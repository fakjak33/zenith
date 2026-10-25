"""Positions and the account: fill, mark, close, roll and settle at expiry.

Cash accounting: opening a credit trade adds the premium to cash and the short
options to the book at a negative value, so NLV = cash + sum(position values)
moves only with the market (and fees). A position's P&L is its cumulative cash
flow plus its current value. Fees: $1 per contract to open, $0 to close
(tastytrade's schedule). Fills: mid -/+ the liquidity-tier half-spread x the
difficulty's slippage factor.

Everything serialises to plain dicts (session_state / the database).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..market import Market
from . import margin, pricing
from .strategies import TEMPLATES, Leg

FEE_OPEN = 1.0


@dataclass
class Position:
    id: int
    ticker: str
    template: str
    legs: list[Leg]
    opened_i: int
    cash_flow: float = 0.0           # $ received (+) / paid (-) incl. fees and assignment
    credit: float = 0.0              # opening net credit in $ (negative = debit)
    status: str = "open"             # open | closed | expired
    closed_i: int = -1
    pnl: float = 0.0                 # realized, once closed
    delta_target: float = 0.16
    width: float = 0.0
    rolls: int = 0
    flags: list[str] = field(default_factory=list)   # events already raised

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("id", "ticker", "template", "opened_i", "cash_flow",
                                           "credit", "status", "closed_i", "pnl", "delta_target",
                                           "width", "rolls")}
        d["legs"] = [lg.to_dict() for lg in self.legs]
        d["flags"] = list(self.flags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        p = cls(**{k: d[k] for k in ("id", "ticker", "template", "opened_i")},
                legs=[Leg.from_dict(x) for x in d["legs"]])
        for k in ("cash_flow", "credit", "status", "closed_i", "pnl", "delta_target", "width",
                  "rolls"):
            setattr(p, k, d[k])
        p.flags = list(d.get("flags", []))
        return p

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def option_legs(self) -> list[Leg]:
        return [lg for lg in self.legs if lg.kind != "stock"]

    def expiry_i(self) -> int:
        ex = [lg.expiry_i for lg in self.option_legs()]
        return min(ex) if ex else -1


@dataclass
class Book:
    cash: float
    start_cash: float
    positions: list[Position] = field(default_factory=list)
    next_id: int = 1
    fees: float = 0.0
    slip: float = 1.0                # 0 = fills at mid
    flat_iv: bool = False            # tests: no skew / earnings (VIX-flat like the vault repro)
    log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"cash": self.cash, "start_cash": self.start_cash, "next_id": self.next_id,
                "fees": self.fees, "slip": self.slip, "flat_iv": self.flat_iv,
                "positions": [p.to_dict() for p in self.positions], "log": self.log[-400:]}

    @classmethod
    def from_dict(cls, d: dict) -> "Book":
        b = cls(d["cash"], d["start_cash"], [Position.from_dict(x) for x in d["positions"]],
                d["next_id"], d.get("fees", 0.0), d.get("slip", 1.0), d.get("flat_iv", False))
        b.log = list(d.get("log", []))
        return b

    # ------------------------------------------------------------ pricing ----
    def leg_mid(self, m: Market, t: str, i: int, lg: Leg) -> float:
        if lg.kind == "stock":
            return m.spot(t, i)
        return float(m.option_mid(t, i, lg.kind == "call", lg.strike, lg.expiry_i, self.flat_iv))

    def leg_fill(self, m: Market, t: str, i: int, lg: Leg, opening: bool, qty: float) -> float:
        """Price per share for trading `qty` of the leg (sign = buy +, sell -)."""
        mid = self.leg_mid(m, t, i, lg)
        if lg.kind == "stock":
            return mid
        h = m.half_spread(t, mid) * self.slip
        return mid + h if qty > 0 else max(mid - h, 0.0)

    def position_value(self, m: Market, p: Position, i: int) -> float:
        v = 0.0
        for lg in p.legs:
            mult = 1.0 if lg.kind == "stock" else 100.0
            v += lg.qty * mult * self.leg_mid(m, p.ticker, i, lg)
        return v

    def position_pnl(self, m: Market, p: Position, i: int) -> float:
        return p.pnl if not p.is_open else p.cash_flow + self.position_value(m, p, i)

    def position_bpr(self, m: Market, p: Position, i: int) -> float:
        if not p.is_open:
            return 0.0
        prices = {k: self.leg_mid(m, p.ticker, i, lg) for k, lg in enumerate(p.legs)
                  if lg.kind != "stock"}
        return margin.bpr(p.legs, m.spot(p.ticker, i), prices)

    def position_greeks(self, m: Market, p: Position, i: int) -> dict:
        g = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        S = m.spot(p.ticker, i)
        for lg in p.legs:
            if lg.kind == "stock":
                g["delta"] += lg.qty
                continue
            sig = float(m.iv(p.ticker, i, lg.strike, lg.expiry_i, self.flat_iv))
            gg = pricing.greeks(lg.kind == "call", S, lg.strike, m.T(i, lg.expiry_i), sig,
                                m.rate[i])
            for k in g:
                g[k] += float(gg[k]) * lg.qty * 100.0
        return g

    # ------------------------------------------------------------- account ----
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    def nlv(self, m: Market, i: int) -> float:
        return self.cash + sum(self.position_value(m, p, i) for p in self.open_positions())

    def bp_used(self, m: Market, i: int) -> float:
        return sum(self.position_bpr(m, p, i) for p in self.open_positions())

    def portfolio_greeks(self, m: Market, i: int) -> dict:
        """Sum of position greeks plus SPY-beta-weighted delta (in SPY shares)."""
        tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "bw_delta": 0.0}
        spy = m.spot("SPY", i) if "SPY" in m.data else None
        for p in self.open_positions():
            g = self.position_greeks(m, p, i)
            for k in ("delta", "gamma", "theta", "vega"):
                tot[k] += g[k]
            if spy:
                beta = m.get(p.ticker, "beta", i)
                beta = beta if beta == beta else 1.0
                tot["bw_delta"] += g["delta"] * beta * m.spot(p.ticker, i) / spy
        return tot

    def note(self, i: int, kind: str, msg: str, pid: int | None = None) -> None:
        self.log.append({"i": int(i), "kind": kind, "msg": msg, "pid": pid})

    # -------------------------------------------------------------- orders ----
    def preview(self, m: Market, t: str, i: int, legs: list[Leg]) -> dict:
        """Price a ticket without trading: credit, BPR, fees, greeks."""
        cash = 0.0
        filled = []
        for lg in legs:
            px = self.leg_fill(m, t, i, lg, True, lg.qty)
            mult = 1.0 if lg.kind == "stock" else 100.0
            cash -= lg.qty * mult * px
            filled.append(Leg(lg.kind, lg.qty, lg.strike, lg.expiry_i, px))
        n_contracts = sum(abs(lg.qty) for lg in legs if lg.kind != "stock")
        prices = {k: self.leg_mid(m, t, i, lg) for k, lg in enumerate(legs) if lg.kind != "stock"}
        tmp = Position(0, t, "", filled, i)
        return {"credit": cash, "fees": FEE_OPEN * n_contracts,
                "bpr": margin.bpr(legs, m.spot(t, i), prices), "legs": filled,
                "greeks": self.position_greeks(m, tmp, i)}

    def open(self, m: Market, t: str, i: int, template: str, legs: list[Leg],
             delta: float = 0.16, width: float = 0.0) -> Position:
        pv = self.preview(m, t, i, legs)
        p = Position(self.next_id, t, template, pv["legs"], i, delta_target=delta, width=width)
        self.next_id += 1
        p.cash_flow = pv["credit"] - pv["fees"]
        p.credit = pv["credit"]
        self.cash += p.cash_flow
        self.fees += pv["fees"]
        self.positions.append(p)
        name = TEMPLATES.get(template, {}).get("name", template)
        self.note(i, "open", f"Opened {name} on {t} for "
                  f"{'credit' if pv['credit'] >= 0 else 'debit'} ${abs(pv['credit']):,.0f}", p.id)
        return p

    def close(self, m: Market, p: Position, i: int, reason: str = "manual") -> float:
        cash = 0.0
        for lg in p.legs:
            px = self.leg_fill(m, p.ticker, i, lg, False, -lg.qty)
            mult = 1.0 if lg.kind == "stock" else 100.0
            cash += lg.qty * mult * px
        p.cash_flow += cash
        self.cash += cash
        p.status, p.closed_i, p.pnl = "closed", i, p.cash_flow
        self.note(i, "close", f"Closed #{p.id} {p.ticker} ({reason}): P&L ${p.pnl:,.0f}", p.id)
        return p.pnl

    def roll(self, m: Market, p: Position, i: int, new_expiry_i: int, mode: str = "out",
             require_credit: bool = True) -> Position | None:
        """Roll out in time (same template and delta at a later expiry) or roll the
        untested side of a two-sided short (same expiry, back to the target delta).
        Returns the new position, or None if a credit was required and not available."""
        from .strategies import build
        if mode == "untested":
            shorts = [lg for lg in p.option_legs() if lg.qty < 0]
            if len({lg.kind for lg in shorts}) < 2:
                return None
            S = m.spot(p.ticker, i)
            untested = max(shorts, key=lambda lg: abs(lg.strike - S))
            new_k = m.strike_for_delta(p.ticker, i, untested.expiry_i, p.delta_target,
                                       untested.kind == "call", self.flat_iv)
            close_px = self.leg_fill(m, p.ticker, i, untested, False, -untested.qty)
            new_leg = Leg(untested.kind, untested.qty, new_k, untested.expiry_i)
            open_px = self.leg_fill(m, p.ticker, i, new_leg, True, new_leg.qty)
            net = (untested.qty * 100 * close_px) - (new_leg.qty * 100 * open_px)
            fee = FEE_OPEN * abs(new_leg.qty)
            if require_credit and net <= 0:
                return None
            new_leg.open_px = open_px
            p.legs = [lg for lg in p.legs if lg is not untested] + [new_leg]
            p.cash_flow += net - fee
            self.cash += net - fee
            self.fees += fee
            p.rolls += 1
            self.note(i, "roll", f"Rolled untested {untested.kind} on #{p.id} {p.ticker} to "
                      f"{new_k:g} for ${net:,.0f}", p.id)
            return p
        legs = build(p.template, m, p.ticker, i, new_expiry_i,
                     contracts=int(max(abs(lg.qty) for lg in p.option_legs()) /
                                   (2 if p.template == "put_bwb" else 1)) or 1,
                     delta=p.delta_target, width=p.width or None, flat=self.flat_iv)
        new_opts = [lg for lg in legs if lg.kind != "stock"]
        close_cash = sum(lg.qty * 100 * self.leg_fill(m, p.ticker, i, lg, False, -lg.qty)
                         for lg in p.option_legs())
        pv = self.preview(m, p.ticker, i, new_opts)
        net = close_cash + pv["credit"]
        if require_credit and net <= 0:
            return None
        # same position object: the old option legs are swapped for the new ones, so
        # any stock (covered call, wheel) and the running P&L stay with the trade
        p.legs = [lg for lg in p.legs if lg.kind == "stock"] + pv["legs"]
        p.cash_flow += net - pv["fees"]
        p.credit += pv["credit"]
        self.cash += net - pv["fees"]
        self.fees += pv["fees"]
        p.rolls += 1
        p.flags = [f for f in p.flags if f not in ("dte21", "target")]
        self.note(i, "roll", f"Rolled #{p.id} {p.ticker} out to {m.dates[new_expiry_i].date()} "
                  f"for {'credit' if net >= 0 else 'debit'} ${abs(net):,.0f}", p.id)
        return p

    def add_legs(self, m: Market, p: Position, i: int, legs: list[Leg], label: str) -> Position:
        """Add legs to an open position, e.g. sell a call against assigned stock (the wheel)."""
        pv = self.preview(m, p.ticker, i, legs)
        p.legs += pv["legs"]
        p.cash_flow += pv["credit"] - pv["fees"]
        p.credit += pv["credit"]
        self.cash += pv["credit"] - pv["fees"]
        self.fees += pv["fees"]
        p.flags = [f for f in p.flags if f not in ("dte21", "target")]
        self.note(i, "adjust", f"{label} on #{p.id} {p.ticker} for ${pv['credit']:,.0f}", p.id)
        return p

    # -------------------------------------------------------------- expiry ----
    def settle(self, m: Market, i: int) -> list[str]:
        """Settle legs expiring at day i (after the close). ITM options become stock."""
        events = []
        for p in self.open_positions():
            due = [lg for lg in p.option_legs() if lg.expiry_i == i]
            if not due:
                continue
            S = m.spot(p.ticker, i)
            shares = 0.0
            for lg in due:
                itm = (S - lg.strike) if lg.kind == "call" else (lg.strike - S)
                if itm > 0.01:
                    dsh = 100.0 * lg.qty * (1.0 if lg.kind == "call" else -1.0)
                    shares += dsh
                    p.cash_flow -= dsh * lg.strike
                    self.cash -= dsh * lg.strike
                    events.append(f"#{p.id} {p.ticker} {lg.kind} {lg.strike:g} "
                                  f"{'assigned' if lg.qty < 0 else 'exercised'}")
            p.legs = [lg for lg in p.legs if lg not in due]
            if shares:
                stock = [lg for lg in p.legs if lg.kind == "stock"]
                if stock:
                    stock[0].qty += shares
                    if abs(stock[0].qty) < 1e-9:
                        p.legs.remove(stock[0])
                else:
                    p.legs.append(Leg("stock", shares, 0.0, -1, 0.0))
            if not p.legs:
                p.status, p.closed_i, p.pnl = "expired", i, p.cash_flow
                self.note(i, "expire", f"#{p.id} {p.ticker} expired: P&L ${p.pnl:,.0f}", p.id)
            elif shares:
                self.note(i, "assign", "; ".join(events[-len(due):]) +
                          " -> now holding stock (sell it, or sell calls against it)", p.id)
        return events

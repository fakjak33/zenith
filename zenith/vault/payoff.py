"""OPTIONS VAULT — payoff curves for the strategy cards (pure, no streamlit).

Every card's default setup is priced on a normalised underlying of 100 with the same flat-vol
Black-Scholes model as pop.py (one IV, no skew, no rates, no costs), so the payoff diagram and the
POP table describe the same trade. Curves are P&L per share (x100 per contract) at three horizons:

  T+0       today, right after entry (zero at spot by construction)
  T+21      21 days later at unchanged IV (the time-exit rule); omitted when an expiry comes first
  expiry    at the first expiration (a calendar's long leg keeps its remaining time value)

Setups come from pop.SETUPS (the cards' stated deltas/DTE). EXTRA_SETUPS adds illustrative shapes for
cards the POP model does not cover; their strikes are ours and every label says so.
"""

from __future__ import annotations

from .pop import IV_DEFAULT, S0, SETUPS, bs, strike_for_delta

# Illustrative shapes for cards without a pop.SETUPS row (strikes/DTE are ours, not tastylive's).
EXTRA_SETUPS = {
    "cash-secured-put": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.30}],
                         "setup": "30-delta put, 45 DTE, cash reserved (delta is ours)"},
    "short-call": {"dte": 45, "legs": [{"type": "call", "qty": -1, "delta": 0.30}],
                   "setup": "30-delta call, 45 DTE (delta is ours)"},
    "ratio-spread": {"dte": 45, "legs": [{"type": "put", "qty": 1, "delta": 0.30},
                                         {"type": "put", "qty": -2, "delta": 0.16}],
                     "setup": "put front ratio: buy 30-delta, sell two 16-delta, 45 DTE (strikes are ours)"},
    "broken-wing-butterfly": {"dte": 45, "legs": [{"type": "put", "qty": 1, "offset": 0},
                                                  {"type": "put", "qty": -2, "offset": -5},
                                                  {"type": "put", "qty": 1, "offset": -15}],
                              "setup": "100/95/85 put broken-wing butterfly, 45 DTE (strikes are ours)"},
    "poor-mans-covered-call": {"legs": [{"type": "call", "qty": 1, "delta": 0.80, "dte": 365},
                                        {"type": "call", "qty": -1, "delta": 0.30, "dte": 45}],
                               "setup": "80-delta 365-DTE call + 30-delta 45-DTE call sold (strikes are ours)"},
    "leaps-stock-replacement": {"dte": 365, "legs": [{"type": "call", "qty": 1, "delta": 0.80}],
                                "setup": "80-delta call, 365 DTE (delta is ours)"},
    "calendar": {"legs": [{"type": "call", "qty": -1, "atm": True, "dte": 40},
                          {"type": "call", "qty": 1, "atm": True, "dte": 75}],
                 "setup": "ATM call calendar, 40/75 DTE (from the card's DTEs)"},
    "diagonal": {"legs": [{"type": "call", "qty": 1, "delta": 0.70, "dte": 75},
                          {"type": "call", "qty": -1, "delta": 0.30, "dte": 40}],
                 "setup": "70-delta 75-DTE call + 30-delta 40-DTE call sold (strikes are ours)"},
    "zero-dte": {"dte": 1, "legs": [{"type": "put", "qty": -1, "delta": 0.16},
                                    {"type": "put", "qty": 1, "delta": 0.05},
                                    {"type": "call", "qty": -1, "delta": 0.16},
                                    {"type": "call", "qty": 1, "delta": 0.05}],
                 "setup": "16/5-delta iron condor, 1 day to expiry (wings are ours)"},
    "earnings-trade": {"dte": 7, "legs": [{"type": "put", "qty": -1, "delta": 0.16},
                                          {"type": "call", "qty": -1, "delta": 0.16}],
                       "setup": "16-delta strangle, 7 DTE, before the IV crush (DTE is ours)"},
    "tail-hedge": {"dte": 90, "legs": [{"type": "put", "qty": 1, "delta": 0.10}],
                   "setup": "10-delta put, 90 DTE (delta and DTE are ours)"},
}
HORIZON = 21


def setup_for(slug: str) -> dict | None:
    s = SETUPS.get(slug) or EXTRA_SETUPS.get(slug)
    if not s:
        return None
    return {"setup": s["setup"], "legs": [{"dte": s.get("dte"), **l} for l in s["legs"]],
            "ours": slug in EXTRA_SETUPS}


def price_legs(legs: list[dict], iv: float = IV_DEFAULT, S: float = S0) -> list[dict]:
    """legs: {type, qty (+long/-short), dte, delta | atm | offset | K}. Adds K and entry price."""
    out = []
    for l in legs:
        if l["type"] == "stock":
            out.append({**l, "K": None, "price": S})
            continue
        T = l["dte"] / 365.0
        if l.get("K") is not None:
            K = float(l["K"])
        elif "delta" in l:
            K = strike_for_delta(l["type"], l["delta"], T, iv, S)
        elif l.get("atm"):
            K = S
        else:
            K = S + l["offset"]
        out.append({**l, "K": K, "price": bs(l["type"], S, K, T, iv)})
    return out


def value_at(legs: list[dict], S: float, days: float, iv: float = IV_DEFAULT) -> float:
    """Position value (per share) after `days`, at underlying S and unchanged IV."""
    v = 0.0
    for l in legs:
        if l["type"] == "stock":
            v += l["qty"] * S
        else:
            v += l["qty"] * bs(l["type"], S, l["K"], max(l["dte"] - days, 0) / 365.0, iv)
    return v


def entry_cost(legs: list[dict]) -> float:
    """>0 = debit paid, <0 = credit received (per share)."""
    return sum(l["qty"] * l["price"] for l in legs)


def curves(legs: list[dict], iv: float = IV_DEFAULT, lo: float = 70.0, hi: float = 130.0,
           n: int = 121) -> list[dict]:
    """Long-form rows {S, horizon, pnl} for the three horizons (P&L per share)."""
    cost = entry_cost(legs)
    opt = [l["dte"] for l in legs if l["type"] != "stock"]
    first = min(opt) if opt else 0
    horizons = [("T+0", 0)]
    if first > HORIZON:
        horizons.append((f"T+{HORIZON}", HORIZON))
    horizons.append(("expiry", first))
    rows = []
    for i in range(n):
        S = lo + (hi - lo) * i / (n - 1)
        for name, d in horizons:
            rows.append({"S": round(S, 2), "horizon": name, "pnl": round(value_at(legs, S, d, iv) - cost, 4)})
    return rows


def summary(legs: list[dict], iv: float = IV_DEFAULT) -> dict:
    """Net premium, breakevens and max profit/loss at first expiry, S from 0 to 400.
    The downside is bounded at S = 0; a P&L still sloping at S = 400 is open-ended (None)."""
    cost = entry_cost(legs)
    opt = [l["dte"] for l in legs if l["type"] != "stock"]
    first = min(opt) if opt else 0
    grid = [0.01] + [0.1 * i for i in range(1, 4001)]
    pnl = [value_at(legs, S, first, iv) - cost for S in grid]
    bes = []
    for s0, p0, s1, p1 in zip(grid, pnl, grid[1:], pnl[1:]):
        if (p0 < 0 <= p1) or (p0 >= 0 > p1):
            bes.append(round(s0 + (s1 - s0) * (0 - p0) / (p1 - p0), 2))
    up_open = abs(pnl[-1] - pnl[-2]) > 1e-6
    mx, mn = max(pnl), min(pnl)
    return {"net": "credit" if cost < 0 else "debit", "premium": round(abs(cost), 3),
            "breakevens": bes,
            "max_profit": None if (up_open and pnl[-1] >= mx - 1e-9) else round(mx, 3),
            "max_loss": None if (up_open and pnl[-1] <= mn + 1e-9) else round(mn, 3),
            "strikes": [round(l["K"], 2) for l in legs if l["K"] is not None]}

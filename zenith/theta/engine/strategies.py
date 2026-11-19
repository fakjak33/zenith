"""Strategy templates in tastytrade style, turned into concrete legs at a date.

Each template lists its legs relative to the short strikes, which are picked by
delta (or at the money); wings sit a fixed dollar width further out. Defaults
mirror the OPTIONS VAULT cards (data/vault/strategies.json): 45 DTE, ~16-delta
shorts for undefined-risk neutral trades, ~30-delta shorts for spreads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..market import Market, round_strike, strike_step

TEMPLATES = {
    "short_strangle": {"name": "Short strangle", "risk": "undefined", "outlook": "neutral",
                       "vault": "short-strangle", "delta": 0.16, "wing": 0},
    "short_straddle": {"name": "Short straddle", "risk": "undefined", "outlook": "neutral",
                       "vault": "short-straddle", "delta": 0.50, "wing": 0},
    "iron_condor": {"name": "Iron condor", "risk": "defined", "outlook": "neutral",
                    "vault": "iron-condor", "delta": 0.16, "wing": 1},
    "iron_fly": {"name": "Iron fly", "risk": "defined", "outlook": "neutral",
                 "vault": "iron-fly", "delta": 0.50, "wing": 1},
    "jade_lizard": {"name": "Jade lizard", "risk": "undefined", "outlook": "bullish",
                    "vault": "jade-lizard", "delta": 0.20, "wing": 1},
    "short_put": {"name": "Short put", "risk": "undefined", "outlook": "bullish",
                  "vault": "short-put", "delta": 0.30, "wing": 0},
    "put_credit_spread": {"name": "Put credit spread", "risk": "defined", "outlook": "bullish",
                          "vault": "put-credit-spread", "delta": 0.30, "wing": 1},
    "call_credit_spread": {"name": "Call credit spread", "risk": "defined", "outlook": "bearish",
                           "vault": "call-credit-spread", "delta": 0.30, "wing": 1},
    "covered_call": {"name": "Covered call", "risk": "stock", "outlook": "bullish",
                     "vault": "covered-call", "delta": 0.30, "wing": 0},
    "put_bwb": {"name": "Put broken-wing butterfly", "risk": "defined", "outlook": "bullish",
                "vault": "broken-wing-butterfly", "delta": 0.30, "wing": 1},
    # ---- buying options: long vol / debit (premium paid is the whole risk)
    "long_call": {"name": "Long call", "risk": "defined", "outlook": "bullish", "vault": "long-call",
                  "delta": 0.50, "wing": 0, "debit": True},
    "long_put": {"name": "Long put", "risk": "defined", "outlook": "bearish", "vault": "long-put",
                 "delta": 0.50, "wing": 0, "debit": True},
    "long_straddle": {"name": "Long straddle", "risk": "defined", "outlook": "neutral", "vault": "long-straddle",
                      "delta": 0.50, "wing": 0, "debit": True},
    "long_strangle": {"name": "Long strangle", "risk": "defined", "outlook": "neutral", "vault": "long-strangle",
                      "delta": 0.25, "wing": 0, "debit": True},
    "call_debit_spread": {"name": "Call debit spread", "risk": "defined", "outlook": "bullish",
                          "vault": "debit-spread", "delta": 0.50, "wing": 1, "debit": True},
    "put_debit_spread": {"name": "Put debit spread", "risk": "defined", "outlook": "bearish",
                         "vault": "debit-spread", "delta": 0.50, "wing": 1, "debit": True},
    "long_butterfly": {"name": "Long call butterfly", "risk": "defined", "outlook": "neutral",
                       "vault": "butterfly", "delta": 0.50, "wing": 1, "debit": True},
    "put_backspread": {"name": "Put backspread (1x2)", "risk": "defined", "outlook": "bearish",
                       "vault": "ratio-spread", "delta": 0.40, "wing": 1, "debit": True},
    "long_calendar": {"name": "Long call calendar", "risk": "defined", "outlook": "neutral", "vault": "calendar",
                      "delta": 0.50, "wing": 0, "debit": True, "calendar": True},
}
LONG_TEMPLATES = [k for k, v in TEMPLATES.items() if v.get("debit")]
SHORT_TEMPLATES = [k for k, v in TEMPLATES.items() if not v.get("debit")]


@dataclass
class Leg:
    kind: str             # "put" | "call" | "stock"
    qty: float            # signed: contracts for options, shares for stock
    strike: float = 0.0
    expiry_i: int = -1
    open_px: float = 0.0  # per share, the fill

    def to_dict(self) -> dict:
        return {"kind": self.kind, "qty": self.qty, "strike": self.strike,
                "expiry_i": self.expiry_i, "open_px": self.open_px}

    @classmethod
    def from_dict(cls, d: dict) -> "Leg":
        return cls(d["kind"], float(d["qty"]), float(d.get("strike", 0.0)),
                   int(d.get("expiry_i", -1)), float(d.get("open_px", 0.0)))


def default_width(S: float) -> float:
    """Wing width: about 3% of spot, rounded to the strike grid (at least one step)."""
    st = strike_step(S)
    return max(st, round(0.03 * S / st) * st)


def build(template: str, m: Market, t: str, i: int, expiry_i: int, contracts: int = 1,
          delta: float | None = None, width: float | None = None, flat: bool = False,
          back_expiry_i: int | None = None) -> list[Leg]:
    """Legs for a template (open_px left at 0; the order fills them). ``back_expiry_i`` is the
    long leg's expiry for calendars."""
    spec = TEMPLATES[template]
    d = spec["delta"] if delta is None else float(delta)
    S = m.spot(t, i)
    w = default_width(S) if width is None else float(width)
    n = int(contracts)
    atm = round_strike(S, S)

    def kp(dd):
        return atm if dd >= 0.5 else m.strike_for_delta(t, i, expiry_i, dd, False, flat)

    def kc(dd):
        return atm if dd >= 0.5 else m.strike_for_delta(t, i, expiry_i, dd, True, flat)

    P = lambda k, q: Leg("put", q * n, k, expiry_i)       # noqa: E731
    C = lambda k, q: Leg("call", q * n, k, expiry_i)      # noqa: E731
    if template in ("short_strangle", "short_straddle"):
        return [P(kp(d), -1), C(kc(d), -1)]
    if template in ("iron_condor", "iron_fly"):
        p, c = kp(d), kc(d)
        return [P(p - w, 1), P(p, -1), C(c, -1), C(c + w, 1)]
    if template == "jade_lizard":
        p, c = kp(0.30 if delta is None else d), kc(d)
        return [P(p, -1), C(c, -1), C(c + w, 1)]
    if template == "short_put":
        return [P(kp(d), -1)]
    if template == "put_credit_spread":
        p = kp(d)
        return [P(p, -1), P(p - w, 1)]
    if template == "call_credit_spread":
        c = kc(d)
        return [C(c, -1), C(c + w, 1)]
    if template == "covered_call":
        return [Leg("stock", 100.0 * n), C(kc(d), -1)]
    if template == "put_bwb":
        k = kp(d)
        return [P(k + w, 1), P(k, -2), P(k - 2 * w, 1)]
    if template == "long_call":
        return [C(kc(d), 1)]
    if template == "long_put":
        return [P(kp(d), 1)]
    if template in ("long_straddle", "long_strangle"):
        return [P(kp(d), 1), C(kc(d), 1)]
    if template == "call_debit_spread":
        c = kc(d)
        return [C(c, 1), C(c + w, -1)]
    if template == "put_debit_spread":
        p = kp(d)
        return [P(p, 1), P(p - w, -1)]
    if template == "long_butterfly":
        return [C(atm - w, 1), C(atm, -2), C(atm + w, 1)]
    if template == "put_backspread":
        p = kp(d)
        return [P(p, -1), P(p - w, 2)]
    if template == "long_calendar":
        back = back_expiry_i
        if back is None:                     # the listed expiry ~30 days after the front month
            back = m.nearest_expiry(t, i, m.days_between(i, expiry_i) + 30)
            if back is None or back <= expiry_i:
                raise ValueError("no later expiry for the calendar's long leg")
        return [C(atm, -1), Leg("call", n, atm, back)]
    raise KeyError(template)

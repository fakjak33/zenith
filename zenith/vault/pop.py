"""OPTIONS VAULT — probability of profit and tail risk for each strategy's default setup.

A small, deterministic model so strategies can be compared on one footing:
- options are priced with Black-Scholes at one implied vol (IV_DEFAULT, near the VIX long-run
  average of about 19 [page:concepts-vix]), no skew, no rates, no costs;
- each strategy uses its default tastylive setup (deltas, DTE) from the strategy cards;
- held to expiration. Two outcome distributions are reported:
    theoretical  terminal price lognormal at the pricing IV (what the trading platform's POP shows);
    realized     terminal price lognormal at REALIZED_RATIO x IV, the average realized/implied ratio
                 in 1990-2026 [repro:vrp_spx]; this is where the variance premium shows up.
- for each: POP (P&L > 0), mean P&L and the average of the worst 5% (CVaR), in units of the credit
  received (short premium) or the debit paid (long premium), plus max loss where defined.

Limits (stated on every row): no skew (real OTM puts are richer, calls cheaper), no management
(50% / 21 DTE changes win rates, see repro:strangle_mgmt), no early exit, one flat vol, normal-ish
tails (real crashes are fatter: repro:expected_move). Use it to compare shapes, not to forecast.

Run: .venv/Scripts/python -m zenith.vault.pop
"""

from __future__ import annotations

import json
import math
import sys

from ..beta.hedge import _ncdf
from ..config import VAULT_DIR

POP_PATH = VAULT_DIR / "pop.json"
IV_DEFAULT = 0.20
REALIZED_RATIO = 0.78
S0 = 100.0
GRID = 4001


def bs(kind: str, S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * _ncdf(d2)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(kind: str, S: float, K: float, T: float, sig: float) -> float:
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    return _ncdf(d1) if kind == "call" else _ncdf(d1) - 1.0


def strike_for_delta(kind: str, delta: float, T: float, sig: float, S: float = S0) -> float:
    """Strike whose absolute delta equals `delta` (bisection; delta in (0, 1))."""
    lo, hi = S * 0.2, S * 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = abs(bs_delta(kind, S, mid, T, sig))
        # call delta falls as strike rises; |put delta| rises as strike rises
        if (kind == "call" and d > delta) or (kind == "put" and d < delta):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _leg_payoff(leg: dict, ST: float) -> float:
    if leg["type"] == "stock":
        return leg["qty"] * ST
    intrinsic = max(0.0, ST - leg["K"]) if leg["type"] == "call" else max(0.0, leg["K"] - ST)
    return leg["qty"] * intrinsic


def build_legs(spec: list[dict], dte: int, iv: float) -> list[dict]:
    """spec legs: {type, qty (+long/-short), delta | atm | strike_offset, dte?}. Adds K and price."""
    T = dte / 365.0
    legs = []
    for s in spec:
        if s["type"] == "stock":
            legs.append({**s, "K": None, "price": S0})
            continue
        if "delta" in s:
            K = strike_for_delta(s["type"], s["delta"], T, iv)
        elif s.get("atm"):
            K = S0
        else:
            K = S0 + s["offset"]
        legs.append({**s, "K": K, "price": bs(s["type"], S0, K, T, iv)})
    return legs


def evaluate(legs: list[dict], dte: int, sig_outcome: float) -> dict:
    """Hold-to-expiration P&L distribution under a lognormal terminal price."""
    T = dte / 365.0
    cost = sum(l["qty"] * l["price"] for l in legs)          # >0 = debit paid, <0 = credit
    unit = abs(cost) if abs(cost) > 1e-9 else 1.0
    has_stock = any(l["type"] == "stock" for l in legs)
    s = sig_outcome * math.sqrt(T)
    mu = -0.5 * s * s                                         # martingale terminal price
    zs = [-6 + 12 * i / (GRID - 1) for i in range(GRID)]
    w = [math.exp(-0.5 * z * z) for z in zs]
    tot = sum(w)
    pnl = []
    for z, wi in zip(zs, w):
        ST = S0 * math.exp(mu + s * z)
        pnl.append((sum(_leg_payoff(l, ST) for l in legs) - cost, wi / tot))
    pop = sum(p for v, p in pnl if v > 1e-9)
    mean = sum(v * p for v, p in pnl)
    acc, cvar_sum = 0.0, 0.0
    for v, p in sorted(pnl):
        take = min(p, 0.05 - acc)
        if take <= 0:
            break
        cvar_sum += v * take
        acc += take
    return {"pop": round(pop, 4), "mean_units": round(mean / unit, 3),
            "cvar5_units": round(cvar_sum / 0.05 / unit, 2), "net": "credit" if cost < 0 else "debit",
            "unit": "position cost" if has_stock else ("credit" if cost < 0 else "debit"),
            "premium": round(abs(cost), 3)}


# Default setups (deltas/DTE) as stated on the strategy cards and their sources.
SETUPS = {
    "short-strangle": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.16},
                                           {"type": "call", "qty": -1, "delta": 0.16}],
                       "setup": "16-delta put + 16-delta call, 45 DTE",
                       "sources": ["page:concepts-strangle", "ep:market-measures-2023-02-22@03:07"]},
    "short-straddle": {"dte": 45, "legs": [{"type": "put", "qty": -1, "atm": True},
                                           {"type": "call", "qty": -1, "atm": True}],
                       "setup": "ATM put + call, 45 DTE", "sources": ["page:concepts-straddle"]},
    "iron-condor": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.16},
                                        {"type": "put", "qty": 1, "delta": 0.05},
                                        {"type": "call", "qty": -1, "delta": 0.16},
                                        {"type": "call", "qty": 1, "delta": 0.05}],
                    "setup": "16-delta shorts, 5-delta wings, 45 DTE",
                    "sources": ["ep:market-measures-2023-02-22@03:07"]},
    "iron-fly": {"dte": 45, "legs": [{"type": "put", "qty": -1, "atm": True},
                                     {"type": "put", "qty": 1, "delta": 0.16},
                                     {"type": "call", "qty": -1, "atm": True},
                                     {"type": "call", "qty": 1, "delta": 0.16}],
                 "setup": "ATM shorts, 16-delta wings, 45 DTE (wing choice is ours)",
                 "sources": ["page:concepts-iron-butterfly"]},
    "short-put": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.30}],
                  "setup": "30-delta put, 45 DTE (delta is ours; no tastylive put delta found)",
                  "sources": ["page:concepts-options-expiration"]},
    "put-credit-spread": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.30},
                                              {"type": "put", "qty": 1, "delta": 0.16}],
                          "setup": "30-delta short / 16-delta long put, 45 DTE (long strike is ours)",
                          "sources": ["page:concepts-vertical-spread"]},
    "call-credit-spread": {"dte": 45, "legs": [{"type": "call", "qty": -1, "delta": 0.30},
                                               {"type": "call", "qty": 1, "delta": 0.16}],
                           "setup": "30-delta short / 16-delta long call, 45 DTE (long strike is ours)",
                           "sources": ["page:concepts-vertical-spread"]},
    "jade-lizard": {"dte": 45, "legs": [{"type": "put", "qty": -1, "delta": 0.16},
                                        {"type": "call", "qty": -1, "delta": 0.16},
                                        {"type": "call", "qty": 1, "delta": 0.05}],
                    "setup": "16-delta put + 16/5-delta call spread, 45 DTE (wing and DTE are ours; "
                             "tastylive's study used 5-7 DTE)",
                    "sources": ["page:concepts-jade-lizard", "ep:market-measures-2026-05-02-2@03:47"]},
    "covered-call": {"dte": 45, "legs": [{"type": "stock", "qty": 1},
                                         {"type": "call", "qty": -1, "delta": 0.30}],
                     "setup": "100 shares + 30-delta call, 45 DTE", "sources": ["page:concepts-covered-call"]},
    "collar": {"dte": 45, "legs": [{"type": "stock", "qty": 1}, {"type": "call", "qty": -1, "delta": 0.30},
                                   {"type": "put", "qty": 1, "delta": 0.30}],
               "setup": "100 shares + 30-delta call sold + 30-delta put bought (deltas are ours)",
               "sources": ["page:concepts-collar-option"]},
    "debit-spread": {"dte": 45, "legs": [{"type": "call", "qty": 1, "atm": True},
                                         {"type": "call", "qty": -1, "delta": 0.30}],
                     "setup": "ATM long call / 30-delta short call, 45 DTE (strikes are ours)",
                     "sources": ["page:concepts-vertical-debit-spread"]},
    "long-call": {"dte": 45, "legs": [{"type": "call", "qty": 1, "atm": True}],
                  "setup": "ATM call, 45 DTE", "sources": ["page:concepts-call-options"]},
    "long-put": {"dte": 45, "legs": [{"type": "put", "qty": 1, "atm": True}],
                 "setup": "ATM put, 45 DTE", "sources": ["page:concepts-put-options"]},
    "long-straddle": {"dte": 45, "legs": [{"type": "put", "qty": 1, "atm": True},
                                          {"type": "call", "qty": 1, "atm": True}],
                      "setup": "ATM put + call bought, 45 DTE", "sources": ["page:concepts-straddle"]},
    "long-strangle": {"dte": 45, "legs": [{"type": "put", "qty": 1, "delta": 0.16},
                                          {"type": "call", "qty": 1, "delta": 0.16}],
                      "setup": "16-delta put + call bought, 45 DTE", "sources": ["page:concepts-strangle"]},
    "butterfly": {"dte": 45, "legs": [{"type": "call", "qty": 1, "offset": -5},
                                      {"type": "call", "qty": -2, "atm": True},
                                      {"type": "call", "qty": 1, "offset": 5}],
                  "setup": "95/100/105 call butterfly, 45 DTE (width is ours)",
                  "sources": ["page:concepts-long-butterfly-spread"]},
    "protective-put": {"dte": 45, "legs": [{"type": "stock", "qty": 1}, {"type": "put", "qty": 1, "delta": 0.30}],
                       "setup": "100 shares + 30-delta put, 45 DTE (delta is ours)",
                       "sources": ["page:concepts-protective-put"]},
}
LIMITS = ("Black-Scholes, flat IV, no skew, no costs, held to expiration (no 50%/21-DTE management), "
          "lognormal tails thinner than real crashes. Stock-based rows measure P&L of the whole position.")


def table(iv: float = IV_DEFAULT, ratio: float = REALIZED_RATIO) -> dict:
    rows = []
    for slug, s in SETUPS.items():
        legs = build_legs(s["legs"], s["dte"], iv)
        theo = evaluate(legs, s["dte"], iv)
        real = evaluate(legs, s["dte"], iv * ratio)
        rows.append({"slug": slug, "setup": s["setup"], "sources": s["sources"],
                     "net": theo["net"], "unit": theo["unit"], "premium_per_share": theo["premium"],
                     "pop_theoretical": theo["pop"], "pop_realized": real["pop"],
                     "mean_units_realized": real["mean_units"], "cvar5_units_realized": real["cvar5_units"],
                     "strikes": [round(l["K"], 2) if l["K"] else "stock" for l in legs]})
    rows.sort(key=lambda r: -r["pop_theoretical"])
    return {"iv": iv, "realized_ratio": ratio, "spot": S0, "limits": LIMITS,
            "sources": ["page:concepts-vix", "repro:vrp_spx"], "rows": rows}


def main(argv=None) -> None:
    t = table()
    POP_PATH.write_text(json.dumps(t, indent=2) + "\n", encoding="utf-8")
    print(f"{'slug':22s} {'POP theo':>8s} {'POP real':>8s} {'mean':>6s} {'CVaR5':>7s}  setup")
    for r in t["rows"]:
        print(f"{r['slug']:22s} {r['pop_theoretical']:8.1%} {r['pop_realized']:8.1%} "
              f"{r['mean_units_realized']:6.2f} {r['cvar5_units_realized']:7.2f}  {r['setup']}")


if __name__ == "__main__":
    sys.exit(main())

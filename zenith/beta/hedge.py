"""CLEAN BETA hedge-condition monitor — pure functions over price series.

  trend_gate   SPY 12-1 month return sign + price vs its 10-month MA
               (Moskowitz-Ooi-Pedersen TSMOM; Faber's 10-month rule).
               Both positive -> ON, both negative -> REDUCED, split -> PARTIAL.
  vol_target   scale = target / forecast vol of the basket, capped. Forecast =
               mean of 1-month and 3-month realized vol (a blend, labelled [E]).
               Risk control, not alpha (Cederburg et al. 2020).
  vrp          VIX minus a HAR-lite forecast of SPY realized vol (equal blend
               of 1-week, 1-month and 3-month realized — Corsi's three HAR
               horizons without fitted coefficients, so [E]). Its 5-year
               percentile sizes the put sleeve: LOW VRP = protection is cheap
               relative to what realized vol has been delivering
               (Israelov & Nielsen 2015) -> spend toward the top of the budget.
  btal_gate    BTAL (long low-beta / short high-beta) 12-1 + 200-day trend.
               Up-trending anti-beta = the high-beta sleeve is fighting the tape.
  put_ladder   advisory 25Δ/10Δ SPY put spreads at three staggered tenors
               from the live chain, Black-Scholes deltas off chain IV.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from ..config import (BETA_HARVEST_MULTIPLE, BETA_PUT_BUDGET, BETA_PUT_LONG_DELTA,
                      BETA_PUT_SHORT_DELTA, BETA_PUT_TENORS, BETA_RISK_FREE, BETA_TREND_EXPOSURE,
                      BETA_TREND_LOOKBACK, BETA_TREND_MA, BETA_TREND_SKIP, BETA_VIX_HARVEST_PCT,
                      BETA_VOL_SCALE_CAP, BETA_VOL_TARGET, BETA_VRP_PCT_WINDOW)
from . import r
from .estimators import TRADING_DAYS

_SQ = math.sqrt(TRADING_DAYS)


def _pct_of_last(s: pd.Series, window: int) -> float | None:
    s = s.dropna().tail(window)
    if len(s) < 60:
        return None
    return float((s < s.iloc[-1]).mean() * 100.0 + (s == s.iloc[-1]).mean() * 50.0)


def tsmom(close: pd.Series) -> dict:
    """12-1 return and price vs 10-month MA for any series."""
    c = close.dropna()
    out = {"price": r(c.iloc[-1], 2) if len(c) else None}
    if len(c) > BETA_TREND_LOOKBACK:
        out["ret_12_1"] = r(c.iloc[-1 - BETA_TREND_SKIP] / c.iloc[-1 - BETA_TREND_LOOKBACK] - 1.0, 4)
    if len(c) >= BETA_TREND_MA:
        ma = c.tail(BETA_TREND_MA).mean()
        out["ma"] = r(ma, 2)
        out["vs_ma"] = r(c.iloc[-1] / ma - 1.0, 4)
    if len(c) >= 200:
        out["vs_200d"] = r(c.iloc[-1] / c.tail(200).mean() - 1.0, 4)
    return out


def trend_gate(spy: pd.Series) -> dict:
    m = tsmom(spy)
    votes = [v > 0 for v in (m.get("ret_12_1"), m.get("vs_ma")) if v is not None]
    if not votes:
        state = "PARTIAL"
    elif all(votes):
        state = "ON"
    elif not any(votes):
        state = "REDUCED"
    else:
        state = "PARTIAL"
    return {**m, "state": state, "exposure": BETA_TREND_EXPOSURE[state]}


def realized(ret: pd.Series, n: int) -> float | None:
    x = ret.dropna().tail(n)
    return float(x.std() * _SQ) if len(x) >= max(3, n // 2) else None


def vol_target(basket_ret: pd.Series) -> dict:
    v1, v3 = realized(basket_ret, 21), realized(basket_ret, 63)
    vals = [v for v in (v1, v3) if v]
    if not vals:
        return {}
    fc = sum(vals) / len(vals)
    scale = min(BETA_VOL_SCALE_CAP, BETA_VOL_TARGET / fc) if fc > 0 else None
    return {"rv_1m": r(v1, 4), "rv_3m": r(v3, 4), "forecast": r(fc, 4),
            "target": BETA_VOL_TARGET, "cap": BETA_VOL_SCALE_CAP, "scale": r(scale, 3)}


def har_forecast(spy_ret: pd.Series) -> pd.Series:
    """Equal-weight blend of trailing 5/21/63-day realized vol (annualized)."""
    sq = spy_ret.pow(2)
    parts = [np.sqrt(sq.rolling(n, min_periods=n).mean() * TRADING_DAYS) for n in (5, 21, 63)]
    return (parts[0] + parts[1] + parts[2]) / 3.0


def vrp(vix: pd.Series, spy: pd.Series) -> dict:
    spy_ret = spy.pct_change()
    fc = har_forecast(spy_ret)
    j = pd.concat([vix / 100.0, fc], axis=1, join="inner").dropna()
    if j.empty:
        return {}
    j.columns = ["iv", "rv"]
    s = j["iv"] - j["rv"]
    vix_pct = _pct_of_last(j["iv"], BETA_VRP_PCT_WINDOW)
    vrp_pct = _pct_of_last(s, BETA_VRP_PCT_WINDOW)
    lo, hi = BETA_PUT_BUDGET
    budget = lo + (hi - lo) * (1.0 - vrp_pct / 100.0) if vrp_pct is not None else (lo + hi) / 2
    hist = s.tail(504)
    return {"vix": r(j["iv"].iloc[-1] * 100, 2), "rv_forecast": r(j["rv"].iloc[-1], 4),
            "vrp": r(s.iloc[-1], 4), "vrp_pct": r(vrp_pct, 1), "vix_pct": r(vix_pct, 1),
            "vrp_mean_5y": r(s.tail(BETA_VRP_PCT_WINDOW).mean(), 4),
            "put_budget": r(budget, 4),
            "harvest_window": vix_pct is not None and vix_pct >= BETA_VIX_HARVEST_PCT,
            "series": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                       "vrp": [r(v, 4) for v in hist.values],
                       "vix": [r(v * 100, 2) for v in j["iv"].tail(504).values],
                       "rv": [r(v, 4) for v in j["rv"].tail(504).values]}}


def btal_gate(btal: pd.Series) -> dict:
    m = tsmom(btal)
    up = [v > 0 for v in (m.get("ret_12_1"), m.get("vs_200d")) if v is not None]
    state = ("HEADWIND" if up and all(up) else "TAILWIND" if up and not any(up) else "NEUTRAL")
    return {**m, "state": state}


# ------------------------------------------------------------ put ladder ----
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_delta(S: float, K: float, T: float, sigma: float, rf: float = BETA_RISK_FREE) -> float | None:
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return None
    d1 = (math.log(S / K) + (rf + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) - 1.0


def _mid(row) -> float | None:
    b, a = row.get("bid"), row.get("ask")
    if b is not None and a is not None and np.isfinite(b) and np.isfinite(a) and a > 0 and a >= b:
        return (b + a) / 2.0
    lp = row.get("lastPrice")
    return float(lp) if lp is not None and np.isfinite(lp) and lp > 0 else None


def _nearest_delta(puts: pd.DataFrame, target: float):
    p = puts.dropna(subset=["delta"])
    if p.empty:
        return None
    return p.iloc[(p["delta"] - target).abs().argsort().iloc[0]]


def rung(spot: float, puts: pd.DataFrame, expiry: str, dte: int, beta: float) -> dict | None:
    """One 25Δ/10Δ put spread from one expiry's put chain."""
    T = dte / 365.0
    p = puts.copy()
    p["delta"] = [bs_put_delta(spot, float(k), T, float(iv) if iv else 0.0)
                  for k, iv in zip(p["strike"], p["impliedVolatility"])]
    p["delta"] = p["delta"].astype(float)
    lg, sh = _nearest_delta(p, BETA_PUT_LONG_DELTA), _nearest_delta(p, BETA_PUT_SHORT_DELTA)
    if lg is None or sh is None or float(sh["strike"]) >= float(lg["strike"]):
        return None
    ml, ms = _mid(lg), _mid(sh)
    if ml is None or ms is None or ml <= ms:
        return None
    cost = ml - ms
    width = float(lg["strike"]) - float(sh["strike"])
    cost_pct = cost / spot
    return {"expiry": expiry, "dte": dte, "long_strike": float(lg["strike"]),
            "short_strike": float(sh["strike"]), "long_delta": r(lg["delta"], 3),
            "short_delta": r(sh["delta"], 3), "long_iv": r(lg["impliedVolatility"], 4),
            "short_iv": r(sh["impliedVolatility"], 4), "cost": r(cost, 2),
            "cost_pct_spot": r(cost_pct, 5), "max_multiple": r(width / cost, 2),
            "long_otm": r(float(lg["strike"]) / spot - 1.0, 4),
            # carry of covering the WHOLE beta-weighted notional with this rung, per year
            "annual_carry_full": r(cost_pct * beta * 365.0 / dte, 5),
            "harvest_at": r(cost * BETA_HARVEST_MULTIPLE, 2)}


def put_ladder(spot: float, chains: list[dict], beta: float, budget: float) -> dict:
    """chains: [{expiry, dte, puts: DataFrame}] -> advisory ladder."""
    rungs = []
    used = set()
    for target in BETA_PUT_TENORS:
        cands = [c for c in chains if c["expiry"] not in used and 30 <= c["dte"] <= 200]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(c["dte"] - target))
        used.add(c["expiry"])
        rr = rung(spot, c["puts"], c["expiry"], c["dte"], beta)
        if rr:
            rr["target_dte"] = target
            rungs.append(rr)
    if not rungs:
        return {"rungs": [], "spot": r(spot, 2)}
    carry = float(np.mean([x["annual_carry_full"] for x in rungs]))
    coverage = min(1.0, budget / carry) if carry > 0 else None
    contracts_per_mm = beta * 1e6 / (100.0 * spot)       # full-coverage contracts per $1M sleeve
    for x in rungs:
        x["contracts_per_mm"] = r(contracts_per_mm * (coverage or 0) / len(rungs), 2)
    return {"rungs": rungs, "spot": r(spot, 2), "ladder_carry_full": r(carry, 5),
            "coverage": r(coverage, 3), "budget": r(budget, 4), "beta": r(beta, 3)}


def alerts(state: dict) -> list[dict]:
    """Plain-language alert list for the monitor."""
    out = []
    tg = state.get("trend") or {}
    if tg.get("state") == "REDUCED":
        out.append({"level": "high", "text": "Trend gate REDUCED: SPY's 12-1 return and its 10-month MA are both "
                                             "negative. Cut sleeve exposure to "
                                             f"{BETA_TREND_EXPOSURE['REDUCED']:.0%}."})
    elif tg.get("state") == "PARTIAL":
        out.append({"level": "med", "text": "Trend gate PARTIAL: the 12-1 return and the 10-month MA disagree."})
    v = state.get("vrp") or {}
    if v.get("harvest_window"):
        out.append({"level": "high", "text": f"VIX at the {v.get('vix_pct'):.0f}th percentile (≥ "
                                             f"{BETA_VIX_HARVEST_PCT:g}th): monetize part of the put "
                                             "ladder and restrike lower. Do NOT switch to selling vol."})
    if v.get("vrp_pct") is not None and v["vrp_pct"] <= 20:
        out.append({"level": "med", "text": f"VRP at the {v['vrp_pct']:.0f}th percentile: index protection is "
                                            "unusually cheap versus realized vol. Lean toward the top of the "
                                            "put budget."})
    b = state.get("btal") or {}
    if b.get("state") == "HEADWIND":
        out.append({"level": "med", "text": "BTAL (anti-beta) is trending up: the high-beta sleeve is fighting "
                                            "the tape."})
    vt = state.get("vol_target") or {}
    if vt.get("scale") is not None and vt["scale"] < 0.75:
        out.append({"level": "med", "text": f"Vol target scale {vt['scale']:.2f}: the basket is running hot "
                                            f"versus the {BETA_VOL_TARGET:.0%} target."})
    return out


def dte(expiry: str, today: date) -> int:
    return (date.fromisoformat(expiry) - today).days

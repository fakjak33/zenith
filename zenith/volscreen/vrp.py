"""Volatility risk premium: implied minus forecast realized, per tenor.

    VRP_T = IV_ex(T) - E[RV over T]      (vol points, decimal; > 0 = options rich)

IV_ex is the ex-earnings constant-maturity ATM vol (an earnings jump is a
separate, priced event — keeping it in would flag every pre-earnings name as
"rich"). The realized side is the GARCH term forecast over the SAME horizon,
so the comparison is like for like; HAR and Yang-Zhang variants are shown for
robustness. Carr & Wu (2009) and Bollerslev, Tauchen & Zhou (2009) document
the premium; Goyal & Saretto (2009) show the cross-section of the HV-IV gap
predicts option returns (sell where IV sits far above HV).

History statistics: z-score of today's VRP30 against its own trailing history,
and the OLS SLOPE of VRP30 over the last 5 / 10 observations (vol points per
day) — the rate of change that feeds the state machine.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _f(x) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def per_tenor(ivx: dict, fc: dict) -> dict:
    return {t: _f(ivx.get(t)) - _f(fc.get(t)) for t in ivx}


def snapshot(ivx: dict, rv: dict, garch_term: dict) -> dict:
    iv30 = _f(ivx.get(30))
    g30 = _f(garch_term.get(30))
    har = _f(rv.get("har30"))
    yz = _f(rv.get("yz20"))
    base = g30 if math.isfinite(g30) else (har if math.isfinite(har) else yz)
    rv252 = _f(rv.get("cc252"))
    return {
        "vrp30": iv30 - base,
        "vrp30_garch": iv30 - g30,
        "vrp30_har": iv30 - har,
        "vrp30_yz": iv30 - yz,
        "iv_rv": iv30 / yz if yz and math.isfinite(yz) and yz > 0 else float("nan"),
        # Goyal-Saretto spread, log(HV) - log(IV): very negative = IV rich vs history
        "gs": (math.log(rv252) - math.log(iv30)) if rv252 > 0 and iv30 > 0 else float("nan"),
        "term_slope": _f(per_tenor(ivx, garch_term).get(90)) - (iv30 - g30),
    }


def ols_slope(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return float("nan")
    x = np.arange(n, dtype=float)
    x -= x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def history_stats(series: pd.Series, z_window: int = 252) -> dict:
    """series: VRP30 by date, oldest first, today's value last."""
    s = pd.Series(series, dtype=float).dropna()
    if not len(s):
        return {"z": float("nan"), "slope5": float("nan"), "slope10": float("nan"), "n": 0}
    hist = s.iloc[-z_window:]
    sd = float(hist.std(ddof=1)) if len(hist) >= 20 else float("nan")
    z = (float(s.iloc[-1]) - float(hist.mean())) / sd if sd and math.isfinite(sd) and sd > 0 else float("nan")
    return {"z": z, "slope5": ols_slope(s.iloc[-5:]), "slope10": ols_slope(s.iloc[-10:]), "n": int(len(s))}


def iv_rank(series: pd.Series, n: int = 252) -> tuple[float, float, int]:
    """(IV rank, IV percentile, observations) of the last value in its trailing window."""
    s = pd.Series(series, dtype=float).dropna().iloc[-n:]
    if len(s) < 20:
        return float("nan"), float("nan"), int(len(s))
    lo, hi, x = float(s.min()), float(s.max()), float(s.iloc[-1])
    rank = (x - lo) / (hi - lo) if hi > lo else float("nan")
    pct = float((s.iloc[:-1] < x).mean())
    return rank, pct, int(len(s))


def sigma_z(series: pd.Series, n: int = 252) -> float:
    """Today's IV in standard deviations of its own trailing-year distribution."""
    s = pd.Series(series, dtype=float).dropna().iloc[-n:]
    if len(s) < 20:
        return float("nan")
    sd = float(s.iloc[:-1].std(ddof=1))
    return (float(s.iloc[-1]) - float(s.iloc[:-1].mean())) / sd if sd > 0 else float("nan")

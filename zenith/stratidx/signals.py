"""Performance and trend signals for one strategy-index level series.

  TSMOM 12-1     sign and vol-scaled size of the return from t-252 to t-21
                 (Moskowitz, Ooi & Pedersen 2012)
  XS rank        percentile of 6M / 12M return across the tracked indices
                 (cross-sectional momentum, Jegadeesh & Titman 1993)
  Donchian       55-day breakout / 20-day position in channel (Turtle rules)
  EWMAC          seven-speed EMA-crossover trend score, -20..+20 (trend/ewmac,
                 Carver's construction; Hurst, Ooi & Pedersen 2017 for the evidence)
  EMA 50/200     simple trend filter
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _ret(s: pd.Series, days: int) -> float | None:
    s = s.dropna()
    if len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)


def _ann(s: pd.Series, years: int) -> float | None:
    s = s.dropna()
    start = s.index[-1] - pd.DateOffset(years=years)
    w = s[s.index >= start]
    if len(w) < 2 or w.index[0] > start + pd.Timedelta(days=10):
        return None
    days = (w.index[-1] - w.index[0]).days
    return float((w.iloc[-1] / w.iloc[0]) ** (365.25 / days) - 1.0)


def performance(s: pd.Series, cash: pd.Series | None = None) -> dict:
    s = s.dropna()
    ytd0 = s[s.index < pd.Timestamp(f"{s.index[-1].year}-01-01")]
    dd = s / s.cummax() - 1.0
    r = np.log(s).diff().dropna()
    def sharpe(n):
        w = r.iloc[-n:]
        if len(w) < n * 0.9:
            return None
        ex = w.mean() * 252
        if cash is not None and len(cash):
            c = np.log(cash.reindex(s.index).ffill()).diff().iloc[-n:]
            ex -= float(c.mean() * 252) if c.notna().any() else 0.0
        sd = w.std() * math.sqrt(252)
        return float(ex / sd) if sd > 0 else None
    return {"last": float(s.iloc[-1]), "date": s.index[-1].date().isoformat(),
            "inception": s.index[0].date().isoformat(),
            "r1d": _ret(s, 1), "r1w": _ret(s, 5), "r1m": _ret(s, 21), "r3m": _ret(s, 63), "r6m": _ret(s, 126),
            "ytd": float(s.iloc[-1] / ytd0.iloc[-1] - 1.0) if len(ytd0) else None,
            "r1y": _ret(s, 252), "a3y": _ann(s, 3), "a5y": _ann(s, 5), "a10y": _ann(s, 10),
            "vol1y": float(r.iloc[-252:].std() * math.sqrt(252)) if len(r) > 200 else None,
            "sharpe1y": sharpe(252), "sharpe3y": sharpe(756),
            "maxdd": float(dd.min()), "dd": float(dd.iloc[-1]),
            "dd_days": int((dd.iloc[::-1].cumprod() != 0).sum()) if dd.iloc[-1] < 0 else 0}


def tsmom(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 260:
        return {"sign": None, "ret": None, "scaled": None}
    r = float(s.iloc[-22] / s.iloc[-253] - 1.0)
    vol = float(np.log(s).diff().iloc[-252:].std() * math.sqrt(252))
    return {"sign": int(np.sign(r)), "ret": r, "scaled": r / vol if vol > 0 else None}


def donchian(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 60:
        return {}
    hi55, lo55 = s.iloc[-56:-1].max(), s.iloc[-56:-1].min()
    hi20, lo20 = s.iloc[-21:-1].max(), s.iloc[-21:-1].min()
    x = float(s.iloc[-1])
    state = "BREAKOUT" if x >= hi55 else "BREAKDOWN" if x <= lo55 else "INSIDE"
    pos = (x - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
    return {"state": state, "pos20": float(min(1.0, max(0.0, pos)))}


def ewmac(s: pd.Series) -> float | None:
    try:
        from ..trend.ewmac import combine, forecasts
        f, _ = forecasts(s.dropna())
        score, _ = combine(f)
        v = score.dropna()
        return float(v.iloc[-1]) if len(v) else None
    except Exception:
        return None


def ema_trend(s: pd.Series) -> str | None:
    s = s.dropna()
    if len(s) < 210:
        return None
    e50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
    e200 = s.ewm(span=200, adjust=False).mean().iloc[-1]
    x = s.iloc[-1]
    return "UP" if x > e50 > e200 else "DOWN" if x < e50 < e200 else "MIXED"


def xs_ranks(levels: dict[str, pd.Series]) -> dict[str, dict]:
    r6 = {k: _ret(v, 126) for k, v in levels.items()}
    r12 = {k: _ret(v, 252) for k, v in levels.items()}
    out = {}
    for name, d in (("r6", r6), ("r12", r12)):
        vals = {k: v for k, v in d.items() if v is not None}
        srt = sorted(vals, key=vals.get)
        for i, k in enumerate(srt):
            out.setdefault(k, {})[name] = (i / (len(srt) - 1)) if len(srt) > 1 else 0.5
    return out


def composite(sig: dict) -> float | None:
    """-1..+1 blend of the four trend reads (equal weights) for the matrix colour."""
    parts = []
    if sig.get("tsmom", {}).get("sign") is not None:
        parts.append(float(sig["tsmom"]["sign"]))
    if sig.get("ewmac") is not None:
        parts.append(max(-1.0, min(1.0, sig["ewmac"] / 10.0)))
    if sig.get("xs", {}).get("r12") is not None:
        parts.append(2 * sig["xs"]["r12"] - 1)
    d = sig.get("donchian", {}).get("state")
    if d:
        parts.append({"BREAKOUT": 1.0, "BREAKDOWN": -1.0}.get(d, 0.0))
    return float(np.mean(parts)) if parts else None

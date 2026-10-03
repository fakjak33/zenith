"""Regime tags at decision time, stored with every guess for later slicing.

All computed from the ticker's FULL history up to (and including) the
decision bar -- never after it:
  * trend    -- above/below SMA-200 and the sign of its slope over 20 bars
  * vol      -- ATR-14 % of price, percentile within the asset's history so far
  * dist_high -- distance from the trailing 52-week high
  * rsi      -- RSI-14 bucket
"""

from __future__ import annotations

import numpy as np

from .indicators import atr, rsi, sma

BARS_PER_YEAR = {"1H": 252 * 7, "4H": 252 * 2, "Daily": 252, "Weekly": 52, "Monthly": 12}


def tags(o, h, l, c, t: int, tf: str = "Daily") -> dict:
    o, h, l, c = (np.asarray(a, float)[:t + 1] for a in (o, h, l, c))
    out: dict = {}
    s200 = sma(c, 200)
    if np.isfinite(s200[-1]):
        above = "above" if c[-1] > s200[-1] else "below"
        slope = "" if len(s200) <= 20 or not np.isfinite(s200[-21]) else (
            "↑" if s200[-1] > s200[-21] else "↓")
        out["trend"] = f"{above} SMA200 {slope}".strip()
    a = atr(h, l, c, 14) / c
    a_hist = a[np.isfinite(a)]
    if a_hist.size >= 60:
        pct = float((a_hist <= a_hist[-1]).mean() * 100)
        out["vol"] = "low vol" if pct < 33.3 else "high vol" if pct > 66.7 else "mid vol"
        out["vol_pct"] = round(pct, 1)
    yr = BARS_PER_YEAR.get(tf, 252)
    if len(h) >= min(yr, 20):
        d = float(c[-1] / h[-yr:].max() - 1)
        out["dist_high_pct"] = round(d * 100, 1)
        out["dist_high"] = ("at 52w high" if d > -0.02 else "-2% to -10%" if d > -0.10
                            else "-10% to -25%" if d > -0.25 else "below -25%")
    r = rsi(c, 14)[-1]
    if np.isfinite(r):
        out["rsi"] = "RSI <30" if r < 30 else "RSI 30-50" if r < 50 else "RSI 50-70" if r < 70 else "RSI >70"
    return out


def summary(tg: dict) -> str:
    return " · ".join(tg[k] for k in ("trend", "vol", "dist_high", "rsi") if tg.get(k))

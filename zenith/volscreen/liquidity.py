"""Option-market liquidity: how tradeable a chain is, and whether that is normal.

Level   option volume, open interest, stock ADV$.
Cost    relative bid/ask width (ask - bid) / mid at the money and at the 25 / 16
        delta wings of the ~30 DTE expiry — what a round trip actually costs.
Depth   listed expiries out to a year, weeklies, strikes within +-10%, quote size.

Stars (1-4) add up points per metric against config.VOLSCREEN_LIQ — in the
spirit of tastytrade's liquidity rating, on OUR thresholds.

Class (what kind of liquid):
  CORE    consistently liquid: >= 3 stars and stable daily volume (low CV)
  SURGE   today's option volume is a multiple of its own 20-day median —
          usually an event (earnings, news); until 5 sessions of option history
          exist, the proxy is the stock's share volume against its own 20-day median
  FADING  recent volume well below its 20-day norm
  THIN    1 star — wide markets, avoid
  ACTIVE  everything else
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import VOLSCREEN_LIQ as L


def _med(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _target_expiry(px: pd.DataFrame, lo: int = 20, hi: int = 50, target: int = 30):
    exps = px[["expiry", "dte"]].drop_duplicates()
    win = exps[(exps["dte"] >= lo) & (exps["dte"] <= hi)]
    pool = win if len(win) else exps[exps["dte"] >= 5]
    if pool.empty:
        return None
    return pool.iloc[(pool["dte"] - target).abs().argsort()].iloc[0]["expiry"]


def metrics(px: pd.DataFrame, spot: float, bars: pd.DataFrame | None) -> dict:
    """Today's liquidity metrics from a priced chain (iv.price_chain output)."""
    if px is None or px.empty:
        return {}
    out = {"opt_vol": float(np.nansum(px["volume"])), "oi": float(np.nansum(px["oi"]))}
    exps = px.loc[px["dte"] <= 365, "expiry"].unique()
    out["n_exp"] = int(len(exps))
    out["weeklies"] = bool(px.loc[(px["dte"] >= 0) & (px["dte"] <= 35), "expiry"].nunique() >= 4)
    e = _target_expiry(px)
    if e is not None:
        g = px[(px["expiry"] == e) & px["quoted"]]
        if len(g):
            F = float(g["F"].iloc[0])
            near = g.iloc[(g["strike"] - F).abs().argsort()].head(4)
            out["atm_spread"] = _med(near["spread_pct"])
            out["quote_size"] = _med(near["bid_size"].fillna(0) + near["ask_size"].fillna(0))
            for lab, dl in (("d25_spread", 0.25), ("d16_spread", 0.16)):
                w = g[g["otm"] & np.isfinite(g["d"])]
                if len(w):
                    pick = w.iloc[(w["d"].abs() - dl).abs().argsort()].head(2)
                    out[lab] = _med(pick["spread_pct"])
            ks = px.loc[px["expiry"] == e, "strike"].unique()
            out["strikes"] = int(np.sum(np.abs(ks / spot - 1.0) <= 0.10)) if spot else 0
            out["liq_dte"] = int(g["dte"].iloc[0])
    if bars is not None and len(bars):
        t = bars.tail(63)
        out["adv_usd"] = _med(t["close"] * t["volume"])
        if len(bars) > 21:
            base = _med(bars["volume"].iloc[-21:-1])
            out["stock_vol_x"] = float(bars["volume"].iloc[-1]) / base if base and base > 0 else float("nan")
    return out


def _pts(v, table) -> int:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0
    for cut, p in table:
        if v >= cut:
            return p
    return 0


def _pts_low(v, table) -> int:
    """Points for 'smaller is better' metrics (spreads)."""
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0
    for cut, p in table:
        if v <= cut:
            return p
    return 0


def points(m: dict, use_spread: bool = True) -> int:
    vol = m.get("opt_vol20") if m.get("opt_vol20") else m.get("opt_vol")
    p = _pts(vol, L["opt_vol"]) + _pts(m.get("oi"), L["oi"]) + _pts(m.get("strikes"), L["strikes"])
    if use_spread:
        p += _pts_low(m.get("atm_spread"), L["atm_spread"]) + _pts_low(m.get("d25_spread"), L["d25_spread"])
    return p


def stars(m: dict, use_spread: bool = True) -> int:
    """1-4 stars. `use_spread=False` for post-close snapshots, whose quotes are
    pulled wide and say nothing about intraday trading cost."""
    p = points(m, use_spread)
    for cut, s in L["stars" if use_spread else "stars_no_spread"]:
        if p >= cut:
            return s
    return 1


def with_history(m: dict, past_vol: list[float]) -> dict:
    """Add 20-day average volume, its CV and the surge / fade ratios from the
    stored daily option-volume history (oldest first, today excluded)."""
    out = dict(m)
    h = [float(x) for x in past_vol if x is not None and math.isfinite(float(x))][-20:]
    today = m.get("opt_vol")
    if len(h) >= 5:
        allv = h + ([today] if today is not None else [])
        out["opt_vol20"] = float(np.mean(allv[-20:]))
        mu = float(np.mean(h))
        out["cv20"] = float(np.std(h, ddof=1) / mu) if mu > 0 and len(h) > 2 else float("nan")
        med = float(np.median(h))
        out["surge_x"] = today / med if med > 0 and today is not None else float("nan")
        last5 = allv[-5:]
        out["fade_x"] = float(np.mean(last5)) / mu if mu > 0 else float("nan")
        out["hist_days"] = len(h)
    else:
        out["hist_days"] = len(h)
    return out


def classify(m: dict, turnover_median: float | None = None) -> str:
    s = m.get("stars", stars(m))
    if s <= 1:
        return "THIN"
    sx = m.get("surge_x")
    if sx is None or not math.isfinite(sx):
        # before 5 sessions of option-volume history: the underlying's own share
        # volume against its 20-day median is the per-name event proxy
        sx = m.get("stock_vol_x")
        if sx is not None and math.isfinite(sx):
            m["surge_proxy"] = True
    if sx is not None and math.isfinite(sx) and sx >= L["surge_x"]:
        return "SURGE"
    fx = m.get("fade_x")
    if fx is not None and math.isfinite(fx) and fx <= L["fade_x"]:
        return "FADING"
    cv = m.get("cv20")
    if s >= 3 and (cv is None or not math.isfinite(cv) or cv <= L["core_cv"]):
        return "CORE"
    return "ACTIVE"


def score(m: dict, use_spread: bool = True) -> float:
    """Continuous liquidity score (for ranking within a star band)."""
    vol = m.get("opt_vol20") or m.get("opt_vol") or 0.0
    oi = m.get("oi") or 0.0
    spr = m.get("atm_spread")
    spr = spr if use_spread and spr is not None and math.isfinite(spr) else 0.0
    return math.log10(1 + vol) + 0.5 * math.log10(1 + oi) - 8.0 * min(spr, 0.25)

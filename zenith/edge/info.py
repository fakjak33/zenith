"""yfinance short-interest + analyst-revision fetchers for the edge screens.

All network access for the SI / revisions screens lives here. Cloned from the
fmom fundamentals-cache pattern: store_cas cache, polite sleep, 200-ticker
checkpoints, never raises. Short interest is FINRA-biweekly so it caches 7d;
revisions cache 3d.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from ..cas import store_cas

_SI_KEY = "edge_shortinterest"
_SI_TTL = 168.0          # 7 days
_REV_KEY = "edge_revisions"
_REV_TTL = 72.0          # 3 days

_SI_FIELDS = ("sharesShort", "shortRatio", "shortPercentOfFloat",
              "floatShares", "marketCap", "recommendationMean",
              "numberOfAnalystOpinions")


def short_interest(tickers: list[str], sleep: float = 0.12,
                   max_age_hours: float = _SI_TTL) -> dict[str, dict]:
    """ticker -> {si_float(%), dtc(days), shares_short, float_shares, mktcap,
    reco_mean, n_analysts}. One .info call each, cached, best-effort."""
    cached = store_cas.cache_get(_SI_KEY, max_age_hours) or {}
    todo = [t for t in tickers if t not in cached]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        for i, t in enumerate(todo):
            try:
                info = yf.Ticker(t).info or {}
                spf = info.get("shortPercentOfFloat")
                cached[t] = {
                    "si_float": round(spf * 100, 3) if spf is not None else None,
                    "dtc": info.get("shortRatio"),
                    "shares_short": info.get("sharesShort"),
                    "float_shares": info.get("floatShares"),
                    "mktcap": info.get("marketCap"),
                    "reco_mean": info.get("recommendationMean"),
                    "n_analysts": info.get("numberOfAnalystOpinions"),
                }
            except Exception:
                cached[t] = {}
            if sleep:
                time.sleep(sleep)
            if i and i % 200 == 0:
                store_cas.cache_put(_SI_KEY, cached)
                print(f"[edge] short-interest {i}/{len(todo)}")
        store_cas.cache_put(_SI_KEY, cached)
    return cached


def _est_rev_pct(tk) -> float | None:
    """% change in the current-FY (+1y) mean EPS estimate, current vs 90d ago."""
    try:
        et = tk.get_eps_trend()
    except Exception:
        return None
    if et is None or getattr(et, "empty", True):
        return None
    try:
        row = et.loc["+1y"] if "+1y" in et.index else et.loc[et.index[-1]]
        cur = row.get("current")
        old = row.get("90daysAgo", row.get("30daysAgo"))
        if cur is None or old is None or pd.isna(cur) or pd.isna(old) or old == 0:
            return None
        return round(float((cur - old) / abs(old)), 5)
    except Exception:
        return None


def _up_frac(tk) -> float | None:
    """Fraction of last-30d EPS revisions that were UP (direction consistency)."""
    try:
        er = tk.get_eps_revisions()
    except Exception:
        return None
    if er is None or getattr(er, "empty", True):
        return None
    try:
        row = er.loc["+1y"] if "+1y" in er.index else er.loc[er.index[-1]]
        up = float(row.get("upLast30days", 0) or 0)
        down = float(row.get("downLast30days", 0) or 0)
        tot = up + down
        return round(up / tot, 4) if tot > 0 else None
    except Exception:
        return None


def _net_reco(tk, since: date) -> float | None:
    """Net upgrades - downgrades over the last 30d, scaled by action count."""
    try:
        ud = tk.upgrades_downgrades
    except Exception:
        return None
    if ud is None or getattr(ud, "empty", True):
        return None
    try:
        recent = ud[ud.index >= pd.Timestamp(since)]
        if recent.empty:
            return None
        grades = recent.get("Action")
        if grades is None:
            return None
        up = int((grades == "up").sum())
        down = int((grades == "down").sum())
        tot = len(recent)
        return round((up - down) / tot, 4) if tot else None
    except Exception:
        return None


def revisions(tickers: list[str], sleep: float = 0.15,
              max_age_hours: float = _REV_TTL) -> dict[str, dict]:
    """ticker -> {est_rev_pct, up_frac, net_reco}. ~3 yfinance calls each;
    cached 3d, checkpointed, best-effort."""
    cached = store_cas.cache_get(_REV_KEY, max_age_hours) or {}
    todo = [t for t in tickers if t not in cached]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        since = date.today() - timedelta(days=30)
        for i, t in enumerate(todo):
            try:
                tk = yf.Ticker(t)
                cached[t] = {"est_rev_pct": _est_rev_pct(tk),
                             "up_frac": _up_frac(tk),
                             "net_reco": _net_reco(tk, since)}
            except Exception:
                cached[t] = {}
            if sleep:
                time.sleep(sleep)
            if i and i % 200 == 0:
                store_cas.cache_put(_REV_KEY, cached)
                print(f"[edge] revisions {i}/{len(todo)}")
        store_cas.cache_put(_REV_KEY, cached)
    return cached

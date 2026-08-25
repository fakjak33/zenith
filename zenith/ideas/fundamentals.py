"""IDEAS fundamentals: a COMMITTED rolling .info cache for the whole scan universe.

fmom.families.replication.get_fundamentals() already pulls the right yfinance
.info fields, but it writes to cas.store_cas, and data/cas/cache/ is
gitignored — CI starts cold every run. A ~1000-ticker .info sweep at
~1.2-1.5s/ticker will not fit inside a 45-minute Action budget from zero.

This module follows mom.universe.refresh_metadata's proven pattern instead: a
COMMITTED data/ideas/fundamentals.json, 30-day TTL, at most META_MAX_PER_RUN
of the stalest tickers refreshed per run, checkpointed every 200. A cold start
fills the whole index over ~1 week and then self-maintains — mom/meta.json did
exactly this and self-healed from cold within the observed range.
"""

from __future__ import annotations

import time
from datetime import date

from . import load, save

META_TTL_DAYS = 30
META_MAX_PER_RUN = 150

# Superset of fmom's replication fields plus what valuation.py / groups.py need
# for quality, growth, leverage, ownership and the consensus-vs-Zenith check
# (spec §18 needs targetMeanPrice + recommendationMean to be real, not invented).
INFO_FIELDS = (
    "marketCap", "priceToBook", "trailingPE", "forwardPE",
    "priceToSalesTrailing12Months", "operatingCashflow", "freeCashflow",
    "dividendYield", "returnOnEquity", "returnOnAssets", "profitMargins",
    "grossMargins", "ebitdaMargins", "revenueGrowth", "earningsGrowth",
    "debtToEquity", "totalDebt", "totalCash", "currentRatio",
    "enterpriseToEbitda", "totalRevenue", "ebitda", "sharesOutstanding",
    "recommendationMean", "numberOfAnalystOpinions",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "heldPercentInstitutions", "heldPercentInsiders", "beta",
)


def _stale(entry: dict | None, today: date, ttl_days: int = META_TTL_DAYS) -> bool:
    if not entry or not entry.get("asof"):
        return True
    try:
        asof = date.fromisoformat(entry["asof"])
    except Exception:
        return True
    return (today - asof).days > ttl_days


def refresh(tickers: list[str], max_per_run: int = META_MAX_PER_RUN,
           ttl_days: int = META_TTL_DAYS, sleep: float = 0.12) -> dict:
    """Best-effort fundamentals cache. Costs ~1-1.5s/ticker via yfinance
    .info, so only the stalest `max_per_run` tickers refresh per run. Result
    is COMMITTED (data/ideas/fundamentals.json), never gitignore-cached —
    a CI runner starts cold every job and this must not restart from zero."""
    today = date.today()
    fund = load("fundamentals", {})
    stale = [t for t in tickers if _stale(fund.get(t), today, ttl_days)]
    todo = stale[:max_per_run]
    if not todo:
        return {"checked": len(tickers), "stale": 0, "refreshed": 0}
    try:
        import yfinance as yf
    except Exception:
        return {"checked": len(tickers), "stale": len(stale), "refreshed": 0,
                "error": "yfinance unavailable"}
    refreshed = 0
    for i, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info or {}
            row = {f: info.get(f) for f in INFO_FIELDS}
            row["asof"] = today.isoformat()
            fund[t] = row
            refreshed += 1
        except Exception:
            pass
        if sleep:
            time.sleep(sleep)
        if i and i % 200 == 0:
            save("fundamentals", fund)
            print(f"[ideas] fundamentals {i}/{len(todo)}")
    save("fundamentals", fund)
    return {"checked": len(tickers), "stale": len(stale), "refreshed": refreshed}


def get(tickers: list[str]) -> dict[str, dict]:
    """Ticker -> cached .info row (may be stale/partial; refresh() maintains
    freshness). Missing tickers are simply absent, never fabricated."""
    fund = load("fundamentals", {})
    return {t: fund[t] for t in tickers if t in fund}

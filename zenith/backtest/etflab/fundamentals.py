"""Fundamental SNAPSHOT data (class B) — current values only, no history.

yfinance exposes today's market cap, P/E, margins, sector, etc., but not a
point-in-time history. So anything here is a *snapshot* held constant across a
backtest → look-ahead/survivorship bias. ``market_caps`` powers optional
market-cap weighting (Phase 1, flagged); the richer ``snapshot`` /
``factor_series`` / ``classify`` power the Phase 2 fundamental screens.
Everything degrades to NaN/"" and never raises.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import DATA_CACHE

_CACHE = DATA_CACHE / "_fundamentals.parquet"   # data/backtest/cache (gitignored)

# Numeric fundamental fields (held as floats in the cache).
_NUM_FIELDS = ("market_cap", "shares", "trailing_pe", "forward_pe",
               "price_to_book", "profit_margin", "dividend_yield", "beta",
               "revenue", "free_cash_flow", "fcf_per_share")
# Categorical fundamental fields (held as strings).
_STR_FIELDS = ("sector", "industry")
_FIELDS = _NUM_FIELDS + _STR_FIELDS

# Maps our field name -> the yfinance ``.info`` key it comes from.
_INFO_MAP = {
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "profit_margin": "profitMargins",
    "dividend_yield": "dividendYield",
    "beta": "beta",
    "revenue": "totalRevenue",
    "free_cash_flow": "freeCashflow",
    "sector": "sector",
    "industry": "industry",
}


def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _load_cache() -> pd.DataFrame:
    if _CACHE.exists():
        try:
            return pd.read_parquet(_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=list(_FIELDS) + ["fetched_at"])


def _save_cache(df: pd.DataFrame) -> None:
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE)
    except Exception:
        pass


def _fetch_one(ticker: str) -> dict:
    """Best-effort single-ticker snapshot via yfinance fast_info + info."""
    import yfinance as yf

    rec = {f: np.nan for f in _NUM_FIELDS}
    rec.update({f: "" for f in _STR_FIELDS})
    tk = yf.Ticker(ticker)

    # fast_info is cheap and reliable for market cap / shares.
    try:
        fi = tk.fast_info
        rec["market_cap"] = _num(fi.get("market_cap"))
        rec["shares"] = _num(fi.get("shares"))
    except Exception:
        pass

    # .info is slower / rate-limited but carries the richer fundamentals.
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    for field, key in _INFO_MAP.items():
        val = info.get(key)
        if field in _STR_FIELDS:
            rec[field] = str(val) if val else ""
        else:
            rec[field] = _num(val)

    # Derive FCF / share when both pieces are present.
    shares = rec.get("shares") or _num(info.get("sharesOutstanding"))
    if not np.isfinite(rec["shares"]) and np.isfinite(_num(shares)):
        rec["shares"] = _num(shares)
    fcf, sh = rec.get("free_cash_flow"), rec.get("shares")
    if np.isfinite(fcf) and np.isfinite(sh) and sh > 0:
        rec["fcf_per_share"] = fcf / sh
    return rec


def snapshot(tickers, max_age_days: int = 30, fields=None) -> pd.DataFrame:
    """Return a snapshot DataFrame (tickers x fields), served from cache when
    fresh enough, otherwise fetched (slow, rate-limited) and cached."""
    tickers = [t.upper() for t in tickers]
    cache = _load_cache()
    now = time.time()
    fresh_cut = now - max_age_days * 86400
    out = {}
    to_fetch = []
    for t in tickers:
        if t in cache.index and float(cache.loc[t].get("fetched_at", 0) or 0) >= fresh_cut:
            out[t] = cache.loc[t].to_dict()
        else:
            to_fetch.append(t)
    for t in to_fetch:
        rec = _fetch_one(t)
        rec["fetched_at"] = now
        out[t] = rec
        cache.loc[t] = rec
    if to_fetch:
        _save_cache(cache)
    df = pd.DataFrame(out).T
    cols = list(fields) if fields else list(_FIELDS)
    keep = [c for c in cols + ["fetched_at"] if c in df.columns] or cols
    return df.reindex(columns=cols + ["fetched_at"])


def factor_series(tickers, field: str, max_age_days: int = 30) -> pd.Series:
    """One numeric fundamental field as a ticker->value Series (NaN-degrading)."""
    try:
        df = snapshot(tickers, max_age_days=max_age_days, fields=[field])
        return pd.to_numeric(df[field], errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=[t.upper() for t in tickers])


def classify(tickers, field: str = "sector", max_age_days: int = 30) -> pd.Series:
    """A categorical fundamental field (sector/industry) as a ticker->str Series."""
    try:
        df = snapshot(tickers, max_age_days=max_age_days, fields=[field])
        return df[field].astype(str).fillna("")
    except Exception:
        return pd.Series("", index=[t.upper() for t in tickers])


def market_caps(tickers) -> pd.Series:
    """Snapshot market caps (NaN where unavailable). Used for cap weighting."""
    return factor_series(tickers, "market_cap")

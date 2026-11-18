"""Real market data via yfinance with a local parquet cache.

A single ticker maps to one parquet file under ``data/backtest/cache/``.
Fixed vs PARALLAX: files older than ``CACHE_TTL_HOURS`` are re-downloaded (the
old cache never refreshed). If the download fails the stale file is used, and
failing that the EPHEMERIS daily panel when it is already on disk.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import DATA_CACHE, CACHE_TTL_HOURS

# Canonical OHLCV column order used throughout the lab.
OHLCV = ["open", "high", "low", "close", "volume"]

# Pandas resample rules for the UI timeframe selector.
TIMEFRAMES = {"Daily": "D", "Weekly": "W-FRI", "Monthly": "ME"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to a coarser timeframe (e.g. 'W-FRI', 'ME')."""
    if rule in (None, "D"):
        return df
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    use = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(use).dropna(how="any")


import re

# yfinance tickers are letters/digits plus a few symbols (e.g. BRK-B, ^GSPC,
# EURUSD=X). Whitelist those only — this also blocks path traversal in the
# cache filename (no slashes, dots-dots, backslashes get through).
_TICKER_RE = re.compile(r"[^A-Z0-9.\-^=]")
_MAX_TICKER_LEN = 15


def clean_ticker(ticker: str) -> str:
    """Uppercase and strip a ticker to safe characters; raise if nothing valid."""
    t = _TICKER_RE.sub("", str(ticker).upper())[:_MAX_TICKER_LEN]
    # collapse any run of dots to a single dot and trim leading/trailing dots
    # (defense in depth against path-traversal artifacts like '..')
    t = re.sub(r"\.+", ".", t).strip(".")
    if not t or t.strip("-^=") == "":
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return t


def _cache_path(ticker: str) -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    safe = clean_ticker(ticker).replace("/", "_")
    path = (DATA_CACHE / f"{safe}.parquet").resolve()
    # ensure the resolved path stays inside the cache directory
    if DATA_CACHE.resolve() not in path.parents:
        raise ValueError(f"Unsafe cache path for ticker {ticker!r}")
    return path


def _download(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    # When no explicit start is given, yfinance defaults to a 1-month window.
    # Use period="max" to cache full available history instead.
    kwargs = dict(auto_adjust=True, progress=False, threads=False)
    if start is None and end is None:
        kwargs["period"] = "max"
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    raw = yf.download(ticker, **kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r}.")

    # yfinance may return a MultiIndex (field, ticker) for single symbols too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    cols = [c for c in OHLCV if c in raw.columns]
    out = raw[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    return out


def prewarm(tickers, force_refresh: bool = False) -> None:
    """Batch-download any UNCACHED tickers in a single yfinance call and write
    each to its parquet cache. Big speed-up vs the per-ticker path when filling
    a large universe (e.g. the ETF Lab). Best-effort: failures are ignored so
    callers fall back to the normal per-ticker fetch.
    """
    import yfinance as yf

    want = []
    for t in tickers:
        try:
            ct = clean_ticker(t)
        except ValueError:
            continue
        if force_refresh or not _cache_path(ct).exists():
            want.append(ct)
    want = list(dict.fromkeys(want))
    if len(want) < 2:
        return
    try:
        raw = yf.download(want, period="max", auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
    except Exception:
        return
    if raw is None or raw.empty:
        return
    for ct in want:
        try:
            sub = raw[ct] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub.rename(columns=str.lower)
            cols = [c for c in OHLCV if c in sub.columns]
            out = sub[cols].dropna(how="all")
            if out.empty:
                continue
            out.index = pd.to_datetime(out.index).tz_localize(None)
            out.index.name = "date"
            out.to_parquet(_cache_path(ct))
        except Exception:
            continue


def get_prices(
    ticker: str,
    start: str | None = "2010-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted OHLCV for ``ticker`` as a DataFrame indexed by date.

    Cached to parquet. On a cache hit the full cached history is returned and
    then sliced to ``[start, end]`` so different date windows share one file.
    """
    ticker = clean_ticker(ticker)
    path = _cache_path(ticker)

    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_HOURS * 3600
    if fresh and not force_refresh:
        df = pd.read_parquet(path)
    else:
        try:
            df = _download(ticker, start=None, end=None)  # cache full history
            df.to_parquet(path)
        except Exception:
            if path.exists():
                df = pd.read_parquet(path)            # stale beats nothing
            else:
                df = _ephemeris_fallback(ticker)
                if df is None:
                    raise

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _ephemeris_fallback(ticker: str) -> pd.DataFrame | None:
    """Daily bars from the EPHEMERIS panel if it is already on disk (no download)."""
    try:
        from ...config import EPHEMERIS_PX_DIR
        from ...ephemeris.store_px import PxStore
        if not (EPHEMERIS_PX_DIR / "daily.parquet").exists():
            return None
        df = PxStore(root=EPHEMERIS_PX_DIR).bars(ticker, "Daily")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df


def get_panel(
    tickers: list[str],
    start: str | None = "2010-01-01",
    end: str | None = None,
    field: str = "close",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide DataFrame of one ``field`` (default close) across tickers.

    Columns are tickers; rows are dates aligned on the union of trading days.
    Tickers that fail to download are skipped with a warning column omitted.
    """
    prewarm(tickers, force_refresh=force_refresh)   # batch-fill the cache first
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            df = get_prices(t, start=start, end=end, force_refresh=force_refresh)
            series[t.upper()] = df[field]
        except Exception as exc:  # noqa: BLE001 — surface per-ticker, keep going
            errors[t.upper()] = str(exc)
            time.sleep(0.1)

    if not series:
        raise ValueError(f"No tickers could be loaded. Errors: {errors}")

    panel = pd.DataFrame(series).sort_index()
    panel.attrs["errors"] = errors
    return panel

"""Real market data via yfinance with a local parquet cache.

A single ticker maps to one parquet file under ``data/backtest/cache/``.
Fixed vs PARALLAX: files older than ``CACHE_TTL_HOURS`` are re-downloaded (the
old cache never refreshed). If the download fails the stale file is used, and
failing that the EPHEMERIS daily panel when it is already on disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
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


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_HOURS * 3600


def _dead_path() -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE / "_no_data.json"


def _dead() -> dict:
    """Tickers yfinance returned nothing for, with when — skipped for CACHE_TTL_HOURS."""
    try:
        d = json.loads(_dead_path().read_text())
        cut = time.time() - CACHE_TTL_HOURS * 3600
        return {k: v for k, v in d.items() if v >= cut}
    except Exception:  # noqa: BLE001
        return {}


def _mark_dead(tickers) -> None:
    if not tickers:
        return
    try:
        d = _dead()
        d.update({t: time.time() for t in tickers})
        _dead_path().write_text(json.dumps(d))
    except Exception:  # noqa: BLE001
        pass


def prewarm(tickers, force_refresh: bool = False, chunk: int = 150) -> set[str]:
    """Batch-download every ticker that is uncached OR stale, ``chunk`` symbols per yfinance
    call, writing each to its parquet cache. Returns the set it attempted, so callers can skip a
    second (serial) download for names the batch couldn't fill. Best-effort: never raises."""
    want = []
    for t in tickers:
        try:
            ct = clean_ticker(t)
        except ValueError:
            continue
        if force_refresh or not _is_fresh(_cache_path(ct)):
            want.append(ct)
    dead = {} if force_refresh else _dead()
    skipped = {t for t in want if t in dead}
    want = [t for t in dict.fromkeys(want) if t not in dead]
    if not want:
        return skipped
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return skipped
    for i in range(0, len(want), chunk):
        part = want[i:i + chunk]
        try:
            raw = yf.download(part if len(part) > 1 else part[0], period="max", auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
        except Exception:  # noqa: BLE001
            continue
        if raw is None or raw.empty:
            continue
        for ct in part:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ct not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[ct]
                else:
                    sub = raw
                sub = sub.rename(columns=str.lower)
                cols = [c for c in OHLCV if c in sub.columns]
                out = sub[cols].dropna(how="all")
                if out.empty:
                    continue
                out.index = pd.to_datetime(out.index).tz_localize(None)
                out.index.name = "date"
                out.to_parquet(_cache_path(ct))
            except Exception:  # noqa: BLE001
                continue
    _mark_dead([t for t in want if not _cache_path(t).exists()])
    return set(want) | skipped


def get_prices(
    ticker: str,
    start: str | None = "2010-01-01",
    end: str | None = None,
    force_refresh: bool = False,
    download: bool = True,
) -> pd.DataFrame:
    """Return adjusted OHLCV for ``ticker`` as a DataFrame indexed by date.

    ``download=False`` never touches the network: a stale cache file (or the EPHEMERIS store)
    is used as-is.

    Cached to parquet. On a cache hit the full cached history is returned and
    then sliced to ``[start, end]`` so different date windows share one file.
    """
    ticker = clean_ticker(ticker)
    path = _cache_path(ticker)

    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_HOURS * 3600
    if (fresh and not force_refresh) or (not download and path.exists()):
        df = pd.read_parquet(path)
    elif not download:
        df = _ephemeris_fallback(ticker)
        if df is None:
            raise ValueError(f"{ticker}: not cached")
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


def _store_panel(tickers: list[str], field: str, max_lag_days: int = 6) -> pd.DataFrame:
    """Wide ``field`` panel for ``tickers`` from the EPHEMERIS daily store (same yfinance
    auto-adjusted prices, published daily as a release asset), read column-wise with a ticker
    filter. Names whose history ends more than ``max_lag_days`` before the store's last bar are
    dropped so the caller refreshes them. Empty frame when the store isn't available."""
    try:
        import pyarrow.parquet as pq

        from ...ephemeris.store_px import ensure_local
        path = ensure_local("daily.parquet")
        if path is None:
            return pd.DataFrame()
        want = sorted({t.upper() for t in tickers})
        tbl = pq.read_table(path, columns=["ticker", "ts", field], filters=[("ticker", "in", want)])
        long = tbl.to_pandas()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if long.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(long["ts"]).dt.tz_localize(None).dt.normalize()
    di, dates = pd.factorize(ts, sort=True)
    ti, names = pd.factorize(long["ticker"], sort=True)
    grid = np.full((len(dates), len(names)), np.nan)
    grid[di, ti] = long[field].to_numpy(float)          # last write wins on duplicates
    wide = pd.DataFrame(grid, index=pd.DatetimeIndex(dates), columns=list(names))
    wide.index.name = "date"
    last = wide.apply(lambda c: c.last_valid_index())
    cut = wide.index.max() - pd.Timedelta(days=max_lag_days)
    keep = [c for c in wide.columns if last[c] is not None and last[c] >= cut]
    # the store itself must be current, or yfinance is the better source
    if wide.index.max() < pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
        return pd.DataFrame()
    return wide[keep]


def get_panel(
    tickers: list[str],
    start: str | None = "2010-01-01",
    end: str | None = None,
    field: str = "close",
    force_refresh: bool = False,
    use_store: bool = True,
) -> pd.DataFrame:
    """Return a wide DataFrame of one ``field`` (default close) across tickers.

    Sources, fastest first: the EPHEMERIS daily store (one columnar read for hundreds of
    names), then ONE batched yfinance pass for anything missing or stale, then the parquet
    cache. No ticker is downloaded twice and nothing is fetched serially. Tickers that can't be
    loaded are listed in ``panel.attrs["errors"]``.
    """
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    want = list(dict.fromkeys(t.upper() for t in tickers))
    store = _store_panel(want, field) if (use_store and not force_refresh and field in OHLCV) else pd.DataFrame()
    for t in store.columns:
        series[t] = store[t].dropna()
    rest = [t for t in want if t not in series]
    tried = prewarm(rest, force_refresh=force_refresh) if rest else set()
    for t in rest:
        try:
            ct = clean_ticker(t)
            df = get_prices(ct, start=None, end=None, force_refresh=False, download=ct not in tried)
            series[t] = df[field]
        except Exception as exc:  # noqa: BLE001 — surface per-ticker, keep going
            errors[t] = str(exc)

    if not series:
        raise ValueError(f"No tickers could be loaded. Errors: {errors}")

    panel = pd.DataFrame(series).sort_index()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    panel = panel.dropna(how="all")
    panel.attrs["errors"] = errors
    panel.attrs["from_store"] = len(store.columns)
    return panel

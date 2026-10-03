"""Nightly price prefetch -> Parquet (published as a GitHub Release asset).

    python -m zenith.ephemeris.prefetch                 # full universe, daily
    python -m zenith.ephemeris.prefetch --limit 60      # quick dev subset

Writes EPHEMERIS_PX_DIR/daily.parquet: columns ticker, ts, open, high, low,
close, volume (float32), sorted by ticker then ts, ONE ROW GROUP PER TICKER so
the app can read a single ticker in milliseconds (store_px.py) without loading
the whole file. Prices are yfinance `auto_adjust=True` -- split- AND
dividend-adjusted OHLC.

Not ZENITH's shared cas.sources.prices.get_history: that one has no retries
and rewrites its per-period cache with only the tickers last requested. This
job downloads directly, in polite batches, with retries and backoff.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EPHEMERIS_FILES, EPHEMERIS_PX_DIR
from . import universe as uni
from .benchmarks import build_base_rates

PX_COLS = ("open", "high", "low", "close", "volume")
BATCH = 40
RETRIES = 4
MIN_BARS = 150


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """One ticker's yfinance frame -> clean float32 OHLCV indexed by naive ts."""
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)
    if not set(PX_COLS) - {"volume"} <= set(df.columns):
        return None
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[list(PX_COLS)].astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # repair inverted bars from bad prints rather than dropping history
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0)
    return df.astype("float32") if len(df) >= MIN_BARS else None


def _download(batch: list[str], interval: str, period: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    for attempt in range(RETRIES):
        try:
            raw = yf.download(batch, period=period, interval=interval, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
            out = {}
            for t in batch:
                try:
                    sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                df = _normalize(sub)
                if df is not None:
                    out[t] = df
            if out or attempt == RETRIES - 1:
                return out
        except Exception as exc:                      # network / rate limit
            print(f"  batch failed ({type(exc).__name__}: {exc}); retry {attempt + 1}")
        time.sleep((2 ** attempt) * 3 + random.random() * 2)
    return {}


def fetch(tickers: list[str], interval: str = "1d", period: str = "max") -> dict[str, pd.DataFrame]:
    got: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]
    for i, b in enumerate(batches, 1):
        got.update(_download(b, interval, period))
        print(f"[{i}/{len(batches)}] {len(got)} tickers so far")
        time.sleep(1.0 + random.random())
    missing = [t for t in tickers if t not in got]
    if missing:                                       # one slower second pass, small batches
        print(f"second pass for {len(missing)} missing")
        for b in [missing[i:i + 10] for i in range(0, len(missing), 10)]:
            got.update(_download(b, interval, period))
            time.sleep(2.0)
    return got


def write_parquet(frames: dict[str, pd.DataFrame], path) -> int:
    """One row group per ticker, sorted by ticker -> O(1) single-ticker reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("ticker", pa.string()), ("ts", pa.timestamp("s"))]
                       + [(c, pa.float32()) for c in PX_COLS])
    tmp = path.with_suffix(".tmp")
    n = 0
    # BYTE_STREAM_SPLIT + zstd is ~2.2x smaller than plain zstd on float prices
    enc = {c: "BYTE_STREAM_SPLIT" for c in PX_COLS} | {"ts": "DELTA_BINARY_PACKED"}
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=6,
                          use_dictionary=["ticker"], column_encoding=enc) as w:
        for t in sorted(frames):
            df = frames[t]
            tbl = pa.table({"ticker": pa.array([t] * len(df), pa.string()),
                            "ts": pa.array(df.index.values.astype("datetime64[s]")),
                            **{c: pa.array(df[c].to_numpy(np.float32)) for c in PX_COLS}},
                           schema=schema)
            w.write_table(tbl, row_group_size=len(df) + 1)
            n += 1
    tmp.replace(path)
    return n


def write_base_rates(daily: dict, rows: list[dict], hourly: dict | None = None) -> dict:
    from .benchmarks import HOURLY_RULES
    from .store_px import resample
    classes = {r["ticker"]: r["cls"] for r in rows}
    table = build_base_rates({t: df.astype("float64") for t, df in daily.items()}, classes, resample)
    if hourly:
        table.update(build_base_rates({t: df.astype("float64") for t, df in hourly.items()}, classes,
                                      resample, HOURLY_RULES))
    EPHEMERIS_FILES["base_rates"].write_text(json.dumps(table, indent=0), encoding="utf-8")
    return table


def read_parquet_frames(path) -> dict[str, pd.DataFrame]:
    """Whole file -> {ticker: frame}, fully in memory (no open handle left behind,
    so the same path can be rewritten afterwards -- matters on Windows)."""
    import pyarrow.parquet as pq
    if path is None or not Path(path).exists():
        return {}
    df = pq.read_table(path).to_pandas()
    out = {}
    for t, g in df.groupby("ticker", sort=False):
        out[t] = g.drop(columns="ticker").set_index("ts").sort_index()
    return out


def merge_frames(old: dict, new: dict) -> dict:
    """Append-only history: yfinance serves ~730 days of 1H bars, so keep the
    older bars from the previous release and let the depth grow nightly."""
    out = dict(old)
    for t, df in new.items():
        if t in out:
            cat = pd.concat([out[t], df])
            out[t] = cat[~cat.index.duplicated(keep="last")].sort_index().astype("float32")
        else:
            out[t] = df
    return out


def base_rates_from_store() -> dict:
    """Rebuild base_rates.json from the local Parquet (no network)."""
    rows = uni.load()
    return write_base_rates(read_parquet_frames(EPHEMERIS_PX_DIR / "daily.parquet"), rows,
                            read_parquet_frames(EPHEMERIS_PX_DIR / "hourly.parquet"))


def _subset(rows: list[dict], limit: int | None) -> list[str]:
    if not limit:
        return [r["ticker"] for r in rows]
    by_cls: dict[str, list[str]] = {}                  # balanced dev subset: a few per class
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r["ticker"])
    per = max(2, limit // max(1, len(by_cls)))
    return [t for ts in by_cls.values() for t in ts[:per]]


def run(limit: int | None = None, hourly: bool = True) -> dict:
    from .store_px import ensure_local
    rows = uni.write()
    tickers = _subset(rows, limit)
    t0 = time.time()
    daily = fetch(tickers, "1d", "max")
    n = write_parquet(daily, EPHEMERIS_PX_DIR / "daily.parquet")
    status = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              "daily": {"requested": len(tickers), "written": n,
                        "missing": sorted(set(tickers) - set(daily))[:200]}}
    hourly_frames = None
    if hourly:
        prev = read_parquet_frames(ensure_local("hourly.parquet", max_age_h=1e9))
        hourly_frames = merge_frames(prev, fetch(tickers, "1h", "730d"))
        nh = write_parquet(hourly_frames, EPHEMERIS_PX_DIR / "hourly.parquet")
        depth = [len(df) for df in hourly_frames.values()]
        status["hourly"] = {"written": nh, "kept_from_previous": len(prev),
                            "median_bars": int(np.median(depth)) if depth else 0}
    write_base_rates(daily, rows, hourly_frames)
    from . import daily as daily_five
    from .store_px import PxStore
    added = daily_five.extend_schedule(PxStore(root=EPHEMERIS_PX_DIR), rows)
    status["daily_five_days_added"] = added
    status["seconds"] = round(time.time() - t0, 1)
    (EPHEMERIS_PX_DIR / "manifest.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    EPHEMERIS_FILES["status"].write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "daily"} | {"written": n}))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-hourly", action="store_true", help="skip the 1H download")
    ap.add_argument("--base-rates-only", action="store_true", help="rebuild base rates from local Parquet")
    a = ap.parse_args()
    base_rates_from_store() if a.base_rates_only else run(a.limit, hourly=not a.no_hourly)

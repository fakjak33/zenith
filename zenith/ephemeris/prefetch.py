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

import numpy as np
import pandas as pd

from ..config import EPHEMERIS_FILES, EPHEMERIS_PX_DIR
from . import universe as uni

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


def run(limit: int | None = None) -> dict:
    rows = uni.write()
    tickers = [r["ticker"] for r in rows]
    if limit:                                          # balanced dev subset: a few per class
        by_cls: dict[str, list[str]] = {}
        for r in rows:
            by_cls.setdefault(r["cls"], []).append(r["ticker"])
        per = max(2, limit // max(1, len(by_cls)))
        tickers = [t for ts in by_cls.values() for t in ts[:per]]
    t0 = time.time()
    frames = fetch(tickers, "1d", "max")
    n = write_parquet(frames, EPHEMERIS_PX_DIR / "daily.parquet")
    status = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              "daily": {"requested": len(tickers), "written": n,
                        "missing": sorted(set(tickers) - set(frames))[:200]},
              "seconds": round(time.time() - t0, 1)}
    (EPHEMERIS_PX_DIR / "manifest.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    EPHEMERIS_FILES["status"].write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "daily"} | {"written": n}))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    run(ap.parse_args().limit)

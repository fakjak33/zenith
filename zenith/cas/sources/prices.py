"""OHLCV price history for the CAS universe (free, via yfinance).

Returns a dict of ticker -> pandas DataFrame with columns
[open, high, low, close, volume] indexed by date. Results are cached to
data/cas/cache/prices.json for the day so reruns don't re-download.
"""

from __future__ import annotations

import io

import pandas as pd

from .. import store_cas


def get_history(tickers: list[str], period: str = "2y",
                max_age_hours: float = 18.0,
                chunk: int = 50) -> tuple[dict[str, pd.DataFrame], dict]:
    """Download daily OHLCV for ``tickers``. Cached. Never raises.

    Returns (data, status) where data maps ticker -> DataFrame and status is a
    dict {ok, n, source, error}.
    """
    key = f"prices_{period}"
    cached = store_cas.cache_get(key, max_age_hours)
    if cached:
        data = {t: pd.read_json(io.StringIO(j), orient="split") for t, j in cached.items()
                if t in tickers}
        if all(t in data for t in tickers):
            return data, {"ok": True, "n": len(data), "source": "yfinance(cache)"}

    # start from whatever the cache already had (so a big master pull can reuse
    # the core-universe download from earlier in the same run)
    data: dict[str, pd.DataFrame] = {}
    if cached:
        for t, j in cached.items():
            if t in tickers:
                try:
                    data[t] = pd.read_json(io.StringIO(j), orient="split")
                except Exception:
                    pass
    todo = [t for t in tickers if t not in data]

    err = ""
    if todo:
        try:
            import yfinance as yf
        except Exception as e:
            return data, {"ok": bool(data), "n": len(data), "source": "yfinance",
                          "error": str(e)[:200]}
        # chunk the download so a ~400-ticker master pull doesn't rate-limit
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]
            try:
                raw = yf.download(batch, period=period, interval="1d",
                                  auto_adjust=True, group_by="ticker", threads=True,
                                  progress=False)
            except Exception as e:
                err = str(e)[:200]
                continue
            for t in batch:
                try:
                    df = raw[t] if len(batch) > 1 else raw
                    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                    df = df.dropna(how="all")
                    if len(df) >= 60:             # need enough history to be useful
                        data[t] = df
                except Exception:
                    continue

    if data:
        store_cas.cache_put(key, {t: df.to_json(orient="split") for t, df in data.items()})
    return data, {"ok": bool(data), "n": len(data), "source": "yfinance",
                  "error": err if not data else ""}

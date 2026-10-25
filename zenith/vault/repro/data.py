"""Thin fetch layer for the reproductions. Raw downloads are cached as CSV/JSON in
data/vault/cache/repro/ (gitignored: third-party data). Network libraries are
imported lazily so the compute modules and tests stay offline."""

from __future__ import annotations

import io
import json
import time
from datetime import datetime

import pandas as pd

from ...config import VAULT_CACHE_DIR

CACHE = VAULT_CACHE_DIR / "repro"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
MAX_AGE_H = 20.0


def _fresh(path, max_age_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 < max_age_h


def _series_csv(path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].dropna().sort_index()


def cboe_index(name: str, refresh: bool = False) -> pd.Series:
    """Daily close of a Cboe strategy benchmark index from its public history CSV."""
    path = CACHE / f"cboe_{name}.csv"
    if refresh or not _fresh(path, MAX_AGE_H):
        from ... import fetch
        r = fetch.get(CBOE_URL.format(name=name))
        if r is None:
            if path.exists():
                return _series_csv(path)
            raise RuntimeError(f"Cboe {name}: download failed")
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.set_index("date").to_csv(path)
        time.sleep(0.5)
    return _series_csv(path)


def yahoo_close(ticker: str, refresh: bool = False) -> pd.Series:
    """Full daily close history (unadjusted for indices; ^VIX/^IRX are levels)."""
    safe = ticker.replace("^", "_")
    path = CACHE / f"yahoo_{safe}.csv"
    if refresh or not _fresh(path, MAX_AGE_H):
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if h is None or h.empty:
            if path.exists():
                return _series_csv(path)
            raise RuntimeError(f"Yahoo {ticker}: no data")
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        CACHE.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_frame().to_csv(path)
        time.sleep(0.5)
    return _series_csv(path)


def spx_vix_rf(refresh: bool = False) -> dict[str, pd.Series]:
    """^GSPC close, ^VIX (decimal), ^IRX (decimal annual yield), ^SP500TR."""
    return {"spx": yahoo_close("^GSPC", refresh), "vix": yahoo_close("^VIX", refresh) / 100.0,
            "rf": yahoo_close("^IRX", refresh) / 100.0, "sptr": yahoo_close("^SP500TR", refresh)}


# --- earnings event study -------------------------------------------------------

def earnings_dates(ticker: str, refresh: bool = False, max_age_h: float = 24 * 14) -> list[str]:
    """Report dates (ISO, past and scheduled) from yfinance, ~100 quarters (back to ~2002)."""
    path = CACHE / "earn" / f"{ticker}.json"
    if not refresh and _fresh(path, max_age_h):
        return json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).get_earnings_dates(limit=100)
        if df is not None and not df.empty:
            out = sorted({ts.date().isoformat() for ts in df.index})
    except Exception:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    time.sleep(0.4)
    return out


def daily_panel(tickers: list[str], refresh: bool = False, chunk: int = 50,
                start: str = "2001-06-01") -> dict[str, pd.DataFrame]:
    """{ticker: DataFrame[close, volume]} of split/dividend-adjusted daily bars."""
    out: dict[str, pd.DataFrame] = {}
    todo = []
    for t in tickers:
        path = CACHE / "px" / f"{t}.csv"
        if not refresh and _fresh(path, 24 * 7):
            out[t] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            todo.append(t)
    import yfinance as yf
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        df = yf.download(part, start=start, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
        for t in part:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                f = sub[["Close", "Volume"]].dropna()
                f.columns = ["close", "volume"]
                f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
            except Exception:
                continue
            if f.empty:
                continue
            p = CACHE / "px" / f"{t}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            f.to_csv(p)
            out[t] = f
        time.sleep(1.0)
    return out


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

"""Point-in-time, price/returns-derived factors for ETF selection.

Every factor takes the wide close-price ``panel`` and an ``asof`` timestamp and
returns a cross-sectional ``ticker -> score`` Series using ONLY data up to
``asof`` (no look-ahead). Higher score = more of the factor; ranking picks the
top/bottom N. These are genuinely valid for historical backtests (unlike the
fundamental snapshot factors in ``fundamentals``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS

# Names the UI/spec recognise as price-derived (class A — backtest-valid).
PRICE_FACTORS = ("momentum", "trailing_return", "low_volatility",
                 "volatility", "beta", "low_beta", "max_drawdown",
                 "dividend_growth", "calendar_return")


def _upto(panel: pd.DataFrame, asof) -> pd.DataFrame:
    return panel.loc[:asof] if asof is not None else panel


def trailing_return(panel: pd.DataFrame, asof=None, lookback: int = 252,
                    skip: int = 0) -> pd.Series:
    """Return over the trailing ``lookback`` bars ending ``skip`` bars before asof
    (skip>0 implements the classic 12-1 momentum exclusion of the last month)."""
    sub = _upto(panel, asof)
    if len(sub) < 2:
        return pd.Series(np.nan, index=panel.columns)
    end = len(sub) - 1 - max(0, skip)
    start = max(0, end - lookback)
    if end <= start:
        return pd.Series(np.nan, index=panel.columns)
    return sub.iloc[end] / sub.iloc[start] - 1.0


def momentum(panel, asof=None, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 style momentum (skip the most recent month by default)."""
    return trailing_return(panel, asof, lookback, skip)


def calendar_return(panel: pd.DataFrame, year: int) -> pd.Series:
    """Total return during a specific calendar ``year`` (for buy-and-hold-next-year)."""
    yr = panel[panel.index.year == year]
    if len(yr) < 2:
        return pd.Series(np.nan, index=panel.columns)
    return yr.iloc[-1] / yr.iloc[0] - 1.0


def volatility(panel: pd.DataFrame, asof=None, lookback: int = 126) -> pd.Series:
    """Annualised volatility of daily returns over the trailing window."""
    sub = _upto(panel, asof).iloc[-(lookback + 1):]
    if len(sub) < 3:
        return pd.Series(np.nan, index=panel.columns)
    return sub.pct_change().std(ddof=1) * np.sqrt(TRADING_DAYS)


def low_volatility(panel, asof=None, lookback: int = 126) -> pd.Series:
    """Score that ranks low-vol names highest (negative vol)."""
    return -volatility(panel, asof, lookback)


def beta(panel: pd.DataFrame, asof=None, market_close: pd.Series | None = None,
         lookback: int = 252) -> pd.Series:
    """Beta of each name vs the market over the trailing window (pairwise-complete, vectorized)."""
    sub = _upto(panel, asof).iloc[-(lookback + 1):]
    if market_close is None or len(sub) < 5:
        return pd.Series(np.nan, index=panel.columns)
    mkt = market_close.reindex(sub.index).pct_change().to_numpy(float)
    A = sub.pct_change().to_numpy(float)
    ok = np.isfinite(A) & np.isfinite(mkt)[:, None]
    n = ok.sum(axis=0).astype(float)
    M = np.where(ok, mkt[:, None], 0.0)
    X = np.where(ok, A, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, mm = X.sum(0) / n, M.sum(0) / n
        cov = (np.where(ok, (X - mx) * (M - mm), 0.0)).sum(0) / (n - 1)
        var = (np.where(ok, (M - mm) ** 2, 0.0)).sum(0) / (n - 1)
        out = np.where((n >= 5) & (var > 1e-18), cov / var, np.nan)
    return pd.Series(out, index=panel.columns)


def low_beta(panel, asof=None, market_close=None, lookback: int = 252) -> pd.Series:
    return -beta(panel, asof, market_close, lookback)


def max_drawdown(panel: pd.DataFrame, asof=None, lookback: int = 252) -> pd.Series:
    """Most negative drawdown over the window (less negative ranks higher)."""
    sub = _upto(panel, asof).iloc[-(lookback + 1):]
    if len(sub) < 3:
        return pd.Series(np.nan, index=panel.columns)
    dd = sub / sub.cummax() - 1.0
    return dd.min()


_DIV_MEMO: dict[str, pd.Series] = {}


def _dividends(tickers) -> dict[str, pd.Series]:
    """Dividend histories, fetched once per process (threaded) and cached on disk for 30 days."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from ..config import DATA_CACHE
    path = DATA_CACHE / "_dividends.parquet"
    if not _DIV_MEMO and path.exists() and time.time() - path.stat().st_mtime < 30 * 86400:
        try:
            df = pd.read_parquet(path)
            for t, g in df.groupby("ticker"):
                _DIV_MEMO[t] = pd.Series(g["amount"].to_numpy(), index=pd.to_datetime(g["date"]))
        except Exception:  # noqa: BLE001
            pass
    need = [t for t in tickers if t not in _DIV_MEMO]
    if need:
        import yfinance as yf

        def one(t):
            try:
                d = yf.Ticker(t).dividends
                d.index = pd.to_datetime(d.index).tz_localize(None)
                return t, d.astype(float)
            except Exception:  # noqa: BLE001
                return t, pd.Series(dtype=float)

        with ThreadPoolExecutor(max_workers=8) as ex:
            for t, d in ex.map(one, need):
                _DIV_MEMO[t] = d
        try:
            rows = [pd.DataFrame({"ticker": t, "date": d.index, "amount": d.to_numpy()})
                    for t, d in _DIV_MEMO.items() if len(d)]
            if rows:
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(rows).to_parquet(path)
        except Exception:  # noqa: BLE001
            pass
    return {t: _DIV_MEMO.get(t, pd.Series(dtype=float)) for t in tickers}


def dividend_growth(tickers, years: int = 5, asof=None) -> pd.Series:
    """Annualised dividend growth over the ``years`` complete calendar years before ``asof``.

    Point-in-time: only dividends paid before ``asof`` count (the old version used the latest
    years at every rebalance, i.e. look-ahead). Histories are fetched once and reused.
    """
    end_year = (pd.Timestamp(asof).year - 1) if asof is not None else None
    out = {}
    for t, div in _dividends(list(tickers)).items():
        if div is None or div.empty:
            out[t] = np.nan
            continue
        ann = div.groupby(div.index.year).sum()
        ey = end_year if end_year is not None else int(ann.index.max())
        ann = ann[(ann.index <= ey) & (ann.index >= ey - years)]
        if len(ann) < 2 or ann.iloc[0] <= 0 or ann.index.max() < ey:
            out[t] = np.nan
            continue
        n = len(ann) - 1
        out[t] = (ann.iloc[-1] / ann.iloc[0]) ** (1.0 / n) - 1.0
    return pd.Series(out)

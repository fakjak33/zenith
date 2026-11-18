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
    """Beta of each name vs the market over the trailing window."""
    sub = _upto(panel, asof).iloc[-(lookback + 1):]
    if market_close is None or len(sub) < 5:
        return pd.Series(np.nan, index=panel.columns)
    mkt = market_close.reindex(sub.index).pct_change()
    var = mkt.var()
    if not np.isfinite(var) or var < 1e-18:
        return pd.Series(np.nan, index=panel.columns)
    aret = sub.pct_change()
    return aret.apply(lambda c: c.cov(mkt) / var)


def low_beta(panel, asof=None, market_close=None, lookback: int = 252) -> pd.Series:
    return -beta(panel, asof, market_close, lookback)


def max_drawdown(panel: pd.DataFrame, asof=None, lookback: int = 252) -> pd.Series:
    """Most negative drawdown over the window (less negative ranks higher)."""
    sub = _upto(panel, asof).iloc[-(lookback + 1):]
    if len(sub) < 3:
        return pd.Series(np.nan, index=panel.columns)
    dd = sub / sub.cummax() - 1.0
    return dd.min()


def dividend_growth(tickers, years: int = 5) -> pd.Series:
    """Annualised dividend growth over ``years`` (uses yfinance dividend history).

    Best-effort and slow (per-ticker); returns NaN where unavailable. Dividend
    *history* is genuinely point-in-time, unlike fundamental snapshots.
    """
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            div = yf.Ticker(t).dividends
            if div is None or div.empty:
                out[t] = np.nan
                continue
            ann = div.groupby(div.index.year).sum()
            ann = ann[ann.index >= ann.index.max() - years]
            if len(ann) < 2 or ann.iloc[0] <= 0:
                out[t] = np.nan
                continue
            n = len(ann) - 1
            out[t] = (ann.iloc[-1] / ann.iloc[0]) ** (1.0 / n) - 1.0
        except Exception:
            out[t] = np.nan
    return pd.Series(out)

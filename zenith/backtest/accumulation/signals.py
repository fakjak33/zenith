"""Indicator signals for accumulation strategies.

All functions take a close-price Series (and params) and return an aligned
Series. Kept dependency-light (pandas/numpy) and reused by the rule classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_from_high(close: pd.Series, rolling: int | None = None) -> pd.Series:
    """Percent below the running peak (0 to -1). ``rolling`` limits the peak
    window (None = all-time high to date)."""
    peak = close.cummax() if rolling is None else close.rolling(rolling, min_periods=1).max()
    return close / peak - 1.0


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def moving_average(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(max(1, int(n)), min_periods=1).mean()


def ma_slope(close: pd.Series, n: int, smooth: int = 5) -> pd.Series:
    """Slope of MA(n) as a fractional change per bar (smoothed)."""
    ma = moving_average(close, n)
    return (ma.pct_change(smooth) / smooth).fillna(0.0)


def pct_from_ma(close: pd.Series, n: int) -> pd.Series:
    """Fractional distance of price above/below MA(n) (Mayer-multiple − 1)."""
    ma = moving_average(close, n)
    return close / ma.replace(0.0, np.nan) - 1.0

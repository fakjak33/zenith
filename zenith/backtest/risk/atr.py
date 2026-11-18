"""Average True Range (Wilder)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(ohlcv: pd.DataFrame) -> pd.Series:
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR via an EWMA with alpha = 1/period."""
    tr = true_range(ohlcv)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()

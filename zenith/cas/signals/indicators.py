"""Indicator toolkit: the time-series / cross-sectional operators used by the
101-formulaic-alpha style library, plus classic technical indicators.

Operators follow the naming convention of the Kakushadze "101 Formulaic Alphas"
paper (rank, ts_rank, delta, decay_linear, correlation, scale, signedpower, …)
so the alpha expressions read the same way. All take/return pandas objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- time-series operators (per-series) -----------------------------------
def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def delta(x: pd.Series, d: int) -> pd.Series:
    return x.diff(d)


def delay(x: pd.Series, d: int) -> pd.Series:
    return x.shift(d)


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).apply(lambda w: pd.Series(w).rank().iloc[-1] / len(w), raw=False)


def ts_min(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).min()


def ts_max(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).max()


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).apply(lambda w: float(np.argmax(w)), raw=True)


def stddev(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).std()


def sma(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).mean()


def ema(x: pd.Series, d: int) -> pd.Series:
    return x.ewm(span=d, adjust=False).mean()


def correlation(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    return a.rolling(d).corr(b)


def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """Weighted moving average with linearly decaying weights (most recent heavy)."""
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d).apply(lambda v: float(np.dot(v, w)), raw=True)


def signedpower(x: pd.Series, a: float) -> pd.Series:
    return np.sign(x) * (np.abs(x) ** a)


def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    s = x.abs().sum()
    return x * (a / s) if s else x


def zscore(x: pd.Series, d: int) -> pd.Series:
    return (x - x.rolling(d).mean()) / (x.rolling(d).std() + 1e-9)


# --- cross-sectional rank (across a DataFrame of assets) -------------------
def rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0,1] across columns (assets), per row (date)."""
    return df.rank(axis=1, pct=True)


# --- classic technical indicators -----------------------------------------
def rsi(close: pd.Series, d: int = 14) -> pd.Series:
    ch = close.diff()
    up = ch.clip(lower=0).rolling(d).mean()
    dn = (-ch.clip(upper=0)).rolling(d).mean()
    rs = up / (dn + 1e-9)
    return 100 - 100 / (1 + rs)


def donchian(high: pd.Series, low: pd.Series, d: int = 20) -> tuple[pd.Series, pd.Series]:
    return high.rolling(d).max(), low.rolling(d).min()


def atr(df: pd.DataFrame, d: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(d).mean()


def realized_vol(close: pd.Series, d: int = 20) -> pd.Series:
    return close.pct_change().rolling(d).std() * np.sqrt(252)


def clip1(x: float) -> float:
    """Clamp to [-1, 1]; NaN -> 0."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(max(-1.0, min(1.0, x)))


def last(x: pd.Series, default: float = 0.0) -> float:
    x = x.dropna()
    return float(x.iloc[-1]) if len(x) else default

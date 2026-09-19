"""Seven-speed EWMAC — pure, vectorized signal math. No I/O, no network.

For one instrument's DAILY adjusted close series p_t, and each speed
(fast, slow) in config.TREND_SPEEDS:

    EMA_N(t)  = p.ewm(span=N, adjust=True).mean()          exponential, NOT simple
    sigma(t)  = EWM std of daily price DIFFERENCES p_t - p_{t-1}   (span 35),
                floored at the 5th percentile of its own trailing 500 days
    raw(t)    = (EMA_fast(t) - EMA_slow(t)) / sigma(t)       unitless trend strength
    f(t)      = clip(raw(t) x scalar_speed, -20, +20)        per-speed forecast
    score(t)  = (f_1 + ... + f_7) / 7                        equal weight, no FDM

Why each choice (the methodology you would otherwise have to reverse-engineer):

  * EXPONENTIAL averages with `adjust=True`: the bias-corrected EWM, which
    weights the first observations correctly instead of seeding the average at
    the first price. At a 512-day span that start-up bias matters; the repo's
    other `ema()` (cas/signals/indicators.py) uses adjust=False, fine for a
    20-day MACD but not here -- so it is deliberately not reused.
  * Dividing by PRICE-DIFFERENCE volatility makes `raw` a pure number: "how
    many daily standard deviations apart are the two averages". It is identical
    under any proportional rescaling of the price series (a split, or the
    dividend back-adjustment yfinance applies), and comparable across a $20 ETF
    and a $900 stock. The vol FLOOR stops a dead-quiet stretch from dividing by
    almost nothing and producing a spurious +/-20 (Carver's robust_vol_calc).
  * The FORECAST SCALAR per speed puts all seven speeds on one scale: without
    it the slow speeds (whose averages sit further apart) would mechanically
    dominate. Fixed published values -- see config.TREND_FORECAST_SCALARS.
  * The +/-20 cap bounds each speed's vote, so the plain mean of seven capped
    forecasts is itself bounded to [-20, +20] -- no second normalization step.

A speed is None (NaN) until the series has at least `slow` bars: an EMA that
has seen fewer observations than its own span is mostly start-up, not signal.

The per-speed normalization is pluggable (NORMALIZERS / config.TREND_NORMALIZATION)
so a future methodology change is one dict entry, not a rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (TREND_FORECAST_CAP, TREND_FORECAST_SCALARS, TREND_MIN_SPEEDS,
                      TREND_NORMALIZATION, TREND_SPEEDS, TREND_VOL_FLOOR_MIN_PERIODS,
                      TREND_VOL_FLOOR_QUANTILE, TREND_VOL_FLOOR_WINDOW, TREND_VOL_MIN_PERIODS,
                      TREND_VOL_SPAN)

SPEED_KEYS = tuple(f"{f}_{s}" for f, s in TREND_SPEEDS)

# A median gap between bars above this many calendar days means the input is
# not a daily series (weekly bars have a 7-day median gap). The whole system is
# specified on daily data; silently accepting weekly bars would stretch every
# span 5x.
_MAX_MEDIAN_GAP_DAYS = 4


def assert_daily(index: pd.Index) -> None:
    """Raise ValueError unless `index` looks like daily bars."""
    if len(index) < 3:
        return
    idx = pd.DatetimeIndex(index)
    gaps = np.diff(idx.asi8) / 86_400e9
    med = float(np.median(gaps))
    if med > _MAX_MEDIAN_GAP_DAYS:
        raise ValueError(f"EWMAC requires DAILY bars; median gap is {med:.1f} days")


def ema(close: pd.Series, span: int) -> pd.Series:
    """Bias-corrected exponential moving average, NaN until `span` bars exist."""
    return close.ewm(span=span, adjust=True, min_periods=span).mean()


def price_vol(close: pd.Series, span: int = TREND_VOL_SPAN,
              floor_window: int = TREND_VOL_FLOOR_WINDOW,
              floor_q: float = TREND_VOL_FLOOR_QUANTILE) -> pd.Series:
    """EWM std of daily price differences (price units), floored at the
    `floor_q` quantile of its own trailing `floor_window` values."""
    vol = close.diff().ewm(span=span, adjust=True, min_periods=TREND_VOL_MIN_PERIODS).std()
    floor = vol.rolling(floor_window, min_periods=TREND_VOL_FLOOR_MIN_PERIODS).quantile(floor_q)
    return vol.where(floor.isna() | (vol >= floor), floor)


def raw_ewmac(close: pd.Series, fast: int, slow: int, vol: pd.Series | None = None) -> pd.Series:
    """(EMA_fast - EMA_slow) / sigma. NaN until `slow` bars exist.

    A series with ZERO price variation (sigma == 0) has no measurable trend, so
    raw is 0 there rather than 0/0 -- the flat-series guard."""
    vol = price_vol(close) if vol is None else vol
    diff = ema(close, fast) - ema(close, slow)
    eps = 1e-12 * close.abs().clip(lower=1.0)
    raw = diff / vol.where(vol > eps)
    raw = raw.mask(diff.notna() & ~(vol > eps), 0.0)
    return raw.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------ normalizers --
# Each maps (raw series, forecast scalar, cap) -> forecast series in [-cap, cap].
def _norm_carver(raw: pd.Series, scalar: float, cap: float) -> pd.Series:
    """Confirmed default: scale, then hard-cap."""
    return (raw * scalar).clip(-cap, cap)


def _norm_sign(raw: pd.Series, scalar: float, cap: float) -> pd.Series:
    """Binary direction only: +/-cap (0 exactly at a tie). Discards strength."""
    return np.sign(raw) * cap


def _norm_tanh(raw: pd.Series, scalar: float, cap: float) -> pd.Series:
    """Smooth squash: ~linear for small forecasts, saturating toward +/-cap."""
    return cap * np.tanh(raw * scalar / cap)


NORMALIZERS = {"carver": _norm_carver, "sign": _norm_sign, "tanh": _norm_tanh}


@dataclass
class EwmacResult:
    """Full daily history for one instrument.

    forecast  -- DataFrame[SPEED_KEYS], each in [-20, +20], NaN where a speed
                 is not yet valid.
    raw       -- DataFrame[SPEED_KEYS], unnormalized (EMA_f - EMA_s)/sigma.
                 Its SIGN is the crossover state (fast above/below slow).
    score     -- equal-weight mean of the valid speeds; NaN where fewer than
                 TREND_MIN_SPEEDS are valid.
    n_valid   -- how many of the seven speeds are valid each day.
    """
    forecast: pd.DataFrame
    raw: pd.DataFrame
    score: pd.Series
    n_valid: pd.Series


def forecasts(close: pd.Series, normalization: str = TREND_NORMALIZATION,
              speeds=TREND_SPEEDS, scalars: dict = TREND_FORECAST_SCALARS,
              cap: float = TREND_FORECAST_CAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecast, raw) DataFrames, one column per speed."""
    norm = NORMALIZERS[normalization]
    close = close.astype(float)
    vol = price_vol(close)
    raw_cols, f_cols = {}, {}
    for (fast, slow), key in zip(speeds, SPEED_KEYS):
        r = raw_ewmac(close, fast, slow, vol)
        raw_cols[key] = r
        f_cols[key] = norm(r, scalars[(fast, slow)], cap)
    return pd.DataFrame(f_cols, index=close.index), pd.DataFrame(raw_cols, index=close.index)


def combine(forecast: pd.DataFrame, min_speeds: int = TREND_MIN_SPEEDS) -> tuple[pd.Series, pd.Series]:
    """Equal-weight Trend Score: the arithmetic mean of the valid speeds'
    forecasts. With all seven valid this is exactly (f1+...+f7)/7, so each
    speed moves the score by exactly (its forecast change)/7."""
    n_valid = forecast.notna().sum(axis=1)
    score = forecast.mean(axis=1, skipna=True).where(n_valid >= min_speeds)
    return score.clip(-TREND_FORECAST_CAP, TREND_FORECAST_CAP), n_valid


def build(close: pd.Series, normalization: str = TREND_NORMALIZATION) -> EwmacResult:
    """Everything for one instrument's close series. Raises ValueError on
    non-daily input."""
    close = close.dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    assert_daily(close.index)
    f, raw = forecasts(close, normalization=normalization)
    score, n_valid = combine(f)
    return EwmacResult(forecast=f, raw=raw, score=score, n_valid=n_valid)


def direction(raw: pd.DataFrame) -> pd.DataFrame:
    """+1 fast above slow, -1 below, 0 exactly equal, NaN where not valid."""
    return np.sign(raw)


def finite_or_none(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None

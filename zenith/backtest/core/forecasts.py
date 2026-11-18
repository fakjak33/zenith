"""Forecast scaling and capping — the heart of the Carver framework.

A 'raw' forecast from a trading rule has arbitrary units. We:
  1. normalize it by its own rolling volatility (so its scale is stable),
  2. multiply by a forecast scalar so the long-run average |forecast| ~= 10,
  3. cap at +/- 20 to limit the influence of any single extreme reading.

GUARD RAIL (fixed vs PARALLAX): the scalar is estimated on an *expanding*
window that only sees bars before t. PARALLAX used the full-sample mean, which
let the future set today's position size. Bars before ``SCALAR_WARMUP`` carry a
zero forecast — no trading until the scalar has data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORECAST_SCALAR_TARGET, FORECAST_CAP, SCALAR_WARMUP


def scale_forecast(
    raw: pd.Series,
    scalar: float | None = None,
    cap: float = FORECAST_CAP,
    causal: bool = True,
    warmup: int = SCALAR_WARMUP,
) -> pd.Series:
    """Scale a raw forecast to target-average magnitude and cap it.

    - ``scalar`` given: use it as-is (e.g. fitted in-sample).
    - ``causal=True`` (default): expanding scalar from bars < t only.
    - ``causal=False``: legacy full-sample scalar (look-ahead; kept only so the
      parity test can isolate the size of the fix).
    """
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scalar is not None:
        return (raw * scalar).clip(-cap, cap)
    if not causal:
        return (raw * estimate_forecast_scalar(raw)).clip(-cap, cap)
    scal = expanding_forecast_scalar(raw, warmup=warmup)
    return (raw * scal).fillna(0.0).clip(-cap, cap)


def expanding_forecast_scalar(raw: pd.Series, target: float = FORECAST_SCALAR_TARGET,
                              warmup: int = SCALAR_WARMUP) -> pd.Series:
    """Per-bar scalar = target / mean(|raw|) over bars strictly before t."""
    mean_abs = raw.abs().expanding(min_periods=warmup).mean().shift(1)
    scal = target / mean_abs.where(mean_abs > 1e-12)
    return scal


def estimate_forecast_scalar(raw: pd.Series, target: float = FORECAST_SCALAR_TARGET) -> float:
    """Scalar that maps mean(|raw|) to ``target`` over the whole series."""
    mean_abs = raw.abs().mean()
    if not np.isfinite(mean_abs) or mean_abs < 1e-12:
        return 1.0
    return float(target / mean_abs)


def normalize_by_vol(signal: pd.Series, vol: pd.Series) -> pd.Series:
    """Divide a raw signal (e.g. a price-difference) by a volatility series."""
    out = signal / vol.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

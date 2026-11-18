import numpy as np
import pandas as pd

from zenith.backtest.core.forecasts import scale_forecast, estimate_forecast_scalar
from zenith.backtest.config import FORECAST_CAP, FORECAST_SCALAR_TARGET


def test_forecast_is_capped():
    raw = pd.Series(np.linspace(-100, 100, 500))
    scaled = scale_forecast(raw)
    assert scaled.abs().max() <= FORECAST_CAP + 1e-9


def test_forecast_scalar_targets_ten():
    rng = np.random.default_rng(0)
    raw = pd.Series(rng.normal(0, 3, 5000))
    scalar = estimate_forecast_scalar(raw)
    scaled = raw * scalar
    # mean absolute scaled forecast should be ~ target (pre-cap)
    assert abs(scaled.abs().mean() - FORECAST_SCALAR_TARGET) < 0.5


def test_handles_nan_and_inf():
    raw = pd.Series([1.0, np.nan, np.inf, -np.inf, 2.0])
    scaled = scale_forecast(raw)
    assert np.isfinite(scaled).all()

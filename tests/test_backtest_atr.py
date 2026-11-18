import numpy as np
import pandas as pd

from zenith.backtest.risk.atr import atr, true_range
from zenith.backtest.risk.overlays import apply_atr_overlay
from zenith.backtest.config import RiskConfig


def _toy_ohlcv(n=100):
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1_000,
    }, index=idx)


def test_atr_positive_and_finite():
    df = _toy_ohlcv()
    a = atr(df, period=14).dropna()
    assert (a > 0).all()
    assert np.isfinite(a).all()


def test_true_range_at_least_high_low():
    df = _toy_ohlcv()
    tr = true_range(df)
    assert (tr >= (df["high"] - df["low"]) - 1e-9).all()


def test_overlay_noop_when_disabled():
    df = _toy_ohlcv()
    pos = pd.Series(1.0, index=df.index)
    gated, events = apply_atr_overlay(pos, df, RiskConfig())
    assert gated.equals(pos)
    assert events.empty


def test_stop_triggers_on_drop():
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    # rise then sharp fall to trip a long stop
    close = pd.Series(np.r_[np.linspace(100, 110, 30), np.linspace(110, 80, 30)], index=idx)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": 1_000}, index=idx)
    pos = pd.Series(1.0, index=idx)
    cfg = RiskConfig(use_atr_stop=True, atr_stop_mult=2.0, atr_period=10, trailing_stop=True)
    gated, events = apply_atr_overlay(pos, df, cfg)
    assert (events["type"] == "stop").any()
    assert (gated == 0.0).any()

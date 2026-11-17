"""Tests for the interval-return distribution / skew helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenith.backtest.engine import returns_dist as rd


def _series(vals, freq="B"):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq=freq)
    return pd.Series(vals, index=idx)


def test_skew_summary_detects_positive_and_negative():
    rng = np.random.default_rng(0)
    pos = _series(np.r_[rng.normal(0, 0.01, 400), [0.4, 0.5, 0.6]])   # right tail
    neg = _series(np.r_[rng.normal(0, 0.01, 400), [-0.4, -0.5, -0.6]])
    assert rd.skew_summary(pos)["label"] == "positive"
    assert rd.skew_summary(neg)["label"] == "negative"


def test_skew_summary_symmetric_is_none():
    rng = np.random.default_rng(1)
    sym = _series(rng.normal(0, 0.01, 2000))
    assert rd.skew_summary(sym)["label"] == "~none"


def test_interval_returns_aggregates_to_monthly():
    daily = _series(np.full(120, 0.001))      # ~6 months of business days
    agg, eff = rd.interval_returns(daily, "Daily", "Monthly")
    assert eff == "Monthly"
    assert len(agg) < len(daily)
    # compounding 0.001 over ~21 business days ≈ 2.1%
    assert 0.015 < agg.iloc[1] < 0.03


def test_interval_finer_than_native_falls_back():
    weekly = _series(np.full(40, 0.002), freq="W-FRI")
    agg, eff = rd.interval_returns(weekly, "Weekly", "Daily")
    assert "native" in eff.lower()
    assert len(agg) == len(weekly.dropna())


def test_candle_is_native():
    s = _series(np.full(30, 0.01))
    agg, eff = rd.interval_returns(s, "Daily", "Candle (native)")
    assert eff == "Candle (native)" and len(agg) == 30

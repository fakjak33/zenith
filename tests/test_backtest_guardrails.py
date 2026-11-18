"""BACKTEST guard rails: no look-ahead, frequency-aware annualization, parity."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.core.forecasts import scale_forecast
from zenith.backtest.core.volatility import bars_per_year
from zenith.backtest.data.providers import resample_ohlcv
from zenith.backtest.data.synthetic import generate
from zenith.backtest.engine import metrics
from zenith.backtest.engine.engine import run_strategy
from zenith.backtest.rules import REGISTRY

SINGLE_ASSET_RULES = [k for k, cls in REGISTRY.items() if not cls.cross_sectional]


@pytest.mark.parametrize("key", SINGLE_ASSET_RULES)
def test_future_shock_cannot_change_the_past(key):
    """Inject a huge move at bar T; every position before T must be unchanged."""
    df = generate("trending", n_days=900, seed=3)
    t = 700
    shocked = df.copy()
    shocked.iloc[t:, :4] = shocked.iloc[t:, :4] * 3.0      # tripling from bar T on
    a = run_strategy(REGISTRY[key](), df)
    b = run_strategy(REGISTRY[key](), shocked)
    pd.testing.assert_series_equal(a.position.iloc[:t], b.position.iloc[:t], check_names=False)
    pd.testing.assert_series_equal(a.returns.iloc[:t], b.returns.iloc[:t], check_names=False)


def test_causal_scalar_differs_from_full_sample_scalar():
    """The PARALLAX full-sample scalar leaks the future; the causal one must not."""
    raw = pd.Series(np.r_[np.ones(200), np.full(200, 10.0)])
    causal = scale_forecast(raw)
    leaky = scale_forecast(raw, causal=False)
    # early bars: leaky version is shrunk by the big future values, causal is not
    assert causal.iloc[100] == pytest.approx(10.0)
    assert leaky.iloc[100] < 5.0
    # warm-up bars carry no forecast
    assert (causal.iloc[:64] == 0).all()


@pytest.mark.parametrize("rule,expected", [("D", 252), ("W-FRI", 52), ("ME", 12)])
def test_bars_per_year_inferred(rule, expected):
    df = resample_ohlcv(generate("gbm", n_days=2000, seed=1), rule)
    assert bars_per_year(df.index) == expected


def test_weekly_sharpe_annualized_with_52():
    df = resample_ohlcv(generate("trending", n_days=3000, seed=4), "W-FRI")
    res = run_strategy(REGISTRY["tsmom"](lookback=12), df)
    assert res.periods == 52
    assert res.stats["Sharpe"] == pytest.approx(metrics.sharpe(res.returns, 52))


def test_costs_reduce_returns():
    from zenith.backtest.config import BacktestConfig
    df = generate("gbm", n_days=800, seed=2)
    free = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(cost_bps=0, slippage_bps=0))
    paid = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(cost_bps=10, slippage_bps=5))
    assert paid.returns.sum() < free.returns.sum()
    assert paid.costs.sum() > 0


def test_extended_stats_present():
    res = run_strategy(REGISTRY["ema"](), generate("trending", n_days=800, seed=8))
    for k in ("Vol", "Skew", "TailRatio", "Ulcer", "ProfitFactor", "MaxDD_bars"):
        assert k in res.stats and np.isfinite(res.stats[k])


def test_dsr_with_one_trial_equals_psr():
    r = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, 2000))
    assert metrics.deflated_sharpe_ratio(r, n_trials=1) == pytest.approx(metrics.probabilistic_sharpe_ratio(r))
    assert metrics.deflated_sharpe_ratio(r, n_trials=50) < metrics.probabilistic_sharpe_ratio(r)

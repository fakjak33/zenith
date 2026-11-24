"""Indicator library: causality, hover coverage, and every indicator backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.backtest import glossary as G
from zenith.backtest.data.synthetic import generate
from zenith.backtest.indicators import CATALOG, FAMILIES
from zenith.backtest.rules import REGISTRY
from zenith.backtest.rules.indicator_rule import band_state
from zenith.backtest.spec import StrategySpec, run_spec


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    df = generate("trending", n_days=1600, seed=11)
    rng = np.random.default_rng(0)
    df["ref_close"] = generate("gbm", n_days=1600, seed=12)["close"].to_numpy()
    df["vix"] = 18 + rng.normal(0, 4, len(df)).cumsum() * 0.05
    df["vix3m"] = df["vix"] + 1.0
    return df


def test_catalog_size_and_families():
    assert len(CATALOG) >= 70
    assert {i.family for i in CATALOG.values()} == set(FAMILIES)


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_indicator_is_causal(frame, key):
    """Value at bar t must not change when bars after t are removed (no look-ahead)."""
    ind = CATALOG[key]
    full = ind(frame).to_numpy()
    cut = 1100
    part = ind(frame.iloc[:cut]).to_numpy()
    a, b = full[:cut], part
    assert (np.isfinite(a) == np.isfinite(b)).all(), f"{key}: warm-up differs"
    m = np.isfinite(a)
    assert m.sum() > 100, f"{key}: almost no values"
    assert np.allclose(a[m], b[m], rtol=1e-6, atol=1e-9), f"{key}: future data leaks into the past"


def test_future_shock_does_not_move_past(frame):
    """A huge price shock at bar k cannot change any indicator before k."""
    k = 1200
    shocked = frame.copy()
    shocked.iloc[k:, :4] *= 1.5
    for key, ind in CATALOG.items():
        a, b = ind(frame).iloc[:k], ind(shocked).iloc[:k]
        assert np.allclose(a.fillna(0), b.fillna(0), rtol=1e-9, atol=1e-12), key


def test_every_indicator_has_help_and_evidence():
    for key, ind in CATALOG.items():
        assert len(ind.hover) > 30, key
        assert ind.how and ind.use and ind.evidence, key
        for pn, pt in ind.params.items():
            assert len(pt) == 5 and pt[4], f"{key}.{pn} has no hover help"


def test_every_rule_param_has_help():
    for key, cls in REGISTRY.items():
        for pn in cls.params:
            assert G.param_help(cls, pn), f"{key}.{pn} has no hover help"


def test_metric_help_covers_summary():
    from zenith.backtest.engine import metrics
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 600))
    for k in metrics.summary(r):
        assert G.metric_help(k), k


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_every_indicator_backtests(frame, key):
    spec = StrategySpec(src="Synthetic", name="trending", rule=f"i_{key}", params={})
    aux = {c: frame[c] for c in ("ref_close", "vix", "vix3m")}
    res = run_spec(spec, frame[["open", "high", "low", "close", "volume"]], aux)
    assert np.isfinite(res.stats["Sharpe"]) or res.forecast.abs().sum() == 0
    assert res.forecast.abs().max() <= 20 + 1e-9


def test_band_state_holds_until_midpoint():
    v = pd.Series([50, 25, 28, 45, 55, 75, 72, 49, 50])
    s = band_state(v, 30, 70).tolist()
    assert s == [0, 1, 1, 1, 0, -1, -1, 0, 0]


def test_trend_indicators_win_on_trending_and_reversion_on_mean_reverting():
    """Known-answer check: SMA cross should beat a z-score fade on trending data, and vice versa."""
    tr = generate("trending", n_days=3000, seed=5)
    mr = generate("mean_reverting", n_days=3000, seed=5)
    sharpe = lambda df, rule: run_spec(StrategySpec(src="Synthetic", rule=rule, params={}), df).stats["Sharpe"]  # noqa: E731
    assert sharpe(tr, "i_sma") > sharpe(tr, "i_zscore")
    assert sharpe(mr, "i_zscore") > sharpe(mr, "i_sma")


def test_ref_ticker_only_in_hash_when_used():
    a = StrategySpec(rule="i_rsi", ref_ticker="QQQ")
    b = StrategySpec(rule="i_rsi", ref_ticker="SPY")
    assert a.hash() == b.hash()
    c = StrategySpec(rule="i_rel_strength", ref_ticker="QQQ")
    d = StrategySpec(rule="i_rel_strength", ref_ticker="SPY")
    assert c.hash() != d.hash()
    # old specs keep their hashes
    assert StrategySpec(rule="ema").hash() == StrategySpec.from_dict(
        {k: v for k, v in StrategySpec(rule="ema").to_dict().items() if k != "ref_ticker"}).hash()

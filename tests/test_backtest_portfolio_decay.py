"""Phase 5: portfolio optimisers, Sharpe-difference CI, decay monitor, and the pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from zenith.backtest.monitor.decay import cone, cusum, dd_breaker, month_ranks, rolling_alpha, verdict
from zenith.backtest.portfolio_opt import (METHODS, bucket_of, combine_with, erc_weights, hrp_weights,
                                           max_sharpe_weights, min_var_weights, optimised_weights, sharpe_diff_ci,
                                           sharpe_vs_n)
from zenith.backtest.spec import StrategySpec


@pytest.fixture(scope="module")
def rets() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=1500)
    common = rng.normal(0, 0.006, 1500)
    return pd.DataFrame({f"s{i}": common * (i % 2) + rng.normal(0.0003, 0.004 * (1 + i), 1500) for i in range(5)},
                        index=idx)


def test_erc_equalises_risk_contributions(rets):
    S = rets.cov().to_numpy()
    w = erc_weights(S)
    rc = w * (S @ w)
    assert np.allclose(rc / rc.sum(), 1 / len(w), atol=1e-4) and abs(w.sum() - 1) < 1e-9


def test_min_var_is_lowest_variance(rets):
    S = rets.cov().to_numpy()
    w = min_var_weights(S)
    eq = np.full(len(S), 1 / len(S))
    assert w @ S @ w <= eq @ S @ eq + 1e-12 and (w >= -1e-9).all()


def test_hrp_and_max_sharpe_are_valid_weights(rets):
    S = rets.cov().to_numpy()
    for w in (hrp_weights(S), max_sharpe_weights(S, rets.mean().to_numpy())):
        assert abs(w.sum() - 1) < 1e-9 and (w >= -1e-9).all()


@pytest.mark.parametrize("method", list(METHODS))
def test_weights_are_causal(rets, method):
    a = optimised_weights(rets, method, 252, buckets={"s0": "trend", "s1": "reversion"})
    b = optimised_weights(rets.iloc[:1100], method, 252, buckets={"s0": "trend", "s1": "reversion"})
    assert np.allclose(a.iloc[:1100].to_numpy(), b.to_numpy(), equal_nan=True)
    out = combine_with(rets, method, 252, buckets={})
    assert out["returns"].notna().all()


def test_sharpe_diff_ci_brackets_zero_for_identical(rets):
    d, lo, hi = sharpe_diff_ci(rets["s0"], rets["s0"])
    assert d == 0 and lo <= 0 <= hi


def test_sharpe_vs_n_grows_for_uncorrelated(rets):
    sv = sharpe_vs_n(rets[["s0", "s2", "s4"]])
    assert sv["theory"].iloc[-1] >= sv["theory"].iloc[0]


def test_bucket_of():
    assert bucket_of(StrategySpec(rule="ema")) == "trend"
    assert bucket_of(StrategySpec(rule="i_rsi")) == "reversion"
    assert bucket_of(StrategySpec(mode="ml")) == "ml"


def test_decay_detects_a_broken_strategy():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-04", periods=1500)
    r = pd.Series(np.r_[rng.normal(0.0008, 0.01, 1000), rng.normal(-0.0012, 0.01, 500)], index=idx)
    ref, live = r.iloc[:1000], r.iloc[1000:]
    cs = cusum(ref, live)
    cn = cone(ref, live, n_paths=400)
    v = verdict(cs, cn, dd_breaker(ref, len(live), n_paths=400), live, month_ranks(ref, live))
    assert cs["alarm"] is not None and v["state"] != "WORKING"


def test_decay_quiet_for_a_working_strategy():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-04", periods=1500)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500), index=idx)
    ref, live = r.iloc[:1000], r.iloc[1000:]
    cs = cusum(ref, live)
    assert cs["alarm"] is None
    v = verdict(cs, cone(ref, live, n_paths=400), dd_breaker(ref, len(live), n_paths=400), live,
                month_ranks(ref, live))
    assert v["state"] in ("WORKING", "WATCH")


def test_rolling_alpha_shape():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-04", periods=800)
    b = pd.Series(rng.normal(0, 0.01, 800), index=idx)
    r = 0.5 * b + 0.0004 + rng.normal(0, 0.005, 800)
    ra = rolling_alpha(r, b, window=252)
    assert {"alpha", "lo", "hi", "beta"} <= set(ra.columns) and ra["beta"].between(0.3, 0.7).all()


def _store_app(src: str, specs: dict) -> AppTest:
    from zenith.backtest import library
    at = AppTest.from_string(src, default_timeout=900)
    store = {}
    for nm, sp in specs.items():
        library.save(store, nm, sp)
    at.session_state["bt_saved"] = store
    return at


SPECS = {"trend": StrategySpec(src="Synthetic", name="trending", rule="ema"),
         "rev": StrategySpec(src="Synthetic", name="mean_reverting", rule="i_zscore"),
         "rsi": StrategySpec(src="Synthetic", name="regime", rule="i_rsi2")}


def test_portfolio_page_methods_and_compare():
    src = "import streamlit as st\nfrom zenith.backtest.views import portfolio_view\nportfolio_view.render()\n"
    at = _store_app(src, SPECS)
    at.session_state["pf_w"] = "hrp"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="pf_mrun").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_library_decay_monitor_renders():
    src = "import streamlit as st\nfrom zenith.backtest.views import library_view\nlibrary_view.render()\n"
    at = _store_app(src, SPECS)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

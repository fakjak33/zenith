"""Phase 3 validation: PBO, CPCV, Reality Check/SPA, block length, challenge sim, stress, and the panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from zenith.backtest.data.synthetic import generate
from zenith.backtest.engine.montecarlo import (auto_block_length, challenge_sim, trade_returns,
                                               trade_shuffle_drawdowns)
from zenith.backtest.robust.overfit import cpcv, n_cpcv_paths, pbo, reality_check, top_clusters
from zenith.backtest.robust.stress import SCENARIOS, scenario_table, shocked
from zenith.backtest.spec import StrategySpec, run_spec


@pytest.fixture(scope="module")
def noise() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, (2000, 20)), index=pd.bdate_range("2005-01-03", periods=2000))


def test_pbo_is_half_on_noise_and_low_with_real_edge(noise):
    # one noise draw can land anywhere near 0.5; the average over several should be close to it
    vals = [pbo(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (1600, 15))), max_combos=300)["pbo"]
            for s in range(6)]
    assert 0.35 < float(np.mean(vals)) < 0.65
    edge = noise.copy()
    edge.iloc[:, 3] += 0.002
    assert pbo(edge, max_combos=600)["pbo"] < 0.05


def test_cpcv_path_count_and_coverage(noise):
    out = cpcv(noise, n_groups=6, k_test=2)
    assert out["n_paths"] == n_cpcv_paths(6, 2) == 5
    assert out["paths"].notna().all().all()          # every bar is covered on every path
    assert out["n_splits"] == 15


def test_cpcv_purges_around_test_groups(noise):
    # a variant that is great only right next to the test groups must not be picked thanks to leakage
    out = cpcv(noise, n_groups=6, k_test=2, embargo=20)
    assert len(out["picks"]) == 15


def test_reality_check_noise_vs_edge(noise):
    rc = reality_check(noise, n_boot=300)
    assert rc["rc_p"] > 0.1 and rc["spa_p"] > 0.1
    edge = noise.copy()
    edge.iloc[:, 0] += 0.002
    rc2 = reality_check(edge, n_boot=300)
    assert rc2["rc_p"] < 0.05 and rc2["spa_p"] < 0.05 and rc2["best"] == 0


def test_auto_block_length_grows_with_dependence():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=3000)
    ar = np.zeros(3000)
    for i in range(1, 3000):
        ar[i] = 0.8 * ar[i - 1] + iid[i]
    assert auto_block_length(iid) < 3 < auto_block_length(ar)


def test_challenge_probabilities_and_leverage():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0006, 0.01, 2500))
    a = challenge_sim(r, days=60, leverage=1.0, n_sims=800)
    assert abs(a["pass"] + a["fail"] + a["timeout"] - 1) < 1e-9
    hi = challenge_sim(r, days=60, leverage=4.0, n_sims=800)
    assert hi["fail"] > a["fail"]                       # more leverage, more blow-ups


def test_trade_shuffle_keeps_trades():
    tr = np.array([0.05, -0.03, 0.02, -0.04, 0.06, -0.01, 0.03, -0.02])
    dds = trade_shuffle_drawdowns(tr, n_sims=200)
    assert (dds <= 0).all() and dds.min() >= -0.12


def test_trade_returns_segments():
    idx = pd.bdate_range("2020-01-01", periods=6)
    pos = pd.Series([0, 1, 1, 0, -1, 0.0], index=idx)
    r = pd.Series([0, 0, 0.1, 0.1, 0, 0.05], index=idx)
    tr = trade_returns(r, pos)
    assert np.allclose(tr, [0.21, 0.05])


def test_shocks_and_scenarios():
    df = generate("trending", n_days=1500, seed=3)
    v2 = shocked(df, "vol x2")
    assert np.log(v2["close"]).diff().std() == pytest.approx(2 * np.log(df["close"]).diff().std(), rel=1e-6)
    flat = shocked(df, "trend off")
    assert flat["close"].iloc[-1] == pytest.approx(df["close"].iloc[0], rel=1e-6)
    t = scenario_table(StrategySpec(src="Synthetic", rule="ema"), df)
    assert list(t["scenario"]) == list(SCENARIOS)
    assert t.set_index("scenario").loc["costs x3", "Sharpe"] < t.set_index("scenario").loc["base", "Sharpe"]


def test_top_clusters():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.integers(5, 100, 60), "y": rng.integers(5, 100, 60), "Sharpe": rng.normal(size=60)})
    tc = top_clusters(df, "Sharpe", ["x", "y"], top=30, k=3)
    assert len(tc) == 3 and tc["size"].sum() == 30


SRC = "import streamlit as st\nfrom zenith.backtest.views import lab\nlab.render()\n"


@pytest.mark.parametrize("sub,button", [("OVERFITTING", "bt_of_run"), ("CHALLENGE", "bt_ch_run"),
                                        ("STRESS", "bt_st_run")])
def test_panels_run_on_click(sub, button):
    at = AppTest.from_string(SRC, default_timeout=600)
    at.session_state["bt_src"] = "Synthetic"
    at.session_state["bt_synth"] = "trending"
    at.session_state["bt_step"] = "ROBUSTNESS"
    at.session_state["bt_rsub"] = sub
    at.session_state["bt_of_n"] = 5
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key=button).click().run()
    assert not at.exception, [e.value for e in at.exception]

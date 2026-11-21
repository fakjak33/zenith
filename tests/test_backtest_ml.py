"""ML LAB: leakage-free splits, known-answer learning, ml-mode causality, extras, and the page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from zenith.backtest.data.synthetic import generate
from zenith.backtest.ml.core import (MODELS, build_features, fwd_return, make_target, scorecard, splits,
                                     to_position, walk_forward)
from zenith.backtest.ml.extras import decompose, meta_label, trades_table, two_prop_z, wilson
from zenith.backtest.ml.strategy import MLStrategy
from zenith.backtest.spec import StrategySpec, run_spec

CFG = {"indicators": ["rsi", "tsmom", "er"], "lags": 3}


@pytest.fixture(scope="module")
def mr() -> pd.DataFrame:
    return generate("mean_reverting", n_days=2500, seed=2)


@pytest.mark.parametrize("scheme", ["expanding", "rolling", "split", "purged_kfold"])
def test_splits_never_train_on_overlapping_targets(scheme):
    h = 5
    for tr, te in splits(2000, scheme, n_folds=5, min_train=400, train_len=600, purge=h, embargo=3):
        assert len(set(tr) & set(te)) == 0
        lo = te.min()
        if scheme != "purged_kfold":
            assert tr.max() < lo - h + 1               # targets of training rows end before the test starts
        else:
            assert not np.any((tr >= lo - h) & (tr <= te.max() + 3))


def test_features_are_causal(mr):
    a = build_features(mr, {**CFG, "memory": 40, "calendar": True})
    b = build_features(mr.iloc[:1800], {**CFG, "memory": 40, "calendar": True})
    pd.testing.assert_frame_equal(a.iloc[:1800], b, check_freq=False)


def test_triple_barrier_labels_binary(mr):
    y = make_target(mr, "triple_barrier", 5, 1.0).dropna()
    assert set(y.unique()) <= {0.0, 1.0} and len(y) > 2000


def test_model_learns_mean_reversion_but_not_a_random_walk(mr):
    def lift(df):
        X, y, fr = build_features(df, CFG), make_target(df, "direction"), fwd_return(df)
        wf = walk_forward(X, y, "logistic", "clf")
        s = scorecard(wf["pred"], wf["y"], fr, to_position(wf["pred"], "clf"), "clf", wf["folds"], cost_bps=0)
        return s["accuracy"] - s["naive"]

    assert lift(mr) > 0.03
    rw = [lift(generate("gbm", n_days=2500, seed=s)) for s in (2, 3, 4, 5)]
    assert abs(float(np.mean(rw))) < 0.02


@pytest.mark.parametrize("model", [m for m, s in MODELS.items() if s.task in ("clf", "both") and m != "stack"])
def test_every_classifier_runs(mr, model):
    X, y = build_features(mr, CFG), make_target(mr, "direction")
    wf = walk_forward(X, y, model, "clf", n_folds=3)
    assert wf["pred"].dropna().between(0, 1).all()


def test_regressors_run(mr):
    X, fr = build_features(mr, CFG), fwd_return(mr)
    for m in ("linear", "ridge", "lasso", "gbm"):
        assert walk_forward(X, fr, m, "reg", n_folds=3)["pred"].notna().sum() > 500


def test_ml_mode_forecast_is_causal(mr):
    cfg = {"features": CFG, "retrain": 100, "min_train": 600}
    full = MLStrategy(cfg).forecast(mr)
    part = MLStrategy(cfg).forecast(mr.iloc[:1900])
    assert np.allclose(full.iloc[:1900], part)
    res = run_spec(StrategySpec(src="Synthetic", mode="ml", ml=cfg), mr)
    assert res.forecast.abs().max() <= 20 and res.stats["Sharpe"] > 0


def test_ml_spec_hash_roundtrip():
    s = StrategySpec(mode="ml", ml={"model": "gbm"})
    assert StrategySpec.from_json(s.to_json()).hash() == s.hash()
    assert StrategySpec(mode="ml", ml={"model": "logistic"}).hash() != s.hash()


def test_wilson_and_z():
    lo, hi = wilson(60, 100)
    assert 0.5 < lo < 0.6 < hi < 0.7
    z, p = two_prop_z(70, 100, 50, 100)
    assert z > 2.5 and p < 0.01


def test_meta_label_runs(mr):
    res = run_spec(StrategySpec(src="Synthetic", rule="i_rsi2"), mr)
    out = meta_label(trades_table(res.position, res.returns), build_features(mr, CFG), "logistic", 0.5)
    assert out["n_all"] > 50 and 0 <= out["win_taken"] <= 1


def test_decompose_runs(mr):
    d = decompose(build_features(mr, CFG), fwd_return(mr))
    assert d["rmse_dec"] > 0 and 0 <= d["size_gap_p"] <= 1


SRC = "import streamlit as st\nfrom zenith.backtest.views import ml_view\nml_view.render()\n"


def test_ml_page_full_click_through():
    at = AppTest.from_string(SRC, default_timeout=900)
    at.session_state["ml_src"] = "Synthetic"
    at.session_state["ml_synth"] = "mean_reverting"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="ml_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    for b in ("ml_cvrun", "ml_on_run", "ml_meta_run", "ml_dec_run", "ml_save"):
        at.button(key=b).click().run()
        assert not at.exception, (b, [e.value for e in at.exception])

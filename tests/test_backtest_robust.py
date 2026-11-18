"""Phase 2: spec, ledger, sensitivity/plateau, permutation, noise, walk-forward, ablation."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest import ledger as lg
from zenith.backtest.data.synthetic import generate
from zenith.backtest.robust.ablation import ablation, paired_sharpe_ci
from zenith.backtest.robust.sensitivity import default_values, plateau, sweep, to_grid
from zenith.backtest.robust.shuffle import add_noise, noise_curve, permutation_test, permute_ohlcv
from zenith.backtest.robust.walkforward import walk_forward
from zenith.backtest.spec import StrategySpec, apply_filters, run_spec, split_holdout


@pytest.fixture(scope="module")
def trending():
    return generate("trending", n_days=2500, seed=21)


def test_spec_roundtrip_and_hash():
    s = StrategySpec(rule="tsmom", params={"lookback": 120}, filters=[{"key": "trend", "on": True,
                                                                         "params": {"length": 200}}],
                     notes="trend persists")
    t = StrategySpec.from_json(s.to_json())
    assert t == s and t.hash() == s.hash()
    assert s.hash() == StrategySpec.from_dict({**s.to_dict(), "notes": "edited"}).hash()   # notes ≠ new trial
    assert s.hash() != s.with_params(lookback=121).hash()
    assert s.family() == s.with_params(lookback=60).family()


def test_split_holdout():
    df = generate("gbm", n_days=1000)
    ins, locked = split_holdout(df, 30)
    assert len(ins) == 700 and locked == df.index[700]
    assert split_holdout(df, 0)[1] is None


def test_filters_only_remove_exposure(trending):
    s = StrategySpec(src="Synthetic", rule="ema")
    fc = s.strategy().forecast(trending)
    for key in ("trend", "volregime", "momconfirm"):
        g = apply_filters(fc, trending, [{"key": key, "on": True, "params": {}}])
        assert (g.abs() <= fc.abs() + 1e-12).all()
        assert ((g == 0) | (g == fc)).all()


def test_ledger_trials_and_reveal(monkeypatch):
    monkeypatch.setenv("ZENITH_LOCAL", "0")
    led = lg._blank()
    assert lg.record(led, "fam", ["a", "b"]) == 2
    assert lg.record(led, "fam", "a") == 2
    assert lg.trials(led, "other") == 1
    r = lg.reveal(led, "fam", "a", "{}")
    assert lg.reveal(led, "fam", "zzz", "{}") == r          # one-time only
    assert not lg.contaminated(led, "fam", "a")
    assert lg.contaminated(led, "fam", "b")


def test_plateau_prefers_broad_optimum_over_spike():
    x = np.arange(15)
    z = np.exp(-((x - 4) ** 2) / 8.0)       # broad hill at 4
    z[12] = 1.3                               # lone spike at 12
    pl = plateau(pd.DataFrame([z], index=[0], columns=x))
    assert pl["peak"]["x"] == 12
    assert abs(pl["plateau"]["x"] - 4) <= 1
    assert pl["robustness"] < 1.0


def test_sweep_and_grid(trending):
    s = StrategySpec(src="Synthetic", rule="ema", params={"fast": 16, "slow": 64})
    xs = default_values(s, "fast", 5)
    df, hashes = sweep(s, trending, "fast", xs, "slow", [32, 64, 128])
    assert len(df) == len(xs) * 3 and len(set(hashes)) == len(hashes)
    g = to_grid(df, "Sharpe")
    assert g.shape == (3, len(xs))


def test_permutation_preserves_path_ends_and_bars(trending):
    p = permute_ohlcv(trending, np.random.default_rng(0))
    assert p["close"].iloc[-1] == pytest.approx(trending["close"].iloc[-1], rel=1e-9)
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()
    assert not np.allclose(p["close"].to_numpy(), trending["close"].to_numpy())


def test_permutation_detects_planted_edge(trending):
    s = StrategySpec(src="Synthetic", rule="tsmom", params={"lookback": 90})
    out = permutation_test(s, trending, n=60, seed=3)
    assert out["p"] < 0.05


def test_permutation_does_not_flag_random_walk():
    df = generate("gbm", n_days=2500, seed=8)
    s = StrategySpec(src="Synthetic", rule="tsmom", params={"lookback": 90})
    out = permutation_test(s, df, n=60, seed=4)
    assert out["p"] > 0.05


def test_noise_k0_is_the_base_run(trending):
    s = StrategySpec(src="Synthetic", rule="ema")
    nc = noise_curve(s, trending, ks=(0.0, 0.5), seeds=3)
    base = run_spec(s, trending).stats["Sharpe"]
    assert nc[nc.k == 0]["Sharpe"].iloc[0] == pytest.approx(base)
    noisy = add_noise(trending, 0.5, np.random.default_rng(1))
    assert (noisy["high"] >= noisy["low"]).all()


def test_walk_forward_retunes(trending):
    s = StrategySpec(src="Synthetic", rule="tsmom", params={"lookback": 90})
    wf = walk_forward(s, trending, "lookback", [30, 60, 120, 240], n_folds=3)
    assert len(wf["folds"]) == 3
    assert wf["oos"].index.is_monotonic_increasing
    assert np.isfinite(wf["wf_sharpe"])


def test_noop_filter_adds_no_value(trending):
    s = StrategySpec(src="Synthetic", rule="ema",
                     filters=[{"key": "volregime", "on": True, "params": {"lookback": 252, "max_pct": 100}}])
    tbl, hashes = ablation(s, trending)
    row = tbl[tbl.variant == "only volregime"].iloc[0]
    assert row["ΔSharpe"] == pytest.approx(0.0)
    assert row["CI lo"] <= 0 <= row["CI hi"]
    assert row["verdict"] == "no clear value"


def test_paired_ci_brackets_true_difference():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.001, 0.01, 3000))
    d, lo, hi = paired_sharpe_ci(a, a * 0.5, 252)
    assert d == pytest.approx(0.0, abs=1e-9) and lo <= 0 <= hi

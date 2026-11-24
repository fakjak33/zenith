"""Phase 1: explicit sizing modes, synthetic seeds / ticker kinds, diversification math."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.config import BacktestConfig
from zenith.backtest.core.volatility import blended_vol, position_from_forecast
from zenith.backtest.data.synthetic import TICKER_KINDS, from_ticker, generate, stationary_bootstrap_idx
from zenith.backtest.engine import diversify as dv
from zenith.backtest.engine.engine import run_single
from zenith.backtest.risk.atr import atr
from zenith.backtest.spec import StrategySpec, run_spec


@pytest.fixture(scope="module")
def df():
    return generate("trending", n_days=1500, seed=7)


def _fc(df, v=10.0):
    return pd.Series(v, index=df.index)


# ------------------------------------------------------------------ sizing ----
def test_fixed_pct_matches_hand_calc(df):
    px = df["close"]
    u = position_from_forecast(_fc(df), px, 1e6, 0.2, sizing_mode="fixed_pct", fixed_pct=0.10)
    assert np.allclose((u * px).iloc[-1], 100_000)            # 10% of $1m at forecast 10


def test_fixed_pct_scale_toggle(df):
    px = df["close"]
    on = position_from_forecast(_fc(df, 20), px, 1e6, 0.2, sizing_mode="fixed_pct", fixed_pct=0.1)
    off = position_from_forecast(_fc(df, 20), px, 1e6, 0.2, sizing_mode="fixed_pct", fixed_pct=0.1,
                                 scale_by_forecast=False)
    assert np.allclose((on * px).iloc[-1], 200_000) and np.allclose((off * px).iloc[-1], 100_000)


def test_fixed_risk_loses_risk_pct_at_stop(df):
    px = df["close"]
    dist = 3 * atr(df)
    u = position_from_forecast(_fc(df), px, 1e6, 0.2, sizing_mode="fixed_risk", risk_pct=0.10,
                               stop_distance=dist, scale_by_forecast=False)
    i = -1
    assert np.isclose(u.iloc[i] * dist.iloc[i], 100_000)        # moving 3 ATR against = 10% of capital


def test_vol_target_matches_formula(df):
    px = df["close"]
    u = position_from_forecast(_fc(df), px, 1e6, 0.25, sizing_mode="vol_target")
    bv = blended_vol(px)
    assert np.isclose(u.iloc[-1], 1e6 * 0.25 / (bv.iloc[-1] * px.iloc[-1]))


def test_leverage_cap(df):
    px = df["close"]
    u = position_from_forecast(_fc(df), px, 1e6, 0.2, sizing_mode="fixed_pct", fixed_pct=3.0, max_leverage=1.5)
    assert ((u * px).abs() <= 1.5e6 + 1e-6).all()


def test_engine_fixed_risk_runs(df):
    r = run_single(_fc(df), df, bt=BacktestConfig(sizing_mode="fixed_risk", risk_pct=0.01, stop_mult=2.0))
    assert r.position.abs().sum() > 0 and np.isfinite(r.stats["Sharpe"])


def test_spec_fixed_risk_forces_stop():
    s = StrategySpec(sizing_mode="fixed_risk")
    assert s.risk().use_atr_stop and s.bt().sizing_mode == "fixed_risk"


# -------------------------------------------------------------------- hash ----
def test_hash_ignores_unused_knobs():
    a = StrategySpec(sizing_mode="fixed_pct", fixed_pct=0.1, target_vol=0.2)
    b = StrategySpec(sizing_mode="fixed_pct", fixed_pct=0.1, target_vol=0.35)
    assert a.hash() == b.hash()                                   # target vol unused under fixed %
    assert a.hash() != StrategySpec(sizing_mode="fixed_pct", fixed_pct=0.2).hash()
    v = StrategySpec()
    assert v.hash() == StrategySpec(fixed_pct=0.5, risk_pct=0.05).hash()   # vol target ignores both


def test_hash_stable_for_old_specs():
    # a spec saved before the new fields existed keeps its hash
    old = {k: v for k, v in StrategySpec().to_dict().items()
           if k not in ("fixed_pct", "risk_pct", "scale_by_forecast", "max_leverage", "seed", "synth_base",
                         "start_date", "end_date", "lock_years")}
    import hashlib
    import json
    old.pop("notes")
    want = hashlib.sha1(json.dumps(old, sort_keys=True, default=str).encode()).hexdigest()[:12]
    assert StrategySpec().hash() == want


def test_seed_changes_synthetic_hash_and_family():
    a = StrategySpec(src="Synthetic", name="trending", seed=1)
    b = StrategySpec(src="Synthetic", name="trending", seed=2)
    assert a.hash() != b.hash() and a.family() != b.family()
    assert StrategySpec(seed=1).hash() == StrategySpec(seed=2).hash()      # market data ignores seed


# --------------------------------------------------------------- synthetic ----
def test_generate_seed_reproducible():
    a, b = generate("regime", 500, seed=3), generate("regime", 500, seed=3)
    assert a.equals(b) and not a.equals(generate("regime", 500, seed=4))


@pytest.mark.parametrize("kind", TICKER_KINDS)
def test_from_ticker_kinds(df, kind):
    out = from_ticker(kind, df, seed=5)
    assert len(out) == len(df) - 1 and (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
    assert out.equals(from_ticker(kind, df, seed=5))
    assert not out["close"].equals(from_ticker(kind, df, seed=6)["close"])


def test_shuffle_keeps_return_distribution(df):
    out = from_ticker("shuffle", df, seed=1)
    a = np.round(df["close"].pct_change().dropna().to_numpy(), 10)
    b = np.round(out["close"].pct_change().dropna().to_numpy(), 10)
    assert np.isin(b, a).all()                                 # same bars, new order


def test_stationary_bootstrap_blocks():
    idx = stationary_bootstrap_idx(1000, 5000, 20, np.random.default_rng(0))
    runs = np.mean(np.diff(idx) == 1)
    assert 0.9 < runs < 0.99                                     # mean block ≈ 20 → ~95% continuation


def test_synthetic_spec_runs_on_ticker_kind(df, monkeypatch):
    spec = StrategySpec(src="Synthetic", name="bootstrap", seed=9, holdout=0)
    res = run_spec(spec, from_ticker("bootstrap", df, seed=9))
    assert len(res.returns) > 100


# ---------------------------------------------------------- diversification ----
def _pair(rho, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0004, 0.01, n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(0.0004, 0.01, n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(b, idx), pd.Series(a, idx)


def test_corr_and_beta():
    s, u = _pair(0.6)
    d = dv.analyse(s, u, 252)
    assert abs(d["corr"] - 0.6) < 0.05 and abs(d["beta"] - 0.6) < 0.07


def test_add_rule_equivalent_to_alpha():
    for rho in (-0.3, 0.2, 0.9):
        s, u = _pair(rho, seed=int(rho * 10) + 5)
        d = dv.analyse(s, u, 252)
        assert d["adds"] == (d["alpha"] > 0)                       # SR_s > ρ·SR_u  ⇔  μ_s > β·μ_u


def test_add_rule_matches_frontier_slope():
    s, u = _pair(0.3, seed=11)
    d = dv.analyse(s, u, 252)
    fr = d["frontier"]
    assert d["adds"] == (fr["Sharpe"].iloc[1] > fr["Sharpe"].iloc[0])


def test_frontier_endpoints():
    s, u = _pair(0.1)
    fr = dv.blend_frontier(s, u, 252)
    assert np.isclose(fr["Sharpe"].iloc[0], dv._sharpe(u, 252)) and np.isclose(fr["Sharpe"].iloc[-1],
                                                                                dv._sharpe(s, 252))

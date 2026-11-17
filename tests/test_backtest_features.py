import numpy as np
import pandas as pd

from zenith.backtest.config import BacktestConfig, RiskConfig
from zenith.backtest.data.synthetic import generate
from zenith.backtest.rules import REGISTRY
from zenith.backtest.engine.engine import run_strategy
from zenith.backtest.engine import montecarlo, metrics, kelly


def test_sizing_modes_differ():
    df = generate("trending", n_days=800, seed=5)
    eqs = {}
    for m in ["vol_target", "fixed_pct", "fixed_dollar"]:
        r = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(sizing_mode=m))
        eqs[m] = r.equity.iloc[-1]
    assert abs(eqs["vol_target"] - eqs["fixed_pct"]) > 1.0


def test_percent_stop_triggers():
    n = 80
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.r_[np.linspace(100, 120, 40), np.linspace(120, 80, 40)], index=idx)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": 1000}, index=idx)
    from zenith.backtest.risk.overlays import apply_risk_overlay
    pos = pd.Series(1.0, index=idx)
    cfg = RiskConfig(use_pct_stop=True, pct_stop=0.05, trailing_stop=True)
    gated, ev = apply_risk_overlay(pos, df, cfg)
    assert (ev["type"] == "stop").any()
    assert (gated == 0.0).any()


def test_atr_stop_still_works_after_refactor():
    from zenith.backtest.risk.overlays import apply_atr_overlay, apply_risk_overlay
    assert apply_atr_overlay is apply_risk_overlay


def test_kelly():
    assert kelly.kelly_discrete(0.6, 2.0) > 0
    assert kelly.kelly_discrete(0.3, 1.0) == 0.0  # no edge -> 0
    df = generate("trending", n_days=800, seed=2)
    r = run_strategy(REGISTRY["ema"](), df)
    kc = kelly.kelly_continuous(r.returns)
    assert kc["half"] == kc["full"] / 2


def test_capture_and_drawdown():
    df = generate("gbm", n_days=600, seed=3)
    r = run_strategy(REGISTRY["ema"](), df)
    bench = df["close"].pct_change()
    assert metrics.annual_vol(r.returns) >= 0
    dd = metrics.drawdown_series(r.returns)
    assert (dd <= 1e-9).all()  # drawdown never positive
    uc = metrics.upside_capture(r.returns, bench)
    dc = metrics.downside_capture(r.returns, bench)
    assert np.isfinite(uc) and np.isfinite(dc)


def test_mc_paths_shape():
    df = generate("trending", n_days=600, seed=4)
    r = run_strategy(REGISTRY["ema"](), df)
    paths = montecarlo.bootstrap_paths(r.returns, n_sims=30, capital=1e6)
    assert paths.shape[1] == 30
    assert (paths.iloc[0] > 0).all()


def test_timeframe_scaling():
    from zenith.backtest.config import scale_params_for_tf
    out = scale_params_for_tf({"fast": 16, "slow": 60, "speed": 1.0}, "Weekly")
    assert out["slow"] == 12 and out["fast"] == 3  # divided by ~5
    assert out["speed"] == 1.0  # non-time param unchanged
    same = scale_params_for_tf({"fast": 16}, "Daily")
    assert same["fast"] == 16

"""Phase 4: generic knob sweeps for every BACKTEST tool."""

import numpy as np
import pandas as pd

from zenith.backtest.data.synthetic import generate
from zenith.backtest.options import engine as E
from zenith.backtest.robust.sensitivity import default_values, knob_choices, set_knob, sweep
from zenith.backtest.spec import StrategySpec
from zenith.backtest.views.options_view import apply_option_knobs, option_knobs
from zenith.backtest.views.portfolio_view import vol_target_returns
from zenith.backtest.views.sensitivity_panel import Knob, grid_run


def test_knob_defaults():
    k = Knob("a", "a", 20, 5, 120, integer=True)
    v = k.defaults(7)
    assert all(isinstance(x, int) for x in v) and min(v) >= 5 and max(v) <= 120 and 20 in range(min(v), max(v) + 1)
    assert Knob("c", "c", "x", choices=["x", "y"]).defaults(5) == ["x", "y"]
    lg = Knob("l", "l", 100, 10, 1000, log=True).defaults(5)
    assert np.isclose(lg[1] / lg[0], lg[2] / lg[1], rtol=0.01)


def test_grid_run_shapes_and_holes():
    kx, ky = Knob("x", "x", 1, 0, 2), Knob("y", "y", 1, 0, 2)

    def run(v):
        if v["x"] == 2 and v["y"] == 2:
            raise ValueError("hole")
        return {"Sharpe": v["x"] - v["y"]}

    df = grid_run(run, kx, [0, 1, 2], ky, [0, 1, 2], ["Sharpe"])
    assert len(df) == 9 and df["Sharpe"].isna().sum() == 1


def test_lab_knobs_cover_filters_and_sizing():
    spec = StrategySpec(sizing_mode="fixed_risk", filters=[{"key": "trend", "on": True, "params": {"length": 200}}])
    ks = knob_choices(spec)
    assert "f:trend:length" in ks and "s:risk_pct" in ks and "s:atr_stop_mult" in ks and "s:target_vol" not in ks
    assert set_knob(spec, "f:trend:length", 100).filters[0]["params"]["length"] == 100
    assert set_knob(spec, "s:risk_pct", 0.02).risk_pct == 0.02
    assert set_knob(spec, "fast", 8).params["fast"] == 8
    assert len(default_values(spec, "s:risk_pct", 5)) >= 3


def test_lab_sweep_over_spec_knob():
    df = generate("trending", 1200, seed=3)
    spec = StrategySpec(src="Synthetic", name="trending", holdout=0)
    out, hashes = sweep(spec, df, "s:target_vol", [0.1, 0.2, 0.4])
    assert np.allclose(out["Sharpe"], out["Sharpe"].iloc[0], atol=1e-6)       # Sharpe is leverage-free
    assert out["Vol"].is_monotonic_increasing if "Vol" in out else True
    assert len(set(hashes)) == 3


def test_option_knobs_apply():
    s = E.TEMPLATES["Iron condor (16Δ / 5Δ)"]
    names = [k.name for k in option_knobs(s)]
    assert names[:4] == ["leg0", "leg1", "leg2", "leg3"] and "dte" in names
    s2 = apply_option_knobs(s, {"leg0": 0.2, "dte": 45, "take_profit": 0.5})
    assert s2.legs[0].value == 0.2 and s2.dte == 45 and s2.take_profit == 0.5 and s.legs[0].value == 0.16


def test_portfolio_vol_target_scales():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.02, 2000), index=pd.bdate_range("2010-01-01", periods=2000))
    out = vol_target_returns(r, 0.10)
    assert abs(out.iloc[300:].std() * np.sqrt(252) - 0.10) < 0.02
    assert vol_target_returns(r, 0.0).equals(r)

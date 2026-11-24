"""No-code builder: conditions, blends, regime rules, templates, round-trips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.builder import TEMPLATES, apply_regime, conditions_state, random_mask_test, sentence
from zenith.backtest.data.synthetic import generate
from zenith.backtest.spec import StrategySpec, run_spec

CLOSE = {"src": "close"}


def N(v):
    return {"src": "num", "value": v}


def I(k, **p):
    return {"src": "ind", "key": k, "params": p}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate("trending", n_days=2500, seed=21)


def test_state_machine_entry_exit_and_flip():
    idx = pd.bdate_range("2020-01-01", periods=8)
    frame = pd.DataFrame({"close": [1, 5, 6, 2, 1, 9, 9, 1.0]}, index=idx)
    frame["open"] = frame["high"] = frame["low"] = frame["close"]
    conds = {"long_entry": {"rows": [{"a": CLOSE, "op": ">", "b": N(4)}]},
             "long_exit": {"rows": [{"a": CLOSE, "op": "<", "b": N(3)}]},
             "short_entry": {"rows": [{"a": CLOSE, "op": "<", "b": N(1.5)}]}}
    s = conditions_state(conds, frame).tolist()
    # bar0 short (1<1.5); bar1 long entry flips? short has no exit rule -> exits when entry stops, then long
    assert s == [-1, 1, 1, 0, -1, 1, 1, -1]


def test_empty_exit_means_hold_while_entry_true(df):
    conds = {"long_entry": {"rows": [{"a": CLOSE, "op": ">", "b": I("sma", length=50)}]}}
    s = conditions_state(conds, df)
    sma = df["close"].rolling(50).mean()
    assert ((s == 1) == (df["close"] > sma)).all()


def test_crosses_above_fires_once():
    idx = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({"close": [1, 3, 4, 2, 5.0]}, index=idx)
    from zenith.backtest.builder.compose import row_mask
    m = row_mask({"a": CLOSE, "op": "crosses above", "b": N(2.5)}, frame).tolist()
    assert m == [False, True, False, False, True]


def test_or_join(df):
    from zenith.backtest.builder.compose import group_mask
    g = {"join": "OR", "rows": [{"a": I("rsi", length=14), "op": "<", "b": N(30)},
                                {"a": I("rsi", length=14), "op": ">", "b": N(70)}]}
    m = group_mask(g, df)
    from zenith.backtest.indicators import CATALOG
    r = CATALOG["rsi"](df, length=14)
    assert (m == ((r < 30) | (r > 70))).all()


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_templates_run_and_are_causal(df, name):
    conds = TEMPLATES[name]["conds"]
    spec = StrategySpec(src="Synthetic", mode="conditions", conditions=conds)
    res = run_spec(spec, df)
    assert np.isfinite(res.stats["Sharpe"])
    full = conditions_state(conds, df)
    part = conditions_state(conds, df.iloc[:1800])
    assert (full.iloc[:1800] == part).all()
    assert sentence(conds)


def test_blend_is_weighted_and_capped(df):
    spec = StrategySpec(src="Synthetic", mode="blend", blend=[{"rule": "ema", "params": {}, "weight": 1.0},
                                                              {"rule": "i_tsmom", "params": {}, "weight": 3.0}])
    fc = spec.strategy().forecast(df)
    assert fc.abs().max() <= 20 + 1e-9
    only = StrategySpec(src="Synthetic", mode="blend", blend=[{"rule": "ema", "params": {}, "weight": 1.0}])
    single = StrategySpec(src="Synthetic", rule="ema")
    assert np.allclose(only.strategy().forecast(df), single.strategy().forecast(df))


def test_regime_filter_zeroes_outside_allowed(df):
    fc = pd.Series(10.0, index=df.index)
    reg = {"on": True, "det": "trend", "use": "filter", "allow": ["uptrend"], "min_bars": 1}
    out = apply_regime(reg, fc, df)
    from zenith.backtest.regime import detect
    lab = detect("trend", df)
    assert (out[lab != "uptrend"] == 0).all() and (out[lab == "uptrend"] == 10).all()


def test_regime_size_and_switch(df):
    fc = pd.Series(10.0, index=df.index)
    lab_reg = {"on": True, "det": "trend", "min_bars": 1}
    out = apply_regime({**lab_reg, "use": "size", "scale": {"uptrend": 1.5, "downtrend": 0.5}}, fc, df)
    assert set(np.round(out.dropna().unique(), 6)) <= {15.0, 5.0, 0.0}
    sw = apply_regime({**lab_reg, "use": "switch", "switch": {"downtrend": "flat"}}, fc, df)
    from zenith.backtest.regime import detect
    assert (sw[detect("trend", df) == "downtrend"] == 0).all()


def test_random_mask_test_finds_real_filter_and_rejects_luck():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    lab = pd.Series(np.repeat(["good", "bad"] * 15, 100), index=idx)
    r = pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    r[lab.shift(1) == "good"] += 0.002              # real edge in 'good'
    real = random_mask_test(r, lab, {"good"}, n=300)
    assert real["p"] < 0.05
    noise = pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    luck = random_mask_test(noise, lab, {"good"}, n=300)
    assert luck["p"] > 0.05


def test_compose_spec_hash_and_roundtrip():
    spec = StrategySpec(mode="conditions", conditions=TEMPLATES["Faber 10-month trend"]["conds"],
                        regime={"on": True, "det": "vol", "use": "filter", "allow": ["calm"]})
    back = StrategySpec.from_json(spec.to_json())
    assert back.hash() == spec.hash()
    # changing the unused single-rule fields does not change a conditions strategy's hash
    assert StrategySpec(**{**spec.to_dict(), "rule": "sma"}).hash() == spec.hash()
    # default spec hash is untouched by the new fields
    assert StrategySpec(mode="single", blend=[], conditions={}, regime={}).hash() == StrategySpec().hash()


def test_conditions_widgets_roundtrip():
    from streamlit.testing.v1 import AppTest
    from zenith.backtest.views import common as Cm
    spec = StrategySpec(src="Synthetic", name="trending", mode="conditions",
                        conditions=TEMPLATES["Connors RSI(2) pullback"]["conds"],
                        regime={"on": True, "det": "trend", "params": {"length": 200, "band": 0.0}, "min_bars": 3,
                                "use": "filter", "allow": ["uptrend"]})
    src = ("import streamlit as st\nfrom zenith.backtest.views import common as C\n"
           "st.session_state['_spec'] = C.builder().to_json()\n")
    at = AppTest.from_string(src, default_timeout=120)
    for k, v in Cm.spec_to_widgets(spec).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rebuilt = StrategySpec.from_json(at.session_state["_spec"])
    assert rebuilt.conditions == spec.conditions
    assert rebuilt.regime["allow"] == ["uptrend"] and rebuilt.regime["det"] == "trend"

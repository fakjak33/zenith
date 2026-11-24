"""LAB page, headless: every step renders, heavy steps run on click, OOS reveal flow."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

SRC = ("import streamlit as st\n"
       "from zenith.backtest.views import lab\nlab.render()\n")


def _app(step: str, **state) -> AppTest:
    at = AppTest.from_string(SRC, default_timeout=300)
    at.session_state["bt_src"] = "Synthetic"
    at.session_state["bt_synth"] = "trending"
    at.session_state["bt_step"] = step
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)


@pytest.mark.parametrize("step", ["DEFINE", "BUILD", "ATTRIBUTION", "OUT-OF-SAMPLE"])
def test_steps_render(step):
    _app(step)


def test_sensitivity_sweep_runs_and_counts_trials():
    at = _app("SENSITIVITY", bt_sn=5, bt_s2c=False)
    at.button(key="bt_srun").click().run()
    assert not at.exception, [e.value for e in at.exception]
    t = _text(at)
    assert "PLATEAU PICK" in t and "ROBUSTNESS" in t
    trials = at.session_state["bt_ledger"]["trials"]
    assert max(len(v) for v in trials.values()) >= 5


def test_permutation_runs():
    at = _app("ROBUSTNESS", bt_pn=20)
    at.button(key="bt_prun").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "p = " in _text(at)


def test_filters_ablation_runs():
    at = _app("FILTERS", bt_f_trend=True)
    at.button(key="bt_arun").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_oos_reveal_then_contamination():
    at = _app("OUT-OF-SAMPLE")
    assert "REVEAL" in " ".join(b.label for b in at.button)
    at.checkbox(key="bt_sure").check().run()
    at.button(key="bt_reveal").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "OUT-OF-SAMPLE SHARPE" in _text(at)
    # change the strategy after the reveal -> contaminated
    at.session_state["bt_p_ema_fast"] = 20
    at.run()
    assert "CONTAMINATED" in _text(at)


def test_attribution_step_renders_with_kpis():
    at = _app("ATTRIBUTION")
    t = _text(at)
    assert "ALPHA / YR" in t and "RETURN FROM BETA" in t


def test_verdict_card_on_oos_step():
    at = _app("OUT-OF-SAMPLE")
    t = _text(at)
    assert "evidence strength" in t.lower() and "Permutation test" in t


def test_portfolio_page_renders_empty():
    at = AppTest.from_string("from zenith.backtest.views import portfolio_view\nportfolio_view.render()\n",
                             default_timeout=120)
    at.session_state["bt_saved"] = {}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "SAVE STRATEGIES FIRST" in _text(at)


def test_portfolio_page_combines_synthetic_specs():
    from zenith.backtest import library
    from zenith.backtest.spec import StrategySpec
    store = {}
    for nm, kind, rule in (("a", "trending", "ema"), ("b", "mean_reverting", "meanrev"), ("c", "regime", "tsmom")):
        library.save(store, nm, StrategySpec(src="Synthetic", name=kind, rule=rule))
    at = AppTest.from_string("from zenith.backtest.views import portfolio_view\nportfolio_view.render()\n",
                             default_timeout=300)
    at.session_state["bt_saved"] = store
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "PORTFOLIO SHARPE" in _text(at)


_OPT_CACHE = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "stratidx" / "cache" / "VIX.csv"


@pytest.mark.skipif(not _OPT_CACHE.exists(), reason="needs cached Cboe VIX history")
def test_options_page_renders():
    at = AppTest.from_string("from zenith.backtest.views import options_view\noptions_view.render()\n",
                             default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    t = _text(at)
    assert "MODELED" in t and "SHARPE" in t


def test_learn_page_renders():
    at = AppTest.from_string("from zenith.backtest.views import learn_view\nlearn_view.render()\n", default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Permutation" in _text(at)


def test_library_lists_and_compares_synthetic():
    from zenith.backtest import library
    from zenith.backtest.spec import StrategySpec
    store = {}
    library.save(store, "trend", StrategySpec(src="Synthetic", name="trending", rule="ema"))
    library.save(store, "mr", StrategySpec(src="Synthetic", name="mean_reverting", rule="meanrev"))
    at = AppTest.from_string("from zenith.backtest.views import library_view\nlibrary_view.render()\n",
                             default_timeout=300)
    at.session_state["bt_saved"] = store
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "RESEARCH LEDGER" in _text(at)


def test_spec_to_widgets_roundtrip():
    """LIBRARY › open in LAB must rebuild the identical spec (same hash)."""
    from zenith.backtest.spec import StrategySpec
    from zenith.backtest.views import common as Cm
    spec = StrategySpec(src="Synthetic", name="regime", rule="tsmom", params={"lookback": 120, "smooth": 3},
                        filters=[{"key": "trend", "on": True, "params": {"length": 150}}], holdout=0,
                        lock_years=3.5, start_date="2008-06-02", cost_bps=2.0, buffer=0.1)
    at = AppTest.from_string(SRC, default_timeout=300)
    for k, v in Cm.spec_to_widgets(spec).items():
        at.session_state[k] = v
    at.session_state["bt_step"] = "DEFINE"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # read the spec LAB built back out of the ledger's trial family + recompute through the builder path
    rebuilt = StrategySpec(src=at.session_state["bt_src"], name=at.session_state["bt_synth"],
                           tf=at.session_state["bt_tf"], start_year=at.session_state["bt_start"].year,
                           start_date=at.session_state["bt_start"].isoformat(),
                           rule=at.session_state["bt_rule"],
                           params={k: at.session_state[f"bt_p_tsmom_{k}"] for k in ("lookback", "smooth")},
                           filters=[{"key": "trend", "on": True, "params": {"length": at.session_state["bt_fp_trend_length"]}}],
                           holdout=0, lock_years=at.session_state["bt_lock"], cost_bps=at.session_state["bt_cost"],
                           buffer=at.session_state["bt_buf"] / 100)
    assert rebuilt.params == spec.params and rebuilt.filters == spec.filters and rebuilt.lock_years == 3.5
    assert rebuilt.start_date == spec.start_date


@pytest.mark.skipif(not (__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "backtest" / "cache"
                         / "SPY.parquet").exists(), reason="needs cached SPY prices")
def test_accumulation_page_renders():
    at = AppTest.from_string("from zenith.backtest.views import accumulation_view\naccumulation_view.render()\n",
                             default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "FINAL EQUITY" in _text(at)


def test_accumulation_page_synthetic():
    at = AppTest.from_string("from zenith.backtest.views import accumulation_view\naccumulation_view.render()\n",
                             default_timeout=300)
    at.session_state["ac_src"] = "Synthetic"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "RETURN ON SAVED" in _text(at)

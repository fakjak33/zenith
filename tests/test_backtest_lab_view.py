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

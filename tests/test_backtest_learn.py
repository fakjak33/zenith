"""LEARN: every mode and every lesson renders; the glossary covers every hover source."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from zenith.backtest.views.learn_lessons import LESSONS

SRC = "import streamlit as st\nfrom zenith.backtest.views import learn_view\nlearn_view.render()\n"


@pytest.mark.parametrize("mode", ["PATH", "GLOSSARY", "REFERENCE"])
def test_modes_render(mode):
    at = AppTest.from_string(SRC, default_timeout=120)
    at.session_state["ln_mode"] = mode
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("lesson", list(LESSONS))
def test_lessons_render(lesson):
    at = AppTest.from_string(SRC, default_timeout=300)
    at.session_state["ln_mode"] = "LESSONS"
    at.session_state["ln_pick"] = lesson
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if lesson == "Leakage and purging":
        at.button(key="lk_run").click().run()
        assert not at.exception, [e.value for e in at.exception]
        shuffled, purged, naive = at.session_state["lk_out"]
        assert shuffled > naive + 0.1                      # leakage looks like skill
        assert abs(purged - naive) < 0.03                  # honest validation finds nothing beyond the base rate


def test_glossary_covers_everything():
    from zenith.backtest.indicators import CATALOG
    from zenith.backtest.views.learn_view import _glossary
    g = _glossary()
    assert len(g) > len(CATALOG) + 40
    assert g["meaning"].str.len().min() > 10
    assert {"metric", "regime", "ML model", "validation", "portfolio weighting"} <= set(g["area"].str.split(" · ").str[0])

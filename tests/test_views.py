"""Headless view rendering checks (streamlit.testing.v1.AppTest).

Views read only committed JSON under data/, so these are offline. They guard
the render path end-to-end: no exceptions, and the research/explanation
surfaces (key-findings strip, methodology sections) actually appear.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from zenith import config


def _render(src: str) -> tuple[AppTest, str]:
    at = AppTest.from_string(src, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(str(m.value) for m in at.markdown)
    text += " ".join(str(c.value) for c in at.caption)
    return at, text


@pytest.mark.skipif(not config.PRETOM_FILES["basket"].exists(),
                    reason="no committed pretom data")
def test_pretom_view_renders():
    at, text = _render("from zenith.pretom import view\nview.render()\n")
    assert "key findings" in text.lower()
    assert "Turn of the month" in text
    assert "Verdict from our data" in text          # honest TOM verdict line
    assert "Nathan, Suominen" in text


@pytest.mark.skipif(not config.PEAD_FILES["signals"].exists(),
                    reason="no committed pead data")
def test_pead_view_renders():
    at, text = _render("from zenith.pead import view\nview.render()\n")
    assert "key findings" in text.lower()
    assert "Announcement premium" in text
    assert "Frazzini" in text
    if config.PEAD_FILES["eap"].exists():
        assert "Skew check" in text                 # right-skew caveat shown

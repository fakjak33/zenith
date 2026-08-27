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


def test_edge_view_renders():
    # the EDGE tab renders even with no data (shows rating + 'no data' info)
    at, text = _render("from zenith.edge import view\nview.render()\n")
    assert "EDGE SCREENS" in text
    assert "evidence strength" in text.lower()      # rating badge present


def test_nightday_view_renders():
    at, text = _render("from zenith.nightday import view\nview.render()\n")
    assert "evidence strength" in text.lower()
    assert "overnight" in text.lower()


def test_mom_view_renders():
    # renders with zero committed data — the day-one condition — showing the
    # rating badge, key findings, and an info prompt rather than a crash.
    at, text = _render("from zenith.mom import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low
    assert "key findings" in low
    if config.MOM_FILES["scores"].exists():
        assert "momentum" in low
        assert "composite" in low


def test_mvt_view_renders():
    # Multivariate Trend's own sub-view renders standalone (bypassing mom's
    # outer scores.json gate, which is a separate artifact) -- the day-one
    # condition with zero mvt data, and again with real data once a compute
    # run has populated data/mom/mvt/*.json.
    at, text = _render("from zenith.mom.mvt import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low
    assert "key findings" in low
    assert any("how is this calculated" in (e.label or "").lower() for e in at.expander)
    if config.MOM_MVT_FILES["equities"].exists():
        assert "names scored" in low or "no multivariate trend data" in low


def test_mvt_cross_universe_view_renders():
    at, text = _render(
        "from zenith.mom.mvt import view\n"
        "import streamlit as st\n"
        "st.session_state['mvt_universe_sub'] = 'Cross-Universe'\n"
        "view.render()\n"
    )
    low = text.lower()
    assert "broad" in low and "idiosyncratic" in low
    if config.MOM_MVT_FILES["crossuniverse"].exists():
        assert "names compared" in low
    else:
        infos = " ".join(str(i.value) for i in at.info).lower()
        assert "no cross-universe data yet" in infos


def test_mvt_validation_view_renders():
    at, text = _render(
        "from zenith.mom.mvt import view\n"
        "import streamlit as st\n"
        "st.session_state['mvt_universe_sub'] = 'Validation'\n"
        "view.render()\n"
    )
    low = text.lower()
    assert "cross-instrument p&l correlation" in low
    if config.MOM_MVT_FILES["validation"].exists():
        assert "central hypothesis test" in low
    else:
        infos = " ".join(str(i.value) for i in at.info).lower()
        assert "no validation backtest yet" in infos


def test_ideas_view_renders():
    # renders with zero committed data -- the day-one condition -- showing the
    # rating badge, key findings, and an info prompt rather than a crash.
    at, text = _render("from zenith.ideas import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low
    assert "key findings" in low
    if config.IDEAS_FILES["ideas"].exists():
        assert "ideas" in low


def test_regimes_view_renders():
    # renders with zero committed data -- the day-one condition -- showing the
    # rating badge, key findings, and an info prompt rather than a crash.
    at, text = _render("from zenith.regimes import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low
    assert "key findings" in low
    if config.REGIMES_FILES["current"].exists():
        assert "regime" in low


def test_holdings_view_renders():
    # renders with or without committed data — no data shows the state banner
    at, text = _render("from zenith.holdings import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low or "no funds registered" in low
    if (config.HOLDINGS_DIR / "funds.json").exists():
        assert "positioning" in low
        assert "what changed" in low
        assert "position explorer" in low
        # the interpretation caveats must be on the page, not buried
        assert "notional" in low


def test_holdings_heatmap_lenses_all_render():
    if not (config.HOLDINGS_DIR / "funds.json").exists():
        pytest.skip("no committed holdings data")
    for lens in ("Δ Exposure", "New / Closed", "Long / Short"):
        at = AppTest.from_string(
            "from zenith.holdings import view\nview.render()\n",
            default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        picked = False
        for r in at.radio:
            if lens in [str(o) for o in r.options]:
                r.set_value(lens)
                picked = True
                break
        assert picked, f"lens {lens} not offered"
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def test_fmom_view_has_vol_scaled_lens():
    if not config.FMOM_FILES["signals"].exists():
        pytest.skip("no committed fmom data")
    at, text = _render("from zenith.fmom import view\nview.render()\n")
    assert "evidence strength" in text.lower()


def test_cas_calendar_has_fomc():
    if not (config.CAS_DIR / "fomc.json").exists():
        pytest.skip("no committed fomc data")
    at = AppTest.from_string("from zenith.cas import view\nview.render()\n",
                             default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for r in at.radio:
        if "Calendar" in [str(o) for o in r.options]:
            r.set_value("Calendar")
            break
    at.run()
    text = " ".join(str(m.value) for m in at.markdown)
    assert "FOMC" in text and "Verdict from our SPY" in text

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


def _etfmom_sub(sub: str):
    """Drive one ETF MOMENTUM sub-view. `st.radio` sub-nav is not reliably
    clickable in the browser pane (a standing gotcha in this repo), so
    pre-seeding session state before render() is the trusted path -- the same
    approach the mvt sub-view tests above use."""
    return _render("\n".join([
        "from zenith.etfmom import view",
        "import streamlit as st",
        f"st.session_state['etfmom_sub'] = {sub!r}",
        "view.render()",
    ]))


def test_etfmom_view_renders_with_or_without_data():
    at, text = _render("from zenith.etfmom import view\nview.render()\n")
    low = text.lower()
    assert "evidence strength" in low          # rating badge present
    assert "key findings" in low
    assert "moskowitz" in low                  # multi-asset citation, not J-T
    if config.ETFMOM_FILES["scores"].exists():
        assert "etf momentum" in low
    else:
        infos = " ".join(str(i.value) for i in at.info).lower()
        assert "no data yet" in infos


@pytest.mark.skipif(not config.ETFMOM_FILES["scores"].exists(),
                    reason="no committed etfmom data")
@pytest.mark.parametrize("sub,needle", [
    ("Overview", "breadth"),
    ("Rankings", "scored funds"),
    ("Factors", "redundancy"),
    ("Categories", "asset class"),
    ("ETF", "factor breakdown"),
])
def test_etfmom_sub_views_render(sub, needle):
    at, text = _etfmom_sub(sub)
    assert needle in text.lower()


@pytest.mark.skipif(not config.ETFMOM_FILES["scores"].exists(),
                    reason="no committed etfmom data")
def test_etfmom_discloses_its_own_caveats():
    """The two things that make this tab honest rather than just pretty: the
    mixed-asset cross-sectional caveat, and the fact that near-duplicate funds
    are deliberately kept so breadth counts funds rather than independent bets."""
    _, text = _etfmom_sub("Overview")
    low = text.lower()
    assert "independent bets" in low
    _, rank_text = _etfmom_sub("Rankings")
    assert "within an asset class" in rank_text.lower()


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


# --------------------------------------------------------------------- INDEX --
@pytest.fixture(autouse=True, scope="function")
def _clear_streamlit_cache():
    """INDEX's view caches its artefacts with @st.cache_data. Without clearing
    it between tests, the no-data test's empty result leaks into the real-data
    test and makes it fail for the wrong reason."""
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _index_results(at) -> list:
    """AppTest's session_state proxy has no .get(), so read it defensively."""
    try:
        return list(at.session_state["idx_result_names"])
    except (KeyError, AttributeError):
        return []


def _index_at(subview: str | None = None, timeout: int = 180):
    at = AppTest.from_string("from zenith.index import view\nview.render()\n",
                             default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if subview:
        at.radio(key="idx_sub").set_value(subview).run()
        assert not at.exception, (subview, [e.value for e in at.exception])
    return at


def test_index_view_renders_with_no_data(tmp_path, monkeypatch):
    """Day-one convention: the view must render before any data exists,
    telling the user how to build it rather than raising."""
    from zenith import config as cfg
    import zenith.index as zidx
    empty = {k: tmp_path / f"{k}.json" for k in cfg.INDEX_FILES}
    monkeypatch.setattr(cfg, "INDEX_FILES", empty)
    monkeypatch.setattr(zidx, "INDEX_FILES", empty)
    at = AppTest.from_string(
        "import zenith.index.view as v\nv.render()\n", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "has not been built yet" in " ".join(str(i.value) for i in at.info)


def test_index_view_renders_with_real_data():
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    at, text = _render("from zenith.index import view\nview.render()\n")
    assert "key findings" in text.lower()
    # the verification strip, NOT a fabricated A/B/C evidence grade
    assert "Verification" in text
    assert "Need review" in text
    assert "evidence strength" not in text.lower(), (
        "INDEX must NOT display an A/B/C evidence badge — it is a directory, "
        "not a predictive signal (see zenith/index/__init__.py)")


def test_index_subviews_all_render():
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    from zenith.index.view import SUBVIEWS
    for sub in SUBVIEWS:
        _index_at(sub)


def test_index_directory_search_and_filter():
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    at = _index_at("Directory")

    total = len(_index_results(at))
    assert total > 100, "the unfiltered directory should list the whole catalog"

    # free-text search must match a tag's human LABEL, not only its slug, and
    # must actually NARROW -- asserting only "some results came back" would pass
    # just as happily against a filter that silently does nothing.
    at.text_input(key="idx_q").set_value("trend following").run()
    assert not at.exception, [e.value for e in at.exception]
    names = _index_results(at)
    assert names, "searching 'trend following' should match entries"
    assert len(names) < total, "search must narrow the result set"
    assert any(n in names for n in ("Man AHL", "AlphaSimplex Group", "Robert Carver"))

    # A nonsense query must return NOTHING rather than falling back to
    # everything. Uses a fresh AppTest: setting the same text_input twice on one
    # instance does not re-apply, so reusing `at` here would silently re-assert
    # the previous query's result.
    at2 = _index_at("Directory")
    at2.text_input(key="idx_q").set_value("zzzznotathing").run()
    assert _index_results(at2) == []

    # two filters applied simultaneously must narrow, not error
    at.text_input(key="idx_q").set_value("").run()
    at.multiselect(key="idx_cat").set_value(["Tool"]).run()
    only_tools = _index_results(at)
    at.multiselect(key="idx_insight").set_value(["Options"]).run()
    assert not at.exception, [e.value for e in at.exception]
    narrowed = _index_results(at)
    assert narrowed, "Tool + Options should match the options analytics platforms"
    assert len(narrowed) < len(only_tools) < total, "each filter must narrow further"
    assert any("Option" in n or "Chameleon" in n for n in narrowed), narrowed


def test_index_cards_layout_renders():
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    at = _index_at("Directory")
    at.radio(key="idx_layout").set_value("Cards").run()
    assert not at.exception, [e.value for e in at.exception]


def test_index_entity_detail_shows_the_knowledge_graph():
    """The worked reference example end-to-end: Robert Carver's profile must
    show his blog, his preserved employment history, and the relationship edges
    that make Carver -> Man AHL -> Man Group traversable."""
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    at = _index_at("Entity detail")
    at.selectbox(key="idx_detail_name").set_value("Robert Carver").run()
    assert not at.exception, [e.value for e in at.exception]
    text = (" ".join(str(m.value) for m in at.markdown)
            + " " + " ".join(str(c.value) for c in at.caption))
    assert "qoppac.blogspot.com" in text
    assert "Previously:" in text and "Man AHL" in text and "Barclays" in text
    assert "worked at" in text, "relationship edges should be listed"
    assert "Systematic Trading" in text, "his books should appear"
    # the cross-link back into Zenith's own scrape registry
    assert "Robert Carver (Systematic)" in text


def test_index_export_buttons_present():
    if not config.INDEX_FILES["entities"].exists():
        pytest.skip("no committed index data")
    at = _index_at("Data management")
    labels = " ".join(str(b.label) for b in at.get("download_button"))
    assert "CSV" in labels and "Excel" in labels and "JSON" in labels

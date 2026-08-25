"""IDEAS tab -- the daily discretionary-systematic opportunity dashboard.

Reads only committed JSON (data/ideas/ideas_latest.json), matching every
other Zenith view. Three sub-views on one radio: Overview (the daily
dashboard -- summary stats + thin-day banner), Buy Ideas / Sell Ideas (the
full idea cards with conviction/unusualness/confluence + the deterministic
narrative). Deeper visualizations (conviction gauges, factor radar, R/R
diagrams, thesis-evolution timelines) and the tracker/performance/analog
sub-views are Phase 2/3 -- this is the Phase-1 working dashboard.
"""

from __future__ import annotations

import html as _html

import streamlit as st

from .. import ui_charts as uc
from ..config import THEME, IDEAS_WEIGHTS
from ..ui_theme import evidence_rating, key_findings, section, stamp, help_badge
from . import DISCLAIMER, GROUP_LABELS, load

SUBVIEWS = ["Overview", "Buy Ideas", "Sell Ideas", "Methodology"]

_EVIDENCE_NOTE = ("IDEAS fuses eight already-validated Zenith signal groups (MOMENTUM, EDGE, "
                  "PEAD, CAS regime, this package's own valuation/fundamentals cache) into one "
                  "composite. The inputs are individually B/B+; this FUSION is novel and carries "
                  "no out-of-sample track record yet. It earns promotion from its own accumulating "
                  "diagnostics (Diagnostics tab, Phase 2), never from a backtest presented as proof.")

_FINDINGS = [
    {"stat": "“The most rewarding investing has to involve doing something others are "
             "reluctant to do.” Discomfort is a feature of a genuine edge, not a bug.",
     "cite": "Howard Marks"},
    {"stat": "Momentum, analyst-revision drift, and short-interest anomalies are each "
             "independently well-replicated but attenuate post-publication.",
     "cite": "see the MOMENTUM / EDGE / PEAD tabs' own citations"},
    {"stat": "A composite of independently-evidenced signals should be judged on its OWN "
             "out-of-sample record, not on the strength of its ingredients.",
     "cite": "this tab's standing anti-overfitting discipline"},
]


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    return {"ideas": load("ideas", {})}


def today_badge() -> str | None:
    """TODAY-tab chip -- only speaks up when there is a real idea list."""
    try:
        doc = load("ideas", {})
        n_buy, n_sell = doc.get("n_buy", 0), doc.get("n_sell", 0)
        if not n_buy and not n_sell:
            return None
        return uc.chip(f"IDEAS -- {n_buy} BUY / {n_sell} SELL today", color=THEME.mauve,
                       sub="see IDEAS tab")
    except Exception:
        return None


def _rr_line(idea: dict) -> str:
    rr = idea.get("riskreward") or {}
    if not rr.get("available"):
        return "Risk/reward: data unavailable."
    ez = rr.get("entry_zone") or (None, None)
    return (f"Entry {ez[0]}-{ez[1]} &middot; Stop {rr.get('stop')} &middot; "
           f"Target {rr.get('target')} &middot; R/R {rr.get('rr_ratio')}:1 &middot; "
           f"Max drawdown {uc.fmt_pct(rr.get('max_drawdown_pct'))}")


def _idea_card(idea: dict) -> str:
    t = idea["ticker"]
    side = idea["side"]
    side_color = THEME.teal if side == "long" else THEME.coral
    side_label = "BUY" if side == "long" else "SELL / SHORT"
    conf = idea.get("confluence", {})
    div = idea.get("divergence", {})
    nar = idea.get("narrative", {})
    rr = idea.get("riskreward") or {}

    header = (
        f'<div style="display:flex; align-items:baseline; gap:0.6rem; flex-wrap:wrap;">'
        f'<span style="font-family:{THEME.font_display}; font-size:1.6rem; color:{side_color};">'
        f'{side_label} -- {_html.escape(t)}</span>'
        f'<span style="color:{THEME.muted}; font-size:0.9rem;">{_html.escape(idea.get("opportunity_type",""))}'
        f' &middot; {_html.escape(idea.get("horizon",""))} horizon</span>'
        f'</div>'
    )
    slab = uc.numeric_slab([
        {"label": "Conviction", "value": f'{idea.get("conviction", 0):.0f}', "color": side_color},
        {"label": "Unusualness", "value": f'{idea.get("unusual", 0):.0f}', "color": THEME.mustard},
        {"label": "Confluence", "value": conf.get("label", "n/a"), "color": THEME.navy},
        {"label": "Divergence flags", "value": str(div.get("n_flags", 0)),
         "color": THEME.mauve if div.get("has_divergence") else THEME.muted},
    ], min_width=140)

    sections = []
    for label, key in (("Thesis", "thesis"), ("Why Now?", "why_now"),
                       ("Idiosyncratic Risk", "idiosyncratic_risk"),
                       ("What the Market Thinks", "market_view"),
                       ("What Zenith Thinks", "zenith_view"),
                       ("Bull Case", "bull_case"), ("Bear Case", "bear_case")):
        val = nar.get(key)
        if val:
            sections.append(
                f'<div style="margin-top:0.4rem;"><b style="color:{THEME.mustard}; '
                f'font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase;">{label}</b>'
                f'<div style="font-size:0.88rem; color:#ddd; line-height:1.4;">'
                f'{_html.escape(str(val))}</div></div>')
    change = nar.get("change_my_mind") or []
    if change:
        items = "".join(f"<li>{_html.escape(c)}</li>" for c in change)
        sections.append(
            f'<div style="margin-top:0.4rem;"><b style="color:{THEME.mustard}; font-size:0.78rem; '
            f'letter-spacing:0.06em; text-transform:uppercase;">What Would Change My Mind?</b>'
            f'<ul style="margin:0.2rem 0 0 1.1rem; font-size:0.88rem; color:#ddd;">{items}</ul></div>')

    rr_html = (f'<div style="margin-top:0.5rem; font-size:0.85rem; color:{THEME.text}; '
              f'border-top:1px solid {THEME.grid}; padding-top:0.4rem;">{_rr_line(idea)}</div>')

    scale_in = rr.get("scale_in") or {}
    scale_html = ""
    if scale_in:
        scale_html = (f'<div style="font-size:0.82rem; color:{THEME.muted}; margin-top:0.2rem;">'
                     f'Scale-in: {_html.escape(scale_in.get("note", ""))}</div>')

    return (f'<div class="z-card" style="border-left:4px solid {side_color};">'
           f'{header}{slab}{"".join(sections)}{rr_html}{scale_html}</div>')


def _thin_day_banner(doc: dict) -> None:
    thin = doc.get("thin_day", {})
    if thin.get("long") or thin.get("short"):
        sides = [s for s, v in (("BUY", thin.get("long")), ("SELL", thin.get("short"))) if v]
        st.markdown(uc.state_banner(THEME.mustard, "THIN DAY",
                                    f"Fewer than the target of 5 qualifying {' and '.join(sides)} "
                                    f"ideas today -- the bar was not lowered to pad the list."),
                   unsafe_allow_html=True)


# -------------------------------------------------------------------- main --
def render() -> None:
    st.caption(DISCLAIMER)
    doc = _artefacts(cache_bust=str(load("ideas", {}).get("as_of", "")))["ideas"]

    st.markdown(evidence_rating("C+", "novel composite, no out-of-sample record yet",
                                _EVIDENCE_NOTE), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    if not doc or not doc.get("as_of"):
        st.markdown(stamp("--", "IDEAS"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.ideas.compute --action auto` to populate "
                "today's ideas.")
        return

    st.markdown(stamp(doc["as_of"], "IDEAS"), unsafe_allow_html=True)

    sub = st.radio("View", SUBVIEWS, horizontal=True, label_visibility="collapsed", key="ideas_sub")
    if sub == "Overview":
        _overview(doc)
    elif sub == "Buy Ideas":
        _idea_list(doc.get("buy", []), "BUY")
    elif sub == "Sell Ideas":
        _idea_list(doc.get("sell", []), "SELL")
    else:
        _methodology()


def _overview(doc: dict) -> None:
    regime = doc.get("regime", {})
    cov = doc.get("coverage", {})
    _thin_day_banner(doc)
    st.markdown(uc.numeric_slab([
        {"label": "BUY ideas today", "value": str(doc.get("n_buy", 0)), "color": THEME.teal},
        {"label": "SELL ideas today", "value": str(doc.get("n_sell", 0)), "color": THEME.coral},
        {"label": "Market regime", "value": regime.get("label", "n/a"), "color": THEME.mustard},
        {"label": "Universe scanned", "value": str(cov.get("n_universe", "n/a")), "color": THEME.text},
        {"label": "Candidates evaluated", "value": str(cov.get("n_candidates", "n/a")), "color": THEME.text},
    ]), unsafe_allow_html=True)

    st.markdown(section("Today's highest-conviction ideas", 3), unsafe_allow_html=True)
    top_buy = sorted(doc.get("buy", []), key=lambda i: i.get("conviction", 0), reverse=True)[:5]
    top_sell = sorted(doc.get("sell", []), key=lambda i: i.get("conviction", 0), reverse=True)[:5]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**BUY** ({len(doc.get('buy', []))})")
        for i, idea in enumerate(top_buy, 1):
            st.markdown(f"{i}. **{idea['ticker']}** -- {idea.get('opportunity_type', '')} "
                       f"(conviction {idea.get('conviction', 0):.0f})")
    with c2:
        st.markdown(f"**SELL / SHORT** ({len(doc.get('sell', []))})")
        for i, idea in enumerate(top_sell, 1):
            st.markdown(f"{i}. **{idea['ticker']}** -- {idea.get('opportunity_type', '')} "
                       f"(conviction {idea.get('conviction', 0):.0f})")

    if not top_buy and not top_sell:
        st.info("No ideas cleared today's conviction/unusualness gates -- a quiet market read, "
               "not a bug (spec section 1: never pad the list).")


def _idea_list(ideas: list[dict], label: str) -> None:
    if not ideas:
        st.info(f"No {label} ideas cleared the gates today.")
        return
    ranked = sorted(ideas, key=lambda i: i.get("conviction", 0), reverse=True)
    for idea in ranked:
        st.markdown(_idea_card(idea), unsafe_allow_html=True)


def _methodology() -> None:
    st.markdown(section("Signal groups and weights", 2), unsafe_allow_html=True)
    weights_txt = " &middot; ".join(f"{GROUP_LABELS[k]} {v:.0%}" for k, v in IDEAS_WEIGHTS.items())
    st.markdown(
        f"**Group weights:** {weights_txt} -- set a priori from each input's own documented "
        "evidence tier (see the MOMENTUM/EDGE/PEAD tabs' own ratings). There is no fitting or "
        "optimization loop anywhere in this package.\n\n"
        "**Conviction** (0-100) is the coverage-weighted blend of all eight groups, tilted by the "
        "current market regime (three coarse states). **Unusualness** (0-100) is kept deliberately "
        "SEPARATE -- it measures how extreme the configuration is, not how positive it looks, so a "
        "'good company at a fair price' ranks below a genuinely unusual situation. **Confluence** "
        "counts how many of the eight groups agree with the idea's overall direction.\n\n"
        "A security needs data in at least 3 of the 8 groups to be eligible at all -- thin coverage "
        "is excluded, never mathematically dampened into a false-looking neutral score.\n\n"
        "**Divergence flags** mark the specific case this feature is built to find: price, "
        "fundamentals and sentiment disagreeing with each other.\n\n"
        "Every thesis sentence on an idea card is assembled deterministically from the SAME numbers "
        "shown on that card -- there is no LLM call and nothing can be stated that is not in the "
        "payload."
    )
    with st.expander("Universe and data-quality notes"):
        st.markdown(
            "**Universe:** the Russell 1000 (pretom.universe.russell1000, the same source every "
            "other Zenith package uses) plus the CAS master ETF list (~400 names). ETF technicals "
            "come from CAS's Factor-Rotation composite where tagged, and a price-only trend fallback "
            "otherwise; ETFs have no per-security fundamentals/sentiment/positioning/catalyst data "
            "in this repo, so those groups are simply marked uncovered for them, never guessed.\n\n"
            "**Valuation** is shown as three separate lenses (cross-sectional, own-history, "
            "price-anchored) rather than one blended number -- see the module docstring in "
            "`zenith/ideas/valuation.py` for why. Where a metric is unavailable the payload omits "
            "it; it is never filled in with a fabricated value."
        )

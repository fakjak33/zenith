"""REGIMES tab — macro regime intelligence & early-warning system.

Reads only committed JSON (data/regimes/*) — no network calls from the view,
same convention every Zenith package follows. Four sub-views on one radio:
Overview (headline regime + confidence + momentum) -> Growth & Inflation
(the axis-by-axis explainability every card needs, spec section 43) ->
Dimensions (the six secondary regimes running alongside the quadrant) ->
History (the reconstructed timeline + transition segments, spec section 4).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import DISCLAIMER, DIMENSIONS, DIMENSION_LABELS, load

SUBVIEWS = ["Overview", "Growth & Inflation", "Dimensions", "History"]

_EVIDENCE_NOTE = ("Growth/inflation quadrant frameworks are well-established institutionally "
                  "(S&P's published regime research, business-cycle models used across macro "
                  "asset allocation), but THIS implementation is novel with zero out-of-sample "
                  "record. It earns promotion from its own calibration-vs-NBER tracking "
                  "(coming in Phase 2) or not at all.")

_FINDINGS = [
    {"stat": "Regime classification requires many indicators per axis, not one series.",
     "cite": "S&P regime research methodology"},
    {"stat": "A quadrant call should persist before being declared — one noisy release "
             "should not flip the headline.",
     "cite": "spec section 44"},
    {"stat": "NBER has dated US recessions back to 1854 — an external label this engine "
             "can be calibrated against.",
     "cite": "NBER Business Cycle Dating Committee"},
]


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    return {"current": load("current", {}), "dimensions": load("dimensions", {}),
           "timeline": load("timeline", {}), "status": load("status", {})}


def today_badge() -> str | None:
    try:
        cur = load("current", {})
        regime = cur.get("regime")
        if not regime:
            return None
        conf = cur.get("confidence")
        color = THEME.mustard if cur.get("transitioning") else THEME.teal
        sub = f"confidence {conf:.0f}" if conf is not None else "see REGIMES tab"
        return uc.chip(f"REGIMES — {regime}", color=color, sub=sub)
    except Exception:
        return None


def _regime_color(regime: str | None) -> str:
    return {"Goldilocks / Reflation": THEME.teal, "Overheating": THEME.orange,
           "Stagflation": THEME.coral, "Deflation / Slowdown": THEME.navy}.get(regime, THEME.muted)


def render() -> None:
    st.caption(DISCLAIMER)
    status = load("status", {})
    art = _artefacts(cache_bust=str(status.get("date", "")))
    current = art["current"]

    st.markdown(evidence_rating("C+", "novel composite, zero out-of-sample record yet", _EVIDENCE_NOTE),
               unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    if not current or not current.get("regime"):
        st.markdown(stamp("—", "REGIMES"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.regimes.compute --action auto` to populate "
               "the macro panel and classify the current regime.")
        return

    st.markdown(stamp(current.get("as_of", "—"), "REGIMES"), unsafe_allow_html=True)

    with st.expander("Methodology — how the regime is classified"):
        st.markdown(
            "**Two axes, many indicators each.** Growth and inflation are each scored from "
            "10-12 FRED series; a series counts as 'rising' if its own direction-adjusted "
            "z-score is higher than it was 3 months ago. The axis is 'rising' if a MAJORITY "
            "of its covered indicators are rising (breadth), not from the level of any single "
            "series.\n\n"
            "**Persistence.** A quadrant only becomes the DECLARED regime after holding for "
            "2 consecutive monthly readings — short of that, the prior declared regime is "
            "shown with a 'transition underway' flag rather than a flip nobody would trust.\n\n"
            "**Point-in-time.** Every series is shifted by its own real publication lag before "
            "being used, so no historical month's classification uses data unpublished at "
            "that time (spec section 42). This removes TIMING look-ahead but not REVISION "
            "look-ahead — FRED revises payrolls/GDP for years after first print — which a "
            "separate, optional local audit (not yet run) is designed to measure.\n\n"
            "**Six secondary dimensions** (monetary, liquidity, credit, financial conditions, "
            "dollar, volatility) are scored independently on the Dimensions tab and never "
            "folded into the quadrant call.")

    sub = st.radio("View", SUBVIEWS, horizontal=True, label_visibility="collapsed", key="regimes_sub")
    if sub == "Overview":
        _overview(current)
    elif sub == "Growth & Inflation":
        _axes(current)
    elif sub == "Dimensions":
        _dimensions(art["dimensions"])
    else:
        _history(art["timeline"])


# -------------------------------------------------------------------- overview
def _overview(current: dict) -> None:
    regime = current.get("regime")
    conf = current.get("confidence")
    mtm = current.get("momentum", {})
    color = _regime_color(regime)

    if current.get("transitioning"):
        st.markdown(uc.state_banner(THEME.mustard, "TRANSITION UNDERWAY",
                                    f"Raw signal points to {current.get('raw_regime')}, but it "
                                    f"has not yet persisted long enough to become the declared "
                                    f"regime (streak: {current.get('streak_months')} month(s))."),
                   unsafe_allow_html=True)
    else:
        st.markdown(uc.state_banner(color, "CURRENT REGIME", regime or "—"), unsafe_allow_html=True)

    st.markdown(uc.numeric_slab([
        {"label": "Regime", "value": regime or "—", "color": color},
        {"label": "Confidence", "value": f"{conf:.0f}/100" if conf is not None else "—",
         "color": THEME.text},
        {"label": "Regime Momentum", "value": f"{mtm.get('score'):+.0f}" if mtm.get("score") is not None else "—",
         "color": THEME.teal if (mtm.get("score") or 0) >= 0 else THEME.coral},
        {"label": "Persisted", "value": f"{current.get('streak_months', 0)} mo", "color": THEME.muted},
        {"label": "Last calculated", "value": current.get("as_of", "—"), "color": THEME.muted},
    ]), unsafe_allow_html=True)

    if mtm.get("narrative"):
        st.markdown(section("Regime momentum", 1), unsafe_allow_html=True)
        st.markdown(mtm["narrative"])

    st.markdown(section("Growth vs Inflation — the two axes", 2), unsafe_allow_html=True)
    g, i = current.get("growth", {}), current.get("inflation", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Growth", "Rising" if g.get("rising") else ("Falling" if g.get("rising") is False else "—"),
                  help="Majority of covered growth indicators rising over the trailing 3 months.")
        if g.get("n_total"):
            st.caption(f"{g.get('n_rising')} of {g.get('n_total')} indicators rising — "
                      f"breadth {g.get('breadth'):.0%}" if g.get("breadth") is not None else "")
    with c2:
        st.metric("Inflation", "Rising" if i.get("rising") else ("Falling" if i.get("rising") is False else "—"),
                  help="Majority of covered inflation indicators rising over the trailing 3 months.")
        if i.get("n_total"):
            st.caption(f"{i.get('n_rising')} of {i.get('n_total')} indicators rising — "
                      f"breadth {i.get('breadth'):.0%}" if i.get("breadth") is not None else "")


# ---------------------------------------------------------------- axes detail
def _axis_table(axis: dict) -> pd.DataFrame:
    rows = axis.get("indicators", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Rising"] = df["rising"].map({True: "↑", False: "↓", None: "—"})
    return pd.DataFrame({
        "Indicator": df["label"], "Latest value": df["raw_value"], "Z-score": df["z"],
        "Rising (3mo)": df["Rising"], "Publication lag (d)": df["lag_days"], "Freq": df["freq"],
    })


def _axes(current: dict) -> None:
    pick = st.radio("Axis", ["Growth", "Inflation"], horizontal=True,
                    label_visibility="collapsed", key="regimes_axis_pick")
    axis = current.get("growth", {}) if pick == "Growth" else current.get("inflation", {})
    st.caption(f"{axis.get('n_rising', '—')} of {axis.get('n_total', '—')} indicators rising over "
              f"the trailing 3 months — the SAME test used for every other explainability card "
              f"in this package.")
    df = _axis_table(axis)
    if df.empty:
        st.info("No indicator data yet.")
        return
    try:
        sty = df.style.map(lambda v: uc.grad_diverging(v, 2.0), subset=["Z-score"])
        st.dataframe(sty, use_container_width=True, hide_index=True, height=min(560, 45 + 34 * len(df)))
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ dimensions
def _dimensions(doc: dict) -> None:
    dims = doc.get("dimensions", {})
    if not dims:
        st.info("No dimension data yet.")
        return
    pick = st.radio("Dimension", [d for d in DIMENSIONS if d in dims], horizontal=True,
                    label_visibility="collapsed", format_func=lambda d: DIMENSION_LABELS[d],
                    key="regimes_dim_pick")
    d = dims[pick]
    st.markdown(uc.numeric_slab([
        {"label": "State", "value": d.get("state", "—"), "color": THEME.text},
        {"label": "Composite Z", "value": f"{d.get('composite_z'):+.2f}" if d.get("composite_z") is not None else "—",
         "color": THEME.teal if (d.get("composite_z") or 0) >= 0 else THEME.coral},
        {"label": "Coverage", "value": f"{d.get('coverage_n', 0)} indicators", "color": THEME.muted},
    ]), unsafe_allow_html=True)
    rows = d.get("indicators", [])
    if not rows:
        st.info("No indicator data yet.")
        return
    df = pd.DataFrame(rows)
    df["Rising"] = df["rising"].map({True: "↑", False: "↓", None: "—"})
    disp = pd.DataFrame({"Indicator": df["label"], "Latest value": df["raw_value"],
                        "Z-score": df["z"], "Rising (3mo)": df["Rising"],
                        "Publication lag (d)": df["lag_days"], "Freq": df["freq"]})
    try:
        sty = disp.style.map(lambda v: uc.grad_diverging(v, 2.0), subset=["Z-score"])
        st.dataframe(sty, use_container_width=True, hide_index=True, height=min(560, 45 + 34 * len(disp)))
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------- history
def _history(doc: dict) -> None:
    segs = doc.get("segments", [])
    trans = doc.get("transitions", [])
    months = doc.get("months", [])
    if not segs:
        st.info("No historical timeline yet.")
        return

    st.markdown(section("Regime timeline — segments", 3), unsafe_allow_html=True)
    seg_df = pd.DataFrame(segs)
    st.dataframe(seg_df.rename(columns={"regime": "Regime", "start": "Start", "end": "End",
                                        "n_months": "Months"}),
                use_container_width=True, hide_index=True)

    if trans:
        st.markdown(section("Transitions — Regime → New Regime", 5), unsafe_allow_html=True)
        for t in trans:
            st.markdown(f"**{t['from_regime']}** (through {t['from_end']}) → "
                       f"**{t['to_regime']}** (from {t['to_start']})")

    if months:
        st.markdown(section("Confidence over time", 1), unsafe_allow_html=True)
        mdf = pd.DataFrame(months)
        mdf["month"] = pd.to_datetime(mdf["month"])

        def build(alt):
            return (alt.Chart(mdf).mark_line(point=False).encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("confidence:Q", title="Confidence", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("declared_regime:N", title="Regime", legend=alt.Legend(labelLimit=0)),
                tooltip=["month:T", "declared_regime:N", "confidence:Q"],
            ).properties(height=280))

        uc.render_chart(build, fallback=mdf[["month", "confidence", "declared_regime"]])

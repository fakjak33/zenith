"""REGIMES tab — macro regime intelligence & early-warning system.

Reads only committed JSON (data/regimes/*) — no network calls from the view,
same convention every Zenith package follows. Twelve sub-views on one radio:
Overview (headline regime + confidence + momentum) -> Growth & Inflation
(the axis-by-axis explainability every card needs, spec section 43) ->
Dimensions (the six secondary regimes running alongside the quadrant) ->
Transitions (empirical base-rate probabilities, spec section 8) -> What's
Changing (per-indicator momentum deltas + the Regime Change Score, spec
sections 6/27) -> Performance (asset-class/factor stats by historical
regime, spec sections 10/11/23) -> Analogs (nearest historical months, spec
section 33) -> Accuracy (calibration vs NBER, spec section 32) -> Themes
(Dollar/Fiscal/Yen/AI/Crypto/Geopolitical, spec sections 13-19) -> Scenarios
("What If?" contingency planning, spec section 25) -> Alerts (spec section
26) -> History (the reconstructed timeline + transition segments, spec
section 4).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import DISCLAIMER, DIMENSIONS, DIMENSION_LABELS, load

SUBVIEWS = ["Overview", "Growth & Inflation", "Dimensions", "Transitions", "What's Changing",
           "Performance", "Analogs", "Accuracy", "Themes", "Scenarios", "Alerts", "History"]

REGIME_LABELS_ORDER = ("Goldilocks / Reflation", "Overheating", "Stagflation", "Deflation / Slowdown")

_EVIDENCE_NOTE = ("Growth/inflation quadrant frameworks are well-established institutionally "
                  "(S&P's published regime research, business-cycle models used across macro "
                  "asset allocation), but THIS implementation is novel with limited out-of-sample "
                  "record. It earns further promotion from its own accumulating calibration-vs-"
                  "NBER tracking (see the Accuracy tab) or not at all.")

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
           "timeline": load("timeline", {}), "status": load("status", {}),
           "transitions": load("transitions", {}), "changes": load("changes", {}),
           "crossasset": load("crossasset", {}), "performance": load("performance", {}),
           "analogs": load("analogs", {}), "accuracy": load("accuracy", {}),
           "themes": load("themes", {}), "scenarios": load("scenarios", {}),
           "alerts": load("alerts", {})}


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
            "folded into the quadrant call.\n\n"
            "**Transition probabilities are EMPIRICAL BASE RATES**, read straight off the "
            "reconstructed timeline (never a fitted model) — every cell shows the count `n` "
            "backing it, and a cell with too few observations says so instead of guessing.\n\n"
            "**Theme signal scores are NOT probabilities.** Only the growth/inflation quadrant "
            "has the historical-timeline infrastructure to back a genuine empirical probability; "
            "themes (Dollar/Fiscal/Yen/AI/Crypto/Geopolitical) get an honestly-named 0-100 "
            "signal score for series-backed themes, or an evidence board (Fact/Interpretation/"
            "Forecast/Speculation, mined from this app's own research archive) for policy-driven "
            "ones — never an invented probability with no series behind it.")

    sub = st.radio("View", SUBVIEWS, horizontal=True, label_visibility="collapsed", key="regimes_sub")
    if sub == "Overview":
        _overview(current, art["alerts"])
    elif sub == "Growth & Inflation":
        _axes(current)
    elif sub == "Dimensions":
        _dimensions(art["dimensions"])
    elif sub == "Transitions":
        _transitions(art["transitions"])
    elif sub == "What's Changing":
        _changes(art["changes"], art["crossasset"])
    elif sub == "Performance":
        _performance(art["performance"])
    elif sub == "Analogs":
        _analogs(art["analogs"])
    elif sub == "Accuracy":
        _accuracy(art["accuracy"])
    elif sub == "Themes":
        _themes(art["themes"])
    elif sub == "Scenarios":
        _scenarios(art["scenarios"])
    elif sub == "Alerts":
        _alerts(art["alerts"])
    else:
        _history(art["timeline"])


# -------------------------------------------------------------------- overview
def _overview(current: dict, alerts_doc: dict) -> None:
    regime = current.get("regime")
    conf = current.get("confidence")
    mtm = current.get("momentum", {})
    color = _regime_color(regime)

    triggered = alerts_doc.get("alerts", [])
    if triggered:
        n_high = sum(1 for a in triggered if a.get("severity") == "high")
        st.markdown(uc.state_banner(THEME.coral if n_high else THEME.mustard, "ACTIVE ALERTS",
                                    f"{len(triggered)} alert(s) triggered — see the Alerts tab"),
                   unsafe_allow_html=True)

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


# ----------------------------------------------------------------- transitions
def _transitions(doc: dict) -> None:
    cur = doc.get("current", {})
    if not cur.get("available"):
        st.info("No transition data yet.")
        return
    st.caption("Empirical base rates read off the reconstructed timeline — every cell shows "
              "`n`, the count of historical months backing it. No fitting, no forecast, just "
              "counted frequency.")
    st.markdown(section(f"From {cur['regime']} — where has history gone next?", 2), unsafe_allow_html=True)

    def _table(table: dict, label: str) -> pd.DataFrame:
        rows = []
        for h_days, cell in table.items():
            row = {"Horizon": f"{h_days}d", "n (start)": cell["n_start"]}
            for regime, d in cell["destinations"].items():
                row[regime] = "—" if d["p"] is None else f"{d['p']:.0%} (n={d['n']})"
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.set_index("Horizon") if not df.empty else df

    st.markdown("**Unconditional**")
    df = _table(cur["unconditional"], "unconditional")
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("No data.")

    mc = cur.get("momentum_conditioned")
    if mc:
        bucket = (cur.get("momentum_bucket") or "").replace("_", " ")
        st.markdown(f"**Conditioned on today's momentum ({bucket})**")
        df2 = _table(mc, "conditioned")
        st.dataframe(df2, use_container_width=True) if not df2.empty else st.info("No data.")
        st.caption("A single improving/deteriorating split, not full terciles — smaller "
                  "conditioned cells are exactly why `n` is always shown.")


# --------------------------------------------------------------- what's changing
def _changes(doc: dict, cross_doc: dict) -> None:
    score = doc.get("regime_change_score", {})
    s = score.get("score")
    if s is not None:
        band = score.get("band", "—")
        color = {"Stable": THEME.teal, "Early Signals": THEME.mustard, "Emerging": THEME.orange,
                "Significant Transition": THEME.coral, "Major Regime Shift": THEME.coral}.get(band, THEME.muted)
        st.markdown(uc.state_banner(color, "REGIME CHANGE SCORE", f"{s:.0f}/100 — {band}"),
                   unsafe_allow_html=True)
        st.markdown(uc.numeric_slab([
            {"label": "Score", "value": f"{s:.0f}/100", "color": color},
            {"label": "Indicators flipped", "value": str(score.get("n_flipped", 0)), "color": THEME.text},
            {"label": "Indicators covered", "value": str(score.get("n_covered", 0)), "color": THEME.muted},
            {"label": "Avg flip magnitude", "value": f"{score.get('avg_flip_magnitude', 0):.2f}σ",
             "color": THEME.muted},
        ]), unsafe_allow_html=True)
        st.caption("Breadth AND magnitude across ALL 8 dimensions, not a threshold on any single "
                  "indicator — spec section 27's own warning: 'avoid triggering a major regime "
                  "alert because of one noisy indicator.'")
    else:
        st.info("No change-score data yet.")

    rows = doc.get("indicators", [])
    if rows:
        st.markdown(section("What is changing — per-indicator momentum", 4), unsafe_allow_html=True)
        df = pd.DataFrame(rows)
        df = df.reindex(df["delta_3m"].abs().sort_values(ascending=False).index)
        disp = pd.DataFrame({"Indicator": df["label"], "Dimension": df["dimension"],
                            "Z-score": df["z"], "Δ 1mo": df["delta_1m"], "Δ 3mo": df["delta_3m"]})
        try:
            sty = disp.style.map(lambda v: uc.grad_diverging(v, 1.5), subset=["Δ 1mo", "Δ 3mo"])
            st.dataframe(sty, use_container_width=True, hide_index=True, height=min(560, 45 + 34 * len(disp)))
        except Exception:
            st.dataframe(disp, use_container_width=True, hide_index=True)

    conf = cross_doc.get("confirmation", {})
    if conf.get("n_total"):
        st.markdown(section("Cross-asset confirmation", 0), unsafe_allow_html=True)
        st.caption(f"{conf['n_confirming']} of {conf['n_total']} independent market checks agree "
                  f"with the classifier's growth/inflation read.")
        for c in conf.get("checks", []):
            icon = "✓" if c["confirms"] else "✗"
            st.markdown(f"{icon} **{c['check']}** — expects {c['expects']}, observed {c['observed']}")

    div = cross_doc.get("divergences", [])
    if div:
        st.markdown(section("Divergences to investigate", 5), unsafe_allow_html=True)
        st.markdown(uc.note_strip("Not automatically bullish or bearish — investigate",
                              [f"{d['flag']}: classifier expects {d['classifier_expects']}, "
                               f"market shows {d['market_shows']}" for d in div]),
                   unsafe_allow_html=True)


# ------------------------------------------------------------------- performance
def _performance(doc: dict) -> None:
    universe = st.radio("Universe", ["Asset Classes", "Factors"], horizontal=True,
                        label_visibility="collapsed", key="regimes_perf_universe")
    table = doc.get("asset", {}) if universe == "Asset Classes" else doc.get("factor", {})
    if not table:
        st.info("No performance data yet.")
        return
    regime = st.radio("Regime", list(dict.fromkeys(REGIME_LABELS_ORDER)), horizontal=True,
                      label_visibility="collapsed", key="regimes_perf_regime")
    rows = []
    for ticker, data in table.items():
        stats = data.get("by_regime", {}).get(regime)
        if stats:
            rows.append({"Ticker": ticker, "Label": data["label"], **stats})
    if not rows:
        st.info(f"No sufficient history for {regime} yet.")
        return
    df = pd.DataFrame(rows).sort_values("avg_return", ascending=False)
    disp = pd.DataFrame({
        "Ticker": df["Ticker"], "Label": df["Label"], "Months (n)": df["n_months"],
        "Avg Return": df["avg_return"], "Median Return": df["median_return"],
        "Win Rate": df["win_rate"], "Vol (ann.)": df["volatility_ann"],
        "Max Drawdown": df["max_drawdown"], "Sharpe-like": df["sharpe_like"],
    })
    try:
        sty = (disp.style.map(lambda v: uc.grad_diverging(v, 0.03), subset=["Avg Return", "Median Return"])
               .format({"Avg Return": lambda v: uc.fmt_pct(v, 2), "Median Return": lambda v: uc.fmt_pct(v, 2),
                       "Win Rate": lambda v: uc.fmt_pct(v, 0, signed=False),
                       "Vol (ann.)": lambda v: uc.fmt_pct(v, 0, signed=False),
                       "Max Drawdown": lambda v: uc.fmt_pct(v, 1, signed=False)}))
        st.dataframe(sty, use_container_width=True, hide_index=True, height=min(600, 45 + 34 * len(disp)))
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption("Regime tagging is RETROSPECTIVE (uses the 2-month persistence rule, filled "
              "forward before persistence cleared) — this is standard regime-conditional "
              "analysis methodology, not a claim every month shown was tradable in real time "
              "at its start.")


# ----------------------------------------------------------------------- analogs
def _analogs(doc: dict) -> None:
    rows = doc.get("analogs", [])
    if not rows:
        st.info(doc.get("note") or "No analog data yet.")
        return
    st.caption(doc.get("note", ""))
    df = pd.DataFrame(rows)
    disp = pd.DataFrame({"Month": df["month"], "Distance": df["distance"],
                        "Shared dimensions": df["n_shared_dimensions"],
                        "Fwd 3mo SPY": df["forward"].apply(lambda f: f.get("3")),
                        "Fwd 6mo SPY": df["forward"].apply(lambda f: f.get("6")),
                        "Fwd 12mo SPY": df["forward"].apply(lambda f: f.get("12"))})
    st.dataframe(disp, use_container_width=True, hide_index=True)

    dists = doc.get("distributions", {})
    if dists:
        st.markdown(section("Forward outcome distribution — context, not a forecast", 3),
                   unsafe_allow_html=True)
        cols = st.columns(len(dists))
        for col, (h, d) in zip(cols, dists.items()):
            with col:
                if d.get("n"):
                    st.markdown(uc.numeric_slab([
                        {"label": f"{h}mo median (n={d['n']})", "value": uc.fmt_pct(d["median"]),
                         "color": THEME.teal if d["median"] >= 0 else THEME.coral},
                    ]), unsafe_allow_html=True)
                    st.caption(f"IQR {uc.fmt_pct(d['q25'])} to {uc.fmt_pct(d['q75'])} · "
                              f"win rate {d['win_rate']:.0%} · best {uc.fmt_pct(d['best'])} · "
                              f"worst {uc.fmt_pct(d['worst'])}")
                else:
                    st.caption(f"{h}mo: no data")


# ---------------------------------------------------------------------- accuracy
def _accuracy(doc: dict) -> None:
    if not doc.get("available"):
        st.info(doc.get("reason") or "No accuracy data yet.")
        return
    ll = doc.get("lead_lag", {})
    st.markdown(section("Calibration vs NBER recession dating", 2), unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "NBER recessions", "value": str(ll.get("n_nber_recessions", 0)), "color": THEME.text},
        {"label": "Matched", "value": str(ll.get("n_matched", 0)), "color": THEME.teal},
        {"label": "Missed (false negatives)", "value": str(ll.get("n_false_negatives", 0)), "color": THEME.coral},
        {"label": "Our signal episodes", "value": str(ll.get("n_our_signal_episodes", 0)), "color": THEME.text},
        {"label": "False positives", "value": str(ll.get("n_false_positives", 0)), "color": THEME.coral},
        {"label": "Avg lead (months)", "value": (f"{ll.get('avg_lead_months'):+.1f}"
                                                 if ll.get("avg_lead_months") is not None else "—"),
         "color": THEME.muted},
    ]), unsafe_allow_html=True)
    st.caption("Our 'recession signal' = declared regime is Deflation/Slowdown — a reasonable "
              "proxy, NOT identical to NBER's own multi-indicator definition. Read this as "
              "'does the engine's own regime call track recessions', not NBER-equivalence.")
    matches = ll.get("matches", [])
    if matches:
        st.dataframe(pd.DataFrame(matches).rename(columns={
            "nber_start": "NBER start", "our_first_signal": "Our first signal",
            "lead_months": "Lead (months, + = we led)"}), use_container_width=True, hide_index=True)

    brier = doc.get("brier", {})
    if brier.get("brier") is not None:
        st.markdown(section("Brier score (IN-SAMPLE)", 5), unsafe_allow_html=True)
        st.markdown(uc.numeric_slab([
            {"label": f"{brier.get('horizon_months')}mo horizon (n={brier.get('n')})",
             "value": f"{brier['brier']:.3f}",
             "color": THEME.teal if brier["brier"] < 0.25 else THEME.coral},
        ]), unsafe_allow_html=True)
        st.caption(brier.get("note", ""))


# ------------------------------------------------------------------------ themes
def _themes(doc: dict) -> None:
    themes = doc.get("themes", {})
    if not themes:
        st.info("No theme data yet.")
        return
    pick = st.radio("Theme", ["Dollar", "Fiscal / Treasury", "Yen / FX", "AI Investment",
                              "Crypto Regulatory", "Geopolitical"], horizontal=True,
                    label_visibility="collapsed", key="regimes_theme_pick")
    key = {"Dollar": "dollar", "Fiscal / Treasury": "fiscal", "Yen / FX": "yen",
          "AI Investment": "ai", "Crypto Regulatory": "crypto", "Geopolitical": "geopolitical"}[pick]
    t = themes.get(key, {})

    if "signal_score" in t:
        score = t["signal_score"]
        st.markdown(uc.numeric_slab([
            {"label": "Signal Score (not a probability)", "value": f"{score:.0f}/100" if score is not None else "—",
             "color": THEME.teal if (score or 50) >= 50 else THEME.coral},
        ]), unsafe_allow_html=True)
    if "intervention_risk" in t:
        tier = t["intervention_risk"]
        color = {"Extreme": THEME.coral, "High": THEME.orange, "Moderate": THEME.mustard,
                "Low": THEME.teal}.get(tier, THEME.muted)
        st.markdown(uc.state_banner(color, "INTERVENTION RISK", tier or "—"), unsafe_allow_html=True)

    if t.get("context"):
        st.markdown(section("Context", 1), unsafe_allow_html=True)
        st.markdown(t["context"])
    if t.get("note"):
        st.caption(t["note"])
    if t.get("framework_note"):
        st.markdown(uc.note_strip("How to read this evidence board", [t["framework_note"]]),
                   unsafe_allow_html=True)

    ev = t.get("evidence", [])
    if ev:
        st.markdown(section(f"Evidence board ({len(ev)})", 3), unsafe_allow_html=True)
        cat_colors = {"Fact": THEME.teal, "Interpretation": THEME.mustard,
                     "Forecast": THEME.orange, "Speculation": THEME.coral}
        for item in ev:
            c = cat_colors.get(item["category"], THEME.muted)
            st.markdown(f'{uc.chip(item["category"], color=c)} **[{item["title"]}]({item["link"]})** '
                       f'— {item["source"]}', unsafe_allow_html=True)


# --------------------------------------------------------------------- scenarios
def _scenarios(doc: dict) -> None:
    quad = doc.get("quadrant_scenarios", [])
    dim = doc.get("dimension_scenarios", [])
    if not quad and not dim:
        st.info("No scenario data yet.")
        return
    st.caption("Contingency planning, NOT predictions (spec section 25) — 'if this happens, "
              "here is the implied regime and what has historically worked in it.'")
    names = [s["name"] for s in quad] + [s["name"] for s in dim]
    pick = st.selectbox("Scenario", names, key="regimes_scenario_pick")
    scenario = next((s for s in quad + dim if s["name"] == pick), None)
    if not scenario:
        return
    st.markdown(f"**{scenario['description']}**" if scenario.get("description") else "")

    if scenario.get("grounded"):
        st.markdown(uc.state_banner(_regime_color(scenario["implied_regime"]), "IMPLIED REGIME",
                                    scenario["implied_regime"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Historical beneficiaries**")
            for b in scenario.get("historical_beneficiaries", []):
                st.markdown(f"- {b['label']} ({b['ticker']}) — {uc.fmt_pct(b['avg_return'])} avg/mo, "
                           f"win rate {b['win_rate']:.0%}")
        with c2:
            st.markdown("**Historical laggards**")
            for l in scenario.get("historical_losers", []):
                st.markdown(f"- {l['label']} ({l['ticker']}) — {uc.fmt_pct(l['avg_return'])} avg/mo, "
                           f"win rate {l['win_rate']:.0%}")
    else:
        st.info(scenario.get("caveat", "Qualitative scenario — no regime-conditional backtest."))
        tickers = scenario.get("relevant_tickers", {})
        if tickers:
            st.markdown("**Relevant tickers**")
            for tk, lbl in tickers.items():
                st.markdown(f"- {tk} — {lbl}")
        if scenario.get("note"):
            st.caption(scenario["note"])


# ------------------------------------------------------------------------ alerts
def _alerts(doc: dict) -> None:
    triggered = doc.get("alerts", [])
    if not triggered:
        st.markdown(uc.state_banner(THEME.teal, "NO ACTIVE ALERTS",
                                    "Nothing has crossed a threshold in the trailing 30 days."),
                   unsafe_allow_html=True)
        return
    sev_colors = {"high": THEME.coral, "medium": THEME.mustard, "low": THEME.muted}
    for a in triggered:
        st.markdown(uc.state_banner(sev_colors.get(a["severity"], THEME.muted),
                                    a["severity"].upper(), a["title"]), unsafe_allow_html=True)
        st.markdown(f"**What changed:** {a.get('what_changed', '—')}")
        st.markdown(f"**Why it matters:** {a.get('description', '—')}")
        st.markdown(f"**Watch next:** {a.get('watch_next', '—')}")
        st.divider()

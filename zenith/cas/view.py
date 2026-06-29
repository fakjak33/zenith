"""Streamlit rendering for the CAS tab. Kept out of app.py so the viewer stays
thin. Reuses the Zenith theme/section helpers; uses altair (ships with Streamlit)
for the overlap heatmap."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import store_cas, registry, contingency
from .universe import label_of
from ..ui_theme import section, help_badge
from . import DISCLAIMER
from .help_text import HELP

_STATE_COLOR = {"buy": "#2ec4b6", "neutral": "#b8b8b8", "sell": "#ff5a3c"}

SEGMENT_TABS = ["Overview", "Strategies", "Flows & positioning", "Themes",
                "Factor Rotation", "History & hit-rate", "Rebalance", "Contingency",
                "BCT consensus", "Models & notes"]

# short hover help per segment tab
_TAB_HELP = {
    "Overview": HELP["overlap"], "Strategies": "Per-asset technical/statistical strategy battery.",
    "Flows & positioning": HELP["flows"], "Themes": HELP["themes"],
    "Factor Rotation": HELP["frm"], "History & hit-rate": HELP["hitrate"],
    "Rebalance": HELP["rebalance"], "Contingency": "Pre-planned playbooks that arm on triggers.",
    "BCT consensus": HELP["consensus"], "Models & notes": "Tune model weights and log research.",
}


def _signals_df() -> pd.DataFrame:
    rows = store_cas.load("signals", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render() -> None:
    status = store_cas.load("status", {})
    st.caption(DISCLAIMER)
    if not status:
        st.info("No CAS data yet. Run `python -m zenith.cas.compute --cadence weekly` "
                "(or wait for the scheduled Action) to populate signals.")
        return
    regime = status.get("regime", "?")
    st.caption(f"As of {status.get('date','?')} · cadence {status.get('cadence','?')} · "
               f"{status.get('n_signals',0)} signals · {status.get('n_assets',0)} assets · "
               f"regime: **{regime}**")

    seg = st.radio("CAS segment", SEGMENT_TABS, horizontal=True, label_visibility="collapsed")
    if _TAB_HELP.get(seg):
        st.caption(f"ⓘ {_TAB_HELP[seg]}")
    df = _signals_df()

    if seg == "Overview":
        _overview(df)
    elif seg == "Strategies":
        _segment(df, "strategies", "Per-asset strategy signals (101-alpha + classics)",
                 help_key="signal")
    elif seg == "Flows & positioning":
        _flows(df)
    elif seg == "Themes":
        _segment(df, "themes", "Sector / theme relative-strength & breadth", help_key="themes")
    elif seg == "Factor Rotation":
        _factor_rotation(df)
    elif seg == "History & hit-rate":
        _history()
    elif seg == "Rebalance":
        _rebalance()
    elif seg == "Contingency":
        _contingency()
    elif seg == "BCT consensus":
        _consensus()
    elif seg == "Models & notes":
        _registry()


def _overview(df: pd.DataFrame) -> None:
    cons = store_cas.load("consensus", [])
    ovl = store_cas.load("overlap", {})
    ranked = ovl.get("ranked", [])

    st.markdown(section("Highest-conviction overlap (signals aligned across segments)", 0,
                        help=HELP["overlap"]), unsafe_allow_html=True)
    if ranked:
        top = pd.DataFrame(ranked).head(20)
        top["label"] = top["asset"].map(label_of)
        st.dataframe(top[["asset", "label", "asset_class", "net", "buy_count",
                          "sell_count", "n_segments", "mean_strength"]],
                     use_container_width=True, height=400)

    st.markdown(section("Overlap heatmap — assets × signal families", 2, help=HELP["signal"]),
                unsafe_allow_html=True)
    _heatmap(df)

    st.markdown(section("BCT consensus leaderboard", 4, help=HELP["consensus"]),
                unsafe_allow_html=True)
    if cons:
        cdf = pd.DataFrame(cons).head(20)
        cdf["label"] = cdf["asset"].map(label_of)
        st.dataframe(cdf[["asset", "label", "state", "composite", "behavioral",
                          "physical", "entanglement", "signal_variety", "entropy",
                          "confidence"]], use_container_width=True, height=400)


def _heatmap(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No signals.")
        return
    try:
        import altair as alt
    except Exception:
        st.dataframe(df.pivot_table(index="asset", columns="family", values="signal",
                                    aggfunc="first"), use_container_width=True)
        return
    top_assets = (df.groupby("asset")["signal"].apply(lambda s: s.abs().mean())
                  .sort_values(ascending=False).head(25).index.tolist())
    sub = df[df["asset"].isin(top_assets)]
    chart = (alt.Chart(sub).mark_rect().encode(
        x=alt.X("family:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("asset:N", title=None),
        color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                        legend=alt.Legend(title="signal")),
        tooltip=["asset", "family", "segment", "signal", "state"],
    ).properties(height=520))
    st.altair_chart(chart, use_container_width=True)


def _segment(df: pd.DataFrame, segment: str, title: str, help_key: str | None = None) -> None:
    st.markdown(section(title, 0, help=HELP.get(help_key) if help_key else None),
                unsafe_allow_html=True)
    if df.empty:
        st.caption("No signals.")
        return
    sub = df[df["segment"] == segment].copy()
    assets = sorted(sub["asset"].unique())
    pick = st.multiselect("Filter assets", assets, default=[])
    if pick:
        sub = sub[sub["asset"].isin(pick)]
    sub["label"] = sub["asset"].map(label_of)
    st.dataframe(sub[["asset", "label", "family", "state", "signal", "horizon",
                      "confidence", "rationale"]].sort_values("signal", ascending=False),
                 use_container_width=True, height=560)


def _flows(df: pd.DataFrame) -> None:
    _segment(df, "flows", "Flows & positioning (COT proxy + volume proxy)")
    st.markdown(section("Dynamics with no free feed (proxy / manual)", 2), unsafe_allow_html=True)
    from .signals.flows import MANUAL_DYNAMICS
    st.caption("These institutional dynamics are paywalled (SpotGamma, EPFR, etc.). "
               "They appear here as placeholders — wire a paid feed or hand-enter later:")
    st.write(", ".join(d.replace("_", " ") for d in MANUAL_DYNAMICS))


def _factor_rotation(df: pd.DataFrame) -> None:
    from .universe import frm_tag, REGION_LABEL
    st.markdown(section("Factor Rotation Momentum", 0, help=HELP["frm"]), unsafe_allow_html=True)
    st.caption("Style factors + industries + region-sectors traded via US-listed ETFs. "
               "Time-series momentum (each ETF on its own trend) + cross-sectional rotation "
               "(ranked within peer groups). Gupta & Kelly (2019) + Man Group style-trend.")

    sub = df[df["segment"] == "factor_rotation"].copy() if not df.empty else pd.DataFrame()
    if not sub.empty:
        tags = sub["asset"].map(lambda t: frm_tag(t) or {})
        sub["group"] = [d.get("group", "") for d in tags]
        sub["label"] = [d.get("label", "") for d in tags]
        sub["region"] = [REGION_LABEL.get(d.get("region"), d.get("region", "")) for d in tags]

        st.markdown(section("Style composite — factor × region", 2, help=HELP["frm_composite"]),
                    unsafe_allow_html=True)
        styles = sub[(sub["family"] == "frm_composite") & (sub["group"] == "style")]
        _frm_heatmap(styles)

        st.markdown(section("All factor-rotation signals", 4, help=HELP["frm_ts_mom"]),
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        groups = sorted(g for g in sub["group"].unique() if g)
        gpick = c1.multiselect("Filter group", groups, default=[], help=HELP["frm_group"])
        fpick = c2.multiselect("Filter family", sorted(sub["family"].unique()), default=[])
        view = sub
        if gpick:
            view = view[view["group"].isin(gpick)]
        if fpick:
            view = view[view["family"].isin(fpick)]
        st.dataframe(view[["asset", "group", "label", "region", "family", "state", "signal",
                           "confidence", "rationale"]].sort_values("signal", ascending=False),
                     use_container_width=True, height=440)
    else:
        st.caption("No factor-rotation signals yet. Run `python -m zenith.cas.compute` to populate.")

    _frm_backtest()


def _frm_heatmap(comp: pd.DataFrame) -> None:
    if comp.empty:
        st.caption("No composite style signals.")
        return
    try:
        import altair as alt
        chart = (alt.Chart(comp).mark_rect().encode(
            x=alt.X("region:N", title=None),
            y=alt.Y("label:N", title=None),
            color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                            legend=alt.Legend(title="composite")),
            tooltip=["asset", "label", "region", "signal", "state"],
        ).properties(height=320))
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(comp.pivot_table(index="label", columns="region", values="signal",
                                      aggfunc="first"), use_container_width=True)


def _frm_backtest() -> None:
    st.markdown(section("Academic backtest — Factor Momentum Everywhere (replication)", 0,
                        help=HELP["backtest"]), unsafe_allow_html=True)
    bt = store_cas.load("backtest", {})
    if not bt:
        st.caption("No backtest yet. Run `python -m zenith.cas.backtest.factor_momentum` "
                   "(or wait for the monthly Action).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Factors tested", bt.get("n_factors", 0), help=HELP["ar1"])
    c2.metric("Positive / Significant AR(1)",
              f"{bt.get('n_positive', 0)} / {bt.get('n_significant', 0)}", help=HELP["t_stat"])
    c3.metric("Combined TS Sharpe", bt.get("combined_ts_sharpe", "—"), help=HELP["ts_sharpe"])
    st.caption(f"As of {bt.get('as_of','?')} · {bt.get('source','')} · "
               "replicates the paper's autocorrelation + TS factor-momentum result on the "
               "publicly available academic factor set.")
    facts = bt.get("factors", {})
    if facts:
        rows = [{"factor": k, **v} for k, v in facts.items()]
        cols = ["factor", "region", "ar1", "t_stat", "positive", "significant",
                "ts_sharpe", "n_months"]
        st.dataframe(pd.DataFrame(rows)[cols].sort_values("ts_sharpe", ascending=False),
                     use_container_width=True, height=360)


def _history() -> None:
    from .universe import frm_tag, REGION_LABEL
    st.markdown(section("Signal history — how a signal has evolved", 0, help=HELP["history"]),
                unsafe_allow_html=True)
    hist = store_cas.load("history", [])
    if hist:
        hdf = pd.DataFrame(hist)
        assets = sorted(hdf["asset"].unique())
        default_ix = assets.index("MTUM") if "MTUM" in assets else 0
        pick = st.selectbox("Asset", assets, index=default_ix,
                            help="Pick a factor/industry ETF to see its signal trend.")
        one = hdf[hdf["asset"] == pick].copy()
        tag = frm_tag(pick) or {}
        st.caption(f"{tag.get('label', pick)} · {REGION_LABEL.get(tag.get('region'), '')} · "
                   f"families: {', '.join(sorted(one['family'].unique()))}")
        try:
            import altair as alt
            one["date"] = pd.to_datetime(one["date"])
            chart = (alt.Chart(one).mark_line(point=False).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("signal:Q", scale=alt.Scale(domain=[-1, 1]), title="signal"),
                color=alt.Color("family:N", legend=alt.Legend(title="family")),
                tooltip=["date:T", "family:N", "signal:Q"],
            ).properties(height=300))
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.line_chart(one.pivot_table(index="date", columns="family", values="signal"))
    else:
        st.caption("No signal history yet — it backfills on the next CAS run "
                   "(`python -m zenith.cas.compute`).")

    st.markdown(section("Predictive hit-rate — did the signal call the move?", 2, help=HELP["hitrate"]),
                unsafe_allow_html=True)
    hr = store_cas.load("hitrate", {})
    if not hr:
        st.caption("No hit-rate yet. Run the CAS compute to backfill it.")
        return
    st.caption(f"{hr.get('model','')} · {hr.get('n_assets',0)} assets · as of {hr.get('as_of','?')} · "
               "% of directional calls that matched the actual forward return (50% = coin-flip).")
    byh = hr.get("by_horizon", {})
    if byh:
        order = ["1m", "3m", "6m", "12m"]
        cols = st.columns(len(order))
        for i, h in enumerate(order):
            rec = byh.get(h, {})
            rate = rec.get("hit_rate")
            cols[i].metric(f"{h} hit-rate",
                           "—" if rate is None else f"{rate:.0%}",
                           help=f"n = {rec.get('n', 0)} observations")
    bg = hr.get("by_group", {})
    if bg:
        rows = []
        for grp, d in bg.items():
            row = {"group": grp}
            for h in ("1m", "3m", "6m", "12m"):
                r = d.get(h, {})
                row[h] = None if r.get("hit_rate") is None else round(r["hit_rate"], 3)
            rows.append(row)
        st.markdown(section("Hit-rate by group", 4), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)


def _rebalance() -> None:
    st.markdown(section("Upcoming rebalance & key dates", 0), unsafe_allow_html=True)
    events = store_cas.load("rebalance", [])
    if events:
        st.dataframe(pd.DataFrame(events)[["date", "days_until", "event", "kind", "note"]],
                     use_container_width=True, height=420)
    else:
        st.caption("No upcoming events loaded.")


def _contingency() -> None:
    st.markdown(section("Contingency playbook", 0), unsafe_allow_html=True)
    for s in contingency.load():
        flag = "🔴 ACTIVE" if s.get("active") else "⚪ standby"
        with st.expander(f"{flag} — {s['name']}  ·  {s.get('status','')}",
                         expanded=bool(s.get("active"))):
            st.markdown(f"**Trigger:** {s['trigger']}")
            st.markdown(f"**Proactive:** {s['proactive']}")
            st.markdown(f"**Reactive:** {s['reactive']}")


def _consensus() -> None:
    st.markdown(section("Behavioral Consensus Theory — ensemble view", 0), unsafe_allow_html=True)
    st.caption("BCT-inspired ensemble (honest interpretation, not literal physics): "
               "behavioral = positioning/mean-reversion extremes · physical = trend/regime/vol · "
               "entanglement = cross-horizon agreement. Variety/entropy = Ashby's-Law analogue.")
    cons = store_cas.load("consensus", [])
    if cons:
        st.dataframe(pd.DataFrame(cons), use_container_width=True, height=560)


def _registry() -> None:
    st.markdown(section("Models & research notes", 0,
                        help="Tune how much each model counts toward the consensus, and log "
                             "research that should shape the models."), unsafe_allow_html=True)
    reg = registry.load()

    st.markdown("**Add research — paste an abstract, link a URL, or upload a paper**")
    fam = st.selectbox("Family this informs", sorted(reg["weights"].keys()),
                       help="Which model family the research bears on.")
    title = st.text_input("Note title / paper")
    url = st.text_input("Source URL (optional)",
                        help="A link to the paper/insight. The /zenith-research Claude skill can "
                             "later turn a flagged note into a signal-module scaffold.")
    upload = st.file_uploader("Upload a paper (PDF/txt, optional)", type=["pdf", "txt", "md"])
    abstract = st.text_area("Paste abstract or summary", height=150)
    w = st.slider("Set weight for this family", 0.0, 2.0, float(reg["weights"].get(fam, 1.0)), 0.05)
    if st.button("Save note + weight"):
        src = url.strip()
        body = abstract.strip()
        if upload is not None:
            src = src or upload.name
            if not body:
                try:
                    body = upload.getvalue().decode("utf-8", "ignore")[:4000]
                except Exception:
                    body = f"(uploaded {upload.name}; binary — process with /zenith-research)"
        status = "pending-review" if (src or upload) and not body.strip() else ""
        registry.add_note(fam, title or (src or "(untitled)"), body,
                          weight_adjustment=w, source=src, status=status)
        st.success(f"Saved note against '{fam}' (weight {w}). "
                   + ("Flagged pending-review for the /zenith-research skill." if status else ""))

    st.markdown(section("Current family weights", 2), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"family": k, "weight": v} for k, v in reg["weights"].items()]),
                 use_container_width=True, height=280)

    st.markdown(section("Research notes log", 4), unsafe_allow_html=True)
    if reg["notes"]:
        for n in reg["notes"][:30]:
            flag = f"  ·  🟡 {n['status']}" if n.get("status") else ""
            st.markdown(f"**{n['title']}**  ·  _{n['family']}_  ·  {n['ts']}{flag}")
            if n.get("source"):
                st.caption(f"source: {n['source']}")
            if n.get("abstract"):
                st.caption(n["abstract"][:600])
    else:
        st.caption("No notes yet.")

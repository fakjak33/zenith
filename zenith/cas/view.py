"""Streamlit rendering for the CAS tab. Kept out of app.py so the viewer stays
thin. Reuses the Zenith theme/section helpers; uses altair (ships with Streamlit)
for the overlap heatmap."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import store_cas, registry, contingency
from .universe import label_of
from ..ui_theme import section
from . import DISCLAIMER

_STATE_COLOR = {"buy": "#2ec4b6", "neutral": "#b8b8b8", "sell": "#ff5a3c"}

SEGMENT_TABS = ["Overview", "Strategies", "Flows & positioning", "Themes",
                "Rebalance", "Contingency", "BCT consensus", "Models & notes"]


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
    df = _signals_df()

    if seg == "Overview":
        _overview(df)
    elif seg == "Strategies":
        _segment(df, "strategies", "Per-asset strategy signals (101-alpha + classics)")
    elif seg == "Flows & positioning":
        _flows(df)
    elif seg == "Themes":
        _segment(df, "themes", "Sector / theme relative-strength & breadth")
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

    st.markdown(section("Highest-conviction overlap (signals aligned across segments)", 0),
                unsafe_allow_html=True)
    if ranked:
        top = pd.DataFrame(ranked).head(20)
        top["label"] = top["asset"].map(label_of)
        st.dataframe(top[["asset", "label", "asset_class", "net", "buy_count",
                          "sell_count", "n_segments", "mean_strength"]],
                     use_container_width=True, height=400)

    st.markdown(section("Overlap heatmap — assets × signal families", 2), unsafe_allow_html=True)
    _heatmap(df)

    st.markdown(section("BCT consensus leaderboard", 4), unsafe_allow_html=True)
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


def _segment(df: pd.DataFrame, segment: str, title: str) -> None:
    st.markdown(section(title, 0), unsafe_allow_html=True)
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
    st.markdown(section("Models & research notes", 0), unsafe_allow_html=True)
    reg = registry.load()

    st.markdown("**Refine a model by pasting a research summary / abstract**")
    fam = st.selectbox("Family", sorted(reg["weights"].keys()))
    title = st.text_input("Note title / paper")
    abstract = st.text_area("Paste abstract or summary", height=150)
    w = st.slider("Set weight for this family", 0.0, 2.0, float(reg["weights"].get(fam, 1.0)), 0.05)
    if st.button("Save note + weight"):
        registry.add_note(fam, title or "(untitled)", abstract, weight_adjustment=w)
        st.success(f"Saved note against '{fam}' and set weight to {w}.")

    st.markdown(section("Current family weights", 2), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"family": k, "weight": v} for k, v in reg["weights"].items()]),
                 use_container_width=True, height=280)

    st.markdown(section("Research notes log", 4), unsafe_allow_html=True)
    if reg["notes"]:
        for n in reg["notes"][:30]:
            st.markdown(f"**{n['title']}**  ·  _{n['family']}_  ·  {n['ts']}")
            if n.get("abstract"):
                st.caption(n["abstract"][:600])
    else:
        st.caption("No notes yet.")

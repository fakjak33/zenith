"""Streamlit rendering for the Weekly Brief tab. Concise, visual, Zenith-themed."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import load, DISCLAIMER
from ..ui_theme import section, card_html

_PCT = "%+.1f%%"


def _pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].apply(lambda v: None if v is None else round(float(v) * 100, 1))
    return df


def render() -> None:
    brief = load("brief", {})
    st.caption(DISCLAIMER)
    if not brief:
        st.info("No Weekly Brief yet. Run `python -m zenith.brief.compute` "
                "(or wait for the scheduled Action).")
        return
    st.caption(f"Week of {brief.get('as_of','?')}")

    _talking_points(brief.get("talking_points", []))
    _overview(brief.get("market_overview", []))
    _sectors_industries(brief)
    _rates(brief.get("rates", {}))
    _additional_charts(brief)
    _earnings(brief.get("earnings", []))
    _cas(brief.get("cas", {}), brief.get("gex", []))
    _news(brief.get("news", []))


def _talking_points(pts: list[str]) -> None:
    st.markdown(section("Talking points — the week in plain English", 0,
                        help="A 1-minute, jargon-free read on current conditions."),
                unsafe_allow_html=True)
    if not pts:
        st.caption("—")
        return
    st.markdown("\n".join(f"- {p}" for p in pts))


def _overview(ov: list[dict]) -> None:
    st.markdown(section("Market overview — what's moving", 1,
                        help="Key markets with 1-week / 1-month / 3-month / YTD performance and a "
                             "6-month trend. Green = up."), unsafe_allow_html=True)
    if not ov:
        st.caption("No data.")
        return
    df = pd.DataFrame(ov)
    df = _pct_cols(df, ["w1", "m1", "m3", "ytd"])
    show = df[["label", "last", "w1", "m1", "m3", "ytd", "spark"]].rename(
        columns={"label": "Market", "last": "Last", "w1": "1W", "m1": "1M",
                 "m3": "3M", "ytd": "YTD", "spark": "6M trend"})
    try:
        st.dataframe(show, use_container_width=True, height=595, hide_index=True,
                     column_config={
                         "1W": st.column_config.NumberColumn(format=_PCT),
                         "1M": st.column_config.NumberColumn(format=_PCT),
                         "3M": st.column_config.NumberColumn(format=_PCT),
                         "YTD": st.column_config.NumberColumn(format=_PCT),
                         "6M trend": st.column_config.LineChartColumn(width="medium"),
                     })
    except Exception:
        st.dataframe(show.drop(columns=["6M trend"]), use_container_width=True, hide_index=True)


def _bar(rows: list[dict], value: str, title: str) -> None:
    if not rows:
        st.caption("No data.")
        return
    df = pd.DataFrame(rows)
    df[value + "_pct"] = df[value].apply(lambda v: None if v is None else round(v * 100, 2))
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{value}_pct:Q", title=f"{title} (%)"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(alt.datum[f"{value}_pct"] >= 0,
                                alt.value("#2ec4b6"), alt.value("#ff5a3c")),
            tooltip=["label", f"{value}_pct"],
        ).properties(height=max(180, 22 * len(df))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df[["label", value + "_pct"]], use_container_width=True, hide_index=True)


def _sectors_industries(brief: dict) -> None:
    st.markdown(section("Sector & industry breakdown", 2,
                        help="11 SPDR sectors and a broad set of industry ETFs, ranked by 1-week "
                             "performance, plus the week's best/worst S&P 500 movers."),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Sectors — 1 week**")
        _bar(brief.get("sectors", []), "w1", "1W")
    with c2:
        st.caption("**Top industries — 1 week**")
        _bar(brief.get("industries", [])[:15], "w1", "1W")

    heat = brief.get("stock_heatmap", {})
    if heat:
        br = heat.get("breadth", {})
        if br.get("pct_above_50dma") is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("S&P 500 breadth", f"{br['pct_above_50dma']:.0%}",
                      help="% of S&P 500 names above their 50-day moving average.")
            m2.metric("Above 200-day", f"{br.get('pct_above_200dma',0):.0%}")
            m3.metric("Universe", br.get("n", 0))
        b1, b2 = st.columns(2)
        with b1:
            st.caption("**Best movers (1W)**")
            best = _pct_cols(pd.DataFrame(heat.get("best_1w", [])), ["w1", "m1"])
            st.dataframe(best, use_container_width=True, hide_index=True, height=300,
                         column_config={"w1": st.column_config.NumberColumn("1W", format=_PCT),
                                        "m1": st.column_config.NumberColumn("1M", format=_PCT)})
        with b2:
            st.caption("**Worst movers (1W)**")
            worst = _pct_cols(pd.DataFrame(heat.get("worst_1w", [])), ["w1", "m1"])
            st.dataframe(worst, use_container_width=True, hide_index=True, height=300,
                         column_config={"w1": st.column_config.NumberColumn("1W", format=_PCT),
                                        "m1": st.column_config.NumberColumn("1M", format=_PCT)})


def _rates(rt: dict) -> None:
    st.markdown(section("Rate expectations", 3,
                        help="Fed funds target + a rough market-implied 12-month path. Full "
                             "FedWatch probabilities require CME data (paid)."),
                unsafe_allow_html=True)
    ff = rt.get("fed_funds", {})
    if ff:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fed funds target",
                  f"{ff.get('target_lower','?')}–{ff.get('target_upper','?')}%")
        c2.metric("Effective", f"{ff.get('effective','?')}%")
        c3.metric("Implied 12m", f"{ff.get('implied_12m_change_bps','?')} bps",
                  help=ff.get("note", ""))
        c4.metric("Next FOMC", (ff.get("next_fomc") or ["—"])[0])
        if ff.get("next_fomc"):
            st.caption("Upcoming FOMC decisions: " + " · ".join(ff["next_fomc"]))

    st.markdown(section("Fixed income — curve, credit & global", 4,
                        help="US Treasury curve, credit spreads, inflation breakeven and major "
                             "global 10-year yields (FRED)."), unsafe_allow_html=True)
    curve = rt.get("curve", {})
    if not curve:
        st.caption("No rates data.")
        return
    rows = [{"id": k, "label": v["label"], "value": v["value"]} for k, v in curve.items()]
    tdf = pd.DataFrame(rows)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(tdf[["label", "value"]].rename(columns={"label": "Series", "value": "%"}),
                     use_container_width=True, hide_index=True, height=420)
    with c2:
        _curve_chart(curve)


def _curve_chart(curve: dict) -> None:
    order = [("DGS3MO", "3M"), ("DGS6MO", "6M"), ("DGS1", "1Y"), ("DGS2", "2Y"),
             ("DGS5", "5Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]
    pts = [{"tenor": lbl, "i": i, "yield": curve[sid]["value"]}
           for i, (sid, lbl) in enumerate(order) if sid in curve]
    if len(pts) < 2:
        st.caption("Curve unavailable.")
        return
    df = pd.DataFrame(pts)
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_line(point=True, color="#2a9bc4").encode(
            x=alt.X("i:O", axis=alt.Axis(labelExpr="", title="tenor"),
                    sort=None),
            y=alt.Y("yield:Q", title="yield %", scale=alt.Scale(zero=False)),
            tooltip=["tenor", "yield"],
        ).properties(height=400))
        text = ch.mark_text(dy=-12, color="#fff").encode(text="tenor")
        st.altair_chart(ch + text, use_container_width=True)
    except Exception:
        st.line_chart(df.set_index("tenor")["yield"])


def _additional_charts(brief: dict) -> None:
    from ..cas import store_cas
    from ..cas.universe import frm_tag, REGION_LABEL
    st.markdown(section("Additional charts — factor rotation", 5,
                        help="The CAS Factor-Rotation composite by style × region: which factors "
                             "are trending across the US, developed and emerging markets."),
                unsafe_allow_html=True)
    frm = [s for s in store_cas.load("factor_rotation", [])
           if s.get("family") == "frm_composite"]
    styles = []
    for s in frm:
        tag = frm_tag(s["asset"]) or {}
        if tag.get("group") == "style":
            styles.append({"label": tag["label"], "signal": s["signal"],
                           "region": REGION_LABEL.get(tag.get("region"), tag.get("region", ""))})
    if not styles:
        st.caption("No factor-rotation data yet (run the CAS compute).")
        return
    df = pd.DataFrame(styles)
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_rect().encode(
            x=alt.X("region:N", title=None),
            y=alt.Y("label:N", title=None),
            color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1])),
            tooltip=["label", "region", "signal"],
        ).properties(height=300))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df.pivot_table(index="label", columns="region", values="signal"),
                     use_container_width=True)


def _earnings(earn: list[dict]) -> None:
    st.markdown(section("Earnings — major reports ahead", 1,
                        help="Upcoming earnings dates for mega-cap names (next ~10 days)."),
                unsafe_allow_html=True)
    if not earn:
        st.caption("No major earnings flagged in the window.")
        return
    st.dataframe(pd.DataFrame(earn).rename(columns={"ticker": "Ticker", "date": "Date"}),
                 use_container_width=True, hide_index=True, height=min(420, 40 + 28 * len(earn)))


def _cas(cas: dict, gx: list[dict]) -> None:
    st.markdown(section("CAS highlights — top signals", 2,
                        help="The strongest Factor-Rotation and consensus signals from the CAS "
                             "monitor this week."), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Factor Rotation — most bullish**")
        top = cas.get("frm_top", [])
        if top:
            st.dataframe(pd.DataFrame([{"asset": s["asset"], "signal": s["signal"]} for s in top]),
                         use_container_width=True, hide_index=True, height=220)
    with c2:
        st.caption("**Factor Rotation — most bearish**")
        bot = cas.get("frm_bottom", [])
        if bot:
            st.dataframe(pd.DataFrame([{"asset": s["asset"], "signal": s["signal"]} for s in bot]),
                         use_container_width=True, hide_index=True, height=220)

    st.markdown(section("Options — dealer gamma (proxy)", 4,
                        help="Rough net dealer gamma from yfinance option chains. Positive = "
                             "dealers dampen moves; negative = moves can accelerate. A proxy."),
                unsafe_allow_html=True)
    if gx:
        cols = st.columns(len(gx))
        for i, g in enumerate(gx):
            cols[i].metric(f"{g['ticker']} net GEX", f"{g['net_gex_bn']:+.2f} bn",
                           help=g["regime"])
    else:
        st.caption("GEX unavailable (no option data).")


def _news(news: list[dict]) -> None:
    st.markdown(section("Market-moving news", 1,
                        help="Selective: news prioritised toward names that actually moved this "
                             "week, drawn from Zenith's feeds."), unsafe_allow_html=True)
    if not news:
        st.caption("No news items.")
        return
    html = "".join(card_html({"source": ("● " if n.get("hot") else "") + n["source"],
                              "title": n["title"], "link": n.get("link", "#")})
                   for n in news)
    st.markdown(html, unsafe_allow_html=True)

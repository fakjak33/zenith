"""Streamlit rendering for the Weekly Brief tab. Concise, visual, Zenith-themed.

Tables carry per-column "?" help (st.column_config help); comparative charts are
normalized (rebased-to-100) so different markets are visually comparable.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from . import load, DISCLAIMER
from .sources import GROUPS
from ..ui_theme import section, card_html
from ..cas.universe import REGION_LABEL

_PCT = "%+.1f%%"

# per-column hover help, reused across tables
COLS = {
    "1W": "1-week price return.", "1M": "1-month price return.",
    "3M": "3-month price return.", "YTD": "Year-to-date price return.",
    "Last": "Latest price.", "Move 1W": "The stock's 1-week return.",
    "Surprise": "Reported EPS vs the consensus estimate (%).",
    "EPS est.": "Consensus EPS estimate for the upcoming report.",
    "Signal": "Model signal, -1 (strong sell) to +1 (strong buy).",
}


def _pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].apply(lambda v: None if v is None else round(float(v) * 100, 1))
    return df


def _num(col, help_key=None):
    return st.column_config.NumberColumn(col, format=_PCT, help=COLS.get(help_key or col, ""))


def render() -> None:
    brief = load("brief", {})
    st.caption(DISCLAIMER)
    if not brief:
        st.info("No Weekly Brief yet. Run `python -m zenith.brief.compute` "
                "(or wait for the scheduled Action).")
        return
    st.caption(f"Week of {brief.get('as_of','?')}")

    _talking_points(brief.get("talking_points", []))
    _overview(brief.get("market_overview", {}))
    _sectors_industries(brief)
    _rates(brief.get("rates", {}))
    _rotation(brief.get("rotation", {}))
    _earnings(brief.get("earnings", {}))
    _cas(brief.get("cas", {}), brief.get("gex", []))
    _news(brief.get("news", []))


def _talking_points(pts: list[str]) -> None:
    st.markdown(section("Talking points — the week in plain English", 0,
                        help="A 1-minute, jargon-free read on current conditions."),
                unsafe_allow_html=True)
    st.markdown("\n".join(f"- {p}" for p in pts) if pts else "—")


# --- comparative (rebased-to-100) line chart -------------------------------
def _rebased_chart(rows: list[dict], window, height: int = 300) -> None:
    recs = []
    for r in rows:
        ser = r.get("series") or []
        if not ser:
            continue
        df = pd.DataFrame(ser)
        df["d"] = pd.to_datetime(df["d"])
        if window == "YTD":
            df = df[df["d"] >= pd.Timestamp(date.today().year, 1, 1)]
        else:
            df = df.tail(int(window))
        if len(df) < 2 or df["c"].iloc[0] <= 0:
            continue
        df = df.assign(pct=(df["c"] / df["c"].iloc[0] - 1.0) * 100.0, label=r["label"])
        recs.append(df[["d", "pct", "label"]])
    if not recs:
        st.caption("No series to chart.")
        return
    big = pd.concat(recs)
    try:
        import altair as alt
        ch = (alt.Chart(big).mark_line().encode(
            x=alt.X("d:T", title=None),
            y=alt.Y("pct:Q", title="% change"),
            color=alt.Color("label:N", legend=alt.Legend(title=None)),
            tooltip=["label", alt.Tooltip("d:T"), alt.Tooltip("pct:Q", format="+.1f")],
        ).properties(height=height))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.line_chart(big.pivot_table(index="d", columns="label", values="pct"))


def _perf_table(rows: list[dict], name_col: str = "Market") -> None:
    df = _pct_cols(pd.DataFrame(rows), ["w1", "m1", "m3", "ytd"])
    show = df[["label", "last", "w1", "m1", "m3", "ytd", "spark"]].rename(
        columns={"label": name_col, "last": "Last", "w1": "1W", "m1": "1M",
                 "m3": "3M", "ytd": "YTD", "spark": "6M trend"})
    try:
        st.dataframe(show, use_container_width=True, hide_index=True,
                     height=min(560, 45 + 35 * len(show)),
                     column_config={
                         "Last": st.column_config.NumberColumn(help=COLS["Last"]),
                         "1W": _num("1W"), "1M": _num("1M"), "3M": _num("3M"), "YTD": _num("YTD"),
                         "6M trend": st.column_config.LineChartColumn("6M trend", width="medium"),
                     })
    except Exception:
        st.dataframe(show.drop(columns=["6M trend"]), use_container_width=True, hide_index=True)


def _overview(ov: dict) -> None:
    st.markdown(section("Market overview — what's moving", 1,
                        help="Key markets by asset class. Tables show 1W/1M/3M/YTD; the chart "
                             "rebases every market to 0% so you can compare them directly."),
                unsafe_allow_html=True)
    if not ov:
        st.caption("No data.")
        return
    win_label = st.radio("Comparison window", ["1 week", "1 month", "YTD"],
                         horizontal=True, key="ov_win")
    win = {"1 week": 5, "1 month": 21, "YTD": "YTD"}[win_label]
    tabs = st.tabs([GROUPS[k][0] for k in ov.keys() if k in GROUPS])
    for tab, key in zip(tabs, [k for k in ov.keys() if k in GROUPS]):
        with tab:
            rows = ov.get(key, [])
            if not rows:
                st.caption("No data.")
                continue
            _perf_table(rows)
            _rebased_chart(rows, win)


def _bar(rows: list[dict], value: str, title: str, height_per: int = 22) -> None:
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
        ).properties(height=max(160, height_per * len(df))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df[["label", value + "_pct"]], use_container_width=True, hide_index=True)


def _movers_table(rows: list[dict], label: str) -> None:
    df = _pct_cols(pd.DataFrame(rows), ["w1", "m1"])
    cols = [c for c in ["ticker", "name", "w1", "m1"] if c in df]
    st.dataframe(df[cols].rename(columns={"ticker": "Ticker", "name": "Company",
                                          "w1": "1W", "m1": "1M"}),
                 use_container_width=True, hide_index=True, height=300,
                 column_config={"1W": _num("1W"), "1M": _num("1M")})


def _sectors_industries(brief: dict) -> None:
    st.markdown(section("Sector & industry breakdown", 2,
                        help="11 SPDR sectors and a broad set of industry ETFs by 1-week return, "
                             "plus the week's S&P 500 leaders and laggards."), unsafe_allow_html=True)
    sectors = brief.get("sectors", [])
    st.caption("**Sectors — 1 week** (bar) and rebased relative performance (chart)")
    c1, c2 = st.columns([2, 3])
    with c1:
        _bar(sectors, "w1", "1W")
    with c2:
        _rebased_chart(sectors, 21, height=max(200, 22 * len(sectors)))

    industries = brief.get("industries", [])
    i1, i2 = st.columns(2)
    with i1:
        st.caption("**Top industries (1W)**")
        _bar(industries[:15], "w1", "1W")
    with i2:
        st.caption("**Bottom industries (1W)**")
        _bar(industries[-15:], "w1", "1W")

    heat = brief.get("stock_heatmap", {})
    if heat:
        br = heat.get("breadth", {})
        if br.get("pct_above_50dma") is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("S&P 500 breadth", f"{br['pct_above_50dma']:.0%}",
                      help="% of S&P 500 names above their 50-day moving average.")
            m2.metric("Above 200-day", f"{br.get('pct_above_200dma',0):.0%}",
                      help="% above their 200-day moving average — a longer-trend breadth read.")
            m3.metric("Universe", br.get("n", 0))
        b1, b2 = st.columns(2)
        with b1:
            st.caption("**Leaders (1W)**")
            _movers_table(heat.get("leaders_1w", []), "Leaders")
        with b2:
            st.caption("**Laggards (1W)**")
            _movers_table(heat.get("laggards_1w", []), "Laggards")


def _rates(rt: dict) -> None:
    st.markdown(section("Rate expectations", 3,
                        help="Fed funds target + a rough market-implied 12-month path. Full "
                             "FedWatch probabilities require CME data (paid)."), unsafe_allow_html=True)
    ff = rt.get("fed_funds", {})
    if ff:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fed funds target", f"{ff.get('target_lower','?')}–{ff.get('target_upper','?')}%")
        c2.metric("Effective", f"{ff.get('effective','?')}%")
        c3.metric("Implied 12m", f"{ff.get('implied_12m_change_bps','?')} bps", help=ff.get("note", ""))
        c4.metric("Next FOMC", (ff.get("next_fomc") or ["—"])[0])
        if ff.get("next_fomc"):
            st.caption("Upcoming FOMC decisions: " + " · ".join(ff["next_fomc"]))

    st.markdown(section("Fixed income — curve, credit & global", 4,
                        help="US Treasury curve, credit spreads, the 10y inflation breakeven, and "
                             "major global 10-year yields (FRED)."), unsafe_allow_html=True)
    curve = rt.get("curve", {})
    if not curve:
        st.caption("No rates data.")
        return
    rows = [{"Series": v["label"], "%": v["value"]} for v in curve.values()]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=440)
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
        base = alt.Chart(df).encode(
            x=alt.X("i:O", sort=None, axis=alt.Axis(title="tenor", labelExpr="")),
            y=alt.Y("yield:Q", title="yield %", scale=alt.Scale(zero=False)))
        st.altair_chart(base.mark_line(point=True, color="#2a9bc4")
                        + base.mark_text(dy=-12, color="#fff").encode(text="tenor"),
                        use_container_width=True)
    except Exception:
        st.line_chart(df.set_index("tenor")["yield"])


def _rotation(rot: dict) -> None:
    st.markdown(section("Factor rotation — by timeframe", 5,
                        help="Cross-sectional momentum: each factor/industry ranked vs its peer "
                             "group over the chosen look-back. +1 = strongest, -1 = weakest."),
                unsafe_allow_html=True)
    if not rot:
        st.caption("No rotation data (run the brief compute).")
        return
    tf = st.radio("Look-back", ["1w", "1m", "3m", "6m", "1y"], index=1, horizontal=True, key="rot_tf")
    rows = rot.get(tf, [])
    styles = [r for r in rows if r.get("group") == "style"]
    for r in styles:
        r["region_lbl"] = REGION_LABEL.get(r.get("region"), r.get("region", ""))
    st.caption("Equity styles × region. EM Growth has no liquid US-listed ETF, so that cell is blank.")
    if styles:
        df = pd.DataFrame(styles)
        try:
            import altair as alt
            ch = (alt.Chart(df).mark_rect().encode(
                x=alt.X("region_lbl:N", title=None),
                y=alt.Y("label:N", title=None),
                color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1])),
                tooltip=["label", "region_lbl", "signal"],
            ).properties(height=300))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.dataframe(df.pivot_table(index="label", columns="region_lbl", values="signal"),
                         use_container_width=True)
    with st.expander("Industries & region-sectors — leaders / laggards this timeframe"):
        non = sorted([r for r in rows if r.get("group") != "style"],
                     key=lambda r: r["signal"], reverse=True)
        for r in non:
            r["label2"] = f"{r['label']}"
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Top**")
            st.dataframe(pd.DataFrame(non[:12])[["label", "signal"]] if non else pd.DataFrame(),
                         use_container_width=True, hide_index=True)
        with c2:
            st.caption("**Bottom**")
            st.dataframe(pd.DataFrame(non[-12:][::-1])[["label", "signal"]] if non else pd.DataFrame(),
                         use_container_width=True, hide_index=True)


def _earnings(earn: dict) -> None:
    st.markdown(section("Earnings — recent reactions & what's ahead", 1,
                        help="Recent reports (with the stock's 1-week move) and upcoming reports, "
                             "from the Nasdaq earnings calendar."), unsafe_allow_html=True)
    if not earn or (not earn.get("recent") and not earn.get("upcoming")):
        st.caption("No earnings in the window.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Recent**")
        rec = earn.get("recent", [])
        if rec:
            df = _pct_cols(pd.DataFrame(rec), ["move_1w"])
            cols = [c for c in ["ticker", "name", "date", "surprise", "move_1w"] if c in df]
            st.dataframe(df[cols].rename(columns={"ticker": "Ticker", "name": "Company",
                         "date": "Date", "surprise": "Surprise", "move_1w": "Move 1W"}),
                         use_container_width=True, hide_index=True, height=360,
                         column_config={"Surprise": st.column_config.TextColumn(help=COLS["Surprise"]),
                                        "Move 1W": _num("Move 1W")})
        else:
            st.caption("—")
    with c2:
        st.caption("**Upcoming**")
        up = earn.get("upcoming", [])
        if up:
            df = pd.DataFrame(up)
            cols = [c for c in ["ticker", "name", "date", "time", "eps_forecast"] if c in df]
            st.dataframe(df[cols].rename(columns={"ticker": "Ticker", "name": "Company",
                         "date": "Date", "time": "Time", "eps_forecast": "EPS est."}),
                         use_container_width=True, hide_index=True, height=360,
                         column_config={"EPS est.": st.column_config.TextColumn(help=COLS["EPS est."])})
        else:
            st.caption("—")


def _cas(cas: dict, gx: list[dict]) -> None:
    st.markdown(section("CAS highlights — strongest signals", 2,
                        help="The most bullish and most bearish Factor-Rotation composite signals "
                             "from the CAS monitor (green = buy, red = sell)."), unsafe_allow_html=True)
    rows = []
    for s in cas.get("frm_top", []):
        rows.append({"label": s["asset"], "signal": s["signal"]})
    for s in cas.get("frm_bottom", []):
        rows.append({"label": s["asset"], "signal": s["signal"]})
    if rows:
        df = pd.DataFrame(rows).drop_duplicates("label")
        try:
            import altair as alt
            ch = (alt.Chart(df).mark_bar().encode(
                x=alt.X("signal:Q", title="composite signal", scale=alt.Scale(domain=[-1, 1])),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.condition(alt.datum.signal >= 0, alt.value("#2ec4b6"), alt.value("#ff5a3c")),
                tooltip=["label", "signal"],
            ).properties(height=max(180, 26 * len(df))))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No CAS signals yet — run the CAS compute.")

    st.markdown(section("Options — dealer gamma (proxy)", 4,
                        help="Rough net dealer gamma from yfinance option chains. Positive = "
                             "dealers dampen moves; negative = moves can accelerate. A proxy."),
                unsafe_allow_html=True)
    if gx:
        cols = st.columns(len(gx))
        for i, g in enumerate(gx):
            cols[i].metric(f"{g['ticker']} net GEX", f"{g['net_gex_bn']:+.2f} bn", help=g["regime"])
    else:
        st.caption("GEX unavailable (no option data).")


def _news(news: list[dict]) -> None:
    st.markdown(section("Market-moving news — the names that moved", 1,
                        help="Ticker-specific headlines for the week's biggest movers (Google News)."),
                unsafe_allow_html=True)
    if not news:
        st.caption("No news items.")
        return
    for n in news:
        mv = n.get("move")
        badge = f" ({mv*100:+.1f}%)" if isinstance(mv, (int, float)) else ""
        st.markdown(f"**{n['ticker']}** — {n.get('name','')}{badge}")
        html = "".join(card_html({"source": "news", "title": it["title"], "link": it.get("link", "#")})
                       for it in n.get("items", []))
        st.markdown(html, unsafe_allow_html=True)

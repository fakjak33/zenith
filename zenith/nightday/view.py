"""Streamlit rendering for the NIGHT & DAY tab — overnight vs intraday.

Leads with the live SPY split (how much of the trailing gain was earned
overnight vs intraday), ETF cumulative decomposition charts, an R1000 scatter of
overnight-share vs total return, and the ranked overnight-momentum / tug-of-war
screens. Reads only committed JSON (data/nightday/).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load
from . import history as nhist
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section, stamp


def _cum_chart(series: list[dict], title: str) -> None:
    if not series:
        return
    df = pd.DataFrame(series)
    long = df.melt("d", value_vars=["overnight", "intraday", "total"],
                   var_name="stream", value_name="cum")
    long["d"] = pd.to_datetime(long["d"])
    try:
        import altair as alt
        ch = (alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("d:T", title=None, axis=alt.Axis(labelLimit=0)),
            y=alt.Y("cum:Q", title="cumulative return", axis=alt.Axis(format="+.0%")),
            color=alt.Color("stream:N", title=None,
                            scale=alt.Scale(domain=["overnight", "intraday", "total"],
                                            range=[THEME.teal, THEME.coral, THEME.muted]),
                            legend=alt.Legend(orient="top")),
            tooltip=["d:T", "stream:N", alt.Tooltip("cum:Q", format="+.1%")]))
        st.altair_chart(ch.properties(height=240, title=title), use_container_width=True)
    except Exception:
        st.line_chart(df.set_index("d")[["overnight", "intraday", "total"]])


def _scatter(rows: list[dict]) -> None:
    pts = [r for r in rows if r.get("share_overnight") is not None
           and r.get("cum_total") is not None]
    if len(pts) < 10:
        return
    df = pd.DataFrame([{"ticker": r["ticker"], "sector": r.get("sector") or "?",
                        "overnight share": r["share_overnight"],
                        "total return": r["cum_total"]} for r in pts])
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_circle(opacity=0.6, size=45).encode(
            x=alt.X("overnight share:Q", axis=alt.Axis(format="+.0%", labelLimit=0),
                    title="share of drift earned overnight"),
            y=alt.Y("total return:Q", axis=alt.Axis(format="+.0%", labelLimit=0)),
            color=alt.Color("sector:N", legend=alt.Legend(orient="right", labelLimit=0)),
            tooltip=["ticker", "sector", alt.Tooltip("overnight share:Q", format="+.0%"),
                     alt.Tooltip("total return:Q", format="+.0%")]))
        st.altair_chart(ch.properties(height=340), use_container_width=True)
    except Exception:
        pass


def _screen_table(rows: list[dict], key: str) -> None:
    if not rows:
        st.caption("— none —")
        return
    cols = ["rank", "ticker", "name", "sector", "on_mom", "overnight_avg_bp",
            "intraday_avg_bp", "share_overnight", "tug"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(400, 45 + 34 * len(df)))


def _tracker() -> None:
    summ = nhist.summarize(load("history", {}).get("rows", []))
    if not summ:
        st.caption("Tracker fills in as +20td horizons mature (screen starts live).")
        return
    cols = st.columns(3)
    for i, side in enumerate(("long", "short", "all")):
        c = summ.get(side)
        if c:
            cols[i].metric(f"overnight-mom {side} n={c['n']}",
                           f"{c['avg']*100:+.2f}% avg", f"{c['win_rate']:.0%} win",
                           delta_color="off")


def today_badge() -> str | None:
    return None


def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating(
        "B", "overnight vs intraday",
        "Strong, long-lived statistics (Lou-Polk-Skouras 2019): nearly all the "
        "market's gain is overnight, and firm-level overnight/intraday tilts "
        "persist for years. But literal overnight capture pays the open/close "
        "spread daily — a monitor and tilt input, not a trading system."),
        unsafe_allow_html=True)
    st.markdown(key_findings([
        {"stat": "Almost all of the U.S. market's cumulative gain accrues "
                 "OVERNIGHT (close->open); intraday is roughly flat or negative.",
         "cite": "Lou, Polk & Skouras (2019)"},
        {"stat": "Firm-level overnight AND intraday returns each show strong "
                 "continuation that persists for years (offsetting reversal).",
         "cite": "Lou, Polk & Skouras (2019)"},
    ]), unsafe_allow_html=True)

    panel = load("panel", {})
    screen = load("screen", {})
    if not panel.get("etfs") and not screen.get("ranked"):
        st.info("No NIGHT & DAY data yet — run `python -m zenith.nightday.compute` "
                "or wait for the nightly Action.")
        return
    st.markdown(stamp(panel.get("as_of", screen.get("as_of", "?")), "NIGHT & DAY"),
                unsafe_allow_html=True)

    # --- headline SPY split ---------------------------------------------------
    spy = (panel.get("etfs") or {}).get("SPY", {}).get("stats")
    if spy:
        st.markdown(section("The split — where the return is made (SPY, ~2y)", 3),
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overnight cumulative", f"{spy['cum_overnight']*100:+.1f}%",
                  help="If you had held SPY only from close to open.")
        m2.metric("Intraday cumulative", f"{spy['cum_intraday']*100:+.1f}%",
                  help="If you had held SPY only from open to close.")
        m3.metric("Overnight avg", f"{spy['overnight_avg_bp']:+.1f} bp/day",
                  help=f"t={spy.get('overnight_t')}")
        m4.metric("Intraday avg", f"{spy['intraday_avg_bp']:+.1f} bp/day",
                  help=f"t={spy.get('intraday_t')}")
        sp = load("panel", {}).get("etfs", {}).get("SPY", {}).get("series", [])
        _cum_chart(sp, "SPY — overnight vs intraday, cumulative")

    # --- ETF panel ------------------------------------------------------------
    etfs = panel.get("etfs", {})
    if etfs:
        st.markdown(section("Factor ETFs — overnight vs intraday", 1),
                    unsafe_allow_html=True)
        pick = st.selectbox("ETF", [t for t in etfs if t != "SPY"] or list(etfs),
                            key="nd_etf")
        _cum_chart(etfs.get(pick, {}).get("series", []),
                   f"{pick} — {etfs.get(pick, {}).get('name', pick)}")

    # --- R1000 scatter --------------------------------------------------------
    if screen.get("ranked"):
        st.markdown(section("Russell 1000 — overnight share vs total return", 2,
                            help="Right = a stock earns most of its drift "
                                 "overnight; up = higher total return."),
                    unsafe_allow_html=True)
        _scatter(screen["ranked"])

    # --- ranked screens -------------------------------------------------------
    st.markdown(section("Overnight-momentum screen (LPS continuation)", 4,
                        help="Vol-scaled last-month overnight return. Long = "
                             "strongest overnight tilt; short = weakest."),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("▲ strongest overnight momentum")
        _screen_table(screen.get("long", []), "nd_long")
    with c2:
        st.caption("▼ weakest overnight momentum")
        _screen_table(screen.get("short", []), "nd_short")

    st.markdown(section("Live tracker — overnight-momentum vs SPY (+20td)", 0),
                unsafe_allow_html=True)
    _tracker()

    with st.expander("How NIGHT & DAY works — the research"):
        st.markdown(
            "**Decomposition**: every day splits into an OVERNIGHT return "
            "(close->open) and an INTRADAY return (open->close); they compound "
            "to the usual close-to-close total. Lou, Polk & Skouras (2019, JFE, "
            "'A Tug of War') document that almost all of the market's cumulative "
            "gain has come overnight while intraday is roughly flat, and that "
            "firm-level overnight and intraday tilts each show strong "
            "CONTINUATION lasting years, with an offsetting cross-period "
            "reversal.\n\n"
            "**The screen**: the overnight-momentum signal is last month's "
            "summed overnight return, vol-scaled — high = a persistent "
            "overnight bid. The tug-of-war spread is overnight-minus-intraday "
            "cumulative. Decile members are snapshotted and scored vs SPY at "
            "+20 trading days.\n\n"
            "**Honest caveat (tier B)**: the statistics are strong, but "
            "capturing the overnight premium literally means trading the "
            "open/close every day and paying that spread — which eats the edge. "
            "Treat this as a research monitor and a *tilt/timing* input (e.g. "
            "which names to prefer holding overnight), not a standalone "
            "strategy.")

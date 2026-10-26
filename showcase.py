"""ZENITH//TERM style prototype -- a one-page preview of the redesign on real data.

    streamlit run showcase.py --server.port 8602

Nothing here is wired into app.py yet: it exists so the look can be judged (and
changed) before every tab is restyled.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from zenith import ui_altair, ui_term as zt
from zenith.ui_term import TERM as T

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ZENITH//TERM", layout="wide", page_icon="▣")
st.markdown(zt.CSS, unsafe_allow_html=True)
ui_altair.enable()


def j(rel: str, default=None):
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_resource(show_spinner="booting market feed…")
def market():
    from zenith.theta.market import Market
    return Market.load()


m = market()
etf = j("data/etfmom/scores_latest.json", {})
rows = etf.get("rows", [])
sectors = j("data/mom/sectors.json", {}).get("sectors", {})
classes = j("data/etfmom/categories.json", {}).get("asset_classes", {})
reg = j("data/regimes/current.json", {})
as_of = etf.get("as_of", "—")

st.markdown(zt.boot([("SYS", "ONLINE"), ("DATA", as_of), ("MODULES", "21"), ("FREE_ONLY", "✓"),
                     ("LINK", "GITHUB-RELEASE")], "> zenith research terminal ·"), unsafe_allow_html=True)
st.markdown(zt.nav(["HOME", "MARKETS", "SIGNALS", "PORTFOLIO", "OPTIONS", "RESEARCH", "GAMES"], "HOME",
                   "▲ 3 ALERTS · ⌘K"), unsafe_allow_html=True)

movers = sorted(rows, key=lambda r: -abs(r.get("composite") or 0))[:40]
st.markdown(zt.tape([(r["ticker"], float(r.get("composite") or 0)) for r in movers]), unsafe_allow_html=True)

# ---------------------------------------------------------------- row 1 ------
c1, c2, c3 = st.columns([1.25, 1, 1])
with c1:
    with st.container(border=True):
        st.markdown(zt.titlebar("MARKET STATUS", T.phosphor, "daily close"), unsafe_allow_html=True)
        if m is not None:
            i = len(m) - 1
            lines = []
            for t in ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "XLE", "SMH"):
                if t in m.data:
                    c = m.data[t]["close"]
                    ch = c[i] / c[i - 1] - 1
                    col = T.pos if ch >= 0 else T.neg
                    lines.append(f'<div style="display:grid; grid-template-columns: 4rem 1fr 5rem 4rem; '
                                 f'font-size:0.92rem; border-bottom:1px dashed {T.grid}; padding:0.12rem 0;">'
                                 f'<b style="color:{T.text}">{t}</b><span style="color:{T.muted}">{m.meta[t]["name"][:22]}</span>'
                                 f'<span style="text-align:right">{c[i]:,.2f}</span>'
                                 f'<span style="text-align:right; color:{col}">{ch:+.2%}</span></div>')
            st.markdown("".join(lines), unsafe_allow_html=True)
            spy = pd.DataFrame({"d": m.dates[-130:], "SPY": m.data["SPY"]["close"][-130:]})
            spark = alt.Chart(spy).mark_area(line={"color": T.phosphor, "strokeWidth": 1.5},
                                              color=alt.Gradient(gradient="linear", stops=[
                                                  alt.GradientStop(color=zt._rgba(T.phosphor, 0.0), offset=0),
                                                  alt.GradientStop(color=zt._rgba(T.phosphor, 0.35), offset=1)],
                                                  x1=1, x2=1, y1=1, y2=0)).encode(
                x=alt.X("d:T", axis=None), y=alt.Y("SPY:Q", axis=None, scale=alt.Scale(zero=False))
            ).properties(height=70)
            st.altair_chart(spark, use_container_width=True)
with c2:
    with st.container(border=True):
        st.markdown(zt.titlebar("THREAT LEVEL", T.amber, "volatility"), unsafe_allow_html=True)
        if m is not None:
            vix = m.vix[-1] * 100
            v1y = m.vix[-252:] * 100
            pct = float((v1y < vix).mean())
            col = T.pos if vix < 18 else (T.amber if vix < 28 else T.neg)
            st.markdown(zt.counters([
                {"label": "VIX", "value": f"{vix:.1f}", "sub": f"1y pctile {pct:.0%}", "color": col},
                {"label": "SPY IV RANK", "value": f"{m.data['SPY']['iv_rank'][-1] * 100:.0f}",
                 "sub": "trailing year", "color": T.cyan},
            ]), unsafe_allow_html=True)
            vx = pd.DataFrame({"d": m.dates[-252:], "VIX": v1y})
            st.altair_chart(alt.Chart(vx).mark_line(color=T.amber, strokeWidth=1.4, interpolate="step-after")
                            .encode(x=alt.X("d:T", title=None), y=alt.Y("VIX:Q", title=None))
                            .properties(height=110), use_container_width=True)
with c3:
    with st.container(border=True):
        st.markdown(zt.titlebar("REGIME", T.cyan, reg.get("latest_month", "")), unsafe_allow_html=True)
        st.markdown(f'<div class="zt-label">growth × inflation quadrant</div>'
                    f'<div class="zt-big zt-glow" style="color:{T.cyan}; font-size:2.4rem;">'
                    f'{reg.get("regime", "—").upper()}</div>', unsafe_allow_html=True)
        st.markdown(zt.counters([
            {"label": "CONFIDENCE", "value": f"{reg.get('confidence', 0):.1f}", "sub": "of 5", "color": T.cyan},
            {"label": "STREAK", "value": f"{reg.get('streak_months', 0)}M", "sub": "months in regime",
             "color": T.mauve},
        ]), unsafe_allow_html=True)

# ---------------------------------------------------------------- row 2 ------
c4, c5 = st.columns([1.5, 1])
with c4:
    with st.container(border=True):
        st.markdown(zt.titlebar("SECTOR MOMENTUM MATRIX", T.phosphor, "russell 1000 · mean composite"),
                    unsafe_allow_html=True)
        cells = [{"name": k[:18], "value": v.get("mean", 0.0),
                  "sub": f"{v.get('pct_bullish', 0):.0%} bull · n{v.get('n', 0)}"}
                 for k, v in sorted(sectors.items(), key=lambda kv: -kv[1].get("mean", 0)) if k != "Unknown"]
        st.markdown(zt.cells(cells, cols=4, cap=4.0), unsafe_allow_html=True)
with c5:
    with st.container(border=True):
        st.markdown(zt.titlebar("CROSS-ASSET SIGNAL", T.magenta, "etf momentum by class"), unsafe_allow_html=True)
        st.markdown(zt.meters([(k, v.get("mean", 0.0)) for k, v in
                               sorted(classes.items(), key=lambda kv: -kv[1].get("mean", 0))
                               if k != "Unknown"], cap=5.0),
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- row 3 ------
c6, c7 = st.columns([1.6, 1])
with c6:
    with st.container(border=True):
        st.markdown(zt.titlebar("SPY vs MODELED IV / REALIZED", T.phosphor, "2y · theta run feed"),
                    unsafe_allow_html=True)
        if m is not None:
            n = 504
            d = m.data["SPY"]
            df = pd.DataFrame({"d": m.dates[-n:], "price": d["close"][-n:], "IV": d["iv30"][-n:],
                               "RV": d["rv21"][-n:]})
            px = alt.Chart(df).mark_line(color=T.phosphor, strokeWidth=1.8).encode(
                x=alt.X("d:T", title=None), y=alt.Y("price:Q", title="SPY", scale=alt.Scale(zero=False)))
            vol = alt.Chart(df.melt("d", ["IV", "RV"], "s", "v")).mark_line(strokeWidth=1.3).encode(
                x=alt.X("d:T", title=None), y=alt.Y("v:Q", title="VOL", axis=alt.Axis(format="%")),
                color=alt.Color("s:N", scale=alt.Scale(domain=["IV", "RV"], range=[T.amber, T.cyan]),
                                legend=alt.Legend(orient="top", title=None)))
            gap = alt.Chart(df).mark_area(opacity=0.18, color=T.amber).encode(x="d:T", y="IV:Q", y2="RV:Q")
            st.altair_chart(px.properties(height=190), use_container_width=True)
            st.altair_chart((gap + vol).properties(height=150), use_container_width=True)
with c7:
    with st.container(border=True):
        st.markdown(zt.titlebar("THETA RUN · LIVE BOOK", T.amber, "demo"), unsafe_allow_html=True)
        st.markdown(zt.counters([
            {"label": "NET LIQ", "value": "$104.2K", "sub": "+4.2% · day 63/126", "color": T.phosphor},
            {"label": "VS THE BOT", "value": "+1.8%", "sub": "bot $102.4K", "color": T.phosphor},
            {"label": "BP IN USE", "value": "31%", "sub": "guide 25–50%", "color": T.amber},
            {"label": "Θ / DAY", "value": "$96", "sub": "0.09% NLV", "color": T.cyan},
        ]), unsafe_allow_html=True)
        st.markdown(f'<div class="zt-label" style="margin-top:0.6rem;">coach</div>'
                    f'<div style="border-left:3px solid {T.amber}; padding:0.3rem 0.6rem; font-size:0.82rem;">'
                    f'<b style="color:{T.amber}">21 DTE · CLOSE OR ROLL</b><br>#3 IWM strangle is at 21 DTE. '
                    f'Close or roll to the next cycle rather than hold into expiration week.</div>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.button("+1 DAY", use_container_width=True)
        b2.button("▶▶ EVENT", type="primary", use_container_width=True)
        b3.button("CLOSE #3", use_container_width=True)

# ---------------------------------------------------------------- row 4 ------
with st.container(border=True):
    st.markdown(zt.titlebar("ETF MOMENTUM · TOP OF BOOK", T.cyan, f"{len(rows)} scored"), unsafe_allow_html=True)
    top = sorted(rows, key=lambda r: r.get("rank") or 9999)[:14]
    tdf = pd.DataFrame([{"rank": r.get("rank"), "ticker": r["ticker"], "name": r["name"][:38],
                         "class": r.get("asset_class"), "composite": r.get("composite"),
                         "pctile": r.get("pctile"), "state": r.get("state")} for r in top])
    t1, t2 = st.columns([1.3, 1])
    t1.dataframe(tdf, hide_index=True, use_container_width=True, height=420, column_config={
        "pctile": st.column_config.ProgressColumn("pctile", min_value=0, max_value=100, format="%d"),
        "composite": st.column_config.NumberColumn("composite", format="%+.2f")})
    fac = pd.DataFrame([{"ticker": r["ticker"], "factor": k, "score": v}
                        for r in top for k, v in (r.get("factor_scores") or {}).items()])
    if not fac.empty:
        hm = alt.Chart(fac).mark_rect(stroke=T.bg, strokeWidth=2).encode(
            x=alt.X("factor:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("ticker:N", title=None, sort=list(tdf["ticker"])),
            color=alt.Color("score:Q", scale=alt.Scale(range=[T.coral, "#3a1410", T.glass,
                                                                         "#0f5c33", T.phosphor],
                                                       domain=[-1, -0.3, 0, 0.3, 1], clamp=True), legend=None),
            tooltip=["ticker", "factor", alt.Tooltip("score:Q", format="+.2f")])
        txt = alt.Chart(fac).mark_text(font="VT323", fontSize=15).encode(
            x="factor:N", y=alt.Y("ticker:N", sort=list(tdf["ticker"])),
            text=alt.Text("score:Q", format="+.1f"),
            color=alt.condition("abs(datum.score) > 0.45", alt.value("#000"), alt.value(T.text)))
        t2.altair_chart((hm + txt).properties(height=420), use_container_width=True)

st.markdown(f'<div class="zt-status" style="margin-top:1rem;">> end of transmission · '
            f'<b>ZENITH//TERM prototype</b> · real data, not wired into the app yet</div>',
            unsafe_allow_html=True)

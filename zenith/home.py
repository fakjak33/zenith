"""DASHBOARD: the ZENITH home page (ui-3 layout).

Hero band with the horizon art, then panels that summarise every engine and link
into it: market overview, momentum signals, trend following, macro regime,
research feed, THETA RUN, ideas & alerts, holdings and module status. Reads only
committed artefacts and the THETA RUN market table; every panel degrades to a
quiet empty state when its data is missing.
"""

from __future__ import annotations

import html
import json
from datetime import date

import pandas as pd
import streamlit as st

from . import art, nav, store
from . import ui_kit as uk
from .chrome import market, topbar
from .config import DATA_DIR, THEME as T
from .viz.radial import regime_quadrant

ORIENTATION = (
    "**Zenith is many tools in one, built from free data.** The sidebar groups them: "
    "**MARKETS** (Today's research feed, the Weekly Brief, macro Regimes, Night & Day), "
    "**SIGNALS** (CAS model monitor, stock and ETF Momentum, seven-speed Trend Following, Factor "
    "Momentum, rated Edge Screens, PEAD, PRETOM, Ideas), **PORTFOLIO** (Clean Beta, fund Holdings), "
    "**OPTIONS** (the evidence-graded Options Vault and the THETA RUN game), **GAMES** (Ephemeris) "
    "and **RESEARCH** (the Index directory, Archive, Sources). Every engine shows an evidence "
    "rating (A/B/C); every number is decision-support, not investment advice.")


def _j(rel: str, default=None):
    try:
        return json.loads((DATA_DIR / rel).read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_data(ttl=600, show_spinner=False)
def _data() -> dict:
    mom = _j("mom/scores_latest.json", {}) or {}
    etf = _j("etfmom/scores_latest.json", {}) or {}
    tr = _j("trend/etfs/latest.json", {}) or {}
    trs = _j("trend/stocks/latest.json", {}) or {}
    reg = _j("regimes/current.json", {}) or {}
    tl = (_j("regimes/timeline.json", {}) or {}).get("months", [])
    brief = _j("brief/brief.json", {}) or {}
    ideas = _j("ideas/ideas_latest.json", {}) or {}
    hold = _j("holdings/dbmf/latest.json", {}) or {}
    return {"mom": mom.get("rows", []), "etf": etf.get("rows", []), "trend": tr.get("rows", []),
            "trend_stocks": trs.get("rows", []), "speeds": tr.get("speeds", []), "regime": reg,
            "trail": [(m.get("growth_z"), m.get("infl_z")) for m in tl[-24:]],
            "talking": brief.get("talking_points") or [], "ideas": ideas,
            "holdings": hold.get("summary") or {}, "as_of": mom.get("as_of") or etf.get("as_of") or "—"}


def _link(slug: str, label: str = "OPEN ▸") -> None:
    if slug in nav.PAGES:
        st.page_link(nav.PAGES[slug], label=label)


def _bar(v: float, cap: float, w: int = 90) -> str:
    x = max(-1.0, min(1.0, v / cap if cap else 0))
    col = T.teal if v >= 0 else T.coral
    return (f'<span style="display:inline-block; width:{w}px; height:7px; background:{T.grid}; position:relative;">'
            f'<span style="position:absolute; left:0; top:0; bottom:0; width:{abs(x) * 100:.0f}%; '
            f'background:{col};"></span></span>')


def _hero(d: dict) -> None:
    tp = d["talking"][0] if d["talking"] else ""
    if isinstance(tp, dict):
        tp = tp.get("text") or tp.get("point") or ""
    st.markdown(
        f'<div class="z-hero">{art.horizon(1400, 220)}'
        f'<div class="z-hero-copy"><div class="k">MARKET INTELLIGENCE · {html.escape(d["as_of"])}</div>'
        f'<div class="h">REAL DATA.<br>DEEPER INSIGHT.<br>BETTER PORTFOLIOS.</div>'
        f'<div class="p">{html.escape(str(tp)[:220])}</div></div></div>', unsafe_allow_html=True)


def _market_panel() -> None:
    m = market()
    with uk.panel("MARKET OVERVIEW", T.teal, "daily close"):
        if m is None:
            st.markdown(uk.empty_state(art.orbits(90), "Market feed loads after the nightly THETA build."),
                        unsafe_allow_html=True)
            return
        i = len(m) - 1
        rows = []
        for t in ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "EEM"):
            if t not in m.data:
                continue
            c = m.data[t]["close"]
            ch = c[i] / c[i - 1] - 1
            col = T.teal if ch >= 0 else T.coral
            rows.append(f'<div style="display:grid; grid-template-columns:3.4rem 1fr 66px 4.6rem 4.4rem; gap:0.5rem; '
                        f'align-items:center; font-size:0.74rem; padding:0.2rem 0; border-bottom:1px solid {T.grid};">'
                        f'<b style="font-weight:600;">{t}</b><span style="color:{T.muted}; overflow:hidden; '
                        f'white-space:nowrap; text-overflow:ellipsis;">{html.escape(m.meta[t]["name"][:20])}</span>'
                        f'{uk.sparkline_svg(c[i - 40:i + 1], col, 66, 16)}'
                        f'<span style="text-align:right;">{c[i]:,.2f}</span>'
                        f'<span style="text-align:right; color:{col};">{ch:+.2%}</span></div>')
        v = m.vix
        rows.append(f'<div style="display:grid; grid-template-columns:3.4rem 1fr 66px 4.6rem 4.4rem; gap:0.5rem; '
                    f'align-items:center; font-size:0.74rem; padding:0.2rem 0;"><b style="font-weight:600;">VIX</b>'
                    f'<span style="color:{T.muted};">volatility</span>'
                    f'{uk.sparkline_svg(v[i - 40:i + 1] * 100, T.amber, 66, 16)}'
                    f'<span style="text-align:right;">{v[i] * 100:.2f}</span>'
                    f'<span style="text-align:right; color:{T.amber};">{v[i] / v[i - 1] - 1:+.2%}</span></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)


def _momentum_panel(d: dict) -> None:
    with uk.panel("MOMENTUM SIGNALS", T.teal, "composite · −20..+20"):
        kind = st.segmented_control("Universe", ["EQUITIES", "ETFS"], default="EQUITIES",
                                    key="home_mom_u", label_visibility="collapsed")
        rows = d["mom"] if kind != "ETFS" else d["etf"]
        top = sorted(rows, key=lambda r: -(r.get("composite") or -99))[:10]
        if not top:
            st.caption("No momentum data yet.")
            return
        facs = ["ts", "xsec", "breakout", "speed", "strength", "mvt"]
        head = "".join(f'<span style="width:12px; text-align:center;">{f[0].upper()}</span>' for f in facs)
        out = [f'<div style="display:grid; grid-template-columns:3.6rem 3.2rem 1fr auto; gap:0.5rem; font-size:0.58rem; '
               f'letter-spacing:0.14em; color:{T.faint}; padding-bottom:0.3rem;"><span>TICKER</span><span>SCORE</span>'
               f'<span></span><span style="display:flex; gap:3px;">{head}</span></div>']
        for r in top:
            fs = r.get("factor_scores") or {}
            dots = "".join(f'<i title="{f} {fs.get(f, 0):+.2f}" style="width:12px; height:9px; display:block; '
                           f'background:{uk.rgba(T.teal if (fs.get(f) or 0) >= 0 else T.coral, 0.2 + 0.8 * min(1, abs(fs.get(f) or 0)))};"></i>'
                           for f in facs)
            out.append(f'<div style="display:grid; grid-template-columns:3.6rem 3.2rem 1fr auto; gap:0.5rem; '
                       f'align-items:center; font-size:0.74rem; padding:0.16rem 0;"><b style="font-weight:600;">'
                       f'{html.escape(r["ticker"])}</b><span style="color:{T.teal};">{r.get("composite", 0):+.1f}</span>'
                       f'{_bar(r.get("composite", 0), 20, 120)}<span style="display:flex; gap:3px;">{dots}</span></div>')
        st.markdown("".join(out), unsafe_allow_html=True)
        _link("momentum" if kind != "ETFS" else "etf-momentum")


def _trend_panel(d: dict) -> None:
    with uk.panel("TREND FOLLOWING", T.amber, "7-speed EWMAC"):
        rows = d["trend"]
        speeds = [s.replace("_", "/") for s in d["speeds"]] or [f"S{k}" for k in range(7)]
        if not rows:
            st.caption("No trend data yet.")
            return
        import altair as alt
        agg = pd.DataFrame([r["forecasts"] for r in rows if r.get("forecasts")], columns=speeds).mean()
        df = pd.DataFrame({"speed": speeds, "forecast": agg.values})
        ch = alt.Chart(df).mark_bar(size=16).encode(
            x=alt.X("speed:N", sort=speeds, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("forecast:Q", title="avg forecast", scale=alt.Scale(domain=[-20, 20])),
            color=alt.Color("speed:N", sort=speeds, legend=None,
                            scale=alt.Scale(range=[T.teal, T.cyan, T.ice, T.violet, T.magenta, T.amber, T.coral])),
            tooltip=["speed", alt.Tooltip("forecast:Q", format="+.1f")]).properties(height=150)
        st.altair_chart(ch, use_container_width=True)
        pick = {r["ticker"]: r for r in rows}
        out = []
        for t in ("SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "EEM"):
            r = pick.get(t)
            if r:
                out.append(f'<div style="display:grid; grid-template-columns:3.2rem 3.2rem 1fr; gap:0.5rem; '
                           f'align-items:center; font-size:0.72rem; padding:0.12rem 0;"><b style="font-weight:600;">{t}</b>'
                           f'<span style="color:{T.teal if r["score"] >= 0 else T.coral};">{r["score"]:+.1f}</span>'
                           f'{_bar(r["score"], 20, 140)}</div>')
        st.markdown("".join(out), unsafe_allow_html=True)
        _link("trend")


def _regime_panel(d: dict) -> None:
    reg = d["regime"]
    with uk.panel("MACRO REGIME", T.violet, str(reg.get("latest_month", ""))):
        trail = [(g, i) for g, i in d["trail"] if g is not None and i is not None]
        st.markdown(regime_quadrant(trail, 300), unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.58rem; letter-spacing:0.2em; color:{T.faint};">CURRENT REGIME</div>'
                    f'<div style="font-size:1.05rem; font-weight:300; letter-spacing:0.08em;">'
                    f'{html.escape(str(reg.get("regime", "—")).upper())}</div>'
                    f'<div style="font-size:0.7rem; color:{T.muted};">confidence {reg.get("confidence", 0):.1f}/5 · '
                    f'{reg.get("streak_months", 0)} months · trail = last 24 months</div>', unsafe_allow_html=True)
        _link("regimes")


def _feed_panel() -> None:
    with uk.panel("MARKET INSIGHTS & RESEARCH", T.cyan, "today"):
        items = store.load_latest()
        if not items:
            dates = store.archive_dates()
            items = store.load_archive(dates[0]) if dates else []
        items = [i for i in items if i.get("category") != "news"][:6] or items[:6]
        if not items:
            st.caption("No research items yet.")
        out = []
        for it in items:
            c = T.section_colors[hash(it["source"]) % len(T.section_colors)]
            out.append(f'<div style="display:grid; grid-template-columns:4px 1fr; gap:0.7rem; padding:0.35rem 0; '
                       f'border-bottom:1px solid {T.grid};"><span style="background:{c};"></span><div>'
                       f'<a href="{html.escape(it["link"])}" target="_blank" style="color:{T.text}; text-decoration:none; '
                       f'font-size:0.8rem;">{html.escape(it["title"][:96])}</a>'
                       f'<div style="font-size:0.6rem; letter-spacing:0.14em; color:{T.faint};">'
                       f'{html.escape(it["source"].upper())} · {html.escape(it.get("category", "").upper())}</div></div></div>')
        st.markdown("".join(out), unsafe_allow_html=True)
        _link("today", "ALL RESEARCH ▸")


def _theta_panel() -> None:
    with uk.panel("SIMULATION · THETA RUN", T.amber, "options game"):
        ch = (_j("theta/challenges.json", {}) or {}).get("challenges", [])
        pick = ch[date.today().toordinal() % len(ch)] if ch else None
        st.markdown(f'<div style="display:flex; gap:1rem; align-items:center;">{art.orbits(120, seed=9)}<div>'
                    f'<div style="font-size:0.58rem; letter-spacing:0.2em; color:{T.faint};">TODAY\'S CHALLENGE</div>'
                    f'<div style="font-size:1.05rem; font-weight:300; letter-spacing:0.08em;">'
                    f'{html.escape(pick["title"]) if pick else "QUICK RUN"}</div>'
                    f'<div style="font-size:0.72rem; color:{T.muted}; line-height:1.5;">'
                    f'{html.escape(pick["brief"][:140]) if pick else ""}</div></div></div>', unsafe_allow_html=True)
        _link("theta-run", "▶ PLAY THETA RUN")


def _ideas_panel(d: dict) -> None:
    with uk.panel("IDEAS & ALERTS", T.coral, "daily"):
        ideas = d["ideas"]
        rows = (ideas.get("buy") or [])[:3] + (ideas.get("sell") or [])[:3]
        if not rows:
            st.markdown(uk.empty_state(art.grid_globe(84), "No high-conviction ideas cleared the bar today — "
                                       "a quiet tape is an answer too."), unsafe_allow_html=True)
        else:
            out = []
            for r in rows:
                side = str(r.get("side") or r.get("direction") or "").upper()
                c = T.teal if side.startswith("B") or side.startswith("L") else T.coral
                out.append(f'<div style="display:flex; justify-content:space-between; font-size:0.76rem; padding:0.25rem 0; '
                           f'border-bottom:1px solid {T.grid};"><span><b>{html.escape(str(r.get("ticker", "")))}</b> '
                           f'<span style="color:{T.muted};">{html.escape(str(r.get("thesis") or r.get("headline") or "")[:46])}</span></span>'
                           f'<span style="color:{c}; border:1px solid {c}; padding:0 0.35rem; font-size:0.6rem;">{html.escape(side or "IDEA")}</span></div>')
            st.markdown("".join(out), unsafe_allow_html=True)
        _link("ideas")


def _holdings_panel(d: dict) -> None:
    with uk.panel("PORTFOLIO · DBMF EXPOSURE", T.mint, "gross by asset class"):
        bac = (d["holdings"] or {}).get("by_asset_class") or []
        if not bac:
            st.caption("No holdings snapshot yet.")
            return
        cols = [T.teal, T.amber, T.violet, T.cyan, T.coral, T.mint]
        parts = [(b["asset_class"].upper(), abs(b.get("gross", 0)), cols[k % len(cols)]) for k, b in enumerate(bac)]
        net = d["holdings"].get("net", 0)
        st.markdown(uk.ring_donut(parts, 150, f"{d['holdings'].get('gross', 0):.0%}", "GROSS"), unsafe_allow_html=True)
        st.caption(f"net {net:+.0%} of NAV · {d['holdings'].get('n_positions', 0)} positions")
        _link("holdings")


def _status_panel() -> None:
    with uk.panel("MODULE STATUS", T.cyan, "every engine"):
        chips, failed = nav.badges()
        for f, chip in chips:
            a, b = st.columns([6, 1])
            a.markdown(chip, unsafe_allow_html=True)
            with b:
                _link(f.slug, "▸")
        if failed:
            st.caption(f"Status unavailable this run: {', '.join(failed)}.")


def render() -> None:
    topbar("DASHBOARD")
    d = _data()
    _hero(d)
    c1, c2 = st.columns([1.05, 1.95])
    with c1:
        _market_panel()
    with c2:
        a, b = st.columns(2)
        with a:
            _momentum_panel(d)
        with b:
            _trend_panel(d)
    c3, c4, c5 = st.columns([1.25, 1, 1.15])
    with c3:
        _feed_panel()
    with c4:
        _regime_panel(d)
    with c5:
        _theta_panel()
        _holdings_panel(d)
    c6, c7 = st.columns([1, 1.4])
    with c6:
        _ideas_panel(d)
    with c7:
        _status_panel()
    with st.expander("WHAT IS ZENITH? · a 20-second orientation"):
        st.markdown(ORIENTATION)

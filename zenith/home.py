"""HOME: the ZENITH//TERM dashboard -- market status, volatility, regime, the
momentum matrix, every module's status chip and quick launch into the games.

Reads only committed artefacts and the THETA RUN market table; every panel
degrades to a caption when its data is missing.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from . import nav
from . import ui_term as zt
from .config import DATA_DIR, THEME as T

ORIENTATION = (
        "**Zenith is many tools in one, all built from free data:**\n"
        "1. **Weekly Brief** — a concise, visual market read: what moved, sectors & industries, "
        "rates & fixed income, earnings, factor rotation, and plain-English talking points.\n"
        "2. **CAS** — a *model & signal monitor*: many models score every ETF **buy → neutral → sell**, "
        "combined into a consensus, with factor rotation, per-ticker detail, and a signal hit-rate.\n"
        "3. **PRETOM** — a monthly *short screener* built on the intramonth momentum cycle: "
        "it ranks the Russell 1000 by distance below the 52-week high and tracks a short-bias "
        "basket through the pre-month-end selling window, plus a turn-of-the-month evidence "
        "monitor (the classic long side, rebuilt nightly from SPY since 1993).\n"
        "4. **PEAD** — a daily *post-earnings drift screener*: Russell 1000 reporters that beat "
        "(missed) **and** closed up (down) vs the market get a 0-100 signal score built from five "
        "published anomalies, with every pick tracked at +5/+20/+60 days and before its next report — "
        "plus the *announcement premium*: upcoming scheduled reporters and the tracked long-side "
        "premium around every announcement.\n"
        "5. **FACTOR MOMENTUM** — Gupta & Kelly's *'Factor Momentum Everywhere'* run live, three "
        "ways: tradable factor ETFs, Man-style 13-style composites, and the academic 65-factor "
        "set — each with time-series (TSFM) and cross-sectional (CSFM) 1-1 models, tracked "
        "monthly.\n"
        "6. **EDGE SCREENS** — four *rated* cross-sectional screens (option IV spread, analyst "
        "revisions, short interest / days-to-cover, lottery MAXβ), each with an honest "
        "evidence tier and long/short ranked ideas.\n"
        "7. **NIGHT & DAY** — *overnight vs intraday* return decomposition: where the return is "
        "really made, plus an overnight-momentum screen (Lou-Polk-Skouras).\n"
        "8. **HOLDINGS** — *fund position intelligence*: a systematic fund's own daily portfolio "
        "disclosure, stored as dated snapshots so you can see what it holds today, what changed "
        "since yesterday, and how the whole book has evolved. First fund: **DBMF**, the managed-"
        "futures replicator.\n"
        "9. **MOMENTUM** — a Russell 1000 *multi-factor stock momentum engine*: six documented "
        "momentum dimensions (time-series, breakout, cross-sectional, trend speed / GMMA, momentum "
        "strength, multivariate trend) blended into a transparent -20..+20 composite, with a "
        "full-index heatmap, sortable rankings, sector breadth, a pairwise relative-strength engine "
        "across the Russell 1000 and a broad ETF universe, and a per-stock drill-down explaining "
        "exactly why each score landed where it did.\n"
        "10. **TREND FOLLOWING** — a *multi-speed systematic trend dashboard*: seven EWMAC speeds "
        "(2/8 through 128/512 days), each a volatility-normalized forecast, averaged with equal weight "
        "into a -20..+20 Trend Score for every stock in MOMENTUM's universe and every ETF in ETF "
        "MOMENTUM's, showing whether a trend runs across every horizon or only at one. It includes "
        "trigger and crossover tracking, a trend-structure map, and each asset's full daily signal "
        "history.\n"
        "11. **CLEAN BETA** — a *quality high-beta screener + hedge monitor*: ranks the Russell 1000 "
        "by a Quality-Beta Score that rewards beta coming from market correlation (Welch bswa beta, "
        "high ρ, low idiosyncratic vol, no event noise, sound fundamentals) rather than noise, builds a "
        "diversified 30-50 name basket, and monitors the cheapest ways to hedge it — trend gate, vol "
        "target, volatility risk premium, BTAL regime and an advisory index put-spread ladder.\n"
        "12. **IDEAS** — a *discretionary-systematic opportunity engine*: fuses MOMENTUM/EDGE/PEAD/"
        "CAS/valuation signals into a daily ranked BUY/SELL idea list, scoring conviction and "
        "unusualness SEPARATELY so genuinely idiosyncratic setups outrank merely-popular ones, with "
        "a deterministic thesis (bull/bear case, idiosyncratic risk, invalidation) for every idea.\n"
        "13. **REGIMES** — a *macro regime intelligence & early-warning system*: classifies the "
        "current growth/inflation quadrant from ~45 point-in-time FRED indicators (with a "
        "persistence requirement so one noisy release can't flip the headline), scores six "
        "secondary regimes (monetary, liquidity, credit, financial conditions, dollar, "
        "volatility) alongside it, and reconstructs the full historical regime timeline back "
        "to 1990.\n"
        "14. **INDEX** — the *Master List*: a curated directory of the financial information "
        "ecosystem itself — institutions, people, academic sources, podcasts and tools. "
        "Deduplicated, classified against an extensible taxonomy, link-verified, and "
        "extended by harvesting the full archives of 14 finance podcasts into a guest "
        "database, so it answers *who is doing interesting work, where do they work, and "
        "where do I read them*. Includes a traversable knowledge graph, a discovery "
        "surface for names your own list does not contain, and a CSV/Excel/JSON export.\n"
        "15. **EPHEMERIS** — a *blind-chart tape-reading trainer*: a random historical chart with "
        "ticker, dates and price level hidden; call UP or DOWN over the next N candles, watch the "
        "future replay, and every call is scored against the base rate and a trend rule on the "
        "same charts. Profiles are one click: type a handle, bookmark the URL.\n"
        "16. **OPTIONS VAULT** — an *evidence-graded options reference*: 30 strategy cards (when, "
        "setup, manage, exit, with payoff diagrams), 15 edges, decision trees for opening, managing, "
        "rolling and taking losses, and a sleeve-sizing calculator, distilled from tastylive research "
        "and checked against the literature and our own free-data reproductions. Every number links "
        "to its episode, page, paper or test.\n"
        "17. **THETA RUN** — a *gamified options-portfolio speedrun*: screen real historical markets "
        "by IV, IV rank and earnings, open tastytrade-style trades, then manage the book day by day "
        "(21 DTE, profit targets, rolls, assignment) against buy-and-hold SPY and a mechanical "
        "tastytrade bot on the same tape, coached by OPTIONS VAULT. Crisis challenges, XP and "
        "achievements; real prices, modeled option prices.\n"
        "18. **Today / Archive** — a *research & insights aggregator* pulling institutional & academic "
        "sources (Fed, NBER, BIS, quant desks, journals) into one deduped feed.\n\n"
        "Every feature shows an **evidence-strength rating (A/B/C)** so you can tell the strong "
        "signals from the weak ones. Every number is **decision-support, not investment advice**. "
        "Hover any **?** for a definition."
)


def _j(rel: str, default=None):
    try:
        return json.loads((DATA_DIR / rel).read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def _market():
    try:
        from .theta.market import Market
        return Market.load()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _data() -> dict:
    etf = _j("etfmom/scores_latest.json", {}) or {}
    return {"etf": etf.get("rows", []), "as_of": etf.get("as_of", "—"),
            "sectors": (_j("mom/sectors.json", {}) or {}).get("sectors", {}),
            "classes": (_j("etfmom/categories.json", {}) or {}).get("asset_classes", {}),
            "regime": _j("regimes/current.json", {}) or {}}


def _window(title: str, color: str, tag: str = ""):
    box = st.container(border=True)
    box.markdown(zt.titlebar(title, color, tag), unsafe_allow_html=True)
    return box


def _link(slug: str, label: str) -> None:
    if slug in nav.PAGES:
        st.page_link(nav.PAGES[slug], label=label)


def render() -> None:
    d = _data()
    m = _market()
    rows = d["etf"]
    if rows:
        movers = sorted(rows, key=lambda r: -abs(r.get("composite") or 0))[:40]
        st.markdown(zt.tape([(r["ticker"], float(r.get("composite") or 0)) for r in movers]),
                    unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1, _window("MARKET STATUS", T.phosphor, "daily close"):
        if m is None:
            st.caption("Market feed not loaded yet.")
        else:
            i = len(m) - 1
            lines = []
            for t in ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "XLE", "SMH"):
                if t not in m.data:
                    continue
                c = m.data[t]["close"]
                ch = c[i] / c[i - 1] - 1
                col = T.pos if ch >= 0 else T.neg
                lines.append(f'<div style="display:grid; grid-template-columns: 4rem 1fr 5rem 4.4rem; '
                             f'font-size:0.92rem; border-bottom:1px dashed {T.grid}; padding:0.12rem 0;">'
                             f'<b style="color:{T.text}">{t}</b><span style="color:{T.muted}">'
                             f'{m.meta[t]["name"][:22]}</span><span style="text-align:right">{c[i]:,.2f}</span>'
                             f'<span style="text-align:right; color:{col}">{ch:+.2%}</span></div>')
            st.markdown("".join(lines), unsafe_allow_html=True)
            spy = pd.DataFrame({"d": m.dates[-130:], "SPY": m.data["SPY"]["close"][-130:]})
            st.altair_chart(alt.Chart(spy).mark_area(
                line={"color": T.phosphor, "strokeWidth": 1.5},
                color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0, stops=[
                    alt.GradientStop(color=T.bg, offset=0), alt.GradientStop(color=T.phosphor_dim, offset=1)])
            ).encode(x=alt.X("d:T", axis=None), y=alt.Y("SPY:Q", axis=None, scale=alt.Scale(zero=False))
                     ).properties(height=70), use_container_width=True)
    with c2, _window("THREAT LEVEL", T.amber, "volatility"):
        if m is None:
            st.caption("Market feed not loaded yet.")
        else:
            vix = m.vix[-1] * 100
            v1y = m.vix[-252:] * 100
            col = T.pos if vix < 18 else (T.amber if vix < 28 else T.neg)
            st.markdown(zt.counters([
                {"label": "VIX", "value": f"{vix:.1f}", "sub": f"1y pctile {(v1y < vix).mean():.0%}",
                 "color": col},
                {"label": "SPY IV RANK", "value": f"{m.data['SPY']['iv_rank'][-1] * 100:.0f}",
                 "sub": "trailing year", "color": T.cyan}]), unsafe_allow_html=True)
            vx = pd.DataFrame({"d": m.dates[-252:], "VIX": v1y})
            st.altair_chart(alt.Chart(vx).mark_line(color=T.amber, strokeWidth=1.4, interpolate="step-after")
                            .encode(x=alt.X("d:T", title=None), y=alt.Y("VIX:Q", title=None))
                            .properties(height=110), use_container_width=True)
    with c3, _window("REGIME", T.cyan, d["regime"].get("latest_month", "")):
        reg = d["regime"]
        st.markdown(f'<div class="zt-label">growth × inflation quadrant</div>'
                    f'<div class="zt-big zt-glow" style="color:{T.cyan}; font-size:2.3rem;">'
                    f'{str(reg.get("regime", "—")).upper()}</div>', unsafe_allow_html=True)
        st.markdown(zt.counters([
            {"label": "CONFIDENCE", "value": f"{reg.get('confidence', 0):.1f}", "sub": "of 5", "color": T.cyan},
            {"label": "STREAK", "value": f"{reg.get('streak_months', 0)}M", "sub": "months in regime",
             "color": T.mauve}]), unsafe_allow_html=True)
        _link("regimes", "OPEN REGIMES ▸")

    c4, c5 = st.columns([1.5, 1])
    with c4, _window("SECTOR MOMENTUM MATRIX", T.phosphor, "russell 1000 · mean composite"):
        cells = [{"name": k[:18], "value": v.get("mean", 0.0),
                  "sub": f"{v.get('pct_bullish', 0):.0%} bull · n{v.get('n', 0)}"}
                 for k, v in sorted(d["sectors"].items(), key=lambda kv: -kv[1].get("mean", 0))
                 if k != "Unknown"]
        if cells:
            st.markdown(zt.cells(cells, cols=4, cap=4.0), unsafe_allow_html=True)
        else:
            st.caption("No momentum data yet.")
        _link("momentum", "OPEN MOMENTUM ▸")
    with c5, _window("CROSS-ASSET SIGNAL", T.magenta, "etf momentum by class"):
        st.markdown(zt.meters([(k, v.get("mean", 0.0)) for k, v in
                               sorted(d["classes"].items(), key=lambda kv: -kv[1].get("mean", 0))
                               if k != "Unknown"], cap=5.0), unsafe_allow_html=True)
        _link("etf-momentum", "OPEN ETF MOMENTUM ▸")

    c6, c7 = st.columns([1.6, 1])
    with c6, _window("MODULE STATUS", T.cyan, "every page, one line"):
        chips, failed = nav.badges()
        for f, chip in chips:
            a, b = st.columns([5, 1])
            a.markdown(chip, unsafe_allow_html=True)
            if f.slug in nav.PAGES:
                b.page_link(nav.PAGES[f.slug], label="OPEN ▸")
        if failed:
            st.caption(f"Status unavailable this run: {', '.join(failed)} (open the page directly).")
    with c7, _window("PLAY", T.amber, "games"):
        st.markdown(f'<div style="font-size:0.85rem; line-height:1.45;">'
                    f'<b style="color:{T.amber}">THETA RUN</b> — run an options book through Lehman, '
                    f'Volmageddon or COVID against a mechanical tastytrade bot.<br><br>'
                    f'<b style="color:{T.phosphor}">EPHEMERIS</b> — call blind charts up or down against '
                    f'the base rate.</div>', unsafe_allow_html=True)
        _link("theta-run", "▶ START THETA RUN")
        _link("ephemeris", "▶ PLAY EPHEMERIS")
        _link("options-vault", "▣ OPEN THE OPTIONS VAULT")

    with st.expander("What is Zenith?  ·  a 20-second orientation"):
        st.markdown(ORIENTATION)

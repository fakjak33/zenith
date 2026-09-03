"""ZENITH — Streamlit viewer.   Run:  streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from zenith import store
from zenith.sources import SOURCES
from zenith.auth import require_password
from zenith.ui_theme import CSS, BANNER, section, card_html

_FAVICON = Path(__file__).parent / "assets" / "favicon.png"
_PAGE_ICON = str(_FAVICON) if _FAVICON.exists() else "▲"
st.set_page_config(page_title="ZENITH // research", layout="wide", page_icon=_PAGE_ICON)
st.markdown(CSS, unsafe_allow_html=True)
require_password()
st.markdown(BANNER, unsafe_allow_html=True)

with st.expander("What is Zenith?  ·  a 20-second orientation"):
    st.markdown(
        "**Zenith is six tools in one, all built from free data:**\n"
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
        "10. **IDEAS** — a *discretionary-systematic opportunity engine*: fuses MOMENTUM/EDGE/PEAD/"
        "CAS/valuation signals into a daily ranked BUY/SELL idea list, scoring conviction and "
        "unusualness SEPARATELY so genuinely idiosyncratic setups outrank merely-popular ones, with "
        "a deterministic thesis (bull/bear case, idiosyncratic risk, invalidation) for every idea.\n"
        "11. **REGIMES** — a *macro regime intelligence & early-warning system*: classifies the "
        "current growth/inflation quadrant from ~45 point-in-time FRED indicators (with a "
        "persistence requirement so one noisy release can't flip the headline), scores six "
        "secondary regimes (monetary, liquidity, credit, financial conditions, dollar, "
        "volatility) alongside it, and reconstructs the full historical regime timeline back "
        "to 1990.\n"
        "12. **INDEX** — the *Master List*: a curated directory of the financial information "
        "ecosystem itself — institutions, people, academic sources, podcasts and tools. "
        "Deduplicated, classified against an extensible taxonomy, link-verified, and "
        "extended by harvesting the full archives of 14 finance podcasts into a guest "
        "database, so it answers *who is doing interesting work, where do they work, and "
        "where do I read them*. Includes a traversable knowledge graph, a discovery "
        "surface for names your own list does not contain, and a CSV/Excel/JSON export.\n"
        "13. **Today / Archive** — a *research & insights aggregator* pulling institutional & academic "
        "sources (Fed, NBER, BIS, quant desks, journals) into one deduped feed.\n\n"
        "Every feature shows an **evidence-strength rating (A/B/C)** so you can tell the strong "
        "signals from the weak ones. Every number is **decision-support, not investment advice**. "
        "Hover any **?** for a definition.")


def render_items(items, empty="Nothing here yet."):
    if not items:
        st.caption(empty)
        return
    html = "".join(card_html(it) for it in items)
    st.markdown(html, unsafe_allow_html=True)


VIEWS = ["Everything", "Insights — text-only", "Insights — charts & visuals",
         "Research & papers", "News"]


def insights_research_news(items, key_prefix):
    """Render the standard Insights(text/visual) + Research + News layout,
    with a 'View' dropdown to jump straight to one section."""
    insights = [i for i in items if i["category"] == "insight"]
    research = [i for i in items if i["category"] == "research"]
    news = [i for i in items if i["category"] == "news"]

    # controls: a sort/view dropdown + an optional source filter, side by side
    c1, c2 = st.columns([1, 2])
    view = c1.selectbox("View", VIEWS, index=0, key=f"{key_prefix}_view")
    srcs = sorted({i["source"] for i in items})
    chosen = c2.multiselect("Filter sources", srcs, default=[], key=f"{key_prefix}_flt")
    if chosen:
        insights = [i for i in insights if i["source"] in chosen]
        research = [i for i in research if i["source"] in chosen]
        news = [i for i in news if i["source"] in chosen]

    text_only = [i for i in insights if i["visual"] == "text"]
    visual = [i for i in insights if i["visual"] == "visual"]

    show_all = view == "Everything"
    if show_all or view == "Insights — text-only":
        st.markdown(section(f"Insights — text-only ({len(text_only)})", 0), unsafe_allow_html=True)
        render_items(text_only, "No text-only insights.")
    if show_all or view == "Insights — charts & visuals":
        st.markdown(section(f"Insights — charts / tables / visuals ({len(visual)})", 2),
                    unsafe_allow_html=True)
        render_items(visual, "No visual insights.")
    if show_all or view == "Research & papers":
        st.markdown(section(f"Research & working papers ({len(research)})", 4),
                    unsafe_allow_html=True)
        render_items(research, "No research items.")
    # News stays de-emphasized: collapsed expander when browsing everything,
    # but shown in full if the user explicitly selects the News view.
    if view == "News":
        st.markdown(section(f"News ({len(news)})", 1), unsafe_allow_html=True)
        render_items(news, "No news items.")
    elif show_all and news:
        with st.expander(f"News (minimized) — {len(news)} items", expanded=False):
            render_items(news)


(tab_today, tab_brief, tab_cas, tab_pretom, tab_pead, tab_fmom, tab_edge,
 tab_nightday, tab_holdings, tab_mom, tab_etfmom, tab_ideas, tab_regimes, tab_index,
 tab_archive, tab_sources) = st.tabs(
    ["TODAY", "WEEKLY BRIEF", "CAS", "PRETOM", "PEAD", "FACTOR MOMENTUM",
     "EDGE SCREENS", "NIGHT & DAY", "HOLDINGS", "MOMENTUM", "ETF MOMENTUM",
     "IDEAS", "REGIMES", "INDEX", "ARCHIVE", "SOURCES"])

with tab_today:
    from zenith.ui_theme import stamp
    from zenith.pretom.view import today_badge
    from zenith.pead.view import today_badge as pead_today_badge
    from zenith.edge.view import today_badge as edge_today_badge
    from zenith.holdings.view import today_badge as holdings_today_badge
    from zenith.mom.view import today_badge as mom_today_badge
    from zenith.etfmom.view import today_badge as etfmom_today_badge
    from zenith.ideas.view import today_badge as ideas_today_badge
    from zenith.regimes.view import today_badge as regimes_today_badge
    from zenith.index.view import today_badge as index_today_badge
    for _b in (today_badge(), pead_today_badge(), edge_today_badge(),
               holdings_today_badge(), mom_today_badge(), etfmom_today_badge(),
               ideas_today_badge(), regimes_today_badge(), index_today_badge()):
        if _b:
            st.markdown(_b, unsafe_allow_html=True)
    latest = store.load_latest()
    dates = store.archive_dates()
    st.markdown(stamp(dates[0] if dates else "—", "Today"), unsafe_allow_html=True)
    st.caption(f"{len(latest)} new items. New items only — already-seen items don't repeat.")
    if not latest and dates:
        latest = store.load_archive(dates[0])
    insights_research_news(latest, "today")

with tab_brief:
    st.markdown(section("Weekly Brief — markets in one read", 0), unsafe_allow_html=True)
    from zenith.brief import view as brief_view
    brief_view.render()

with tab_cas:
    st.markdown(section("CAS — Complex Adaptive Systems monitor", 2), unsafe_allow_html=True)
    from zenith.cas import view as cas_view
    cas_view.render()

with tab_pretom:
    st.markdown(section("PRETOM — intramonth momentum short screener", 4),
                unsafe_allow_html=True)
    from zenith.pretom import view as pretom_view
    pretom_view.render()

with tab_pead:
    st.markdown(section("PEAD — post-earnings drift screener", 1),
                unsafe_allow_html=True)
    from zenith.pead import view as pead_view
    pead_view.render()

with tab_fmom:
    st.markdown(section("FACTOR MOMENTUM — Gupta & Kelly, three ways", 3),
                unsafe_allow_html=True)
    from zenith.fmom import view as fmom_view
    fmom_view.render()

with tab_edge:
    st.markdown(section("EDGE SCREENS — rated cross-sectional signals", 3),
                unsafe_allow_html=True)
    from zenith.edge import view as edge_view
    edge_view.render()

with tab_nightday:
    st.markdown(section("NIGHT & DAY — overnight vs intraday", 1),
                unsafe_allow_html=True)
    from zenith.nightday import view as nightday_view
    nightday_view.render()

with tab_holdings:
    st.markdown(section("HOLDINGS — fund position intelligence", 5),
                unsafe_allow_html=True)
    from zenith.holdings import view as holdings_view
    holdings_view.render()

with tab_mom:
    st.markdown(section("MOMENTUM — Russell 1000 stock momentum engine", 4),
                unsafe_allow_html=True)
    from zenith.mom import view as mom_view
    mom_view.render()

with tab_etfmom:
    st.markdown(section("ETF MOMENTUM — the momentum engine over the full ETF universe", 2),
                unsafe_allow_html=True)
    from zenith.etfmom import view as etfmom_view
    etfmom_view.render()

with tab_ideas:
    st.markdown(section("IDEAS — discretionary-systematic opportunity engine", 3),
                unsafe_allow_html=True)
    from zenith.ideas import view as ideas_view
    ideas_view.render()

with tab_regimes:
    st.markdown(section("REGIMES — macro regime intelligence & early-warning system", 2),
                unsafe_allow_html=True)
    from zenith.regimes import view as regimes_view
    regimes_view.render()

with tab_index:
    st.markdown(section("INDEX — the Master List: financial intelligence directory", 5),
                unsafe_allow_html=True)
    from zenith.index import view as index_view
    index_view.render()

with tab_archive:
    st.markdown(section("Archive", 0), unsafe_allow_html=True)
    dates = store.archive_dates()
    if not dates:
        st.info("No archived days yet. The nightly scrape will populate this.")
    else:
        day = st.selectbox("Pick a day", dates)
        items = store.load_archive(day)
        st.caption(f"{day} — {len(items)} items")
        q = st.text_input("Search titles/sources", "")
        if q.strip():
            ql = q.lower()
            items = [i for i in items if ql in i["title"].lower() or ql in i["source"].lower()]
        insights_research_news(items, "arch")

with tab_sources:
    from zenith.ui_theme import stamp
    st.markdown(section("Sources", 0), unsafe_allow_html=True)
    _sdates = store.archive_dates()
    st.markdown(stamp(_sdates[0] if _sdates else "—", "Sources"), unsafe_allow_html=True)
    st.caption("Feeds-first registry. Disabled sources have no confirmed free feed yet "
               "(extensible over time). 'Last run' reflects the most recent scrape.")
    status = {s["source"]: s for s in store.load_status()}
    rows = []
    for s in SOURCES:
        st_row = status.get(s.name, {})
        last = "ok" if st_row.get("ok") else ("err" if st_row else "—")
        rows.append({
            "Source": s.name, "Category": s.category, "Kind": s.kind,
            "Enabled": "✓" if s.enabled else "—",
            "Last run": last, "New (last run)": str(st_row.get("new", "")),
            "Note": s.note,
        })
    # health summary
    n_total = len(SOURCES)
    n_enabled = sum(1 for s in SOURCES if s.enabled and s.url)
    n_ok = sum(1 for r in rows if r["Last run"] == "ok")
    n_err = sum(1 for r in rows if r["Last run"] == "err")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered sources", n_total, help="Total sources in the catalog (enabled + documented).")
    m2.metric("Enabled", n_enabled, help="Sources with a confirmed URL that are attempted each run.")
    m3.metric("OK last run", n_ok, help="Returned items on the most recent scrape.")
    m4.metric("Errored last run", n_err, help="Reachable-but-failed; see the Note column for why.")
    by_cat = {c: sum(1 for s in SOURCES if s.category == c and s.enabled)
              for c in ("insight", "research", "news")}
    st.caption(f"Enabled by category — insights: {by_cat['insight']} · research: {by_cat['research']} "
               f"· news: {by_cat['news']}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=560, hide_index=True)

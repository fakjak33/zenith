"""ZENITH — Streamlit viewer.   Run:  streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from zenith import store
from zenith.sources import SOURCES
from zenith.auth import require_password
from zenith.ui_theme import CSS, BANNER, section, card_html

st.set_page_config(page_title="ZENITH // research", layout="wide", page_icon="▲")
st.markdown(CSS, unsafe_allow_html=True)
require_password()
st.markdown(BANNER, unsafe_allow_html=True)


def render_items(items, empty="Nothing here yet."):
    if not items:
        st.caption(empty)
        return
    html = "".join(card_html(it) for it in items)
    st.markdown(html, unsafe_allow_html=True)


def insights_research_news(items, key_prefix):
    """Render the standard Insights(text/visual) + Research + News layout."""
    insights = [i for i in items if i["category"] == "insight"]
    research = [i for i in items if i["category"] == "research"]
    news = [i for i in items if i["category"] == "news"]

    # optional source filter
    srcs = sorted({i["source"] for i in items})
    chosen = st.multiselect("Filter sources", srcs, default=[], key=f"{key_prefix}_flt")
    if chosen:
        insights = [i for i in insights if i["source"] in chosen]
        research = [i for i in research if i["source"] in chosen]
        news = [i for i in news if i["source"] in chosen]

    text_only = [i for i in insights if i["visual"] == "text"]
    visual = [i for i in insights if i["visual"] == "visual"]

    st.markdown(section(f"Insights — text-only ({len(text_only)})", 0), unsafe_allow_html=True)
    render_items(text_only, "No text-only insights.")
    st.markdown(section(f"Insights — charts / tables / visuals ({len(visual)})", 2),
                unsafe_allow_html=True)
    render_items(visual, "No visual insights.")
    st.markdown(section(f"Research & working papers ({len(research)})", 4), unsafe_allow_html=True)
    render_items(research, "No research items.")
    # News is intentionally de-emphasized — collapsed by default, off to the side.
    if news:
        with st.expander(f"News (minimized) — {len(news)} items", expanded=False):
            render_items(news)


tab_today, tab_archive, tab_sources = st.tabs(["TODAY", "ARCHIVE", "SOURCES"])

with tab_today:
    latest = store.load_latest()
    dates = store.archive_dates()
    st.caption(f"Most recent scrape: {dates[0] if dates else '—'} · "
               f"{len(latest)} new items. New items only — already-seen items don't repeat.")
    if not latest and dates:
        latest = store.load_archive(dates[0])
    insights_research_news(latest, "today")

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
    st.markdown(section("Sources", 0), unsafe_allow_html=True)
    st.caption("Feeds-first registry. Disabled sources have no confirmed free feed yet "
               "(extensible over time). 'Last run' reflects the most recent scrape.")
    status = {s["source"]: s for s in store.load_status()}
    rows = []
    for s in SOURCES:
        st_row = status.get(s.name, {})
        rows.append({
            "Source": s.name, "Category": s.category, "Kind": s.kind,
            "Enabled": "✓" if s.enabled else "—",
            "Last run": ("ok" if st_row.get("ok") else ("err" if st_row else "—")),
            "New (last run)": st_row.get("new", ""),
            "Note": s.note,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=560)

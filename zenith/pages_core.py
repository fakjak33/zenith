"""The core ZENITH pages that are not feature packages: TODAY, ARCHIVE, SOURCES."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import store
from .sources import SOURCES
from . import ui_kit as uk
from .config import THEME as T
from .ui_theme import card_html, section, stamp
from .viz.charts import calendar_heat

VIEWS = ["Everything", "Insights — text-only", "Insights — charts & visuals",
         "Research & papers", "News"]


def render_items(items, empty="Nothing here yet."):
    if not items:
        st.caption(empty)
        return
    st.markdown("".join(card_html(it) for it in items), unsafe_allow_html=True)


def insights_research_news(items, key_prefix):
    """The standard Insights(text/visual) + Research + News layout, with a 'View'
    dropdown to jump straight to one section."""
    insights = [i for i in items if i["category"] == "insight"]
    research = [i for i in items if i["category"] == "research"]
    news = [i for i in items if i["category"] == "news"]

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
        st.markdown(section(f"Research & working papers ({len(research)})", 4), unsafe_allow_html=True)
        render_items(research, "No research items.")
    # News stays de-emphasized: collapsed when browsing everything, full if chosen.
    if view == "News":
        st.markdown(section(f"News ({len(news)})", 1), unsafe_allow_html=True)
        render_items(news, "No news items.")
    elif show_all and news:
        with st.expander(f"News (minimized) — {len(news)} items", expanded=False):
            render_items(news)


@st.cache_data(ttl=900, show_spinner=False)
def _flow(days: int = 35) -> pd.DataFrame:
    """Items per source per day over the last `days` archived days."""
    rows = []
    for d in store.archive_dates()[:days]:
        for it in store.load_archive(d):
            rows.append({"day": d, "source": it.get("source", "?"), "category": it.get("category", "")})
    return pd.DataFrame(rows)


def _flow_panel(title: str = "RESEARCH FLOW") -> None:
    """Signature: a source x day pulse map of everything the scrape collected."""
    df = _flow()
    if df.empty:
        return
    top = df["source"].value_counts().head(18).index
    g = (df[df["source"].isin(top)].groupby(["source", "day"]).size().rename("n").reset_index())
    g["day"] = g["day"].str[5:]
    cap = max(float(g["n"].quantile(0.9)), 1.0)
    import altair as alt
    with uk.panel(title, T.cyan, f"top {len(top)} sources · last {df['day'].nunique()} days"):
        ch = alt.Chart(g).mark_circle(opacity=0.9).encode(
            x=alt.X("day:O", title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=9)),
            y=alt.Y("source:N", title=None, sort=list(top)),
            size=alt.Size("n:Q", legend=None, scale=alt.Scale(range=[8, 260], domain=[0, cap], clamp=True)),
            color=alt.Color("n:Q", legend=None, scale=alt.Scale(range=[T.line, T.cyan, T.teal], domain=[0, cap / 2, cap],
                                                                 clamp=True)),
            tooltip=["source", "day", "n"]).properties(height=max(200, 17 * len(top)))
        st.altair_chart(ch, use_container_width=True)
        mix = df["category"].value_counts()
        st.markdown(uk.kpis([{"label": k, "value": f"{v:,}", "color": c} for (k, v), c in
                             zip(mix.items(), [T.teal, T.amber, T.violet, T.cyan])]), unsafe_allow_html=True)


def today() -> None:
    latest = store.load_latest()
    dates = store.archive_dates()
    st.markdown(stamp(dates[0] if dates else "—", "Today"), unsafe_allow_html=True)
    st.caption(f"{len(latest)} new items. New items only — already-seen items don't repeat. "
               "Module status chips live on HOME.")
    if not latest and dates:
        latest = store.load_archive(dates[0])
    _flow_panel()
    insights_research_news(latest, "today")


def archive() -> None:
    dates = store.archive_dates()
    if not dates:
        st.info("No archived days yet. The nightly scrape will populate this.")
        return
    _flow_panel("ARCHIVE PULSE")
    day = st.selectbox("Pick a day", dates)
    items = store.load_archive(day)
    st.caption(f"{day} — {len(items)} items")
    q = st.text_input("Search titles/sources", "")
    if q.strip():
        ql = q.lower()
        items = [i for i in items if ql in i["title"].lower() or ql in i["source"].lower()]
    insights_research_news(items, "arch")


def sources() -> None:
    sdates = store.archive_dates()
    st.markdown(stamp(sdates[0] if sdates else "—", "Sources"), unsafe_allow_html=True)
    st.caption("Feeds-first registry. Disabled sources have no confirmed free feed yet "
               "(extensible over time). 'Last run' reflects the most recent scrape.")
    status = {s["source"]: s for s in store.load_status()}
    rows = []
    for s in SOURCES:
        st_row = status.get(s.name, {})
        last = "ok" if st_row.get("ok") else ("err" if st_row else "—")
        rows.append({"Source": s.name, "Category": s.category, "Kind": s.kind,
                     "Enabled": "✓" if s.enabled else "—", "Last run": last,
                     "New (last run)": str(st_row.get("new", "")), "Note": s.note})
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
    health = {(r["Category"], r["Source"][:22]): (1.0 if r["Last run"] == "ok" else (-1.0 if r["Last run"] == "err" else None))
              for r in rows if r["Enabled"] == "✓"}
    with uk.panel("SOURCE HEALTH", T.teal, "teal = ok · coral = error · grey = no run"):
        cats = sorted({c for c, _ in health})
        for c in cats:
            names = [n for cc, n in health if cc == c]
            dots = "".join(f'<i title="{n}" style="display:inline-block; width:12px; height:12px; margin:2px; '
                           f'background:{T.teal if health[(c, n)] == 1 else (T.coral if health[(c, n)] == -1 else T.line)};"></i>'
                           for n in names)
            st.markdown(f'<div style="display:flex; align-items:center; gap:0.8rem;"><span class="z-lbl" '
                        f'style="width:6rem;">{c}</span><div>{dots}</div></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=560, hide_index=True)

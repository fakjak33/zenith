"""The one registry of ZENITH pages: sidebar sections, URL slugs, icons, page
headers and HOME status chips all derive from FEATURES.

With st.navigation each rerun executes only the open page -- the old st.tabs
layout ran all twenty tabs on every click (~30 s).
"""

from __future__ import annotations

import html
import importlib
from dataclasses import dataclass

import streamlit as st

from .config import THEME as T


@dataclass(frozen=True)
class Feature:
    group: str
    label: str            # page title (the old tab label)
    slug: str             # URL path, stable for bookmarks
    module: str | None    # zenith.<pkg>.view with render(); None = core page in pages_core
    header: str           # "TITLE — what it is", shown as the page header
    color: int = 0        # accent index into THEME.section_colors
    badge: bool = False   # contributes a status chip to HOME
    icon: str = ":material/radio_button_unchecked:"


FEATURES: tuple[Feature, ...] = (
    Feature("MARKETS", "TODAY", "today", None, "TODAY — new research & insights", 0,
            icon=":material/feed:"),
    Feature("MARKETS", "WEEKLY BRIEF", "brief", "zenith.brief.view", "WEEKLY BRIEF — markets in one read", 1,
            icon=":material/article:"),
    Feature("MARKETS", "REGIMES", "regimes", "zenith.regimes.view",
            "REGIMES — macro regime intelligence & early warning", 2, True, ":material/public:"),
    Feature("MARKETS", "NIGHT & DAY", "nightday", "zenith.nightday.view",
            "NIGHT & DAY — where the return is really made", 3, icon=":material/contrast:"),
    Feature("SIGNALS", "CAS", "cas", "zenith.cas.view", "CAS — complex adaptive systems monitor", 2,
            icon=":material/hub:"),
    Feature("SIGNALS", "MOMENTUM", "momentum", "zenith.mom.view", "MOMENTUM — Russell 1000 stock momentum engine",
            0, True, ":material/speed:"),
    Feature("SIGNALS", "ETF MOMENTUM", "etf-momentum", "zenith.etfmom.view",
            "ETF MOMENTUM — the momentum engine across the ETF universe", 2, True, ":material/donut_large:"),
    Feature("SIGNALS", "TREND FOLLOWING", "trend", "zenith.trend.view",
            "TREND FOLLOWING — seven-speed EWMAC, stocks & ETFs", 4, True, ":material/trending_up:"),
    Feature("SIGNALS", "FACTOR MOMENTUM", "factor-momentum", "zenith.fmom.view",
            "FACTOR MOMENTUM — Gupta & Kelly, three ways", 3, icon=":material/stacked_line_chart:"),
    Feature("SIGNALS", "EDGE SCREENS", "edge", "zenith.edge.view", "EDGE SCREENS — rated cross-sectional signals",
            1, True, ":material/filter_alt:"),
    Feature("SIGNALS", "PEAD", "pead", "zenith.pead.view", "PEAD — post-earnings drift screener", 1, True,
            ":material/event_upcoming:"),
    Feature("SIGNALS", "PRETOM", "pretom", "zenith.pretom.view", "PRETOM — intramonth momentum short screener",
            3, True, ":material/calendar_month:"),
    Feature("SIGNALS", "IDEAS", "ideas", "zenith.ideas.view", "IDEAS — discretionary-systematic opportunities",
            0, True, ":material/lightbulb:"),
    Feature("PORTFOLIO", "CLEAN BETA", "clean-beta", "zenith.beta.view",
            "CLEAN BETA — quality high-beta screener & hedge monitor", 0, True, ":material/shield:"),
    Feature("PORTFOLIO", "HOLDINGS", "holdings", "zenith.holdings.view", "HOLDINGS — fund position intelligence",
            5, True, ":material/account_balance:"),
    Feature("OPTIONS", "OPTIONS VAULT", "options-vault", "zenith.vault.view",
            "OPTIONS VAULT — evidence-graded strategies, edges & mechanics", 4, True, ":material/menu_book:"),
    Feature("OPTIONS", "THETA RUN", "theta-run", "zenith.theta.view",
            "THETA RUN — options-portfolio speedrun vs the tape and the bot", 0, True, ":material/sports_esports:"),
    Feature("GAMES", "EPHEMERIS", "ephemeris", "zenith.ephemeris.view",
            "EPHEMERIS — blind-chart tape-reading trainer", 1, True, ":material/candlestick_chart:"),
    Feature("RESEARCH", "INDEX", "index", "zenith.index.view",
            "INDEX — the master list of the financial information ecosystem", 5, True, ":material/share:"),
    Feature("RESEARCH", "ARCHIVE", "archive", None, "ARCHIVE — every scraped day", 0, icon=":material/inventory_2:"),
    Feature("RESEARCH", "SOURCES", "sources", None, "SOURCES — the feed registry", 0, icon=":material/rss_feed:"),
)
GROUPS = ("MARKETS", "SIGNALS", "PORTFOLIO", "OPTIONS", "GAMES", "RESEARCH")
PAGES: dict[str, "st.Page"] = {}          # slug -> st.Page, filled by build()


def render_feature(module_path: str, feature: str) -> None:
    """Import a feature's view module and render it, containing any failure to
    THIS page. The error is shown, not swallowed; Streamlit logs the traceback.
    Stale modules can survive a Streamlit Cloud redeploy, so the hint says reboot."""
    try:
        importlib.import_module(module_path).render()
    except Exception as exc:
        st.error(f"The {feature} page could not be loaded: {type(exc).__name__}: {exc}")
        st.caption("The rest of the app is unaffected. If this followed a fresh deploy, "
                   "rebooting the app usually clears it — stale modules can survive a "
                   "redeploy even when the code on disk is correct.")


def badges() -> tuple[list[tuple[Feature, str]], list[str]]:
    """(feature, chip html) for every badge-carrying feature; failures are named, never raised."""
    chips, failed = [], []
    for f in FEATURES:
        if not f.badge or not f.module:
            continue
        try:
            b = importlib.import_module(f.module).today_badge()
            if b:
                chips.append((f, b))
        except Exception:
            failed.append(f.label)
    return chips, failed


def page_header(f: Feature) -> str:
    """Large light title + one-line description, with the page's accent tick."""
    title, _, desc = f.header.partition(" — ")
    c = T.section_colors[f.color % len(T.section_colors)]
    return (f'<div class="z-phead" style="display:flex; align-items:flex-end; gap:1rem; margin:0 0 1.1rem 0;">'
            f'<div style="width:3px; height:2.6rem; background:{c};"></div><div>'
            f'<div style="font-size:0.58rem; letter-spacing:0.3em; color:{T.faint};">{html.escape(f.group)}</div>'
            f'<div style="font-size:1.75rem; font-weight:300; letter-spacing:0.12em; line-height:1.15;">'
            f'{html.escape(title)}</div>'
            f'<div style="font-size:0.74rem; color:{T.muted}; letter-spacing:0.04em;">{html.escape(desc)}</div>'
            f'</div></div>')


def _page_fn(f: Feature):
    def run() -> None:
        from .chrome import topbar
        topbar(f"{f.group} / {f.label}")
        st.markdown(page_header(f), unsafe_allow_html=True)
        if f.module:
            render_feature(f.module, f.label)
        else:
            from . import pages_core
            getattr(pages_core, f.slug)()
    run.__name__ = f"page_{f.slug.replace('-', '_')}"
    return run


def build():
    """st.navigation in the sidebar: DASHBOARD first, then the six sections."""
    from . import home
    PAGES.clear()
    PAGES["home"] = st.Page(home.render, title="DASHBOARD", url_path="home", default=True,
                            icon=":material/grid_view:")
    groups: dict[str, list] = {"": [PAGES["home"]]}
    for f in FEATURES:
        PAGES[f.slug] = st.Page(_page_fn(f), title=f.label, url_path=f.slug, icon=f.icon)
        groups.setdefault(f.group, []).append(PAGES[f.slug])
    return st.navigation(groups, position="sidebar", expanded=True)

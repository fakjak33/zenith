"""The one registry of ZENITH pages: navigation groups, URL slugs, section headers
and TODAY/HOME status chips all derive from FEATURES.

With st.navigation each rerun executes only the open page -- the old st.tabs
layout ran all twenty tabs on every click (~30 s).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import streamlit as st

from .ui_theme import section


@dataclass(frozen=True)
class Feature:
    group: str
    label: str            # page title (the old tab label)
    slug: str             # URL path, stable for bookmarks
    module: str | None    # zenith.<pkg>.view with render(); None = core page in pages_core
    header: str           # section header shown above the page
    color: int = 0        # section colour index
    badge: bool = False   # contributes a status chip to HOME


FEATURES: tuple[Feature, ...] = (
    Feature("MARKETS", "TODAY", "today", None, "TODAY — new research & insights", 0),
    Feature("MARKETS", "WEEKLY BRIEF", "brief", "zenith.brief.view", "Weekly Brief — markets in one read", 1),
    Feature("MARKETS", "REGIMES", "regimes", "zenith.regimes.view",
            "REGIMES — macro regime intelligence & early-warning system", 2, True),
    Feature("MARKETS", "NIGHT & DAY", "nightday", "zenith.nightday.view", "NIGHT & DAY — overnight vs intraday", 3),
    Feature("SIGNALS", "CAS", "cas", "zenith.cas.view", "CAS — Complex Adaptive Systems monitor", 2),
    Feature("SIGNALS", "MOMENTUM", "momentum", "zenith.mom.view", "MOMENTUM — Russell 1000 stock momentum engine",
            0, True),
    Feature("SIGNALS", "ETF MOMENTUM", "etf-momentum", "zenith.etfmom.view",
            "ETF MOMENTUM — the momentum engine over the full ETF universe", 2, True),
    Feature("SIGNALS", "TREND FOLLOWING", "trend", "zenith.trend.view",
            "TREND FOLLOWING — seven-speed EWMAC, stocks & ETFs", 4, True),
    Feature("SIGNALS", "FACTOR MOMENTUM", "factor-momentum", "zenith.fmom.view",
            "FACTOR MOMENTUM — Gupta & Kelly, three ways", 3),
    Feature("SIGNALS", "EDGE SCREENS", "edge", "zenith.edge.view", "EDGE SCREENS — rated cross-sectional signals",
            1, True),
    Feature("SIGNALS", "PEAD", "pead", "zenith.pead.view", "PEAD — post-earnings drift screener", 1, True),
    Feature("SIGNALS", "PRETOM", "pretom", "zenith.pretom.view", "PRETOM — intramonth momentum short screener",
            3, True),
    Feature("SIGNALS", "IDEAS", "ideas", "zenith.ideas.view", "IDEAS — discretionary-systematic opportunity engine",
            0, True),
    Feature("PORTFOLIO", "CLEAN BETA", "clean-beta", "zenith.beta.view",
            "CLEAN BETA — quality high-beta screener + convex hedge monitor", 0, True),
    Feature("PORTFOLIO", "HOLDINGS", "holdings", "zenith.holdings.view", "HOLDINGS — fund position intelligence",
            5, True),
    Feature("OPTIONS", "OPTIONS VAULT", "options-vault", "zenith.vault.view",
            "OPTIONS VAULT — evidence-graded options strategies, edges & mechanics", 4, True),
    Feature("OPTIONS", "THETA RUN", "theta-run", "zenith.theta.view",
            "THETA RUN — options-portfolio speedrun vs the tape and the bot", 0, True),
    Feature("GAMES", "EPHEMERIS", "ephemeris", "zenith.ephemeris.view",
            "EPHEMERIS — blind-chart tape-reading trainer", 1, True),
    Feature("RESEARCH", "INDEX", "index", "zenith.index.view",
            "INDEX — the Master List: financial intelligence directory", 5, True),
    Feature("RESEARCH", "ARCHIVE", "archive", None, "ARCHIVE — every scraped day", 0),
    Feature("RESEARCH", "SOURCES", "sources", None, "SOURCES — the feed registry", 0),
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


def _page_fn(f: Feature):
    def run() -> None:
        st.markdown(section(f.header, f.color), unsafe_allow_html=True)
        if f.module:
            render_feature(f.module, f.label)
        else:
            from . import pages_core
            getattr(pages_core, f.slug)()
    run.__name__ = f"page_{f.slug.replace('-', '_')}"
    return run


def build():
    """st.navigation with HOME first, then the groups (position=top)."""
    from . import home
    PAGES.clear()
    PAGES["home"] = st.Page(home.render, title="HOME", url_path="home", default=True, icon=":material/terminal:")
    groups: dict[str, list] = {"": [PAGES["home"]]}
    for f in FEATURES:
        PAGES[f.slug] = st.Page(_page_fn(f), title=f.label, url_path=f.slug)
        groups.setdefault(f.group, []).append(PAGES[f.slug])
    return st.navigation(groups, position="top")

"""Zenith config: paths + ported Parallax theme."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
LATEST_JSON = DATA_DIR / "latest.json"
SEEN_JSON = DATA_DIR / "seen.json"
STATUS_JSON = DATA_DIR / "status.json"
USAGE_JSON = DATA_DIR / "apify_usage.json"

# --- CAS (Complex Adaptive Systems) monitor -------------------------------
CAS_DIR = DATA_DIR / "cas"
CAS_ARCHIVE_DIR = CAS_DIR / "archive"
CAS_CACHE_DIR = CAS_DIR / "cache"            # raw data caches (prices, cot, …)
# one JSON file per CAS artefact, mirroring the feeds-side store
CAS_FILES = {
    "signals": CAS_DIR / "signals_latest.json",
    "positioning": CAS_DIR / "positioning.json",
    "themes": CAS_DIR / "themes.json",
    "rebalance": CAS_DIR / "rebalance.json",
    "consensus": CAS_DIR / "consensus.json",
    "overlap": CAS_DIR / "overlap.json",
    "registry": CAS_DIR / "registry.json",
    "contingency": CAS_DIR / "contingency.json",
    "status": CAS_DIR / "status.json",
    "factor_rotation": CAS_DIR / "factor_rotation.json",
    "backtest": CAS_DIR / "backtest_factor_momentum.json",
    "history": CAS_DIR / "history.json",
    "hitrate": CAS_DIR / "hitrate.json",
    "price_panel": CAS_DIR / "price_panel.json",   # committed: powers app price overlays
}

# --- Weekly Brief (market commentary) -------------------------------------
BRIEF_DIR = DATA_DIR / "brief"
BRIEF_FILES = {
    "brief": BRIEF_DIR / "brief.json",          # the assembled weekly brief
}

for _d in (DATA_DIR, ARCHIVE_DIR, CAS_DIR, CAS_ARCHIVE_DIR, CAS_CACHE_DIR, BRIEF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# polite scraping
USER_AGENT = "ZenithResearchAggregator/0.1 (+personal research; respects robots.txt)"
REQUEST_TIMEOUT = 15
MAX_ITEMS_PER_SOURCE = 25       # cap per source per run
CLASSIFY_FETCH = True           # fetch article pages to classify text vs visual
CLASSIFY_MAX_FETCH = 120        # cap page fetches per run (politeness/time)

# A realistic browser User-Agent for the *direct* fetch tier. Many sites that
# return 403 to a bot UA serve fine to a normal browser UA — this unblocks a
# lot of "blocked" sources for free, before we ever fall back to Apify.
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}

# --- Apify fallback (only used when the direct tier is blocked) -------------
# Token comes from env / Streamlit secret / GitHub Action secret — never hard-coded.
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "").strip()
# Default to Apify's general-purpose Store crawler in cheerio mode (no browser =
# cheapest). For the hardest Cloudflare/JS sites, set APIFY_CRAWLER=playwright.
APIFY_ACTOR = os.environ.get("APIFY_ACTOR", "apify/website-content-crawler").strip()
APIFY_CRAWLER = os.environ.get("APIFY_CRAWLER", "cheerio").strip()   # cheerio | playwright
# Residential proxy reaches the hardest anti-bot sites but costs much more than
# the FREE tier supports — off by default. Set APIFY_RESIDENTIAL=1 if you upgrade.
APIFY_RESIDENTIAL = os.environ.get("APIFY_RESIDENTIAL", "").strip() in ("1", "true", "True")
APIFY_TIMEOUT = int(os.environ.get("APIFY_TIMEOUT", "120"))
# Soft monthly safety cap (USD). The FREE plan grants ~$5/mo; stop calling Apify
# once we estimate we've spent this much so a run can never blow the budget.
APIFY_MONTHLY_BUDGET_USD = float(os.environ.get("APIFY_MONTHLY_BUDGET_USD", "4.0"))
APIFY_ENABLED = bool(APIFY_TOKEN)

# --- Firecrawl fallback (free-tier alternative, tried before Apify) ---------
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "").strip()
FIRECRAWL_TIMEOUT = int(os.environ.get("FIRECRAWL_TIMEOUT", "45"))


@dataclass(frozen=True)
class Theme:
    bg: str = "#000000"
    panel: str = "#0b0b0b"
    grid: str = "#2c2c2c"
    border: str = "#ffffff"
    teal: str = "#2ec4b6"
    coral: str = "#ff5a3c"
    orange: str = "#ff8c2b"
    mustard: str = "#ffc857"
    mauve: str = "#c46b8b"
    mint: str = "#7bdcb5"
    navy: str = "#2a9bc4"
    text: str = "#ffffff"
    muted: str = "#b8b8b8"
    font_display: str = "'VT323', 'Space Mono', 'Courier New', monospace"
    font_body: str = "'Space Mono', 'Share Tech Mono', 'Courier New', monospace"
    section_colors: tuple = ("#2ec4b6", "#ffc857", "#ff5a3c", "#c46b8b", "#2a9bc4", "#7bdcb5")


THEME = Theme()

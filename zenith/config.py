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
    "rotation": CAS_DIR / "rotation.json",         # factor rotation by look-back window
    "fomc": CAS_DIR / "fomc.json",                 # FOMC cycle evidence + countdown
}

# --- Weekly Brief (market commentary) -------------------------------------
BRIEF_DIR = DATA_DIR / "brief"
BRIEF_FILES = {
    "brief": BRIEF_DIR / "brief.json",          # the assembled weekly brief
}

# --- PRETOM (intramonth momentum short screener) ---------------------------
PRETOM_DIR = DATA_DIR / "pretom"
PRETOM_ARCHIVE_DIR = PRETOM_DIR / "archive"      # one basket snapshot per month
PRETOM_FILES = {
    "basket": PRETOM_DIR / "basket_latest.json",   # current month's ranked basket
    "history": PRETOM_DIR / "history.json",        # tracker rows across all baskets
    "status": PRETOM_DIR / "status.json",          # state machine + run health
    "universe": PRETOM_DIR / "universe.json",      # last-good Russell 1000 snapshot
    "tom": PRETOM_DIR / "tom_longrun.json",        # SPY turn-of-the-month evidence
}

# --- PEAD (post-earnings announcement drift screener) -----------------------
PEAD_DIR = DATA_DIR / "pead"
PEAD_ARCHIVE_DIR = PEAD_DIR / "archive"          # one signal sheet per reaction day
PEAD_FILES = {
    "signals": PEAD_DIR / "signals_latest.json",   # recent sheets + active book + drift curve
    "history": PEAD_DIR / "history.json",          # append-only picks + horizon evaluations
    "status": PEAD_DIR / "status.json",            # run health + day state
    "eap": PEAD_DIR / "eap.json",                  # announcement premium: upcoming + summary
    "eap_history": PEAD_DIR / "eap_history.json",  # append-only announcement windows
}

# --- FMOM (factor momentum — Gupta & Kelly TSFM/CSFM) -----------------------
FMOM_DIR = DATA_DIR / "fmom"
FMOM_ARCHIVE_DIR = FMOM_DIR / "archive"          # one signal snapshot per month
FMOM_FILES = {
    "signals": FMOM_DIR / "signals_latest.json",   # current formation month, all models
    "history": FMOM_DIR / "history.json",          # append-only picks + evaluations
    "backtest": FMOM_DIR / "backtest.json",        # published/ETF-panel backtests
    "holdings": FMOM_DIR / "holdings_latest.json", # replication layer bin members
    "etf_catalog": FMOM_DIR / "etf_catalog.json",  # Morningstar strategic-beta catalog
    "osap_catalog": FMOM_DIR / "osap_catalog.json",  # OSAP SignalDoc definitions
    "screens": FMOM_DIR / "screens_latest.json",   # per-characteristic stock screens
    "status": FMOM_DIR / "status.json",            # run health + per-source as-of dates
}

# --- EDGE (cross-sectional screeners: IV spread / revisions / SI / lottery) -
EDGE_DIR = DATA_DIR / "edge"
EDGE_FILES = {
    "ivspread": EDGE_DIR / "ivspread.json",        # Cremers-Weinbaum IV spread ranks
    "revisions": EDGE_DIR / "revisions.json",      # analyst estimate-revision ranks
    "shortint": EDGE_DIR / "shortint.json",        # short interest / days-to-cover
    "lottery": EDGE_DIR / "lottery.json",          # MAX / MAX-beta lottery short screen
    "history": EDGE_DIR / "history.json",          # append-only decile snapshots + eval
    "status": EDGE_DIR / "status.json",            # run health per screen
}

# --- NIGHT & DAY (overnight vs intraday return decomposition) ----------------
NIGHTDAY_DIR = DATA_DIR / "nightday"
NIGHTDAY_FILES = {
    "panel": NIGHTDAY_DIR / "panel.json",          # ETF overnight/intraday cumulative series
    "screen": NIGHTDAY_DIR / "screen.json",        # R1000 ranked overnight/intraday/tug-of-war
    "history": NIGHTDAY_DIR / "history.json",       # decile snapshots + horizon eval
    "status": NIGHTDAY_DIR / "status.json",
}

# --- MOMENTUM (Russell 1000 multi-factor stock momentum engine) ------------
# Five factors (time-series, breakout, cross-sectional, trend speed / GMMA,
# momentum strength) blended into a transparent -20..+20 composite. Big
# per-day artefacts are written COMPACT (indent=None) — see mom/__init__.py
# _write(); at ~1000 rows/day, pretty-printing would add ~300MB/yr to git.
MOM_DIR = DATA_DIR / "mom"
MOM_HISTORY_DIR = MOM_DIR / "history"        # sharded data/mom/history/<YYYY>.json
MOM_FILES = {
    "scores": MOM_DIR / "scores_latest.json",       # full R1000 ranked composite + factors
    "detail": MOM_DIR / "detail_latest.json",       # per-stock factor internals (MAs, slopes, …)
    "sectors": MOM_DIR / "sectors.json",            # sector/industry aggregates
    "diagnostics": MOM_DIR / "diagnostics.json",    # factor correlation matrix, IC, hit rates
    "meta": MOM_DIR / "meta.json",                  # ticker -> name/sector/industry/mktcap cache
    "membership": MOM_DIR / "membership.json",      # append-only point-in-time constituents
    "picks": MOM_DIR / "picks.json",                # append-only decile pick tracker + eval
    "status": MOM_DIR / "status.json",
}

# Horizon weights shared by the time-series, breakout and cross-sectional
# factors. Tilted toward the slower legs (12-1, 6M) per the "medium/long-term,
# not day-trading" brief; the 1M leg is kept small because short-horizon
# returns are known to reverse (Jegadeesh 1990), not disappear.
MOM_HORIZON_WEIGHTS = {
    "12_1": 0.30,   # 12-month return excluding the most recent month (classic UMD spec)
    "12m": 0.15,
    "9m": 0.15,
    "6m": 0.20,
    "3m": 0.15,
    "1m": 0.05,
}

# Composite factor weights. Not equal: TS and cross-sectional momentum carry
# the most independent information (MOP 2012 decompose the premium into
# distinct auto-covariance vs cross-serial components); breakout is a
# non-linear read of the same path as TS so it is downweighted; speed and
# strength are both MA-derived (the likeliest redundant pair) and split
# .15/.20, with strength weighted higher because its acceleration/quality
# terms carry information neither TS nor speed contains. Configurable without
# touching code; the app also renders an equal-weight composite for
# comparison and flags factor-pair correlations above 0.85.
MOM_WEIGHTS = {
    "ts": 0.25,
    "xsec": 0.25,
    "breakout": 0.15,
    "speed": 0.15,
    "strength": 0.20,
}

# Moving averages for the trend-speed / GMMA factor and chart (trading days).
MOM_MA_PERIODS = (9, 21, 50, 100, 200, 250, 400)

# Signal-state bands on the -20..+20 composite (threshold, label), descending.
MOM_STATES = (
    (15.0, "EXTREME BULLISH"),
    (10.0, "STRONG BULLISH"),
    (5.0, "BULLISH"),
    (-5.0, "NEUTRAL"),
    (-10.0, "BEARISH"),
    (-15.0, "STRONG BEARISH"),
    (-20.0, "EXTREME BEARISH"),
)

# The point at which point-in-time R1000 membership tracking began (PRETOM's
# universe.json first commit). Historical MOMENTUM scores before this date use
# TODAY's constituents and are survivorship-biased; the UI marks the boundary.
MOM_MEMBERSHIP_START = "2026-07-15"

# --- IDEAS (discretionary-systematic opportunity engine) --------------------
# Fusion layer over MOMENTUM/EDGE/PEAD/FMOM/CAS — not a new data pipeline (see
# zenith/ideas/__init__.py for the full architecture note). Its own committed
# fundamentals cache follows mom.universe.refresh_metadata's proven pattern
# (rolling TTL, capped per-run refresh, self-heals from cold) rather than
# fmom's gitignored store_cas cache, because this package's nightly Action
# needs the data warm on every CI run, not just locally.
IDEAS_DIR = DATA_DIR / "ideas"
IDEAS_ARCHIVE_DIR = IDEAS_DIR / "archive"
IDEAS_FILES = {
    "ideas": IDEAS_DIR / "ideas_latest.json",           # today's ranked BUY/SELL ideas, full payload
    "candidates": IDEAS_DIR / "candidates.json",        # top/bottom ~150 by unusualness (pre-selection pool)
    "universe_scores": IDEAS_DIR / "universe_scores.json",  # compact {ticker: conviction/unusual} for the whole scan universe
    "fundamentals": IDEAS_DIR / "fundamentals.json",    # rolling .info cache (30d TTL, ~150/run)
    "tracker": IDEAS_DIR / "tracker.json",              # append-only thesis-status history per idea
    "performance": IDEAS_DIR / "performance.json",      # closed-idea returns vs benchmark
    "diagnostics": IDEAS_DIR / "diagnostics.json",      # engine's own out-of-sample IC/hit-rate (accumulates)
    "analog_bank": IDEAS_DIR / "analog_bank.json",      # historical setups for the analog engine (local backfill)
    "status": IDEAS_DIR / "status.json",
}

# Scan universe: Russell 1000 (pretom.universe.russell1000) + the CAS tagged
# ETF set (cas.universe.frm_universe, ~335 names). Written as a flag so a
# future Russell 3000 pass is a config change, not a rewrite — see
# ideas/__init__.py's note on why R1000 was chosen for the MVP (every existing
# Zenith signal is already computed on exactly this universe).
IDEAS_UNIVERSE_SCOPE = "r1000_etf"    # r1000_etf | r3000_etf (not yet implemented)

# Eight signal groups, each scored in [-1, +1] with an explicit coverage flag
# (a stock missing options data is renormalized over what IS available, never
# silently scored neutral — see mom.engine._weighted, reused here). Weights
# are set A PRIORI from each input's own documented evidence tier (EDGE
# screens already carry A/B/C ratings) and are NEVER fitted to historical
# returns — there is no optimization loop anywhere in this package (see
# ideas/__init__.py's anti-overfitting note, spec §21). Rationale per weight:
#   technicals   .20 — MOMENTUM's 5-factor composite is this repo's most
#                       validated single input (B+, its own IC diagnostic).
#   sentiment    .15 — analyst revisions (EDGE B) + IV spread (EDGE C+,
#                       partly a borrow-fee proxy) blended, revisions-led.
#   positioning  .15 — short interest / crowding (EDGE B gross, C net of
#                       borrow fees) + institutional ownership level.
#   valuation    .15 — cross-sectional lens is real today; the two historical
#                       lenses are weaker (see ideas/valuation.py) so this
#                       group's OWN internal weighting already discounts them.
#   fundamentals .10 — quality/growth/leverage overlay from .info fields, no
#                       in-repo replication study behind it, kept modest.
#   catalyst     .10 — PEAD's post-earnings-drift composite (B) + the
#                       announcement-premium calendar (B).
#   risk_reward  .10 — structural score (stop distance vs target distance,
#                       ATR-based) plus the MAXβ lottery penalty (EDGE B+,
#                       Bali-Ince-Ozsoylev 2026) — a risk filter, not itself
#                       a return-predicting signal.
#   macro        .05 — regime conditioning (cas.signals.regime), deliberately
#                       small: it moves every stock the same direction by
#                       default and mainly acts as a tilt/gate, not a scorer.
IDEAS_WEIGHTS = {
    "technicals": 0.20, "sentiment": 0.15, "positioning": 0.15, "valuation": 0.15,
    "fundamentals": 0.10, "catalyst": 0.10, "risk_reward": 0.10, "macro": 0.05,
}

# Quality gates for the daily BUY/SELL selection (spec §1: never pad to a
# quota). Both floors must clear; a thin day shows fewer than 5 ideas and says
# so via a state banner rather than lowering the bar.
IDEAS_GATES = {
    "min_conviction": 62.0,        # 0-100
    "min_unusual": 55.0,           # 0-100
    "min_coverage_n": 3,           # minimum covered signal groups (of 8) to be eligible at all --
                                    # a security scored on 1-2 groups can look artificially extreme
                                    # (its whole conviction/unusual read rides on that one group), so
                                    # thin coverage is excluded rather than mathematically dampened
                                    # (spec section 29's "never silently fabricate" principle applied
                                    # to breadth, not just to individual missing fields).
    "min_adv_usd": 5_000_000.0,    # 63d avg-dollar-volume liquidity floor
    "target_n_per_side": 5,        # soft target, never forced
    "max_n_per_side": 10,
    "max_etf_slots_per_side": 2,   # ETFs cannot crowd out stock-picking (spec §2)
}

# Broad-beta instruments that carry a hard unusualness discount (spec §2: "do
# not repeatedly generate Buy SPY unless genuinely unusual"). Extend over time.
IDEAS_OBVIOUS_TICKERS = {
    "SPY", "VOO", "IVV", "VTI", "QQQ", "IWM", "IWB", "VONE", "DIA",
    "EFA", "VEA", "VWO", "EEM", "AGG", "BND", "IEI", "IEF", "TLT", "GOVT",
    "LQD", "HYG", "GLD", "SLV",
}

# Deterministic-narrative & selection weights conditioned by market regime
# (spec §23): coarse 3-state, never a fitted model — "risk-off" tilts weight
# toward quality/valuation and away from momentum/lottery-style names.
IDEAS_REGIME_TILTS = {
    "risk-on": {"technicals": 1.10, "risk_reward": 0.90},
    "neutral / transition": {},
    "risk-off": {"technicals": 0.85, "valuation": 1.15, "fundamentals": 1.10},
}

# --- HOLDINGS (fund position intelligence — DBMF first) ---------------------
# One sub-directory per tracked fund so a second fund is a registry entry, not
# a schema change: data/holdings/<fund>/{latest,history,changes,status}.json
# plus data/holdings/<fund>/archive/YYYY-MM-DD.json (the source of truth).
HOLDINGS_DIR = DATA_DIR / "holdings"
HOLDINGS_FUNDS_JSON = HOLDINGS_DIR / "funds.json"   # registry snapshot for the app
HOLDINGS_ARTEFACTS = ("latest", "history", "changes", "status")


def holdings_dir(fund: str) -> Path:
    """Per-fund artefact directory (created on demand)."""
    d = HOLDINGS_DIR / fund
    d.mkdir(parents=True, exist_ok=True)
    return d


def holdings_archive_dir(fund: str) -> Path:
    """Per-fund daily snapshot directory (created on demand)."""
    d = HOLDINGS_DIR / fund / "archive"
    d.mkdir(parents=True, exist_ok=True)
    return d


def holdings_files(fund: str) -> dict:
    """Named artefact paths for one fund, mirroring the *_FILES convention."""
    d = holdings_dir(fund)
    return {
        "latest": d / "latest.json",      # current snapshot + summary + 1-day changes
        "history": d / "history.json",    # date-indexed per-position weight/notional series
        "changes": d / "changes.json",    # change events + precomputed window rankings
        "status": d / "status.json",      # run health + source freshness + quality flags
    }


for _d in (DATA_DIR, ARCHIVE_DIR, CAS_DIR, CAS_ARCHIVE_DIR, CAS_CACHE_DIR, BRIEF_DIR,
           PRETOM_DIR, PRETOM_ARCHIVE_DIR, PEAD_DIR, PEAD_ARCHIVE_DIR,
           FMOM_DIR, FMOM_ARCHIVE_DIR, EDGE_DIR, NIGHTDAY_DIR, HOLDINGS_DIR,
           MOM_DIR, MOM_HISTORY_DIR, IDEAS_DIR, IDEAS_ARCHIVE_DIR):
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

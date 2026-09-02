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

# Composite factor weights. Originally ts/xsec .25/.25, breakout .15,
# speed/strength .15/.20 (see git history) -- rebalanced when Multivariate
# Trend (mvt) was added, per a live measurement, not a guess: the engine's
# own factor-correlation diagnostic already flagged ts/xsec at Spearman 0.997
# (near-total redundancy -- both derive from the same vol-adjusted-return
# grid via different monotonic transforms), while a synthetic-panel test of
# mvt's residual (common-factor-removed) pairwise trend against xsec came in
# at ~0.73-0.74 -- genuinely the least redundant addition available. ts/xsec
# are trimmed .05 each (still tied for the largest single-factor weight) and
# mvt enters at .20, matched to them, rather than starting small and having
# to be "discovered" later. breakout/speed/strength take a small proportional
# trim to fund it. This is still a documented judgment call, NOT a fitted
# optimum -- see mvt.horizons.erc_weights() and data/mom/weighting.json for
# the equal-risk-contribution lens computed nightly from the LIVE correlation
# matrix, which is what should actually move this dict if the user adopts it
# (config.MOM_WEIGHT_MODE). The app renders equal-weight AND erc composites
# alongside the declared one for comparison, and flags factor-pair
# correlations above 0.85.
MOM_WEIGHTS = {
    "ts": 0.20,
    "xsec": 0.20,
    "breakout": 0.12,
    "speed": 0.13,
    "strength": 0.15,
    "mvt": 0.20,
}

# "declared" (MOM_WEIGHTS above, the live default) or "erc" (equal-risk-
# contribution weights recomputed nightly from the live factor correlation
# matrix -- see mvt/horizons.py:erc_weights). Flip only after reviewing
# data/mom/weighting.json / the Momentum tab's weighting comparison; never
# silently. "declared" until the user decides otherwise (see MOM §41 --
# reproducibility of the existing scoring approach is the default posture).
MOM_WEIGHT_MODE = "declared"

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

# --- MOMENTUM > Multivariate Trend (pairwise relative-strength / residual
# momentum sub-signal) ------------------------------------------------------
# Lives under mom/ (it is MOMENTUM's 6th factor, not a new top-level feature).
# Two universes: equities reuse mom.universe.constituents() (the R1000, no
# second pipeline); ETFs union cas.universe.master_etfs() + frm_tickers() +
# the Morningstar etf_catalog.json (~935 raw, gated down -- see
# mvt/universe.py). The NxN pairwise matrix is NEVER committed (see
# mvt/pairwise.py's storage-architecture note) -- only the per-horizon return
# vectors + PCA loadings/eigenvalues/idio vols needed to reconstruct any pair
# on demand are.
MOM_MVT_DIR = MOM_DIR / "mvt"
MOM_MVT_HISTORY_DIR = MOM_MVT_DIR / "history"     # sharded rank-evolution snapshots (Phase 3)
MOM_MVT_FILES = {
    "equities": MOM_MVT_DIR / "equities_latest.json",   # R1000 mvt scores + reconstruction inputs
    "etfs": MOM_MVT_DIR / "etfs_latest.json",           # ETF universe mvt scores + reconstruction inputs
    "etf_meta": MOM_MVT_DIR / "etf_meta.json",          # ticker -> name/category/asset-class tags cache
    "weighting": MOM_DIR / "weighting.json",            # declared/equal/erc comparison, factors + horizons
    "status": MOM_MVT_DIR / "status.json",
    "validation": MOM_MVT_DIR / "validation.json",      # Phase 2: Models A/B/C/D backtest + correlation study
    "crossuniverse": MOM_MVT_DIR / "crossuniverse.json",  # Phase 3: equity-vs-sector-ETF broad/idiosyncratic split
}
# Note: the relative-strength NETWORK (section 22) is deliberately NOT
# precomputed/committed -- it's built live in the view from a user-chosen
# subset (mvt/network.py), the same "reconstruct on demand from the
# committed vectors, never persist a big derived artifact" pattern as the
# interactive pairwise matrix itself.

# --- Phase 2 validation (mvt/validate.py) -----------------------------------
# Models A (pure time-series) / B (pure cross-sectional) / C (pure
# multivariate trend) / D (full declared-weight composite) backtested with
# monthly rebalancing over whatever price history is already cached (see
# validate.py's own module docstring for why monthly, and the honest
# survivorship/history-depth limitations -- this is NOT a claim of covering
# 2008 or 2020, which would need a much deeper, slower repull AND would
# carry much more severe survivorship bias from projecting today's R1000
# list backward that many years).
MOM_MVT_VALIDATION_MONTHS = 58          # ~the full depth of the cached 5y equity price pull
MOM_MVT_VALIDATION_DECILE = 0.1         # top/bottom decile = the long/short bucket per model
MOM_MVT_VALIDATION_MODELS = ("A_timeseries", "B_crosssectional", "C_multivariate", "D_combined")
MOM_MVT_VALIDATION_MODEL_LABELS = {
    "A_timeseries": "Model A -- Traditional (time-series) trend",
    "B_crosssectional": "Model B -- Cross-sectional momentum",
    "C_multivariate": "Model C -- Multivariate trend (residual pairwise)",
    "D_combined": "Model D -- Combined (full declared-weight Momentum)",
}
# Stress windows identified EMPIRICALLY from the cached SPY series's own
# drawdown history (not assumed from real-world calendar dates) -- see
# validate.py's docstring. Only windows the cached history actually reaches
# are populated; anything earlier is honestly reported as unavailable
# rather than faked.
#
# "2022_drawdown" is INTENTIONALLY kept even though a live run shows it
# unavailable: SPY's own PRICE history reaches back to 2021-08, but the
# factor computations built on top of it (mom.factors.MIN_BARS=460,
# mvt's own MOM_MVT_MIN_BARS + MOM_MVT_COV_WINDOW=504) need ~1.8-2 years of
# TRAILING history before they can score anything at all -- so the first
# ~20 of the nominal 58 backtest months (verified: a 58-month run actually
# produced only 38 usable months) can't produce a single decile portfolio,
# and 2022 falls entirely inside that dead zone. Leaving the window
# DEFINED rather than deleting it means the validation report states this
# limitation explicitly (validate.summarize's by_stress_window output)
# instead of silently pretending the window was never asked for.
MOM_MVT_STRESS_WINDOWS = {
    "2022_drawdown": ("2022-01-01", "2022-10-15"),
    "2023_recovery": ("2023-01-01", "2023-12-31"),
    "2024_2025_bull": ("2024-01-01", "2026-08-31"),
}

# Horizons the multivariate engine scores, as (lookback_days, skip_days) --
# identical spec to mom.factors.HORIZON_SPEC so the two signals are directly
# comparable on the methodology panel.
MOM_MVT_HORIZON_SPEC = {
    "12_1": (252, 21),
    "12m": (252, 0),
    "9m": (189, 0),
    "6m": (126, 0),
    "3m": (63, 0),
    "1m": (21, 0),
}

# Minimum trading days of aligned history for a name to enter the mvt panel
# at all (needs a real 12M return AND enough tail for a stable covariance
# estimate over the trailing window used for the PCA factor model).
MOM_MVT_MIN_BARS = 280
# Trailing window for the covariance/PCA estimate AND the window residual
# returns are computed over. Two independent constraints set the floor:
#   (a) must exceed the longest horizon lookback (12M = 252 trading days)
#       by at least one day -- the 9-12M disjoint increment needs a return
#       252 days back, i.e. >=253 observations, or the longest horizons
#       silently come back empty.
#   (b) far more binding at R1000 scale: with N ~ 1000 instruments, a
#       covariance/PCA estimate needs T well above N or the eigen-structure
#       is dominated by sampling noise (classic N>>T random-matrix-theory
#       distortion -- Marchenko-Pastur). VERIFIED while building this: on a
#       synthetic 1000-name panel with a known ~11-factor structure, fitting
#       PCA on only 273 observations picked up spurious factors and
#       measurably degraded the residual signal (its correlation with total
#       cross-sectional momentum dropped to ~0.20 instead of the ~0.73 a
#       correctly-specified fit produces on the SAME data with a longer
#       window). 504 trading days (~2y) is comfortably affordable -- MOMENTUM
#       already pulls 5y of history for the whole Russell 1000 -- and keeps
#       N/T close to 2 rather than 4, which is what actually fixed it.
MOM_MVT_COV_WINDOW = 504

# Explicit leveraged/inverse ETF exclusion (name regex in mvt/universe.py
# catches most, but tickers whose fund name doesn't self-describe, or whose
# metadata cache hasn't refreshed yet, need a hard list). Kept short and
# reviewed, not exhaustive -- the empirical vol/correlation backstop in
# mvt/universe.py is the real safety net.
MOM_MVT_LEVERAGED_EXCLUDE = frozenset({
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXU", "SPXS", "UPRO", "SPXL", "TMF", "TMV",
    "TBT", "TBF", "TTT", "UVXY", "SVXY", "VIXY", "VXX", "SH", "PSQ", "DOG", "RWM",
    "DXD", "QID", "SDS", "SSO", "QLD", "DDM", "MVV", "TNA", "TZA", "FAS", "FAZ",
    "LABU", "LABD", "YINN", "YANG", "NAIL", "DRN", "DRV", "ERX", "ERY", "NUGT",
    "DUST", "JNUG", "JDST", "GUSH", "DRIP", "UCO", "SCO", "BOIL", "KOLD", "UGL",
    "ZSL", "AGQ", "UWM", "TWM", "URTY", "SRTY", "UDOW", "SDOW", "TECL", "TECS",
    "CURE", "PILL", "WEBL", "WEBS", "BNKU", "FNGU", "FNGD", "UMDD", "SMDD",
})

for _d in (MOM_MVT_DIR, MOM_MVT_HISTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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

# --- REGIMES (macro regime intelligence & early-warning system) ------------
# Growth/inflation four-quadrant classifier (S&P-style: two dimensions, many
# indicators each, a persistence requirement before a quadrant is "declared")
# PLUS six secondary dimensions (monetary, liquidity, credit, financial
# conditions, dollar, volatility) that run alongside it — see
# regimes/series.py for the indicator registry and regimes/classify.py for
# the methodology. Historical reconstruction runs on a MONTHLY grid (matches
# the cadence of the headline indicators themselves; daily series are
# resampled to month-end) using POINT-IN-TIME values only — every series is
# shifted by its own real publication lag (registry field) before being
# used, so no historical month's classification uses data that had not yet
# been published as of that month (spec section 42's look-ahead-bias rule).
# This eliminates *timing* look-ahead but not *revision* look-ahead (FRED
# revises payrolls/GDP for years); regimes/vintage.py is a separate, optional,
# LOCAL-ONLY audit that measures the residual with true ALFRED vintages on a
# handful of headline series and publishes it as a calibration caveat rather
# than silently ignoring it.
REGIMES_DIR = DATA_DIR / "regimes"
REGIMES_JOURNAL_DIR = REGIMES_DIR / "journal"        # sharded data/regimes/journal/<YYYY>.json
REGIMES_FILES = {
    "macro_raw": REGIMES_DIR / "macro_raw.json",       # committed raw per-series points (warm-start cache)
    "current": REGIMES_DIR / "current.json",           # today's classification + full explainability
    "timeline": REGIMES_DIR / "timeline.json",         # full monthly reconstruction + transition boundaries
    "dimensions": REGIMES_DIR / "dimensions.json",     # latest reading, all 8 dimensions
    "status": REGIMES_DIR / "status.json",
    "vintage_audit": REGIMES_DIR / "vintage_audit.json",  # optional, local-only (see regimes/vintage.py)
    # --- Phase 2 ---
    "transitions": REGIMES_DIR / "transitions.json",   # empirical base-rate transition probability tables
    "changes": REGIMES_DIR / "changes.json",           # "what is changing" deltas + Regime Change Score
    "crossasset": REGIMES_DIR / "crossasset.json",     # cross-asset confirmation checks + divergence flags
    "performance": REGIMES_DIR / "performance.json",   # asset-class/factor performance by historical regime
    "analogs": REGIMES_DIR / "analogs.json",           # nearest historical months + forward SPY outcomes
    "accuracy": REGIMES_DIR / "accuracy.json",         # calibration vs NBER USREC (lead/lag + in-sample Brier)
    # --- Phase 3 ---
    "themes": REGIMES_DIR / "themes.json",             # quant-scored + evidence-board theme monitors
    "scenarios": REGIMES_DIR / "scenarios.json",       # "What If?" contingency scenarios
    "alerts": REGIMES_DIR / "alerts.json",             # currently-active regime alerts
}

# Historical reconstruction starts here, not at each series' own inception:
# several inflation-expectation/credit-spread series (breakevens, AAA10Y)
# only begin in the 1980s-2000s, and the coverage-aware composite already
# down-weights thin coverage — but bounding the START keeps every
# reconstructed month at a reasonable breadth AND keeps the timeline anchored
# to canonical, checkable regimes (1990-92 slowdown, 1994-95 tightening,
# 1998 LTCM, 2000-02 dotcom, 2008-09 GFC, 2020 COVID shock, 2021-22
# inflation) — the exact validation set in the plan's verification section.
REGIMES_HISTORY_START = "1990-01-01"

# A quadrant must hold for this many CONSECUTIVE monthly readings before it
# is "declared" the current regime; short of that it renders as "emerging /
# transition underway" rather than a flip (spec section 44: communicate
# uncertainty, not false precision). Two months, not one, so a single noisy
# release cannot flip the headline.
REGIMES_PERSISTENCE_MONTHS = 2

# Rolling window (months) for z-scoring each indicator against its OWN
# history before combining into a dimension composite — trailing, not
# expanding-from-inception, so a composite from 2024 isn't implicitly graded
# against 1990s levels of a structurally different economy. min 24mo so no
# indicator is z-scored against fewer than 2 years of its own history.
REGIMES_ZSCORE_WINDOW_MONTHS = 120
REGIMES_ZSCORE_MIN_MONTHS = 24

# Minimum covered indicators for a dimension composite to be shown as "real"
# rather than "insufficient coverage" (spec section 29 applied to breadth,
# exactly IDEAS_GATES["min_coverage_n"]'s reasoning reused here).
REGIMES_MIN_COVERAGE = 3

for _d in (REGIMES_DIR, REGIMES_JOURNAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- INDEX (Master List: financial intelligence directory & knowledge graph) -
# A DIRECTORY, not a signal engine: it maps the information ecosystem (firms,
# people, academic sources, podcasts, tools) rather than scoring securities.
# Entities and relationships are stored SEPARATELY -- entities.json is the
# catalog, relationships.json is a plain edge list -- so Phase 2 (podcast guest
# edges) and Phase 3 (network rendering) add edges without reshaping the
# catalog. `seed/` holds the user's raw source lists committed verbatim as
# provenance, so any enrichment can always be traced back to what was supplied.
INDEX_DIR = DATA_DIR / "index"
INDEX_SEED_DIR = INDEX_DIR / "seed"          # raw supplied lists, committed as provenance
INDEX_FILES = {
    "entities": INDEX_DIR / "entities.json",           # the catalog
    "relationships": INDEX_DIR / "relationships.json",  # edge list {source,target,type}
    "links": INDEX_DIR / "links.json",                 # per-URL health from links.py
    "status": INDEX_DIR / "status.json",               # run health + counts + as-of
    # --- Phase 2 (podcast intelligence) — reserved, not yet written ---
    "podcasts": INDEX_DIR / "podcasts.json",           # feed registry + harvest state
    "episodes": INDEX_DIR / "episodes.json",           # harvested episode archive
}

# Link checking is a live network sweep, so it is rate-limited and never run
# from the view. A URL is re-checked only once its last check is this stale.
INDEX_LINK_TTL_DAYS = 14
INDEX_LINK_TIMEOUT = 12
INDEX_LINK_MAX_PER_RUN = 400        # ceiling on URLs probed in one compute run
INDEX_LINK_WORKERS = 8              # modest concurrency; these are other people's servers

# An entity needs this fraction of its "useful" fields populated before quality.py
# will call the profile complete. Deliberately not 1.0: many legitimate entries
# (a journal, a screener) have no founder, location or strategy tags and should
# not be perpetually flagged incomplete for lacking fields that do not apply.
INDEX_COMPLETENESS_TARGET = 0.6

# --- Phase 2: podcast intelligence ------------------------------------------
# Feed descriptions run to many KB of show notes, sponsor copy and timestamps,
# but a guest's affiliation is stated in the opening sentences. Keeping a
# bounded prefix is what lets ~5,700 episodes stay a sane committed artifact.
INDEX_EPISODE_SUMMARY_CHARS = 700

# A parsed guest name is only trusted enough to become a Person entity at or
# above this confidence. Lower-confidence parses are still RECORDED against the
# episode (so nothing is lost and the yield is auditable) but do not manufacture
# directory entries -- a directory full of misparsed fragments is worse than a
# smaller correct one.
INDEX_GUEST_MIN_CONFIDENCE = "medium"

# A firm named in episode metadata only becomes its own directory ENTITY once it
# has been seen this many times. A one-off mention is recorded as the guest's
# affiliation text (losing nothing) but does not manufacture an organisation
# stub with no URL -- otherwise several hundred unverifiable entries would swamp
# the review queue and drown the curated catalog.
INDEX_FIRM_ENTITY_MIN_MENTIONS = 3

# Appearances kept inline on a person's entity record. The full episode archive
# lives in episodes.json; the entity carries the most recent few so a profile is
# readable without loading thousands of rows.
INDEX_INLINE_APPEARANCES = 12

for _d in (INDEX_DIR, INDEX_SEED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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

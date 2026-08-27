"""MOMENTUM — a Russell 1000 multi-factor stock momentum engine for Zenith.

Every trading day, every current Russell 1000 constituent is scored on six
distinct, individually-documented momentum dimensions and blended into one
transparent composite in [-20, +20]:

  1. Time-series momentum   — is THIS stock's own trailing return positive or
     negative, vol-adjusted, across 12-1/12/9/6/3/1-month horizons
     (Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum").
  2. Breakout momentum      — has the stock's DAILY CLOSE broken its own
     trailing high/low over the same horizons (classic Donchian-style
     trend-following logic).
  3. Cross-sectional momentum — how strong is the stock's (vol-adjusted)
     return RELATIVE to the rest of the Russell 1000 (Jegadeesh & Titman
     1993, "Returns to Buying Winners and Selling Losers").
  4. Trend speed / GMMA     — how quickly is the moving-average structure
     itself changing: alignment, fresh crossovers, expansion/compression
     (Guppy Multiple Moving Averages, extended to 9..400 day averages).
  5. Momentum strength / trend quality — are moving-average slopes rising,
     falling, accelerating or decelerating, and is the trend statistically
     smooth (Da, Gurun & Warachka 2014, "Frog in the Pan: Continuous
     Information and Momentum") rather than one discrete jump.
  6. Multivariate Trend     — is the stock outperforming its PEERS across a
     broad pairwise relative-strength grid, and does that outperformance
     survive once common market/sector factors are statistically removed
     (Blitz, Huij & Martens 2011, "Residual Momentum"; see mom/mvt/ for the
     full pairwise engine, its own Equities/ETFs sub-tab, and the honest
     redundancy measurements that led to residualizing rather than using
     the naive pairwise spread).

The six factors are deliberately NOT equal-weighted (see config.MOM_WEIGHTS
and its inline rationale) and are NOT black-boxed: every score decomposes
into its six named contributions, and the app renders the factor
correlation matrix so redundancy claims are checked, not assumed.

Evidence tier B+: momentum is one of the most replicated cross-sectional
anomalies in finance, but it also crashes hard and predictably in panic
states (Daniel & Moskowitz 2016, "Momentum Crashes"; Barroso & Santa-Clara
2015), and large-cap momentum (this universe) is its most crowded corner.
This is decision-support and a research monitor, not investment advice, and
NOT a promise of forward returns.

Prices are yfinance daily OHLC with auto_adjust=True (splits AND dividends
applied) — the only self-consistent choice for both return calculations and
trailing-high/low breakout levels, since an unadjusted series breaks at every
split. Free / best-effort data only, computed offline in a GitHub Action that
commits JSON under data/mom/. Views read only committed JSON, except the
individual-stock GMMA chart, which fetches that one ticker's price history
on demand (cached) for chart fidelity the nightly artefacts don't carry.
"""

from __future__ import annotations

import json

from ..config import MOM_FILES

DISCLAIMER = ("MOMENTUM scores every current Russell 1000 constituent on six documented "
              "momentum dimensions (time-series, breakout, cross-sectional, trend speed, "
              "momentum strength, multivariate trend) and blends them into a transparent "
              "-20..+20 composite. Momentum is well-replicated academically but crashes hard "
              "in panic states and is most crowded in large caps (this universe). "
              "Decision-support and a research monitor, not investment advice.")

SURVIVORSHIP_NOTE = ("Scores from 2026-07-15 onward use the Russell 1000 as point-in-time "
                     "constituted (tracked via this app's own membership archive). Scores "
                     "before that date reuse TODAY's constituents and are survivorship-biased "
                     "upward — a name that was later removed from the index is absent from "
                     "those historical cross-sections. Historical charts mark the boundary.")

HORIZONS = ("12_1", "12m", "9m", "6m", "3m", "1m")
HORIZON_LABELS = {"12_1": "12-1M", "12m": "12M", "9m": "9M", "6m": "6M", "3m": "3M", "1m": "1M"}
FACTORS = ("ts", "breakout", "xsec", "speed", "strength", "mvt")
FACTOR_LABELS = {
    "ts": "Time-Series Momentum",
    "breakout": "Breakout Momentum",
    "xsec": "Cross-Sectional Momentum",
    "speed": "Trend Speed",
    "strength": "Momentum Strength",
    "mvt": "Multivariate Trend",
}


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(MOM_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write a MOM artefact. Large daily artefacts (scores/detail) should pass
    indent=None — at ~1000 rows/day, pretty-printing adds ~300MB/yr of git
    history for zero benefit (nobody diffs a 1000-row JSON by eye); the small
    artefacts (sectors/diagnostics/meta/status) keep indent=2 so diffs stay
    readable, matching every other Zenith feature package."""
    MOM_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    MOM_FILES[name].write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

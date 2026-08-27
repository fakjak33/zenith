"""Multivariate Trend — MOMENTUM's 6th factor: pairwise relative-strength /
residual momentum across an instrument's own universe.

Traditional momentum asks "is this instrument trending relative to its OWN
history" (time-series) or "is it outperforming the rest of the universe"
(cross-sectional, a single aggregate rank). This module asks a more specific
question inspired by Quantica's reported pairwise approach: for every
instrument, build the FULL set of pairwise relative-return relationships
against its peers, and see whether the outperformance survives once the
common (market/sector/style) factors shared by the whole universe are
stripped out.

Two layers are computed for every pair, at every horizon:

  * RAW    — vol-normalized total-return spread (r_A - r_B) / sigma_AB.
             This is the naive, spec-literal pairwise construction. A
             synthetic-panel test run before building this (see the plan
             file) found it Spearman ~0.99 correlated with plain
             cross-sectional momentum -- it is arithmetically dominated by
             each instrument's own return, so on its own it is cross-
             sectional momentum wearing a costume. Shipped anyway, honestly
             labelled, as the spec's own "Raw / Naive" score (see MOM
             multivariate-trend spec section 12).

  * RESIDUAL — the same spread computed on RESIDUAL returns, after removing
             a k-factor statistical (PCA) model of the whole universe's
             common movement (see factors.py). This is what actually
             injects cross-instrument correlation STRUCTURE into the signal
             rather than just re-deriving cross-sectional rank: the same
             synthetic-panel test found it only ~0.73-0.74 correlated with
             cross-sectional momentum -- genuinely different information.
             This is the "Normalized" score fed into the Momentum composite,
             and it corresponds to Blitz, Huij & Martens (2011) "Residual
             Momentum" (J. Empirical Finance).

Both layers preserve the full pairwise structure (peer win-rate, strongest/
weakest relationships, drill-down) -- residualizing does not throw any of
the spec's required pairwise detail away, it just changes what the spread is
measured ON.

The NxN pairwise matrix itself is a cheap, exactly-vectorized computation
(benchmarked at N=1000, T=504: ~0.07s, ~8MB) and is intentionally NEVER
committed to disk -- see pairwise.py's storage-architecture note. What IS
committed is the smaller set of inputs (return vectors + PCA loadings/
eigenvalues/idiosyncratic vols) that reconstruct any pair's spread exactly,
on demand, in the view.

Evidence tier: C+ on day one, matching this repo's own anti-overfitting
convention (see mom/__init__.py's DISCLAIMER). Residual momentum is a
published, replicated anomaly; THIS composite -- this universe, this factor
count, this horizon blend -- has no out-of-sample record of its own yet.
Promotion comes only from this app's own accumulating validation diagnostics
(Phase 2), never from a backtest tuned to look good.
"""

from __future__ import annotations

import json

from ...config import MOM_MVT_FILES

DISCLAIMER = (
    "Multivariate Trend scores every instrument against ALL of its peers (not just its own "
    "history) across six horizons, using both a raw vol-normalized spread and a residual "
    "spread with common market/sector factors removed. The residual score is genuinely less "
    "redundant with cross-sectional momentum than the raw one (measured, not assumed) and is "
    "what feeds the Momentum composite. New, unproven signal (evidence tier C+) -- decision-"
    "support and a research monitor, not investment advice."
)

HORIZONS = ("12_1", "12m", "9m", "6m", "3m", "1m")
HORIZON_LABELS = {"12_1": "12-1M", "12m": "12M", "9m": "9M", "6m": "6M", "3m": "3M", "1m": "1M"}
INCREMENTS = ("0_1m", "1_3m", "3_6m", "6_9m", "9_12m")
INCREMENT_LABELS = {
    "0_1m": "0-1M", "1_3m": "1-3M", "3_6m": "3-6M", "6_9m": "6-9M", "9_12m": "9-12M",
}
UNIVERSES = ("equities", "etfs")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(MOM_MVT_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write an mvt artefact. `equities`/`etfs` carry a per-horizon NxN-sized
    set of vectors for ~1000/900 names -- written COMPACT (indent=None),
    matching mom/__init__.py's documented convention for the same reason
    (pretty-printing ~1000 rows/day is pure git bloat, nobody diffs it by
    eye). Small artefacts (etf_meta/weighting/status) keep indent=2."""
    MOM_MVT_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    MOM_MVT_FILES[name].write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

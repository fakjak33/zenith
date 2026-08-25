"""REGIMES dimension composites — combine each dimension's per-series
z-scores (macro.py) into one composite-per-month, coverage-aware (spec
section 29 applied to breadth: a dimension scored from one series when it
usually has ten is a different kind of number and must say so, exactly
IDEAS_GATES["min_coverage_n"]'s reasoning, reused here as
config.REGIMES_MIN_COVERAGE).

Two dimensions (growth, inflation) feed the quadrant classifier (classify.py)
and are handled there in more depth (per-indicator breadth, not just the
composite). The other six are scored and labelled independently HERE and
never folded into the quadrant — spec section 12's "the four-quadrant
framework should be the top-level regime, not the entire system."
"""

from __future__ import annotations

import pandas as pd

from . import DIMENSIONS, DIMENSION_STATES, DIMENSION_NEUTRAL_BAND
from .series import BY_DIMENSION, BY_ID


def composite_series(z_df: pd.DataFrame, dimension: str) -> tuple[pd.Series, pd.Series]:
    """(composite, coverage) monthly Series for one dimension: composite is
    the mean of that month's AVAILABLE direction-adjusted z-scores (missing
    series don't drag it toward zero — same coverage-aware renormalization
    principle as mom.engine._weighted), coverage is how many were available."""
    cols = [s.id for s in BY_DIMENSION.get(dimension, ()) if s.id in z_df.columns]
    if not cols:
        empty = pd.Series(dtype=float, index=z_df.index)
        return empty, empty.astype("Int64")
    sub = z_df[cols]
    coverage = sub.notna().sum(axis=1)
    composite = sub.mean(axis=1, skipna=True)
    return composite, coverage


def state_label(dimension: str, z: float | None, min_coverage_ok: bool) -> str:
    if not min_coverage_ok or z is None or pd.isna(z):
        return "Insufficient coverage"
    pos, neg = DIMENSION_STATES.get(dimension, ("Elevated", "Depressed"))
    if z >= DIMENSION_NEUTRAL_BAND:
        return pos
    if z <= -DIMENSION_NEUTRAL_BAND:
        return neg
    return "Neutral"


def latest_indicator_rows(z_df: pd.DataFrame, raw_df: pd.DataFrame, dimension: str) -> list[dict]:
    """One row per series in this dimension, latest month: id, label, raw
    (post-transform, pre-direction — what a human reads), z (direction-
    adjusted), direction, rising (bool: is the direction-adjusted z above its
    own trailing-3-month value — the same test classify.py uses for growth/
    inflation breadth, applied here for the secondary dimensions' own
    explainability)."""
    rows = []
    for spec in BY_DIMENSION.get(dimension, ()):
        if spec.id not in z_df.columns:
            continue
        zc, rc = z_df[spec.id], raw_df.get(spec.id, pd.Series(dtype=float))
        if zc.dropna().empty:
            continue
        z_latest = zc.iloc[-1]
        z_prev3 = zc.iloc[-4] if len(zc) > 3 else None
        rows.append({
            "id": spec.id, "label": spec.label,
            "raw_value": None if rc.empty or pd.isna(rc.iloc[-1]) else round(float(rc.iloc[-1]), 4),
            "z": None if pd.isna(z_latest) else round(float(z_latest), 3),
            "direction": spec.direction,
            "rising": (None if (pd.isna(z_latest) or z_prev3 is None or pd.isna(z_prev3))
                      else bool(z_latest > z_prev3)),
            "lag_days": spec.lag_days, "freq": spec.freq,
        })
    return rows


def latest_summary(z_df: pd.DataFrame, raw_df: pd.DataFrame, min_coverage: int) -> dict[str, dict]:
    """Full current-reading summary for every SECONDARY dimension (excludes
    growth/inflation, which classify.py owns) — composite z, coverage,
    state label, and every indicator's latest reading for the explainability
    panel (spec section 43: never a bare label with no evidence listed)."""
    out = {}
    for dim in DIMENSIONS:
        if dim in ("growth", "inflation"):
            continue
        composite, coverage = composite_series(z_df, dim)
        n = int(coverage.iloc[-1]) if len(coverage) and not pd.isna(coverage.iloc[-1]) else 0
        z_latest = None if composite.empty or pd.isna(composite.iloc[-1]) else float(composite.iloc[-1])
        out[dim] = {
            "composite_z": None if z_latest is None else round(z_latest, 3),
            "coverage_n": n,
            "min_coverage": min_coverage,
            "state": state_label(dim, z_latest, n >= min_coverage),
            "indicators": latest_indicator_rows(z_df, raw_df, dim),
        }
    return out

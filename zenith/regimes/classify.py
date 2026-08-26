"""REGIMES quadrant classifier — growth x inflation, the top-level regime.

A quadrant is about DIRECTION (is growth/inflation accelerating or
decelerating), never the raw LEVEL — every institutional framework surveyed
for this feature (S&P's published regime research, Fidelity's business-cycle
model, GTAA-style growth/inflation quadrants) classifies this way, and it is
the only definition consistent with the spec's own worked example
("Overheating" = growth AND inflation both rising, regardless of whether
growth is merely below-trend-but-improving).

"Rising" for one axis = BREADTH of its own indicators' 3-month momentum, not
the sign of the composite alone — spec section 43 requires "N of M
indicators positive", so breadth IS the primary signal here, and the
composite z is used only as a magnitude/confidence input and as the
tie-breaker when breadth is exactly split.

PERSISTENCE (spec section 44, "communicate uncertainty, not false
precision"): the raw month-to-month quadrant call can flip on one release. A
quadrant becomes the DECLARED regime only after holding for
config.REGIMES_PERSISTENCE_MONTHS consecutive months; short of that, the
month is shown as the prior declared regime with "transition underway"
rather than a flip nobody would trust.
"""

from __future__ import annotations

import math

import pandas as pd

from ..config import REGIMES_MIN_COVERAGE, REGIMES_PERSISTENCE_MONTHS
from . import REGIME_LABELS
from .dimensions import composite_series, latest_indicator_rows


def _axis_breadth(z_df: pd.DataFrame, dimension_cols: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(breadth, n_rising, n_total) monthly Series for one axis: breadth is
    the fraction of covered indicators whose direction-adjusted z rose over
    the trailing 3 months (diff(3) > 0) — the exact test dimensions.py's
    per-indicator `rising` flag uses, so the timeline and the explainability
    cards agree with each other by construction."""
    if not dimension_cols:
        empty = pd.Series(dtype=float, index=z_df.index)
        return empty, empty, empty
    diff3 = z_df[dimension_cols].diff(3)
    total = diff3.notna().sum(axis=1)
    rising = (diff3 > 0).sum(axis=1).where(total > 0)
    breadth = (rising / total).where(total > 0)
    return breadth, rising, total.where(total > 0)


def _axis_state(breadth: float | None, composite_diff3: float | None) -> bool | None:
    if breadth is None or (isinstance(breadth, float) and math.isnan(breadth)):
        return None
    if breadth > 0.5:
        return True
    if breadth < 0.5:
        return False
    if composite_diff3 is None or (isinstance(composite_diff3, float) and math.isnan(composite_diff3)):
        return None
    return bool(composite_diff3 > 0)


def _confidence(breadth_g: float, n_g: float, breadth_i: float, n_i: float,
                full_n: int = 6) -> float | None:
    if any(pd.isna(v) for v in (breadth_g, n_g, breadth_i, n_i)):
        return None
    clarity_g, clarity_i = abs(breadth_g - 0.5) * 2.0, abs(breadth_i - 0.5) * 2.0
    cov_g, cov_i = min(1.0, n_g / full_n), min(1.0, n_i / full_n)
    return round(100.0 * 0.5 * (clarity_g * cov_g + clarity_i * cov_i), 1)


def classify_timeline(z_df: pd.DataFrame) -> pd.DataFrame:
    """The full monthly reconstruction: one row per month with the axis
    breadth/state, the raw (unpersisted) quadrant call, the PERSISTED
    declared regime, streak length, and confidence. This IS the timeline
    history.py reconstructs — classify.py owns the methodology, history.py
    owns turning it into a "Regime -> Transition -> New Regime" narrative."""
    from .series import BY_DIMENSION
    growth_cols = [s.id for s in BY_DIMENSION.get("growth", ()) if s.id in z_df.columns]
    infl_cols = [s.id for s in BY_DIMENSION.get("inflation", ()) if s.id in z_df.columns]

    growth_composite, growth_cov = composite_series(z_df, "growth")
    infl_composite, infl_cov = composite_series(z_df, "inflation")
    growth_breadth, growth_n_rising, growth_n_total = _axis_breadth(z_df, growth_cols)
    infl_breadth, infl_n_rising, infl_n_total = _axis_breadth(z_df, infl_cols)
    growth_diff3, infl_diff3 = growth_composite.diff(3), infl_composite.diff(3)

    rows = []
    declared, streak, last_raw = None, 0, None
    for t in z_df.index:
        bg, bi = growth_breadth.get(t), infl_breadth.get(t)
        ng, ni = growth_n_total.get(t, 0), infl_n_total.get(t, 0)
        g_rising = _axis_state(bg, growth_diff3.get(t))
        i_rising = _axis_state(bi, infl_diff3.get(t))
        enough_coverage = (not pd.isna(ng) and ng >= REGIMES_MIN_COVERAGE
                          and not pd.isna(ni) and ni >= REGIMES_MIN_COVERAGE)
        raw_regime = (REGIME_LABELS.get((g_rising, i_rising))
                     if enough_coverage and g_rising is not None and i_rising is not None else None)

        if raw_regime == last_raw and raw_regime is not None:
            streak += 1
        else:
            streak = 1 if raw_regime is not None else 0
        last_raw = raw_regime

        if raw_regime is not None and streak >= REGIMES_PERSISTENCE_MONTHS:
            declared = raw_regime
        transitioning = raw_regime is not None and raw_regime != declared

        rows.append({
            "month": t, "growth_z": growth_composite.get(t), "growth_breadth": bg,
            "growth_n_rising": None if pd.isna(ng) else int(growth_n_rising.get(t, 0)),
            "growth_n_total": None if pd.isna(ng) else int(ng),
            "growth_rising": g_rising,
            "infl_z": infl_composite.get(t), "infl_breadth": bi,
            "infl_n_rising": None if pd.isna(ni) else int(infl_n_rising.get(t, 0)),
            "infl_n_total": None if pd.isna(ni) else int(ni),
            "infl_rising": i_rising,
            "raw_regime": raw_regime, "declared_regime": declared, "streak": streak,
            "transitioning": transitioning,
            "confidence": _confidence(bg, ng, bi, ni),
        })
    return pd.DataFrame(rows).set_index("month")


def current(z_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict:
    """The latest month's classification PLUS the per-indicator breakdown
    every card needs (spec section 43: never a bare label)."""
    timeline = classify_timeline(z_df)
    if timeline.empty:
        return {"regime": None, "confidence": None, "note": "no data"}
    last = timeline.iloc[-1]

    def _int_or_none(v):
        return None if v is None or pd.isna(v) else int(v)

    return {
        # NOTE: named "latest_month", not "as_of" -- compute.py's caller does
        # {"as_of": today.isoformat(), **current}, and a same-named key here
        # would silently win the dict-literal merge, replacing the actual
        # compute-run date with this timeline month-end label (which can be
        # a few days into the FUTURE relative to today, since the monthly
        # grid labels the in-progress current month by its eventual
        # month-end). A real bug, caught live: the app showed "DATA AS OF
        # 2026-08-31" while the actual run date was 2026-08-25.
        "latest_month": timeline.index[-1].date().isoformat(),
        "regime": last["declared_regime"],
        "raw_regime": last["raw_regime"],
        "transitioning": bool(last["transitioning"]),
        "streak_months": int(last["streak"]),
        "confidence": last["confidence"],
        "growth": {
            "rising": last["growth_rising"], "breadth": last["growth_breadth"],
            "n_rising": _int_or_none(last["growth_n_rising"]), "n_total": _int_or_none(last["growth_n_total"]),
            "z": None if pd.isna(last["growth_z"]) else round(float(last["growth_z"]), 3),
            "indicators": latest_indicator_rows(z_df, raw_df, "growth"),
        },
        "inflation": {
            "rising": last["infl_rising"], "breadth": last["infl_breadth"],
            "n_rising": _int_or_none(last["infl_n_rising"]), "n_total": _int_or_none(last["infl_n_total"]),
            "z": None if pd.isna(last["infl_z"]) else round(float(last["infl_z"]), 3),
            "indicators": latest_indicator_rows(z_df, raw_df, "inflation"),
        },
    }

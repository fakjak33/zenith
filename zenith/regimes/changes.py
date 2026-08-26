"""REGIMES "What Is Changing?" (spec section 6) + the Regime Change Score
(spec section 27).

Rather than a static "inflation = 3.1%" reading, this shows MOMENTUM deltas —
which indicators flipped direction recently — across ALL eight dimensions,
then rolls that up into one 0-100 Regime Change Score measuring the BREADTH
and MAGNITUDE of change across independent indicators (spec section 27's own
words: "avoid triggering a major regime alert because of one noisy
indicator" — this is why the score is a breadth measure across many series,
not a threshold on any single one).
"""

from __future__ import annotations

import pandas as pd

from .series import BY_ID

CHANGE_SCORE_BANDS = (
    (80.0, "Major Regime Shift"), (60.0, "Significant Transition"),
    (40.0, "Emerging"), (20.0, "Early Signals"), (0.0, "Stable"),
)


def band_for(score: float) -> str:
    for threshold, label in CHANGE_SCORE_BANDS:
        if score >= threshold:
            return label
    return CHANGE_SCORE_BANDS[-1][1]


def indicator_deltas(z_df: pd.DataFrame) -> list[dict]:
    """Every indicator's z-score change over the trailing 1 and 3 months —
    "inflation momentum ↑, wage pressure ↑" made concrete and per-series."""
    if z_df.empty:
        return []
    rows = []
    for sid in z_df.columns:
        spec = BY_ID.get(sid)
        s = z_df[sid].dropna()
        if len(s) < 4 or spec is None:
            continue
        d1 = s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else None
        d3 = s.iloc[-1] - s.iloc[-4] if len(s) >= 4 else None
        rows.append({
            "id": sid, "label": spec.label, "dimension": spec.dimension,
            "z": round(float(s.iloc[-1]), 3),
            "delta_1m": None if d1 is None else round(float(d1), 3),
            "delta_3m": None if d3 is None else round(float(d3), 3),
            "direction_1m": None if d1 is None else ("up" if d1 > 0 else ("down" if d1 < 0 else "flat")),
        })
    return rows


def regime_change_score(z_df: pd.DataFrame, window_months: int = 2) -> dict:
    """0-100: the fraction of ALL covered indicators (across all 8
    dimensions, not just growth/inflation) that FLIPPED direction (their
    trailing-window delta changed sign) over `window_months`, scaled to
    0-100 and weighted by the average MAGNITUDE of the flips so a broad,
    forceful move scores higher than a broad but marginal one."""
    if z_df.shape[0] < window_months + 2:
        return {"score": None, "band": None, "n_flipped": 0, "n_covered": 0}
    prev_delta = z_df.diff(1).iloc[-(window_months + 1)]
    cur_delta = z_df.diff(1).iloc[-1]
    covered = prev_delta.notna() & cur_delta.notna()
    n_covered = int(covered.sum())
    if n_covered == 0:
        return {"score": None, "band": None, "n_flipped": 0, "n_covered": 0}
    flipped = covered & (((prev_delta > 0) & (cur_delta < 0)) | ((prev_delta < 0) & (cur_delta > 0)))
    n_flipped = int(flipped.sum())
    breadth = n_flipped / n_covered
    magnitude = float(cur_delta[flipped].abs().mean()) if n_flipped else 0.0
    magnitude_factor = min(1.0, magnitude / 1.0)   # a full 1-sigma-unit flip saturates the magnitude term
    score = round(100.0 * breadth * (0.5 + 0.5 * magnitude_factor), 1)
    return {"score": score, "band": band_for(score), "n_flipped": n_flipped, "n_covered": n_covered,
           "breadth": round(breadth, 3), "avg_flip_magnitude": round(magnitude, 3)}

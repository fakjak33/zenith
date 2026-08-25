"""REGIMES momentum — are the forces DEFINING the current regime strengthening
or weakening (spec section 7), independent of the regime label itself.

Reuses `mom.normalize.ols_slope_r2` (this repo's one existing trend-slope
primitive, built for MOMENTUM's "momentum strength" factor and documented
there as "no polyfit/linregress/gradient exists elsewhere in this repo") on
the trailing 6 months of each axis's composite z — R² weights the slope by
how CLEAN the trend actually is, so a noisy wiggle contributes less than a
steady move of the same magnitude.
"""

from __future__ import annotations

import pandas as pd

from ..mom.normalize import ols_slope_r2

TREND_WINDOW_MONTHS = 6


def regime_momentum(growth_composite: pd.Series, infl_composite: pd.Series,
                    regime_label: str | None) -> dict:
    """Signed momentum score in roughly [-100, 100]: positive means the
    forces that DEFINE the current regime are strengthening. Sign convention
    is regime-relative — e.g. under Overheating (growth rising, inflation
    rising) a rising growth slope helps the score and a rising inflation
    slope ALSO helps (both axes are "supposed" to be rising); under
    Goldilocks a rising inflation slope HURTS the score even though it isn't
    inherently changing the growth read."""
    g_slope, g_r2 = ols_slope_r2(growth_composite, window=TREND_WINDOW_MONTHS)
    i_slope, i_r2 = ols_slope_r2(infl_composite, window=TREND_WINDOW_MONTHS)
    if g_slope is None or i_slope is None:
        return {"score": None, "growth_slope": g_slope, "growth_r2": g_r2,
                "infl_slope": i_slope, "infl_r2": i_r2, "narrative": "Insufficient history."}

    # which direction each axis is "supposed" to move for the CURRENT regime
    growth_wants_up = regime_label in ("Goldilocks / Reflation", "Overheating")
    infl_wants_up = regime_label in ("Overheating", "Stagflation")
    g_dir = 1 if growth_wants_up else -1
    i_dir = 1 if infl_wants_up else -1

    raw = 50.0 * (g_dir * g_slope * (g_r2 or 0) + i_dir * i_slope * (i_r2 or 0))
    score = round(max(-100.0, min(100.0, raw)), 1)

    g_word = "strengthened" if g_slope > 0 else "weakened"
    i_word = "accelerated" if i_slope > 0 else "moderated"
    narrative = (f"Growth signals have {g_word} over the trailing {TREND_WINDOW_MONTHS} months "
                f"(trend R²={g_r2:.2f}) while inflation signals have {i_word} "
                f"(trend R²={i_r2:.2f}).")
    return {"score": score, "growth_slope": round(g_slope, 4), "growth_r2": round(g_r2, 3),
           "infl_slope": round(i_slope, 4), "infl_r2": round(i_r2, 3), "narrative": narrative}

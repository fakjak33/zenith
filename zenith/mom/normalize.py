"""MOMENTUM normalization primitives — pure functions, no I/O, no network.

Every factor in this package outputs a score in [-1, +1]. This module is the
one place that logic lives, so the normalization each factor uses is legible
without hunting through five files (per the project's own "document exactly
how each factor is normalized" requirement).

Three normalizations are used across the five factors:
  * winsorize_xs  — clip cross-sectional outliers before scaling anything
                     (reuses zenith.fmom.core.winsorize, the repo's only
                     existing winsorizer).
  * tanh_clip     — a smooth, bounded squashing function for a
                     vol-adjusted / z-like statistic. Preferred over a hard
                     clip because it compresses tails continuously rather
                     than flattening them at a wall.
  * inv_normal    — the van der Waerden inverse-normal transform of a
                     percentile rank. Spreads the crowded middle of a
                     cross-sectional rank distribution and compresses the
                     tails less aggressively than a raw percentile map, which
                     is the standard alternative to a plain linear mapping
                     for turning a rank into a score.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..fmom.core import winsorize as _fmom_winsorize
from ..edge.common import pct_ranks


def winsorize_xs(values: list[float], p: float = 0.01) -> list[float]:
    """Cross-sectional winsorization at the p / 1-p quantiles (default 1%).

    Thin wrapper around `fmom.core.winsorize` (a pandas Series clip at the
    p/1-p quantiles) so callers can stay in plain lists/dicts.
    """
    if not values:
        return []
    s = _fmom_winsorize(pd.Series(values, dtype=float), p)
    return s.tolist()


def pct_rank_xs(values: list[float]) -> list[float]:
    """0-100 cross-sectional percentile rank, ties averaged. Reuses
    `edge.common.pct_ranks` — the repo's one existing percentile-rank
    implementation — rather than a second copy."""
    return pct_ranks(values)


def tanh_clip(x: float, scale: float = 1.0) -> float:
    """Smooth bounded squash to (-1, +1): tanh(x / scale). NaN/inf -> 0.0."""
    if x is None or not math.isfinite(x):
        return 0.0
    return float(math.tanh(x / scale)) if scale else 0.0


def clip1(x: float) -> float:
    """Hard clamp to [-1, 1]. NaN/inf -> 0.0."""
    if x is None or not math.isfinite(x):
        return 0.0
    return max(-1.0, min(1.0, float(x)))


def inv_normal(pctile_0_100: float, floor: float = 0.005, ceil: float = 0.995) -> float:
    """Van der Waerden inverse-normal transform of a 0-100 percentile rank,
    scaled to land in roughly [-1, +1].

    A raw linear percentile map (p/50 - 1) treats the crowded middle of the
    cross-section the same as the sparse tails. Passing the percentile
    through the inverse standard-normal CDF spreads the middle (where most of
    the discriminating information actually lives, because with N~1000 names
    ranks cluster tightly) and lets the true tail names stand out, while the
    `floor`/`ceil` clamp keeps the very extremes from blowing up (Phi^-1(1)
    is infinite). Divide by 2 so a name at the 99.5th percentile scores
    ~+1.29, comfortably inside the [-1, 1] band without a second clip.
    """
    if pctile_0_100 is None or not math.isfinite(pctile_0_100):
        return 0.0
    p = max(floor, min(ceil, float(pctile_0_100) / 100.0))
    z = float(_ndtri(p))
    return clip1(z / 2.0)


def _ndtri(p: float) -> float:
    """Inverse standard-normal CDF (probit). numpy has no public ndtri, but
    scipy is not a Zenith dependency, so this is Acklam's rational
    approximation (max abs error ~1.15e-9), which is more than sufficient for
    a display-scale momentum signal."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    # Coefficients for the rational approximation (Peter Acklam, 2003).
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def ols_slope_r2(y: pd.Series, window: int | None = None) -> tuple[float | None, float | None]:
    """OLS slope and R-squared of `y` regressed on a simple time index
    (0..n-1), i.e. the trend slope and how well it fits.

    New primitive: no polyfit/linregress/gradient exists elsewhere in this
    repo. Used both for the momentum-strength "trend quality" component
    (fit on log(price)) and available for slope-of-anything callers.

    Returns (None, None) if fewer than 3 finite points are available.
    `window` takes only the trailing N observations of `y` first.
    """
    s = y.dropna().astype(float)
    if window:
        s = s.tail(window)
    n = len(s)
    if n < 3:
        return None, None
    x = np.arange(n, dtype=float)
    yv = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(yv)):
        mask = np.isfinite(yv)
        x, yv = x[mask], yv[mask]
        n = len(yv)
        if n < 3:
            return None, None
    x_mean, y_mean = x.mean(), yv.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx <= 0:
        return None, None
    sxy = ((x - x_mean) * (yv - y_mean)).sum()
    slope = float(sxy / sxx)
    yhat = y_mean + slope * (x - x_mean)
    ss_res = float(((yv - yhat) ** 2).sum())
    ss_tot = float(((yv - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return slope, max(0.0, min(1.0, r2))

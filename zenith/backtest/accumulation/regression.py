"""Rolling linear & logarithmic regression channels with ±k·σ bands.

A trend-line fit over a trailing window plus standard-deviation bands of the
residuals. Log mode fits log(price) (straight line on a log chart = constant
growth rate), which suits long-horizon equity/crypto accumulation. Buy near the
lower band, sell near the upper band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_channel(close: pd.Series, lookback: int = 252, k: float = 2.0,
                       log: bool = True) -> pd.DataFrame:
    """Return DataFrame [fit, upper, lower, z] aligned to ``close``.

    fit = rolling regression value; upper/lower = fit ± k·σ(residuals); z =
    standardized distance of price from the fit (negative = cheap).
    """
    y_all = np.log(close.clip(lower=1e-9)) if log else close.astype(float)
    n = len(close)
    lookback = max(10, int(lookback))
    fit = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    x = np.arange(lookback, dtype=float)
    xc = x - x.mean()
    denom = (xc**2).sum()

    yv = y_all.to_numpy()
    for i in range(lookback - 1, n):
        win = yv[i - lookback + 1: i + 1]
        slope = (xc * (win - win.mean())).sum() / denom
        intercept = win.mean()  # centered x → intercept at window midpoint mean
        pred_last = intercept + slope * xc[-1]
        resid = win - (intercept + slope * xc)
        fit[i] = pred_last
        sd[i] = resid.std(ddof=1)

    fit_s = pd.Series(fit, index=close.index)
    sd_s = pd.Series(sd, index=close.index)
    upper = fit_s + k * sd_s
    lower = fit_s - k * sd_s
    z = (y_all - fit_s) / sd_s.replace(0.0, np.nan)

    if log:
        out = pd.DataFrame({"fit": np.exp(fit_s), "upper": np.exp(upper),
                            "lower": np.exp(lower), "z": z})
    else:
        out = pd.DataFrame({"fit": fit_s, "upper": upper, "lower": lower, "z": z})
    return out

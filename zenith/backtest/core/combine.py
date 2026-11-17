"""Forecast combination with a Forecast Diversification Multiplier (FDM).

Combining several rules' forecasts reduces noise, but a simple weighted
average of correlated forecasts shrinks magnitude below the ~10 target. The
FDM scales the combination back up based on how diversifying the rules are
(lower average correlation => higher FDM). Result is re-capped at +/- 20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORECAST_CAP


def forecast_diversification_multiplier(forecasts: pd.DataFrame, weights: np.ndarray) -> float:
    """FDM = 1 / sqrt(w' C w), where C is the forecast correlation matrix.

    Clamped to [1, 2.5] as Carver recommends to avoid runaway leverage from
    noisy correlation estimates.
    """
    if forecasts.shape[1] == 1:
        return 1.0
    corr = forecasts.corr().fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 1.0)
    var = float(weights @ corr @ weights)
    if var <= 1e-12:
        return 1.0
    return float(np.clip(1.0 / np.sqrt(var), 1.0, 2.5))


def combine_forecasts(
    forecasts: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Weighted blend of scaled forecasts, FDM-adjusted and re-capped.

    ``forecasts`` maps rule-name -> already-scaled forecast series. Equal
    weights are used if ``weights`` is None.
    """
    df = pd.DataFrame(forecasts).fillna(0.0)
    names = list(df.columns)

    if weights is None:
        w = np.full(len(names), 1.0 / len(names))
    else:
        w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
        total = w.sum()
        w = w / total if total > 0 else np.full(len(names), 1.0 / len(names))

    fdm = forecast_diversification_multiplier(df, w)
    combined = (df.to_numpy() @ w) * fdm
    return pd.Series(combined, index=df.index).clip(-cap, cap)

"""Kelly criterion position sizing helpers.

Two flavors:
  - Discrete Kelly from a win probability and payoff ratio (the classic
    f* = p - q/b).
  - Continuous Kelly from a return series (f* = mean / variance), which gives
    the growth-optimal leverage; most quants use a fraction of it (half-Kelly)
    because full Kelly is very volatile and sensitive to estimation error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def kelly_discrete(win_prob: float, win_loss_ratio: float) -> float:
    """Classic Kelly fraction: f* = p - (1-p)/b. Clamped to [0, 1]."""
    p = min(max(win_prob, 0.0), 1.0)
    b = max(win_loss_ratio, 1e-9)
    f = p - (1.0 - p) / b
    return float(max(0.0, min(1.0, f)))


def kelly_continuous(returns: pd.Series, periods: int = TRADING_DAYS) -> dict[str, float]:
    """Continuous Kelly leverage from a return stream.

    Returns per-period and annualized full-Kelly leverage plus half-Kelly.
    f* = mean / variance (per period). Annualized leverage is the same number
    (leverage is unitless) but we also report the implied annual growth proxy.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    var = r.var(ddof=1)
    if len(r) < 8 or var < 1e-12:
        return {"full": 0.0, "half": 0.0, "quarter": 0.0}
    f = float(r.mean() / var)
    return {"full": f, "half": f / 2.0, "quarter": f / 4.0}

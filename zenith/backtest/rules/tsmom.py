"""Time-series momentum (Moskowitz, Ooi & Pedersen 2012).

Forecast = past N-day return scaled by volatility. Positive past return =>
long. This is the academically canonical 'trend' factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.volatility import ew_vol, bars_per_year


@register
class TimeSeriesMomentum(Strategy):
    key = "tsmom"
    label = "Time-Series Momentum"
    params = {
        "lookback": (90, 10, 2000, 5),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "lookback"

    plain = (
        "Looks at whether an asset went up or down over the last several months, and bets "
        "the same direction continues."
    )
    analogy = (
        "Like judging a football team by their record this season rather than their "
        "reputation. Whoever has been winning is the one you back next week."
    )
    how_it_works = (
        "Measure the total return over the lookback window - often about a year.",
        "Divide by how jumpy the asset is, so 10% in a calm bond counts more than in a wild stock.",
        "Positive past return means buy; negative means sell short.",
        "Position size scales with how strong that risk-adjusted past move was.",
    )
    works_when = (
        "Almost everywhere, historically - this is the single best-replicated effect in "
        "the whole app, found across stocks, bonds, currencies and commodities over more "
        "than a century."
    )
    fails_when = (
        "Sharp reversals at turning points, and crisis periods specifically: research "
        "finds the predictability trend-following relies on can weaken to roughly a third "
        "of its normal strength when markets are in crisis."
    )
    evidence = (
        "Moskowitz, Ooi & Pedersen (2012), the canonical time-series momentum paper. See "
        "[Alpha Architect's review of the historical evidence]"
        "(https://alphaarchitect.com/time-series-momentum-aka-trend-following-the-historical-evidence/) "
        "for the century-plus replication and the caveats about shorter horizons."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        past_ret = close / close.shift(lb) - 1.0
        # scale by annualized vol so the signal is risk-adjusted
        vol_annual = ew_vol(close, annualize=True)
        sharpe_like = past_ret / (vol_annual * np.sqrt(lb / bars_per_year(close.index)))
        fc = sharpe_like.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

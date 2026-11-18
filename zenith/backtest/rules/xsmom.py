"""Cross-sectional momentum (Jegadeesh & Titman 1993).

Ranks assets by past return across the universe and goes long winners /
short losers. This rule is inherently multi-asset, so it overrides
``forecast_panel`` rather than ``raw_forecast``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import scale_forecast


@register
class CrossSectionalMomentum(Strategy):
    key = "xsmom"
    label = "Cross-Sectional Momentum"
    cross_sectional = True
    params = {
        "lookback": (126, 20, 2000, 5),
        "skip": (21, 0, 120, 1),   # skip most-recent days (reversal control)
    }
    spectrum_param = "lookback"

    plain = (
        "Compares all the assets in your list against each other, buys the ones that have "
        "risen most and sells the ones that have risen least."
    )
    analogy = (
        "Like a school sports day ranking. It does not matter how fast anyone ran in "
        "absolute terms - only who finished ahead of whom."
    )
    how_it_works = (
        "Measure each asset's return over the lookback window.",
        "Deliberately skip the most recent few weeks, because very recent moves tend to reverse.",
        "Rank everything against everything else and score it relative to the group average.",
        "Go long the top of the ranking and short the bottom.",
    )
    works_when = (
        "You give it several instruments to compare - it needs a field to rank. With one "
        "ticker selected it has nothing to do."
    )
    fails_when = (
        "'Momentum crashes' - sharp rebounds after a market bottom, when the beaten-down "
        "losers you are short rocket upward."
    )
    evidence = (
        "Jegadeesh & Titman (1993), one of the most cited findings in finance and the "
        "origin of the momentum factor. Note it is a *relative* bet, so it can lose money "
        "in a rising market if your longs rise less than your shorts."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError("xsmom is cross-sectional; use forecast_panel.")

    def forecast_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        lb = int(self.values["lookback"])
        skip = int(self.values["skip"])
        closes = pd.DataFrame({t: df["close"] for t, df in panel.items()}).sort_index()
        past_ret = closes.shift(skip) / closes.shift(lb) - 1.0

        # cross-sectional z-score each day => demeaned long/short forecast
        mean = past_ret.mean(axis=1)
        std = past_ret.std(axis=1).replace(0.0, np.nan)
        z = past_ret.sub(mean, axis=0).div(std, axis=0)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # scale each column to the standard forecast magnitude
        return pd.DataFrame({c: scale_forecast(z[c]) for c in z.columns})

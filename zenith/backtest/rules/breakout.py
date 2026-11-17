"""Donchian channel breakout.

Forecast = position of price within its N-day high/low range, centered and
scaled so a fresh N-day high reads near +20 and a fresh low near -20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class Breakout(Strategy):
    key = "breakout"
    # Renamed from "Donchian Breakout" so it is not confused with the Turtle rule,
    # which is what most people mean by a Donchian breakout. Registry key unchanged.
    label = "Donchian Channel Position"
    params = {
        "lookback": (40, 5, 1000, 1),
        "smooth": (10, 1, 200, 1),
    }
    spectrum_param = "lookback"

    plain = (
        "Measures where today's price sits inside its recent high-to-low range. Near the "
        "top of the range it buys, near the bottom it sells."
    )
    analogy = (
        "Like a fuel gauge for the recent trading range. Full tank means price is pressing "
        "against its recent highs; empty means it is scraping the lows."
    )
    how_it_works = (
        "Find the highest and lowest price of the last N bars.",
        "Work out where today sits between them, as a position from 0 to 1.",
        "Recentre so mid-range is zero, the top is positive and the bottom negative.",
        "Smooth it, so it drifts rather than flipping on a single bar.",
    )
    works_when = "Prices grind steadily toward one end of their range and keep going."
    fails_when = "Prices oscillate within a stable range - the signal just swings with them."
    evidence = (
        "NOTE: this is a smooth *position-in-range* signal, not a true breakout rule. It "
        "is always in the market to some degree and never fires a discrete entry. If you "
        "want the classic buy-the-new-high behaviour with pyramiding and channel exits, "
        "use **Turtle Breakout (Dennis)** instead - the two are often confused."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        roll_max = close.rolling(lb).max()
        roll_min = close.rolling(lb).min()
        rng = (roll_max - roll_min).replace(0.0, np.nan)
        # position in range: 0 at low, 1 at high -> center to [-0.5, 0.5]
        pos = (close - roll_min) / rng - 0.5
        smoothed = pos.rolling(int(self.values["smooth"])).mean()
        return smoothed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        return {
            f"Donchian high ({lb})": close.rolling(lb).max(),
            f"Donchian low ({lb})": close.rolling(lb).min(),
        }

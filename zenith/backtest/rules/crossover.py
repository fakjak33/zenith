"""Generic MA crossover with a binary-ish, smoothed forecast.

Distinct from the EWMAC trend rule: this measures the *percentage* gap
between two SMAs (a classic golden/death-cross proxy) and smooths it, giving
a forecast that saturates rather than scaling with raw price difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class Crossover(Strategy):
    key = "crossover"
    label = "MA Crossover (%gap)"
    params = {
        "fast": (50, 5, 1000, 1),
        "slow": (200, 20, 2000, 1),
        "smooth": (5, 1, 100, 1),
    }
    spectrum_param = "fast"

    plain = (
        "The classic 'golden cross / death cross'. It measures how far apart two average "
        "prices are as a percentage, so a 50-day average above a 200-day one is a buy."
    )
    analogy = (
        "Like comparing this month's average temperature to this year's. If the month is "
        "running 5% warmer, summer is coming - and the size of that gap is the signal."
    )
    how_it_works = (
        "Take a short average and a long average of the price.",
        "Express the gap between them as a percentage of the long average.",
        "Smooth that percentage so single odd days do not whipsaw you.",
        "A positive percentage buys, a negative one sells.",
    )
    works_when = "Slow, major trend changes - this is a deliberately sluggish rule."
    fails_when = (
        "You need to react quickly. By the time a 50/200 cross confirms, a large part of "
        "the move has already happened."
    )
    evidence = (
        "The most widely watched signal in retail technical analysis. Included partly so "
        "you can test it honestly rather than take the folklore on trust - compare its "
        "Sharpe to the EMA rule on the same data."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = close.rolling(int(self.values["fast"])).mean()
        slow = close.rolling(int(self.values["slow"])).mean()
        pct_gap = (fast - slow) / slow.replace(0.0, np.nan)
        return pct_gap.rolling(int(self.values["smooth"])).mean().fillna(0.0)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"SMA fast ({int(self.values['fast'])})": close.rolling(int(self.values["fast"])).mean(),
            f"SMA slow ({int(self.values['slow'])})": close.rolling(int(self.values["slow"])).mean(),
        }

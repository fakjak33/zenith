"""Guppy Multiple Moving Average (GMMA).

Two EMA ribbons: a short-term group (traders) and a long-term group
(investors). Forecast = separation between the two ribbon means, normalized
by price vol. Wide positive separation => strong, agreed-upon uptrend.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol

_SHORT = (3, 5, 8, 10, 12, 15)
_LONG = (30, 35, 40, 45, 50, 60)


@register
class GMMA(Strategy):
    key = "gmma"
    label = "Guppy MMA"
    params = {
        # a single 'speed' multiplier scales both ribbons for the spectrum
        "speed": (1.0, 0.5, 6.0, 0.1),
    }
    spectrum_param = "speed"

    plain = (
        "Uses two whole bundles of averages - a fast bundle representing short-term "
        "traders and a slow bundle representing long-term investors - and buys when the "
        "two groups clearly agree the market is rising."
    )
    analogy = (
        "Like checking whether both the day-trippers and the season-ticket holders are "
        "buying. When both crowds move the same way and the gap between them widens, "
        "the move has broad agreement behind it."
    )
    how_it_works = (
        "Build six fast averages (the traders) and six slow ones (the investors).",
        "Average each bundle to get one line per group.",
        "Measure the gap between the two lines, adjusted for how jumpy the price is.",
        "A wide, positive gap means both crowds agree it is going up - buy strongly.",
    )
    works_when = "A trend has genuine breadth of participation, not just a one-day spike."
    fails_when = (
        "The bundles tangle together in a sideways market, producing a signal near zero "
        "and a lot of small, pointless trades."
    )
    evidence = (
        "Daryl Guppy's Multiple Moving Average, a practitioner technique. Weaker academic "
        "support than trend or carry - treat it as a variation on trend-following rather "
        "than an independent effect, and check its correlation to the EMA rule in the "
        "Diagnostics tab before assuming it adds anything."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        speed = float(self.values["speed"])
        short = pd.concat(
            [close.ewm(span=max(2, int(s * speed))).mean() for s in _SHORT], axis=1
        ).mean(axis=1)
        long = pd.concat(
            [close.ewm(span=max(3, int(s * speed))).mean() for s in _LONG], axis=1
        ).mean(axis=1)
        vol = ew_vol(close, annualize=False) * close
        return normalize_by_vol(short - long, vol)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        speed = float(self.values["speed"])
        lines = {}
        for s in _SHORT:
            lines[f"short {int(s*speed)}"] = close.ewm(span=max(2, int(s * speed))).mean()
        for s in _LONG:
            lines[f"long {int(s*speed)}"] = close.ewm(span=max(3, int(s * speed))).mean()
        return lines

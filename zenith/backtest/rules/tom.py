"""Turn-of-the-month seasonality.

Equity indices have historically earned an outsized share of their total return
in a narrow window straddling the month boundary: the last few trading days of
one month and the first few of the next. This rule holds the market only in
that window and sits in cash the rest of the time.

Pure calendar logic — it reads nothing but the DatetimeIndex, so it needs no
extra data and cannot be curve-fit to price history beyond its two parameters.

Like ``turtle`` it overrides ``forecast``: the signal is on/off, so estimating a
scalar from mean(|forecast|) would just rescale the on-state arbitrarily. In the
window it emits the standard 'full risk budget' forecast (+10); outside, zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..config import FORECAST_SCALAR_TARGET


@register
class TurnOfMonth(Strategy):
    key = "tom"
    label = "Turn-of-Month"
    params = {
        "pre": (3, 0, 10, 1),    # trading days before month end
        "post": (3, 0, 10, 1),   # trading days after month start
    }
    spectrum_param = "pre"

    plain = (
        "Owns the market only around the turn of each month — the last few trading days "
        "and the first few of the next — and holds nothing the rest of the time."
    )
    analogy = (
        "Like showing up to a restaurant only during the lunch rush. You are not "
        "predicting anything about the food; you just know that is when the crowd arrives."
    )
    how_it_works = (
        "Look at the calendar, not the price.",
        "Count the last few trading days of the month and the first few of the next.",
        "Hold the market during that window only.",
        "Sit in cash on every other day.",
    )
    works_when = (
        "The historical pattern keeps repeating — commonly explained by pension and "
        "payroll money arriving on a monthly schedule."
    )
    fails_when = (
        "The pattern is arbitraged away, or the market has a bad month-end. It is also "
        "invested only ~30% of the time, so it will badly trail buy-and-hold in a "
        "straight-up year even if the effect is real. Treat it as a diversifier, not a "
        "core holding — and be honest that calendar effects are the category most prone "
        "to being a coincidence someone found by looking hard enough."
    )
    evidence = (
        "Long documented across equity indices — see "
        "[Quantpedia's turn-of-the-month strategy](https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes). "
        "Robust across many countries and decades, but it is a statistical pattern "
        "with no risk-based explanation, so treat it more sceptically than trend or carry."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        idx = ohlcv.index
        if not isinstance(idx, pd.DatetimeIndex):
            return pd.Series(0.0, index=idx)
        pre = int(self.values["pre"])
        post = int(self.values["post"])

        # Rank each bar within its own calendar month, from the front and the back.
        month = pd.Series(idx.to_period("M"), index=idx)
        from_start = month.groupby(month).cumcount()                    # 0,1,2,...
        from_end = month.groupby(month).cumcount(ascending=False)       # ...,2,1,0

        in_window = (from_end < pre) | (from_start < post)
        return in_window.astype(float)

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        """On/off signal at the standard full-risk forecast level (see docstring)."""
        return self.raw_forecast(ohlcv) * FORECAST_SCALAR_TARGET

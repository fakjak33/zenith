"""Carry proxy for ETFs.

True carry needs a term structure / yield. For equity & commodity ETFs we
approximate the carry premium with a long-horizon risk-adjusted drift (the
asset's own tendency to earn a positive return per unit risk), smoothed.
This is a deliberately simple stand-in so the 'carry' style is represented;
swap in real yield/roll data later for futures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.volatility import ew_vol, bars_per_year


@register
class CarryProxy(Strategy):
    key = "carry"
    label = "Carry (drift proxy)"
    params = {
        "lookback": (252, 60, 2000, 5),
    }
    spectrum_param = "lookback"

    plain = (
        "Favours assets that reward you just for holding them, measured here by their "
        "long-run tendency to drift upward relative to how risky they are."
    )
    analogy = (
        "Like choosing a rental property by its rent-to-price ratio rather than guessing "
        "whether house prices will rise. You get paid for holding it, regardless."
    )
    how_it_works = (
        "Measure the asset's annualised return over a long window - about a year by default.",
        "Divide by how volatile it is, giving return per unit of risk.",
        "Smooth the result so it changes slowly.",
        "A high value means holding is well rewarded - buy it.",
    )
    works_when = "Assets with a genuine structural yield: bonds, dividend payers, some commodities."
    fails_when = (
        "Carry trades famously 'go up by the stairs and down by the elevator' - they earn "
        "steadily and then lose a lot at once in a crisis."
    )
    evidence = (
        "IMPORTANT CAVEAT: this is a *stand-in*, not real carry. True carry needs yield or "
        "futures roll data, which this app does not have on the free data feed. It uses "
        "long-run risk-adjusted drift as a proxy, which behaves like a very slow trend "
        "rule. Real carry is well supported academically; treat this particular "
        "implementation as a placeholder until proper yield data is wired in."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        ann_ret = (close / close.shift(lb)) ** (bars_per_year(close.index) / lb) - 1.0
        vol_annual = ew_vol(close, annualize=True)
        carry = ann_ret / vol_annual.replace(0.0, np.nan)
        return carry.rolling(20).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)

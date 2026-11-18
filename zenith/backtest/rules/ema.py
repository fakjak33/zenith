"""EMA trend rule.

Carver's EWMAC: difference between a fast and slow EWMA, normalized by price
volatility so the forecast scale is stationary. Positive => uptrend.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol


@register
class EMACrossover(Strategy):
    key = "ema"
    label = "EMA Trend (EWMAC)"
    params = {
        "fast": (16, 2, 1000, 1),
        "slow": (64, 4, 2000, 1),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "fast"

    plain = (
        "Follows the trend. It compares a fast-reacting average price to a slow-moving "
        "one: when the fast one is above the slow one the market is rising, so it buys."
    )
    analogy = (
        "Like judging whether a hill goes up by comparing where you are standing to "
        "where you were a mile back. Recent position versus longer-term position tells "
        "you the slope."
    )
    how_it_works = (
        "Keep two running averages of the price: one quick, one slow.",
        "Subtract the slow one from the quick one - positive means rising.",
        "Divide by how jumpy the price normally is, so calm and wild markets are comparable.",
        "The bigger the gap, the bigger the position - it scales smoothly, it is not just on/off.",
    )
    works_when = "Markets move in sustained directions for weeks or months at a time."
    fails_when = (
        "Sideways, choppy markets. The two averages keep crossing back and forth and you "
        "pay costs on every flip-flop without catching a real move."
    )
    evidence = (
        "Robert Carver's EWMAC, the core building block of his systematic trading books, "
        "and the industry-standard implementation of trend-following used by managed "
        "futures funds."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = int(self.values["fast"])
        slow = int(self.values["slow"])
        ewmac = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
        # normalize by price * daily vol => unit-free trend strength
        vol = ew_vol(close, annualize=False) * close
        fc = normalize_by_vol(ewmac, vol)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"EMA fast ({int(self.values['fast'])})": close.ewm(span=int(self.values["fast"])).mean(),
            f"EMA slow ({int(self.values['slow'])})": close.ewm(span=int(self.values["slow"])).mean(),
        }

"""Short-term reversal (Jegadeesh 1990; Lehmann 1990).

Forecast = minus the last few days' return, divided by volatility. Whatever
just fell hardest is bought; whatever just spiked is sold.

Distinct from ``meanrev``, which measures distance from a 20-day moving average
in standard deviations. This one looks only at the very recent *return* over a
1-5 day window, which is the horizon the reversal literature actually documents.

Kept deliberately simple so the app can demonstrate the punchline of that
literature: the raw effect is real, and trading costs eat it. Turn the cost and
slippage sliders up and watch the equity curve invert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.volatility import ew_vol


@register
class ShortTermReversal(Strategy):
    key = "strev"
    label = "Short-Term Reversal"
    params = {
        "lookback": (5, 1, 60, 1),
        "smooth": (1, 1, 20, 1),
    }
    spectrum_param = "lookback"

    plain = (
        "Buys whatever just dropped sharply over the last few days and sells whatever "
        "just jumped, betting that very short, sharp moves tend to partly undo themselves."
    )
    analogy = (
        "Like a crowded shop with one till. When everyone rushes to buy at once the "
        "queue overshoots, and a moment later it settles back down. This rule bets on "
        "the settling-back, not on where the shop is going."
    )
    how_it_works = (
        "Measure how much the price moved over the last few days.",
        "Divide by how jumpy the price normally is, so a big move in a calm market counts for more.",
        "Flip the sign: a large drop becomes a buy, a large jump becomes a sell.",
        "Repeat every bar — positions turn over very quickly.",
    )
    works_when = (
        "Short bursts of panic or excitement push a price further than the news justifies, "
        "and it drifts back within days."
    )
    fails_when = (
        "Costs are realistic. This is the honest lesson of this rule: the effect shows up "
        "clearly in raw prices but the research finds ordinary trading costs wipe out the "
        "simple version, because it trades constantly. Raise the cost and slippage sliders "
        "in the sidebar and watch it happen."
    )
    evidence = (
        "Jegadeesh (1990) and Lehmann (1990) documented the effect. Later work "
        "([Quantpedia summary](https://quantpedia.com/strategies/combining-fundamental-fscore-and-equity-short-term-reversals)) "
        "finds the plain version is not profitable after transaction costs, though "
        "variants anchored on fundamentals survive."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        past_ret = close / close.shift(lb) - 1.0
        # scale by the volatility of a move of that horizon, so the signal is
        # comparable across calm and wild periods
        vol = ew_vol(close, annualize=False) * np.sqrt(lb)
        fc = (-past_ret / vol.replace(0.0, np.nan))
        fc = fc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

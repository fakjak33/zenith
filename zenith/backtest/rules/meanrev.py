"""Mean reversion — the academically supported counterpoint to trend.

Forecast = negative z-score of price vs its moving average (buy dips, fade
rallies). Included so the lab can demonstrate regime dependence: this should
WIN on mean-reverting synthetic data and LOSE on trending data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class MeanReversion(Strategy):
    key = "meanrev"
    label = "Mean Reversion (z-score)"
    params = {
        "lookback": (20, 3, 1000, 1),
    }
    spectrum_param = "lookback"

    plain = (
        "Buys when the price drops unusually far below its recent average, and sells "
        "when it shoots unusually far above — betting it snaps back toward normal."
    )
    analogy = (
        "Like a rubber band. Stretch it too far in either direction and it pulls back "
        "to the middle. This rule bets on the snap-back, not on where the band is going."
    )
    how_it_works = (
        "Work out the average price over the last few weeks.",
        "Measure how far today's price sits from that average, in 'unusual-ness' units.",
        "The further below average, the more it buys. The further above, the more it sells.",
        "The bet closes itself as the price returns to average.",
    )
    works_when = (
        "The market is choppy and range-bound, with no strong direction — prices "
        "wander around a level instead of marching away from it."
    )
    fails_when = (
        "A genuine trend starts. Then 'unusually cheap' keeps getting cheaper and you "
        "keep buying all the way down. This is the exact opposite failure mode to the "
        "trend rules, which is why holding both can smooth the ride."
    )
    evidence = (
        "The academic counterpoint to trend-following. Included here mainly as a "
        "controlled experiment: run it on 'mean_reverting' synthetic data and it should "
        "win; run it on 'trending' synthetic data and it should lose. If it does not "
        "behave that way, something is wrong with the backtest, not the market."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        ma = close.rolling(lb).mean()
        sd = close.rolling(lb).std().replace(0.0, np.nan)
        z = (close - ma) / sd
        # negative => fade the move
        return (-z).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        ma = close.rolling(lb).mean()
        sd = close.rolling(lb).std()
        return {
            f"MA ({lb})": ma,
            "+2σ band": ma + 2 * sd,
            "−2σ band": ma - 2 * sd,
        }

"""Low-volatility / betting-against-beta, cross-sectional.

Ranks the universe by trailing realized volatility and goes long the calmest
names, short the wildest. Cross-sectional like ``xsmom``, so it overrides
``forecast_panel`` and needs more than one instrument selected.

The anomaly: riskier stocks have historically *not* paid for their extra risk,
so the low-volatility leg wins on a risk-adjusted basis. Frazzini & Pedersen's
betting-against-beta is the leveraged-constraint explanation of the same effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import scale_forecast
from ..core.volatility import ew_vol


@register
class LowVolatility(Strategy):
    key = "lowvol"
    label = "Low Volatility (BAB)"
    cross_sectional = True
    params = {
        "lookback": (126, 20, 1000, 5),   # window for realized vol
        "skip": (0, 0, 120, 1),           # lag the vol measure (anti-reversal check)
    }
    spectrum_param = "lookback"

    plain = (
        "Buys the steadiest, least jumpy assets in your list and sells the wildest ones, "
        "because calm assets have historically paid better than their risk suggests."
    )
    analogy = (
        "Like picking the reliable used car over the temperamental sports car. The sports "
        "car promises more excitement, but over years of ownership the boring one costs "
        "you less and gets you there just as often."
    )
    how_it_works = (
        "Measure how much each asset in your list has bounced around recently.",
        "Rank them from calmest to wildest.",
        "Go long the calm end of the list, short the wild end.",
        "Re-rank as volatilities change.",
    )
    works_when = (
        "Investors who cannot use leverage bid up risky assets hoping for big returns, "
        "leaving calm assets underpriced. Needs several instruments selected to rank across."
    )
    fails_when = (
        "A sharp recovery rally after a crash — the beaten-up, high-volatility names "
        "bounce hardest and the short leg hurts. Also needs at least 3-4 tickers to mean "
        "anything; with one instrument there is nothing to rank."
    )
    evidence = (
        "Frazzini & Pedersen's betting-against-beta, and the broader low-volatility "
        "anomaly. [Quantpedia's deconstruction](https://quantpedia.com/deconstructing-the-low-volatility-anomaly/) "
        "shows it is not just short-term reversal in disguise — it survives lagging the "
        "volatility measure by a month or more, and survives sector neutrality."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError("lowvol is cross-sectional; use forecast_panel.")

    def forecast_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        lb = int(self.values["lookback"])
        skip = int(self.values["skip"])
        closes = pd.DataFrame({t: df["close"] for t, df in panel.items()}).sort_index()

        rets = closes.pct_change()
        vol = rets.rolling(lb).std()
        if skip:
            vol = vol.shift(skip)

        # cross-sectional z-score, negated so LOW vol => positive forecast
        mean = vol.mean(axis=1)
        std = vol.std(axis=1).replace(0.0, np.nan)
        z = -vol.sub(mean, axis=0).div(std, axis=0)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({c: scale_forecast(z[c]) for c in z.columns})

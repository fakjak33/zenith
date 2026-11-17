"""SMA trend rule — simple-moving-average crossover, vol-normalized."""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol


@register
class SMACrossover(Strategy):
    key = "sma"
    label = "SMA Trend"
    params = {
        "fast": (20, 2, 1000, 1),
        "slow": (100, 5, 2000, 1),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "fast"

    plain = (
        "The same trend-following idea as the EMA rule, but using plain averages that "
        "treat every day in the window equally instead of favouring recent days."
    )
    analogy = (
        "Two ways to judge a student: their average grade over the whole term (this rule), "
        "or an average that weights last week's tests more heavily (the EMA rule)."
    )
    how_it_works = (
        "Average the price over a short window and over a long window.",
        "Subtract the long average from the short one.",
        "Divide by how jumpy the price normally is, so the signal means the same in any market.",
        "Positive means uptrend and it buys; negative means downtrend and it sells.",
    )
    works_when = "The same conditions as any trend rule - long, sustained moves."
    fails_when = (
        "Choppy markets. It also reacts more slowly than the EMA version, so it enters "
        "and exits later - sometimes a good thing (fewer false signals), sometimes costly."
    )
    evidence = (
        "The oldest documented technical rule; tested exhaustively in the academic "
        "literature. Worth running alongside the EMA rule to see how little the choice "
        "between them actually matters - a useful lesson in not over-tuning."
    )

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = int(self.values["fast"])
        slow = int(self.values["slow"])
        diff = close.rolling(fast).mean() - close.rolling(slow).mean()
        vol = ew_vol(close, annualize=False) * close
        fc = normalize_by_vol(diff, vol)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"SMA fast ({int(self.values['fast'])})": close.rolling(int(self.values["fast"])).mean(),
            f"SMA slow ({int(self.values['slow'])})": close.rolling(int(self.values["slow"])).mean(),
        }

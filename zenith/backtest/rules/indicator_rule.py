"""Every catalogue indicator as a backtestable rule.

``IndicatorRule`` turns an indicator into a raw forecast according to its
signal style, then the base class scales it into a Carver forecast. One
subclass per indicator is registered as ``i_<key>`` so the whole LAB pipeline
(sweeps, walk-forward, permutation, OOS) works on all of them unchanged.

Transforms (``side`` = +1 follows the reading, −1 fades it):
  cross   (close − line) ÷ ATR(14)                     → trend-following
  level   the reading itself (already signed)
  band    stateful: long below ``lo`` until it crosses back over the midpoint,
          short above ``hi`` likewise. side +1 = mean reversion (buy oversold),
          side −1 = breakout (buy overbought).
  zscore  z-score of the reading over ``z_len`` bars
  state   the reading vs its own expanding median, in expanding-σ units
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import CATALOG, Indicator
from ..indicators.catalog import atr, zscore
from .base import Strategy
from .registry import register

SIDE_HELP = ("+1 follows the indicator, −1 does the opposite. For oversold/overbought indicators +1 means "
             "'buy when oversold' (mean reversion) and −1 means 'buy when overbought' (breakout).")


def band_state(v: pd.Series, lo: float, hi: float) -> pd.Series:
    """+1 from a close below ``lo`` until the reading crosses the midpoint; −1 mirror for ``hi``."""
    x = v.to_numpy(dtype=float)
    mid = (lo + hi) / 2
    out = np.zeros(len(x))
    pos = 0.0
    for i, val in enumerate(x):
        if not np.isfinite(val):
            pos = 0.0
        elif pos == 0:
            pos = 1.0 if val < lo else -1.0 if val > hi else 0.0
        elif pos > 0 and val >= mid:
            pos = -1.0 if val > hi else 0.0
        elif pos < 0 and val <= mid:
            pos = 1.0 if val < lo else 0.0
        out[i] = pos
    return pd.Series(out, index=v.index)


class IndicatorRule(Strategy):
    indicator: Indicator | None = None
    needs: tuple = ()

    def reading(self, ohlcv: pd.DataFrame) -> pd.Series:
        ip = {k: self.values[k] for k in self.indicator.params}
        return self.indicator(ohlcv, **ip)

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        ind = self.indicator
        v = self.reading(ohlcv)
        side = float(np.sign(self.values.get("side", 1)) or 1)
        style = ind.signal
        if style == "cross":
            if ind.overlay:
                raw = (ohlcv["close"] - v) / atr(ohlcv, 14)
            else:
                raw = v
        elif style == "band":
            raw = band_state(v, float(self.values["lo"]), float(self.values["hi"]))
        elif style == "zscore":
            raw = zscore(v, int(self.values.get("z_len", 252)))
        elif style == "state":
            med = v.expanding(min_periods=60).median()
            sd = v.expanding(min_periods=60).std()
            raw = (v - med) / sd.where(sd > 0)
        else:
            raw = v
        return (side * raw).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def indicator_lines(self, ohlcv):
        if self.indicator.overlay:
            return {self.describe(): self.reading(ohlcv)}
        return {}

    @classmethod
    def explain(cls) -> dict:
        d = super().explain()
        i = cls.indicator
        d.update({"how_calc": i.how, "use": i.use, "video": i.video, "family": i.family, "signal": i.signal})
        return d


_STYLE_TEXT = {
    "cross": "Signal: long while price is above the line, short below, sized by the distance in ATRs.",
    "level": "Signal: the reading itself — positive = long, negative = short.",
    "band": "Signal: buy when it drops below the low level, sell above the high level; exit at the midpoint.",
    "zscore": "Signal: how unusual the reading is versus its own past year (z-score).",
    "state": "Signal: above its own long-run median = long (flip 'side' to fade it). Mostly useful as a "
             "filter or regime variable.",
}


def _make(ind: Indicator) -> type[IndicatorRule]:
    params = {k: v[:4] for k, v in ind.params.items()}
    helps = {k: v[4] for k, v in ind.params.items()}
    if ind.signal == "band":
        lo, hi = float(ind.lo), float(ind.hi)
        span = max(abs(hi - lo), 1e-6)
        step = round(span / 20, 4) or 0.01
        params["lo"] = (lo, lo - 2 * span, hi, step)
        params["hi"] = (hi, lo, hi + 2 * span, step)
        helps["lo"] = "Oversold level: below it the rule buys (side +1)."
        helps["hi"] = "Overbought level: above it the rule sells (side +1)."
    if ind.signal == "zscore":
        params["z_len"] = (252, 20, 1000, 1)
        helps["z_len"] = "Window used to judge how unusual the reading is."
    params["side"] = (1, -1, 1, 2)
    helps["side"] = SIDE_HELP
    attrs = dict(key=f"i_{ind.key}", label=ind.label, params=params, param_help=helps,
                 spectrum_param=next(iter(ind.params), "side"), indicator=ind, needs=ind.needs,
                 plain=ind.hover, analogy=_STYLE_TEXT[ind.signal], how_it_works=(ind.how, ind.use),
                 works_when=ind.use, fails_when="", evidence=ind.evidence)
    return register(type(f"Ind_{ind.key}", (IndicatorRule,), attrs))


for _ind in CATALOG.values():
    _make(_ind)

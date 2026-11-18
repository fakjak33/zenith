"""Turtle breakout (Richard Dennis / Dennis-Eckhardt).

The original Turtle rules, as a Carver-compatible forecast. Unlike the smooth
``breakout`` rule (which reports where price sits *inside* its range), this one
fires on discrete events:

  - ENTER when the close breaks the highest high / lowest low of the last
    ``entry_lb`` bars ("System 1" = 20 bars, "System 2" = 55).
  - ADD a unit each time price moves a further ``add_atr`` x N in your favour,
    up to ``max_units`` — the pyramiding that lets a big trend pay for many
    small losses.
  - EXIT the whole position on the opposite ``exit_lb``-bar channel, or if
    price retraces ``stop_atr`` x N from the last entry.

N is Wilder's ATR, exactly as in the original rules.

Forecast mapping: because the position is a discrete ladder of units, this rule
overrides ``forecast`` rather than going through ``scale_forecast``. A full
``max_units`` stack maps to the forecast cap (+/-20) and one unit to a
proportional fraction, so the ladder survives into the sizing layer instead of
being renormalized away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..config import FORECAST_CAP
from ..risk.atr import atr


@register
class TurtleBreakout(Strategy):
    key = "turtle"
    label = "Turtle Breakout (Dennis)"
    params = {
        "entry_lb": (20, 5, 400, 1),     # breakout channel (20 = S1, 55 = S2)
        "exit_lb": (10, 3, 200, 1),      # opposite channel that closes the trade
        "atr_period": (20, 5, 100, 1),   # N
        "add_atr": (0.5, 0.1, 3.0, 0.1),  # add a unit every this-many N in favour
        "stop_atr": (2.0, 0.5, 10.0, 0.5),  # stop, in N, from the last entry
        "max_units": (4, 1, 8, 1),       # pyramid cap
    }
    spectrum_param = "entry_lb"

    plain = (
        "Buys when the price climbs above its highest point of the last few weeks, "
        "and keeps buying more as it keeps climbing. It sells short the same way "
        "when the price falls below its lowest point."
    )
    analogy = (
        "Like betting on a runner who just broke the course record. You do not try "
        "to guess who will win — you wait for someone to actually pull ahead, back "
        "them, and add to the bet each time they stretch their lead further."
    )
    how_it_works = (
        "Track the highest high and lowest low of the last N bars — that is the channel.",
        "When the price closes above the top of the channel, go long. Below the bottom, go short.",
        "Each time the price moves another half-ATR your way, add another unit (up to 4).",
        "Close everything when the price crosses back through a shorter, opposite channel.",
    )
    works_when = (
        "A market makes a long, sustained move in one direction — the 2020 rally, "
        "a commodity spike, a currency devaluation."
    )
    fails_when = (
        "The market chops sideways. You get repeatedly stopped out by false breakouts. "
        "Expect to lose on most trades — roughly 60-70% — and make it all back on a few huge winners."
    )
    evidence = (
        "The rules Richard Dennis taught his 'Turtles' in 1983. Independently tested since: "
        "[Donchian channels on SAFEX futures](https://open.uct.ac.za/handle/11427/21754) found "
        "materially better risk-adjusted returns, and "
        "[a study of the rules on Chinese futures](https://www.researchgate.net/publication/370698720_Research_on_Quantitative_Trading_Strategies_Based_on_the_Turtle_Trading_Rule) "
        "found excess returns after optimisation. It is a discrete cousin of the "
        "well-replicated time-series momentum effect."
    )

    # --- unit ladder -------------------------------------------------------
    def _units(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Signed unit count per bar (-max_units .. +max_units).

        Walked bar by bar because the position is path-dependent: whether you
        add today depends on where you entered, which depends on an earlier
        breakout. There is no vectorised shortcut that preserves that.
        """
        close = ohlcv["close"]
        entry_lb = int(self.values["entry_lb"])
        exit_lb = int(self.values["exit_lb"])
        add_n = float(self.values["add_atr"])
        stop_n = float(self.values["stop_atr"])
        max_u = int(self.values["max_units"])

        # Channels use only *prior* bars, so a breakout is never compared
        # against the bar that caused it (that would be look-ahead).
        hi = ohlcv["high"].rolling(entry_lb).max().shift(1)
        lo = ohlcv["low"].rolling(entry_lb).min().shift(1)
        exit_hi = ohlcv["high"].rolling(exit_lb).max().shift(1)
        exit_lo = ohlcv["low"].rolling(exit_lb).min().shift(1)
        n = atr(ohlcv, period=int(self.values["atr_period"]))

        c = close.to_numpy(dtype=float)
        hi_a, lo_a = hi.to_numpy(dtype=float), lo.to_numpy(dtype=float)
        xhi_a, xlo_a = exit_hi.to_numpy(dtype=float), exit_lo.to_numpy(dtype=float)
        n_a = n.to_numpy(dtype=float)

        units = np.zeros(len(c), dtype=float)
        pos = 0          # signed unit count
        last_entry = np.nan   # price of the most recent unit added
        unit_n = np.nan       # N at the time of the first entry (Turtles froze it)

        for i in range(len(c)):
            price, big_n = c[i], n_a[i]
            if not np.isfinite(price) or not np.isfinite(big_n) or big_n <= 0:
                units[i] = pos
                continue

            if pos == 0:
                if np.isfinite(hi_a[i]) and price > hi_a[i]:
                    pos, last_entry, unit_n = 1, price, big_n
                elif np.isfinite(lo_a[i]) and price < lo_a[i]:
                    pos, last_entry, unit_n = -1, price, big_n
            elif pos > 0:
                # exit first: the opposite channel, or the stop from last entry
                if (np.isfinite(xlo_a[i]) and price < xlo_a[i]) or \
                        price <= last_entry - stop_n * unit_n:
                    pos, last_entry, unit_n = 0, np.nan, np.nan
                elif pos < max_u and price >= last_entry + add_n * unit_n:
                    pos += 1
                    last_entry = price
            else:
                if (np.isfinite(xhi_a[i]) and price > xhi_a[i]) or \
                        price >= last_entry + stop_n * unit_n:
                    pos, last_entry, unit_n = 0, np.nan, np.nan
                elif pos > -max_u and price <= last_entry - add_n * unit_n:
                    pos -= 1
                    last_entry = price
            units[i] = pos

        return pd.Series(units, index=ohlcv.index)

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Unit ladder normalized to [-1, 1]."""
        return self._units(ohlcv) / float(self.values["max_units"])

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        """Fixed mapping to the forecast scale — see the module docstring.

        Deliberately bypasses ``scale_forecast``: the rule is flat most of the
        time, so estimating a scalar from mean(|forecast|) would inflate a
        single unit up to the cap and erase the pyramid.
        """
        return (self.raw_forecast(ohlcv) * FORECAST_CAP).clip(-FORECAST_CAP, FORECAST_CAP)

    def indicator_lines(self, ohlcv):
        entry_lb = int(self.values["entry_lb"])
        exit_lb = int(self.values["exit_lb"])
        return {
            f"Entry high ({entry_lb})": ohlcv["high"].rolling(entry_lb).max().shift(1),
            f"Entry low ({entry_lb})": ohlcv["low"].rolling(entry_lb).min().shift(1),
            f"Exit high ({exit_lb})": ohlcv["high"].rolling(exit_lb).max().shift(1),
            f"Exit low ({exit_lb})": ohlcv["low"].rolling(exit_lb).min().shift(1),
        }

"""NIGHT & DAY — overnight vs intraday return decomposition for Zenith.

Lou, Polk & Skouras (JFE 2019, "A Tug of War: Overnight versus Intraday
Expected Returns"): almost all of the U.S. market's cumulative gain accrues
OVERNIGHT (close-to-open) while intraday (open-to-close) is roughly flat; and
firm-level overnight and intraday returns each show strong continuation that
persists for years, with an offsetting cross-period reversal.

This tab decomposes each stock/ETF into its two streams:
    overnight_t = open_t / close_{t-1} - 1
    intraday_t  = close_t / open_t - 1
and screens for names with the most persistent overnight or intraday tilt, plus
the "tug of war" spread. Evidence tier B: the statistics are strong, but literal
overnight-only capture pays the full spread every day, so this is a research
monitor / tilt input, not a day-trading system.

Free / best-effort data only (yfinance OHLC), computed offline in a GitHub
Action that commits JSON under data/nightday/. Decision-support, not advice.
"""

from __future__ import annotations

import json

from ..config import NIGHTDAY_FILES

DISCLAIMER = ("NIGHT & DAY decomposes yfinance OHLC into overnight (close->open) and intraday "
              "(open->close) streams (Lou-Polk-Skouras 2019). The overnight tilt is real and "
              "persistent in the data but NOT cheaply tradable — capturing it literally pays the "
              "open/close spread daily. A research monitor and tilt input, not a trading system.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(NIGHTDAY_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    NIGHTDAY_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    NIGHTDAY_FILES[name].write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

"""Trend SYSTEM registry — the extension point for future methodologies.

A trend system maps one instrument's daily close series to an
`ewmac.EwmacResult`-shaped object: per-speed forecasts on the common -20..+20
scale, the raw signal whose sign is the direction, an equal-weight score and a
valid-speed count. Everything downstream (structure, events, persistence, the
whole UI) consumes only that shape, so a Donchian-breakout ladder, an
ATR-normalized trend, or a volatility-scaled variant is one entry here plus a
config key -- not a rewrite.

Only the seven-speed EWMAC exists today, and it is the only system any score
in the app uses.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import ewmac

SYSTEM_REGISTRY: dict[str, Callable[[pd.Series], ewmac.EwmacResult]] = {
    "ewmac": ewmac.build,
}

ACTIVE_SYSTEM = "ewmac"


def build(close: pd.Series, system: str = ACTIVE_SYSTEM) -> ewmac.EwmacResult:
    return SYSTEM_REGISTRY[system](close)

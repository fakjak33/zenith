"""TREND FOLLOWING — a seven-speed EWMAC trend-following system for Zenith.

Every trading day, every Russell 1000 stock (MOMENTUM's universe) and every ETF
in ETF MOMENTUM's universe is scored on seven Exponentially-Weighted Moving
Average Crossovers -- 2/8, 4/16, 8/32, 16/64, 32/128, 64/256, 128/512 days --
each a vol-normalized forecast on a -20..+20 scale, and the seven are averaged
with EQUAL weight into one Trend Score in [-20, +20]. See ewmac.py for the
exact math and config.py's TREND block for every constant.

The point of seven speeds is the TERM STRUCTURE of trend: whether an asset is
trending across every horizon or only at one. structure.py turns the seven
forecasts into breadth, fast/slow group reads and a plain-English structure
label; events.py detects crossovers, trigger/band changes and multi-speed
confirmations; history.py persists all of it daily so any past score or
crossover date can be answered.

Conceptually this is the systematic trend-following of Robert Carver
("Systematic Trading", 2015; pysystemtrade) and the CTA literature (Moskowitz,
Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017, "A Century of Evidence on
Trend-Following Investing"; Baltas & Kosowski 2013). It is not a replica of any
proprietary strategy. A positive score is a TREND SIGNAL -- the asset has been
rising at those horizons -- not a buy recommendation and not a return forecast.

Two differences from MOMENTUM worth knowing: (1) this is purely time-series --
no cross-sectional ranking enters any score, so a stock's score does not depend
on what the rest of the universe did; (2) because of that, and because it uses
only the asset's own price, a historical backfill is honest: there is no
survivorship bias in any single asset's signal history (universe-level breadth
history still reuses today's constituents before config.MOM_MEMBERSHIP_START).
"""

from __future__ import annotations

import json

from ..config import TREND_FILES, TREND_SPEEDS

DISCLAIMER = ("TREND FOLLOWING scores every stock in the MOMENTUM universe and every ETF in the "
              "ETF MOMENTUM universe on seven EWMAC speeds (2/8 through 128/512 days), each a "
              "volatility-normalized trend forecast from -20 to +20, averaged with equal weight into "
              "one Trend Score. A positive score means the asset HAS BEEN trending up at those "
              "horizons — it is a trend signal, not a buy recommendation or a forecast of returns. "
              "Decision-support and a research monitor, not investment advice.")

UNIVERSE_LABELS = {"stocks": "Stocks", "etfs": "ETFs"}

SPEED_KEYS = tuple(f"{f}_{s}" for f, s in TREND_SPEEDS)
SPEED_SHORT = {k: k.replace("_", "/") for k in SPEED_KEYS}          # "2/8"
SPEED_NAMES = dict(zip(SPEED_KEYS, ("Very Fast", "Fast", "Medium-Fast", "Medium",
                                    "Medium-Slow", "Slow", "Very Slow")))
# Horizon bucket per speed -- used to tag events short/medium/long term.
SPEED_HORIZON = dict(zip(SPEED_KEYS, ("short", "short", "short", "medium",
                                      "long", "long", "long")))


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(universe: str, name: str, default=None):
    return _read(TREND_FILES[universe][name], default if default is not None else {})


def save(universe: str, name: str, obj, indent: int | None = 2) -> None:
    """Write a TREND artefact. The large daily ones (latest, recent events)
    pass indent=None -- same reasoning as mom.save: thousands of rows a day
    pretty-printed is hundreds of MB of git history nobody diffs by eye."""
    p = TREND_FILES[universe][name]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

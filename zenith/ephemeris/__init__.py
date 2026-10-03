"""EPHEMERIS — a blind-chart tape-reading trainer.

You see a randomly drawn historical chart (ticker, dates and price level
hidden), call the direction over the next N candles, and the future is then
revealed and scored. Every call is logged to a per-player store, and the stats
engine asks the only question that matters for deliberate practice: are you
beating the BASE RATE and a simple TREND RULE on the very same charts? Beating
50% is not skill when the asset drifts up 55% of the time.

Design notes:
  * Prices never come from a live fetch per guess. A nightly job (prefetch.py)
    builds adjusted OHLCV Parquet and publishes it as a GitHub Release asset;
    store_px.py downloads it once per container.
  * Low-friction profiles: type a handle, done. `?player=handle` in the URL
    means a bookmark is a login. Optional 4-digit PIN for shared deployments.
  * Storage sits behind repo.Repository: Supabase Postgres when the
    `ephemeris_db_url` secret exists, local SQLite otherwise.
  * Lineage: ports the good parts of VELA (../vela) -- rebasing, the bad-data
    filters, class quotas -- and drops its sign-up / recovery-code flow.
"""

from __future__ import annotations

DISCLAIMER = ("EPHEMERIS is a practice game on historical, split- and dividend-adjusted prices. "
              "Paper stakes only. Results measure pattern-reading on past data, not future returns — "
              "not investment advice.")

SURVIVORSHIP_NOTE = ("Survivorship bias: the universe is today's Russell 1000 and currently-listed ETFs. "
                     "Delisted and acquired names are missing, which flatters long calls over long histories.")

# --- paper account -----------------------------------------------------------
START_BALANCE = 10_000.0
BASE_STAKE = 1_000.0
CONVICTION = {"Low": 0.25, "Medium": 0.50, "High": 1.00}
DEFAULT_CONVICTION = "Medium"

# --- game parameters ---------------------------------------------------------
TIMEFRAMES = ("1H", "4H", "Daily", "Weekly", "Monthly")
PHASE1_TIMEFRAMES = ("Daily",)
HORIZONS = (1, 3, 5, 10, 20, 30, 50)
LOOKBACKS = (60, 120, 250, 500)
DEFAULT_HORIZON = 10
DEFAULT_LOOKBACK = 120
WARMUP = 250            # bars kept before the window so SMA-200 is real at its left edge
MODES = ("practice", "daily")

# --- universe classes (display order) ----------------------------------------
CLASSES = ("Russell 1000", "US Equity ETFs", "International Equity", "Bonds",
           "Precious Metals", "Commodities", "Real Estate", "Currencies", "Crypto",
           "Alternatives", "Indices")

# Largest single-bar |move| tolerated in a window before it is called bad data
# (unadjusted split, bad print). Per class, from VELA's build-pool.mjs.
MAX_BAR_MOVE = {"Currencies": 0.15, "Crypto": 0.60, "Russell 1000": 0.40, "Bonds": 0.15,
                "Indices": 0.25}
MAX_BAR_MOVE_DEFAULT = 0.30
NO_VOLUME_CLASSES = frozenset({"Currencies"})     # spot FX has no volume field
MIN_DOLLAR_VOLUME = 1e6                            # window median; adjusted $ — conservative


def stake_for(conviction: str, base: float = BASE_STAKE) -> float:
    return base * CONVICTION.get(conviction, CONVICTION[DEFAULT_CONVICTION])

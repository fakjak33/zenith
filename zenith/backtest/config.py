"""BACKTEST defaults: the Carver pipeline knobs, paths and compute caps.

Ported from PARALLAX (``ghost/config.py``). The retro theme is gone — BACKTEST
pages use ``zenith.config.THEME`` like every other page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import DATA_DIR

# --- Paths -----------------------------------------------------------------
BACKTEST_DIR = DATA_DIR / "backtest"
DATA_CACHE = BACKTEST_DIR / "cache"     # per-ticker parquet (gitignored)
SPECS_DIR = BACKTEST_DIR / "specs"      # saved StrategySpec JSON (committed)
RUNS_DIR = BACKTEST_DIR / "runs"        # cached run results (gitignored)
CACHE_TTL_HOURS = 30                    # re-download a ticker after this

# --- Carver framework defaults --------------------------------------------
TRADING_DAYS = 252              # fallback bars/year for daily data (US calendar)
FORECAST_SCALAR_TARGET = 10.0   # target average absolute forecast
FORECAST_CAP = 20.0             # hard cap on |forecast|
VOL_LOOKBACK = 32               # span (bars) for EW volatility estimate
DEFAULT_TARGET_VOL = 0.20       # 20% annualized vol target
DEFAULT_CAPITAL = 1_000_000.0
SCALAR_WARMUP = 64              # bars before the expanding forecast scalar is trusted

# Trading days represented by one bar at each timeframe — used to scale
# day-based strategy parameters when the candle frequency changes.
TIMEFRAME_DAYS = {"Daily": 1, "Weekly": 5, "Monthly": 21}
TIME_SCALED_PARAMS = {"fast", "slow", "lookback", "smooth", "skip",
                      "entry_lb", "exit_lb", "atr_period"}


# --- Compute caps -----------------------------------------------------------
def is_local() -> bool:
    """True when Zenith runs on the user's PC (caps lifted).

    Set ``ZENITH_LOCAL=1`` (the .claude/launch.json config does) or
    ``local = true`` in ``.streamlit/secrets.toml``.
    """
    if os.environ.get("ZENITH_LOCAL", "") == "1":
        return True
    try:
        import streamlit as st
        return bool(st.secrets.get("local", False))
    except Exception:  # noqa: BLE001 — no secrets file is the cloud default
        return False


@dataclass(frozen=True)
class Caps:
    tickers: int
    grid_side: int
    permutations: int
    mc_paths: int


CLOUD_CAPS = Caps(tickers=12, grid_side=20, permutations=200, mc_paths=1000)
LOCAL_CAPS = Caps(tickers=200, grid_side=60, permutations=5000, mc_paths=10000)


def caps() -> Caps:
    return LOCAL_CAPS if is_local() else CLOUD_CAPS


@dataclass
class BacktestConfig:
    """Backtest-level knobs (costs, capital, vol target)."""

    capital: float = DEFAULT_CAPITAL
    target_vol: float = DEFAULT_TARGET_VOL
    cost_bps: float = 1.0           # per-trade cost in basis points of notional
    slippage_bps: float = 0.5       # half-spread slippage in bps
    vol_lookback: int = VOL_LOOKBACK
    trading_days: int | None = None  # bars/year; None = infer from the data
    use_vol_target: bool = True     # back-compat flag (see sizing_mode)
    direction: str = "both"         # "both" | "long" | "short"
    sizing_mode: str = "vol_target"  # "vol_target" | "fixed_pct" | "fixed_dollar"
    fixed_pct: float = 1.0
    fixed_dollar: float = DEFAULT_CAPITAL
    use_delay: bool = False
    entry_delay: int = 0
    exit_delay: int = 0

    def __post_init__(self):
        if not self.use_vol_target and self.sizing_mode == "vol_target":
            self.sizing_mode = "fixed_pct"
        if self.sizing_mode != "vol_target":
            self.use_vol_target = False


@dataclass
class RiskConfig:
    """Stop-loss / take-profit overlay settings (ATR-based or fixed-percent)."""

    use_atr_stop: bool = False
    use_atr_tp: bool = False
    atr_period: int = 14
    atr_stop_mult: float = 3.0
    atr_tp_mult: float = 6.0
    use_pct_stop: bool = False
    use_pct_tp: bool = False
    pct_stop: float = 0.10
    pct_tp: float = 0.20
    trailing_stop: bool = True


def scale_params_for_tf(values: dict, tf_label: str) -> dict:
    """Convert day-based params into bars at the chosen candle frequency."""
    dpb = TIMEFRAME_DAYS.get(tf_label, 1)
    if dpb <= 1:
        return dict(values)
    out = {}
    for k, v in values.items():
        if k in TIME_SCALED_PARAMS and isinstance(v, (int, float)):
            out[k] = max(2, int(round(v / dpb))) if k != "skip" else max(0, int(round(v / dpb)))
        else:
            out[k] = v
    return out

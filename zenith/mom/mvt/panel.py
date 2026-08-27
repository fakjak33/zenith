"""Multivariate Trend price panel: OHLCV dict -> an aligned, cleaned daily
log-return matrix ready for the pairwise engine.

Pure transform, no I/O -- callers hand in whatever `cas.sources.prices.
get_history` already returned (equities reuse mom.compute's own R1000+SPY
pull; ETFs are a new, separately chunked pull in mvt/compute.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import MOM_MVT_MIN_BARS
from ..normalize import winsorize_xs


def build_return_panel(px: dict[str, pd.DataFrame], min_bars: int = MOM_MVT_MIN_BARS,
                       ffill_limit: int = 3, winsor_p: float = 0.005) -> tuple[pd.DataFrame, dict]:
    """Aligns every ticker's close series on the UNION of trading days seen
    across the universe, forward-fills isolated gaps up to `ffill_limit`
    days (different ETF listing venues / holidays), drops any ticker with
    fewer than `min_bars` non-NaN closes AFTER alignment (stated, not
    silently dropped -- see the `dropped` map in the returned status), and
    returns cross-sectionally winsorized daily log returns.

    Log returns (not simple) are used throughout mvt so that the horizon
    return is an exact sum of its disjoint increments (see horizons.py) --
    simple returns do not compose additively across sub-periods, which
    would break the whole no-double-counting design.
    """
    if not px:
        return pd.DataFrame(), {"n_in": 0, "n_kept": 0, "dropped": {}}

    closes = {}
    for t, df in px.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df["close"].dropna()
        if len(s) >= 60:
            closes[t] = s

    if not closes:
        return pd.DataFrame(), {"n_in": len(px), "n_kept": 0, "dropped": {}}

    panel = pd.DataFrame(closes)
    panel = panel.sort_index()
    panel = panel.ffill(limit=ffill_limit)

    dropped: dict[str, str] = {}
    kept_cols = []
    for t in panel.columns:
        n_valid = panel[t].notna().sum()
        if n_valid < min_bars:
            dropped[t] = f"insufficient_history(<{min_bars}d)"
        else:
            kept_cols.append(t)
    panel = panel[kept_cols]

    log_ret = np.log(panel / panel.shift(1))
    # Cross-sectional winsorization per day guards the covariance/PCA step
    # against one stale-price or bad-print day poisoning every pair that
    # touches it -- the same "don't let one bad instrument contaminate the
    # whole matrix" requirement the spec calls out explicitly (section 28).
    log_ret = log_ret.apply(lambda row: pd.Series(winsorize_xs(row.tolist(), winsor_p), index=row.index)
                            if row.notna().sum() >= 20 else row, axis=1)

    status = {"n_in": len(px), "n_kept": len(kept_cols), "dropped": dropped,
              "n_days": int(len(log_ret))}
    return log_ret, status


def advdollar(px: dict[str, pd.DataFrame], window: int = 21) -> dict[str, float]:
    """Trailing average daily dollar volume per ticker, for the liquidity
    gate (section 28) and as the near-duplicate-cluster liquidity tiebreak
    (universe.duplicate_and_leverage_gate's `adv` argument)."""
    out = {}
    for t, df in px.items():
        if df is None or df.empty or "close" not in df.columns or "volume" not in df.columns:
            continue
        dv = (df["close"] * df["volume"]).dropna().tail(window)
        if len(dv):
            out[t] = float(dv.mean())
    return out

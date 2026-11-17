"""Correlation of a strategy to standard underlyings.

Lets the R&D workflow see, at a glance, how correlated a strategy's returns are
to major asset-class / sector proxies (SPY, QQQ, IWM, TLT, IEI, GLD, USO) — so
you can flip synthetic seeds and judge both profitability and whether the
strategy is just tracking the market. Reuses providers + the timeframe resampling
the diagnostics tab already uses.
"""

from __future__ import annotations

import pandas as pd

from ..data import providers
from ..data.providers import TIMEFRAMES, resample_ohlcv
from .diagnostics import beta_and_correlation

STANDARD_UNDERLYINGS = ["SPY", "QQQ", "IWM", "TLT", "IEI", "GLD", "USO"]


def correlation_panel(strategy_returns: pd.Series, tf: str,
                      start: str | None = None, end: str | None = None,
                      extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-underlying correlation + beta vs ``strategy_returns``.

    Pulls each underlying, resamples to the selected timeframe (so it aligns with
    the strategy's bars), and computes Pearson r + beta on aligned returns.
    Degrades per-ticker — a failed pull is skipped, never raises.
    """
    sr = strategy_returns.dropna()
    rows = []
    tickers = list(dict.fromkeys(list(STANDARD_UNDERLYINGS) + list(extra)))
    for u in tickers:
        try:
            up = providers.get_prices(u, start=start or None, end=end or None)
            up = resample_ohlcv(up, TIMEFRAMES[tf])
            if start:
                up = up[up.index >= pd.Timestamp(start)]
            if end:
                up = up[up.index <= pd.Timestamp(end)]
            ur = up["close"].pct_change()
            aligned = pd.DataFrame({"strat": sr, "u": ur.reindex(sr.index)}).dropna()
            if len(aligned) < 5:
                continue
            bc = beta_and_correlation(aligned["strat"], up["close"])
            rows.append({"Underlying": u, "Correlation": bc["correlation"],
                         "Beta": bc["beta"], "Obs": len(aligned)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Underlying")
    return df

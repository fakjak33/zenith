"""Monte Carlo / bootstrap on the return stream.

Resamples daily returns (stationary block bootstrap) to produce a
*distribution* of Sharpe and max drawdown — so a single lucky path doesn't
masquerade as skill. Returns percentiles for the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def _block_bootstrap(returns: np.ndarray, rng: np.random.Generator,
                     block: int) -> np.ndarray:
    n = len(returns)
    out = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = min(block, n - i)
        for j in range(length):
            out[i + j] = returns[(start + j) % n]
        i += length
    return out


def bootstrap(
    returns: pd.Series,
    n_sims: int = 1000,
    block: int = 20,
    seed: int | None = 7,
    periods: int = 252,
) -> pd.DataFrame:
    """Return a DataFrame of bootstrapped [Sharpe, MaxDD, CAGR] samples."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 30:
        return pd.DataFrame(columns=["Sharpe", "MaxDD", "CAGR"])
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        sample = pd.Series(_block_bootstrap(r, rng, block))
        rows.append({
            "Sharpe": metrics.sharpe(sample, periods),
            "MaxDD": metrics.max_drawdown(sample),
            "CAGR": metrics.cagr(sample, periods),
        })
    return pd.DataFrame(rows)


def bootstrap_paths(
    returns: pd.Series,
    n_sims: int = 200,
    block: int = 20,
    capital: float = 1.0,
    seed: int | None = 7,
) -> pd.DataFrame:
    """Return a DataFrame of bootstrapped *equity paths* (columns = sims).

    Each column is a cumulative equity curve from a block-bootstrapped resample
    of the daily returns, starting at ``capital``. Used to draw the fan of
    possible outcomes behind the actual curve.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 30:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    paths = {}
    for s in range(n_sims):
        sample = _block_bootstrap(r, rng, block)
        paths[s] = capital * np.cumprod(1.0 + sample)
    return pd.DataFrame(paths)


def summarize(boot: pd.DataFrame) -> dict[str, dict[str, float]]:
    """5th/50th/95th percentiles per metric."""
    if boot.empty:
        return {}
    out = {}
    for col in boot.columns:
        out[col] = {
            "p5": float(boot[col].quantile(0.05)),
            "p50": float(boot[col].quantile(0.50)),
            "p95": float(boot[col].quantile(0.95)),
        }
    return out

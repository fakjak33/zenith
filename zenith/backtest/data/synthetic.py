"""Synthetic price generators — 'fake' data with known ground truth.

Used to validate strategies before trusting real-data results: a trend
follower MUST make money on ``trending`` and lose on ``mean_reverting``.
All generators return an OHLCV DataFrame matching the real-data schema so
they are drop-in interchangeable in the backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS

_KINDS = ("gbm", "trending", "mean_reverting", "regime", "fat_tailed")


def _to_ohlcv(close: np.ndarray, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Build a plausible OHLCV frame from a close-price path."""
    close = np.asarray(close, dtype=float)
    intraday = np.abs(rng.normal(0, 0.005, size=close.shape)) * close
    high = close + intraday
    low = close - intraday
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    volume = rng.integers(1_000_000, 5_000_000, size=close.shape)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def generate(
    kind: str = "gbm",
    n_days: int = 1500,
    start: str = "2015-01-01",
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate one synthetic OHLCV path.

    ``kind`` is one of: gbm, trending, mean_reverting, regime, fat_tailed.
    ``mu``/``sigma`` are annualized drift/vol. ``seed`` makes it reproducible.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown kind {kind!r}; choose from {_KINDS}.")

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    dt = 1.0 / TRADING_DAYS
    daily_mu = mu * dt
    daily_sig = sigma * np.sqrt(dt)

    if kind == "gbm":
        shocks = rng.normal(daily_mu, daily_sig, n_days)
        log_path = np.cumsum(shocks)

    elif kind == "trending":
        # persistent drift that flips sign occasionally — momentum-friendly
        regime_len = max(60, n_days // 8)
        signs = rng.choice([-1.0, 1.0], size=(n_days // regime_len) + 1)
        drift = np.repeat(signs, regime_len)[:n_days] * abs(daily_mu) * 3.0
        shocks = drift + rng.normal(0, daily_sig * 0.6, n_days)
        log_path = np.cumsum(shocks)

    elif kind == "mean_reverting":
        # Ornstein-Uhlenbeck around 0 in log space — punishes trend following
        theta, x = 0.05, 0.0
        path = np.empty(n_days)
        for i in range(n_days):
            x += -theta * x + rng.normal(0, daily_sig)
            path[i] = x
        log_path = path

    elif kind == "regime":
        # alternating calm/volatile regimes with differing drift
        path, x = np.empty(n_days), 0.0
        vol_state = daily_sig
        for i in range(n_days):
            if rng.random() < 0.01:
                vol_state = daily_sig * rng.choice([0.5, 1.0, 2.5])
            x += rng.normal(daily_mu * rng.choice([-1, 1]), vol_state)
            path[i] = x
        log_path = path

    else:  # fat_tailed — Student-t shocks
        t_shocks = rng.standard_t(df=3, size=n_days) * daily_sig * 0.6
        log_path = np.cumsum(daily_mu + t_shocks)

    close = s0 * np.exp(log_path)
    return _to_ohlcv(close, index, rng)


def generate_panel(
    n_assets: int = 6,
    kind: str = "trending",
    seed: int | None = 42,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Generate a dict of {SYNTH_i: OHLCV} for cross-sectional testing."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        s = None if seed is None else seed + i
        out[f"SYNTH_{i+1}"] = generate(kind=kind, seed=s, **kwargs)
    return out

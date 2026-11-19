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
# Kinds that learn from a real ticker's history (``from_ticker``).
TICKER_KINDS = ("bootstrap", "shuffle", "gbm_fit", "regime_fit")
KIND_HELP = {
    "trending": "Drift that flips sign every few months: a trend rule MUST win here.",
    "mean_reverting": "Ornstein–Uhlenbeck in log price: trend rules should lose, reversal rules win.",
    "regime": "Calm / normal / wild vol regimes with random drift.",
    "fat_tailed": "Student-t (3 d.f.) shocks: crashes and spikes far more often than a bell curve.",
    "gbm": "Pure random walk with drift: no rule has an edge, so any 'edge' you see is noise.",
    "bootstrap": "Stationary block bootstrap of the base ticker's real bars (mean block 20 days): keeps its "
                 "fat tails, vol clustering and short-run autocorrelation, but reshuffles history.",
    "shuffle": "The base ticker's real bars in random order: same return distribution, but every trend and "
               "vol cluster destroyed. A trend edge that survives this is suspect.",
    "gbm_fit": "Random walk with the base ticker's own drift and vol: the 'no edge' world at realistic scale.",
    "regime_fit": "Two-state (calm / stressed) Markov model fitted to the base ticker's vol, drift and "
                  "regime persistence.",
}


def _to_ohlcv(close: np.ndarray, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Build a plausible OHLCV frame from a close-price path."""
    close = np.asarray(close, dtype=float)
    intraday = np.abs(rng.normal(0, 0.005, size=close.shape)) * close
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    # bars must contain their own open and close (PARALLAX's did not ~1/3 of the time)
    high = np.maximum(openp, close) + intraday
    low = np.minimum(openp, close) - intraday
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


# ------------------------------------------------------------ from a ticker ----
def _bar_ratios(base: pd.DataFrame) -> np.ndarray:
    """Each real bar as (open, high, low, close) relative to the previous close."""
    b = base[["open", "high", "low", "close"]].astype(float).dropna()
    prev = b["close"].shift(1)
    rel = b.div(prev, axis=0).iloc[1:]
    rel = rel[(rel > 0).all(axis=1)]
    return rel.to_numpy()


def _from_ratios(rel: np.ndarray, index: pd.DatetimeIndex, s0: float, rng: np.random.Generator) -> pd.DataFrame:
    close = s0 * np.cumprod(rel[:, 3])
    prev = np.concatenate([[s0], close[:-1]])
    o, h, lo = rel[:, 0] * prev, rel[:, 1] * prev, rel[:, 2] * prev
    high = np.maximum.reduce([h, o, close])
    low = np.minimum.reduce([lo, o, close])
    vol = rng.integers(1_000_000, 5_000_000, size=close.shape)
    return pd.DataFrame({"open": o, "high": high, "low": low, "close": close, "volume": vol}, index=index)


def stationary_bootstrap_idx(n_src: int, n_out: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap: geometric block lengths, wrap-around."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n_out, dtype=int)
    cur = int(rng.integers(n_src))
    for i in range(n_out):
        if i > 0 and rng.random() < p:
            cur = int(rng.integers(n_src))
        idx[i] = cur
        cur = (cur + 1) % n_src
    return idx


def from_ticker(kind: str, base: pd.DataFrame, seed: int | None = 42, mean_block: float = 20.0) -> pd.DataFrame:
    """A synthetic path learned from ``base`` (a real OHLCV frame), on the base's own dates.

    Every kind is reproducible from ``seed`` and has the same length and calendar as ``base``,
    so it slots into the LAB in place of the real series.
    """
    if kind not in TICKER_KINDS:
        raise ValueError(f"Unknown ticker kind {kind!r}; choose from {TICKER_KINDS}.")
    rel = _bar_ratios(base)
    if len(rel) < 60:
        raise ValueError("base ticker needs at least 60 bars")
    rng = np.random.default_rng(seed)
    n = len(rel)
    index = base.index[-n:]
    s0 = float(base["close"].dropna().iloc[0])
    if kind == "bootstrap":
        return _from_ratios(rel[stationary_bootstrap_idx(n, n, mean_block, rng)], index, s0, rng)
    if kind == "shuffle":
        return _from_ratios(rel[rng.permutation(n)], index, s0, rng)

    lr = np.log(rel[:, 3])
    if kind == "gbm_fit":
        shocks = rng.normal(lr.mean(), lr.std(ddof=1), n)
    else:  # regime_fit — two vol states split at the median of 20-day realised vol
        rv = pd.Series(lr).rolling(20, min_periods=5).std().bfill().to_numpy()
        hi = rv > np.median(rv)
        mu = (lr[~hi].mean(), lr[hi].mean())
        sd = (lr[~hi].std(ddof=1), lr[hi].std(ddof=1))
        stay = []
        for st_ in (False, True):
            m = hi[:-1] == st_
            stay.append(float(np.mean(hi[1:][m] == st_)) if m.any() else 0.95)
        state = bool(rng.random() < hi.mean())
        shocks = np.empty(n)
        for i in range(n):
            if rng.random() > stay[int(state)]:
                state = not state
            shocks[i] = rng.normal(mu[int(state)], sd[int(state)])
    close = s0 * np.exp(np.cumsum(shocks))
    return _to_ohlcv(close, index, rng)

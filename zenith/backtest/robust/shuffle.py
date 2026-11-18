"""Permutation, noise and synthetic-market tests.

- **Permutation** (Masters' bar permutation): shuffle the bars' intrabar shapes
  and overnight gaps. The shuffled market keeps the same return distribution but
  has no time structure, so a rule that exploits real structure should beat
  nearly every shuffle. With ``grid`` set, the *whole optimisation* is re-run
  on every shuffle (in-sample permutation test): you compare your best variant
  with the best variant luck could find, which is the honest comparison.
- **Noise**: perturb returns by k·σ and watch Sharpe decay. A rule that
  collapses at small k was keyed to exact prices.
- **Synthetic markets**: known-answer worlds. A trend rule should make money on
  'trending', lose on 'mean_reverting', and earn ~0 on a random walk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.providers import resample_ohlcv, TIMEFRAMES
from ..data.synthetic import generate
from ..spec import StrategySpec, run_spec
from .runner import pmap


def permute_ohlcv(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One bar-permuted copy of ``df`` (first bar kept as the anchor)."""
    lo, hi, lw, lc = (np.log(df[c].to_numpy(float)) for c in ("open", "high", "low", "close"))
    n = len(df)
    gap = lo[1:] - lc[:-1]                    # overnight move
    rh, rl, rc = hi[1:] - lo[1:], lw[1:] - lo[1:], lc[1:] - lo[1:]   # intrabar shape vs open
    i_bar = rng.permutation(n - 1)
    i_gap = rng.permutation(n - 1)
    gap, rh, rl, rc = gap[i_gap], rh[i_bar], rl[i_bar], rc[i_bar]
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    o[0], h[0], l[0], c[0] = lo[0], hi[0], lw[0], lc[0]
    for t in range(1, n):
        o[t] = c[t - 1] + gap[t - 1]
        h[t], l[t], c[t] = o[t] + rh[t - 1], o[t] + rl[t - 1], o[t] + rc[t - 1]
    out = pd.DataFrame({"open": np.exp(o), "high": np.exp(h), "low": np.exp(l), "close": np.exp(c)},
                       index=df.index)
    if "volume" in df:
        vol = df["volume"].to_numpy()
        out["volume"] = np.r_[vol[0], vol[1:][i_bar]]
    return out


def _best(spec: StrategySpec, df: pd.DataFrame, aux, grid) -> float:
    if not grid:
        return run_spec(spec, df, aux).stats["Sharpe"]
    param, values = grid
    return max(run_spec(spec.with_params(**{param: v}), df, aux).stats["Sharpe"] for v in values)


def permutation_test(spec: StrategySpec, ohlcv: pd.DataFrame, n: int = 100, grid: tuple | None = None,
                     aux: dict | None = None, seed: int = 11, progress=None) -> dict:
    real = _best(spec, ohlcv, aux, grid)
    seeds = np.random.SeedSequence(seed).spawn(n)

    def one(ss):
        return _best(spec, permute_ohlcv(ohlcv, np.random.default_rng(ss)), aux, grid)

    perms = np.array(pmap(one, seeds, progress), dtype=float)
    p = (1 + np.sum(perms >= real)) / (n + 1)
    return {"real": float(real), "perms": perms, "p": float(p), "n": n, "optimised": bool(grid)}


def add_noise(df: pd.DataFrame, k: float, rng: np.random.Generator) -> pd.DataFrame:
    lr = np.log(df["close"]).diff().fillna(0.0).to_numpy()
    sig = lr[1:].std()
    noisy = lr + rng.normal(0.0, k * sig, len(lr)) * (np.arange(len(lr)) > 0)
    new_close = df["close"].iloc[0] * np.exp(np.cumsum(noisy))
    ratio = new_close / df["close"].to_numpy()
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = df[c].to_numpy() * ratio
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    return out


def noise_curve(spec: StrategySpec, ohlcv: pd.DataFrame, ks=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
                seeds: int = 12, aux: dict | None = None, progress=None) -> pd.DataFrame:
    jobs = [(k, s) for k in ks for s in range(seeds if k > 0 else 1)]

    def one(j):
        k, s = j
        df = ohlcv if k == 0 else add_noise(ohlcv, k, np.random.default_rng(1000 + s))
        return {"k": k, "seed": s, "Sharpe": run_spec(spec, df, aux).stats["Sharpe"]}

    return pd.DataFrame(pmap(one, jobs, progress))


SYNTH_KINDS = ("trending", "mean_reverting", "gbm", "regime", "fat_tailed")


def synthetic_suite(spec: StrategySpec, seeds: int = 6, n_days: int = 3000, progress=None) -> pd.DataFrame:
    jobs = [(k, s) for k in SYNTH_KINDS for s in range(seeds)]

    def one(j):
        k, s = j
        df = resample_ohlcv(generate(k, n_days=n_days, seed=500 + s), TIMEFRAMES[spec.tf])
        return {"market": k, "seed": s, "Sharpe": run_spec(spec, df).stats["Sharpe"]}

    return pd.DataFrame(pmap(one, jobs, progress))

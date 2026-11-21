"""Overfitting statistics on a family of strategy variants.

All of these work on a matrix of per-bar returns, one column per variant you
tried (a parameter grid), so the variants are run once and every test reuses
them.

- PBO via CSCV (Bailey, Borwein, López de Prado & Zhu 2017): split time into S
  blocks, for every way of picking half of them as "in-sample" find the best
  variant there and see where it ranks on the other half. PBO = how often the
  in-sample winner lands in the bottom half out-of-sample.
- CPCV (López de Prado 2018): N groups, k held out per split, pick the best
  variant on the rest (purged + embargoed around the test groups), stitch the
  held-out returns into φ = k/N·C(N,k) full out-of-sample paths.
- Reality Check (White 2000) and SPA (Hansen 2005): is the best variant's mean
  return better than zero once you account for having searched over all of
  them? Stationary-bootstrap p-values.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from ..engine.montecarlo import auto_block_length


def _sharpe_cols(x: np.ndarray, periods: int = 252) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mu / sd * np.sqrt(periods), 0.0)


# ------------------------------------------------------------------ CSCV ----
def pbo(R: pd.DataFrame, n_blocks: int = 16, max_combos: int = 2000, seed: int = 0,
        periods: int = 252) -> dict:
    """Probability of backtest overfitting. ``R`` = bars × variants."""
    X = R.fillna(0.0).to_numpy()
    T, N = X.shape
    S = max(4, n_blocks - n_blocks % 2)
    if N < 2 or T < S * 10:
        return {"pbo": np.nan, "logits": np.array([]), "is_sr": np.array([]), "oos_sr": np.array([]), "n": 0}
    blocks = np.array_split(np.arange(T), S)
    combos = list(combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    logits, is_sr, oos_sr = [], [], []
    for c in combos:
        ins = np.concatenate([blocks[i] for i in c])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        s_in, s_out = _sharpe_cols(X[ins], periods), _sharpe_cols(X[oos], periods)
        best = int(np.argmax(s_in))
        rank = (s_out < s_out[best]).sum() + 0.5 * ((s_out == s_out[best]).sum() - 1) + 1   # 1..N
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_sr.append(s_in[best])
        oos_sr.append(s_out[best])
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits, "is_sr": np.array(is_sr),
            "oos_sr": np.array(oos_sr), "n": len(combos)}


# ------------------------------------------------------------------ CPCV ----
def cpcv(R: pd.DataFrame, n_groups: int = 6, k_test: int = 2, embargo: int = 5, periods: int = 252) -> dict:
    """Combinatorial purged CV. Returns the OOS path returns (bars × φ) and per-split picks."""
    X = R.fillna(0.0).to_numpy()
    T, N = X.shape
    groups = np.array_split(np.arange(T), n_groups)
    splits = list(combinations(range(n_groups), k_test))
    n_paths = k_test * len(splits) // n_groups
    # which path each (split, group) contributes to: the j-th time a group is tested -> path j
    seen = {g: 0 for g in range(n_groups)}
    path_ret = np.full((T, n_paths), np.nan)
    picks = []
    for sp in splits:
        test = np.concatenate([groups[g] for g in sp])
        bad = np.zeros(T, dtype=bool)
        for g in sp:
            a, b = groups[g][0], groups[g][-1]
            bad[max(0, a - embargo): min(T, b + 1 + embargo)] = True       # purge + embargo around test
        train = np.where(~bad)[0]
        best = int(np.argmax(_sharpe_cols(X[train], periods)))
        picks.append({"split": sp, "pick": R.columns[best]})
        for g in sp:
            j = seen[g]
            if j < n_paths:
                path_ret[groups[g], j] = X[groups[g], best]
            seen[g] += 1
    paths = pd.DataFrame(path_ret, index=R.index, columns=[f"path {i + 1}" for i in range(n_paths)])
    sr = _sharpe_cols(paths.to_numpy(), periods)
    return {"paths": paths, "sharpe": sr, "picks": picks, "n_splits": len(splits), "n_paths": n_paths}


# ----------------------------------------------------- Reality Check / SPA ----
def _stationary_indices(T: int, block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(block, 1.0)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    jumps = rng.random(T) < p
    starts = rng.integers(0, T, T)
    for t in range(1, T):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
    return idx


def reality_check(R: pd.DataFrame, bench: pd.Series | None = None, n_boot: int = 500, seed: int = 0,
                  block: float | None = None) -> dict:
    """White's Reality Check and Hansen's SPA p-values for 'the best variant beats the benchmark'.

    ``bench`` defaults to zero (cash): the question is whether the best variant earned a positive mean at all."""
    X = R.fillna(0.0).to_numpy()
    if bench is not None:
        X = X - bench.reindex(R.index).fillna(0.0).to_numpy()[:, None]
    T, N = X.shape
    if T < 100 or N < 1:
        return {"rc_p": np.nan, "spa_p": np.nan, "best": None, "block": np.nan}
    block = block or max(2.0, float(np.median([auto_block_length(X[:, j]) for j in range(min(N, 20))])))
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1e-12)
    v_rc = np.sqrt(T) * mu.max()
    t_stat = np.sqrt(T) * mu / sd
    v_spa = max(t_stat.max(), 0.0)
    thresh = -sd * np.sqrt(2 * np.log(np.log(T)) / T)
    mu_c = np.where(mu >= thresh, mu, 0.0)                    # Hansen's consistent re-centring
    rc_n = spa_n = 0
    for _ in range(n_boot):
        idx = _stationary_indices(T, block, rng)
        mb = X[idx].mean(axis=0)
        rc_n += np.sqrt(T) * (mb - mu).max() >= v_rc
        spa_n += max((np.sqrt(T) * (mb - mu_c) / sd).max(), 0.0) >= v_spa
    return {"rc_p": rc_n / n_boot, "spa_p": spa_n / n_boot, "best": R.columns[int(np.argmax(mu))],
            "block": block, "best_t": float(t_stat.max())}


# ------------------------------------------------------------- grid views ----
def top_clusters(df: pd.DataFrame, metric: str, cols: list[str], top: int = 30, k: int = 3, seed: int = 0) -> pd.DataFrame:
    """SharpeEdge: k-means on the top-N parameter sets; the member nearest each centroid is a robust pick."""
    d = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top)
    if len(d) < k + 1:
        return pd.DataFrame()
    from sklearn.cluster import KMeans
    Z = d[cols].astype(float).to_numpy()
    mu, sd = Z.mean(0), Z.std(0)
    Zs = (Z - mu) / np.where(sd > 0, sd, 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Zs)
    out = []
    for c in range(k):
        m = km.labels_ == c
        dist = ((Zs[m] - km.cluster_centers_[c]) ** 2).sum(1)
        rep = d[m].iloc[int(np.argmin(dist))]
        out.append({"cluster": c + 1, "size": int(m.sum()), **{cn: rep[cn] for cn in cols},
                    metric: rep[metric], f"mean {metric}": float(d[m][metric].mean())})
    return pd.DataFrame(out).sort_values("size", ascending=False)


def n_cpcv_paths(n_groups: int, k_test: int) -> int:
    return k_test * comb(n_groups, k_test) // n_groups


def variant_returns(spec, ohlcv: pd.DataFrame, px: str, xs: list, py: str | None = None, ys: list | None = None,
                    aux: dict | None = None, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Run every grid cell once and keep its per-bar returns (bars × variants) plus the trial hashes."""
    from .runner import pmap
    from .sensitivity import set_knob
    from ..spec import run_spec
    combos = [(x, y) for y in (ys or [None]) for x in xs]

    def one(c):
        x, y = c
        s = set_knob(spec, px, x)
        if py:
            s = set_knob(s, py, y)
        try:
            r = run_spec(s, ohlcv, aux).returns
        except Exception:  # noqa: BLE001
            r = pd.Series(np.nan, index=ohlcv.index)
        return (f"{x}" if y is None else f"{x}·{y}"), r, s.hash()

    out = pmap(one, combos, progress)
    R = pd.DataFrame({k: r for k, r, _ in out})
    return R, [h for *_, h in out]

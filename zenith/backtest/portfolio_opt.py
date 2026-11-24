"""PORTFOLIO optimisers for a book of strategies (Phase 5).

Every method is causal: weights are set at each rebalance from the trailing
window of returns strictly before it, then held. Carver's IDM scales the mix
back up to the components' risk (see ``portfolio.py``). The honest comparison
is each method against plain 1/N, walk-forward, with a bootstrap interval on
the Sharpe difference (SharpeEdge; DeMiguel, Garlappi & Uppal 2009).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio import _idm, _weights

METHODS = {
    "equal": "Equal weight (1/N) — fits nothing, hard to beat (DeMiguel, Garlappi & Uppal 2009).",
    "inverse_vol": "Inverse volatility — calmer strategies get more weight.",
    "erc": "Equal risk contribution — each strategy adds the same risk (Maillard, Roncalli & Teiletche 2010).",
    "min_var": "Minimum variance — the least volatile long-only mix (needs a good covariance).",
    "max_sharpe": "Max Sharpe — tilts to the best past Sharpe; means and correlations shrunk, still the most "
                  "overfit-prone.",
    "hrp": "Hierarchical risk parity — cluster by correlation, split risk top-down (López de Prado 2016).",
    "buckets": "Buckets first — equal across strategy types (trend / reversion / other), then equal within "
               "(SharpeEdge's a-priori buckets).",
}


def _cov(win: np.ndarray, shrink: bool = True) -> np.ndarray:
    sd = win.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1e-12)
    if shrink and win.shape[1] > 1:
        from ..volscreen.clusters import shrunk_corr
        C, _ = shrunk_corr(pd.DataFrame(win))
    else:
        C = np.corrcoef(win.T) if win.shape[1] > 1 else np.ones((1, 1))
    return C * np.outer(sd, sd)


def erc_weights(S: np.ndarray, iters: int = 500) -> np.ndarray:
    n = len(S)
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        rc = w * (S @ w)
        w_new = w * (rc.sum() / n / np.where(rc > 1e-18, rc, 1e-18)) ** 0.5
        w_new /= w_new.sum()
        done = np.abs(w_new - w).max() < 1e-10
        w = w_new
        if done:
            break
    return w


def _slsqp(obj, n: int) -> np.ndarray:
    from scipy.optimize import minimize
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"maxiter": 200})
    w = np.clip(res.x if res.success else x0, 0, None)
    return w / w.sum() if w.sum() > 0 else x0


def min_var_weights(S: np.ndarray) -> np.ndarray:
    return _slsqp(lambda w: w @ S @ w, len(S))


def max_sharpe_weights(S: np.ndarray, mu: np.ndarray, shrink_mu: float = 0.5) -> np.ndarray:
    m = (1 - shrink_mu) * mu + shrink_mu * mu.mean()
    return _slsqp(lambda w: -(w @ m) / np.sqrt(max(w @ S @ w, 1e-18)), len(S))


def hrp_weights(S: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    n = len(S)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    D = np.sqrt(np.clip((1 - C) / 2, 0, None))
    np.fill_diagonal(D, 0)
    order = list(leaves_list(linkage(squareform(D, checks=False), "single")))
    w = np.ones(n)

    def cvar(ix):
        sub = S[np.ix_(ix, ix)]
        iv = 1 / np.diag(sub)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            a, b = c[: len(c) // 2], c[len(c) // 2:]
            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def bucket_weights(cols: list, buckets: dict) -> np.ndarray:
    b = [buckets.get(c, "other") for c in cols]
    kinds = sorted(set(b))
    return np.array([1.0 / len(kinds) / b.count(x) for x in b])


def optimised_weights(rets: pd.DataFrame, method: str, window: int = 252, rebalance: int = 21,
                      buckets: dict | None = None) -> pd.DataFrame:
    """Weights set at each rebalance from the trailing ``window`` bars strictly before it, held until the next."""
    if method in ("equal", "inverse_vol"):
        return _weights(rets, method, window)
    n = rets.shape[1]
    if method == "buckets":
        return pd.DataFrame([bucket_weights(list(rets.columns), buckets or {})] * len(rets), index=rets.index,
                            columns=rets.columns)
    X = rets.to_numpy(float)
    W = np.full((len(rets), n), np.nan)
    minp = max(60, window // 2)
    last = np.full(n, 1.0 / n)
    for i in range(len(rets)):
        if i >= minp and (i - minp) % rebalance == 0:
            win = X[max(0, i - window): i]
            S = _cov(win)
            if method == "erc":
                last = erc_weights(S)
            elif method == "min_var":
                last = min_var_weights(S)
            elif method == "max_sharpe":
                last = max_sharpe_weights(S, win.mean(axis=0))
            elif method == "hrp":
                last = hrp_weights(S)
        W[i] = last
    return pd.DataFrame(W, index=rets.index, columns=rets.columns)


def combine_with(rets: pd.DataFrame, method: str, window: int = 252, rebalance: int = 21,
                 buckets: dict | None = None) -> dict:
    rets = rets.dropna(how="any")
    w = optimised_weights(rets, method, window, rebalance, buckets)
    idm = _idm(rets, w, window)
    return {"returns": (rets * w).sum(axis=1) * idm, "weights": w, "idm": idm}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, periods: int = 252, n: int = 1000, block: int = 20,
                   seed: int = 0) -> tuple[float, float, float]:
    """Sharpe(a) − Sharpe(b) with a paired moving-block bootstrap 95% interval."""
    d = pd.concat([a, b], axis=1).dropna().to_numpy()
    if len(d) < 100:
        return (np.nan, np.nan, np.nan)

    def sr(x):
        s = x.std(axis=0, ddof=1)
        return x.mean(axis=0) / np.where(s > 0, s, np.nan) * np.sqrt(periods)

    obs = sr(d)
    rng = np.random.default_rng(seed)
    T = len(d)
    k = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        idx = (rng.integers(0, T - block, k)[:, None] + np.arange(block)).ravel()[:T]
        s = sr(d[idx])
        diffs[i] = s[0] - s[1]
    return float(obs[0] - obs[1]), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def sharpe_vs_n(rets: pd.DataFrame, periods: int = 252, samples: int = 30, seed: int = 0) -> pd.DataFrame:
    """Average equal-weight Sharpe for random subsets of size k, and the textbook curve
    SR_N = SR̄·√(N / (1 + (N−1)ρ̄))."""
    rng = np.random.default_rng(seed)
    cols = list(rets.columns)
    N = len(cols)
    sr_each = rets.mean() / rets.std() * np.sqrt(periods)
    corr = rets.corr().to_numpy()
    rho = (corr.sum() - N) / max(N * (N - 1), 1)
    rows = []
    for k in range(1, N + 1):
        vals = []
        for _ in range(samples if k < N else 1):
            p = rets[list(rng.choice(cols, k, replace=False))].mean(axis=1)
            vals.append(p.mean() / p.std() * np.sqrt(periods) if p.std() > 0 else np.nan)
        rows.append({"N": k, "actual": float(np.nanmean(vals)),
                     "theory": float(sr_each.mean() * np.sqrt(k / (1 + (k - 1) * max(rho, 0))))})
    return pd.DataFrame(rows)


TREND_FAMILIES = {"trend"}
REVERSION_FAMILIES = {"meanrev", "momentum"}
CLASSIC_BUCKETS = {"ema": "trend", "sma": "trend", "gmma": "trend", "crossover": "trend", "tsmom": "trend",
                   "breakout": "trend", "turtle": "trend", "meanrev": "reversion", "strev": "reversion",
                   "carry": "carry", "tom": "calendar", "xsmom": "trend", "lowvol": "other"}


def bucket_of(spec) -> str:
    """Strategy type for the a-priori buckets."""
    if spec.mode == "single":
        if spec.rule.startswith("i_"):
            from .indicators import CATALOG
            fam = CATALOG[spec.rule[2:]].family
            side = (spec.params or {}).get("side", 1)
            if fam in TREND_FAMILIES:
                return "trend" if side >= 0 else "reversion"
            if fam in REVERSION_FAMILIES:
                return "reversion"
            return fam
        return CLASSIC_BUCKETS.get(spec.rule, "other")
    return {"blend": "blend", "conditions": "rules", "ml": "ml"}.get(spec.mode, "other")

"""Correlation baskets: group underlyings that move together, keep one of each.

1. Daily log returns over a window (60 / 120 / 250 days).
2. Correlation shrunk toward the constant-correlation target (Ledoit & Wolf
   2004, "Honey, I shrunk the sample covariance matrix") — sample
   correlations of a few hundred names over a few months are noisy; the
   shrinkage intensity is the closed-form optimum.
3. Distance d = sqrt(2 (1 - rho)) and AVERAGE-LINKAGE agglomerative
   clustering (Lance-Williams update), numpy only.
4. Cut where linkage distance crosses rho* (auto k, clamped), or at a chosen k.
5. The uncorrelated basket: the best-scoring liquid name of each cluster,
   skipping any that correlates above `max_rho` with one already picked.

Diversification is reported as the effective number of independent bets
(eigenvalue participation ratio; beta.estimators.effective_dimension).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def returns_matrix(closes: dict[str, pd.Series], window: int = 120, min_frac: float = 0.8) -> pd.DataFrame:
    df = pd.DataFrame({t: s for t, s in closes.items() if s is not None and len(s)}).sort_index()
    r = np.log(df).diff().iloc[-window:]
    keep = [c for c in r if r[c].notna().mean() >= min_frac]
    return r[keep].fillna(0.0)


def shrunk_corr(r: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf constant-correlation shrinkage on standardised returns."""
    X = r.to_numpy(float)
    T, N = X.shape
    X = (X - X.mean(0)) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), 1.0)
    S = X.T @ X / (T - 1)
    rbar = (S.sum() - N) / (N * (N - 1))
    F = np.full((N, N), rbar)
    np.fill_diagonal(F, 1.0)
    # pi: sum of asymptotic variances of the sample correlations
    Y = X * X
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi = pi_mat.sum()
    gamma = np.linalg.norm(S - F, "fro") ** 2
    kappa = (pi - np.trace(pi_mat)) / gamma if gamma > 0 else 1.0
    delta = float(min(1.0, max(0.0, kappa / T)))
    C = delta * F + (1 - delta) * S
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0), delta


def avg_linkage(C: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Merges (a, b, distance, size) like scipy's linkage; new cluster ids n, n+1, ..."""
    n = len(C)
    D = np.sqrt(np.maximum(2.0 * (1.0 - C), 0.0))
    np.fill_diagonal(D, np.inf)
    size = {i: 1 for i in range(n)}
    active = list(range(n))
    ids = list(range(n))                       # row -> cluster id
    merges = []
    D = D.copy()
    alive = np.ones(n, dtype=bool)
    for step in range(n - 1):
        sub = np.where(alive[:, None] & alive[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if i > j:
            i, j = j, i
        d = float(sub[i, j])
        a, b = ids[i], ids[j]
        na, nb = size[a], size[b]
        merges.append((a, b, d, na + nb))
        new = n + step
        # Lance-Williams average linkage: d(k, i u j) = (na d_ki + nb d_kj) / (na + nb)
        row = (na * D[i] + nb * D[j]) / (na + nb)
        D[i, :] = row
        D[:, i] = row
        D[i, i] = np.inf
        alive[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        ids[i] = new
        size[new] = na + nb
    del active
    return merges


def cut(merges, n: int, k: int) -> np.ndarray:
    """Cluster label per leaf after undoing the last k - 1 merges."""
    k = max(1, min(k, n))
    parent = list(range(2 * n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for step, (a, b, _, _) in enumerate(merges[: n - k]):
        new = n + step
        parent[find(a)] = new
        parent[find(b)] = new
    roots = [find(i) for i in range(n)]
    uniq = {r: c for c, r in enumerate(dict.fromkeys(roots))}
    return np.array([uniq[r] for r in roots])


def auto_k(merges, n: int, rho_star: float = 0.5, lo: int = 6, hi: int = 15) -> int:
    """Number of clusters left when merging stops at correlation rho_star."""
    dmax = math.sqrt(2.0 * (1.0 - rho_star))
    done = sum(1 for m in merges if m[2] <= dmax)
    return int(max(lo, min(hi, n - done)))


def leaf_order(merges, n: int) -> list[int]:
    """Dendrogram leaf order (seriation: similar names sit next to each other)."""
    children = {n + s: (a, b) for s, (a, b, _, _) in enumerate(merges)}
    if not merges:
        return list(range(n))
    out, stack = [], [n + len(merges) - 1]
    while stack:
        x = stack.pop()
        if x < n:
            out.append(x)
        else:
            a, b = children[x]
            stack.extend([b, a])
    return out


def pick_basket(tickers: list[str], labels, C: np.ndarray, scores: dict[str, float],
                eligible: set[str], max_rho: float = 0.6) -> list[str]:
    """Best-scoring eligible name per cluster, best clusters first, with a
    pairwise correlation cap against names already chosen."""
    idx = {t: i for i, t in enumerate(tickers)}
    best = {}
    for t, lab in zip(tickers, labels):
        if t in eligible and math.isfinite(scores.get(t, float("nan"))):
            if lab not in best or scores[t] > scores[best[lab]]:
                best[lab] = t
    chosen: list[str] = []
    for t in sorted(best.values(), key=lambda x: -scores[x]):
        if all(C[idx[t], idx[c]] <= max_rho for c in chosen):
            chosen.append(t)
    return chosen


def build(closes: dict[str, pd.Series], scores: dict[str, float], eligible: set[str],
          window: int = 120, k: int | None = None, max_rho: float = 0.6) -> dict:
    from ..beta.estimators import effective_dimension

    r = returns_matrix(closes, window)
    tickers = list(r.columns)
    n = len(tickers)
    if n < 4:
        return {"window": window, "tickers": tickers, "clusters": [], "basket": []}
    C, delta = shrunk_corr(r)
    merges = avg_linkage(C)
    kk = k or auto_k(merges, n)
    labels = cut(merges, n, kk)
    basket = pick_basket(tickers, labels, C, scores, eligible, max_rho)
    order = leaf_order(merges, n)
    clusters = []
    for lab in sorted(set(labels.tolist())):
        mem = [tickers[i] for i in range(n) if labels[i] == lab]
        ii = [tickers.index(m) for m in mem]
        sub = C[np.ix_(ii, ii)]
        intra = float((sub.sum() - len(ii)) / (len(ii) * (len(ii) - 1))) if len(ii) > 1 else 1.0
        rep = next((b for b in basket if b in mem), None)
        clusters.append({"id": int(lab), "members": mem, "rep": rep, "intra_rho": intra})

    def eff(names):
        if len(names) < 2:
            return float(len(names))
        ii = [tickers.index(t) for t in names]
        v = effective_dimension(C[np.ix_(ii, ii)])
        return float(v) if v is not None else float("nan")

    return {"window": window, "tickers": tickers, "labels": labels.tolist(), "k": int(kk),
            "shrinkage": delta, "merges": [[int(a), int(b), float(d), int(s)] for a, b, d, s in merges],
            "order": order, "clusters": clusters, "basket": basket,
            "eff_basket": eff(basket), "eff_universe": eff(tickers),
            "corr": np.round(C, 3).tolist()}

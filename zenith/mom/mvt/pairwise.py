"""The pairwise relative-strength engine: vectorized NxN spread computation,
exactly as benchmarked (N=1000, T=504: ~0.07s, ~8MB) before this feature was
built -- see mvt/__init__.py for why that number matters (it makes the
section-26 "computationally intelligent, no naive N^2" requirement a
non-issue rather than an architecture problem).

Two layers, same math, different input series (see mvt/__init__.py):
  * RAW      on total returns      -> `spread_matrix(total_returns, ...)`
  * RESIDUAL on PCA residuals      -> `spread_matrix(residual_returns, ...)`

STORAGE ARCHITECTURE: the NxN matrix this module builds is NEVER written to
disk. What gets committed (by mvt/compute.py) is the smaller set of inputs
that reconstruct it exactly on demand: horizon return vectors (total AND
residual), the per-name variance needed for the spread-vol formula, and the
PCA loadings/eigenvalues if a caller needs to rebuild a residual vector for
an arbitrary sub-window. A committed NxN would also make "compare any two
tickers the user picks" impossible for anything the nightly job didn't
already choose to store -- keeping only the vectors keeps the interactive
matrix (mvt/view.py) genuinely interactive over ANY subset, not just a
pre-baked top-K.

A vs B is exactly -(B vs A) by construction (spread_matrix is antisymmetric,
diag zero) -- section 26's "don't calculate duplicate pairs twice" is
therefore structural, not a caching decision: only the upper triangle is
ever touched for anything that doesn't need the full antisymmetric matrix
(peer percentiles do -- one row's percentile needs the sign of every
column -- but that's still one O(N^2) pass, not two).
"""

from __future__ import annotations

import numpy as np


def spread_matrix(r: np.ndarray, var: np.ndarray, horizon_days: int,
                  min_spread_vol: float = 1e-8) -> np.ndarray:
    """r: (N,) horizon return vector (log-return sum over the horizon, total
    OR residual). var: (N,) per-name variance of the SAME return series over
    the estimation window (not of the horizon return itself -- this is the
    per-period variance used to scale up to horizon-vol via sqrt(horizon)).

    Returns D, an (N,N) antisymmetric matrix: D[i,j] = (r[i]-r[j]) /
    (spread_vol[i,j] * sqrt(horizon_days)). spread_vol[i,j] =
    sqrt(var[i]+var[j]) is a conservative (zero-covariance) upper bound on
    the true pairwise spread vol; it deliberately does NOT use the i-j
    covariance term (which would need the full NxN covariance rather than
    just the diagonal) -- close enough for a normalized SIGNAL, and it keeps
    this function O(N) in memory beyond the one NxN output, needing only the
    two length-N vectors already computed by factors.fit()/panel.py.

    `min_spread_vol` guards near-duplicate pairs (should already be gated
    out by universe.duplicate_and_leverage_gate, but this is the numerical
    backstop so a division never actually blows up even if a caller skips
    that gate).
    """
    n = len(r)
    spread_vol = np.sqrt(np.maximum(var[:, None] + var[None, :], min_spread_vol ** 2))
    spread_vol = np.maximum(spread_vol, min_spread_vol)
    D = (r[:, None] - r[None, :]) / (spread_vol * np.sqrt(max(horizon_days, 1)))
    np.fill_diagonal(D, 0.0)
    return D


def peer_stats(D: np.ndarray) -> dict[str, np.ndarray]:
    """Row-wise aggregate stats from an (N,N) antisymmetric spread matrix:
      win_frac    -- fraction of the N-1 peers this name's spread beats (>0)
      mean_spread -- mean pairwise spread vs all peers (the naive aggregate;
                     see mvt/__init__.py on why this alone is ~redundant
                     with cross-sectional rank)
      median_spread, std_spread -- distribution shape vs peers
    """
    n = D.shape[0]
    if n < 2:
        return {"win_frac": np.zeros(n), "mean_spread": np.zeros(n),
                "median_spread": np.zeros(n), "std_spread": np.zeros(n)}
    wins = (D > 0).sum(axis=1) / (n - 1)
    mean_spread = D.sum(axis=1) / (n - 1)
    median_spread = np.median(D, axis=1)
    std_spread = D.std(axis=1)
    return {"win_frac": wins, "mean_spread": mean_spread,
            "median_spread": median_spread, "std_spread": std_spread}


def strongest_weakest(D: np.ndarray, tickers: list[str], i: int, top: int = 5) -> dict:
    """For ticker at row `i`: its `top` strongest (most positive spread) and
    weakest (most negative spread) pairwise relationships -- section 10's
    "pair-wise strongest/weakest relationships" drilldown. O(N log N) per
    call, only run for the one instrument the user has selected, never for
    the whole universe."""
    row = D[i]
    others = [j for j in range(len(row)) if j != i]
    order = sorted(others, key=lambda j: row[j])
    weakest_idx = order[:top]
    strongest_idx = order[::-1][:top]
    return {
        "strongest": [{"ticker": tickers[j], "spread": round(float(row[j]), 4)} for j in strongest_idx],
        "weakest": [{"ticker": tickers[j], "spread": round(float(row[j]), 4)} for j in weakest_idx],
    }


def submatrix(D: np.ndarray, tickers: list[str], subset: list[str]) -> tuple[np.ndarray, list[str]]:
    """Slices D down to a user-chosen subset of tickers (for the interactive
    matrix view -- see mvt/__init__.py's storage-architecture note: this is
    what lets the UI stay interactive over any basket without a committed
    NxN)."""
    idx = {t: k for k, t in enumerate(tickers)}
    order = [idx[t] for t in subset if t in idx]
    found = [tickers[k] for k in order]
    if not order:
        return np.zeros((0, 0)), []
    return D[np.ix_(order, order)], found

"""Multivariate Trend statistical factor model: a k-factor PCA decomposition
of the return panel.

One computation does three jobs (see mvt/__init__.py's module docstring for
the full rationale):

  1. Makes the covariance usable. With N ~ 1000 names and T ~ 252-504 days,
     the sample covariance is rank-deficient (N > T); a k-factor +
     idiosyncratic-diagonal model is well-conditioned where the raw sample
     covariance is not.
  2. Supplies the RESIDUAL return series pairwise.py needs for the
     normalized/residual score -- the part of each instrument's return that
     is NOT explained by the universe's own dominant common movements.
  3. Is the section-20 regime diagnostic: eigenvalues, explained variance,
     and an effective-factor count (Shannon-entropy based). Purely
     descriptive -- see the docstring on `effective_factor_count` for the
     explicit "this is not a signal" framing the spec requires.

No look-ahead: the factor model is fit on a single TRAILING window ending at
the current day (config.MOM_MVT_COV_WINDOW, default 252d) -- it never uses
any observation the caller hasn't already restricted the panel to.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...config import MOM_MVT_COV_WINDOW


def _k_from_variance(eigvals_desc: np.ndarray, threshold: float = 0.60, k_min: int = 3,
                     k_max: int = 15) -> int:
    """Fixed, documented rule for the factor count -- NOT tuned to make any
    particular result look good (section 33's "avoid overfitting" applies
    here as much as to the weighting): take the smallest k that explains at
    least `threshold` of total variance, clamped to [k_min, k_max]. A market-
    plus-a-handful-of-sector-factors structure typically needs single digits
    of factors to clear 60% of variance in a broad equity/ETF panel; the cap
    keeps the model from chasing noise eigenvalues when it doesn't."""
    total = eigvals_desc.sum()
    if total <= 0:
        return k_min
    cum = np.cumsum(eigvals_desc) / total
    k = int(np.searchsorted(cum, threshold) + 1)
    return max(k_min, min(k_max, k))


def fit(returns: pd.DataFrame, window: int = MOM_MVT_COV_WINDOW) -> dict | None:
    """Fits the k-factor model on the trailing `window` days of `returns`
    (a wide DataFrame, columns=tickers, already aligned/winsorized by
    panel.py). Returns None if there isn't enough history for a stable fit.

    Returns a dict of numpy arrays / plain values:
      tickers        -- column order, so callers can align back to names
      loadings       -- (N, k) factor loadings (eigenvectors, unit-normed)
      eigenvalues    -- (k,) the retained eigenvalues, descending
      factor_returns -- (window, k) the k orthogonal factor return series
      idio_vol       -- (N,) residual (idiosyncratic) daily vol per name
      residuals      -- (window, N) the residual return matrix used by
                        pairwise.py's residual layer
      explained_variance_ratio -- float, fraction of total variance the
                        retained k factors explain
      effective_factor_count   -- float, Shannon-entropy based (section 20)
      k              -- factors retained
    """
    R = returns.tail(window)
    R = R.dropna(axis=1, thresh=max(20, int(window * 0.6)))
    R = R.fillna(0.0)
    tickers = list(R.columns)
    n, N = R.shape
    if N < 5 or n < 20:
        return None

    X = R.to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    C = np.cov(X, rowvar=False)

    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    w = np.clip(w, 0.0, None)

    k = _k_from_variance(w)
    k = min(k, N - 1, n - 1) if min(N, n) > 1 else 1
    k = max(k, 1)
    loadings = V[:, :k]
    eigenvalues = w[:k]

    factor_returns = X @ loadings                      # (n, k)
    # OLS betas of each name on the k factor series (factors are orthogonal
    # by construction, so this is just a projection, not a numerically
    # fragile inversion).
    denom = (factor_returns ** 2).sum(axis=0)
    denom[denom == 0] = 1.0
    betas = (factor_returns.T @ X) / denom[:, None]     # (k, N)
    fitted = factor_returns @ betas                      # (n, N)
    residuals = X - fitted
    idio_vol = residuals.std(axis=0)

    total_var = float(w.sum())
    explained = float(eigenvalues.sum() / total_var) if total_var > 0 else 0.0

    return {
        "tickers": tickers,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "betas": betas,
        "factor_returns": factor_returns,
        "residuals": residuals,
        "idio_vol": idio_vol,
        "explained_variance_ratio": round(explained, 4),
        "effective_factor_count": effective_factor_count(w),
        "k": k,
        "n_obs": n,
    }


def effective_factor_count(eigvals_desc: np.ndarray) -> float:
    """Shannon-entropy-based effective number of factors (section 20):
    exp(entropy of the normalized eigenvalue spectrum). Ranges from 1 (one
    factor explains everything -- maximally macro-dominated) to N (all
    eigenvalues equal -- maximally idiosyncratic/diversified).

    THIS IS A DESCRIPTIVE REGIME DIAGNOSTIC, NOT A SIGNAL. A low reading
    means recent cross-instrument behavior is being explained by fewer
    dominant statistical factors -- it says nothing, on its own, about
    whether trend-following will work well or poorly. The UI must not
    render this as "low factor count -> buy trend" (the spec explicitly
    warns against exactly that framing)."""
    w = np.asarray(eigvals_desc, dtype=float)
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    p = w / w.sum()
    entropy = -float((p * np.log(p)).sum())
    return round(float(math.exp(entropy)), 3)

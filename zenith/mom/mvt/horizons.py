"""Disjoint-horizon decomposition and equal-risk-contribution (ERC) weighting.

Two distinct jobs share this module because they are the SAME piece of math
applied at two levels (per the plan's design: one implementation, one test
suite):

  1. `disjoint_increments()` fixes the nested-horizon double-counting the
     spec calls out (section 11): 12M literally contains 1M/3M/6M/9M, so a
     naive weighted average of the six nested horizons overweights whatever
     is common to all of them (recent return). Building on non-overlapping
     increments (0-1M, 1-3M, 3-6M, 6-9M, 9-12M) instead means each increment
     is informationally distinct BY CONSTRUCTION -- no correlation
     estimation, no fitting, nothing to overfit.

  2. `erc_weights()` is a genuinely different, complementary idea: given
     that a set of inputs (increments OR whole factors) ARE correlated
     (which the nested horizons no longer are once decomposed, but the six
     MOMENTUM FACTORS very much still are -- ts/xsec measured at 0.997),
     equal-risk-contribution asks each input to contribute an equal SHARE OF
     COMPOSITE VARIANCE rather than an equal share of raw weight. A highly
     redundant pair (ts/xsec) each get less weight under ERC than under
     equal weighting, because together they'd otherwise double-count. This
     is a correlation-structure calculation with NO return/performance data
     involved -- nothing here is fit to a backtest, which is what makes it
     defensible under the spec's section 33 "avoid overfitting" requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import HORIZONS, INCREMENTS


# Each nested horizon as a sum of the disjoint increments it's built from.
# "12_1" = 12-month momentum excluding the most recent month (classic UMD
# spec) = every increment except "0_1m".
HORIZON_INCREMENTS = {
    "1m": ("0_1m",),
    "3m": ("0_1m", "1_3m"),
    "6m": ("0_1m", "1_3m", "3_6m"),
    "9m": ("0_1m", "1_3m", "3_6m", "6_9m"),
    "12m": ("0_1m", "1_3m", "3_6m", "6_9m", "9_12m"),
    "12_1": ("1_3m", "3_6m", "6_9m", "9_12m"),
}

# (lookback_days, skip_days) bounds for each disjoint increment, matching
# the trading-day conventions mom.factors.HORIZON_SPEC already uses (21d/mo).
INCREMENT_SPEC = {
    "0_1m": (0, 21),
    "1_3m": (21, 63),
    "3_6m": (63, 126),
    "6_9m": (126, 189),
    "9_12m": (189, 252),
}


def increment_return(log_close: pd.Series, start_back: int, end_back: int) -> float | None:
    """Log-return over the disjoint window [t-end_back, t-start_back], i.e.
    the price move that happened BETWEEN those two points back from today --
    not the cumulative return from today. `log_close` is log(price)."""
    n = len(log_close)
    if n < end_back + 1:
        return None
    end_val = log_close.iloc[-1 - start_back]
    start_val = log_close.iloc[-1 - end_back]
    if not (np.isfinite(end_val) and np.isfinite(start_val)):
        return None
    return float(end_val - start_val)


def increments_for_panel(log_returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Given a (T, N) daily log-return panel, returns {increment_key ->
    (N,) Series of that disjoint window's summed log return per ticker}.
    Summing log returns over a window IS the window's log return, which is
    exactly why disjoint increments recombine additively into any nested
    horizon (unlike simple returns)."""
    log_close = log_returns.cumsum()
    out = {}
    for key, (start_back, end_back) in INCREMENT_SPEC.items():
        col_vals = {}
        for t in log_returns.columns:
            r = increment_return(log_close[t], start_back, end_back)
            if r is not None:
                col_vals[t] = r
        out[key] = pd.Series(col_vals)
    return out


def nested_from_increments(inc_returns: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Reconstructs each nested horizon (12_1/12m/9m/6m/3m/1m) as the exact
    sum of its disjoint increments -- so the nested horizons stay
    independently visible per spec section 6 while sharing one underlying,
    non-double-counted computation.

    A ticker missing from even ONE required increment is dropped from the
    nested horizon's result entirely, rather than left in with a NaN value.
    This matters more than it looks: `increment_return()` already excludes
    a ticker from a single increment cleanly (returns None -> the ticker's
    index label simply never enters that increment's Series -- an interior
    data gap, e.g. a trading halt surviving the panel's 3-day forward-fill
    limit, is the real-world cause). But summing Series with MISMATCHED
    indices (pandas' default when you add two Series) silently produces NaN
    for any label not present in every operand -- so a ticker present in
    4 of 5 increments would otherwise come out of `sum()` as NaN rather
    than being cleanly excluded. A real production bug caught from this:
    that lone NaN return, once it reached the pairwise engine, poisoned
    EVERY OTHER ticker's raw_score to exactly 0.0 -- one row of the NxN
    spread matrix containing a NaN taints every row's sum once you total
    across peers, which is exactly the section-28 "one bad instrument must
    not contaminate the whole matrix" failure mode this method exists to
    prevent. `.dropna()` after the sum makes the exclusion explicit and
    total, matching how every other insufficient-data case in this pipeline
    is handled (excluded, never a silent NaN passed downstream)."""
    out = {}
    for h, keys in HORIZON_INCREMENTS.items():
        parts = [inc_returns[k] for k in keys if k in inc_returns]
        if not parts:
            continue
        combined = sum(parts[1:], parts[0]) if len(parts) > 1 else parts[0]
        out[h] = combined.dropna()
    return out


# ------------------------------------------------------------------ ERC ---
def erc_weights(corr: pd.DataFrame | np.ndarray, vol: np.ndarray | None = None,
                max_iter: int = 500, tol: float = 1e-10) -> dict:
    """Equal-risk-contribution weights over a correlation (or covariance, if
    `vol` is None and the input already has variances on the diagonal)
    matrix. Contribution of input i to composite variance is
    w_i * (Sigma w)_i / (w' Sigma w); ERC finds w (long-only, sums to 1)
    where every contribution is equal.

    Solved by the standard fixed-point / Newton-lite iteration (no scipy
    dependency -- not a Zenith requirement): start equal-weighted, at each
    step move weight away from inputs whose current contribution exceeds
    the (dynamic) target share, renormalize, repeat. Converges reliably for
    the small (5-10 input) problems this module is used for (factor weights,
    horizon weights) -- this is NOT a general-purpose portfolio optimizer
    and isn't claimed to be one.

    Returns {"weights": {name: w}, "contributions": {name: c}, "iterations": n,
    "converged": bool}. If Sigma isn't valid (fewer than 2 usable inputs,
    non-finite entries), falls back to equal weights with converged=False.
    """
    if isinstance(corr, pd.DataFrame):
        names = list(corr.columns)
        Sigma = corr.to_numpy(dtype=float)
    else:
        Sigma = np.asarray(corr, dtype=float)
        names = [str(i) for i in range(Sigma.shape[0])]

    n = Sigma.shape[0]
    if n < 2 or not np.all(np.isfinite(Sigma)):
        eq = {nm: 1.0 / max(n, 1) for nm in names}
        return {"weights": eq, "contributions": eq, "iterations": 0, "converged": False}

    if vol is not None:
        v = np.asarray(vol, dtype=float)
        D = np.outer(v, v)
        Sigma = Sigma * D

    # Guard against a non-positive-semidefinite input (e.g. a Spearman
    # matrix with pairwise-deleted missing data can be slightly indefinite)
    # by nudging the diagonal -- keeps the iteration numerically stable
    # without materially changing well-conditioned inputs.
    Sigma = Sigma + np.eye(n) * 1e-8

    w = np.full(n, 1.0 / n)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Sw = Sigma @ w
        port_var = float(w @ Sw)
        if port_var <= 0:
            break
        contrib = w * Sw / port_var
        target = 1.0 / n
        # Multiplicative update: scale each weight by (target/contrib)^0.5,
        # a damped step toward equal contribution; renormalize to sum to 1.
        ratio = np.sqrt(np.clip(target / np.maximum(contrib, 1e-12), 0.2, 5.0))
        w_new = w * ratio
        w_new = np.clip(w_new, 1e-6, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            converged = True
            break
        w = w_new

    Sw = Sigma @ w
    port_var = float(w @ Sw)
    contrib = (w * Sw / port_var) if port_var > 0 else np.full(n, 1.0 / n)

    return {
        "weights": {nm: round(float(x), 6) for nm, x in zip(names, w)},
        "contributions": {nm: round(float(x), 6) for nm, x in zip(names, contrib)},
        "iterations": it,
        "converged": converged,
    }

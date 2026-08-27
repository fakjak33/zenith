"""Combines the pairwise engine + factor model + horizon machinery into the
per-instrument Multivariate Trend scores: raw/naive and normalized/residual,
both mapped to -20..+20, plus consistency/conviction diagnostics.

This is the module mvt/compute.py calls once per universe (equities, ETFs)
per run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import inv_normal, clip1
from . import HORIZONS
from . import factors as mvt_factors
from . import horizons as mvt_horizons
from . import pairwise as pw


def _score_from_return_vector(r: pd.Series, var: pd.Series, horizon_days: int) -> tuple[dict, np.ndarray, list[str]]:
    """One horizon's full pairwise pass for one return vector: builds the
    NxN spread matrix (in memory only, per pairwise.py's storage note),
    reduces it to row stats, and returns per-ticker peer stats + the raw
    matrix (for optional drilldown) + the ticker order. Each ticker's own
    return value is carried through too (`r`) -- together with the single
    per-ticker variance already available separately, that is ALL a caller
    needs to call pairwise.spread_matrix() again later for an arbitrary
    subset (the view's interactive-matrix reconstruction; see mvt/compute.py
    and pairwise.py's storage-architecture note) without ever persisting
    the NxN itself."""
    tickers = list(r.index)
    D = pw.spread_matrix(r.to_numpy(dtype=float), var.reindex(tickers).to_numpy(dtype=float), horizon_days)
    stats = pw.peer_stats(D)
    out = {}
    for i, t in enumerate(tickers):
        out[t] = {
            "win_frac": float(stats["win_frac"][i]),
            "mean_spread": float(stats["mean_spread"][i]),
            "r": float(r.loc[t]),
        }
    return out, D, tickers


def compute_universe_scores(log_returns: pd.DataFrame, cov_window: int) -> dict | None:
    """Full pipeline for one universe's aligned, winsorized log-return panel:

      1. Fit the k-factor PCA model on the trailing `cov_window` (factors.fit).
      2. Build disjoint increments on BOTH total returns and PCA residuals.
      3. For every nested horizon, run the pairwise engine on total returns
         (RAW layer) and on residual returns (RESIDUAL layer) -- see
         mvt/__init__.py for why these are kept as two named layers rather
         than blended into one number.
      4. Aggregate horizons with the naive weights (raw score) and with ERC
         weights computed from the residual layer's own cross-horizon
         correlation (normalized score) -- per horizons.py's "one
         implementation, two levels" design.
      5. Map both to -20..+20 via the same inv_normal transform xsec uses,
         so the scale matches the rest of the Momentum tab.

    Returns None if the fit or the return panel is unusable (caller then
    excludes this universe's contribution for the run, same fail-soft
    posture as mom.compute's own missing-data handling).
    """
    if log_returns is None or log_returns.empty or log_returns.shape[1] < 10:
        return None

    # The longest disjoint increment (9-12M) needs a return 252 trading days
    # back, i.e. >=253 observations in the SAME window the residuals are
    # computed over -- widen defensively so a caller passing too small a
    # window doesn't silently starve the 12M/12-1 horizons (see the
    # config.MOM_MVT_COV_WINDOW comment for the same reasoning).
    cov_window = max(cov_window, 260)

    fit = mvt_factors.fit(log_returns, window=cov_window)
    if fit is None:
        return None

    tickers = fit["tickers"]
    residual_df = pd.DataFrame(fit["residuals"], columns=tickers)
    # NOT fillna(0.0)'d here (unlike factors.fit()'s internal covariance
    # copy) -- a ticker with a genuine interior data gap (survives
    # factors.fit()'s 60%-coverage dropna filter but still has NaN days
    # inside this window) must be EXCLUDED from any horizon it can't
    # honestly compute, not given a fabricated zero return. That exclusion
    # is handled correctly downstream by horizons.nested_from_increments()
    # (its own docstring has the full incident writeup) -- see that
    # function for why a plain Series-addition of increments would
    # otherwise leak a NaN into one ticker's return and, via the pairwise
    # engine's row-sum, silently zero out raw_score for the ENTIRE universe.
    total_df = log_returns[tickers].tail(fit["n_obs"]).reset_index(drop=True)

    total_var = total_df.var(ddof=1)
    resid_var = residual_df.var(ddof=1)

    inc_total = mvt_horizons.increments_for_panel(total_df)
    inc_resid = mvt_horizons.increments_for_panel(residual_df)
    nested_total = mvt_horizons.nested_from_increments(inc_total)
    nested_resid = mvt_horizons.nested_from_increments(inc_resid)

    horizon_days = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252, "12_1": 231}

    raw_by_h, resid_by_h = {}, {}
    matrices = {}
    for h in HORIZONS:
        r_total = nested_total.get(h)
        r_resid = nested_resid.get(h)
        if r_total is None or r_resid is None or r_total.empty:
            continue
        common = r_total.index.intersection(r_resid.index)
        if len(common) < 10:
            continue
        stats_total, D_total, order = _score_from_return_vector(
            r_total.loc[common], total_var.loc[common], horizon_days[h])
        stats_resid, D_resid, _ = _score_from_return_vector(
            r_resid.loc[common], resid_var.loc[common], horizon_days[h])
        raw_by_h[h] = stats_total
        resid_by_h[h] = stats_resid
        matrices[h] = {"raw": D_total, "residual": D_resid, "tickers": order}

    if not raw_by_h:
        return None

    from ...config import MOM_HORIZON_WEIGHTS
    all_tickers = sorted(set().union(*[set(v) for v in raw_by_h.values()]))

    # --- horizon correlation for ERC (on the RESIDUAL layer's win_frac,
    # since that is the layer feeding the normalized score) ----------------
    resid_frame = pd.DataFrame({h: {t: v["win_frac"] for t, v in resid_by_h[h].items()} for h in resid_by_h})
    resid_frame = resid_frame.reindex(all_tickers)
    horizon_corr = resid_frame.corr(method="spearman").fillna(0.0)
    n_h = len(resid_by_h)
    eq_weights = {h: 1.0 / n_h for h in resid_by_h}
    if horizon_corr.shape[0] >= 2:
        erc = mvt_horizons.erc_weights(horizon_corr)
        # Shrink 50% toward equal weight. Pure ERC has no notion of SIGNAL
        # QUALITY, only of correlation -- verified on a synthetic panel while
        # building this: the least-correlated horizon (shortest, noisiest)
        # got ~37% of the weight unshrunk, which is a real risk-parity
        # result but not necessarily a good SIGNAL result (the shortest
        # horizon is also the most reversal-prone -- Jegadeesh 1990, the
        # same reasoning config.MOM_HORIZON_WEIGHTS already documents for
        # the existing ts/xsec factors). A fixed 50/50 shrinkage toward the
        # simple, transparent default is the repo's own stated preference
        # (spec section 33: "prefer a transparent/simple weighting method"
        # when an approach isn't statistically defensible on available
        # data) -- not tuned to any particular outcome.
        erc_weights = {h: 0.5 * erc["weights"].get(h, 0.0) + 0.5 * eq_weights[h] for h in resid_by_h}
    else:
        erc_weights = eq_weights

    rows = []
    for t in all_tickers:
        raw_vals = {h: raw_by_h[h][t]["mean_spread"] for h in raw_by_h if t in raw_by_h[h]}
        resid_vals = {h: resid_by_h[h][t]["mean_spread"] for h in resid_by_h if t in resid_by_h[h]}
        raw_pct = {h: raw_by_h[h][t]["win_frac"] * 100.0 for h in raw_by_h if t in raw_by_h[h]}
        resid_pct = {h: resid_by_h[h][t]["win_frac"] * 100.0 for h in resid_by_h if t in resid_by_h[h]}

        raw_agg = _weighted_mean(raw_vals, MOM_HORIZON_WEIGHTS)

        rows.append({
            "ticker": t,
            "raw_percentiles": {h: round(v, 2) for h, v in raw_pct.items()},
            "residual_percentiles": {h: round(v, 2) for h, v in resid_pct.items()},
            "residual_mean_spread": {h: round(v, 4) for h, v in resid_vals.items()},
            "raw_score": round(clip1(raw_agg) * 20.0, 3) if raw_agg is not None else None,
            "normalized_score": round(_pctile_to_score(resid_pct, erc_weights), 3) if resid_pct else None,
            "consistency": _consistency(resid_pct),
            # Reconstruction inputs for the interactive matrix (see
            # pairwise.py's storage-architecture note): the per-horizon
            # total/residual return values plus this ticker's single
            # per-series variance (spread_matrix scales it to horizon-vol
            # internally via sqrt(horizon_days), so ONE variance per series
            # -- not one per horizon -- is all that's needed). Together
            # these exactly reproduce any submatrix the view asks for
            # without the NxN ever touching disk.
            "total_return": {h: round(raw_by_h[h][t]["r"], 6) for h in raw_by_h if t in raw_by_h[h]},
            "residual_return": {h: round(resid_by_h[h][t]["r"], 6) for h in resid_by_h if t in resid_by_h[h]},
            "total_var": round(float(total_var.get(t)), 8) if t in total_var.index and pd.notna(total_var.get(t)) else None,
            "resid_var": round(float(resid_var.get(t)), 8) if t in resid_var.index and pd.notna(resid_var.get(t)) else None,
        })

    return {
        "rows": rows,
        "erc_horizon_weights": {k: round(v, 4) for k, v in erc_weights.items()},
        "explained_variance_ratio": fit["explained_variance_ratio"],
        "effective_factor_count": fit["effective_factor_count"],
        "k_factors": fit["k"],
        "n_tickers": len(all_tickers),
        "matrices": matrices,     # in-memory only -- compute.py does NOT persist this key
    }


def _weighted_mean(values: dict, weights: dict) -> float | None:
    tw = tv = 0.0
    for k, w in weights.items():
        v = values.get(k)
        if v is None:
            continue
        tv += w * v
        tw += w
    return (tv / tw) if tw > 0 else None


def _pctile_to_score(pctiles: dict, weights: dict) -> float:
    """Averages the van der Waerden-transformed per-horizon percentiles
    (weighted by the ERC horizon weights), then scales to -20..+20 -- the
    same normalization xsec already uses (mom.normalize.inv_normal), so
    Multivariate Trend sits on an identical scale to every other factor."""
    tw = tv = 0.0
    for h, p in pctiles.items():
        w = weights.get(h, 0.0)
        if w <= 0:
            continue
        tv += w * inv_normal(p)
        tw += w
    z = (tv / tw) if tw > 0 else 0.0
    return clip1(z) * 20.0


def _consistency(pctiles: dict) -> dict:
    """Timeframe agreement (section 21): how many horizons agree on
    direction (>55 bullish / <45 bearish), plus short-vs-long acceleration
    (1M percentile minus 12M percentile) as a simple, transparent momentum-
    of-momentum read."""
    if not pctiles:
        return {"n_bullish": 0, "n_bearish": 0, "n_neutral": 0, "agreement": None, "acceleration": None}
    n_bull = sum(1 for v in pctiles.values() if v >= 55)
    n_bear = sum(1 for v in pctiles.values() if v <= 45)
    n = len(pctiles)
    n_neutral = n - n_bull - n_bear
    agreement = max(n_bull, n_bear) / n
    accel = None
    if "1m" in pctiles and "12m" in pctiles:
        accel = round(pctiles["1m"] - pctiles["12m"], 2)
    return {"n_bullish": n_bull, "n_bearish": n_bear, "n_neutral": n_neutral,
            "agreement": round(agreement, 3), "acceleration": accel}

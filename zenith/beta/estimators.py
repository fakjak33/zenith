"""CLEAN BETA estimators — pure numpy, no I/O.

bswa  : Welch (2022), "Simply Better Market Betas", Critical Finance Review
        11(1). Slope-winsorize each stock return into the band
        [(1-δ)·r_m, (1+δ)·r_m] (lower/upper taken as min/max so the band is
        right on down days too), then weighted least squares with weights
        exp(-λ·age_in_years). δ=3, λ=2 are his recommended defaults. Uses raw
        daily returns, not excess returns: at a daily horizon the risk-free
        rate is ~0.016%/day and moves the slope by far less than rounding.
ols   : the plain market model over the last BETA_LOOKBACK days — beta, ρ,
        R², annualized IVOL (residual sd), total vol and the count of days
        with |residual| > k·σ (event days). Kept for comparison and because
        IVOL and ρ are defined on it.
vasicek: cross-sectional Bayesian shrinkage of OLS beta toward the universe
        mean (Vasicek 1973) — computed so the PARALLAX grid has all three.
"""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS = 252


def bswa(r_i: np.ndarray, r_m: np.ndarray, delta: float = 3.0, lam: float = 2.0) -> float | None:
    """Welch slope-winsorized, age-decayed beta. Inputs are aligned daily
    returns, OLDEST FIRST (age is measured back from the last observation)."""
    r_i = np.asarray(r_i, dtype=float)
    r_m = np.asarray(r_m, dtype=float)
    ok = np.isfinite(r_i) & np.isfinite(r_m)
    n = len(r_i)
    if ok.sum() < 20:
        return None
    a, b = (1.0 - delta) * r_m, (1.0 + delta) * r_m
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    rw = np.clip(r_i, lo, hi)
    age_years = (n - 1 - np.arange(n)) / TRADING_DAYS
    w = np.exp(-lam * age_years)
    w, x, y = w[ok], r_m[ok], rw[ok]
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    vx = (w * (x - xm) ** 2).sum()
    if vx <= 0:
        return None
    return float((w * (x - xm) * (y - ym)).sum() / vx)


def ols_stats(r_i: np.ndarray, r_m: np.ndarray, jump_sigma: float = 4.0) -> dict | None:
    """Market-model statistics on aligned daily returns."""
    r_i = np.asarray(r_i, dtype=float)
    r_m = np.asarray(r_m, dtype=float)
    ok = np.isfinite(r_i) & np.isfinite(r_m)
    x, y = r_m[ok], r_i[ok]
    if len(x) < 20:
        return None
    vx = x.var()
    if vx <= 0:
        return None
    beta = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    sd_y = y.std()
    rho = float(np.corrcoef(x, y)[0, 1]) if sd_y > 0 else 0.0
    rsd = resid.std(ddof=2) if len(resid) > 2 else float("nan")
    jumps = int((np.abs(resid) > jump_sigma * rsd).sum()) if rsd > 0 else 0
    return {"beta": beta, "rho": rho, "r2": rho * rho,
            "ivol": float(rsd * math.sqrt(TRADING_DAYS)),
            "vol": float(sd_y * math.sqrt(TRADING_DAYS)),
            "jumps": jumps, "n": int(len(x))}


def vasicek(betas: dict[str, float], ses: dict[str, float]) -> dict[str, float]:
    """Shrink each OLS beta toward the cross-sectional mean:
    b* = (σ²_xs·b + se²·μ) / (σ²_xs + se²)."""
    vals = np.array([v for v in betas.values() if v is not None and math.isfinite(v)])
    if len(vals) < 3:
        return dict(betas)
    mu, var_xs = float(vals.mean()), float(vals.var())
    out = {}
    for t, b in betas.items():
        se2 = ses.get(t)
        if b is None or se2 is None or not math.isfinite(se2):
            out[t] = b
            continue
        out[t] = (var_xs * b + se2 * mu) / (var_xs + se2) if var_xs + se2 > 0 else b
    return out


def beta_se2(r_i: np.ndarray, r_m: np.ndarray) -> float | None:
    """Squared standard error of the OLS slope (for Vasicek)."""
    r_i = np.asarray(r_i, dtype=float)
    r_m = np.asarray(r_m, dtype=float)
    ok = np.isfinite(r_i) & np.isfinite(r_m)
    x, y = r_m[ok], r_i[ok]
    n = len(x)
    if n < 20:
        return None
    sxx = ((x - x.mean()) ** 2).sum()
    if sxx <= 0:
        return None
    b = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    resid = y - y.mean() - b * (x - x.mean())
    return float((resid ** 2).sum() / (n - 2) / sxx)


def effective_bets(weights: np.ndarray, cov: np.ndarray) -> float | None:
    """Effective number of uncorrelated bets: inverse Herfindahl of each
    principal component's share of portfolio variance (Meucci 2009).
    A basket of perfectly correlated names -> 1; N independent equal-vol
    names equally weighted -> N."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if w.size == 0 or cov.shape != (w.size, w.size):
        return None
    lam, vec = np.linalg.eigh(cov)
    lam = np.clip(lam, 0.0, None)
    contrib = (vec.T @ w) ** 2 * lam
    tot = contrib.sum()
    if tot <= 0:
        return None
    p = contrib / tot
    return float(1.0 / (p ** 2).sum())


def effective_dimension(corr: np.ndarray) -> float | None:
    """Weight-free count of independent bets: (Σλ)² / Σλ² over the eigenvalues
    of the names' correlation matrix (the participation ratio). N perfectly
    correlated names -> 1; N uncorrelated names -> N. Complements
    effective_bets, which for any long-only equity basket sits near 1 because
    the market factor carries almost all of the PORTFOLIO's variance."""
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or not np.isfinite(c).all():
        return None
    lam = np.clip(np.linalg.eigvalsh(c), 0.0, None)
    s2 = (lam ** 2).sum()
    return float(lam.sum() ** 2 / s2) if s2 > 0 else None


def pct_rank(values: list[float | None]) -> list[float | None]:
    """0-100 percentile ranks; None stays None (edge.common.pct_ranks semantics
    on the non-missing subset)."""
    from ..edge.common import pct_ranks
    idx = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    ranks = pct_ranks([values[i] for i in idx])
    out: list[float | None] = [None] * len(values)
    for i, r in zip(idx, ranks):
        out[i] = r
    return out

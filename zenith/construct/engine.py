"""CONSTRUCT — the maths behind good portfolio construction, as small testable functions.

Every function here backs one chapter of the guide or the sandbox:

- diversification: equal-weight portfolio vol vs N and ρ; Carver's diversification multiplier
- volatility drag: geometric ≈ arithmetic − σ²/2
- Shannon's demon / volatility pumping: closed-form growth of a constant-mix portfolio and
  path simulations; the rebalancing (diversification) return ½(Σwσ² − σ_p²)
- the add-to-portfolio test: SR_new > ρ·SR_port ⇔ μ_new > β·μ_port
- weighting: equal, inverse vol, equal risk contribution, minimum variance, maximum Sharpe
  (sample / Ledoit–Wolf shrunk / Michaud-resampled); risk contributions
- rebalancing: none / calendar / threshold bands, with costs and turnover
- leverage: Kelly growth curve and the odds of ever seeing a given drawdown
- crisis behaviour: correlations in the worst months vs all months
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------------------------------------ diversification
def ew_portfolio_vol(n: int | np.ndarray, sigma: float, rho: float) -> np.ndarray:
    """Vol of an equal-weight portfolio of n assets, each vol σ, all pairwise correlation ρ.
    σ_p² = σ²·(1/n + (1 − 1/n)·ρ) → floor of σ·√ρ as n grows."""
    n = np.asarray(n, float)
    return sigma * np.sqrt(1.0 / n + (1.0 - 1.0 / n) * rho)


def diversification_multiplier(w: np.ndarray, corr: np.ndarray, floor_negative: bool = True,
                               cap: float = 2.5) -> float:
    """Carver's instrument / forecast diversification multiplier 1/√(wᵀHw) (negative ρ floored at 0)."""
    w = np.asarray(w, float)
    H = np.array(corr, float)
    if floor_negative:
        H = np.clip(H, 0.0, 1.0)
        np.fill_diagonal(H, 1.0)
    v = float(w @ H @ w)
    return float(min(cap, 1.0 / np.sqrt(v))) if v > 1e-12 else 1.0


# -------------------------------------------------------------- vol drag
def geometric_from_arithmetic(mu: float | np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Compound (geometric) growth ≈ arithmetic mean − σ²/2 (log-normal approximation)."""
    return np.asarray(mu) - 0.5 * np.asarray(sigma) ** 2


# ------------------------------------------------------- Shannon's demon
def constant_mix_growth(w: float | np.ndarray, up: float, down: float, p: float = 0.5) -> np.ndarray:
    """Expected log growth per step of holding fraction w in a coin-flip asset (+up / −down with
    probability p / 1−p) and 1−w in cash, rebalanced every step."""
    w = np.asarray(w, float)
    return p * np.log1p(w * up) + (1 - p) * np.log1p(-w * down)


def kelly_fraction(up: float, down: float, p: float = 0.5) -> float:
    """Growth-optimal constant fraction for the coin-flip asset (closed form, clipped to [0, 1])."""
    f = (p * up - (1 - p) * down) / (up * down)
    return float(np.clip(f, 0.0, 1.0))


def shannon_paths(n_paths: int = 200, n_steps: int = 100, up: float = 1.0, down: float = 0.5, p: float = 0.5,
                  w: float = 0.5, every: int = 1, band: float = 0.0, cost: float = 0.0,
                  seed: int = 7) -> dict:
    """Simulate the asset alone vs a w / (1−w) asset/cash mix rebalanced every ``every`` steps
    (or only when the weight drifts more than ``band`` from target). Returns wealth paths."""
    rng = np.random.default_rng(seed)
    flips = rng.random((n_paths, n_steps)) < p
    r = np.where(flips, up, -down)
    asset = np.cumprod(1 + r, axis=1)
    mix = np.empty_like(asset)
    turnover = np.zeros(n_paths)
    for k in range(n_paths):
        a, c = w, 1.0 - w                                      # dollars in asset and cash
        for t in range(n_steps):
            a *= 1 + r[k, t]
            V = a + c
            wt = a / V if V > 0 else w
            due = (band > 0 and abs(wt - w) > band) or (band <= 0 and (t + 1) % every == 0)
            if due and V > 0:
                trade = abs(wt - w) * V
                turnover[k] += trade / V
                V -= trade * cost
                a, c = w * V, (1 - w) * V
            mix[k, t] = a + c
    return {"asset": asset, "mix": mix, "turnover": turnover / n_steps}


def rebalancing_premium(weights: np.ndarray, vols: np.ndarray, corr: np.ndarray) -> float:
    """Diversification return (Booth & Fama; Willenbrock): ½(Σ wᵢσᵢ² − σ_p²), per period of the vols."""
    w, s = np.asarray(weights, float), np.asarray(vols, float)
    cov = np.outer(s, s) * np.asarray(corr, float)
    return float(0.5 * (np.sum(w * s ** 2) - w @ cov @ w))


# ----------------------------------------------------- add-to-portfolio test
def add_test(sr_new: float, sr_port: float, rho: float) -> dict:
    """Adding a little of a new return stream raises the portfolio Sharpe iff SR_new > ρ·SR_port.
    The same condition in return terms: μ_new > β_new|port · μ_port (positive alpha vs the book)."""
    hurdle = rho * sr_port
    return {"hurdle": hurdle, "adds": bool(sr_new > hurdle), "margin": sr_new - hurdle}


def add_test_returns(new: pd.Series, port: pd.Series, periods: int = TRADING_DAYS) -> dict:
    df = pd.concat([new.rename("n"), port.rename("p")], axis=1).dropna()
    if len(df) < 30:
        return {"ok": False}
    a, b = df["n"], df["p"]
    sr = lambda x: float(x.mean() / x.std() * np.sqrt(periods)) if x.std() > 0 else 0.0  # noqa: E731
    rho = float(a.corr(b))
    beta = float(a.cov(b) / b.var()) if b.var() > 0 else float("nan")
    t = add_test(sr(a), sr(b), rho)
    mu_n, mu_p = float(a.mean() * periods), float(b.mean() * periods)
    return {"ok": True, "sr_new": sr(a), "sr_port": sr(b), "rho": rho, "beta": beta, "mu_new": mu_n,
            "mu_port": mu_p, "alpha": mu_n - beta * mu_p, **t,
            "adds_beta_form": bool(mu_n > beta * mu_p), "vol_new": float(a.std() * np.sqrt(periods)),
            "vol_port": float(b.std() * np.sqrt(periods))}


def optimal_mix(sr_new: float, sr_port: float, rho: float) -> float:
    """Risk share of the new stream that maximises the Sharpe of a two-stream mix (equal-vol streams)."""
    # max-Sharpe weights ∝ Σ⁻¹μ with unit vols: w ∝ [SR_p − ρSR_n, SR_n − ρSR_p]
    wn, wp = sr_new - rho * sr_port, sr_port - rho * sr_new
    if wn <= 0:
        return 0.0
    if wp <= 0:
        return 1.0
    return float(wn / (wn + wp))


# ---------------------------------------------------------------- weights
def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf (2004) shrinkage of the sample covariance towards a scaled identity."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.sum((S - F) ** 2) / p
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / (n ** 2) / p
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S


def _slsqp(obj, n: int, w0: np.ndarray | None = None, jac=None) -> np.ndarray:
    from scipy.optimize import minimize
    w0 = np.full(n, 1.0 / n) if w0 is None else w0
    res = minimize(obj, w0, jac=jac, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x if res.success else w0, 0, None)
    return w / w.sum()


def min_variance(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights."""
    cov = np.asarray(cov, float)
    c = cov / np.trace(cov) * cov.shape[0]                      # rescale: better conditioned, same argmin
    return _slsqp(lambda w: w @ c @ w, cov.shape[0], jac=lambda w: 2 * c @ w)


def max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only maximum-Sharpe weights (μᵀw / √(wᵀΣw))."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    if np.all(mu <= 0):
        return min_variance(cov)
    return _slsqp(lambda w: -(mu @ w) / np.sqrt(max(w @ cov @ w, 1e-18)), len(mu))


def risk_parity(cov: np.ndarray, iters: int = 500) -> np.ndarray:
    """Equal risk contribution (Maillard, Roncalli & Teïletche 2010), long-only, by fixed point."""
    n = cov.shape[0]
    w = 1.0 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        target = rc.sum() / n
        w = w * (target / np.maximum(rc, 1e-18)) ** 0.5
        w /= w.sum()
    return w


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    v = float(w @ cov @ w)
    return w * (cov @ w) / v if v > 0 else np.full(len(w), np.nan)


def resampled_max_sharpe(R: np.ndarray, n: int = 60, seed: int = 3) -> np.ndarray:
    """Michaud (1989): average the max-Sharpe weights over bootstrap resamples of the history."""
    rng = np.random.default_rng(seed)
    T = len(R)
    acc = np.zeros(R.shape[1])
    for _ in range(n):
        S = R[rng.integers(0, T, T)]
        acc += max_sharpe(S.mean(0), np.cov(S.T))
    return acc / n


SCHEMES = {
    "equal": "Equal weight (1/N) — fits nothing; hard to beat out of sample (DeMiguel et al. 2009)",
    "inverse_vol": "Inverse volatility — each asset gets the same standalone risk",
    "risk_parity": "Equal risk contribution — each asset adds the same share of portfolio risk",
    "min_variance": "Minimum variance — lowest-risk mix (long only)",
    "max_sharpe": "Maximum Sharpe, sample estimates — the textbook optimiser (error-maximising)",
    "max_sharpe_shrunk": "Maximum Sharpe, Ledoit–Wolf covariance and means shrunk 50% to the average",
    "max_sharpe_resampled": "Maximum Sharpe, Michaud-resampled (averaged over 60 bootstraps)",
}


def weights_for(scheme: str, R: pd.DataFrame) -> np.ndarray:
    X = R.dropna().to_numpy(float)
    n = X.shape[1]
    cov = np.cov(X.T) if n > 1 else np.array([[X.var()]])
    if scheme == "equal" or n == 1:
        return np.full(n, 1.0 / n)
    if scheme == "inverse_vol":
        iv = 1 / np.sqrt(np.diag(cov))
        return iv / iv.sum()
    if scheme == "risk_parity":
        return risk_parity(cov)
    if scheme == "min_variance":
        return min_variance(cov)
    if scheme == "max_sharpe":
        return max_sharpe(X.mean(0), cov)
    if scheme == "max_sharpe_shrunk":
        mu = X.mean(0)
        return max_sharpe(0.5 * mu + 0.5 * mu.mean(), ledoit_wolf(X))
    if scheme == "max_sharpe_resampled":
        return resampled_max_sharpe(X)
    raise KeyError(scheme)


# ----------------------------------------------------------- rebalancing
REBALANCE = ("none", "monthly", "quarterly", "annual", "band")


def run_mix(R: pd.DataFrame, w: np.ndarray, rule: str = "quarterly", band: float = 0.05,
            cost_bps: float = 5.0, vol_target: float = 0.0, vol_lookback: int = 63) -> dict:
    """Daily backtest of a fixed target mix with drift, a rebalance rule ('none', calendar, or
    'band' = only when any weight is more than ``band`` away from target), costs on turnover and
    an optional portfolio vol target (scaled with trailing vol known the day before, capped 3x)."""
    R = R.dropna()
    X = R.to_numpy(float)
    w_t = np.asarray(w, float)
    cur = w_t.copy()
    nxt = R.index[1:].to_period("M").append(pd.PeriodIndex([None], freq="M"))
    is_end = np.asarray(R.index.to_period("M") != nxt)              # last trading day of each month
    months = np.asarray(R.index.month)
    when = {"monthly": is_end, "quarterly": is_end & np.isin(months, [3, 6, 9, 12]),
            "annual": is_end & (months == 12)}.get(rule)
    out, turn, nreb = np.empty(len(R)), 0.0, 0
    for i in range(len(R)):
        rp = float(cur @ X[i])
        cur = cur * (1 + X[i]) / (1 + rp) if 1 + rp > 0 else w_t.copy()
        c = 0.0
        if rule == "band":
            due = bool(np.max(np.abs(cur - w_t)) > band)
        else:
            due = bool(when[i]) if when is not None else False
        if due:
            tr = float(np.abs(cur - w_t).sum())
            c = tr * cost_bps / 1e4
            turn += tr
            nreb += 1
            cur = w_t.copy()
        out[i] = rp - c
    r = pd.Series(out, index=R.index)
    if vol_target:
        vol = r.rolling(vol_lookback, min_periods=20).std().shift(1) * np.sqrt(TRADING_DAYS)
        r = r * (vol_target / vol).clip(upper=3.0).fillna(1.0)
    yrs = len(r) / TRADING_DAYS
    return {"returns": r, "turnover_per_year": turn / max(yrs, 1e-9), "rebalances": nreb}


def stats(r: pd.Series, periods: int = TRADING_DAYS) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0
    vol = float(r.std() * np.sqrt(periods))
    dd = float((eq / eq.cummax() - 1).min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0,
            "MaxDD": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "Arith": float(r.mean() * periods)}


def random_portfolios(mu: np.ndarray, cov: np.ndarray, n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Long-only Dirichlet portfolios for the frontier cloud (annualised inputs)."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.full(len(mu), 0.6), n)
    ret = W @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return pd.DataFrame({"ret": ret, "vol": vol, "sharpe": np.where(vol > 0, ret / vol, np.nan)})


# ---------------------------------------------------------------- leverage
def growth_vs_leverage(mu: float, sigma: float, rf: float = 0.0, L: np.ndarray | None = None) -> pd.DataFrame:
    """Expected log growth of L× leverage on an asset with arithmetic excess μ−rf and vol σ:
    g(L) = rf + L(μ−rf) − L²σ²/2. Peak at the Kelly leverage (μ−rf)/σ²."""
    L = np.linspace(0, 4, 161) if L is None else np.asarray(L, float)
    g = rf + L * (mu - rf) - 0.5 * L ** 2 * sigma ** 2
    return pd.DataFrame({"L": L, "growth": g, "vol": L * sigma})


def prob_ever_drawdown(g: float, vol: float, dd: float) -> float:
    """Chance a GBM with log-drift g and vol σ ever falls to (1−dd) of its STARTING value:
    (1−dd)^(2g/σ²) when g > 0, else 1. (A floor on the drawdown risk, which is from peaks.)"""
    if g <= 0 or vol <= 0:
        return 1.0
    return float((1 - dd) ** (2 * g / vol ** 2))


# ---------------------------------------------------------------- crisis
def crisis_table(R: pd.DataFrame, anchor: str, q: float = 0.10) -> pd.DataFrame:
    """Per asset: correlation to ``anchor`` in all months vs the anchor's worst-q months,
    and average monthly return in those months."""
    m = (1 + R.dropna(how="all")).resample("ME").prod() - 1
    m = m.dropna(subset=[anchor])
    worst = m[anchor] <= m[anchor].quantile(q)
    rows = []
    for c in m.columns:
        if c == anchor:
            continue
        s = m[[c, anchor]].dropna()
        w = s[worst.reindex(s.index).fillna(False)]
        rows.append({"asset": c, "corr_all": float(s[c].corr(s[anchor])),
                     "corr_crisis": float(w[c].corr(w[anchor])) if len(w) > 4 else np.nan,
                     "avg_in_crisis_%": float(w[c].mean() * 100) if len(w) else np.nan,
                     "hit_in_crisis_%": float((w[c] > 0).mean() * 100) if len(w) else np.nan,
                     "crisis_months": int(len(w))})
    return pd.DataFrame(rows).set_index("asset") if rows else pd.DataFrame()


def trend_sleeve(px: pd.DataFrame, lookback: int = 252, target_vol: float = 0.10) -> pd.Series:
    """A simple time-series momentum sleeve (Moskowitz, Ooi & Pedersen 2012): long assets above
    their 12-month level, short below, each vol-scaled, equal-weighted, rebalanced monthly."""
    r = px.pct_change()
    sig = np.sign(px / px.shift(lookback) - 1)
    vol = r.ewm(span=60, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    pos = (sig * (target_vol / vol)).clip(-3, 3)
    pos = pos.resample("ME").last().reindex(px.index, method="ffill").shift(1)
    return (pos * r).mean(axis=1, skipna=True).fillna(0.0)

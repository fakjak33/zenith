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
- drawdown budgets: crisis replay, historical max DD and option-implied tails as stress lenses;
  ETL / ETG from the smile (Breeden-Litzenberger) or from history; max-ETG/ETL weights fitted to a budget
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


# ======================================================= drawdown budgets & tails
# Chapters 09-10 and the sandbox DRAWDOWN BUDGET tab. Start from the drawdown the holder can live
# with, check the portfolio against it through three lenses (crisis replay, historical max
# drawdown, option-implied tail), then spend the budget where the upside tail is largest relative
# to the downside tail (ETG / ETL). Tails are measured against cash: option forwards already carry
# the cash rate, and the historical scenarios treat cash as a zero-return asset.

CRISES = {
    "2000-02 dot-com": ("2000-03-24", "2002-10-09"),
    "2008-09 GFC": ("2007-10-09", "2009-03-09"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}


def napkin_equity_weight(budget: float, eq_loss: float, def_ret: float = 0.0) -> float:
    """Equity weight whose crash loss just fills the budget, the rest in a defensive asset:
    -budget = w*(-eq_loss) + (1-w)*def_ret  =>  w = (budget + def_ret) / (eq_loss + def_ret).
    The 60/40 rule: 30% budget, 50% equity crash, flat bonds -> 60%."""
    den = eq_loss + def_ret
    if den <= 0:
        return 1.0
    return float(np.clip((budget + def_ret) / den, 0.0, 1.0))


def _ncdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def _nppf(p):
    from scipy.special import ndtri
    return ndtri(p)


def bs_delta_strike(sigma: float, T: float, delta: float, F: float = 1.0) -> float:
    """Strike with Black (forward) delta ``delta`` (negative for puts). Put: N(d1)-1 = delta;
    call: N(d1) = delta; d1 = (ln F/K + s^2/2)/s with s = sigma*sqrt(T)  =>  ln K/F = s^2/2 - d1*s."""
    d1 = float(_nppf(1 + delta if delta < 0 else delta))
    s = sigma * np.sqrt(T)
    return float(F * np.exp(0.5 * s * s - d1 * s))


def _black_call(F, K, sigma, T):
    s = np.maximum(sigma * np.sqrt(T), 1e-12)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return F * _ncdf(d1) - K * _ncdf(d1 - s)


def rn_density(ivs, deltas, T: float, n: int = 1601) -> tuple[np.ndarray, np.ndarray]:
    """Risk-neutral distribution of R = S_T/F - 1 from a few delta-bucket IVs (Breeden &
    Litzenberger 1978). Each bucket sits at its own strike; IV is interpolated linearly in
    log-moneyness and held flat beyond the outer buckets; call prices are differenced twice.
    Negative density from an uneven smile is clipped and the rest renormalised.
    Returns (R grid, probability per grid point)."""
    ivs, deltas = np.asarray(ivs, float), np.asarray(deltas, float)
    ok = np.isfinite(ivs) & (ivs > 0)
    ivs, deltas = ivs[ok], deltas[ok]
    if len(ivs) < 3:
        raise ValueError("need at least three finite delta-bucket IVs")
    k = np.array([0.0 if d == 0.5 else np.log(bs_delta_strike(v, T, d)) for v, d in zip(ivs, deltas)])
    o = np.argsort(k)
    k, ivs = k[o], ivs[o]
    width = 6 * float(ivs.max()) * np.sqrt(T) + 0.05
    kg = np.linspace(-width, width, n)
    K = np.exp(kg)
    C = _black_call(1.0, K, np.interp(kg, k, ivs), T)
    pdf = np.clip(np.gradient(np.gradient(C, K), K), 0, None)
    p = pdf * np.gradient(K)
    p = p / p.sum()
    m = float((K * p).sum())                                    # E[S_T/F] must be 1: undo clipping bias
    return K / m - 1.0, p


def tail_measures(x, p=None, q: float = 0.10) -> dict:
    """Expected tail loss / gain at level q, ETG/|ETL| and skewness of scenarios x with
    probabilities p (equal if None). ETL = mean of the worst q, ETG = mean of the best q."""
    x = np.asarray(x, float)
    p = np.full(len(x), 1.0) if p is None else np.asarray(p, float)
    ok = np.isfinite(x) & np.isfinite(p)
    x, p = x[ok], p[ok]
    if len(x) < 2 or p.sum() <= 0:
        return {"etl": np.nan, "etg": np.nan, "ratio": np.nan, "skew": np.nan, "mean": np.nan, "vol": np.nan}
    p = p / p.sum()
    o = np.argsort(x)
    x, p = x[o], p[o]

    def tail(xs, ps):
        take = np.clip(q - (np.cumsum(ps) - ps), 0, ps)        # partial weight on the boundary point
        return float((xs * take).sum() / take.sum())

    etl, etg = tail(x, p), tail(x[::-1], p[::-1])
    mu = float((x * p).sum())
    sd = float(np.sqrt(((x - mu) ** 2 * p).sum()))
    return {"etl": etl, "etg": etg, "ratio": etg / -etl if etl < 0 else np.nan,
            "skew": float(((x - mu) ** 3 * p).sum() / sd ** 3) if sd > 0 else 0.0, "mean": mu, "vol": sd}


def clean_smile(ivs, deltas, max_step: float = 1.35) -> tuple[list[float], int]:
    """Drop illiquid wing quotes. Walking out from ATM on each side, a bucket whose IV is more than
    ``max_step`` times the next bucket in is treated as a bad print, and so is everything beyond it
    (e.g. an EEM 10-delta call at 73% next to a 27% 15-delta). Returns (ivs, number dropped)."""
    ivs = [float(v) if v is not None and np.isfinite(v) and v > 0 else np.nan for v in ivs]
    deltas = [float(d) for d in deltas]
    dropped = 0
    for side in (-1, 1):
        idx = sorted([i for i, d in enumerate(deltas) if d != 0.5 and np.sign(d) == side],
                     key=lambda i: abs(deltas[i]), reverse=True)          # 35 -> 25 -> 15 -> 10 (inner to outer)
        inner = ivs[deltas.index(0.5)] if 0.5 in deltas else np.nan
        bad = False
        for i in idx:
            if bad or (np.isfinite(inner) and np.isfinite(ivs[i]) and ivs[i] > max_step * inner):
                bad = True
                dropped += int(np.isfinite(ivs[i]))
                ivs[i] = np.nan
            elif np.isfinite(ivs[i]):
                inner = ivs[i]
    return ivs, dropped


def implied_tails(ivs, deltas, T: float, q: float = 0.10) -> dict:
    """Option-implied tails at one tenor: where the 10-delta put and call strikes sit, plus the
    risk-neutral ETL / ETG / ratio / skew. Bad wing prints are dropped first (``clean_smile``).
    {} if the smile is unusable."""
    deltas = [float(d) for d in deltas]
    ivs, dropped = clean_smile(ivs, deltas)
    try:
        x, p = rn_density(ivs, deltas, T)
    except ValueError:
        return {}
    out = tail_measures(x, p, q)
    out["dropped"] = dropped
    for nm, d in (("put10", -0.10), ("call10", 0.10)):
        v = ivs[deltas.index(d)] if d in deltas else np.nan
        out[f"{nm}_iv"] = v
        out[f"{nm}_dist"] = bs_delta_strike(v, T, d) - 1 if np.isfinite(v) and v > 0 else np.nan
    return out


def horizon_returns(R: pd.DataFrame, horizon: int = 63, step: int = 5) -> pd.DataFrame:
    """Overlapping ``horizon``-day compound returns sampled every ``step`` days: a joint scenario
    set that keeps the real co-movement between assets."""
    lv = np.log1p(R.fillna(0.0)).cumsum()
    return np.expm1(lv.shift(-horizon) - lv).iloc[::step].dropna()


def crisis_returns(px: pd.DataFrame, windows: dict | None = None) -> pd.DataFrame:
    """Point-to-point return of each asset over each crisis window (equity peak -> trough); rows
    are crises, NaN where the window is outside the price history. A buy-and-hold portfolio's
    return over a window is exactly sum(w_i * r_i)."""
    windows = CRISES if windows is None else windows
    px = px.sort_index()
    rows = {}
    for nm, (a, b) in windows.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        row = {}
        for c in px.columns:
            s = px[c].dropna()
            ok = len(s) and s.index[0] <= a + pd.Timedelta(days=5) and s.index[-1] >= b
            row[c] = float(s.loc[:b].iloc[-1] / s.loc[:a].iloc[-1] - 1) if ok and len(s.loc[:a]) else np.nan
        rows[nm] = row
    return pd.DataFrame(rows).T.reindex(columns=px.columns)


def max_drawdown(r) -> float:
    eq = np.cumprod(1 + np.nan_to_num(np.asarray(r, float)))
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else np.nan


def stress_lenses(w, R: pd.DataFrame | None = None, crisis: pd.DataFrame | None = None,
                  implied_etl=None, crisis_sign=None, rule: str = "quarterly") -> dict:
    """Portfolio loss (as a positive number) under three lenses; cash is 1 - sum(w).

    crisis  - worst sum(w_i * r_i) across the crisis windows that every held asset covers
    history - max drawdown of the rebalanced mix (cash held at its weight) over the sample
    implied - option-implied ETL summed across assets (comonotonic, so an upper bound); an asset
              that GAINED on average in the crises gets no charge and no credit (conservative)
    worst / binding - the largest of the three: the budget has to survive all of them."""
    w = np.asarray(w, float)
    held = w > 1e-9
    out = {"crisis": np.nan, "crisis_name": "", "history": np.nan, "implied": np.nan}
    if crisis is not None and len(crisis):
        c = crisis.loc[:, held].dropna(how="any")
        if len(c):
            pl = c.to_numpy() @ w[held]
            i = int(np.argmin(pl))
            out["crisis"], out["crisis_name"] = float(max(-pl[i], 0.0)), str(c.index[i])
    if R is not None and len(R) > 20 and w.sum() > 0:
        out["history"] = float(-max_drawdown(run_mix(R, w / w.sum(), rule)["returns"].to_numpy() * w.sum())) \
            if R.shape[1] == len(w) else np.nan
    if implied_etl is not None:
        e = np.nan_to_num(np.asarray(implied_etl, float))
        if crisis_sign is not None:
            e = np.where(np.nan_to_num(np.asarray(crisis_sign, float)) > 0, 0.0, e)
        out["implied"] = float(-(w * np.minimum(e, 0)).sum())
    vals = {k: out[k] for k in ("crisis", "history", "implied") if np.isfinite(out[k])}
    out["binding"] = max(vals, key=vals.get) if vals else ""
    out["worst"] = max(vals.values()) if vals else np.nan
    return out


def headroom(budget: float, worst: float) -> dict:
    """Budget minus the binding stressed loss. Losses scale ~linearly with exposure, so the risk
    you could run is budget / worst times today's (>1 = room to add, <1 = take risk off)."""
    if not np.isfinite(worst) or worst <= 0:
        return {"headroom": np.nan, "scale": np.nan}
    return {"headroom": budget - worst, "scale": budget / worst}


def portfolio_tails(w, H: pd.DataFrame, q: float = 0.10) -> dict:
    """ETL / ETG / ratio / skew of the portfolio across joint horizon scenarios H."""
    return tail_measures(H.to_numpy() @ np.asarray(w, float), None, q)


def max_convexity(H: pd.DataFrame, q: float = 0.10, max_w: float = 1.0, w0=None) -> np.ndarray:
    """Long-only, fully invested weights maximising ETG/ETL across joint scenarios H. The ratio is
    scale-free (cash changes the size of the bet, not its shape), so the drawdown budget is
    applied afterwards by ``fit_to_budget``. Several starts guard against the ratio's flat spots."""
    from scipy.optimize import minimize
    X = H.to_numpy()
    n = X.shape[1]
    max_w = max(float(max_w), 1.0 / n + 1e-9)

    def neg(w):
        t = tail_measures(X @ w, None, q)
        return -t["ratio"] if np.isfinite(t["ratio"]) else 1e3 - t["etg"]

    starts = [np.full(n, 1 / n)] + ([np.asarray(w0, float)] if w0 is not None else [])
    starts += [np.minimum(np.eye(n)[i] * 0.5 + 0.5 / n, max_w) for i in range(n)]
    best, best_v = starts[0], neg(starts[0])
    for s in starts:
        s = s / s.sum()
        res = minimize(neg, s, method="SLSQP", bounds=[(0.0, max_w)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                       options={"maxiter": 200, "ftol": 1e-9, "eps": 1e-4})
        x = np.clip(res.x, 0, None)
        x = x / x.sum() if x.sum() > 0 else s
        if np.any(x > max_w + 1e-6):
            continue
        v = neg(x)
        if v < best_v:
            best, best_v = x, v
    return best


def fit_to_budget(w, budget: float, lens_fn, tol: float = 1e-4) -> tuple[float, dict]:
    """Largest exposure s in (0, 1] (the rest in cash) whose binding stressed loss <= budget.
    ``lens_fn(weights) -> stress_lenses dict``. Bisection: losses rise with exposure."""
    w = np.asarray(w, float) / np.sum(w)
    full = lens_fn(w)
    if not np.isfinite(full["worst"]) or full["worst"] <= budget:
        return 1.0, full
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lens_fn(w * mid)["worst"] <= budget else (lo, mid)
    return lo, lens_fn(w * lo)


def implied_tail_breach(px: pd.Series, vol: pd.Series, horizon: int = 63, q: float = 0.10) -> pd.DataFrame:
    """How often the realised move went past the market's own priced tail. Each day, the
    lognormal q-quantile of the next ``horizon`` trading days is set from an implied-vol index
    quoted in % (VIX3M for SPX); records it, the realised forward return and whether it broke
    through. VIX3M already sits above ATM vol (it includes the skew), so this is a mid-smile tail."""
    vol = vol.dropna() / 100.0
    fwd = px.dropna().shift(-horizon) / px.dropna() - 1
    df = pd.concat({"vol": vol, "fwd": fwd}, axis=1).dropna()
    s = df["vol"] * np.sqrt(horizon / TRADING_DAYS)
    df["tail"] = np.exp(float(_nppf(q)) * s - 0.5 * s * s) - 1
    df["breach"] = df["fwd"] < df["tail"]
    df["excess"] = (df["fwd"] - df["tail"]).where(df["breach"])
    return df


def skew_zscore(s: pd.Series, window: int = 3 * TRADING_DAYS, min_periods: int = TRADING_DAYS) -> pd.Series:
    """Where a skew measure sits against its own trailing window, in standard deviations."""
    s = s.dropna()
    roll = s.rolling(window, min_periods=min_periods)
    return (s - roll.mean()) / roll.std()

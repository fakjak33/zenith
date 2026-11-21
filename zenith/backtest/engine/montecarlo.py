"""Monte Carlo / bootstrap on the return stream.

Resamples daily returns (stationary block bootstrap) to produce a
*distribution* of Sharpe and max drawdown — so a single lucky path doesn't
masquerade as skill. Returns percentiles for the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def _block_bootstrap(returns: np.ndarray, rng: np.random.Generator,
                     block: int) -> np.ndarray:
    n = len(returns)
    out = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = min(block, n - i)
        for j in range(length):
            out[i + j] = returns[(start + j) % n]
        i += length
    return out


def bootstrap(
    returns: pd.Series,
    n_sims: int = 1000,
    block: int = 20,
    seed: int | None = 7,
    periods: int = 252,
) -> pd.DataFrame:
    """Return a DataFrame of bootstrapped [Sharpe, MaxDD, CAGR] samples."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 30:
        return pd.DataFrame(columns=["Sharpe", "MaxDD", "CAGR"])
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        sample = pd.Series(_block_bootstrap(r, rng, block))
        rows.append({
            "Sharpe": metrics.sharpe(sample, periods),
            "MaxDD": metrics.max_drawdown(sample),
            "CAGR": metrics.cagr(sample, periods),
        })
    return pd.DataFrame(rows)


def bootstrap_paths(
    returns: pd.Series,
    n_sims: int = 200,
    block: int = 20,
    capital: float = 1.0,
    seed: int | None = 7,
) -> pd.DataFrame:
    """Return a DataFrame of bootstrapped *equity paths* (columns = sims).

    Each column is a cumulative equity curve from a block-bootstrapped resample
    of the daily returns, starting at ``capital``. Used to draw the fan of
    possible outcomes behind the actual curve.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 30:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    paths = {}
    for s in range(n_sims):
        sample = _block_bootstrap(r, rng, block)
        paths[s] = capital * np.cumprod(1.0 + sample)
    return pd.DataFrame(paths)


def summarize(boot: pd.DataFrame) -> dict[str, dict[str, float]]:
    """5th/50th/95th percentiles per metric."""
    if boot.empty:
        return {}
    out = {}
    for col in boot.columns:
        out[col] = {
            "p5": float(boot[col].quantile(0.05)),
            "p50": float(boot[col].quantile(0.50)),
            "p95": float(boot[col].quantile(0.95)),
        }
    return out


# ------------------------------------------------------------ Phase 3 additions ----
def auto_block_length(x, c: float = 2.0) -> float:
    """Politis & White (2004, corrected by Patton, Politis & White 2009) optimal block length for the
    stationary bootstrap: long enough to keep the series' short-run dependence, no longer."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return 1.0
    x = x - x.mean()
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = np.ceil(min(3 * np.sqrt(n), n / 3))
    var = x @ x / n
    if var <= 0:
        return 1.0
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(mmax + 1)])
    rho = acov / var
    thr = c * np.sqrt(np.log10(n) / n)
    mhat = mmax
    for m in range(1, mmax - kn + 1):
        if np.all(np.abs(rho[m: m + kn]) < thr):
            mhat = m
            break
    M = min(2 * max(mhat, 1), mmax)
    ks = np.arange(-M, M + 1)
    t = np.abs(ks) / M
    lam = np.where(t <= 0.5, 1.0, np.where(t <= 1, 2 * (1 - t), 0.0))
    g = acov[np.abs(ks)]
    G = float((lam * np.abs(ks) * g).sum())
    sig = float((lam * g).sum())
    d_sb = 2 * sig ** 2
    if d_sb <= 0:
        return 1.0
    b = (2 * G ** 2 / d_sb) ** (1 / 3) * n ** (1 / 3)
    return float(np.clip(b, 1.0, bmax))


def trade_returns(returns: pd.Series, position: pd.Series) -> np.ndarray:
    """Compound return of each trade (a run of same-sign position)."""
    pos = np.sign(position.reindex(returns.index).fillna(0.0))
    held = pos.shift(1).fillna(0.0)                  # the return at t belongs to the position held into t
    grp = (held != held.shift()).cumsum()
    out = []
    for _, g in returns.groupby(grp):
        if held.loc[g.index[0]] != 0:
            out.append(float(np.prod(1 + g.to_numpy()) - 1))
    return np.array(out)


def trade_shuffle_drawdowns(trade_r: np.ndarray, n_sims: int = 1000, seed: int = 7) -> np.ndarray:
    """Max drawdown of the equity curve under random re-orderings of the same trades (SharpeEdge)."""
    if len(trade_r) < 5:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for i in range(n_sims):
        eq = np.cumprod(1 + rng.permutation(trade_r))
        out[i] = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return out


def challenge_sim(returns: pd.Series, target: float = 0.10, max_loss: float = 0.10, daily_limit: float = 0.05,
                  days: int = 60, leverage: float = 1.0, n_sims: int = 2000, block: float | None = None,
                  trailing: bool = False, seed: int = 7) -> dict:
    """Prop-firm style challenge on block-bootstrapped returns (SharpeEdge): pass when equity reaches
    +target before breaching -max_loss (static from start, or trailing from the high) or a one-day loss
    worse than daily_limit, within ``days``. Returns pass / fail / timeout probabilities and days-to-pass."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy() * leverage
    n = len(r)
    if n < 30:
        return {}
    block = block or auto_block_length(r / max(leverage, 1e-9))
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1.0)
    outcome = np.empty(n_sims, dtype=object)
    t_pass = np.full(n_sims, np.nan)
    final = np.empty(n_sims)
    for s in range(n_sims):
        eq, hi, i = 1.0, 1.0, int(rng.integers(n))
        res = "timeout"
        for d in range(days):
            if d and rng.random() < p:
                i = int(rng.integers(n))
            else:
                i = (i + 1) % n
            day = r[i]
            eq *= 1 + day
            hi = max(hi, eq)
            if day <= -daily_limit:
                res = "fail"
                break
            floor = (hi if trailing else 1.0) * (1 - max_loss)
            if eq <= floor:
                res = "fail"
                break
            if eq >= 1 + target:
                res = "pass"
                t_pass[s] = d + 1
                break
        outcome[s] = res
        final[s] = eq - 1
    return {"pass": float((outcome == "pass").mean()), "fail": float((outcome == "fail").mean()),
            "timeout": float((outcome == "timeout").mean()), "days_to_pass": t_pass[np.isfinite(t_pass)],
            "final": final, "block": block}

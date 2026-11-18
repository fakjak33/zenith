"""Performance metrics, including overfitting-aware Sharpe variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.std(ddof=1) < 1e-12 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd < 1e-12:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def cagr(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.empty:
        return 0.0
    total = (1.0 + r).prod()
    years = len(r) / periods
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + _clean(returns)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (fraction below the prior peak) over time."""
    eq = equity_curve(returns)
    if eq.empty:
        return eq
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else 0.0


def annual_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else 0.0


def _capture(returns: pd.Series, bench: pd.Series, upside: bool) -> float:
    """Up/down capture ratio vs a benchmark return series (in %)."""
    df = pd.concat([_clean(returns).rename("r"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return 0.0
    mask = df["b"] > 0 if upside else df["b"] < 0
    sub = df[mask]
    if sub.empty or abs(sub["b"].mean()) < 1e-12:
        return 0.0
    return float(sub["r"].mean() / sub["b"].mean() * 100)


def upside_capture(returns: pd.Series, bench: pd.Series) -> float:
    return _capture(returns, bench, upside=True)


def downside_capture(returns: pd.Series, bench: pd.Series) -> float:
    return _capture(returns, bench, upside=False)


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, periods) / mdd) if mdd > 1e-9 else 0.0


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    nonzero = r[r != 0]
    return float((nonzero > 0).mean()) if len(nonzero) else 0.0


def turnover(position: pd.Series) -> float:
    """Average daily absolute change in position (proxy for trading activity)."""
    return float(position.diff().abs().mean())


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               periods: int = TRADING_DAYS) -> float:
    """PSR: probability the true Sharpe exceeds ``benchmark_sr`` (Bailey & Lopez de Prado).

    Accounts for track-record length, skew, and kurtosis.
    """
    r = _clean(returns)
    n = len(r)
    if n < 8 or r.std(ddof=1) < 1e-12:
        return 0.0
    sr = sharpe(r, periods) / np.sqrt(periods)          # per-period SR
    sr_bench = benchmark_sr / np.sqrt(periods)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom < 1e-12:
        return 0.0
    z = (sr - sr_bench) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sr_std: float = 1.0, periods: int = TRADING_DAYS) -> float:
    """DSR: PSR against a benchmark inflated for ``n_trials`` independent tries.

    This is the headline overfitting metric for spectrum testing: testing many
    variants raises the bar a strategy must clear to be considered 'real'.
    """
    if n_trials <= 1:
        # one trial = no selection to deflate for (PARALLAX's formula hit ppf(0) = -inf here → DSR of 100%)
        return probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, periods=periods)
    euler = 0.5772156649
    # expected max Sharpe of n_trials random strategies (annualized)
    e_max = trial_sr_std * (
        (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(returns, benchmark_sr=e_max, periods=periods)


def skew(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(stats.skew(r)) if len(r) > 3 else 0.0


def tail_ratio(returns: pd.Series) -> float:
    """|95th pct| / |5th pct| of returns — >1 means fatter right tail."""
    r = _clean(returns)
    r = r[r != 0]
    if len(r) < 20:
        return 0.0
    lo = abs(np.percentile(r, 5))
    return float(abs(np.percentile(r, 95)) / lo) if lo > 1e-12 else 0.0


def ulcer_index(returns: pd.Series) -> float:
    """RMS of percentage drawdowns — penalizes depth *and* duration."""
    dd = drawdown_series(returns)
    return float(np.sqrt((dd.pow(2)).mean())) if not dd.empty else 0.0


def max_dd_duration(returns: pd.Series) -> int:
    """Longest run of bars spent below a prior equity peak."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    under = (dd < 0).astype(int).to_numpy()
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def profit_factor(returns: pd.Series) -> float:
    r = _clean(returns)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 1e-12 else 0.0


def summary(returns: pd.Series, position: pd.Series | None = None,
            n_trials: int = 1, periods: int = TRADING_DAYS) -> dict[str, float]:
    """One-stop metrics dict for the UI table, annualized with ``periods``."""
    out = {
        "CAGR": cagr(returns, periods),
        "Vol": annual_vol(returns, periods),
        "Sharpe": sharpe(returns, periods),
        "Sortino": sortino(returns, periods),
        "MaxDD": max_drawdown(returns),
        "MaxDD_bars": max_dd_duration(returns),
        "Calmar": calmar(returns, periods),
        "HitRate": hit_rate(returns),
        "ProfitFactor": profit_factor(returns),
        "Skew": skew(returns),
        "TailRatio": tail_ratio(returns),
        "Ulcer": ulcer_index(returns),
        "PSR": probabilistic_sharpe_ratio(returns, periods=periods),
        "DSR": deflated_sharpe_ratio(returns, n_trials=n_trials, periods=periods),
    }
    if position is not None:
        out["Turnover"] = turnover(position)
    return out

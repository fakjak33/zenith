"""Overfitting diagnostics: correlation matrices and walk-forward splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BacktestConfig, RiskConfig
from .engine import run_strategy
from . import metrics


def beta_and_correlation(strategy_returns: pd.Series, underlying_close: pd.Series) -> dict[str, float]:
    """Beta and correlation of a strategy's returns vs the underlying ETF.

    Beta = cov(strat, mkt) / var(mkt); correlation is Pearson r. Aligned on
    common dates with NaNs dropped.
    """
    mkt = underlying_close.pct_change()
    df = pd.concat([strategy_returns.rename("strat"), mkt.rename("mkt")], axis=1).dropna()
    if len(df) < 3 or df["mkt"].var() < 1e-18:
        return {"beta": 0.0, "correlation": 0.0}
    cov = df["strat"].cov(df["mkt"])
    beta = cov / df["mkt"].var()
    corr = df["strat"].corr(df["mkt"])
    return {"beta": float(beta), "correlation": float(corr)}


def return_correlation(results: dict) -> pd.DataFrame:
    """Correlation matrix of strategy *return* streams.

    Two 'different' rules with 0.95 return correlation are not diversifying —
    this exposes that directly. ``results`` maps label -> BacktestResult.
    """
    rets = pd.DataFrame({label: res.returns for label, res in results.items()})
    return rets.corr()


def forecast_correlation(forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation matrix of raw/scaled forecast streams."""
    return pd.DataFrame(forecasts).corr()


def walk_forward(
    strategy,
    ohlcv: pd.DataFrame,
    n_splits: int = 4,
    bt: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
) -> pd.DataFrame:
    """Sequential in-sample/out-of-sample Sharpe per split.

    Splits the history into ``n_splits`` contiguous OOS windows; for each, the
    'in-sample' is everything before it. Large IS>>OOS gaps flag overfitting.
    """
    idx = ohlcv.index
    bounds = np.linspace(0, len(idx), n_splits + 2, dtype=int)
    rows = []
    for k in range(1, len(bounds) - 1):
        is_end = bounds[k]
        oos_end = bounds[k + 1]
        is_data = ohlcv.iloc[:is_end]
        oos_data = ohlcv.iloc[is_end:oos_end]
        if len(is_data) < 60 or len(oos_data) < 20:
            continue
        is_res = run_strategy(strategy, is_data, bt=bt, risk=risk)
        oos_res = run_strategy(strategy, oos_data, bt=bt, risk=risk)
        rows.append({
            "split": k,
            "is_start": str(is_data.index[0].date()),
            "oos_period": f"{oos_data.index[0].date()}..{oos_data.index[-1].date()}",
            "IS_Sharpe": is_res.stats["Sharpe"],
            "OOS_Sharpe": oos_res.stats["Sharpe"],
        })
    return pd.DataFrame(rows)

"""Multi-asset rebalancing portfolio backtester (the ETF Lab engine).

Models a fund as a set of (signed) target weights that are reset to target on
each rebalance date and drift with prices in between. Uses the standard
*rebalanced-weights return* approach — portfolio return = Σ wᵢ·rᵢ — rather than
share/cash bookkeeping, which cleanly handles long/short, gross leverage,
transaction costs, expense-ratio drag, and short borrow costs.

No look-ahead: the weights earned on day *t* are the ones set at the prior
rebalance (targets become effective the bar after they're set).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS
from ..engine import metrics


@dataclass
class PortfolioConfig:
    capital: float = 1_000_000.0
    rebalance: str = "Monthly"          # Daily|Weekly|Monthly|Quarterly|Biannual|Annual
    cost_bps: float = 1.0               # commission on traded notional
    slippage_bps: float = 0.5           # half-spread on traded notional
    expense_ratio_annual: float = 0.0   # fund expense ratio (drag on NAV)
    borrow_bps: float = 50.0            # annual borrow cost charged on short notional
    max_weight: float = 1.0             # per-name cap (applied in the weight schedule)
    # direction is enforced upstream when building weights; kept for reference
    direction: str = "long"             # long | short | long_short


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame               # active (incoming) weights per day, dates x tickers
    turnover: pd.Series                 # one-way turnover on each rebalance date
    costs_paid: pd.Series               # cumulative transaction cost (fraction of NAV)
    expense_drag: pd.Series             # cumulative expense-ratio + borrow drag
    stats: dict = field(default_factory=dict)


_FREQ = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Annual": "Y"}


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    """First trading bar of each rebalance period within ``index``."""
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index
    if rebalance == "Daily":
        return index
    if rebalance == "Biannual":
        # first bar of each half-year (Jan & Jul)
        key = index.year * 2 + (index.month > 6).astype(int)
    else:
        freq = _FREQ.get(rebalance, "M")
        key = np.asarray(pd.PeriodIndex(index, freq=freq).asi8)
    changed = np.r_[True, key[1:] != key[:-1]]
    return index[changed]


def run_portfolio(panel: pd.DataFrame, weight_schedule: pd.DataFrame,
                  cfg: PortfolioConfig) -> PortfolioResult:
    """Backtest a portfolio.

    panel: wide close prices (dates x tickers).
    weight_schedule: signed target weights (rebalance-date rows x tickers); long
        positive, short negative. Rows define the rebalance dates.
    """
    panel = panel.sort_index()
    cols = list(panel.columns)
    idx = panel.index
    px = panel.to_numpy(dtype=float)
    rets = np.array(panel.pct_change(fill_method=None).to_numpy(dtype=float), copy=True)
    rets[~np.isfinite(rets)] = 0.0

    sched = weight_schedule.reindex(columns=cols).fillna(0.0)
    sched = sched[sched.index.isin(idx)]
    targets = {idx.get_loc(ts): row for ts, row in zip(sched.index, sched.to_numpy(dtype=float))}

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10_000.0
    er_daily = cfg.expense_ratio_annual / TRADING_DAYS
    borrow_daily = (cfg.borrow_bps / 10_000.0) / TRADING_DAYS

    n = len(cols)
    w = np.zeros(n)                     # incoming weights for the current day
    net_list, turn_list, cost_cum, drag_cum = [], [], [], []
    wsnap = np.empty((len(idx), n))
    cum_cost = 0.0
    cum_drag = 0.0

    for i in range(len(idx)):
        r = np.nan_to_num(rets[i])
        wsnap[i] = w                    # weights active during this day

        # 1) earn the day's return on incoming weights
        r_p = float(np.dot(w, r))
        short_gross = float(-w[w < 0].sum())

        # 2) drift weights to end of day
        denom = 1.0 + r_p
        if abs(denom) > 1e-12:
            w = w * (1.0 + r) / denom

        # 3) rebalance at end of day (effective next bar); charge cost today
        cost_today = 0.0
        turn_today = 0.0
        if i in targets:
            target = targets[i]
            valid = np.isfinite(px[i])          # only hold names with a price today
            target = np.where(valid, target, 0.0)
            turn_today = float(np.abs(target - w).sum())
            cost_today = turn_today * cost_rate
            w = target

        drag_today = er_daily + borrow_daily * short_gross
        cum_cost += cost_today
        cum_drag += drag_today
        net_list.append(r_p - cost_today - drag_today)
        turn_list.append(turn_today)
        cost_cum.append(cum_cost)
        drag_cum.append(cum_drag)

    returns = pd.Series(net_list, index=idx)
    equity = cfg.capital * (1.0 + returns).cumprod()
    weights = pd.DataFrame(wsnap, index=idx, columns=cols)
    turnover = pd.Series(turn_list, index=idx)
    stats = metrics.summary(returns)
    stats["FinalEquity"] = float(equity.iloc[-1]) if len(equity) else cfg.capital
    stats["TotalReturn%"] = (stats["FinalEquity"] / cfg.capital - 1.0) * 100
    stats["AnnVol%"] = metrics.annual_vol(returns) * 100
    stats["AvgTurnover"] = float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0
    stats["TotalCost%"] = cum_cost * 100
    return PortfolioResult(equity, returns, weights, turnover,
                           pd.Series(cost_cum, index=idx),
                           pd.Series(drag_cum, index=idx), stats)

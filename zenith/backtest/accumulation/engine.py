"""Accumulation backtester: trigger-based buying/selling vs DCA & buy-and-hold.

Model: you start with ``initial_cash`` and add ``contribution`` every cadence
period into a cash reserve. A *buy rule* deploys a fraction of available cash
when it fires; an optional *sell rule* moves a fraction of holdings back to cash.
Idle cash optionally earns ``cash_yield_annual``. Equity = shares·price + cash,
so every strategy is compared on the *same total money in* (dry powder counts).

Benchmarks (same contribution schedule):
- **DCA**: invest each contribution immediately every period.
- **Buy & hold (lump)**: invest the entire planned capital at t0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


@dataclass
class AccumConfig:
    initial_cash: float = 10_000.0
    contribution: float = 1_000.0
    cadence: str = "Weekly"            # Daily | Weekly | Monthly
    # deploy_mode: how much cash to spend each time the buy rule fires
    #   "pct_cash"      -> deploy_fraction of available cash
    #   "fixed_dollar"  -> deploy_dollar (capped at available cash)
    deploy_mode: str = "pct_cash"
    deploy_fraction: float = 1.0       # used when deploy_mode == "pct_cash"
    deploy_dollar: float = 5_000.0     # used when deploy_mode == "fixed_dollar"
    min_signal_gap: int = 0            # min bars between buys (0 = no limit)
    sell_fraction: float = 0.25        # fraction of shares sold per sell signal
    cash_yield_annual: float = 0.0     # optional yield on idle cash


@dataclass
class AccumResult:
    equity: pd.Series
    cash: pd.Series
    shares: pd.Series
    invested: pd.Series                # cumulative money contributed (dry powder incl.)
    deployed: pd.Series                # cumulative capital actually put to work (cost basis)
    profit: pd.Series                  # equity − contributed (flat at 0 until first deploy)
    stats: dict


def _contribution_mask(index: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    """Boolean mask: True on the first bar of each contribution period."""
    if cadence == "Daily":
        return np.ones(len(index), dtype=bool)
    freq = "W" if cadence == "Weekly" else "M"
    periods = pd.PeriodIndex(index, freq=freq)
    return np.r_[True, periods[1:] != periods[:-1]]


def run_accumulation(close: pd.Series, buy_rule, sell_rule, cfg: AccumConfig,
                     ctx: dict | None = None) -> AccumResult:
    ctx = ctx or {}
    close = close.dropna()
    idx = close.index
    px = close.to_numpy(dtype=float)
    contrib_mask = _contribution_mask(idx, cfg.cadence)

    buy_sig = buy_rule.signal(close, ctx).reindex(idx).fillna(False).to_numpy()
    buy_w = buy_rule.weight(close, ctx).reindex(idx).fillna(1.0).clip(0, 1).to_numpy()
    sell_sig = (sell_rule.signal(close, ctx).reindex(idx).fillna(False).to_numpy()
                if sell_rule is not None else np.zeros(len(idx), bool))

    daily_yield = (1 + cfg.cash_yield_annual) ** (1 / TRADING_DAYS) - 1
    cash = cfg.initial_cash
    shares = 0.0
    invested = cfg.initial_cash
    deployed = 0.0          # running cost basis of shares currently held
    last_buy = -10**9
    eq, csh, shs, inv, dep = [], [], [], [], []

    for i in range(len(idx)):
        if i > 0 and daily_yield:
            cash *= (1 + daily_yield)
        if contrib_mask[i] and i > 0:
            cash += cfg.contribution
            invested += cfg.contribution
        # sell first (realize before buy) — reduce cost basis proportionally
        if sell_sig[i] and shares > 0:
            frac = min(1.0, cfg.sell_fraction)
            sell_sh = shares * frac
            cash += sell_sh * px[i]
            shares -= sell_sh
            deployed *= (1 - frac)
        if buy_sig[i] and cash > 0 and (i - last_buy) >= cfg.min_signal_gap:
            if cfg.deploy_mode == "fixed_dollar":
                spend = min(cash, cfg.deploy_dollar * float(buy_w[i]))
            else:  # pct_cash
                spend = cash * cfg.deploy_fraction * float(buy_w[i])
            if spend > 0:
                shares += spend / px[i]
                cash -= spend
                deployed += spend
                last_buy = i
        eq.append(shares * px[i] + cash)
        csh.append(cash); shs.append(shares); inv.append(invested); dep.append(deployed)

    equity = pd.Series(eq, index=idx)
    invested_s = pd.Series(inv, index=idx)
    deployed_s = pd.Series(dep, index=idx)
    shares_s = pd.Series(shs, index=idx)
    holdings_s = shares_s * close                       # market value of the asset position
    profit_s = equity - invested_s                      # P/L vs money contributed
    stats = _accum_stats(equity, invested_s, close,
                         deployed=deployed_s, holdings=holdings_s)
    return AccumResult(equity, pd.Series(csh, index=idx), shares_s,
                       invested_s, deployed_s, profit_s, stats)


def benchmarks(close: pd.Series, cfg: AccumConfig) -> dict[str, pd.Series]:
    """DCA and lump buy-and-hold equity curves on the same contribution plan."""
    close = close.dropna()
    idx = close.index
    px = close.to_numpy(dtype=float)
    mask = _contribution_mask(idx, cfg.cadence)

    # DCA: invest each contribution immediately
    shares, eq = 0.0, []
    shares += cfg.initial_cash / px[0]
    for i in range(len(idx)):
        if mask[i] and i > 0:
            shares += cfg.contribution / px[i]
        eq.append(shares * px[i])
    dca = pd.Series(eq, index=idx)

    # Lump: deploy the entire planned capital at t0
    total = cfg.initial_cash + cfg.contribution * (mask.sum() - 1)
    lump = (close / px[0]) * total

    return {"DCA": dca, "Buy & Hold (lump)": lump}


def _accum_stats(equity: pd.Series, invested: pd.Series, close: pd.Series,
                 deployed: pd.Series | None = None,
                 holdings: pd.Series | None = None) -> dict:
    """Accumulation stats.

    ``invested`` = cumulative cash contributed (dry powder included).
    ``deployed`` = capital actually put to work (cost basis); defaults to
    ``invested`` for benchmarks that deploy everything. ``holdings`` = market
    value of the asset position (defaults to equity, i.e. fully invested).
    ReturnOnContributed% measures return on every dollar saved; ReturnOnDeployed%
    isolates the return on money actually at work (excludes dry-powder drag).
    """
    from ..engine import metrics
    ret = equity.pct_change().fillna(0.0)
    mkt = close.pct_change().reindex(ret.index).fillna(0.0)
    var = mkt.var()
    beta = float(ret.cov(mkt) / var) if var > 1e-18 else 0.0
    alpha_daily = ret.mean() - beta * mkt.mean()
    final_inv = float(invested.iloc[-1])
    final_eq = float(equity.iloc[-1])
    final_dep = float(deployed.iloc[-1]) if deployed is not None else final_inv
    final_hold = float(holdings.iloc[-1]) if holdings is not None else final_eq
    return {
        "FinalEquity": final_eq,
        "Contributed": final_inv,
        "Deployed": final_dep,
        "Profit": final_eq - final_inv,
        "ReturnOnContributed%": (final_eq / final_inv - 1) * 100 if final_inv else 0.0,
        "ReturnOnDeployed%": (final_hold / final_dep - 1) * 100 if final_dep > 1e-9 else 0.0,
        "AnnVol%": metrics.annual_vol(ret) * 100,
        "MaxDD%": metrics.max_drawdown(ret) * 100,
        "Beta": beta,
        "Alpha(ann)%": alpha_daily * TRADING_DAYS * 100,
        "Corr": float(ret.corr(mkt)) if var > 1e-18 else 0.0,
    }

"""Vectorized backtest engine.

Given a continuous position (in units) and prices, compute daily P&L net of
transaction costs and slippage. Works for a single instrument; portfolio
backtests sum instrument-level P&L with diversification handled upstream.

Trades execute at the next day's price (positions are lagged one day) to
avoid look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BacktestConfig, RiskConfig
from ..core.volatility import position_from_forecast, bars_per_year
from ..risk.overlays import apply_atr_overlay
from . import metrics


@dataclass
class BacktestResult:
    returns: pd.Series          # daily strategy returns (fraction of capital)
    equity: pd.Series           # equity curve (starts at capital)
    position: pd.Series         # position in units, post-overlay
    forecast: pd.Series         # scaled forecast that drove it
    events: pd.DataFrame        # ATR entry/stop/tp blotter
    stats: dict                 # metrics.summary output
    label: str = ""
    periods: int = 252          # bars/year used to annualize
    gross_returns: pd.Series | None = None
    costs: pd.Series | None = None


def buffer_and_round(position: pd.Series, width: pd.Series, multiplier: float | None) -> pd.Series:
    """Only trade when the target leaves a ±width band around the current holding (Carver's
    buffer), then hold whole contracts when ``multiplier`` is set. Causal: uses bar t only."""
    tgt = position.to_numpy(float)
    w = np.nan_to_num(width.reindex(position.index).to_numpy(float), nan=0.0)
    out = np.zeros_like(tgt)
    cur = 0.0
    m = multiplier or 0.0
    for i in range(len(tgt)):
        lo, hi = tgt[i] - w[i], tgt[i] + w[i]
        if m:
            if w[i] <= 0:
                cur = np.round(tgt[i] / m) * m
            elif cur < lo or cur > hi:
                lo_c, hi_c = np.ceil(lo / m) * m, np.floor(hi / m) * m
                # band narrower than one contract → nearest whole contract to the target
                cur = (lo_c if cur < lo else hi_c) if lo_c <= hi_c else np.round(tgt[i] / m) * m
        elif cur < lo:
            cur = lo
        elif cur > hi:
            cur = hi
        out[i] = cur
    return pd.Series(out, index=position.index)


def run_single(
    forecast: pd.Series,
    ohlcv: pd.DataFrame,
    bt: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
    n_trials: int = 1,
    label: str = "",
) -> BacktestResult:
    """Backtest one instrument given an already-scaled forecast."""
    bt = bt or BacktestConfig()
    risk = risk or RiskConfig()

    close = ohlcv["close"].reindex(forecast.index).ffill()
    periods = bt.trading_days or bars_per_year(close.index)

    # forecast -> position in units per the selected sizing mode
    position = position_from_forecast(
        forecast, close, capital=bt.capital,
        target_vol=bt.target_vol, vol_span=bt.vol_lookback,
        sizing_mode=bt.sizing_mode, fixed_pct=bt.fixed_pct,
        fixed_dollar=bt.fixed_dollar, periods=periods,
    )

    # direction filter: long-only / short-only
    if bt.direction == "long":
        position = position.clip(lower=0.0)
    elif bt.direction == "short":
        position = position.clip(upper=0.0)

    # optional execution delay (wait N bars before acting on a signal)
    if bt.use_delay and (bt.entry_delay or bt.exit_delay):
        from .execution import apply_delay
        position = apply_delay(position, bt.entry_delay, bt.exit_delay)

    # Carver buffering and whole-contract rounding (futures)
    if bt.buffer > 0 or bt.multiplier:
        f10 = position_from_forecast(forecast * 0 + 10.0, close, capital=bt.capital, target_vol=bt.target_vol,
                                     vol_span=bt.vol_lookback, sizing_mode=bt.sizing_mode, fixed_pct=bt.fixed_pct,
                                     fixed_dollar=bt.fixed_dollar, periods=periods).abs()
        position = buffer_and_round(position, f10 * bt.buffer, bt.multiplier)

    # ATR stop/take-profit overlay
    position, events = apply_atr_overlay(position, ohlcv.reindex(forecast.index), risk)

    # lag position by 1 day: today's signal trades tomorrow
    pos_lag = position.shift(1).fillna(0.0)

    price_ret = close.pct_change().fillna(0.0)
    # P&L in currency = units * price * return; normalize by capital -> fraction
    pnl_currency = pos_lag * close.shift(1).fillna(close.iloc[0]) * price_ret
    gross_ret = pnl_currency / bt.capital

    # costs: charge on traded notional (change in position)
    traded_units = position.diff().abs().fillna(0.0)
    traded_notional = traded_units * close
    cost_rate = (bt.cost_bps + bt.slippage_bps) / 10_000.0
    cost_ret = (traded_notional * cost_rate) / bt.capital
    if bt.multiplier and bt.fee_per_contract:
        cost_ret = cost_ret + (traded_units / bt.multiplier) * bt.fee_per_contract / bt.capital

    net_ret = (gross_ret - cost_ret).fillna(0.0)
    equity = bt.capital * (1.0 + net_ret).cumprod()
    stats = metrics.summary(net_ret, position=position, n_trials=n_trials, periods=periods)

    return BacktestResult(
        returns=net_ret, equity=equity, position=position,
        forecast=forecast, events=events, stats=stats, label=label,
        periods=periods, gross_returns=gross_ret, costs=cost_ret,
    )


def run_strategy(
    strategy,
    ohlcv: pd.DataFrame,
    bt: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
    n_trials: int = 1,
) -> BacktestResult:
    """Convenience: compute a single-instrument strategy's forecast then backtest."""
    forecast = strategy.forecast(ohlcv)
    return run_single(forecast, ohlcv, bt=bt, risk=risk,
                      n_trials=n_trials, label=strategy.describe())

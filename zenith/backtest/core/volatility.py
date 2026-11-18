"""Volatility estimation and volatility-targeted position sizing.

Carver sizes positions so each instrument contributes a target risk. We use
an exponentially-weighted per-bar return vol, annualize it with the bar
frequency actually present in the data, and derive units from the forecast.

GUARD RAILS (fixed vs PARALLAX):
- annualization is inferred from the index (252 daily / 52 weekly / 12
  monthly) instead of a fixed 256, so weekly and monthly runs are correct;
- the warm-up period is left empty instead of back-filled from future vol;
- vol-target sizing uses Carver's blended vol (70% recent / 30% long-run), not raw recent vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, VOL_LOOKBACK, FORECAST_SCALAR_TARGET


def bars_per_year(index: pd.Index) -> int:
    """Infer bars/year from the median spacing of a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return TRADING_DAYS
    days = float(pd.Series(index).diff().dt.days.median())
    if days <= 1.6:
        return TRADING_DAYS
    if days <= 8:
        return 52
    if days <= 35:
        return 12
    if days <= 100:
        return 4
    return 1


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple per-bar percentage returns."""
    return prices.pct_change()


def ew_vol(prices: pd.Series, span: int = VOL_LOOKBACK, annualize: bool = False,
           periods: int | None = None) -> pd.Series:
    """Exponentially-weighted per-bar return volatility (causal, floored)."""
    rets = daily_returns(prices)
    vol = rets.ewm(span=span, min_periods=max(2, span // 2)).std()
    vol = vol.clip(lower=1e-6)
    if annualize:
        vol = vol * np.sqrt(periods or bars_per_year(prices.index))
    return vol


def blended_vol(prices: pd.Series, span: int = VOL_LOOKBACK, periods: int | None = None,
                long_weight: float = 0.3) -> pd.Series:
    """Carver's sizing vol: 70% recent EW vol + 30% its own long-run average (up to ~10 years).

    Pure short-run vol invites extreme leverage in calm markets (ES in 2006: 6% vol → 7x).
    The long-run anchor is a trailing mean of past vol only, so it stays causal.
    """
    ew = ew_vol(prices, span=span, annualize=True, periods=periods)
    n = (periods or bars_per_year(prices.index)) * 10
    long_run = ew.rolling(n, min_periods=span).mean()
    return ((1 - long_weight) * ew + long_weight * long_run.fillna(ew))


def position_from_forecast(
    forecast: pd.Series,
    prices: pd.Series,
    capital: float,
    target_vol: float,
    vol_span: int = VOL_LOOKBACK,
    use_vol_target: bool = True,
    sizing_mode: str | None = None,
    fixed_pct: float = 1.0,
    fixed_dollar: float | None = None,
    periods: int | None = None,
) -> pd.Series:
    """Convert a capped forecast into a position in *units* (shares).

    - ``"vol_target"``: units = (f/10) * capital*target_vol / (σ_annual * price)
    - ``"fixed_pct"``: deploy ``fixed_pct`` of capital at f=10.
    - ``"fixed_dollar"``: deploy ``fixed_dollar`` notional at f=10.
    """
    if sizing_mode is None:
        sizing_mode = "vol_target" if use_vol_target else "fixed_pct"

    if sizing_mode == "vol_target":
        price_vol_annual = blended_vol(prices, span=vol_span, periods=periods)
        risk_budget = capital * target_vol
        units = (forecast / FORECAST_SCALAR_TARGET) * risk_budget / (price_vol_annual * prices)
    elif sizing_mode == "fixed_dollar":
        dollar = capital if fixed_dollar is None else fixed_dollar
        units = (forecast / FORECAST_SCALAR_TARGET) * dollar / prices
    else:  # fixed_pct
        units = (forecast / FORECAST_SCALAR_TARGET) * fixed_pct * capital / prices
    return units.replace([np.inf, -np.inf], np.nan).fillna(0.0)

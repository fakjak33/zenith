"""REGIMES performance-by-regime (spec sections 10, 11, 23): "what historically
worked when the world looked like this?" for both asset classes and factors.

IMPORTANT HONESTY CAVEAT, surfaced in the UI everywhere this module's output
appears: the DECLARED regime a month is tagged with uses the persistence
rule (2 consecutive readings) and, for months before that persistence
cleared, is filled forward from the prior declared regime — meaning this
tagging is NOT identical to "what a trader would have known to trade on
that exact day." It's t retrospective regime-conditional performance (the
standard way this kind of analysis is done across the industry: Fidelity's,
S&P's and most macro-cycle asset-allocation research all condition on the
SAME kind of after-the-fact-confirmed regime label), not a claim that the
regime was tradable in real time at the start of every month shown.

Asset-class and factor proxies are free-data ETFs (yfinance via the
existing cached `cas.sources.prices` fetcher) — the same universe class
CAS/MOMENTUM/IDEAS already draw on, no new data source.
"""

from __future__ import annotations

import pandas as pd

from ..cas.sources import prices as cas_prices
from . import REGIME_LABELS
from .macro import month_ends

ASSET_PROXIES = {
    "SPY": "US Large Cap Equities", "QQQ": "Nasdaq 100", "IWM": "US Small Cap",
    "EFA": "International Developed Equities", "EEM": "Emerging Markets",
    "TLT": "Long Treasuries (20y+)", "IEF": "7-10y Treasuries", "LQD": "IG Corporate Credit",
    "HYG": "High Yield Credit", "GLD": "Gold", "SLV": "Silver", "DBC": "Broad Commodities",
    "USO": "Crude Oil", "UUP": "US Dollar Index", "BTC-USD": "Bitcoin",
}

FACTOR_PROXIES = {
    "VLUE": "Value", "MTUM": "Momentum", "QUAL": "Quality", "USMV": "Low Volatility",
    "SIZE": "Small Size", "IWF": "Growth",
}

MIN_MONTHS_FOR_STATS = 4


def monthly_returns(tickers: list[str], ends: pd.DatetimeIndex | None = None) -> dict[str, pd.Series]:
    """ticker -> monthly return Series aligned to the same month-end grid the
    classifier uses, so it can be joined directly to declared_regime."""
    ends = ends if ends is not None else month_ends()
    px, status = cas_prices.get_history(list(tickers), period="max")
    out: dict[str, pd.Series] = {}
    for t in tickers:
        df = px.get(t)
        if df is None or df.empty:
            continue
        close = df["close"].dropna()
        close.index = pd.to_datetime(close.index)
        monthly = close.reindex(close.index.union(ends)).ffill().reindex(ends)
        out[t] = monthly.pct_change()
    return out


def _stats(returns: pd.Series) -> dict | None:
    r = returns.dropna()
    if len(r) < MIN_MONTHS_FOR_STATS:
        return None
    nav = (1.0 + r).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    vol_ann = float(r.std() * (12 ** 0.5))
    mean_ann = float(r.mean() * 12)
    sharpe = round(mean_ann / vol_ann, 3) if vol_ann > 0 else None
    return {
        "n_months": int(len(r)), "avg_return": round(float(r.mean()), 4),
        "median_return": round(float(r.median()), 4), "win_rate": round(float((r > 0).mean()), 4),
        "volatility_ann": round(vol_ann, 4), "max_drawdown": round(dd, 4),
        "sharpe_like": sharpe,
    }


def by_regime(returns: pd.Series, declared_regime: pd.Series) -> dict:
    """{regime_label: stats-or-None} for one asset/factor's return series."""
    aligned = declared_regime.reindex(returns.index)
    out = {}
    for regime in dict.fromkeys(REGIME_LABELS.values()):
        mask = aligned == regime
        out[regime] = _stats(returns[mask])
    return out


def build_performance_table(declared_regime: pd.Series, universe: str = "asset") -> dict:
    """Full table for every ticker in ASSET_PROXIES or FACTOR_PROXIES,
    regime-conditional. `universe`: 'asset' | 'factor'."""
    proxies = ASSET_PROXIES if universe == "asset" else FACTOR_PROXIES
    ends = declared_regime.index
    rets = monthly_returns(list(proxies), ends)
    out = {}
    for t, label in proxies.items():
        if t not in rets:
            continue
        out[t] = {"label": label, "by_regime": by_regime(rets[t], declared_regime)}
    return out

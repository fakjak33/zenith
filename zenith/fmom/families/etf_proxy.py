"""ETF-proxy factor family: monthly SPY-excess returns per factor ETF.

The tradable family — every "factor" is a real, liquid US ETF. Returns are
measured in excess of SPY so a proxy's market beta doesn't read as factor
momentum; the long leg is directly implementable (buy the ETFs), the short
leg doubles as an avoid/underweight list.
"""

from __future__ import annotations

import pandas as pd

from .. import core
from ..catalog import ETF_PROXIES, BENCHMARK


def tickers() -> list[str]:
    return [p["ticker"] for p in ETF_PROXIES] + [BENCHMARK]


def build_panel(px: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """px: ticker -> daily OHLCV frame (cas.sources.prices.get_history shape).
    Returns (monthly excess-return panel keyed by factor name, meta)."""
    bench = px.get(BENCHMARK)
    if bench is None or bench.empty:
        return pd.DataFrame(), {"ok": False, "error": f"no {BENCHMARK} history"}
    bench_m = core.monthly_returns(bench["close"])

    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    for p in ETF_PROXIES:
        df = px.get(p["ticker"])
        if df is None or df.empty:
            missing.append(p["ticker"])
            continue
        m = core.monthly_returns(df["close"])
        cols[p["factor"]] = (m - bench_m).dropna()
    panel = pd.DataFrame(cols).sort_index()
    # drop the still-forming current month: the last bar of a resample("ME") on
    # a mid-month pull is partial, and a 1-1 signal must form on a full month
    today = pd.Timestamp.today()
    panel = panel[panel.index < today.replace(day=1).normalize()]
    meta = {"ok": not panel.empty, "n_factors": panel.shape[1],
            "missing": missing, "benchmark": BENCHMARK,
            "last_month": panel.index[-1].strftime("%Y-%m") if len(panel) else None}
    return panel, meta

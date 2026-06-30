"""Themes segment — which sectors / industries / themes / asset-classes are
gaining or losing momentum, the kind of rotation that precedes rallies (semis,
gold, DRAM/memory, EM, etc.).

Combines two free inputs:
  * Theme/sector ETF relative strength vs SPY (price-based, always available).
  * Finviz group performance (sector or industry) for breadth confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal
from ..universe import asset_class_of, label_of, SECTORS, THEMES


def _relative_strength(close: pd.Series, bench: pd.Series, lb: int = 63) -> float:
    if len(close) < lb + 2 or len(bench) < lb + 2:
        return 0.0
    rs = (close.iloc[-1] / close.iloc[-lb]) / (bench.iloc[-1] / bench.iloc[-lb]) - 1.0
    return ind.clip1(np.tanh(5 * rs))


def _trend_vs_sma(close: pd.Series, d: int = 200) -> float:
    m = ind.sma(close, d)
    if pd.isna(m.iloc[-1]) or m.iloc[-1] == 0:
        return 0.0
    return ind.clip1((close.iloc[-1] / m.iloc[-1] - 1.0) * 6)


def compute(data: dict[str, pd.DataFrame], finviz_groups: list[dict] | None = None) -> list[dict]:
    out: list[dict] = []
    bench = data.get("SPY", {}).get("close") if "SPY" in data else None

    for t in {**SECTORS, **THEMES}:
        df = data.get(t)
        if df is None or bench is None:
            continue
        close = df["close"]
        rs1 = _relative_strength(close, bench, 21)
        rs3 = _relative_strength(close, bench, 63)
        rs6 = _relative_strength(close, bench, 126)
        blend = ind.clip1(0.6 * rs3 + 0.4 * rs6)
        ac, lbl = asset_class_of(t), label_of(t)
        out.append(make_signal(
            t, "themes", "relative_strength", blend, asset_class=ac, horizon="months",
            source="yfinance", rationale=f"{lbl} RS vs SPY: 3m {rs3:+.2f}, 6m {rs6:+.2f}"))
        out.append(make_signal(
            t, "themes", "rs_short", rs1, asset_class=ac, horizon="weeks",
            source="yfinance", rationale=f"{lbl} 1-month relative strength vs SPY {rs1:+.2f}"))
        out.append(make_signal(
            t, "themes", "rs_long", rs6, asset_class=ac, horizon="months",
            source="yfinance", rationale=f"{lbl} 6-month relative strength vs SPY {rs6:+.2f}"))
        out.append(make_signal(
            t, "themes", "trend_vs_200dma", _trend_vs_sma(close, 200), asset_class=ac,
            horizon="months", source="yfinance",
            rationale=f"{lbl} price vs its 200-day average (absolute trend)"))

    # Finviz breadth confirmation at the sector level (best-effort)
    if finviz_groups:
        for g in finviz_groups:
            name = str(g.get("name", ""))
            perf_m = g.get("perf_month") or g.get("performance_month") or 0
            try:
                pm = float(str(perf_m).replace("%", "")) / 100.0
            except ValueError:
                continue
            out.append(make_signal(
                name, "themes", "finviz_group_momentum", ind.clip1(np.tanh(8 * pm)),
                asset_class="sector", horizon="weeks", source="finviz", confidence="low",
                rationale=f"Finviz {name} 1m perf {pm:+.1%}"))
    return out

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


def compute(data: dict[str, pd.DataFrame], finviz_groups: list[dict] | None = None) -> list[dict]:
    out: list[dict] = []
    bench = data.get("SPY", {}).get("close") if "SPY" in data else None

    for t in {**SECTORS, **THEMES}:
        df = data.get(t)
        if df is None or bench is None:
            continue
        rs3 = _relative_strength(df["close"], bench, 63)
        rs6 = _relative_strength(df["close"], bench, 126)
        sig = ind.clip1(0.6 * rs3 + 0.4 * rs6)
        out.append(make_signal(
            t, "themes", "relative_strength", sig, asset_class=asset_class_of(t),
            horizon="months", source="yfinance",
            rationale=f"{label_of(t)} RS vs SPY: 3m {rs3:+.2f}, 6m {rs6:+.2f}"))

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

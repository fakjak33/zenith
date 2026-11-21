"""STRESS: how the strategy behaves in named crises and under synthetic shocks.

Crisis replay reuses PORTFOLIO › CONSTRUCT's crisis windows (plus a few more);
shocks rebuild the price path so the strategy re-trades it causally:
  vol ×2     daily log returns stretched around their mean (same path shape, twice the swings)
  gap −10%   three overnight −10% gaps at evenly spaced dates
  costs ×3   commission and slippage tripled
  trend off  the path's drift removed (a market with no trend to catch)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...construct.engine import CRISES, crisis_returns, max_drawdown, tail_measures
from ..spec import run_spec

EXTRA_CRISES = {
    "1987 crash": ("1987-10-02", "1987-10-19"),
    "1998 LTCM": ("1998-07-17", "1998-10-08"),
    "2011 euro crisis": ("2011-04-29", "2011-10-03"),
    "2015-16 China / oil": ("2015-07-20", "2016-02-11"),
    "2018 Q4 sell-off": ("2018-09-20", "2018-12-24"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}
WINDOWS = {**EXTRA_CRISES, **CRISES}


def _rebuild(df: pd.DataFrame, lr: pd.Series) -> pd.DataFrame:
    """New OHLC with close following log returns ``lr``; intrabar shape kept relative to close."""
    c0 = df["close"].iloc[0]
    close = c0 * np.exp(lr.fillna(0.0).cumsum())
    ratio = close / df["close"]
    out = df.copy()
    for k in ("open", "high", "low", "close"):
        if k in out:
            out[k] = df[k] * ratio
    return out


def shocked(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    lr = np.log(df["close"]).diff()
    if kind == "vol x2":
        mu = lr.mean()
        return _rebuild(df, mu + 2 * (lr - mu))
    if kind == "gap -10%":
        lr = lr.copy()
        for f in (0.3, 0.55, 0.8):
            lr.iloc[int(len(lr) * f)] += np.log(0.9)
        return _rebuild(df, lr)
    if kind == "trend off":
        return _rebuild(df, lr - lr.mean())
    return df


SCENARIOS = ("base", "costs x3", "vol x2", "gap -10%", "trend off")


def scenario_table(spec, ins: pd.DataFrame, aux: dict | None = None) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        df = shocked(ins, sc)
        res = run_spec(spec, df, aux, cost_mult=3.0 if sc == "costs x3" else 1.0)
        s = res.stats
        tm = tail_measures(res.returns.to_numpy(), q=0.05)
        rows.append({"scenario": sc, "Sharpe": s["Sharpe"], "CAGR": s["CAGR"], "MaxDD": s["MaxDD"],
                     "worst 5% bar": tm["etl"], "best 5% bar": tm["etg"]})
    return pd.DataFrame(rows)


def crisis_table(strategy_r: pd.Series, bench_r: pd.Series) -> pd.DataFrame:
    px = pd.DataFrame({"strategy": (1 + strategy_r.fillna(0)).cumprod(),
                       "buy & hold": (1 + bench_r.fillna(0)).cumprod()})
    t = crisis_returns(px, WINDOWS).dropna(how="all")
    if t.empty:
        return t
    dd = {}
    for nm, (a, b) in WINDOWS.items():
        if nm in t.index:
            seg = strategy_r.loc[a:b]
            dd[nm] = max_drawdown(seg.to_numpy()) if len(seg) > 1 else np.nan
    t["strategy max DD in window"] = pd.Series(dd)
    t["start"] = [WINDOWS[n][0] for n in t.index]
    return t.sort_values("start")

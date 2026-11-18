"""Benchmarks: pick the right yardstick for the strategy's goal.

A strategy is only 'good' relative to the cheapest thing that does the same
job. Long-only equity timing must beat simply holding the asset; an absolute-
return strategy must beat cash; a crisis-alpha strategy is judged by what it
does when equities fall; a carry/income strategy by a balanced portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BENCHMARKS = {
    "bh": "Buy & hold (same instrument)",
    "SPY": "S&P 500 (SPY, total return)",
    "6040": "60/40 (SPY / IEF, monthly rebalance)",
    "cash": "Cash (3-month T-bills)",
}
EQUITY_LIKE = {"SPY", "QQQ", "IWM", "EFA", "EEM", "^GSPC", "ES=F", "NQ=F", "DIA", "VTI"}


@dataclass(frozen=True)
class Suggestion:
    key: str
    why: str


def suggest(goal: str, name: str, direction: str, src: str) -> list[Suggestion]:
    """Ordered benchmark suggestions (first = primary) for a goal and instrument."""
    eq = name.upper() in EQUITY_LIKE
    out: list[Suggestion] = []
    if goal.startswith("Beat buy"):
        out.append(Suggestion("bh", f"you said the goal is to beat holding {name}; that is the cheapest rival"))
    elif goal.startswith("Crisis"):
        out.append(Suggestion("SPY", "crisis alpha is judged by what it does when equities fall: watch the "
                                     "down-capture and convexity"))
    elif goal.startswith("Income"):
        out.append(Suggestion("6040", "an income/carry book competes with a plain balanced portfolio"))
    else:   # absolute return
        out.append(Suggestion("cash", "an absolute-return strategy must first beat risk-free cash"))
    if direction == "long" and "bh" not in [s.key for s in out]:
        out.append(Suggestion("bh", f"long-only timing of {name} must beat just holding it"))
    if eq and "SPY" not in [s.key for s in out]:
        out.append(Suggestion("SPY", "the strategy trades equities, so check it isn't just market beta"))
    if not eq and src == "Market" and "bh" not in [s.key for s in out]:
        out.append(Suggestion("bh", f"holding {name} is the passive version of this exposure"))
    for k in BENCHMARKS:
        if k not in [s.key for s in out]:
            out.append(Suggestion(k, ""))
    return out


def cash_index() -> pd.Series:
    """Daily T-bill cash index from FRED DTB3 (shares the STRATEGY INDICES cache)."""
    try:
        from ...cas.sources.fred import get_series
        out, _ = get_series(["DTB3"], limit=None, cache_key="stratidx_fred")
        pts = out.get("DTB3") or []
        if not pts:
            return pd.Series(dtype=float)
        y = pd.Series([p["value"] / 100 for p in pts], index=pd.to_datetime([p["date"] for p in pts]))
        idx = pd.bdate_range(y.index[0], pd.Timestamp.today())
        y = y.reindex(idx).ffill()
        return (1 + y.shift(1).fillna(0) / 252).cumprod()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def to_returns(level: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Per-bar returns of a price/index level, sampled on the strategy's bars."""
    if level is None or level.empty:
        return pd.Series(0.0, index=index)
    lv = level.sort_index()
    lv = lv[~lv.index.duplicated()]
    s = lv.reindex(lv.index.union(index)).ffill().reindex(index)
    return s.pct_change().fillna(0.0)


def sixty_forty(spy: pd.Series, ief: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """60/40 returns on the strategy's bars, rebalanced to target at each month end."""
    a, b = to_returns(spy, index), to_returns(ief, index)
    out = []
    we = 0.6
    month = None
    for t in index:
        if month is not None and t.month != month:
            we = 0.6
        month = t.month
        ra, rb = a.loc[t], b.loc[t]
        r = we * ra + (1 - we) * rb
        out.append(r)
        g = 1 + r
        we = we * (1 + ra) / g if g else 0.6
    return pd.Series(out, index=index)

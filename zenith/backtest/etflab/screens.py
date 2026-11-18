"""Declarative ETF spec + point-in-time selection / weight-schedule builder.

An ``ETFSpec`` describes how to build a fund: a universe, an optional ranking
factor (top/bottom N) or an explicit basket, optional filters, a weighting
scheme, a rebalance cadence, and a direction (long / short / long_short).
``build_weight_schedule`` turns that into signed target weights on each
rebalance date — recomputing selection from data available *as of* that date.

Factors split into two classes. **Class A** (price/returns-derived) is genuinely
point-in-time and backtest-valid. **Class B** (the fundamental snapshot — P/E,
FCF/share, sector, etc.) is *today's* value held constant across history, so any
screen using it carries look-ahead/survivorship bias (``spec_is_lookahead``
flags this so the UI can warn). Filters and class-B ranking are the Phase 2
additions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import factors, weighting, fundamentals
from ..data.universe import UNIVERSES
from ..data.stocks500 import STOCKS_500

# Class-A (price-derived) rankers: name -> callable(panel, asof, mkt, lb, sk).
_PRICE_RANK = {
    "momentum": lambda p, a, m, lb, sk: factors.momentum(p, a, lb, sk),
    "trailing_return": lambda p, a, m, lb, sk: factors.trailing_return(p, a, lb, sk),
    "calendar_return": lambda p, a, m, lb, sk: factors.calendar_return(
        p, (a.year - 1) if a is not None else int(p.index.year.max())),
    "low_volatility": lambda p, a, m, lb, sk: factors.low_volatility(p, a, lb),
    "volatility": lambda p, a, m, lb, sk: factors.volatility(p, a, lb),
    "beta": lambda p, a, m, lb, sk: factors.beta(p, a, m, lb),
    "low_beta": lambda p, a, m, lb, sk: factors.low_beta(p, a, m, lb),
    "max_drawdown": lambda p, a, m, lb, sk: factors.max_drawdown(p, a, lb),
    "dividend_growth": lambda p, a, m, lb, sk: factors.dividend_growth(list(p.columns)),
}

# Class-B (fundamental snapshot) rankers: name -> (snapshot field, sign).
# A negative sign turns a "low X" tilt (value) into a high-is-best score.
_FUND_NUM = {
    "pe": ("trailing_pe", 1.0),
    "low_pe": ("trailing_pe", -1.0),
    "forward_pe": ("forward_pe", 1.0),
    "low_forward_pe": ("forward_pe", -1.0),
    "fcf_per_share": ("fcf_per_share", 1.0),
    "revenue": ("revenue", 1.0),
    "profit_margin": ("profit_margin", 1.0),
    "low_profit_margin": ("profit_margin", -1.0),
    "price_to_book": ("price_to_book", 1.0),
    "low_price_to_book": ("price_to_book", -1.0),
    "dividend_yield": ("dividend_yield", 1.0),
    "market_cap_value": ("market_cap", 1.0),
}
# Class-B categorical fields usable in filters.
_FUND_STR = {"sector": "sector", "industry": "industry"}

# Everything that touches the fundamental snapshot (drives the look-ahead warning).
CLASS_B_FACTORS = set(_FUND_NUM) | set(_FUND_STR)


@dataclass
class FilterRule:
    factor: str
    op: str            # "<=" | ">=" | "<" | ">" | "==" | "!=" | "in" | "not in"
    value: object


@dataclass
class SelectionSpec:
    universe: object = "Sector ETFs"   # named key or explicit list
    explicit: list | None = None       # explicit basket (overrides ranking)
    rank_factor: str | None = None
    short_rank_factor: str | None = None  # separate ranking for the short leg
    rank_lookback: int = 252
    rank_skip: int = 0
    top_n: int | None = None           # longs from the top of the ranking
    bottom_n: int | None = None        # shorts from the bottom of the ranking
    filters: list = field(default_factory=list)


@dataclass
class ETFSpec:
    name: str
    selection: SelectionSpec
    weighting: str = "equal"
    rebalance: str = "Monthly"
    direction: str = "long"            # long | short | long_short
    manual_weights: dict | None = None
    max_weight: float = 1.0
    notes: str = ""
    phase: int = 1
    enabled: bool = True


def resolve_universe(u) -> list[str]:
    if isinstance(u, (list, tuple, set)):
        return list(u)
    if u in ("Stocks (500)", "IWB (large-cap proxy)", "IWB"):
        return list(STOCKS_500)
    return list(UNIVERSES.get(u, []))


def spec_is_lookahead(spec: ETFSpec) -> bool:
    """True if the spec relies on the fundamental snapshot (look-ahead/survivorship)."""
    if spec.weighting in weighting.SNAPSHOT_METHODS:
        return True
    sel = spec.selection
    if sel.rank_factor in CLASS_B_FACTORS or sel.short_rank_factor in CLASS_B_FACTORS:
        return True
    return any(f.factor in CLASS_B_FACTORS for f in (sel.filters or []))


def _valid_at(panel: pd.DataFrame, asof, tickers) -> list[str]:
    sub = panel.loc[:asof]
    if sub.empty:
        return []
    last = sub.iloc[-1]
    return [t for t in tickers if t in panel.columns and np.isfinite(last.get(t, np.nan))]


# --- filtering --------------------------------------------------------------

def _match_num(val, op: str, thr) -> bool:
    if val is None or not np.isfinite(val):
        return False          # can't verify → exclude (conservative)
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        return False
    return {
        "<=": val <= thr, "<": val < thr, ">=": val >= thr, ">": val > thr,
        "==": val == thr, "!=": val != thr,
    }.get(op, False)


def _match_str(val: str, op: str, target) -> bool:
    v = str(val or "").strip().lower()
    if not v:
        return False
    if op in ("in", "not in"):
        opts = [str(x).strip().lower() for x in (target if isinstance(target, (list, tuple, set)) else [target])]
        hit = v in opts
        return hit if op == "in" else not hit
    t = str(target).strip().lower()
    if op == "==":
        return v == t
    if op == "!=":
        return v != t
    if op == "contains":
        return t in v
    return False


def _numeric_values(factor: str, uni, panel, asof, market_close) -> pd.Series:
    if factor in _FUND_NUM:
        field, _ = _FUND_NUM[factor]
        return fundamentals.factor_series(uni, field).reindex([t.upper() for t in uni]).set_axis(uni)
    fn = _PRICE_RANK.get(factor)
    if fn is not None:
        return fn(panel[uni], asof, market_close, 252, 0).reindex(uni)
    return pd.Series(np.nan, index=uni)


def _apply_filters(uni, panel, asof, market_close, filters) -> list[str]:
    keep = list(uni)
    for f in filters or []:
        if not keep:
            break
        if f.factor in _FUND_STR:
            vals = fundamentals.classify(keep, _FUND_STR[f.factor])
            vals = vals.reindex([t.upper() for t in keep]).set_axis(keep)
            keep = [t for t in keep if _match_str(vals.get(t, ""), f.op, f.value)]
        else:
            vals = _numeric_values(f.factor, keep, panel, asof, market_close)
            keep = [t for t in keep if _match_num(vals.get(t, np.nan), f.op, f.value)]
    return keep


# --- ranking ----------------------------------------------------------------

def _score(factor: str, panel, asof, market_close, lb, sk) -> pd.Series:
    fn = _PRICE_RANK.get(factor)
    if fn is not None:
        return fn(panel, asof, market_close, lb, sk)
    if factor in _FUND_NUM:
        field, sign = _FUND_NUM[factor]
        s = fundamentals.factor_series(list(panel.columns), field)
        s = s.reindex([t.upper() for t in panel.columns]).set_axis(list(panel.columns))
        return sign * s
    return pd.Series(np.nan, index=panel.columns)


def _ranked(factor, panel, uni, asof, market_close, sel):
    if not factor:
        return None
    s = _score(factor, panel[uni], asof, market_close, sel.rank_lookback, sel.rank_skip)
    s = s.reindex(uni).dropna().sort_values(ascending=False)
    return s if not s.empty else None


def selected_at(spec: ETFSpec, panel: pd.DataFrame, asof=None,
                market_close: pd.Series | None = None):
    """Return (longs, shorts) chosen from data available up to ``asof``."""
    sel = spec.selection
    if sel.explicit:
        return _valid_at(panel, asof, sel.explicit), []

    uni = [t for t in resolve_universe(sel.universe) if t in panel.columns]
    uni = _valid_at(panel, asof, uni)
    uni = _apply_filters(uni, panel, asof, market_close, sel.filters)
    if not uni:
        return [], []

    # No ranking → the (filtered) universe is the long basket.
    if not sel.rank_factor and not sel.short_rank_factor:
        return uni, []

    longs, shorts = [], []
    if spec.direction in ("long", "long_short"):
        scores = _ranked(sel.rank_factor, panel, uni, asof, market_close, sel)
        if scores is not None:
            longs = list(scores.index[:sel.top_n]) if sel.top_n else list(scores.index)
        elif sel.rank_factor is None:
            longs = uni
    if spec.direction in ("short", "long_short"):
        sf = sel.short_rank_factor or sel.rank_factor
        sscores = _ranked(sf, panel, uni, asof, market_close, sel)
        if sscores is not None:
            n = sel.bottom_n or sel.top_n
            shorts = list(sscores.index[-n:]) if n else []
        shorts = [t for t in shorts if t not in longs]   # don't long & short the same name
    return longs, shorts


def build_weight_schedule(spec: ETFSpec, panel: pd.DataFrame, rebal_dts,
                          market_close: pd.Series | None = None) -> pd.DataFrame:
    """Signed target weights per rebalance date (longs sum +1, shorts sum -1)."""
    rows = {}
    for asof in rebal_dts:
        longs, shorts = selected_at(spec, panel, asof, market_close)
        w = {}
        if longs:
            lw = weighting.compute(spec.weighting, longs, panel, asof,
                                   spec.manual_weights, spec.max_weight)
            for t, x in lw.items():
                w[t] = w.get(t, 0.0) + x
        if shorts:
            sw = weighting.compute(spec.weighting, shorts, panel, asof,
                                   None, spec.max_weight)
            for t, x in sw.items():
                w[t] = w.get(t, 0.0) - x
        rows[asof] = w
    if not rows:
        return pd.DataFrame(columns=panel.columns)
    return pd.DataFrame(rows).T.reindex(columns=panel.columns).fillna(0.0)

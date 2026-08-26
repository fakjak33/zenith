"""REGIMES historical analog engine (spec section 33 / 22): "have we seen
this before?"

The analog VECTOR is this package's own 8 dimension composites (growth,
inflation, monetary, liquidity, credit, financial_conditions, dollar,
volatility) at each historical month — genuinely richer than a price-only
vector, and every dimension in it is something this package already
computes for other purposes, so there is no new data pull here.

Distance is a coverage-aware Euclidean distance in z-space (missing
dimensions in either month are simply excluded from that pair's distance,
never imputed as zero — imputing zero would silently claim "average" for a
dimension with no data, which is exactly the fabrication spec section 29
prohibits). A minimum-overlap requirement excludes pairs that would
otherwise match on 1-2 dimensions by coincidence.

Forward outcomes are shown as a DISTRIBUTION (median / IQR / win rate /
best / worst, always with `n`), never a point forecast — spec section 33:
"do not treat historical analogs as predictions."
"""

from __future__ import annotations

import pandas as pd

from ..cas.sources import prices as cas_prices
from . import DIMENSIONS
from .macro import month_ends

MIN_SHARED_DIMENSIONS = 5
EXCLUDE_RECENT_MONTHS = 6      # a month can't be its own analog, nor a trivial neighbor
DEFAULT_TOP_K = 5
FORWARD_HORIZONS_MONTHS = (3, 6, 12)


def _dimension_composites(z_df: pd.DataFrame) -> pd.DataFrame:
    """month x dimension matrix (one composite column per dimension, mean of
    that dimension's available series that month) — the analog vector."""
    from .series import BY_DIMENSION
    cols = {}
    for dim in DIMENSIONS:
        ids = [s.id for s in BY_DIMENSION.get(dim, ()) if s.id in z_df.columns]
        cols[dim] = z_df[ids].mean(axis=1, skipna=True) if ids else pd.Series(index=z_df.index, dtype=float)
    return pd.DataFrame(cols)


def _distance(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    diff = (a - b).dropna()
    n = len(diff)
    if n < MIN_SHARED_DIMENSIONS:
        return None, n
    return float((diff ** 2).sum() ** 0.5) / (n ** 0.5), n   # RMS distance, scale-comparable across n


def nearest_analogs(z_df: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Nearest historical months to the LATEST month, by dimension-composite
    distance, excluding the trailing EXCLUDE_RECENT_MONTHS (trivially
    similar to "today")."""
    dims = _dimension_composites(z_df)
    if len(dims) < EXCLUDE_RECENT_MONTHS + 10:
        return []
    today_vec = dims.iloc[-1]
    candidates = dims.iloc[:-EXCLUDE_RECENT_MONTHS]
    scored = []
    for month, row in candidates.iterrows():
        dist, n_shared = _distance(today_vec, row)
        if dist is None:
            continue
        scored.append({"month": month, "distance": dist, "n_shared_dimensions": n_shared})
    scored.sort(key=lambda r: r["distance"])
    return scored[:top_k]


def forward_outcomes(analogs: list[dict], spy_returns: pd.Series) -> list[dict]:
    """For each analog month, the ACTUAL forward SPY return at each horizon —
    context, not a forecast."""
    out = []
    idx = list(spy_returns.index)
    for a in analogs:
        row = dict(a)
        row["month"] = a["month"].date().isoformat()
        row["forward"] = {}
        if a["month"] not in idx:
            out.append(row)
            continue
        pos = idx.index(a["month"])
        for h in FORWARD_HORIZONS_MONTHS:
            if pos + h < len(spy_returns):
                window = spy_returns.iloc[pos + 1:pos + h + 1]
                row["forward"][str(h)] = None if window.isna().any() else round(float((1 + window).prod() - 1), 4)
            else:
                row["forward"][str(h)] = None
        out.append(row)
    return out


def outcome_distribution(analog_rows: list[dict], horizon: int) -> dict:
    """Median/IQR/win-rate/best/worst across the analog set's forward
    returns at one horizon — the "have we seen this before" answer, always
    with n so a 2-analog distribution can't masquerade as a robust read."""
    vals = [r["forward"].get(str(horizon)) for r in analog_rows if r["forward"].get(str(horizon)) is not None]
    if not vals:
        return {"n": 0}
    s = pd.Series(vals, dtype=float)
    return {"n": int(len(s)), "median": round(float(s.median()), 4),
           "q25": round(float(s.quantile(0.25)), 4), "q75": round(float(s.quantile(0.75)), 4),
           "win_rate": round(float((s > 0).mean()), 4), "best": round(float(s.max()), 4),
           "worst": round(float(s.min()), 4)}


def build(z_df: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> dict:
    analogs = nearest_analogs(z_df, top_k=top_k)
    if not analogs:
        return {"analogs": [], "distributions": {}, "note": "Insufficient history for analog search yet."}
    ends = month_ends()
    px, _status = cas_prices.get_history(["SPY"], period="max")
    spy = px.get("SPY")
    if spy is None or spy.empty:
        return {"analogs": [{"month": a["month"].date().isoformat(), "distance": round(a["distance"], 3),
                            "n_shared_dimensions": a["n_shared_dimensions"], "forward": {}} for a in analogs],
               "distributions": {}, "note": "SPY price data unavailable this run."}
    close = spy["close"].dropna()
    close.index = pd.to_datetime(close.index)
    monthly_close = close.reindex(close.index.union(ends)).ffill().reindex(ends)
    spy_returns = monthly_close.pct_change()

    rows = forward_outcomes(analogs, spy_returns)
    for r in rows:
        r["distance"] = round(r["distance"], 3)
    dists = {str(h): outcome_distribution(rows, h) for h in FORWARD_HORIZONS_MONTHS}
    return {"analogs": rows, "distributions": dists,
           "note": f"{len(rows)} nearest historical months by 8-dimension macro similarity; "
                   f"forward SPY returns are historical outcomes, NOT a forecast."}

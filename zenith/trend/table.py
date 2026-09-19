"""Pure screening logic for the All Trends table: frame building, filters,
and every sort the view offers. No Streamlit here, so it is unit-tested
directly (tests/test_trend.py) rather than only through a rendered page."""

from __future__ import annotations

import pandas as pd

from . import SPEED_KEYS, SPEED_SHORT

# label -> (column, descending-by-default). Any column missing from a frame
# (e.g. the Momentum join when that artefact is absent) falls back to score.
SORTS: dict[str, tuple[str, bool]] = {
    "Trend Score": ("score", True),
    **{f"EWMAC {SPEED_SHORT[k]}": (f"f_{k}", True) for k in SPEED_KEYS},
    "Recent change (5d)": ("d5", True),
    "Change (20d)": ("d20", True),
    "Recent trigger": ("last_event_date", True),
    "Latest crossover": ("latest_cross_date", True),
    "Acceleration (fast − slow)": ("slope", True),
    "Speeds bullish": ("n_bull", True),
    "Momentum score": ("momentum", True),
    "Trend − Momentum": ("trend_minus_mom", True),
    "Ticker": ("ticker", False),
    "Name": ("name", False),
    "Sector / category": ("group", False),
}


def frame(rows: list[dict], group_col: str = "sector") -> pd.DataFrame:
    """Scored rows -> a flat DataFrame with one f_<speed> column per speed and
    the derived sort keys the table needs."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fc = df["forecasts"].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [None] * 7)
    for j, k in enumerate(SPEED_KEYS):
        df[f"f_{k}"] = fc.apply(lambda v, j=j: v[j] if j < len(v) else None).astype(float)
    if "last_event_date" not in df:
        df["last_event_date"] = df.get("last_event", pd.Series([None] * len(df))).apply(
            lambda e: e.get("date") if isinstance(e, dict) else None)
    df["group"] = df[group_col] if group_col in df else ""
    return df


def sort(df: pd.DataFrame, key: str, ascending: bool | None = None) -> pd.DataFrame:
    col, desc = SORTS.get(key, ("score", True))
    if col not in df.columns:
        col, desc = "score", True
    asc = (not desc) if ascending is None else ascending
    return df.sort_values([col, "score"], ascending=[asc, False], na_position="last", kind="mergesort")


def filter_rows(df: pd.DataFrame, query: str = "", groups: list | None = None,
                group_col: str = "sector", structures: list | None = None,
                score_range: tuple[float, float] | None = None, hide_partial: bool = False,
                min_bull: int | None = None, hide_cash_like: bool = False) -> pd.DataFrame:
    out = df
    if query and query.strip():
        q = query.strip().lower()
        out = out[out["ticker"].str.lower().str.contains(q, regex=False)
                  | out["name"].fillna("").str.lower().str.contains(q, regex=False)]
    if groups:
        out = out[out[group_col].isin(groups)]
    if structures:
        out = out[out["structure"].isin(structures)]
    if score_range is not None:
        lo, hi = score_range
        out = out[(out["score"] >= lo) & (out["score"] <= hi)]
    if hide_cash_like and "cash_like" in out:
        out = out[~out["cash_like"].fillna(False).astype(bool)]
    if hide_partial and "partial" in out:
        out = out[~out["partial"].fillna(False).astype(bool)]
    if min_bull is not None:
        # "n speeds agree with the score's direction"
        agree = out.apply(lambda r: r["n_bull"] if r["score"] >= 0 else 7 - r["n_bull"], axis=1) \
            if len(out) else pd.Series(dtype=float)
        out = out[agree >= min_bull] if len(out) else out
    return out

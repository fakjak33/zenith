"""AQR-inspired factor family: the Gupta & Kelly 65 via published returns.

Every factor in catalog.AQR65 with a live source becomes a monthly long-short
return series pulled from the Ken French Data Library (univariate-sort
Hi30−Lo30 spreads and published factor series) or AQR's public monthly
workbooks (QMJ, BAB, HML-Devil). Factors without a free monthly-updating
source stay listed for honest coverage but carry no live returns — the OSAP
deep-history panel (backtest-only, annual updates) covers a much wider set.

Unlike the ETF/Man families these series are ALREADY long-short factor
returns, so no benchmark subtraction is applied. Ken French publishes with a
~1-2 month lag; the panel's last month can trail the ETF families and each
signal row carries its own source month.
"""

from __future__ import annotations

import pandas as pd

from ..catalog import AQR65, AQR_SUPPLEMENTAL
from ...cas.backtest import factor_data


def live_rows() -> list[dict]:
    return [r for r in AQR65 + AQR_SUPPLEMENTAL if r["live_source"]]


def coverage() -> dict:
    """Honest coverage stats for the UI badge."""
    live = sum(1 for r in AQR65 if r["live_source"])
    repl = sum(1 for r in AQR65 if r["yf_char"])
    return {"defined": len(AQR65), "live": live, "replicable": repl,
            "supplemental": len(AQR_SUPPLEMENTAL)}


def _load_series(row: dict) -> pd.Series | None:
    src = row["live_source"]
    if src == "sort":
        return factor_data.load_french_sorts(row["ds"], row["long_col"],
                                             row["short_col"])
    if src == "factor":
        return factor_data.load_french_factor(row["ds"], row["col"])
    if src == "aqr":
        return factor_data.load_aqr_series(row["aqr_key"])
    return None


def build_panel(_px=None) -> tuple[pd.DataFrame, dict]:
    """Monthly long-short return panel keyed by factor abbreviation. The px
    argument is ignored (kept for a uniform family-builder signature)."""
    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    for row in live_rows():
        s = _load_series(row)
        if s is None or s.empty:
            missing.append(row["abbr"])
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
        cols[row["abbr"]] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()
    if not panel.empty:
        today = pd.Timestamp.today()
        panel = panel[panel.index < today.replace(day=1).normalize()]
    meta = {"ok": not panel.empty, "n_factors": panel.shape[1],
            "missing": missing, "benchmark": "long-short (none)",
            "coverage": coverage(),
            "last_month": panel.index[-1].strftime("%Y-%m") if len(panel) else None}
    return panel, meta


def build_osap_panel() -> tuple[pd.DataFrame, dict]:
    """Deep-history OSAP long-short panel (backtest-only)."""
    series = factor_data.load_osap()
    if not series:
        return pd.DataFrame(), {"ok": False, "n_factors": 0,
                                "error": "OSAP data unavailable (optional)"}
    panel = pd.DataFrame(series).sort_index()
    panel.index = pd.to_datetime(panel.index) + pd.offsets.MonthEnd(0)
    # drop predictors that are all-NaN or absurd
    panel = panel.loc[:, panel.notna().sum() >= 60]
    meta = {"ok": not panel.empty, "n_factors": panel.shape[1],
            "benchmark": "long-short (none)", "backtest_only": True,
            "last_month": panel.index[-1].strftime("%Y-%m") if len(panel) else None}
    return panel, meta

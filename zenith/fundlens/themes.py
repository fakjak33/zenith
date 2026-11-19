"""Theme scan: find themes / tilts / groups having an outlier year that may be skewing the data.

For every grouping dimension (strategic-beta group, index selection factors, index weighting,
objective, style box, momentum tag, ...) each group with 3+ funds is compared with the whole
comparison universe:

  1Y excess     = group median 1-year return - universe median 1-year return
  Prior excess  = same, using ex-latest-year 3Y returns (the two years before the latest 12 months)
  Median skew   = group median Recency Skew (1Y peer percentile - ex-latest-year percentile)

Flags (thresholds in profile `recency`):
  Hot year          1Y excess >= theme_gap and median skew >= theme_skew   -> this year is unusual for the group
  Persistent leader 1Y excess >= theme_gap but prior excess >= theme_gap/2 -> strong now AND before (not an outlier)
  Cold year         1Y excess <= -theme_gap and median skew <= -theme_skew
Purely diagnostic: it never changes scores. Fund-level recency control is the `recency.mode` setting.
"""
from __future__ import annotations

import math

import pandas as pd

from .lenses import NOT_CLASSIFIED, group_values
from .quality import QualityLog
from .scoring import ScoreResult

DEFAULT_DIMENSIONS = [
    {"field": "sb_group", "label": "Strategic Beta Group"},
    {"field": "index_selection", "label": "Index Selection factor", "explode": True},
    {"field": "index_weighting", "label": "Index Weighting", "explode": True},
    {"field": "objective", "label": "Objective"},
    {"field": "equity_style_box", "label": "Style Box"},
    {"field": "strategic_beta", "label": "Strategic Beta (Y/N)"},
    {"field": "momentum_tilt", "label": "Momentum strategy (name / selection)"},
    {"field": "sector_tilt", "label": "Sector overweight (5+ pts vs benchmark)", "explode": True},
    {"field": "sector_under", "label": "Sector underweight (5+ pts vs benchmark)", "explode": True},
    {"field": "category", "label": "Morningstar Category"},
]
SKIP = {NOT_CLASSIFIED, "Not applicable", "No"}


def theme_scan(df: pd.DataFrame, s: ScoreResult, rank_off: pd.Series, prof: dict, codes: dict,
               log: QualityLog) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rc = prof["recency"]
    gap, skew_thr = float(rc["theme_gap"]), float(rc["theme_skew"])
    dims = rc.get("theme_dimensions") or DEFAULT_DIMENSIONS
    r1 = pd.to_numeric(df.get("nav_ret_12m"), errors="coerce") if "nav_ret_12m" in df else None
    if r1 is None or r1.notna().sum() < 5:
        return pd.DataFrame(), {}
    ytd = pd.to_numeric(df["nav_ret_ytd"], errors="coerce") if "nav_ret_ytd" in df else None
    xr = pd.to_numeric(df["nav_ret_xr_3y"], errors="coerce") if "nav_ret_xr_3y" in df else None
    sk = pd.to_numeric(df["recency_skew"], errors="coerce") if "recency_skew" in df else None
    n_rank = int(s.rank.notna().sum())
    top_cut = max(1, math.ceil(n_rank * 0.2))
    u1, uy = r1.median(), (ytd.median() if ytd is not None else None)
    ux = xr.median() if xr is not None else None

    rows, fund_hot = [], {}
    for dim in dims:
        f = dim["field"]
        if f not in df:
            continue
        members: dict[str, list[str]] = {}
        for t in df.index:
            for g in group_values(df.at[t, f], f, bool(dim.get("explode")), codes, {}):
                members.setdefault(g, []).append(t)
        for g, tick in members.items():
            if g in SKIP or len(tick) < 3 or len(tick) >= len(df):
                continue
            m1 = r1[tick].median()
            if pd.isna(m1):
                continue
            ex1 = m1 - u1
            exx = (xr[tick].median() - ux) if xr is not None and xr[tick].notna().sum() >= 2 else None
            msk = sk[tick].median() if sk is not None and sk[tick].notna().any() else None
            ranked = [t for t in tick if pd.notna(s.rank[t])]
            top_now = sum(1 for t in ranked if s.rank[t] <= top_cut)
            top_off = sum(1 for t in ranked if pd.notna(rank_off.get(t)) and rank_off[t] <= top_cut)
            chg = [rank_off[t] - s.rank[t] for t in ranked if pd.notna(rank_off.get(t))]
            flag = ""
            if ex1 >= gap and exx is not None and exx >= gap / 2:
                flag = "Persistent leader"
            elif ex1 >= gap and msk is not None and msk >= skew_thr:
                flag = "Hot year"
            elif ex1 <= -gap and msk is not None and msk <= -skew_thr:
                flag = "Cold year"
            rows.append({"dimension": dim["label"], "group": g, "n": len(tick), "median_1y": m1, "excess_1y": ex1,
                         "excess_ytd": (ytd[tick].median() - uy) if ytd is not None and uy is not None else None,
                         "excess_prior": exx, "median_skew": msk,
                         "top_quintile_now": top_now, "top_quintile_off": top_off,
                         "avg_rank_change": (sum(chg) / len(chg)) if chg else None,
                         "flag": flag, "tickers": ", ".join(sorted(tick))})
            if flag == "Hot year":
                for t in tick:
                    fund_hot.setdefault(t, []).append(g if dim["field"] != "momentum_tilt" else "Momentum")
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, {}
    order = {"Hot year": 0, "Cold year": 1, "Persistent leader": 2, "": 3}
    tab = tab.sort_values(["flag", "excess_1y"], key=lambda c: c.map(order) if c.name == "flag" else -c).reset_index(drop=True)

    for _, r in tab[tab["flag"] == "Hot year"].head(6).iterrows():
        log.add("warning", "Themes",
                f"Hot year - {r['dimension']} '{r['group']}' ({r['n']} funds): median 1Y return {r['excess_1y']:+.1f} pts vs "
                f"the comparison, median recency skew {r['median_skew']:+.0f} pts"
                + (f", prior 2 years {r['excess_prior']:+.1f} pts" if r["excess_prior"] is not None and not pd.isna(r["excess_prior"]) else "")
                + f"; top-quintile funds {r['top_quintile_off']} with recency off -> {r['top_quintile_now']} now. "
                  "Latest-year returns for this theme may be skewing rankings.")
    for _, r in tab[tab["flag"] == "Cold year"].head(3).iterrows():
        log.add("info", "Themes", f"Cold year - {r['dimension']} '{r['group']}' ({r['n']} funds): median 1Y return "
                                  f"{r['excess_1y']:+.1f} pts vs the comparison; latest year may understate it.")
    fund_hot = {t: sorted(set(v)) for t, v in fund_hot.items()}
    return tab, fund_hot

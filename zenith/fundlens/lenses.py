"""Lenses: rank funds within groups of any categorical field (objective, strategic-beta group, ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .quality import QualityLog
from .scoring import ScoreResult

NOT_CLASSIFIED = "Not classified"


@dataclass
class Lens:
    field: str
    label: str
    table: pd.DataFrame          # columns: group, group_n, within_rank, ticker (one row per fund-group)
    groups: list[str]            # display order
    unmapped_codes: dict[str, int]


def group_values(value, field: str, explode: bool, codes: dict[str, dict[str, str]],
                 unmapped: dict[str, int]) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() in ("", "-", "Not Applicable"):
        return [NOT_CLASSIFIED if str(value).strip() != "Not Applicable" else "Not applicable"]
    parts = [p.strip() for p in str(value).split(",")] if explode else [str(value).strip()]
    out = []
    cmap = codes.get(field, {})
    for p in parts:
        if re.fullmatch(r"\d+(\.0)?", p):
            code = p.split(".")[0]
            if code in cmap:
                p = cmap[code]
            else:
                unmapped[code] = unmapped.get(code, 0) + 1
                p = f"Code {code} (unmapped)"
        out.append(p)
    return sorted(set(out), key=out.index)


def build_lens(df: pd.DataFrame, s: ScoreResult, field: str, label: str, explode: bool,
               codes: dict[str, dict[str, str]], log: QualityLog, min_groups: int = 1) -> Lens | None:
    if field not in df:
        log.add("info", "Lenses", f"Lens '{label}' skipped: field not present in data")
        return None
    unmapped: dict[str, int] = {}
    rows = []
    for t in s.order:
        for g in group_values(df.at[t, field], field, explode, codes, unmapped):
            rows.append({"group": g, "ticker": t, "rank": s.rank[t]})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return None
    sizes = tab.groupby("group").size()
    if len(sizes) < min_groups:
        return None
    groups = sorted(sizes.index, key=lambda g: (g in (NOT_CLASSIFIED, "Not applicable"), -sizes[g], g))
    tab["group_n"] = tab["group"].map(sizes)
    tab["_g"] = tab["group"].map({g: i for i, g in enumerate(groups)})
    tab["_r"] = tab["rank"].fillna(10 ** 6)
    tab = tab.sort_values(["_g", "_r", "ticker"]).reset_index(drop=True)
    tab["within_rank"] = tab.groupby("group").cumcount() + 1
    tab.loc[tab["rank"].isna(), "within_rank"] = pd.NA
    tab = tab.drop(columns=["_g", "_r"])
    if unmapped:
        codes_txt = ", ".join(f"{c} ({n} fund{'s' if n > 1 else ''})" for c, n in sorted(unmapped.items(), key=lambda kv: int(kv[0])))
        log.add("warning", "Lenses",
                f"{label}: Morningstar numeric codes without a verified label - {codes_txt}. "
                f"Shown as 'Code N (unmapped)'. Confirm in Morningstar and add to codes.yaml.")
    if explode:
        log.add("info", "Lenses", f"{label}: multi-value entries (e.g. 'Dividend,Yield') are split, so a fund can appear in several groups")
    return Lens(field, label, tab, groups, unmapped)


def group_leaders(lenses: list[Lens], top: int = 3) -> pd.DataFrame:
    rows = []
    for L in lenses:
        for g in L.groups:
            sub = L.table[(L.table["group"] == g) & L.table["within_rank"].notna()].head(top)
            for _, r in sub.iterrows():
                rows.append({"lens": L.label, "group": g, "group_n": int(r["group_n"]),
                             "within_rank": int(r["within_rank"]), "ticker": r["ticker"]})
    return pd.DataFrame(rows)

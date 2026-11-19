"""Hard screens. Excluded funds are always listed with the reason."""
from __future__ import annotations

import pandas as pd

from .quality import QualityLog


def apply_screens(df: pd.DataFrame, prof: dict, log: QualityLog) -> tuple[list[str], list[dict]]:
    s = prof.get("screens", {}) or {}
    excluded: list[dict] = []
    keep: list[str] = []
    excl_obj = {o.strip().lower() for o in (s.get("exclude_objectives") or [])}
    excl_tick = {t.strip().upper() for t in (s.get("exclude_tickers") or [])}
    for t, row in df.iterrows():
        reasons = []
        aum, er = row.get("aum"), row.get("net_er")
        if s.get("min_aum_mm") is not None and pd.notna(aum) and aum < float(s["min_aum_mm"]):
            reasons.append(f"AUM ${aum:,.0f}MM < ${float(s['min_aum_mm']):,.0f}MM minimum")
        if s.get("max_net_er") is not None and pd.notna(er) and er > float(s["max_net_er"]):
            reasons.append(f"Net expense {er:.2f}% > {float(s['max_net_er']):.2f}% maximum")
        if s.get("min_history_years") is not None:
            hy = row.get("history_years")
            if pd.isna(hy) or hy < float(s["min_history_years"]):
                reasons.append(f"Track record {0 if pd.isna(hy) else int(hy)}y < {s['min_history_years']}y minimum")
        if s.get("exclude_leveraged") and str(row.get("leveraged") or "").strip().lower() == "yes":
            reasons.append("Leveraged fund")
        if excl_obj and str(row.get("objective") or "").strip().lower() in excl_obj:
            reasons.append(f"Objective '{row.get('objective')}' excluded")
        if t in excl_tick:
            reasons.append("Manually excluded")
        if reasons:
            excluded.append({"ticker": t, "name": row.get("name"), "reason": "; ".join(reasons)})
            log.add("info", "Screens", "Excluded: " + "; ".join(reasons), t)
        else:
            keep.append(t)
    return keep, excluded

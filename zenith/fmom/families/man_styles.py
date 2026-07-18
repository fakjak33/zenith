"""Man-inspired style family: 13 broad equity styles as ETF composites.

Man AHL's style trend strategy trades ~13 aggregated equity styles; the closest
free-data replication is a composite per style built from the Morningstar
strategic-beta catalog (data/fmom/etf_catalog.json): equal-weight mean of the
member ETFs' monthly returns minus SPY, so each style is a market-neutral-ish
return stream like the styles in the paper. Membership is tag-driven and
transparent — the UI shows exactly which funds make up each style.
"""

from __future__ import annotations

import pandas as pd

from .. import core, load
from ..catalog import (BENCHMARK, CORE_STYLE_TAGS, MAN_EXCLUDE_CATEGORY,
                       MAN_EXCLUDE_GROUPS, MAN_MAX_MEMBERS, MAN_MIN_MEMBERS,
                       MAN_STYLES)


def _eligible(etf: dict) -> bool:
    if set(etf.get("groups", [])) & MAN_EXCLUDE_GROUPS:
        return False
    cat = etf.get("category", "")
    return not any(x in cat for x in MAN_EXCLUDE_CATEGORY)


def members(catalog: dict | None = None) -> dict[str, list[dict]]:
    """style -> member ETFs (dicts from the catalog), top MAN_MAX_MEMBERS by
    AUM. 'pure' styles reject funds tagged with 3+ core style groups (those are
    Multifactor); Multifactor takes exactly those funds."""
    catalog = catalog if catalog is not None else load("etf_catalog", {})
    etfs = [e for e in catalog.get("etfs", []) if _eligible(e)]
    out: dict[str, list[dict]] = {}
    for style, rule in MAN_STYLES.items():
        picks = []
        for e in etfs:
            tags = set(e.get("groups", []))
            sheets = set(e.get("sheets", []))
            n_core = len(tags & CORE_STYLE_TAGS)
            is_multi = (n_core >= 3 or "MULTIFACTOR 1" in sheets
                        or "MULTIFACTOR 2" in sheets)
            if rule.get("multifactor"):
                hit = is_multi
            else:
                hit = bool(tags & set(rule["groups"]))
                if hit and rule.get("pure") and is_multi:
                    hit = False                      # that's a multifactor fund
                if hit and tags & set(rule.get("exclude", [])):
                    hit = False                      # off-label tilt
            if hit:
                picks.append(e)
        picks.sort(key=lambda e: e.get("aum_m") or 0.0, reverse=True)
        picks = picks[:MAN_MAX_MEMBERS]
        have = {e["ticker"] for e in picks}
        for extra in rule.get("extra", []):
            if extra["ticker"] not in have:
                picks.append({**extra, "groups": [], "sheets": [],
                              "category": "manual", "aum_m": None, "er": None})
        out[style] = picks
    return out


def tickers(catalog: dict | None = None) -> list[str]:
    mem = members(catalog)
    return sorted({e["ticker"] for v in mem.values() for e in v}) + [BENCHMARK]


def build_panel(px: dict[str, pd.DataFrame],
                catalog: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Monthly SPY-excess composite returns per style. A style only enters a
    month once MAN_MIN_MEMBERS member ETFs have data (composites are equal-
    weight across whichever members are alive, so history extends as far back
    as the three oldest members)."""
    bench = px.get(BENCHMARK)
    if bench is None or bench.empty:
        return pd.DataFrame(), {"ok": False, "error": f"no {BENCHMARK} history"}
    bench_m = core.monthly_returns(bench["close"])

    mem = members(catalog)
    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    membership_meta: dict[str, list[str]] = {}
    for style, etfs in mem.items():
        rets = []
        used = []
        for e in etfs:
            df = px.get(e["ticker"])
            if df is None or df.empty:
                missing.append(e["ticker"])
                continue
            rets.append(core.monthly_returns(df["close"]).rename(e["ticker"]))
            used.append(e["ticker"])
        membership_meta[style] = used
        if len(used) < MAN_MIN_MEMBERS:
            continue
        wide = pd.concat(rets, axis=1)
        alive = wide.notna().sum(axis=1)
        composite = wide.mean(axis=1)[alive >= MAN_MIN_MEMBERS]
        cols[style] = (composite - bench_m).dropna()
    panel = pd.DataFrame(cols).sort_index()
    today = pd.Timestamp.today()
    panel = panel[panel.index < today.replace(day=1).normalize()]
    meta = {"ok": not panel.empty, "n_factors": panel.shape[1],
            "missing": sorted(set(missing)), "benchmark": BENCHMARK,
            "members": membership_meta,
            "last_month": panel.index[-1].strftime("%Y-%m") if len(panel) else None}
    return panel, meta

"""Multivariate Trend universe assembly: the Russell 1000 (equities, reused
verbatim from mom.universe) and a broad, non-leveraged ETF universe (new).

Leverage/inverse exclusion is layered because no single check is sufficient
(verified while building this): only 574 of the ~935 raw union tickers carry
a name at all via the committed Morningstar catalog, and the 361 unnamed
ones include real leveraged/inverse funds (SOXL, SQQQ, SPXU, UVXY, SVXY, SH,
PSQ, DOG, RWM, NAIL, ...). Four layers:

  1. config.MOM_MVT_LEVERAGED_EXCLUDE  -- explicit, reviewed ticker list.
  2. Name regex                        -- catches anything named honestly.
  3. A committed, self-healing metadata cache (mirrors mom.universe's proven
     refresh_metadata pattern) so the 361 unnamed tickers get a name+category
     within about a week, after which the regex applies to them too.
  4. An EMPIRICAL backstop applied once daily return data exists (see
     duplicate_and_leverage_gate) -- a fund with far higher realized vol
     than a broad benchmark AND near-total correlation to it is leveraged
     regardless of what its name says or whether metadata has caught up.

Also handles near-duplicate clustering (SPY/VOO/IVV/SPLG etc. would each
have ~zero spread-vol against each other, which blows up a pairwise
division) -- kept to the single most liquid member of any tightly
correlated cluster.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date

import pandas as pd

from ...config import FMOM_FILES, MOM_MVT_LEVERAGED_EXCLUDE
from ...cas import universe as cas_universe
from .. import universe as mom_universe
from . import load, save

META_TTL_DAYS = 30
META_MAX_PER_RUN = 150

# Matches funds that self-describe as leveraged/inverse. Deliberately narrow:
# a naive match on bare "ultra" or "short" false-positives on names like
# "Ultra Dividend Revenue" (RDIV) or "Short Duration" (SBND) -- so this only
# fires on the specific leveraged/inverse vocabulary funds actually use.
LEV_NAME_RE = re.compile(
    r"(\b[23]x\b|\bultrapro\b|\bultrashort\b|\bdaily\s+(2x|3x)\b|"
    r"\bleveraged\b|\binverse\b|\bbull\s*[23]x\b|\bbear\s*[23]x\b|"
    r"[-−]1x\b|\b(200|300)%\b)",
    re.I,
)


# --------------------------------------------------------------- equities --
def equities_constituents(max_age_hours: float = 168.0) -> tuple[list[dict], dict]:
    """Thin pass-through to mom.universe.constituents() -- MOMENTUM already
    maintains the one Russell 1000 pipeline; mvt does not build a second."""
    return mom_universe.constituents(max_age_hours=max_age_hours)


# -------------------------------------------------------------------- ETFs --
def _etf_catalog() -> dict[str, dict]:
    """ticker -> {name, category, groups, aum_m, er} from the committed
    Morningstar strategic-beta export (data/fmom/etf_catalog.json, 574
    names) -- reused as-is, not re-fetched."""
    if not FMOM_FILES["etf_catalog"].exists():
        return {}
    doc = json.loads(FMOM_FILES["etf_catalog"].read_text(encoding="utf-8"))
    return {e["ticker"]: e for e in doc.get("etfs", []) if e.get("ticker")}


def raw_etf_tickers() -> list[str]:
    """Union of Zenith's existing ETF universes: the curated ~400-name master
    list + the ~335-name factor-rotation tagged universe + the 574-name
    Morningstar catalog -- ~935 unique, verified when this was designed. No
    new external source (per the user's explicit choice)."""
    catalog = _etf_catalog()
    out = set(cas_universe.master_etfs()) | set(cas_universe.frm_tickers()) | set(catalog)
    return sorted(out)


def etf_metadata() -> dict:
    return load("etf_meta", {})


def _stale(entry: dict | None, today: date, ttl_days: int = META_TTL_DAYS) -> bool:
    if not entry or not entry.get("asof"):
        return True
    try:
        asof = date.fromisoformat(entry["asof"])
    except Exception:
        return True
    return (today - asof).days > ttl_days


def refresh_etf_metadata(tickers: list[str], max_per_run: int = META_MAX_PER_RUN,
                         ttl_days: int = META_TTL_DAYS, sleep: float = 0.1) -> dict:
    """Best-effort name/category cache for ETFs the Morningstar catalog
    doesn't name (~361 of ~935). Mirrors mom.universe.refresh_metadata
    exactly (same TTL, same per-run cap, same committed-not-gitignored
    reasoning -- CI starts cold every run) so the whole R1000+ETF metadata
    footprint follows one proven pattern rather than two."""
    catalog = _etf_catalog()
    today = date.today()
    meta = etf_metadata()
    need = [t for t in tickers if t not in catalog and _stale(meta.get(t), today, ttl_days)]
    todo = need[:max_per_run]
    if not todo:
        return {"checked": len(tickers), "stale": 0, "refreshed": 0}
    try:
        import yfinance as yf
    except Exception:
        return {"checked": len(tickers), "stale": len(need), "refreshed": 0,
                "error": "yfinance unavailable"}
    refreshed = 0
    for i, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info or {}
            meta[t] = {
                "name": info.get("longName") or info.get("shortName") or t,
                "category": info.get("category") or "",
                "asof": today.isoformat(),
            }
            refreshed += 1
        except Exception:
            pass
        if sleep:
            time.sleep(sleep)
        if i and i % 100 == 0:
            save("etf_meta", meta)
    save("etf_meta", meta)
    return {"checked": len(tickers), "stale": len(need), "refreshed": refreshed}


def _name_of(ticker: str, catalog: dict, meta: dict) -> str:
    """Best-available display name. Checked in order: the Morningstar
    catalog (574 names), the label already known to cas.universe (the core
    ~53 sector/theme/factor/macro tickers, e.g. SPY/QQQ/TLT/GLD, PLUS the
    ~335-name factor-rotation tag set -- these are Zenith's own curated
    names and would otherwise be wrongly treated as "unnamed" since they
    never needed a yfinance metadata cache), then the mvt metadata cache."""
    if ticker in catalog:
        return catalog[ticker].get("name") or ticker
    known = cas_universe.label_of(ticker)
    if known and known != ticker:
        return known
    return (meta.get(ticker) or {}).get("name") or ""


def is_leveraged_by_name(ticker: str, name: str) -> bool:
    if ticker in MOM_MVT_LEVERAGED_EXCLUDE:
        return True
    return bool(name) and bool(LEV_NAME_RE.search(name))


def tag_of(ticker: str, catalog: dict, meta: dict) -> dict:
    """Best-effort category metadata: asset class / sector / industry /
    factor / geography / strategy, per spec section 24. Derived entirely
    from data Zenith already has -- cas.universe's factor-rotation tags +
    asset_class_of() + the Morningstar catalog's own category/groups -- no
    new taxonomy is built from scratch."""
    tag = cas_universe.frm_tag(ticker) or {}
    cat_row = catalog.get(ticker, {})
    in_core = ticker in cas_universe.all_etfs()
    asset_class = cas_universe.asset_class_of(ticker) if in_core else (tag.get("group") or "")
    if tag:
        label = tag.get("label") or ""
    elif in_core:
        label = cas_universe.label_of(ticker)
    else:
        label = cat_row.get("category") or ""
    return {
        "asset_class": asset_class,
        "label": label,
        "region": tag.get("region") or "",
        "morningstar_category": cat_row.get("category") or (meta.get(ticker) or {}).get("category") or "",
        "morningstar_groups": cat_row.get("groups") or [],
    }


def etf_universe(refresh_meta: bool = True) -> tuple[list[dict], dict]:
    """Final, name/regex-gated ETF universe (before the empirical vol/corr
    backstop, which needs a return panel and runs in panel.py). Every raw
    ticker survives into the returned list with an `included` flag and, if
    excluded, a stated reason -- nothing is silently dropped."""
    catalog = _etf_catalog()
    raw = raw_etf_tickers()
    status = refresh_etf_metadata(raw) if refresh_meta else {"refreshed": 0}
    meta = etf_metadata()

    rows = []
    for t in raw:
        name = _name_of(t, catalog, meta)
        row = {"ticker": t, "name": name or t, **tag_of(t, catalog, meta)}
        if is_leveraged_by_name(t, name):
            row["included"] = False
            row["exclusion_reason"] = "leveraged_or_inverse_by_name"
        elif not name:
            row["included"] = False
            row["exclusion_reason"] = "no_name_metadata_yet"
        else:
            row["included"] = True
        rows.append(row)
    return rows, {"n_raw": len(raw), "meta_refresh": status,
                  "n_excluded_by_name": sum(1 for r in rows if not r["included"])}


# ---------------------------------------------------- empirical backstop ---
def duplicate_and_leverage_gate(returns: pd.DataFrame, adv: dict[str, float] | None = None,
                                benchmarks: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD"),
                                cluster_corr: float = 0.99, lev_corr: float = 0.95,
                                lev_vol_ratio: float = 2.5) -> dict[str, str]:
    """Returns {ticker: reason} for names to drop AFTER name/regex gating,
    using the actual daily-return panel:

      * near-duplicate clustering -- any pair with correlation >= cluster_corr
        (e.g. SPY/VOO/IVV/SPLG) collapses to its most-liquid (by ADV$, else
        alphabetically-first as a stable tiebreak) member. Necessary: two
        near-identical instruments have a spread-vol near zero, which
        explodes (r_A - r_B) / sigma_AB.
      * leverage/inverse backstop -- a fund correlated >= lev_corr with one
        of `benchmarks` AND with realized vol >= lev_vol_ratio x that
        benchmark's vol is leveraged regardless of what its name says (or
        whether metadata has caught up). This is what catches anything the
        name-based layers in this module miss.
    """
    cols = [c for c in returns.columns if returns[c].notna().sum() >= 60]
    if len(cols) < 3:
        return {}
    R = returns[cols].fillna(0.0)
    vol = R.std()
    excluded: dict[str, str] = {}

    present_benchmarks = [b for b in benchmarks if b in R.columns]
    for b in present_benchmarks:
        bvol = vol.get(b)
        if not bvol or bvol <= 0:
            continue
        corr_b = R.corrwith(R[b])
        for t in cols:
            if t == b or t in excluded:
                continue
            c = corr_b.get(t)
            v = vol.get(t)
            if c is None or v is None:
                continue
            if abs(c) >= lev_corr and v >= lev_vol_ratio * bvol:
                excluded[t] = f"empirical_leverage_vs_{b}(corr={c:.2f},vol_ratio={v / bvol:.1f}x)"

    remaining = [c for c in cols if c not in excluded]
    if len(remaining) >= 2:
        C = R[remaining].corr().to_numpy()
        n = len(remaining)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if C[i, j] >= cluster_corr:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            if len(members) < 2:
                continue
            names = [remaining[i] for i in members]
            keeper = max(names, key=lambda t: adv.get(t, 0.0)) if adv else sorted(names)[0]
            for t in names:
                if t != keeper:
                    excluded[t] = f"near_duplicate_of:{keeper}"

    return excluded

"""Top-level pipeline: files + profile -> Comparison objects (full universe, approved-only)."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import __version__
from .derive import add_derived, apply_recency
from .diagnostics import Diagnostics, diagnose
from .dictionary import Dictionary, load_codes, load_dictionary
from .ingest import ApprovedList, Dataset, load_approved, load_dataset
from .lenses import Lens, build_lens, group_leaders
from .profile import load_profile, profile_hash
from .quality import QualityLog
from .scoring import ScoreResult, score
from .screens import apply_screens
from .themes import theme_scan


@dataclass
class Comparison:
    kind: str                         # "full" | "approved"
    title: str
    category: str
    vehicle: str
    asof: dt.date | None
    df: pd.DataFrame                  # universe funds (+ derived), indexed by ticker
    scores: ScoreResult
    diag: Diagnostics
    lenses: list[Lens]
    leaders: pd.DataFrame
    excluded: list[dict]
    approved: ApprovedList | None
    approved_flag: pd.Series | None   # ticker -> bool (universe)
    audit: pd.DataFrame | None
    prof: dict
    d: Dictionary
    dataset: Dataset
    log: QualityLog
    meta: dict = field(default_factory=dict)
    shortlist: list[str] = field(default_factory=list)
    recency: dict = field(default_factory=dict)

    @property
    def ranked(self) -> list[str]:
        return self.scores.order


def _vehicle(ds: Dataset) -> str:
    v = (ds.mode("vehicle") or "").lower()
    if "etf" in v or "exchange" in v:
        return "ETFs"
    if v:
        return "Mutual Funds" if ("open" in v or "mutual" in v or v in ("fe", "mf")) else v.upper()
    return "Funds"


def _category(ds: Dataset) -> str:
    """'Large Value', or 'Large Value & Large Blend' when the input mixes categories."""
    clean = lambda c: re.sub(r"^(US|EAA|Global)\s+Fund\s+", "", str(c)).strip()
    if "category" not in ds.df or ds.df["category"].dropna().empty:
        return "Fund Comparison"
    vc = ds.df["category"].dropna().map(clean).value_counts()
    if vc.iloc[0] / vc.sum() >= 0.8:
        return vc.index[0]
    names = list(vc.index[:3])
    return " & ".join(names) + (" & others" if len(vc) > 3 else "")


def _run_universe(kind: str, tickers: list[str], full: pd.DataFrame, prof: dict, d: Dictionary,
                  log: QualityLog, codes: dict, bench: dict | None = None) -> tuple:
    df = add_derived(full.loc[tickers], prof, log, bench=bench)
    eff = apply_recency(prof)
    s = score(df, eff, d, log)
    diag = diagnose(df, s, eff)
    lenses = [L for L in (build_lens(df, s, lz["field"], lz["label"], bool(lz.get("explode")), codes, log,
                                     int(lz.get("min_groups", 1)))
                          for lz in prof["lenses"]) if L is not None]
    rec = _recency_report(df, s, prof, d, log, codes)
    return df, s, diag, lenses, group_leaders(lenses), rec


def _recency_report(df: pd.DataFrame, s: ScoreResult, prof: dict, d: Dictionary, log: QualityLog,
                    codes: dict) -> dict:
    """Rank with recency control off (for comparison) plus the theme scan."""
    rc = prof["recency"]
    out = {"mode": rc["mode"], "surge_flag": float(rc["surge_flag"]), "rank_off": s.rank}
    if rc["mode"] != "off":
        off = dict(prof, recency=dict(rc, mode="off"))
        out["rank_off"] = score(df, apply_recency(off), d, QualityLog()).rank
    skew = df.get("recency_skew")
    if skew is not None:
        n_surge = int((skew >= out["surge_flag"]).sum())
        if n_surge:
            log.add("info", "Recency", f"{n_surge} fund(s) flagged 'Recent surge': 1-year peer percentile is "
                                       f"{out['surge_flag']:.0f}+ points above their ex-latest-year percentile")
    out["themes"], out["fund_hot"] = theme_scan(df, s, out["rank_off"], prof, codes, log)
    if not out["themes"].empty and (out["themes"]["flag"] == "Hot year").any() and rc["mode"] == "off":
        log.add("warning", "Recency", "Hot-year themes detected while recency control is OFF - consider 'balanced'.")
    return out


def run(data_paths: list[str | Path], profile: str | dict = "us_equity_style",
        approved_path: str | Path | None = None, shortlist: list[str] | None = None,
        approved_require_solicitation: bool = False) -> list[Comparison]:
    d = load_dictionary()
    codes = load_codes()
    prof = load_profile(profile)
    base_log = QualityLog()
    ds = load_dataset(data_paths, base_log, d)
    if ds.df.empty:
        raise ValueError("No funds found in the input file(s).")

    category, vehicle = _category(ds), _vehicle(ds)
    asof = ds.mode("ret_date")
    pre = add_derived(ds.df, prof, QualityLog(), note=False, bench=ds.benchmark)
    eligible, excluded = apply_screens(pre, prof, base_log)

    approved = load_approved(approved_path, base_log) if approved_path else None
    meta = {
        "engine_version": __version__,
        "profile_id": prof.get("id"), "profile_name": prof.get("name"), "profile_hash": profile_hash(prof),
        "files": dict(ds.files), "returns_basis": ", ".join(sorted({"qtr": "quarter-end", "mo": "month-end"}[f] for f in ds.freqs)) or "n/a",
        "portfolio_date": ds.mode("portfolio_date"), "n_input": len(ds.df), "layouts": ds.layouts,
    }
    if approved:
        meta["approved_file"] = {approved.file: approved.hash}

    out = []
    # ---------------------------------------------------------------- full universe
    log = base_log.copy()
    df, s, diag, lenses, leaders, rec = _run_universe("full", eligible, ds.df, prof, d, log, codes, ds.benchmark)
    appr_flag = audit = None
    if approved:
        aset = set(approved.tickers)
        appr_flag = pd.Series({t: t in aset for t in df.index})
        audit = _audit(ds, approved, eligible, excluded, approved_require_solicitation)
    out.append(Comparison("full", f"{category} {vehicle}", category, vehicle, asof, df, s, diag, lenses, leaders,
                          excluded, approved, appr_flag, audit, prof, d, ds, log, dict(meta, n_universe=len(df)),
                          list(shortlist or []), rec))

    # ---------------------------------------------------------------- approved only
    if approved:
        incl = set(audit.loc[audit["status"] == "Included", "ticker"])
        tick = [t for t in eligible if t in incl]
        alog = base_log.copy()
        alog.add("info", "Approved list", f"Approved-only comparison re-ranks {len(tick)} funds among themselves "
                                          f"(percentiles and peer medians recomputed)")
        if len(tick) < 5:
            alog.add("warning", "Approved list", f"Only {len(tick)} approved funds - percentile scores are coarse")
        if tick:
            df2, s2, diag2, lenses2, leaders2, rec2 = _run_universe("approved", tick, ds.df, prof, d, alog, codes, ds.benchmark)
            out.append(Comparison("approved", f"{category} {vehicle} - Approved", category, vehicle, asof, df2, s2,
                                  diag2, lenses2, leaders2, excluded, approved,
                                  pd.Series({t: True for t in df2.index}), audit, prof, d, ds, alog,
                                  dict(meta, n_universe=len(df2)), [t for t in (shortlist or []) if t in df2.index],
                                  rec2))
    return out


def _audit(ds: Dataset, ap: ApprovedList, eligible: list[str], excluded: list[dict], require_sol: bool) -> pd.DataFrame:
    rows = []
    elig = set(eligible)
    exc = {e["ticker"]: e["reason"] for e in excluded}
    aset = set(ap.tickers)
    for t in ds.df.index:
        info = ap.rows.get(t, {})
        cat = info.get("morningstar category")
        sol = info.get("available for solicitation")
        if t in aset:
            if require_sol and (sol or "").lower() != "yes":
                status, reason = "Approved - not solicitable", "Available for Solicitation is not 'Yes'"
            elif t in exc:
                status, reason = "Approved - screened out", exc[t]
            else:
                status, reason = "Included", "Ticker on approved list"
        else:
            status, reason = "Not approved", "Ticker not on approved list"
        rows.append({"ticker": t, "name": ds.df.at[t, "name"], "status": status, "reason": reason,
                     "approved_category": cat, "solicitation": sol,
                     "complexity": info.get("complexity"), "active": info.get("actively managed")})
    for t in ap.tickers:
        if t not in ds.df.index:
            info = ap.rows.get(t, {})
            rows.append({"ticker": t, "name": info.get("name"), "status": "Approved - not in data",
                         "reason": "On approved list but missing from the Morningstar export",
                         "approved_category": info.get("morningstar category"),
                         "solicitation": info.get("available for solicitation"),
                         "complexity": info.get("complexity"), "active": info.get("actively managed")})
    order = {"Included": 0, "Approved - screened out": 1, "Approved - not solicitable": 2,
             "Approved - not in data": 3, "Not approved": 4}
    return pd.DataFrame(rows).sort_values(["status", "ticker"], key=lambda c: c.map(order) if c.name == "status" else c)\
        .reset_index(drop=True)

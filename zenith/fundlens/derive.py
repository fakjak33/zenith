"""Metrics computed by the engine from the raw fields (deterministic, documented)."""
from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pandas as pd

from .dictionary import HORIZON_ORDER
from .dictionary import load_codes
from .quality import QualityLog
from .scoring import pct_rank

MOM_RX = re.compile(r"\bmom(entum|t)\b", re.I)


def annual_years(df: pd.DataFrame, prefix: str = "nav_annual_") -> list[int]:
    """Calendar years present (with any data), newest first."""
    ys = [int(c[len(prefix):]) for c in df.columns
          if c.startswith(prefix) and c[len(prefix):].isdigit() and df[c].notna().any()]
    return sorted(ys, reverse=True)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


SECTORS = ["sec_basic_materials", "sec_consumer_cyclical", "sec_financial", "sec_real_estate", "sec_communication",
           "sec_energy", "sec_industrials", "sec_technology", "sec_consumer_defensive", "sec_healthcare", "sec_utilities"]


def add_derived(df: pd.DataFrame, prof: dict, log: QualityLog, note: bool = True,
                bench: dict | None = None) -> pd.DataFrame:
    """Return a copy of `df` (one comparison universe) with derived columns added.

    Peer-relative fields (% years top half) depend on the universe, so this is
    re-run for the approved-only comparison.
    """
    df = df.copy()
    bench = bench or {}

    # Expense ratio: prospectus net if present, else annual-report net (pasted mutual-fund grids)
    if "annual_net_er" in df:
        er = _num(df, "net_er") if "net_er" in df else pd.Series(np.nan, index=df.index)
        fill = er.isna() & _num(df, "annual_net_er").notna()
        if fill.any():
            df["net_er"] = er.where(~fill, _num(df, "annual_net_er"))
            if note:
                log.add("info", "Data", f"Net expense ratio taken from the annual report for {int(fill.sum())} fund(s) "
                                        "(no prospectus net expense ratio in the input)")

    # Sector active weights vs the benchmark row, and the overweight tags used by the Theme Scan
    have = [c for c in SECTORS if c in df and c in bench]
    if have:
        tilt_pp = float(prof.get("recency", {}).get("sector_tilt_pp", 5.0))
        for c in have:
            df[f"{c}_active"] = _num(df, c) - float(bench[c])
        def tilts(t):
            ow = [(df.at[t, f"{c}_active"], c) for c in have
                  if pd.notna(df.at[t, f"{c}_active"]) and df.at[t, f"{c}_active"] >= tilt_pp]
            return ", ".join(SECTOR_NAMES[c] for _, c in sorted(ow, reverse=True)) or None
        df["sector_tilt"] = [tilts(t) for t in df.index]

        def unders(t):
            uw = [(df.at[t, f"{c}_active"], c) for c in have
                  if pd.notna(df.at[t, f"{c}_active"]) and df.at[t, f"{c}_active"] <= -tilt_pp]
            return ", ".join(SECTOR_NAMES[c] for _, c in sorted(uw)) or None
        df["sector_under"] = [unders(t) for t in df.index]
    sc = prof["scoring"]

    for h in HORIZON_ORDER:
        up, dn = f"up_capture_{h}", f"down_capture_{h}"
        if up in df and dn in df:
            df[f"capture_spread_{h}"] = _num(df, up) - _num(df, dn)
        r2 = f"r_squared_{h}"
        if r2 in df:
            df[f"corr_est_{h}"] = np.sqrt(_num(df, r2).clip(lower=0) / 100.0)

    if "prem_disc_12m" in df:
        df["abs_prem_disc"] = _num(df, "prem_disc_12m").abs()

    years = annual_years(df)
    if years:
        last5 = [y for y in years[:5]]
        cols5 = [f"nav_annual_{y}" for y in last5]
        if len(last5) == 5:
            block = df[cols5].apply(pd.to_numeric, errors="coerce")
            df["worst_year_5y"] = block.min(axis=1).where(block.notna().all(axis=1))

        window = years[: int(sc["calendar_years"])]
        wcols = [f"nav_annual_{y}" for y in window]
        block = df[wcols].apply(pd.to_numeric, errors="coerce")
        med = block.median(axis=0, skipna=True)
        beat = block.ge(med, axis=1).astype(float).where(block.notna())
        n_obs = block.notna().sum(axis=1).astype(float)
        df["top_half_pct"] = (beat.sum(axis=1) / n_obs.where(n_obs > 0) * 100).where(n_obs >= 3)

    # Worst 3M since inception is only comparable for funds that lived through the stress year.
    if "worst_3m" in df:
        sy = int(sc["stress_year"])
        stress_col = f"nav_annual_{sy}"
        if "inception_date" in df and df["inception_date"].notna().any():
            cutoff = dt.date(sy, 1, 1)
            eligible = df["inception_date"].map(lambda d: isinstance(d, dt.date) and d < cutoff).astype(bool)
            basis = f"an inception date before {sy}"
        elif stress_col in df:
            eligible = df[stress_col].notna()
            basis = f"a full {sy} calendar-year return"
        else:
            eligible = df.get("nav_ret_10y", pd.Series(np.nan, index=df.index)).notna()
            basis = "a 10-year track record"
        df["worst_3m_eligible"] = _num(df, "worst_3m").where(eligible)
        n_out = int((df["worst_3m"].notna() & ~eligible).sum())
        if note and n_out:
            log.add("info", "Scoring",
                    f"Worst 3M not scored for {n_out} fund(s) without {basis} (not crisis-tested; would look artificially mild)")

    add_recency(df, prof)

    asof = None
    if "ret_date" in df and df["ret_date"].notna().any():
        asof = max(d for d in df["ret_date"] if isinstance(d, dt.date))

    def hist(row) -> float:
        inc = row.get("inception_date")
        if asof is not None and isinstance(inc, dt.date):
            return round((asof - inc).days / 365.25, 1)
        for h, yrs in (("15y", 15), ("10y", 10), ("5y", 5), ("3y", 3)):
            v = row.get(f"nav_ret_{h}")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(yrs)
        n = sum(1 for y in years if pd.notna(row.get(f"nav_annual_{y}")))
        return float(n) if n else np.nan

    df["history_years"] = df.apply(hist, axis=1) if len(df) else pd.Series(dtype=float)
    return df


def add_recency(df: pd.DataFrame, prof: dict) -> None:
    """Ex-latest-year returns, recency skew/surge and momentum tags (in place)."""
    r1 = _num(df, "nav_ret_12m")
    for h, n in (("3y", 3), ("5y", 5), ("10y", 10), ("15y", 15)):
        if f"nav_ret_{h}" not in df:
            continue
        rn = _num(df, f"nav_ret_{h}")
        growth = (1 + rn / 100.0) ** n / (1 + r1 / 100.0)
        df[f"nav_ret_xr_{h}"] = ((growth.clip(lower=1e-9) ** (1.0 / (n - 1)) - 1) * 100.0).where(rn.notna() & r1.notna())

    xr_cols = [(c, w) for c, w in (("nav_ret_xr_3y", 0.4), ("nav_ret_xr_5y", 0.6)) if c in df]
    if r1.notna().sum() >= 5 and xr_cols:
        p1 = pct_rank(r1, "higher")
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for c, w in xr_cols:
            pc = pct_rank(df[c], "higher")
            num += pc.fillna(0.0) * w
            den += pc.notna().astype(float) * w
        pxr = (num / den).where(den > 0)
        df["recency_skew"] = p1 - pxr
        df["recency_surge"] = df["recency_skew"].clip(lower=0)

    codes = load_codes().get("sb_group", {})
    def is_mom(t) -> bool:
        g = str(df.at[t, "sb_group"]) if "sb_group" in df and pd.notna(df.at[t, "sb_group"]) else ""
        g = codes.get(g.split(".")[0], g)
        sel = str(df.at[t, "index_selection"]) if "index_selection" in df and pd.notna(df.at[t, "index_selection"]) else ""
        name = str(df.at[t, "name"]) if "name" in df and pd.notna(df.at[t, "name"]) else ""
        return g == "Momentum" or "momentum" in sel.lower() or bool(MOM_RX.search(name))
    df["momentum_tilt"] = ["Yes" if is_mom(t) else None for t in df.index]


def apply_recency(prof: dict) -> dict:
    """Expand the profile's recency mode into concrete pillar metrics (engine-time; profile file untouched)."""
    import copy
    rc = prof.get("recency") or {}
    mode = rc.get("mode", "off")
    p = copy.deepcopy(prof)
    if mode == "off":
        return p
    pil = p["pillars"]
    if "returns" in pil and "nav_ret" in pil["returns"]["metrics"]:
        w = pil["returns"]["metrics"]["nav_ret"]
        if mode == "balanced":
            pil["returns"]["metrics"]["nav_ret_xr"] = w
        else:                                            # strict
            del pil["returns"]["metrics"]["nav_ret"]
            pil["returns"]["metrics"]["nav_ret_xr"] = w
    target = "consistency" if pil.get("consistency", {}).get("weight", 0) > 0 else "downside"
    if target in pil:
        pil[target]["metrics"]["recency_surge"] = 1.0 if mode == "balanced" else 2.0
    return p


SECTOR_NAMES = {"sec_basic_materials": "Basic Materials", "sec_consumer_cyclical": "Consumer Cyclical",
                "sec_financial": "Financial Services", "sec_real_estate": "Real Estate",
                "sec_communication": "Communication Services", "sec_energy": "Energy", "sec_industrials": "Industrials",
                "sec_technology": "Technology", "sec_consumer_defensive": "Consumer Defensive",
                "sec_healthcare": "Healthcare", "sec_utilities": "Utilities"}

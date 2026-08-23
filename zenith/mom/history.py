"""MOMENTUM history: sharded composite history + the append-only pick tracker.

Two distinct stores, both append-only:

  * SHARDED daily/weekly history (data/mom/history/<YYYY>.json) — every
    trading day appends {ticker, composite, state}; once a week (Fridays)
    the row also carries the full factor breakdown. `edge`/`nightday`'s
    `history.json` is a single flat file rewritten in full on every append;
    at ~1000 rows/day that becomes a multi-MB daily rewrite within a year.
    Sharding by year means only the CURRENT year's file is ever touched.

  * The decile PICK TRACKER (data/mom/picks.json) — modeled directly on
    `edge.history`: one row per (ticker, asof) for the composite's top/bottom
    decile, entry close filled in, SPY-excess evaluated once each horizon
    (20/60/126 trading days — roughly one, three and six months) has
    elapsed. Sign-adjusted so positive excess always means "the pick worked".
    Only decile members are tracked here, never the full universe (see
    config.py's data-size budget note).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import DISCLAIMER, load, save
from ..config import MOM_HISTORY_DIR
from ..pretom import calendar as cal
from ..pead.signals import _asof_pos

PICK_HORIZONS_TD = (20, 60, 126)   # ~1 / 3 / 6 months


# ---------------------------------------------------------- sharded history --
def _shard_path(year: int) -> Path:
    return MOM_HISTORY_DIR / f"{year}.json"


def _read_shard(year: int) -> dict:
    p = _shard_path(year)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"year": year, "rows": []}


def _write_shard(year: int, doc: dict) -> None:
    MOM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _shard_path(year).write_text(json.dumps(doc, indent=None, ensure_ascii=False), encoding="utf-8")


def append_history(rows: list[dict], today: date, full: bool | None = None) -> int:
    """Append one composite-history row per scored ticker for `today` to the
    current year's shard. `full` (defaults to True on Fridays) additionally
    stores the five factor scores + contributions, so the "historical factor
    table" (not just the composite line) has weekly resolution without
    paying daily storage for it. Idempotent per (date, ticker)."""
    full = (today.weekday() == 4) if full is None else full
    year = today.year
    doc = _read_shard(year)
    existing = doc.get("rows", [])
    seen = {(r["date"], r["ticker"]) for r in existing}
    today_iso = today.isoformat()
    added = 0
    for r in rows:
        if r.get("composite") is None:
            continue
        key = (today_iso, r["ticker"])
        if key in seen:
            continue
        row = {"date": today_iso, "ticker": r["ticker"], "composite": r["composite"], "state": r["state"]}
        if full:
            row["factor_scores"] = r.get("factor_scores")
            row["contributions"] = r.get("contributions")
        existing.append(row)
        seen.add(key)
        added += 1
    doc["rows"] = existing
    doc["as_of"] = today_iso
    _write_shard(year, doc)
    return added


def series_for(ticker: str, start_year: int | None = None, end_year: int | None = None) -> list[dict]:
    """Composite (+ factor, on weeks it was captured) history for one ticker,
    across all available yearly shards in [start_year, end_year], ascending
    by date. Used by the Stock-detail historical score chart."""
    if not MOM_HISTORY_DIR.exists():
        return []
    years = sorted(int(p.stem) for p in MOM_HISTORY_DIR.glob("*.json") if p.stem.isdigit())
    if start_year is not None:
        years = [y for y in years if y >= start_year]
    if end_year is not None:
        years = [y for y in years if y <= end_year]
    out = []
    for y in years:
        doc = _read_shard(y)
        out.extend(r for r in doc.get("rows", []) if r["ticker"] == ticker)
    out.sort(key=lambda r: r["date"])
    return out


# ------------------------------------------------------------- pick tracker --
def make_pick_rows(scored_rows: list[dict], asof: str) -> list[dict]:
    """Rows for the composite's top/bottom decile members (rows already
    carry side="long"/"short" from `edge.common.assemble(..., "composite")`)."""
    rows = []
    for r in scored_rows:
        if r.get("side") not in ("long", "short"):
            continue
        rows.append({
            "ticker": r["ticker"], "asof": asof, "side": r["side"],
            "composite": r.get("composite"), "rank": r.get("rank"), "pctile": r.get("pctile"),
            "entry_close": None,
            "eval": {str(h): {"evaluated": False, "excess": None, "exit_date": None}
                     for h in PICK_HORIZONS_TD},
        })
    return rows


def append_picks(new_rows: list[dict]) -> int:
    doc = load("picks", {"rows": []})
    rows = doc.get("rows", [])
    seen = {(r["ticker"], r["asof"]) for r in rows}
    added = 0
    for r in new_rows:
        key = (r["ticker"], r["asof"])
        if key not in seen:
            rows.append(r)
            seen.add(key)
            added += 1
    rows.sort(key=lambda r: (r["asof"], r["ticker"]))
    doc.update({"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER, "rows": rows})
    save("picks", doc, indent=None)
    return added


def _close_on(series: pd.Series, d: date) -> float | None:
    i = _asof_pos(series.index, d)
    if i is None:
        return None
    di = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
    return float(series.iloc[i]) if di == d else None


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    """Fill entry closes and any matured horizon's excess return. Sign-
    adjusted: positive excess = the pick worked (short excess is negated)."""
    doc = load("picks", {"rows": []})
    changed = 0
    for r in doc.get("rows", []):
        frame = px.get(r["ticker"])
        close = frame["close"] if frame is not None and not frame.empty else None
        if close is None:
            continue
        asof = date.fromisoformat(r["asof"])
        if r.get("entry_close") is None:
            c0 = _close_on(close, asof)
            if c0 is not None:
                r["entry_close"] = round(c0, 4)
        exit_days = cal.trading_days(asof, today)
        for h in PICK_HORIZONS_TD:
            cell = r["eval"][str(h)]
            if cell["evaluated"] or len(exit_days) <= h:
                continue
            ed = exit_days[h]
            c0 = r.get("entry_close")
            c1, s0, s1 = _close_on(close, ed), _close_on(spy_close, asof), _close_on(spy_close, ed)
            if None in (c0, c1, s0, s1) or c0 <= 0 or s0 <= 0:
                continue
            raw = (c1 / c0 - 1.0) - (s1 / s0 - 1.0)
            cell["excess"] = round(raw if r["side"] == "long" else -raw, 6)
            cell["exit_date"] = ed.isoformat()
            cell["evaluated"] = True
            changed += 1
    if changed:
        doc["as_of"] = today.isoformat()
        save("picks", doc, indent=None)
    return changed


def summarize(rows: list[dict]) -> dict:
    """Per-horizon/side win-rate + avg sign-adjusted excess (evaluated rows)."""
    out = {}
    for h in PICK_HORIZONS_TD:
        cell = {}
        for side in ("long", "short", "all"):
            ev = [r["eval"][str(h)]["excess"] for r in rows
                  if (side == "all" or r["side"] == side)
                  and r["eval"][str(h)]["evaluated"] and r["eval"][str(h)]["excess"] is not None]
            if ev:
                s = pd.Series(ev, dtype=float)
                cell[side] = {"n": int(len(s)), "avg": round(float(s.mean()), 6),
                              "win_rate": round(float((s > 0).mean()), 4)}
        if cell:
            out[str(h)] = cell
    return out

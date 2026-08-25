"""REGIMES history: the full monthly timeline -> "Regime -> Transition ->
New Regime" segments (spec section 4), plus the daily regime JOURNAL (spec
section 31) that Phase 2's accuracy tracking will read.

The journal is sharded by year (data/regimes/journal/<YYYY>.json), append-
only, one row per calendar day — the exact pattern `mom/history.py` uses for
its own sharded composite history, for the same reason: a single flat file
rewritten in full every day becomes a multi-MB rewrite within a year.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import REGIMES_JOURNAL_DIR


def segments(timeline: pd.DataFrame) -> list[dict]:
    """Collapse the month-by-month `declared_regime` column into contiguous
    segments: [{regime, start, end, n_months}, ...], ascending by start date.
    Months before the first regime clears its persistence requirement (see
    classify.py) carry declared_regime=None and are skipped — there is no
    "declared" regime to segment yet, which is the honest state, not a gap
    to paper over."""
    out: list[dict] = []
    cur_regime, cur_start = None, None
    for month, row in timeline.iterrows():
        reg = row["declared_regime"]
        if reg != cur_regime:
            if cur_regime is not None:
                out.append({"regime": cur_regime, "start": cur_start.date().isoformat(),
                           "end": prev_month.date().isoformat(),
                           "n_months": int((prev_month.to_period("M") - cur_start.to_period("M")).n) + 1})
            cur_regime, cur_start = reg, month
        prev_month = month
    if cur_regime is not None:
        out.append({"regime": cur_regime, "start": cur_start.date().isoformat(),
                   "end": prev_month.date().isoformat(),
                   "n_months": int((prev_month.to_period("M") - cur_start.to_period("M")).n) + 1})
    return out


def transitions(segs: list[dict]) -> list[dict]:
    """Adjacent-segment pairs — the literal "Regime -> Transition -> New
    Regime" rows the spec's worked example shows."""
    return [{"from_regime": segs[i]["regime"], "from_end": segs[i]["end"],
            "to_regime": segs[i + 1]["regime"], "to_start": segs[i + 1]["start"]}
           for i in range(len(segs) - 1)]


def timeline_to_records(timeline: pd.DataFrame) -> list[dict]:
    """The full monthly timeline as JSON-safe records (one per month) —
    everything transition.py (Phase 2) will need for empirical base rates."""
    recs = []
    for month, row in timeline.iterrows():
        rec = row.to_dict()
        rec["month"] = month.date().isoformat()
        for k, v in list(rec.items()):
            if isinstance(v, float) and pd.isna(v):
                rec[k] = None
            elif hasattr(v, "item"):
                rec[k] = v.item()
        recs.append(rec)
    return recs


# ------------------------------------------------------------------ journal
def _shard_path(year: int) -> Path:
    return REGIMES_JOURNAL_DIR / f"{year}.json"


def _read_shard(year: int) -> dict:
    p = _shard_path(year)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"year": year, "rows": []}


def _write_shard(year: int, doc: dict) -> None:
    REGIMES_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    _shard_path(year).write_text(json.dumps(doc, indent=None, ensure_ascii=False), encoding="utf-8")


def append_journal(entry: dict, today: date | None = None) -> int:
    """Append today's regime call to the journal (idempotent per date — a
    same-day rerun overwrites today's row rather than duplicating it, since
    REGIMES is not trading-day-gated and could plausibly run twice in a day
    during development or a manual dispatch)."""
    today = today or date.today()
    year = today.year
    doc = _read_shard(year)
    rows = [r for r in doc.get("rows", []) if r.get("date") != today.isoformat()]
    rows.append({"date": today.isoformat(), **entry})
    rows.sort(key=lambda r: r["date"])
    doc["rows"] = rows
    doc["as_of"] = today.isoformat()
    _write_shard(year, doc)
    return len(rows)


def journal_years() -> list[int]:
    if not REGIMES_JOURNAL_DIR.exists():
        return []
    return sorted(int(p.stem) for p in REGIMES_JOURNAL_DIR.glob("*.json") if p.stem.isdigit())


def load_journal(start_year: int | None = None, end_year: int | None = None) -> list[dict]:
    years = journal_years()
    if start_year is not None:
        years = [y for y in years if y >= start_year]
    if end_year is not None:
        years = [y for y in years if y <= end_year]
    out: list[dict] = []
    for y in years:
        out.extend(_read_shard(y).get("rows", []))
    out.sort(key=lambda r: r["date"])
    return out

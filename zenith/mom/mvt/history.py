"""Multivariate Trend history: sharded daily rank/score history, mirroring
mom/history.py's own sharded-by-year pattern exactly (data/mom/history/
<YYYY>.json -> data/mom/mvt/history/<universe>_<YYYY>.json).

This is Phase 3's prerequisite for "rank-evolution-over-time" (spec section
10's "a visualization of how the instrument's relative rank has evolved
over time") -- there is no way to chart history before it starts being
recorded. Wired into mvt/compute.py's run_auto() so it begins accumulating
from the day this ships; a freshly-launched instance's rank-evolution chart
will honestly show a single point, exactly like every other Zenith feature's
day-one history chart, and fills in one point per trading day going forward.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ...config import MOM_MVT_HISTORY_DIR


def _shard_path(universe: str, year: int) -> Path:
    return MOM_MVT_HISTORY_DIR / f"{universe}_{year}.json"


def _read_shard(universe: str, year: int) -> dict:
    p = _shard_path(universe, year)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"universe": universe, "year": year, "rows": []}


def _write_shard(universe: str, year: int, doc: dict) -> None:
    MOM_MVT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _shard_path(universe, year).write_text(json.dumps(doc, indent=None, ensure_ascii=False),
                                           encoding="utf-8")


def append_history(rows: list[dict], universe: str, today: date) -> int:
    """Append one compact row per scored ticker (date, ticker, normalized
    score, rank) to the current year's shard for `universe` ("equities" or
    "etfs"). Deliberately minimal per row (not the full pairwise detail --
    that's already committed daily in equities_latest.json/etfs_latest.json,
    this is ONLY for the historical rank-evolution line) to keep the
    sharded file's daily growth comparable to mom/history.py's own budget.
    Idempotent per (date, ticker)."""
    year = today.year
    doc = _read_shard(universe, year)
    existing = doc.get("rows", [])
    seen = {(r["date"], r["ticker"]) for r in existing}
    today_iso = today.isoformat()

    scored = [r for r in rows if r.get("normalized_score") is not None]
    ranked = sorted(scored, key=lambda r: r["normalized_score"], reverse=True)
    rank_of = {r["ticker"]: i + 1 for i, r in enumerate(ranked)}
    n = len(ranked)

    added = 0
    for r in scored:
        key = (today_iso, r["ticker"])
        if key in seen:
            continue
        rank = rank_of[r["ticker"]]
        existing.append({
            "date": today_iso, "ticker": r["ticker"],
            "normalized_score": r["normalized_score"], "raw_score": r.get("raw_score"),
            "rank": rank, "pctile": round(100.0 * (n - rank) / (n - 1), 2) if n > 1 else 50.0,
        })
        seen.add(key)
        added += 1
    doc["rows"] = existing
    doc["as_of"] = today_iso
    doc["n_universe"] = n
    _write_shard(universe, year, doc)
    return added


def series_for(ticker: str, universe: str, start_year: int | None = None,
               end_year: int | None = None) -> list[dict]:
    """One ticker's rank/score history across as many yearly shards as
    exist, oldest first. Empty on day one -- callers should render that
    honestly (a single point once today's run lands) rather than treating
    it as an error."""
    end_year = end_year or date.today().year
    start_year = start_year or end_year
    out = []
    for y in range(start_year, end_year + 1):
        doc = _read_shard(universe, y)
        out.extend(r for r in doc.get("rows", []) if r["ticker"] == ticker)
    return sorted(out, key=lambda r: r["date"])

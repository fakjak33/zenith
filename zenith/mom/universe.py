"""MOMENTUM universe: Russell 1000 constituents, point-in-time membership
archive, and best-effort sector/industry/market-cap metadata.

Reuses `pretom.universe.russell1000()` for the constituent list itself
(Vanguard VONE -> Wikipedia -> committed-snapshot fallback chain) rather than
re-fetching — that pipeline is already the whole repo's single source of
truth for the index, and every other feature package depends on it too. This
module adds three things PRETOM doesn't: a point-in-time membership archive
(so future historical scores aren't survivorship-biased), a one-off seed of
that archive from this repo's OWN git history, and a metadata cache
(industry + market cap, which neither Vanguard nor Wikipedia carry, and
which `.info` is too slow to pull for the whole index every night).
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import date

from ..config import PROJECT_ROOT
from ..pretom import universe as pretom_universe
from . import load, save

META_TTL_DAYS = 30
META_MAX_PER_RUN = 150


def constituents(max_age_hours: float = 168.0) -> tuple[list[dict], dict]:
    """Current Russell 1000 rows (ticker/name/sector/weight_pct) + status.
    Thin pass-through to `pretom.universe.russell1000()` — MOMENTUM does not
    maintain a second universe pipeline."""
    return pretom_universe.russell1000(max_age_hours=max_age_hours)


# ------------------------------------------------------- membership archive --
def membership_archive(tickers: list[str], today: str | None = None) -> int:
    """Append today's constituent list to the point-in-time archive (a no-op
    if today's date is already recorded). Returns 1 if a new entry was
    written, 0 otherwise. This is what makes the archive genuinely
    point-in-time GOING FORWARD; it does not retroactively fix history."""
    today = today or date.today().isoformat()
    doc = load("membership", {"entries": []})
    entries = doc.get("entries", [])
    if entries and entries[-1].get("date") == today:
        return 0
    entries.append({"date": today, "n": len(tickers), "tickers": sorted(set(tickers))})
    doc["entries"] = entries
    save("membership", doc, indent=None)
    return 1


def membership_asof(as_of: str) -> set[str] | None:
    """The most recently archived constituent set on or before `as_of`
    (YYYY-MM-DD), or None if the archive has no entry that early — callers
    must fall back to TODAY's constituents and label the result
    survivorship-biased (see `mom.SURVIVORSHIP_NOTE` /
    `config.MOM_MEMBERSHIP_START`). Assumes archive entries are sorted
    ascending by date, which every writer here maintains."""
    doc = load("membership", {"entries": []})
    best = None
    for e in doc.get("entries", []):
        if e["date"] <= as_of:
            best = e
        else:
            break
    return set(best["tickers"]) if best else None


def seed_membership_from_git(repo_root=None) -> dict:
    """One-off: recover past Russell 1000 constituent lists from this repo's
    OWN git history of data/pretom/universe.json (the file PRETOM commits on
    every live universe refresh). This is the only free point-in-time R1000
    history available — there is no external source for it — and it only
    reaches back to whenever universe.json was first committed, typically a
    matter of weeks, not years (see config.MOM_MEMBERSHIP_START, set from the
    actual observed range). Merges into the same membership.json the daily
    job appends to; safe to re-run, and never overwrites a live-collected
    entry for a date it already has."""
    repo_root = repo_root or PROJECT_ROOT
    rel = "data/pretom/universe.json"
    log = subprocess.run(
        ["git", "log", "--format=%H|%ad", "--date=short", "--", rel],
        cwd=str(repo_root), capture_output=True, text=True, check=True,
    ).stdout.strip()
    doc = load("membership", {"entries": []})
    have = {e["date"] for e in doc.get("entries", [])}
    added = 0
    for line in log.splitlines():
        if "|" not in line:
            continue
        sha, day = line.split("|", 1)
        if day in have:
            continue
        try:
            blob = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=str(repo_root),
                                  capture_output=True, text=True, check=True).stdout
            payload = json.loads(blob)
            tickers = sorted({r["ticker"] for r in payload.get("rows", []) if r.get("ticker")})
        except Exception:
            continue
        if len(tickers) < 800:      # same sanity floor pretom.universe applies to a live fetch
            continue
        doc.setdefault("entries", []).append({"date": day, "n": len(tickers), "tickers": tickers})
        have.add(day)
        added += 1
    doc["entries"] = sorted(doc.get("entries", []), key=lambda e: e["date"])
    save("membership", doc, indent=None)
    return {"added": added, "total": len(doc["entries"])}


# ------------------------------------------------------------------ metadata --
def _stale(entry: dict | None, today: date, ttl_days: int = META_TTL_DAYS) -> bool:
    if not entry or not entry.get("asof"):
        return True
    try:
        asof = date.fromisoformat(entry["asof"])
    except Exception:
        return True
    return (today - asof).days > ttl_days


def refresh_metadata(tickers: list[str], max_per_run: int = META_MAX_PER_RUN,
                     ttl_days: int = META_TTL_DAYS, sleep: float = 0.1) -> dict:
    """Best-effort sector/industry/market-cap cache. Costs ~1-1.5s/ticker via
    yfinance .info, so only the `max_per_run` stalest tickers are refreshed
    per run — the full R1000 fills in over about a week and then
    self-maintains (edge/info.py's checkpoint pattern). Result is COMMITTED
    (data/mom/meta.json), not just gitignore-cached: CI runners start cold on
    every job, and metadata must not restart from zero every night."""
    today = date.today()
    meta = load("meta", {})
    stale = [t for t in tickers if _stale(meta.get(t), today, ttl_days)]
    todo = stale[:max_per_run]
    if not todo:
        return {"checked": len(tickers), "stale": 0, "refreshed": 0}
    try:
        import yfinance as yf
    except Exception:
        return {"checked": len(tickers), "stale": len(stale), "refreshed": 0,
                "error": "yfinance unavailable"}
    refreshed = 0
    for i, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info or {}
            meta[t] = {
                "name": info.get("longName") or info.get("shortName") or t,
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "mktcap": info.get("marketCap"),
                "asof": today.isoformat(),
            }
            refreshed += 1
        except Exception:
            pass
        if sleep:
            time.sleep(sleep)
        if i and i % 100 == 0:
            save("meta", meta)
    save("meta", meta)
    return {"checked": len(tickers), "stale": len(stale), "refreshed": refreshed}

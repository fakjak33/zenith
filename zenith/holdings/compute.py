"""HOLDINGS orchestrator — fetch, validate, archive, derive.

    python -m zenith.holdings.compute --action auto
    python -m zenith.holdings.compute --action backfill --days 365
    python -m zenith.holdings.compute --action rebuild

`auto` deliberately has no trading-day gate. The fund's own published value
date is the real gate: if the page still shows the date we already have, there
is no new data and the run is a no-op. That handles weekends, US holidays and
a publisher running late with one rule instead of three.

Nothing is written unless it validates. A snapshot that fails leaves the last
good `latest.json` untouched and records why in `status.json`, so the app can
say "data unavailable" instead of showing a broken book as current.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from . import (WINDOWS, archive_day, archive_days, compare, history, load,
               load_day, registry, save, save_funds, snapshot)
from .sources import get_adapter

BACKSCAN_DAYS = 30           # window checked for missing snapshots each run
CHECKPOINT_EVERY = 10        # backfill: flush derived artefacts this often


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seg(status: list[dict], name: str, ok: bool, **kw) -> None:
    status.append({"segment": name, "ok": bool(ok), **kw})


def load_snapshots(fund_key: str) -> list[dict]:
    """Every archived snapshot for a fund, oldest first."""
    days = sorted(archive_days(fund_key))
    out = []
    for d in days:
        s = load_day(fund_key, d)
        if s.get("as_of"):
            out.append(s)
    return out


def derive(fund_key: str, snaps: list[dict] | None = None) -> dict:
    """Rebuild history + changes + latest purely from the archive."""
    snaps = snaps if snaps is not None else load_snapshots(fund_key)
    hist = history.build(fund_key, snaps)
    chg = compare.build_changes(fund_key, snaps, hist, WINDOWS)
    save(fund_key, "history", hist)
    save(fund_key, "changes", chg)

    latest = {}
    if snaps:
        curr = snaps[-1]
        prev = snaps[-2] if len(snaps) > 1 else None
        latest = dict(curr)
        latest["previous_as_of"] = prev.get("as_of") if prev else None
        # Sessions skipped between the two snapshots being compared, so the UI
        # can never present a multi-week move as a one-day change.
        latest["previous_gap_sessions"] = (hist.get("gap_before") or [0])[-1]
        latest["changes"] = compare.diff(prev, curr)
        latest["gaps"] = history.gaps(hist, BACKSCAN_DAYS)
        latest["n_snapshots"] = len(snaps)
        latest["first_snapshot"] = snaps[0].get("as_of")
        latest["caps"] = chg["caps"]
        ev = latest["changes"]
        biggest = max(ev, key=lambda e: abs(e["d_weight"]), default=None)
        latest["summary"] = dict(curr.get("summary", {}))
        latest["summary"].update({
            "n_added": sum(1 for e in ev if e["type"] == "new"),
            "n_increased": sum(1 for e in ev if e["type"] == "increased"),
            "n_reduced": sum(1 for e in ev if e["type"] == "reduced"),
            "n_closed": sum(1 for e in ev if e["type"] == "closed"),
            "n_flipped": sum(1 for e in ev if e["type"] == "flipped"),
            "largest_change": ({"id": biggest["id"], "name": biggest["name"],
                                "d_weight": biggest["d_weight"],
                                "type": biggest["type"]} if biggest else None),
        })
        save(fund_key, "latest", latest)
    return {"n_snapshots": len(snaps), "n_dates": hist["n_dates"],
            "n_positions": hist["n_positions"], "n_events": chg["n_events"],
            "gaps": latest.get("gaps", [])}


def run_fund(fund, status: list[dict]) -> dict:
    """One live fetch cycle for one fund."""
    adapter = get_adapter(fund.adapter)
    known = set(archive_days(fund.key))

    try:
        rows, meta = adapter.fetch(fund)
    except Exception as e:
        _seg(status, f"fetch({fund.key})", False, error=str(e)[:200])
        return {"fund": fund.key, "state": "unavailable",
                "error": str(e)[:200]}
    _seg(status, f"fetch({fund.key})", bool(rows), n=len(rows),
         via=meta.get("via", ""), error=meta.get("error", ""))
    if not rows:
        return {"fund": fund.key, "state": "unavailable",
                "error": meta.get("error", "no rows parsed")}

    snaps = load_snapshots(fund.key)
    prev = snaps[-1] if snaps else None
    snap = snapshot.build(fund.key, rows, meta, previous=prev, live=True)

    if not snapshot.is_ok(snap):
        _seg(status, f"validate({fund.key})", False, error=snapshot.reason(snap))
        return {"fund": fund.key, "state": "unavailable",
                "as_of": snap.get("as_of"), "error": snapshot.reason(snap)}
    _seg(status, f"validate({fund.key})", True, n=snap["n_positions"],
         warnings=snap["quality"]["warnings"])

    if snap["as_of"] in known:
        _seg(status, f"publish({fund.key})", True, note="no new value date",
             as_of=snap["as_of"])
        derived = derive(fund.key, snaps)
        return {"fund": fund.key, "state": "unchanged", "as_of": snap["as_of"],
                **derived}

    archive_day(fund.key, snap["as_of"], snap)
    snaps.append(snap)
    derived = derive(fund.key, snaps)
    _seg(status, f"publish({fund.key})", True, as_of=snap["as_of"],
         n=snap["n_positions"])
    return {"fund": fund.key, "state": "updated", "as_of": snap["as_of"],
            **derived}


def run_backfill(fund, days: int, status: list[dict]) -> dict:
    """Replay archived captures of the fund page through the live parser."""
    from .sources import wayback

    snaps = load_snapshots(fund.key)
    known = {s["as_of"] for s in snaps}
    # Captures already replayed. Skipping by capture day (not value date) is
    # what makes a re-run cheap: each archived page is ~3 MB.
    done_captures = {(s.get("source") or {}).get("capture_day")
                     for s in snaps} - {None}
    added, skipped, failed = 0, 0, 0

    for rows, meta in wayback.replay(fund, days=days, skip_days=done_captures):
        cap = meta.get("capture_day", "?")
        if not rows:
            failed += 1
            continue
        prev = [s for s in snaps
                if (s.get("as_of") or "") < (meta.get("value_date") or "")]
        snap = snapshot.build(fund.key, rows, meta,
                              previous=prev[-1] if prev else None, live=False)
        if not snapshot.is_ok(snap):
            failed += 1
            print(f"[holdings] backfill {cap}: rejected — {snapshot.reason(snap)}")
            continue
        if snap["as_of"] in known:
            skipped += 1
            continue
        archive_day(fund.key, snap["as_of"], snap)
        known.add(snap["as_of"])
        snaps.append(snap)
        snaps.sort(key=lambda s: s["as_of"])
        added += 1
        print(f"[holdings] backfill {cap} -> value date {snap['as_of']} "
              f"({snap['n_positions']} positions)")
        if added % CHECKPOINT_EVERY == 0:
            derive(fund.key, snaps)

    _seg(status, f"backfill({fund.key})", True, n=added, skipped=skipped,
         failed=failed)
    return {"fund": fund.key, "added": added, "skipped": skipped,
            "failed": failed, **derive(fund.key)}


def run(action: str, fund_key: str | None, days: int) -> dict:
    funds = ([registry.get(fund_key)] if fund_key
             else registry.enabled_funds())
    status: list[dict] = []
    results = []

    save_funds(registry.registry_snapshot())

    for fund in funds:
        try:
            if action == "backfill":
                res = run_backfill(fund, days, status)
            elif action == "rebuild":
                res = {"fund": fund.key, "state": "rebuilt",
                       **derive(fund.key)}
                _seg(status, f"rebuild({fund.key})", True,
                     n=res.get("n_dates", 0))
            else:
                res = run_fund(fund, status)
        except Exception as e:
            res = {"fund": fund.key, "state": "error", "error": str(e)[:200]}
            _seg(status, f"run({fund.key})", False, error=str(e)[:200])
        results.append(res)

        hist = load(fund.key, "history", {})
        out = {
            "date": date.today().isoformat(),
            "ran_at": _now(),
            "action": action,
            "fund": fund.key,
            "state": res.get("state", action),
            "as_of": res.get("as_of") or hist.get("as_of"),
            "n_snapshots": res.get("n_snapshots", hist.get("n_dates", 0)),
            "n_positions": res.get("n_positions", hist.get("n_positions", 0)),
            "gaps": res.get("gaps", []),
            "error": res.get("error", ""),
            "segments": [s for s in status if f"({fund.key})" in s["segment"]],
        }
        save(fund.key, "status", out)

    for s in status:
        extra = " ".join(f"{k}={v}" for k, v in s.items()
                         if k not in ("segment", "ok") and v not in ("", [], None))
        print(f"  {'ok ' if s['ok'] else 'ERR'} {s['segment']:<24} {extra}")
    return {"results": results, "segments": status}


def main() -> None:
    ap = argparse.ArgumentParser(description="Zenith HOLDINGS compute")
    ap.add_argument("--action", default="auto",
                    choices=["auto", "backfill", "rebuild"])
    ap.add_argument("--fund", default=None, help="fund key (default: all enabled)")
    ap.add_argument("--days", type=int, default=365,
                    help="backfill look-back window in calendar days")
    args = ap.parse_args()
    run(args.action, args.fund, args.days)


if __name__ == "__main__":
    main()

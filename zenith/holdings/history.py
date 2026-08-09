"""Date-indexed position history — the query surface for every historical view.

Shape (compact on purpose: ~18 positions x ~250 days a year is nothing, but
keeping it flat means the app never has to reshape a list of snapshots on
every rerun):

    {"dates": ["2026-08-06", "2026-08-07"],
     "series": {"TU": {"name": ..., "asset_class": ..., "w": [-1.00, -0.96],
                       "n": [-4.15e9, -3.99e9], "contract": [...]}},
     "nav": [...], "sources": [...]}

`null` in `w` means NOT HELD on that date — deliberately distinct from a held
position sitting at zero, and distinct again from a date with no snapshot at
all (which simply is not in `dates`).

History is DERIVED, always rebuilt from `archive/*.json`. The archive is the
append-only thing; making everything else a pure function of it means a schema
change is a re-run, not a migration, and `auto` can never drift from `rebuild`.

Dates stay ISO strings end to end. They are never handed to pandas as an index
and round-tripped through `to_json` — that is what silently rewrote the fmom
caches to 1970.
"""

from __future__ import annotations

from datetime import date

from . import normalize


def build(fund: str, snapshots: list[dict]) -> dict:
    """Assemble history from chronologically ordered snapshots."""
    snaps = [s for s in snapshots if s.get("as_of")]
    snaps.sort(key=lambda s: s["as_of"])
    dates = [s["as_of"] for s in snaps]
    n = len(dates)

    series: dict[str, dict] = {}
    for i, snap in enumerate(snaps):
        for p in snap.get("positions", []):
            s = series.get(p["id"])
            if s is None:
                s = series[p["id"]] = {
                    "name": p["name"], "asset_class": p["asset_class"],
                    "sub_class": p.get("sub_class", ""),
                    "w": [None] * n, "n": [None] * n, "contract": [None] * n,
                }
            s["w"][i] = p["weight"]
            s["n"][i] = p["notional"]
            s["contract"][i] = p.get("contract") or None
            s["name"] = p["name"]           # keep the most recent display name

    return {
        "fund": fund,
        "as_of": dates[-1] if dates else None,
        "dates": dates,
        # Trading sessions missing immediately before each date. Computed once
        # here so no view ever has to touch the calendar, and so a column that
        # silently spans two months can be labelled as such.
        "gap_before": gap_before(dates),
        "nav": [s.get("nav") for s in snaps],
        "sources": [(s.get("source") or {}).get("kind", "?") for s in snaps],
        "series": series,
        "n_dates": n,
        "n_positions": len(series),
    }


def gap_before(dates: list[str]) -> list[int]:
    """Missing trading sessions immediately preceding each date (0 for the first)."""
    if not dates:
        return []
    try:
        from ..pretom import calendar as cal
    except Exception:
        return [0] * len(dates)
    out = [0]
    for a, b in zip(dates, dates[1:]):
        d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
        out.append(len([d for d in cal.trading_days(d0, d1) if d0 < d < d1]))
    return out


def gaps(history: dict, lookback_days: int = 30) -> list[str]:
    """Trading days in the last `lookback_days` with no stored snapshot.

    A CALENDAR window, not a count of snapshots: the question this answers is
    "has the daily job been keeping up lately", and counting backwards through
    a sparse backfill would instead report months-old holes forever.

    Surfaced in status and drawn as explicit holes in the matrix — never
    interpolated across.
    """
    dates = history.get("dates", [])
    if not dates:
        return []
    try:
        from datetime import timedelta

        from ..pretom import calendar as cal
    except Exception:
        return []
    end = date.fromisoformat(dates[-1])
    start = max(end - timedelta(days=lookback_days),
                date.fromisoformat(dates[0]))
    have = set(dates)
    return [d.isoformat() for d in cal.trading_days(start, end)
            if d.isoformat() not in have]


def series_for(history: dict, pid: str) -> list[dict]:
    """One position's full time series as tidy records (nulls dropped)."""
    s = history.get("series", {}).get(pid)
    if not s:
        return []
    out = []
    for d, w, nv, c in zip(history["dates"], s["w"], s["n"], s.get("contract", [])):
        if w is None:
            continue
        out.append({"d": d, "w": w, "n": nv, "contract": c,
                    "direction": normalize.direction_of(nv)})
    return out


def matrix(history: dict, ids: list[str] | None = None,
           since: str | None = None,
           include_collateral: bool = False) -> list[dict]:
    """Tidy (position x date) records for the time-matrix heatmap.

    Emits a record for every (position, date) pair in range, including dates
    where the position was absent (`w=None`, `state="absent"`), so the chart
    can render "not held" as a deliberate mark rather than a hole that reads
    as missing data.
    """
    dates = history.get("dates", [])
    keep = [i for i, d in enumerate(dates) if not since or d >= since]
    recs: list[dict] = []
    for pid, s in history.get("series", {}).items():
        if ids is not None and pid not in ids:
            continue
        if s["asset_class"] == "collateral" and not include_collateral:
            continue
        for i in keep:
            w = s["w"][i]
            recs.append({
                "id": pid, "name": s["name"], "asset_class": s["asset_class"],
                "sub_class": s.get("sub_class", ""), "d": dates[i],
                "w": w, "n": s["n"][i],
                "state": "absent" if w is None else
                         normalize.direction_of(s["n"][i]),
            })
    return recs


def delta_matrix(history: dict, ids: list[str] | None = None,
                 last_n: int | None = None,
                 include_collateral: bool = False) -> list[dict]:
    """Tidy (position x date) records of DAY-OVER-DAY change.

    Companion to `matrix`: that one shows the level of each position, this one
    shows the flow. `event` distinguishes an opening or closing from an
    ordinary size change so the chart can mark them differently — a position
    going from nothing to 15% of NAV is a different act from one drifting from
    14% to 15%.
    """
    dates = history.get("dates", [])
    if len(dates) < 2:
        return []
    start = max(1, len(dates) - last_n) if last_n else 1
    gap = history.get("gap_before") or [0] * len(dates)
    recs: list[dict] = []
    for pid, s in history.get("series", {}).items():
        if ids is not None and pid not in ids:
            continue
        if s["asset_class"] == "collateral" and not include_collateral:
            continue
        for i in range(start, len(dates)):
            a, b = s["w"][i - 1], s["w"][i]
            na, nb = s["n"][i - 1], s["n"][i]
            if a is None and b is None:
                event, dw, dn = "absent", None, None
            elif a is None:
                event, dw, dn = "new", b, nb
            elif b is None:
                event, dw, dn = "closed", -a, -(na or 0.0)
            else:
                event = "move"
                dw, dn = round(b - a, 6), round((nb or 0.0) - (na or 0.0), 2)
            recs.append({"id": pid, "name": s["name"],
                         "asset_class": s["asset_class"],
                         "d": dates[i], "dw": dw, "dn": dn, "event": event,
                         "w": b, "n": s["n"][i],
                         "gap": gap[i] if i < len(gap) else 0,
                         "spans": ("one session" if (i < len(gap) and not gap[i])
                                   else f"{(gap[i] if i < len(gap) else 0) + 1} "
                                        f"sessions — snapshots missing between")})
    return recs


def rank_ids(history: dict, include_collateral: bool = False) -> list[str]:
    """Positions ordered for display: asset class, then average conviction."""
    order = {"rates": 0, "equity": 1, "fx": 2, "commodity": 3,
             "unclassified": 4, "collateral": 5}
    rows = []
    for pid, s in history.get("series", {}).items():
        if s["asset_class"] == "collateral" and not include_collateral:
            continue
        vals = [abs(w) for w in s["w"] if w is not None]
        rows.append((order.get(s["asset_class"], 9),
                     -(sum(vals) / len(vals) if vals else 0.0), pid))
    rows.sort()
    return [r[2] for r in rows]


def position_stats(history: dict, pid: str) -> dict:
    """Everything the position explorer needs about one position."""
    pts = series_for(history, pid)
    s = history.get("series", {}).get(pid, {})
    if not pts:
        return {"id": pid, "held": False}

    weights = [p["w"] for p in pts]
    cur = pts[-1]
    biggest = max(pts, key=lambda p: abs(p["w"]))
    smallest = min(pts, key=lambda p: abs(p["w"]))

    changes = []
    for a, b in zip(pts, pts[1:]):
        if abs(b["w"] - a["w"]) > 1e-9:
            changes.append({"d": b["d"], "d_weight": round(b["w"] - a["w"], 6),
                            "w": b["w"]})
    flips = [b for a, b in zip(pts, pts[1:]) if (a["w"] > 0) != (b["w"] > 0)]
    rolls = sorted({p["contract"] for p in pts if p.get("contract")})

    all_dates = history.get("dates", [])
    first_i = all_dates.index(pts[0]["d"]) if pts[0]["d"] in all_dates else 0
    absent = sum(1 for w in s.get("w", [])[first_i:] if w is None)

    return {
        "id": pid, "held": True, "name": s.get("name", pid),
        "asset_class": s.get("asset_class", ""),
        "sub_class": s.get("sub_class", ""),
        "current_weight": cur["w"], "current_notional": cur["n"],
        "current_direction": cur["direction"], "current_contract": cur["contract"],
        "as_of": cur["d"],
        "first_seen": pts[0]["d"], "last_seen": pts[-1]["d"],
        "days_observed": len(pts),
        "days_absent_since_first": absent,
        "max_weight": {"d": biggest["d"], "w": biggest["w"]},
        "min_weight": {"d": smallest["d"], "w": smallest["w"]},
        "mean_abs_weight": round(sum(abs(w) for w in weights) / len(weights), 6),
        "last_change": changes[-1] if changes else None,
        "n_changes": len(changes),
        "n_direction_flips": len(flips),
        "contracts_seen": rolls,
        "recent_changes": changes[-10:][::-1],
    }

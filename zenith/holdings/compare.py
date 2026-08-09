"""Snapshot differencing — what changed, by how much, and how much that matters.

Two conventions worth stating because they drive everything downstream:

1. "Increased" and "reduced" describe ABSOLUTE exposure. A short that gets more
   short has *increased* its position, even though its signed weight fell. The
   signed move is kept alongside as `d_weight`, and the UI always labels which
   of the two it is showing.

2. Window rankings compare the latest snapshot against the snapshot at or
   before `today - N days` — they are not a sum of daily events. Summing would
   double-count a position that went out and came back; comparing endpoints
   answers the question the user actually asked ("what has DBMF been
   accumulating over the last month?").
"""

from __future__ import annotations

from datetime import date

from . import normalize

# Weight moves smaller than this are rounding, not decisions (5bp of NAV).
MIN_MOVE = 0.0005
# A position with less than this absolute weight is treated as not held.
FLAT = 0.0002
# Robust saturation cap for the heatmap: this quantile of recent |moves|.
CAP_QUANTILE = 0.90
CAP_FLOOR = 0.005            # never make the scale so tight that noise screams
CAP_LOOKBACK_DAYS = 60


def _by_id(snap: dict) -> dict[str, dict]:
    return {p["id"]: p for p in snap.get("positions", [])}


def missing_trading_days(a: str, b: str) -> int:
    """Trading days strictly between two ISO dates (0 = consecutive sessions)."""
    try:
        from ..pretom import calendar as cal
        d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
        return max(0, len([d for d in cal.trading_days(d0, d1)
                           if d0 < d < d1]))
    except Exception:
        return 0


def classify_move(w_from: float | None, w_to: float | None) -> str:
    """One of the CHANGE_TYPES for a single position between two snapshots."""
    held_before = w_from is not None and abs(w_from) > FLAT
    held_after = w_to is not None and abs(w_to) > FLAT
    if not held_before and held_after:
        return "new"
    if held_before and not held_after:
        return "closed"
    if not held_before and not held_after:
        return "held"
    if (w_from > 0) != (w_to > 0):
        return "flipped"
    delta = abs(w_to) - abs(w_from)
    if delta > MIN_MOVE:
        return "increased"
    if delta < -MIN_MOVE:
        return "reduced"
    return "held"


def diff(prev: dict | None, curr: dict, include_held: bool = False) -> list[dict]:
    """Position-level change events between two snapshots."""
    if not curr:
        return []
    old = _by_id(prev) if prev else {}
    new = _by_id(curr)
    a_date = (prev or {}).get("as_of")
    b_date = curr.get("as_of")
    gap = missing_trading_days(a_date, b_date) if (a_date and b_date) else 0

    events: list[dict] = []
    for pid in sorted(set(old) | set(new), key=lambda k: k):
        p_old, p_new = old.get(pid), new.get(pid)
        ref = p_new or p_old
        w_from = p_old["weight"] if p_old else None
        w_to = p_new["weight"] if p_new else None
        kind = classify_move(w_from, w_to)
        if kind == "held" and not include_held:
            continue
        f, t = (w_from or 0.0), (w_to or 0.0)
        n_from = (p_old or {}).get("notional") or 0.0
        n_to = (p_new or {}).get("notional") or 0.0
        pct = ((abs(t) / abs(f) - 1.0) if abs(f) > FLAT else None)
        events.append({
            "from": a_date, "to": b_date, "gap_days": gap,
            "id": pid, "name": ref["name"], "asset_class": ref["asset_class"],
            "sub_class": ref["sub_class"],
            "type": kind,
            "w_from": round(f, 6), "w_to": round(t, 6),
            "d_weight": round(t - f, 6),
            "d_abs_weight": round(abs(t) - abs(f), 6),
            "n_from": round(n_from, 2), "n_to": round(n_to, 2),
            "d_notional": round(n_to - n_from, 2),
            "pct_change": round(pct, 4) if pct is not None else None,
            "direction_from": normalize.direction_of(n_from) if p_old else None,
            "direction_to": normalize.direction_of(n_to) if p_new else None,
            "rolled": bool(p_old and p_new
                           and p_old.get("contract") and p_new.get("contract")
                           and p_old["contract"] != p_new["contract"]),
        })
    events.sort(key=lambda e: -abs(e["d_weight"]))
    return events


def _series_at(history: dict, target: str) -> tuple[str | None, dict]:
    """Weights as at the latest history date on or before `target`."""
    dates = history.get("dates", [])
    idx = None
    for i, d in enumerate(dates):
        if d <= target:
            idx = i
        else:
            break
    if idx is None:
        return None, {}
    out = {}
    for pid, s in history.get("series", {}).items():
        w = s["w"][idx] if idx < len(s["w"]) else None
        if w is not None:
            out[pid] = {"id": pid, "name": s["name"],
                        "asset_class": s["asset_class"],
                        "sub_class": s.get("sub_class", ""),
                        "weight": w,
                        "notional": (s["n"][idx] if idx < len(s["n"]) else None) or 0.0}
    return dates[idx], out


def window_changes(history: dict, days: int,
                   include_collateral: bool = False) -> dict:
    """Endpoint-to-endpoint change over the last `days` calendar days."""
    dates = history.get("dates", [])
    if len(dates) < 2:
        return {"from": None, "to": dates[-1] if dates else None, "days": days,
                "increases": [], "decreases": [], "new": [], "closed": [],
                "n_snapshots": len(dates)}
    end = dates[-1]
    target = (date.fromisoformat(end) - _timedelta(days)).isoformat()
    start, old = _series_at(history, target)
    if start is None or start == end:
        start, old = dates[0], _series_at(history, dates[0])[1]
    _, new = _series_at(history, end)

    rows = []
    for pid in set(old) | set(new):
        ref = new.get(pid) or old[pid]
        if ref["asset_class"] == "collateral" and not include_collateral:
            continue
        w_from = old.get(pid, {}).get("weight")
        w_to = new.get(pid, {}).get("weight")
        kind = classify_move(w_from, w_to)
        f, t = (w_from or 0.0), (w_to or 0.0)
        pct = ((abs(t) / abs(f) - 1.0) if abs(f) > FLAT else None)
        rows.append({"id": pid, "name": ref["name"],
                     "asset_class": ref["asset_class"],
                     "sub_class": ref.get("sub_class", ""),
                     "type": kind, "w_from": round(f, 6), "w_to": round(t, 6),
                     "d_weight": round(t - f, 6),
                     "d_abs_weight": round(abs(t) - abs(f), 6),
                     "pct_change": round(pct, 4) if pct is not None else None})

    moved = [r for r in rows if r["type"] in ("increased", "reduced", "flipped")]
    actual = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return {
        "from": start, "to": end, "days": days,
        # What the caller asked for vs what the archive could actually supply.
        # With a sparse history several windows can resolve to the same pair,
        # and the UI must not present that as, say, a one-day move.
        "requested_days": days,
        "actual_days": actual,
        "exact": actual <= days,
        "n_snapshots": sum(1 for d in dates if start <= d <= end),
        "increases": sorted([r for r in moved if r["d_abs_weight"] > 0],
                            key=lambda r: -r["d_abs_weight"])[:12],
        "decreases": sorted([r for r in moved if r["d_abs_weight"] < 0],
                            key=lambda r: r["d_abs_weight"])[:12],
        "new": sorted([r for r in rows if r["type"] == "new"],
                      key=lambda r: -abs(r["w_to"])),
        "closed": sorted([r for r in rows if r["type"] == "closed"],
                         key=lambda r: -abs(r["w_from"])),
    }


def _timedelta(days: int):
    from datetime import timedelta
    return timedelta(days=days)


def change_cap(events: list[dict], key: str = "d_weight") -> float:
    """Saturation point for the heatmap colour scale.

    Anchored to this fund's OWN recent behaviour rather than a fixed constant,
    so "large" means large for DBMF — a book that shifts a few points of NAV a
    day should not look identical to one that shifts twenty.
    """
    mags = sorted(abs(e.get(key) or 0.0) for e in events
                  if e.get("type") not in ("new", "closed"))
    mags = [m for m in mags if m > 0]
    if not mags:
        return CAP_FLOOR
    i = min(len(mags) - 1, int(round(CAP_QUANTILE * (len(mags) - 1))))
    return max(CAP_FLOOR, round(mags[i], 6))


def build_changes(fund: str, snapshots: list[dict], history: dict,
                  windows: dict[str, int]) -> dict:
    """The committed `changes.json`: per-day events + window rankings + caps."""
    events: list[dict] = []
    for prev, curr in zip(snapshots, snapshots[1:]):
        events.extend(diff(prev, curr))

    end = history.get("dates", [None])[-1]
    recent = events
    if end:
        cutoff = (date.fromisoformat(end) - _timedelta(CAP_LOOKBACK_DAYS)).isoformat()
        recent = [e for e in events if (e.get("to") or "") >= cutoff] or events

    return {
        "fund": fund,
        "as_of": end,
        "n_events": len(events),
        "caps": {"d_weight": change_cap(recent, "d_weight"),
                 "d_notional": change_cap(recent, "d_notional")},
        "latest": [e for e in events if e.get("to") == end],
        "events": events[-2000:],          # plenty of depth, still a small file
        "rankings": {label: window_changes(history, n)
                     for label, n in windows.items()},
    }

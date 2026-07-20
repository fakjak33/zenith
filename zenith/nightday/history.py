"""NIGHT & DAY pick history: overnight-momentum decile snapshots, +20td eval."""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save
from ..pretom import calendar as cal
from ..pead.signals import _asof_pos

HORIZON_TD = 20


def make_rows(screen: dict, asof: str) -> list[dict]:
    out = []
    for side in ("long", "short"):
        for r in screen.get(side, []):
            out.append({"ticker": r["ticker"], "asof": asof, "side": side,
                        "on_mom": r.get("on_mom"), "entry_close": None,
                        "evaluated": False, "excess": None, "exit_date": None})
    return out


def append_rows(new_rows: list[dict]) -> int:
    hist = load("history", {"rows": []})
    rows = hist.get("rows", [])
    seen = {(r["ticker"], r["asof"]) for r in rows}
    added = 0
    for r in new_rows:
        if (r["ticker"], r["asof"]) not in seen:
            rows.append(r)
            seen.add((r["ticker"], r["asof"]))
            added += 1
    rows.sort(key=lambda r: (r["asof"], r["ticker"]))
    hist.update({"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
                 "rows": rows})
    save("history", hist)
    return added


def _close_on(series, d: date):
    i = _asof_pos(series.index, d)
    if i is None:
        return None
    di = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
    return float(series.iloc[i]) if di == d else None


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    hist = load("history", {"rows": []})
    changed = 0
    for r in hist.get("rows", []):
        if r.get("evaluated"):
            continue
        frame = px.get(r["ticker"])
        close = frame["close"] if frame is not None and not frame.empty else None
        if close is None:
            continue
        asof = date.fromisoformat(r["asof"])
        if r.get("entry_close") is None:
            c0 = _close_on(close, asof)
            if c0 is not None:
                r["entry_close"] = round(c0, 4)
        span = cal.trading_days(asof, today)
        if len(span) <= HORIZON_TD:
            continue
        ed = span[HORIZON_TD]
        c0 = r.get("entry_close")
        c1, s0, s1 = _close_on(close, ed), _close_on(spy_close, asof), _close_on(spy_close, ed)
        if None in (c0, c1, s0, s1) or c0 <= 0 or s0 <= 0:
            continue
        raw = (c1 / c0 - 1.0) - (s1 / s0 - 1.0)
        r["excess"] = round(raw if r["side"] == "long" else -raw, 6)
        r["exit_date"] = ed.isoformat()
        r["evaluated"] = True
        changed += 1
    if changed:
        hist["as_of"] = today.isoformat()
        save("history", hist)
    return changed


def summarize(rows: list[dict]) -> dict:
    out = {}
    for side in ("long", "short", "all"):
        ev = [r["excess"] for r in rows if r.get("evaluated")
              and (side == "all" or r["side"] == side)
              and isinstance(r.get("excess"), (int, float))]
        if ev:
            s = pd.Series(ev, dtype=float)
            out[side] = {"n": int(len(s)), "avg": round(float(s.mean()), 6),
                         "win_rate": round(float((s > 0).mean()), 4)}
    return out

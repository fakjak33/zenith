"""Edge-screen pick history: append-only decile snapshots, horizon evaluation.

One row per (screen, ticker, asof_date) for the long/short decile members of
each screen. Entry = the close on asof_date; the excess-vs-SPY return fills once
the screen's horizon (5-20 trading days) has elapsed. Sign-adjusted so positive
= the pick worked (longs up vs SPY, shorts down vs SPY). This is the live "does
this screen actually work on our data" tracker.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save
from ..pretom import calendar as cal
from ..pead.signals import _asof_pos

SURVIVORSHIP_NOTE = ("Snapshots use the Russell 1000 as constituted each run; "
                     "delisted names are absent. Screens start live (no "
                     "backfill), so the tracker is thin until history accrues.")


def make_rows(screen: str, result: dict, asof: str, horizon_td: int) -> list[dict]:
    """Rows for the long+short decile members of one screen result."""
    rows = []
    for side_key in ("long", "short"):
        for r in result.get(side_key, []):
            rows.append({
                "screen": screen, "ticker": r["ticker"], "asof": asof,
                "side": r.get("side", side_key), "horizon_td": horizon_td,
                "rank": r.get("rank"), "pctile": r.get("pctile"),
                "entry_close": None, "evaluated": False, "excess": None,
                "exit_date": None,
            })
    return rows


def append_rows(new_rows: list[dict]) -> int:
    hist = load("history", {"rows": []})
    rows = hist.get("rows", [])
    seen = {(r["screen"], r["ticker"], r["asof"]) for r in rows}
    added = 0
    for r in new_rows:
        key = (r["screen"], r["ticker"], r["asof"])
        if key not in seen:
            rows.append(r)
            seen.add(key)
            added += 1
    rows.sort(key=lambda r: (r["asof"], r["screen"], r["ticker"]))
    hist.update({"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
                 "survivorship_note": SURVIVORSHIP_NOTE, "rows": rows})
    save("history", hist)
    return added


def _close_on(series, d: date):
    i = _asof_pos(series.index, d)
    if i is None:
        return None
    di = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
    return float(series.iloc[i]) if di == d else None


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    """Fill entry closes and matured horizon excess returns. Sign-adjusted:
    positive excess = the pick worked (short returns are negated)."""
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
        exit_day = cal.trading_days(asof, today)
        if len(exit_day) <= r["horizon_td"]:
            continue                      # horizon not reached yet
        ed = exit_day[r["horizon_td"]]
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
    """Per-screen/side win-rate + avg sign-adjusted excess (evaluated rows)."""
    out = {}
    from .common import finite
    for screen in sorted({r["screen"] for r in rows}):
        cell = {}
        for side in ("long", "short", "all"):
            ev = [r["excess"] for r in rows
                  if r["screen"] == screen and r.get("evaluated")
                  and (side == "all" or r["side"] == side) and finite(r.get("excess"))]
            if ev:
                s = pd.Series(ev, dtype=float)
                cell[side] = {"n": int(len(s)), "avg": round(float(s.mean()), 6),
                              "win_rate": round(float((s > 0).mean()), 4)}
        if cell:
            out[screen] = cell
    return out

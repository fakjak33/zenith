"""PEAD pick history: append-only rows, multi-horizon lazy evaluation.

One row per (ticker, report_date). A row is appended when its signal sheet is
finalized (close of the reaction day); the entry price (next session's open)
and each exit horizon fill on LATER runs as the bars print — never fabricated.

Horizons (trading days after the reaction day R):
    h5 / h20 / h60   close of R+5 / R+20 / R+60
    next_earn        close of the trading day BEFORE the next report's own
                     reaction day — out before the next announcement, having
                     captured the pre-announcement drift concentration
                     (Bernard & Thomas 1990).
A pick is ACTIVE until its h60 or next_earn exit, whichever comes first.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save
from ..pretom import calendar as cal
from .signals import _asof_pos, reaction_day, prev_trading_day

HORIZONS = {"h5": 5, "h20": 20, "h60": 60}
DELIST_AFTER_TD = 10        # no bars for this many trading days -> close out

SURVIVORSHIP_NOTE = ("Backfilled picks use CURRENT Russell 1000 membership; "
                     "delisted names are absent, which can flatter both sides.")


def make_row(sig: dict, backfilled: bool = False) -> dict:
    """History row from one eligible signal-sheet row."""
    comp = sig.get("composite") or 0.0
    return {
        "ticker": sig["ticker"], "report_date": sig["report_date"],
        "reaction_day": sig["reaction_day"], "side": sig["side"],
        "composite": comp,
        "score_tercile": ("high" if comp >= 60 else "mid" if comp >= 45 else "low"),
        "sue_source": sig.get("sue_source", ""),
        "cap_tier": (sig.get("context") or {}).get("cap_tier"),
        "backfilled": backfilled,
        "entry_date": None, "entry_open": None, "entry_proxy": False,
        "delisted": False,
        "next_earn_date": (sig.get("context") or {}).get("next_earn_date"),
        "horizons": {h: {"evaluated": False} for h in (*HORIZONS, "next_earn")},
        "active": True,
    }


def append_rows(new_rows: list[dict]) -> int:
    """Append, skipping (ticker, report_date) already present. Idempotent under
    cron retries / backscans. Returns the number actually appended."""
    hist = load("history", {"rows": []})
    rows = hist.get("rows", [])
    seen = {(r["ticker"], r["report_date"]) for r in rows}
    added = 0
    for r in new_rows:
        if (r["ticker"], r["report_date"]) not in seen:
            rows.append(r)
            seen.add((r["ticker"], r["report_date"]))
            added += 1
    rows.sort(key=lambda r: (r["report_date"], r["ticker"]))
    hist.update({"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
                 "survivorship_note": SURVIVORSHIP_NOTE, "rows": rows})
    save("history", hist)
    return added


def refresh_next_earnings(upcoming: dict[str, str | None]) -> int:
    """Update next_earn_date on still-active rows from {ticker: iso date}."""
    hist = load("history", {"rows": []})
    changed = 0
    for r in hist.get("rows", []):
        if not r.get("active"):
            continue
        nxt = upcoming.get(r["ticker"])
        if nxt and nxt > r["report_date"] and nxt != r.get("next_earn_date"):
            r["next_earn_date"] = nxt
            changed += 1
    if changed:
        save("history", hist)
    return changed


def _close_on(series: pd.Series, d: date) -> float | None:
    i = _asof_pos(series.index, d)
    if i is None:
        return None
    d_i = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
    return float(series.iloc[i]) if d_i == d else None


def _exit_dates(r: dict) -> dict[str, date | None]:
    """Scheduled exit date per horizon (rule calendar)."""
    rd = date.fromisoformat(r["reaction_day"])
    tail = cal.trading_days(rd, rd.fromordinal(rd.toordinal() + 120))
    out = {}
    for h, n in HORIZONS.items():
        out[h] = tail[n] if len(tail) > n else None    # tail[0] == R itself
    nxt = r.get("next_earn_date")
    if nxt:
        nxt_reaction = reaction_day(date.fromisoformat(nxt), "")   # assume AH
        out["next_earn"] = prev_trading_day(nxt_reaction)
    else:
        out["next_earn"] = None
    return out


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    """Fill entry opens + every matured horizon; refresh running P&L on active
    rows; apply the delisting close-out. Returns number of horizon fills."""
    hist = load("history", {"rows": []})
    filled = 0
    for r in hist.get("rows", []):
        if all(h.get("evaluated") for h in r["horizons"].values()) and not r.get("active"):
            continue
        frame = px.get(r["ticker"])
        close = frame["close"] if frame is not None and not frame.empty else None
        rd = date.fromisoformat(r["reaction_day"])

        # entry = open of the first session after the reaction day
        if r["entry_open"] is None and frame is not None and not frame.empty:
            e_day = cal.next_trading_day(rd)
            i = _asof_pos(frame.index, e_day)
            if i is not None:
                d_i = (frame.index[i].date() if hasattr(frame.index[i], "date")
                       else frame.index[i])
                if d_i == e_day and "open" in frame.columns:
                    r["entry_date"] = e_day.isoformat()
                    r["entry_open"] = round(float(frame["open"].iloc[i]), 4)
            if r["entry_open"] is None and len(cal.trading_days(rd, today)) > 5:
                base = _close_on(close, rd) if close is not None else None
                if base:
                    r["entry_date"], r["entry_open"] = rd.isoformat(), round(base, 4)
                    r["entry_proxy"] = True

        # delisting: active row with no bars for DELIST_AFTER_TD trading days
        if close is not None and len(close) and r.get("active"):
            last = close.index[-1]
            last_d = last.date() if hasattr(last, "date") else last
            if len(cal.trading_days(last_d, today)) - 1 > DELIST_AFTER_TD:
                r["delisted"] = True

        exits = _exit_dates(r)
        sign = 1.0 if r["side"] == "long" else -1.0
        for h, x_day in exits.items():
            slot = r["horizons"][h]
            if slot.get("evaluated") or r["entry_open"] is None:
                continue
            if r.get("delisted") and close is not None and len(close):
                last = close.index[-1]
                x_day = last.date() if hasattr(last, "date") else last
                slot["delist_exit"] = True
            elif x_day is None or x_day > today or close is None:
                continue
            xc = _close_on(close, x_day) if close is not None else None
            if xc is None and close is not None and len(close):
                i = _asof_pos(close.index, x_day)          # holiday drift: as-of
                xc = float(close.iloc[i]) if i is not None else None
            ec = _close_on(spy_close, x_day)
            e0_i = _asof_pos(spy_close.index, date.fromisoformat(r["entry_date"]))
            if xc is None or ec is None or e0_i is None:
                continue
            spy0 = float(spy_close.iloc[e0_i])
            ret = xc / r["entry_open"] - 1.0
            spy_ret = ec / spy0 - 1.0 if spy0 > 0 else 0.0
            slot.update({"exit_date": x_day.isoformat(),
                         "ret": round(ret, 4),
                         "excess_ret": round(sign * (ret - spy_ret), 4),
                         "evaluated": True})
            filled += 1

        # active = before the earlier of h60 / next_earn exits
        ends = [d for d in (exits.get("h60"), exits.get("next_earn")) if d]
        r["active"] = (not r.get("delisted")
                       and (not ends or today <= min(ends))
                       and not all(r["horizons"][h].get("evaluated")
                                   for h in ("h60", "next_earn")
                                   if exits.get(h) is not None))

        # running mark for the active book
        if r["active"] and r["entry_open"] and close is not None and len(close):
            lc = float(close.iloc[-1])
            last = close.index[-1]
            last_d = last.date() if hasattr(last, "date") else last
            i0 = _asof_pos(spy_close.index, date.fromisoformat(r["entry_date"]))
            li = _asof_pos(spy_close.index, last_d)
            ret = lc / r["entry_open"] - 1.0
            spy_ret = (float(spy_close.iloc[li]) / float(spy_close.iloc[i0]) - 1.0
                       if i0 is not None and li is not None else 0.0)
            r["running"] = {"last_close_date": last_d.isoformat(),
                            "ret": round(ret, 4),
                            "excess_ret": round((1.0 if r["side"] == "long" else -1.0)
                                                * (ret - spy_ret), 4)}
        elif not r["active"]:
            r.pop("running", None)

    hist["as_of"] = today.isoformat()
    save("history", hist)
    return filled


def active_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("active")]


def summarize(rows: list[dict]) -> dict:
    """Descriptive stats per horizon x side x score tercile (+ all-tercile),
    excess-return convention: positive = the pick worked (sign-adjusted)."""
    out = {}
    for h in (*HORIZONS, "next_earn"):
        for side in ("long", "short"):
            for terc in ("all", "high", "mid", "low"):
                vals = [r["horizons"][h]["excess_ret"] for r in rows
                        if r["side"] == side and r["horizons"][h].get("evaluated")
                        and r["horizons"][h].get("excess_ret") is not None
                        and (terc == "all" or r.get("score_tercile") == terc)]
                if not vals:
                    continue
                s = pd.Series(vals)
                out[f"{h}|{side}|{terc}"] = {
                    "n": int(len(s)),
                    "win_rate": round(float((s > 0).mean()), 4),
                    "avg": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "std": round(float(s.std(ddof=0)), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                }
    n_live = sum(1 for r in rows if not r.get("backfilled"))
    return {"cells": out, "n_rows": len(rows), "n_live": n_live,
            "n_active": len(active_rows(rows))}


def drift_curve(rows: list[dict], px: dict, spy_close: pd.Series,
                max_td: int = 60) -> list[dict]:
    """The classic PEAD staircase: average cumulative sign-adjusted excess
    return by trading day since the reaction day, long vs short cohorts."""
    paths = {"long": [], "short": []}
    for r in rows:
        frame = px.get(r["ticker"])
        if frame is None or frame.empty or not r.get("entry_open"):
            continue
        close = frame["close"]
        rd = date.fromisoformat(r["reaction_day"])
        i0 = _asof_pos(close.index, rd)
        s0 = _asof_pos(spy_close.index, rd)
        if i0 is None or s0 is None:
            continue
        base, sbase = float(close.iloc[i0]), float(spy_close.iloc[s0])
        if base <= 0 or sbase <= 0:
            continue
        sign = 1.0 if r["side"] == "long" else -1.0
        path = []
        for k in range(1, max_td + 1):
            if i0 + k >= len(close) or s0 + k >= len(spy_close):
                break
            ret = float(close.iloc[i0 + k]) / base - 1.0
            spy_r = float(spy_close.iloc[s0 + k]) / sbase - 1.0
            path.append(sign * (ret - spy_r))
        if len(path) >= 5:
            paths[r["side"]].append(path)
    out = []
    for k in range(1, max_td + 1):
        pt = {"td": k}
        for side in ("long", "short"):
            vals = [p[k - 1] for p in paths[side] if len(p) >= k]
            if vals:
                pt[f"{side}_avg_excess"] = round(sum(vals) / len(vals), 5)
                pt[f"n_{side}"] = len(vals)
        if len(pt) > 1:
            out.append(pt)
    return out

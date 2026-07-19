"""Earnings-announcement premium (EAP) — the calendar long side of PEAD.

The evidence: stocks earn abnormal returns around their SCHEDULED earnings
announcement dates, before anyone knows the number.

  * Frazzini & Lamont (2007, NBER w13090): long every stock expected to
    announce within the coming month, short every stock not expected to,
    earns over 60 bp per month (strategy variants 7-18%/yr), strongly tied
    to the announcement volume surge and small-investor attention.
  * Savor & Wilson (2016, Journal of Finance): 1974-2009, an equal-weighted
    announcers-minus-non-announcers portfolio earned 0.39% per WEEK (CAPM
    alpha 0.38%/wk, ~20%/yr annualized) - framed as compensation for the
    systematic risk of announcement news.
  * Barber, De George, Lehavy & Trueman (2013, JFE): the premium exists
    around the globe.

Unlike PEAD's drift rows (conditional on the surprise), EAP rows are
UNCONDITIONAL: every liquid Russell 1000 reporter is tracked long through two
windows anchored on its reaction session R (the first session that can trade
the news):

    pre      entry close R-5td  ->  close R-1td   (the run-up, no event risk)
    through  entry close R-5td  ->  close of R    (run-up + the announcement)

Pure math + this feature's own append-only history I/O; all network access
stays in earnings.py / compute.py.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save
from ..pretom import calendar as cal
from .signals import (MIN_ADV_USD, MIN_PRICE, _asof_pos, adv_usd, cap_tier,
                      prev_trading_day, reaction_day)

ENTRY_TD_BEFORE = 5          # entry = close 5 trading days before reaction day
WINDOWS = ("pre", "through")

SURVIVORSHIP_NOTE = ("Backfilled announcements use CURRENT Russell 1000 "
                     "membership; delisted names are absent. Sample starts "
                     "with PEAD's archive (~mid-2024) - decades shorter than "
                     "the published evidence.")


def windows_for(report_date: date, time_str: str) -> dict:
    """Window anchor dates for one scheduled announcement."""
    r = reaction_day(report_date, time_str)
    entry = r
    for _ in range(ENTRY_TD_BEFORE):
        entry = prev_trading_day(entry)
    return {"reaction_day": r, "entry_day": entry,
            "pre_exit": prev_trading_day(r), "through_exit": r}


def _close_on(series: pd.Series | None, d: date) -> float | None:
    """Close exactly on d (None until that bar prints)."""
    if series is None or len(series) == 0:
        return None
    i = _asof_pos(series.index, d)
    if i is None:
        return None
    d_i = series.index[i].date() if hasattr(series.index[i], "date") else series.index[i]
    return float(series.iloc[i]) if d_i == d else None


def _window_leg(close: pd.Series | None, spy: pd.Series | None,
                entry_day: date, exit_day: date) -> dict:
    """One window's {ret, spy_ret, excess, evaluated} - evaluated only once
    both the entry and exit bars have printed for stock AND SPY."""
    c0, c1 = _close_on(close, entry_day), _close_on(close, exit_day)
    s0, s1 = _close_on(spy, entry_day), _close_on(spy, exit_day)
    if None in (c0, c1, s0, s1) or c0 <= 0 or s0 <= 0:
        return {"evaluated": False}
    ret = c1 / c0 - 1.0
    spy_ret = s1 / s0 - 1.0
    return {"evaluated": True, "ret": round(ret, 6),
            "spy_ret": round(spy_ret, 6), "excess": round(ret - spy_ret, 6)}


def make_row(rep: dict, backfilled: bool = False) -> dict:
    """Unevaluated EAP row from one reporter record ({ticker, name,
    report_date, time, mktcap})."""
    w = windows_for(date.fromisoformat(rep["report_date"]), rep.get("time", ""))
    return {
        "ticker": rep["ticker"], "name": rep.get("name", ""),
        "report_date": rep["report_date"],
        "reaction_day": w["reaction_day"].isoformat(),
        "entry_date": w["entry_day"].isoformat(),
        "pre_exit_date": w["pre_exit"].isoformat(),
        "through_exit_date": w["through_exit"].isoformat(),
        "cap_tier": cap_tier(rep.get("mktcap")), "mktcap": rep.get("mktcap"),
        "backfilled": backfilled, "gate": None,
        "pre": {"evaluated": False}, "through": {"evaluated": False},
    }


def evaluate_row(row: dict, frame: pd.DataFrame | None,
                 spy_close: pd.Series | None) -> bool:
    """Fill unevaluated windows in place once bars exist; apply liquidity
    gates at the entry date. Returns True when anything changed."""
    changed = False
    close = frame["close"] if frame is not None and not frame.empty else None
    entry = date.fromisoformat(row["entry_date"])
    if row.get("gate") is None and frame is not None and not frame.empty:
        px_entry = _close_on(close, entry)
        liq = adv_usd(frame, entry)
        gate = ("penny" if px_entry is not None and px_entry <= MIN_PRICE else
                "illiquid" if liq is not None and liq < MIN_ADV_USD else None)
        if gate:
            row["gate"] = gate
            changed = True
    exits = {"pre": row["pre_exit_date"], "through": row["through_exit_date"]}
    for wname, exit_iso in exits.items():
        if row[wname].get("evaluated"):
            continue
        leg = _window_leg(close, spy_close, entry, date.fromisoformat(exit_iso))
        if leg.get("evaluated"):
            row[wname] = leg
            changed = True
    return changed


# --- append-only history (mirrors history.py conventions) --------------------

def append_rows(new_rows: list[dict]) -> int:
    """Append, skipping (ticker, report_date) already present. Idempotent."""
    hist = load("eap_history", {"rows": []})
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
    save("eap_history", hist)
    return added


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    """Evaluate every row with an unevaluated window whose exit date has
    passed. Returns the number of rows changed."""
    hist = load("eap_history", {"rows": []})
    changed = 0
    for r in hist.get("rows", []):
        if r["pre"].get("evaluated") and r["through"].get("evaluated"):
            continue
        if date.fromisoformat(r["through_exit_date"]) > today:
            continue
        if evaluate_row(r, px.get(r["ticker"]), spy_close):
            changed += 1
    if changed:
        hist["as_of"] = today.isoformat()
        save("eap_history", hist)
    return changed


def pending_tickers(today: date) -> list[str]:
    """Tickers with windows still awaiting evaluation (for the price pull)."""
    hist = load("eap_history", {"rows": []})
    return sorted({r["ticker"] for r in hist.get("rows", [])
                   if not (r["pre"].get("evaluated")
                           and r["through"].get("evaluated"))
                   and date.fromisoformat(r["entry_date"]) <= today})


# --- aggregation -------------------------------------------------------------

def _bucket(vals: list[float]) -> dict | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    s = pd.Series(vals, dtype=float)
    return {"n": int(len(s)), "avg": round(float(s.mean()), 6),
            "median": round(float(s.median()), 6),
            "win_rate": round(float((s > 0).mean()), 4)}


def summarize(rows: list[dict]) -> dict:
    """Tracked-premium aggregates for the view: per window overall, by cap
    tier, and a monthly average-excess series (through window)."""
    ok = [r for r in rows if r.get("gate") is None]
    out: dict = {"n_rows": len(rows), "n_gated": len(rows) - len(ok)}
    for w in WINDOWS:
        ev = [r for r in ok if r[w].get("evaluated")]
        out[w] = {
            "overall": _bucket([r[w]["excess"] for r in ev]),
            "by_cap_tier": {tier: _bucket([r[w]["excess"] for r in ev
                                           if r.get("cap_tier") == tier])
                            for tier in ("large", "mid", "small")},
        }
    monthly: dict[str, list[float]] = {}
    for r in ok:
        if r["through"].get("evaluated"):
            monthly.setdefault(r["report_date"][:7], []).append(r["through"]["excess"])
    out["monthly"] = [{"month": m, **(_bucket(v) or {})}
                      for m, v in sorted(monthly.items())]
    return out


def past_announcement_avg(rows: list[dict], ticker: str) -> dict | None:
    """This name's own tracked history: mean through-window excess + count
    (Frazzini & Lamont: the premium concentrates in names whose past
    announcements drew the most attention)."""
    ev = [r["through"]["excess"] for r in rows
          if r["ticker"] == ticker and r.get("gate") is None
          and r["through"].get("evaluated")]
    if not ev:
        return None
    b = _bucket(ev)
    return {"avg_excess": b["avg"], "n": b["n"]} if b else None


def build_upcoming(scheduled: list[dict], hist_rows: list[dict],
                   today: date) -> list[dict]:
    """View-ready upcoming-reporters list from the forward calendar scan."""
    out = []
    for rep in scheduled:
        try:
            w = windows_for(date.fromisoformat(rep["report_date"]),
                            rep.get("time", ""))
        except ValueError:
            continue
        if w["reaction_day"] < today:
            continue
        past = past_announcement_avg(hist_rows, rep["ticker"])
        out.append({
            "ticker": rep["ticker"], "name": rep.get("name", ""),
            "report_date": rep["report_date"], "time": rep.get("time", ""),
            "reaction_day": w["reaction_day"].isoformat(),
            "entry_date": w["entry_day"].isoformat(),
            "eps_consensus": rep.get("eps_consensus"),
            "n_estimates": rep.get("n_estimates"),
            "cap_tier": cap_tier(rep.get("mktcap")), "mktcap": rep.get("mktcap"),
            "past_avg_excess": past["avg_excess"] if past else None,
            "past_n": past["n"] if past else 0,
            "in_window": w["entry_day"] <= today,
        })
    out.sort(key=lambda r: (r["report_date"], -(r["mktcap"] or 0)))
    return out

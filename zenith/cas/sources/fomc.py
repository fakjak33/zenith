"""FOMC cycle time (Cieslak-Morse-Vissing-Jorgensen) — pure functions.

CMVJ (2019, Journal of Finance, "Stock Returns over the FOMC Cycle"): since
1994 the U.S. equity premium was earned entirely in EVEN weeks of FOMC cycle
time — week 0 starting one trading day before a scheduled announcement, then
weeks 2, 4, 6. Even-week day t excess returns averaged ~12bp above odd-week
days, 1994-2016, and the pattern survived their out-of-sample checks through
2016.

The honest update this module exists to show: Uppal ("Does the FOMC Cycle
Still Drive Stock Returns?") finds the even-week pattern is NOT robust once
the sample runs to 2023 (fragile as early as post-2004, gone 2017-2023), and
the biweekly-board-meeting leak mechanism ended in 2004. The related
pre-FOMC announcement drift (Lucca & Moench 2015) has also concentrated into
press-conference meetings only. So: an evidence exhibit + live cycle clock,
NOT a signal.

Meeting dates come from the committed data/cas/fomc_dates.json (1994-2027,
sources stamped inside; scheduled meetings anchor cycle time).
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from ...config import CAS_DIR

_DATES_PATH = CAS_DIR / "fomc_dates.json"

ERAS = (("1994-2016 (CMVJ sample)", "1994-01-01", "2016-12-31"),
        ("2017-now (out of sample)", "2017-01-01", "2099-01-01"))


def load_meetings() -> list[dict]:
    """All meetings [{date, scheduled}] from the committed artifact."""
    try:
        obj = json.loads(_DATES_PATH.read_text(encoding="utf-8"))
        return obj.get("meetings", [])
    except Exception:
        return []


def scheduled_days() -> list[date]:
    return sorted(date.fromisoformat(m["date"]) for m in load_meetings()
                  if m.get("scheduled"))


def next_meeting(d: date, meetings: list[date] | None = None) -> date | None:
    ms = meetings if meetings is not None else scheduled_days()
    future = [m for m in ms if m > d]
    return future[0] if future else None


def prev_meeting(d: date, meetings: list[date] | None = None) -> date | None:
    ms = meetings if meetings is not None else scheduled_days()
    past = [m for m in ms if m <= d]
    return past[-1] if past else None


def cycle_positions(index: pd.DatetimeIndex,
                    meetings: list[date]) -> pd.Series:
    """CMVJ cycle day for every session in a trading-day index: 0 on the
    scheduled announcement day, counted in the index's own business days.
    Days before the first meeting are NaN."""
    mset = {pd.Timestamp(m) for m in meetings}
    pos = np.full(len(index), np.nan)
    last = None
    for i, ts in enumerate(index):
        if ts.normalize() in mset:
            last = i
        if last is not None:
            pos[i] = i - last
    return pd.Series(pos, index=index)


def week_of(cycle_day: float) -> int | None:
    """CMVJ week number: week 0 starts one day BEFORE the announcement, so
    days -1..3 are week 0, 4..8 week 1, ... (here cycle_day >= 0, and the day
    before the NEXT announcement belongs to that next cycle's week 0 — that
    single -1 day is handled by the caller shifting against next meetings)."""
    if cycle_day is None or (isinstance(cycle_day, float) and np.isnan(cycle_day)):
        return None
    return int((cycle_day + 1) // 5)


def even_week_mask(index: pd.DatetimeIndex, meetings: list[date]) -> pd.Series:
    """Boolean per session: does it fall in an EVEN CMVJ cycle week? The day
    immediately before a scheduled announcement is week 0 of the next cycle
    (CMVJ's '-1' day), which this handles explicitly."""
    pos = cycle_positions(index, meetings)
    mset = {pd.Timestamp(m) for m in meetings}
    nxt_is_meeting = pd.Series(
        [index[i + 1].normalize() in mset if i + 1 < len(index) else False
         for i in range(len(index))], index=index)
    week = ((pos + 1) // 5)
    even = (week % 2 == 0)
    even[nxt_is_meeting] = True          # day -1 -> next cycle's week 0
    even[pos.isna() & ~nxt_is_meeting] = False
    return even.astype(bool), pos


def _bucket(daily: pd.Series, min_n: int = 30) -> dict | None:
    r = daily.dropna()
    if len(r) < min_n:
        return None
    sd = float(r.std(ddof=1))
    t = float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else None
    return {"n_days": int(len(r)), "avg_bp": round(float(r.mean()) * 1e4, 2),
            "t_stat": round(t, 2) if t is not None else None}


def era_stats(spy_close: pd.Series, meetings: list[date]) -> dict:
    """Even-week vs odd-week mean daily SPY return by era, plus the pre-FOMC
    (day before announcement) drift by era."""
    close = spy_close.dropna()
    ret = close.pct_change().dropna()
    even, pos = even_week_mask(ret.index, meetings)
    mset = {pd.Timestamp(m) for m in meetings}
    idx = list(ret.index)
    pre_mask = pd.Series(
        [idx[i + 1].normalize() in mset if i + 1 < len(idx) else False
         for i in range(len(idx))], index=ret.index)

    out: dict = {"eras": {}}
    slices = list(ERAS) + [("trailing 24m", None, None)]
    for label, lo, hi in slices:
        if lo is None:
            sub = ret.tail(504)
        else:
            sub = ret.loc[lo:hi]
        e = sub[even.reindex(sub.index).fillna(False)]
        o = sub[~even.reindex(sub.index).fillna(False)]
        rec = {"even": _bucket(e), "odd": _bucket(o), "pre_fomc":
               _bucket(sub[pre_mask.reindex(sub.index).fillna(False)],
                       min_n=10),
               "all": _bucket(sub)}
        both = rec["even"] and rec["odd"]
        rec["even_minus_odd_bp"] = (round(rec["even"]["avg_bp"]
                                          - rec["odd"]["avg_bp"], 2)
                                    if both else None)
        out["eras"][label] = rec
    return out

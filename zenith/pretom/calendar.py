"""NYSE trading calendar + PreTOM state machine (pure functions, no deps).

Trading days are indexed relative to the turn-of-the-month: tau is the last
trading day of the month, tau-1 the one before, tau+1 the first trading day of
the next month. Windows follow Nathan, Suominen & Tasa (2026):

    PreTOM (classic)  [tau-9, tau-4]   six days, the paper's 1980-2025 window
    PreTOM (T+1 span) [tau-9, tau-3]   adds the post-May-2024 marginal day
    Post (reversal)   [tau-3, tau+3]   ~70% of the loser hit reverses here
    TOM (long side)   [tau,   tau+3]   the classic turn-of-the-month window
                                       (Lakonishok & Smidt 1988; Xu & McConnell
                                       2008: all of the market's excess return
                                       1897-2005 fell in these four days)

The rule-based calendar covers regular NYSE holidays. Ad-hoc closures (e.g.
national days of mourning) are caught by verify_against() and only shift day
labels — window stats are computed off actual price rows.
"""

from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta

# Window geometry (offsets vs tau; tau itself is offset 0).
WIN_START = -9          # first PreTOM day
WIN_END_CLASSIC = -4    # classic window close
WIN_END_T1 = -3         # T+1-regime marginal selling day / post-window start
POST_END = 3            # post window runs through tau+3
TOM_END = 3             # turn-of-the-month long window [tau, tau+3]
LOCK_AT = -10           # basket locks the day before the window opens


def _easter(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, _cal.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """NYSE observation shift: Sat -> Fri, Sun -> Mon.

    Exception: when Jan 1 falls on a Saturday the NYSE does not observe it
    (no Dec 31 holiday in the prior year), so New Year's is handled by the
    caller; this helper is for the other fixed-date holidays.
    """
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    hol: set[date] = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 6:                      # Sunday -> Monday Jan 2
        hol.add(date(year, 1, 2))
    elif ny.weekday() != 5:                    # Saturday -> not observed
        hol.add(ny)
    hol.add(_nth_weekday(year, 1, 0, 3))       # MLK Day (3rd Mon Jan)
    hol.add(_nth_weekday(year, 2, 0, 3))       # Washington's Birthday
    hol.add(_easter(year) - timedelta(days=2))  # Good Friday
    hol.add(_last_weekday(year, 5, 0))         # Memorial Day (last Mon May)
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))  # Juneteenth
    hol.add(_observed(date(year, 7, 4)))       # Independence Day
    hol.add(_nth_weekday(year, 9, 0, 1))       # Labor Day (1st Mon Sep)
    hol.add(_nth_weekday(year, 11, 3, 4))      # Thanksgiving (4th Thu Nov)
    hol.add(_observed(date(year, 12, 25)))     # Christmas
    return hol


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def trading_days(start: date, end: date) -> list[date]:
    """All trading days in [start, end], inclusive."""
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def month_trading_days(year: int, month: int) -> list[date]:
    return trading_days(date(year, month, 1),
                        date(year, month, _cal.monthrange(year, month)[1]))


def month_tau(year: int, month: int) -> date:
    return month_trading_days(year, month)[-1]


def month_schedule(year: int, month: int) -> dict:
    """Key dates of the month's PreTOM cycle (values are datetime.date)."""
    td = month_trading_days(year, month)
    tau = td[-1]
    nxt = next_trading_day(tau)
    post = trading_days(nxt, nxt + timedelta(days=10))[:POST_END]
    return {
        "tau": tau,
        "lock_by": td[LOCK_AT - 1],             # tau-10
        "win_start": td[WIN_START - 1],          # tau-9
        "win_end_classic": td[WIN_END_CLASSIC - 1],
        "win_end_t1": td[WIN_END_T1 - 1],
        "post_start": td[WIN_END_T1 - 1],        # post window opens at tau-3
        "post_end": post[POST_END - 1],          # tau+3, in the next month
        "tom_start": tau,                        # TOM long window opens at tau
        "tom_end": post[TOM_END - 1],            # tau+3, same day as post_end
    }


def schedule_iso(sched: dict) -> dict:
    return {k: v.isoformat() for k, v in sched.items()}


def tau_offset(d: date) -> int | None:
    """Offset of d vs its own month's tau (<= 0); None on non-trading days."""
    if not is_trading_day(d):
        return None
    td = month_trading_days(d.year, d.month)
    return td.index(d) - (len(td) - 1)


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def state_for(d: date) -> dict:
    """PreTOM cycle state on date d (evaluated at the next trading day if d
    isn't one, so weekend/holiday views still show a live countdown).

    States: FORMING -> LOCKED -> WINDOW_OPEN -> POST -> (FORMING of next cycle).
    The first three trading days of a month are the *previous* month's POST.
    """
    eff = d if is_trading_day(d) else next_trading_day(d)

    py, pm = _prev_month(eff.year, eff.month)
    prev_sched = month_schedule(py, pm)
    if eff <= prev_sched["post_end"]:
        month, sched = f"{py:04d}-{pm:02d}", prev_sched
        k = len(trading_days(next_trading_day(sched["tau"]), eff))  # tau+k
        state = "POST"
    else:
        month, k = f"{eff.year:04d}-{eff.month:02d}", tau_offset(eff)
        sched = month_schedule(eff.year, eff.month)
        if k < LOCK_AT:
            state = "FORMING"
        elif k < WIN_START:
            state = "LOCKED"
        elif k <= WIN_END_T1:
            state = "WINDOW_OPEN"
        else:
            state = "POST"

    def _tdays_until(target: date) -> int:
        return len(trading_days(eff, target)) - 1 if eff <= target else 0

    return {
        "state": state,
        "month": month,
        "tau_offset": k,
        "is_trading_day": is_trading_day(d),
        "t1_tail": state == "WINDOW_OPEN" and k == WIN_END_T1,
        "days_to_open": _tdays_until(sched["win_start"]),
        "days_to_close_classic": _tdays_until(sched["win_end_classic"]),
        "days_to_close_t1": _tdays_until(sched["win_end_t1"]),
        "days_to_post_end": _tdays_until(sched["post_end"]),
        "schedule": sched,
    }


def completed_months(n: int, today: date) -> list[tuple[int, int]]:
    """Last n (year, month) pairs whose full cycle (through tau+3) has ended."""
    out, y, m = [], today.year, today.month
    while len(out) < n:
        y, m = _prev_month(y, m)
        if month_schedule(y, m)["post_end"] < today:
            out.append((y, m))
    return list(reversed(out))


def verify_against(actual_days) -> list[str]:
    """Cross-check rule calendar vs actual trading dates (e.g. SPY index).

    Returns mismatch descriptions; empty list = ok. Only past dates within the
    actual index's range are compared.
    """
    actual = sorted({d.date() if hasattr(d, "date") else d for d in actual_days})
    if len(actual) < 2:
        return []
    expected = set(trading_days(actual[0], actual[-1]))
    got = set(actual)
    issues = [f"unexpected close {d}" for d in sorted(expected - got)]
    issues += [f"unexpected session {d}" for d in sorted(got - expected)]
    return issues

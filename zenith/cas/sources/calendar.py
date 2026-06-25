"""Rebalance & key-date calendar — computed (no external data needed).

Surfaces the recurring market-structure dates that drive flows: monthly options
expiration (3rd Friday), quarterly triple-witching, month / quarter end, S&P
quarterly index rebalance (effective after 3rd Friday of Mar/Jun/Sep/Dec), and
Russell reconstitution (late June). Returns upcoming events with days-until.
"""

from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta


def _third_friday(year: int, month: int) -> date:
    c = _cal.Calendar(firstweekday=_cal.MONDAY)
    fridays = [d for d in c.itermonthdates(year, month)
               if d.month == month and d.weekday() == _cal.FRIDAY]
    return fridays[2]


def _last_business_day(year: int, month: int) -> date:
    d = date(year, month, _cal.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def upcoming(today: date | None = None, horizon_days: int = 120) -> list[dict]:
    today = today or date.today()
    end = today + timedelta(days=horizon_days)
    events: list[dict] = []

    y, m = today.year, today.month
    for _ in range(6):                       # look a few months ahead
        tf = _third_friday(y, m)
        if today <= tf <= end:
            quarterly = m in (3, 6, 9, 12)
            events.append({"date": tf.isoformat(),
                           "event": "Triple witching" if quarterly else "Monthly OPEX",
                           "kind": "opex", "note": "Quarterly index/futures/options expiry"
                           if quarterly else "Monthly options expiration"})
            if quarterly:
                events.append({"date": tf.isoformat(), "event": "S&P quarterly rebalance",
                               "kind": "index_rebal",
                               "note": "S&P index changes effective after the close"})
        lbd = _last_business_day(y, m)
        if today <= lbd <= end:
            qe = m in (3, 6, 9, 12)
            events.append({"date": lbd.isoformat(),
                           "event": "Quarter-end" if qe else "Month-end",
                           "kind": "period_end",
                           "note": "Pension/40-act rebalancing window"})
        m += 1
        if m > 12:
            m, y = 1, y + 1

    # Russell reconstitution — effective after the last Friday of June
    for yr in (today.year, today.year + 1):
        june_fridays = [d for d in _cal.Calendar().itermonthdates(yr, 6)
                        if d.month == 6 and d.weekday() == _cal.FRIDAY]
        rr = june_fridays[-1]
        if today <= rr <= end:
            events.append({"date": rr.isoformat(), "event": "Russell reconstitution",
                           "kind": "index_rebal", "note": "Annual Russell index rebuild"})

    for e in events:
        e["days_until"] = (date.fromisoformat(e["date"]) - today).days
    events.sort(key=lambda e: e["date"])
    # de-dupe identical (date,event)
    seen, uniq = set(), []
    for e in events:
        k = (e["date"], e["event"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq

"""Daily Five — five blind charts a day, the same for every player.

Fairness and comparability:
  * Fixed settings (Daily candles, 120 visible, 10-candle horizon, default
    indicators, no stops; the pre-commit note is optional) so days compare.
  * Each day's charts come from a seed derived from the date (sha256), drawn
    class-balanced with at most 2 per class and no ticker repeated within 30
    days, then ordered easy / medium x3 / hard by how decisive the move was
    (|10-bar return| in ATR units -- VELA's difficulty shape).
  * The schedule is generated ahead by the nightly job and COMMITTED
    append-only (daily_schedule.json): once a day exists it never changes,
    even though the price store grows and is re-adjusted every night.
  * One attempt per player per day: daily_results' primary key
    (player, date, slot).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date, datetime, timedelta, timezone

import numpy as np

from ..config import EPHEMERIS_DIR, EPHEMERIS_FILES
from . import CLASSES
from .indicators import DAILY_FIVE_SET, atr
from .sampler import NoChartError, draw

SETTINGS = {"tf": "Daily", "lookback": 120, "horizon": 10}
SLOTS = 5
SHAPE = ("easy", "medium", "medium", "medium", "hard")
EPOCH = date(2026, 10, 1)            # Daily Five #1
NO_REPEAT_DAYS = 30
MAX_PER_CLASS = 2
CANDIDATES = 15
INDICATORS = DAILY_FIVE_SET


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def game_number(day: date) -> int:
    return (day - EPOCH).days + 1


def seed_for(day: str) -> int:
    return int(hashlib.sha256(f"ephemeris-daily|{day}".encode()).hexdigest()[:16], 16)


def _move_z(ch) -> float:
    a = atr(ch.h, ch.l, ch.c, 14)[ch.t]
    if not np.isfinite(a) or a <= 0:
        return 0.0
    ret = ch.c[ch.t + ch.horizon] / ch.entry - 1
    return abs(ret) / (a / ch.entry * math.sqrt(ch.horizon))


def build_day(store, universe: list[dict], day: str, recent: set[str] | None = None) -> list[dict]:
    """Five slots for `day`, deterministic for a given store + universe."""
    rng = random.Random(seed_for(day))
    recent = recent or set()
    cands, seen_t = [], set()
    for _ in range(CANDIDATES * 8):
        if len(cands) >= CANDIDATES:
            break
        try:
            ch = draw(store, universe, classes=CLASSES, rng=rng, **SETTINGS)
        except NoChartError:
            break
        if ch.ticker in recent or ch.ticker in seen_t:
            continue
        seen_t.add(ch.ticker)
        cands.append((_move_z(ch), ch))
    if len(cands) < SLOTS:
        return []
    cands.sort(key=lambda x: -x[0])                  # most decisive first

    picked, per_cls = [], {}

    def take(pool):
        for z, ch in pool:
            if ch in [p[1] for p in picked] or per_cls.get(ch.cls, 0) >= MAX_PER_CLASS:
                continue
            per_cls[ch.cls] = per_cls.get(ch.cls, 0) + 1
            picked.append((z, ch))
            return True
        return False

    take(cands)                                       # easy: most decisive
    hard_pool = list(reversed(cands))
    mid = len(cands) // 2
    mids = sorted(cands, key=lambda x: abs(cands.index(x) - mid))
    for _ in range(3):
        take(mids)
    take(hard_pool)                                   # hard: least decisive (chop)
    if len(picked) < SLOTS:                           # class cap too tight -> relax
        for zc in cands:
            if len(picked) >= SLOTS:
                break
            if zc not in picked:
                picked.append(zc)
    order = [picked[0]] + sorted(picked[1:4], key=lambda x: -x[0]) + [picked[4]]
    return [{"slot": i, "ticker": ch.ticker, "cls": ch.cls,
             "decision": ch.decision_date.strftime("%Y-%m-%d"), "difficulty": SHAPE[i],
             "move_z": round(float(z), 2)} for i, (z, ch) in enumerate(order)]


def load_schedule() -> dict:
    try:
        return json.loads(EPHEMERIS_FILES["daily_schedule"].read_text(encoding="utf-8"))
    except Exception:
        return {"epoch": EPOCH.isoformat(), "settings": SETTINGS, "days": {}}


def extend_schedule(store, universe: list[dict], days_ahead: int = 14, start: date | None = None) -> int:
    """Append missing days from `start` (today) to today+days_ahead. Existing
    days are never rewritten."""
    sch = load_schedule()
    days = sch.setdefault("days", {})
    start = start or today_utc()
    added = 0
    for k in range(days_ahead + 1):
        d = start + timedelta(days=k)
        key = d.isoformat()
        if key in days:
            continue
        recent = {s["ticker"] for j in range(1, NO_REPEAT_DAYS + 1)
                  for s in days.get((d - timedelta(days=j)).isoformat(), [])}
        slots = build_day(store, universe, key, recent)
        if len(slots) == SLOTS:
            days[key] = slots
            added += 1
    sch["days"] = dict(sorted(days.items()))
    EPHEMERIS_DIR.mkdir(parents=True, exist_ok=True)
    EPHEMERIS_FILES["daily_schedule"].write_text(json.dumps(sch, indent=1), encoding="utf-8")
    return added


def slots_for(day: date, schedule: dict | None = None) -> list[dict]:
    return ((schedule or load_schedule()).get("days") or {}).get(day.isoformat(), [])


# ------------------------------------------------------------ streaks ----
def streak(completed: set[date], today: date) -> int:
    """Consecutive fully-played days ending today -- or yesterday, so an
    unplayed today doesn't break the streak until it is over."""
    d = today if today in completed else today - timedelta(days=1)
    n = 0
    while d in completed:
        n += 1
        d -= timedelta(days=1)
    return n


def best_streak(completed: set[date]) -> int:
    best = cur = 0
    prev = None
    for d in sorted(completed):
        cur = cur + 1 if prev and (d - prev).days == 1 else 1
        best = max(best, cur)
        prev = d
    return best


def share_text(day: date, results: list[dict], streak_n: int) -> str:
    """Copy-paste result: emoji grid + PnL, no answers revealed."""
    grid = "".join("🟩" if r["win"] else "🟥" for r in results)
    calls = "".join("▲" if r["direction"] > 0 else "▼" for r in results)
    wins = sum(bool(r["win"]) for r in results)
    pnl = sum(float(r["pnl"]) for r in results)
    sign = "+" if pnl >= 0 else "−"
    return (f"EPHEMERIS Daily Five #{game_number(day)} · {day.isoformat()}\n"
            f"{grid}  {wins}/{len(results)}\n{calls}\n"
            f"P&L {sign}${abs(pnl):,.0f} · streak {streak_n}")

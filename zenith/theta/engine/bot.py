"""The mechanical tastytrade benchmark played on the same tape as the player.

One SPY short strangle at a time: 16-delta shorts at the monthly expiry closest
to 45 DTE, closed at 50% of max profit or 21 DTE, whichever comes first, then
reopened the next day. Contracts are sized so the strangle uses about 30% of
net liq in buying power (the middle of tastylive's 25-50% band at normal VIX).
`expiry_mode="calendar"` reproduces the OPTIONS VAULT repro convention (expiry
= last trading day on or before entry + 45 days) for the cross-check test.
"""

from __future__ import annotations

import math

import pandas as pd

from ..market import Market
from . import clock
from .book import Book
from .strategies import build

TICKER = "SPY"
DTE, DELTA, BP_SHARE = 45, 0.16, 0.30


def _expiry(m: Market, i: int, mode: str) -> int | None:
    if mode == "calendar":
        target = m.dates[i] + pd.Timedelta(days=DTE)
        e = int(m.dates.searchsorted(target, side="right") - 1)
        return e if e > i and target <= m.dates[-1] else None
    return m.nearest_expiry(TICKER, i, DTE)


def play(m: Market, start_i: int, end_i: int, cash: float, slip: float = 1.0,
         flat_iv: bool = False, expiry_mode: str = "listed", contracts: int | None = None,
         stop: float | None = None) -> dict:
    """Run the bot; returns {equity: [[i, nlv]], trades: [...], book: Book}."""
    book = Book(cash, cash, slip=slip, flat_iv=flat_iv)
    run = clock.Run(book, start_i, end_i, start_i)
    trades = []
    pos = None
    clock.snapshot(m, run)
    while True:
        i = run.i
        if pos is not None and pos.is_open:
            pnl = book.position_pnl(m, pos, i)
            dte = m.days_between(i, pos.expiry_i())
            hit = pnl >= 0.5 * pos.credit or dte <= 21 or (stop and pnl <= -stop * pos.credit)
            if hit:
                book.close(m, pos, i, "bot rule")
                trades.append({"open_i": pos.opened_i, "close_i": i, "credit": pos.credit,
                               "pnl": pos.pnl})
                pos = None
        if pos is not None and not pos.is_open:          # expired / settled
            trades.append({"open_i": pos.opened_i, "close_i": pos.closed_i, "credit": pos.credit,
                           "pnl": pos.pnl})
            pos = None
        if pos is None and i < end_i and m.live(TICKER, i):
            e = _expiry(m, i, expiry_mode)
            if e is not None and e <= end_i + 60:
                legs = build("short_strangle", m, TICKER, i, e, 1, DELTA, flat=flat_iv)
                n = contracts
                if n is None:
                    per = book.preview(m, TICKER, i, legs)["bpr"]
                    n = max(1, math.floor(BP_SHARE * book.nlv(m, i) / max(per, 1.0)))
                legs = build("short_strangle", m, TICKER, i, e, n, DELTA, flat=flat_iv)
                pos = book.open(m, TICKER, i, "short_strangle", legs, DELTA)
        if run.done:
            break
        clock.step(m, run)
    return {"equity": run.equity, "trades": trades, "book": book}

"""Time: advance the run and raise the events a tastytrade trader reacts to.

Order within a day: move to day i -> settle expiries at i's close -> mark ->
detect events. "To next event" stops on the first day any event fires.

Events (each fires once per position, re-armed by a roll):
  expiry      a leg settled (expired, assigned or exercised)
  dte21       the position's nearest expiry is 21 days out or less
  earnings    the underlying reports tomorrow
  target      P&L >= 50% of the opening credit
  stop        loss >= 2x the opening credit
  bp          buying power in use above the allocation guideline
  end         the scenario is over
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..market import Market
from .book import Book

TARGET = 0.50
STOP = 2.0
DTE_EXIT = 21
BP_GUIDE = 0.50                     # tastylive: 25-50% of net liq in short premium


@dataclass
class Event:
    i: int
    kind: str
    msg: str
    pid: int | None = None

    def to_dict(self) -> dict:
        return {"i": self.i, "kind": self.kind, "msg": self.msg, "pid": self.pid}


@dataclass
class Run:
    """The mutable state of one game: book + clock + equity history."""
    book: Book
    start_i: int
    end_i: int
    i: int
    equity: list[list[float]] = field(default_factory=list)    # [i, nlv, bp_used, theta]

    def to_dict(self) -> dict:
        return {"book": self.book.to_dict(), "start_i": self.start_i, "end_i": self.end_i,
                "i": self.i, "equity": self.equity}

    @classmethod
    def from_dict(cls, d: dict) -> "Run":
        return cls(Book.from_dict(d["book"]), d["start_i"], d["end_i"], d["i"],
                   [list(x) for x in d.get("equity", [])])

    @property
    def done(self) -> bool:
        return self.i >= self.end_i


def snapshot(m: Market, run: Run) -> None:
    b, i = run.book, run.i
    theta = b.portfolio_greeks(m, i)["theta"] if b.open_positions() else 0.0
    row = [i, round(b.nlv(m, i), 2), round(b.bp_used(m, i), 2), round(theta, 2)]
    if run.equity and run.equity[-1][0] == i:
        run.equity[-1] = row
    else:
        run.equity.append(row)


def detect(m: Market, run: Run) -> list[Event]:
    b, i = run.book, run.i
    out: list[Event] = []
    for p in b.open_positions():
        ex = p.expiry_i()
        if ex >= 0 and m.days_between(i, ex) <= DTE_EXIT and "dte21" not in p.flags:
            p.flags.append("dte21")
            out.append(Event(i, "dte21", f"#{p.id} {p.ticker} is at {m.days_between(i, ex)} DTE", p.id))
        nr = m.next_report(p.ticker, i)
        if nr is not None and nr == i + 1 and f"earn{nr}" not in p.flags:
            p.flags.append(f"earn{nr}")
            out.append(Event(i, "earnings", f"{p.ticker} reports tomorrow (#{p.id} is exposed)", p.id))
        if p.credit > 0 and p.option_legs():
            pnl = b.position_pnl(m, p, i)
            if pnl >= TARGET * p.credit and "target" not in p.flags:
                p.flags.append("target")
                out.append(Event(i, "target", f"#{p.id} {p.ticker} hit {pnl / p.credit:.0%} of max profit", p.id))
            if pnl <= -STOP * p.credit and "stop" not in p.flags:
                p.flags.append("stop")
                out.append(Event(i, "stop", f"#{p.id} {p.ticker} is down {-pnl / p.credit:.1f}x the credit", p.id))
    nlv = b.nlv(m, i)
    if nlv > 0 and b.bp_used(m, i) / nlv > BP_GUIDE:
        if not run.equity or len(run.equity) < 2 or run.equity[-2][2] / max(run.equity[-2][1], 1) <= BP_GUIDE:
            out.append(Event(i, "bp", f"Buying power in use {b.bp_used(m, i) / nlv:.0%} of net liq "
                                      f"(guideline <= {BP_GUIDE:.0%})"))
    if run.done:
        out.append(Event(i, "end", "Scenario complete"))
    return out


def step(m: Market, run: Run) -> list[Event]:
    """Advance one trading day."""
    if run.done:
        return [Event(run.i, "end", "Scenario complete")]
    run.i += 1
    events = [Event(run.i, "expiry", e) for e in run.book.settle(m, run.i)]
    snapshot(m, run)
    events += detect(m, run)
    for e in events:
        if e.kind != "expiry":
            run.book.note(e.i, e.kind, e.msg, e.pid)
    return events


def advance(m: Market, run: Run, mode: str = "day", max_days: int = 400) -> list[Event]:
    """mode: 'day' | 'week' (5 trading days) | 'event' (until the first event)."""
    if mode == "day":
        return step(m, run)
    # an empty book has nothing to wait for: "next event" then means a week, not
    # a silent fast-forward to the end of the scenario
    n = 5 if mode == "week" or not run.book.open_positions() else max_days
    out: list[Event] = []
    for _ in range(n):
        ev = step(m, run)
        out += ev
        if run.done or (mode == "event" and ev):
            break
    return out

"""Data-quality log: every assumption, skip and anomaly is recorded, never silent."""
from __future__ import annotations

from dataclasses import dataclass, field

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Entry:
    severity: str
    area: str
    message: str
    ticker: str = ""


@dataclass
class QualityLog:
    entries: list[Entry] = field(default_factory=list)
    conflicts: dict = field(default_factory=dict)      # column -> [count, example] while merging files

    def add(self, severity: str, area: str, message: str, ticker: str = "") -> None:
        e = Entry(severity, area, message, ticker or "")
        if e not in self.entries:
            self.entries.append(e)

    def sorted(self) -> list[Entry]:
        return sorted(self.entries, key=lambda e: (SEVERITY_ORDER.get(e.severity, 9), e.area, e.ticker, e.message))

    def count(self, severity: str) -> int:
        return sum(e.severity == severity for e in self.entries)

    def copy(self) -> "QualityLog":
        return QualityLog(list(self.entries))

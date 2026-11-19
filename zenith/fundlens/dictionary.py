"""Metric dictionary: maps raw Morningstar labels to canonical metric columns."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
METRICS_PATH = PKG_DIR / "metrics.yaml"
USER_METRICS_PATH = PKG_DIR / "metrics_user.yaml"     # your approved additions (app / Claude skill)
CODES_PATH = PKG_DIR / "codes.yaml"

_H = r"(?P<h>ytd|\d+ ?(?:yr|mo)|since incep(?:tion)?)"
_F = r"\((?P<freq>qtr|mo)-end\)"
_Y = r"(?P<y>(?:19|20)\d\d)"

SCORING_HORIZONS = ("3y", "5y", "10y", "15y")
HORIZON_LABEL = {"ytd": "YTD", "1m": "1M", "3m": "3M", "6m": "6M", "12m": "1Y",
                 "3y": "3Y", "5y": "5Y", "10y": "10Y", "15y": "15Y", "si": "Incep"}
HORIZON_ORDER = list(HORIZON_LABEL)


def norm_label(s: object) -> str:
    """Lower-case, collapse whitespace, strip. 'Sortino  Ratio  3 Yr ' -> 'sortino ratio 3 yr'."""
    if s is None:
        return ""
    s = str(s).replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\s*\(\s*", " (", s)          # 'bear market %(mo-end)' -> 'bear market % (mo-end)'
    s = re.sub(r"\s*\)", ")", s)
    s = re.sub("[\u24d8\u2139]", "", s)                       # pasted info icons
    s = re.sub(r"\b(\d+)-year\b", r"\1 yr", s)         # 'information ratio 3-year' -> '3 yr'
    s = re.sub(r"\s+(annlzd|annualized)\b", "", s)     # 'tot ret 3 yr annlzd' -> 'tot ret 3 yr'
    return re.sub(r"\s+", " ", s).strip()


def norm_horizon(tok: str) -> str:
    tok = tok.replace(" ", "")
    if tok == "ytd":
        return "ytd"
    if tok.startswith("sinceincep"):
        return "si"
    m = re.fullmatch(r"(\d+)(yr|mo)", tok)
    if not m:
        return tok
    n, unit = int(m.group(1)), m.group(2)
    if (unit == "mo" and n == 12) or (unit == "yr" and n == 1):
        return "12m"
    return f"{n}{'y' if unit == 'yr' else 'm'}"


@dataclass
class Metric:
    id: str
    label: str
    kind: str
    format: str = "num"
    direction: str = "none"
    horizons: bool = False
    years: bool = False
    derived: bool = False
    benchmark_relative: bool = False
    description: str = ""
    patterns: list[str] = field(default_factory=list)
    regexes: list[re.Pattern] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Match:
    metric: Metric
    column: str          # canonical column id, e.g. sharpe_3y / nav_annual_2024 / net_er
    horizon: str | None
    year: int | None
    freq: str | None     # 'qtr' / 'mo' when the label carried a (qtr-end)/(mo-end) suffix


def _compile(p: str) -> re.Pattern:
    """Patterns are regexes over normalised labels with {H}/{F}/{Y} placeholders."""
    rx = p.replace("{H}", _H).replace("{F}", _F).replace("{Y}", _Y)
    return re.compile(rf"^{rx}$")


class Dictionary:
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.by_id = {m.id: m for m in metrics}

    def match(self, label: object) -> Match | None:
        nl = norm_label(label)
        if not nl:
            return None
        for m in self.metrics:
            if m.derived:
                continue
            for rx in m.regexes:
                mt = rx.match(nl)
                if not mt:
                    continue
                gd = mt.groupdict()
                h = norm_horizon(gd["h"]) if gd.get("h") else None
                y = int(gd["y"]) if gd.get("y") else None
                col = m.id if h is None and y is None else f"{m.id}_{h if h else y}"
                return Match(m, col, h, y, gd.get("freq"))
        return None

    def column_meta(self, column: str) -> tuple[Metric | None, str | None, int | None]:
        """Reverse lookup: 'sharpe_3y' -> (Metric sharpe, '3y', None)."""
        if column in self.by_id:
            return self.by_id[column], None, None
        base, _, suffix = column.rpartition("_")
        m = self.by_id.get(base)
        if m is None:
            return None, None, None
        if suffix.isdigit() and len(suffix) == 4:
            return m, None, int(suffix)
        return m, suffix, None

    def column_label(self, column: str) -> str:
        m, h, y = self.column_meta(column)
        if m is None:
            return column
        if h:
            return f"{m.label} {HORIZON_LABEL.get(h, h)}"
        if y:
            return f"{m.label} {y}" if m.id != "nav_annual" else str(y)
        return m.label


@lru_cache(maxsize=4)
def load_dictionary(path: str | None = None) -> Dictionary:
    """Core dictionary plus metrics_user.yaml (user entries are matched first)."""
    entries = []
    sources = [Path(path)] if path else [USER_METRICS_PATH, METRICS_PATH]
    for src in sources:
        if src.exists():
            entries += (yaml.safe_load(src.read_text(encoding="utf-8")) or {}).get("metrics") or []
    metrics, seen = [], set()
    for d in entries:
        d = dict(d)
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        pats = d.pop("patterns", []) or []
        m = Metric(**d, patterns=pats)
        m.regexes = [_compile(p) for p in pats]
        metrics.append(m)
    return Dictionary(metrics)


def add_user_metric(entry: dict) -> None:
    """Append a metric to metrics_user.yaml (used by the app's Unmapped Metrics panel)."""
    data = {"metrics": []}
    if USER_METRICS_PATH.exists():
        data = yaml.safe_load(USER_METRICS_PATH.read_text(encoding="utf-8")) or {"metrics": []}
        data.setdefault("metrics", [])
    data["metrics"] = [m for m in data["metrics"] if m["id"] != entry["id"]] + [entry]
    USER_METRICS_PATH.write_text("# User-approved metric mappings (loaded before metrics.yaml)\n"
                                 + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_dictionary.cache_clear()


@lru_cache(maxsize=4)
def load_codes(path: str | None = None) -> dict[str, dict[str, str]]:
    raw = yaml.safe_load(Path(path or CODES_PATH).read_text(encoding="utf-8")) or {}
    return {k: {str(c): str(v) for c, v in (m or {}).items()} for k, m in raw.items()}

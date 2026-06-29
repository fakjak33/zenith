"""Weekly Brief — an automated, aesthetic weekly market commentary for Zenith.

A separate top-level tab that recreates (and extends) a hand-made weekly market
update: market overview, sector & industry breakdown, rate expectations, fixed
income, additional charts, earnings, market-moving news, plus CAS highlights,
an options/gamma read, and plain-English "talking points".

Free / best-effort data only (yfinance + FRED + finviz + Zenith's own scrape),
so it runs unattended in the GitHub Action. Figures are decision-support, not
investment advice.
"""

from __future__ import annotations

import json

from ..config import BRIEF_FILES

DISCLAIMER = ("Weekly Brief is auto-generated from free/best-effort data (yfinance, FRED, "
             "finviz, Zenith feeds). Fed-funds odds and dealer-gamma are labelled proxies. "
             "Decision-support, not investment advice.")


def load(name: str = "brief", default=None):
    p = BRIEF_FILES[name]
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}


def save(name: str, obj) -> None:
    p = BRIEF_FILES[name]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

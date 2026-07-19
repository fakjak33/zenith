"""PRETOM — intramonth momentum short screener for Zenith.

Implements the mechanism documented in Nathan, Suominen & Tasa (2026),
"The Intramonth Momentum Cycle": momentum-loser underperformance concentrates
in a six-trading-day window ending four days before month-end (PreTOM =
[tau-9, tau-4], tau = last trading day of the month), driven by predictable
month-end institutional cash demand. The screener ranks the Russell 1000 by
"dispensability" — distance below the 52-week high (the paper's strongest
proxy), liquidity (pressure concentrates in liquid losers), and index weight
(prevalence across funds that may sell to raise cash) — locks the bottom
decile as a short-bias basket before the window opens, and tracks every past
basket's performance through both the classic [tau-9, tau-4] window and the
post-T+1 span [tau-9, tau-3], plus the [tau-3, tau+3] reversal.

The tab also tracks the LONG side of the same month-end liquidity cycle: the
classic turn-of-the-month (TOM) window [tau, tau+3]. Lakonishok & Smidt
(1988, RFS) first documented it in 90 years of DJIA data; Xu & McConnell
(2008, FAJ, "Equity Returns at the Turn of the Month") show that over
1897-2005 - and still over 1987-2005 - essentially ALL of the U.S. equity
market's excess return accrued in those four trading days. International
confirmation: Kunkel, Compton & Beyer (2003). The leading mechanism is the
same institutional/payroll cash cycle PRETOM shorts into (Ogden 1990; Etula,
Rinne, Suominen & Vaittinen 2020, JF, "Dash for Cash"). The in-app evidence
is rebuilt nightly from SPY's full history (1993-) so the user can judge
whether the effect still exists.

Free / best-effort data only (Vanguard VONE holdings + yfinance), computed
offline in a GitHub Action that commits JSON under data/pretom/; the app is a
thin reader. Decision-support, not investment advice.
"""

from __future__ import annotations

import json

from ..config import PRETOM_FILES, PRETOM_ARCHIVE_DIR

DISCLAIMER = ("PRETOM is auto-generated from free/best-effort data (Vanguard VONE holdings, "
              "yfinance). Short side: Nathan, Suominen & Tasa (2026), 'The Intramonth Momentum "
              "Cycle'. TOM long window: Lakonishok & Smidt (1988); Xu & McConnell (2008). "
              "Backfilled months use current index membership (survivorship bias). "
              "Shorting carries unlimited-loss risk. Decision-support, not investment advice.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(name: str, default=None):
    """Load a named PRETOM artefact (see config.PRETOM_FILES)."""
    return _read(PRETOM_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(PRETOM_FILES[name], obj)


# --- per-month basket snapshots: data/pretom/archive/YYYY-MM.json -----------
def archive_month(month: str, obj: dict) -> None:
    _write(PRETOM_ARCHIVE_DIR / f"{month}.json", obj)


def archive_months() -> list[str]:
    return sorted((p.stem for p in PRETOM_ARCHIVE_DIR.glob("*.json")), reverse=True)


def load_month(month: str) -> dict:
    return _read(PRETOM_ARCHIVE_DIR / f"{month}.json", {})

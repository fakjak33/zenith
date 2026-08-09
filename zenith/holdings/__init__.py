"""HOLDINGS — fund position intelligence for Zenith.

Tracks what a fund actually holds, every day, and how those positions change
over time. The first (and currently only) fund is DBMF, the iMGP DBi Managed
Futures Strategy ETF, sub-advised by Dynamic Beta investments.

Why DBMF is the right first subject: as a transparent active ETF it must
publish its complete portfolio every business day under SEC Rule 6c-11, and
because it expresses the entire strategy through futures, that published file
is a *directional exposure statement* rather than a list of stocks. Each row
carries a signed notional market value, so a negative number is a real short
and the weight column is exposure as a fraction of NAV (-0.96 = 96% short
notional), not a percent-of-portfolio in the usual equity-fund sense.

Three interpretation caveats are carried into the UI rather than buried here:

1. DBi's Dynamic Beta Engine regresses the trailing-60-day returns of the
   largest managed-futures hedge funds against a basket of liquid futures and
   holds the best-fit portfolio. It replicates the target's PERFORMANCE, not
   its POSITIONS. What you see is therefore a model estimate of aggregate CTA
   industry positioning, not a window into what those funds actually hold.
2. Notional is not risk-equivalent across asset classes. A -96% 2-year note
   position carries far less interest-rate risk than a -42% 10-year one. We
   deliberately do not vol-scale or duration-adjust — the published numbers
   are shown as published, with the limitation stated.
3. The Treasury-bill sleeve is collateral backing the futures margin, not a
   directional view. It is classified as `collateral` and excluded from the
   long/short exposure arithmetic.

Architecture: `compute.py` runs in a GitHub Action, fetches the fund's own
page, normalises and validates it, and commits JSON under data/holdings/;
the app is a thin reader. `archive/YYYY-MM-DD.json` is the only authoritative
artefact — latest/history/changes are all regenerable from it via
`--action rebuild`, so a schema change never requires refetching.

Free / primary-source data only. Decision-support, not investment advice.
"""

from __future__ import annotations

import json

from ..config import (HOLDINGS_FUNDS_JSON, holdings_archive_dir,
                      holdings_files)

DISCLAIMER = (
    "HOLDINGS reads each fund's own daily portfolio disclosure (SEC Rule 6c-11) "
    "directly from the fund company's website. Figures are shown exactly as "
    "published: for futures, market value is signed NOTIONAL exposure and weight "
    "is a fraction of NAV, not risk. Notional is not comparable across asset "
    "classes, and DBMF's book is a replication portfolio that tracks the "
    "performance — not the positions — of the managed-futures funds it targets. "
    "Decision-support, not investment advice."
)

# Change classification vocabulary, shared by compare.py, view.py and the tests.
CHANGE_TYPES = ("new", "closed", "increased", "reduced", "flipped", "held")

# Ranking windows offered in the UI, in trading-ish calendar days.
WINDOWS = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "6M": 126}


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


def load(fund: str, name: str, default=None):
    """Load a named artefact for one fund (see config.holdings_files)."""
    return _read(holdings_files(fund)[name], default if default is not None else {})


def save(fund: str, name: str, obj) -> None:
    _write(holdings_files(fund)[name], obj)


# --- per-day snapshots: data/holdings/<fund>/archive/YYYY-MM-DD.json --------
def archive_day(fund: str, day: str, obj: dict) -> None:
    _write(holdings_archive_dir(fund) / f"{day}.json", obj)


def archive_days(fund: str) -> list[str]:
    """Archived snapshot dates, newest first."""
    return sorted((p.stem for p in holdings_archive_dir(fund).glob("*.json")),
                  reverse=True)


def load_day(fund: str, day: str) -> dict:
    return _read(holdings_archive_dir(fund) / f"{day}.json", {})


# --- registry snapshot the app reads without importing the adapters ---------
def load_funds() -> dict:
    return _read(HOLDINGS_FUNDS_JSON, {})


def save_funds(obj: dict) -> None:
    _write(HOLDINGS_FUNDS_JSON, obj)

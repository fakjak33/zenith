"""FMOM — factor momentum (TSFM & CSFM) for Zenith.

Implements Gupta & Kelly (2019), "Factor Momentum Everywhere" (JPM): individual
equity factors are strongly persistent month-to-month (AR(1) positive for 59 of
65 factors), so scaling exposure to each factor by its own vol-adjusted
previous-month return — time-series factor momentum (TSFM) — earned a 0.84
Sharpe, and the cross-sectional variant (CSFM, de-meaning formation returns
across factors first) behaves almost identically. Both are run here as 1-1
strategies (form on the previous month, hold one month) across three factor
families:

  * etf  — real, tradable factor-ETF proxies (excess returns vs SPY);
  * man  — 13 broad equity styles built from strategic-beta ETF composites,
           after Man AHL (2026), "Cash (Equities) Is King";
  * aqr  — the academic 65-factor set via published long-short factor returns
           (Ken French / AQR), plus a Russell 1000 replication layer for
           holdings visibility.

Free / best-effort data only, computed offline in a GitHub Action that commits
JSON under data/fmom/; the app is a thin reader. Decision-support, not
investment advice.
"""

from __future__ import annotations

import json

from ..config import FMOM_FILES, FMOM_ARCHIVE_DIR

DISCLAIMER = ("FMOM is auto-generated from free/best-effort data (yfinance ETF prices, "
              "Ken French Data Library, AQR datasets). Based on Gupta & Kelly (2019), "
              "'Factor Momentum Everywhere'. Long-short model returns ignore trading "
              "costs, borrow and slippage; ETF backtests start at fund inception. "
              "Decision-support, not investment advice.")


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
    """Load a named FMOM artefact (see config.FMOM_FILES)."""
    return _read(FMOM_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(FMOM_FILES[name], obj)


# --- per-month signal snapshots: data/fmom/archive/YYYY-MM.json --------------
def archive_month(month: str, obj: dict) -> None:
    _write(FMOM_ARCHIVE_DIR / f"{month}.json", obj)


def archive_months() -> list[str]:
    return sorted((p.stem for p in FMOM_ARCHIVE_DIR.glob("*.json")), reverse=True)


def load_month(month: str) -> dict:
    return _read(FMOM_ARCHIVE_DIR / f"{month}.json", {})

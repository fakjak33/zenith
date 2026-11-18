"""Research ledger: the trial counter and the one-time OOS reveal log.

Both are keyed by ``spec.family()`` (same data + same holdout). The trial count
is every distinct spec hash you have run on that family's in-sample data —
including each cell of a sensitivity sweep — and feeds the deflated Sharpe.

State lives in a plain dict (Streamlit session_state in the app). When ZENITH
runs locally the ledger is also persisted to ``data/backtest/runs/ledger.json``
so the count survives restarts; on the cloud it lasts for the session.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import RUNS_DIR, is_local

LEDGER_FILE = RUNS_DIR / "ledger.json"


def _blank() -> dict:
    return {"trials": {}, "reveals": {}}


def load() -> dict:
    if is_local() and LEDGER_FILE.exists():
        try:
            d = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
            return {"trials": d.get("trials", {}), "reveals": d.get("reveals", {})}
        except Exception:  # noqa: BLE001
            pass
    return _blank()


def save(ledger: dict) -> None:
    if not is_local():
        return
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        LEDGER_FILE.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001 — a read-only disk must not break the page
        pass


def record(ledger: dict, family: str, hashes) -> int:
    """Add trial hashes to a family; returns the family's trial count."""
    pool = set(ledger["trials"].get(family, []))
    before = len(pool)
    pool.update([hashes] if isinstance(hashes, str) else hashes)
    ledger["trials"][family] = sorted(pool)
    if len(pool) != before:
        save(ledger)
    return len(pool)


def trials(ledger: dict, family: str) -> int:
    return max(1, len(ledger["trials"].get(family, [])))


def reveal(ledger: dict, family: str, spec_hash: str, spec_json: str) -> dict:
    """Record the one-time reveal. A second call for the same family is a no-op."""
    if family not in ledger["reveals"]:
        ledger["reveals"][family] = {"hash": spec_hash, "spec": spec_json,
                                     "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                     "trials_at_reveal": trials(ledger, family)}
        save(ledger)
    return ledger["reveals"][family]


def revealed(ledger: dict, family: str) -> dict | None:
    return ledger["reveals"].get(family)


def contaminated(ledger: dict, family: str, spec_hash: str) -> bool:
    """True once OOS has been seen and the strategy has since been changed."""
    r = revealed(ledger, family)
    return bool(r) and r["hash"] != spec_hash

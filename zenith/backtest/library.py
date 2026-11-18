"""Saved strategies. Locally they are written to ``data/backtest/specs/<name>.json``
(meant to be committed, so a strategy's exact definition is versioned); on the cloud
they live for the session. PORTFOLIO combines them; LIBRARY (phase 5) browses them."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .config import SPECS_DIR, is_local
from .spec import StrategySpec


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
    return s or "strategy"


def load_disk() -> dict[str, dict]:
    out = {}
    if SPECS_DIR.exists():
        for p in sorted(SPECS_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out[d.get("name", p.stem)] = d
            except Exception:  # noqa: BLE001
                continue
    return out


def save(store: dict, name: str, spec: StrategySpec, note: str = "") -> dict:
    rec = {"name": name.strip() or spec.strategy().describe(), "spec": spec.to_dict(), "hash": spec.hash(),
           "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "note": note}
    store[rec["name"]] = rec
    if is_local():
        try:
            SPECS_DIR.mkdir(parents=True, exist_ok=True)
            (SPECS_DIR / f"{_slug(rec['name'])}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return rec


def delete(store: dict, name: str) -> None:
    store.pop(name, None)
    if is_local():
        p = SPECS_DIR / f"{_slug(name)}.json"
        if p.exists():
            p.unlink()


def spec_of(rec: dict) -> StrategySpec:
    return StrategySpec.from_dict(rec["spec"])

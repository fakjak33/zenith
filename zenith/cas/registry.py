"""Model & research-notes registry.

Lets the user refine the system over time by (a) tuning each family's weight and
parameters and (b) pasting research-paper summaries/abstracts that get logged
against a family. Honest scope: this is structured note-keeping plus manual /
simple weight tuning — it does NOT auto-learn from the pasted text. The composite
in consensus.py can read these weights; notes are shown in the UI for context.
"""

from __future__ import annotations

from datetime import datetime

from . import store_cas

# Default per-family weights (1.0 = neutral). Editable via the UI.
DEFAULT_WEIGHTS = {
    "momentum": 1.0, "ma_cross_50_200": 1.0, "mean_reversion": 1.0,
    "donchian_breakout": 1.0, "trend_following": 1.0, "vol_regime": 1.0,
    "formulaic_alpha_combo": 1.0, "relative_strength": 1.0,
    "cot_leveraged_positioning": 1.0, "cot_managed_money_positioning": 1.0,
    "cot_asset_manager_positioning": 1.0, "risk_regime": 1.0,
    "frm_ts_mom": 1.0, "frm_cs_region": 1.0,
    "frm_cs_peer": 1.0, "frm_composite": 1.0,
}


def load() -> dict:
    reg = store_cas.load("registry", {})
    if not reg:
        reg = {"weights": dict(DEFAULT_WEIGHTS), "notes": []}
        store_cas.save("registry", reg)
    reg.setdefault("weights", dict(DEFAULT_WEIGHTS))
    reg.setdefault("notes", [])
    return reg


def save(reg: dict) -> None:
    store_cas.save("registry", reg)


def weight_of(family: str) -> float:
    return float(load()["weights"].get(family, 1.0))


def set_weight(family: str, weight: float) -> None:
    reg = load()
    reg["weights"][family] = round(float(weight), 3)
    save(reg)


def add_note(family: str, title: str, abstract: str,
             weight_adjustment: float | None = None,
             source: str = "", status: str = "") -> dict:
    """Append a research note (e.g. a pasted abstract or an uploaded paper) against
    a family, and optionally nudge that family's weight.

    ``source`` records where it came from (URL / filename); ``status`` lets the
    in-app uploader flag a note as 'pending-review' for the /zenith-research skill
    to process into a scaffold later."""
    reg = load()
    note = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "family": family,
        "title": title.strip()[:200],
        "abstract": abstract.strip()[:4000],
    }
    if source:
        note["source"] = source.strip()[:500]
    if status:
        note["status"] = status.strip()[:40]
    if weight_adjustment is not None:
        reg["weights"][family] = round(float(weight_adjustment), 3)
        note["weight_set_to"] = reg["weights"][family]
    reg["notes"].insert(0, note)
    save(reg)
    return note

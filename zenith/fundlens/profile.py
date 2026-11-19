"""Scoring profiles (YAML presets with `extends` inheritance)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .dictionary import PKG_DIR, SCORING_HORIZONS

PROFILE_DIR = PKG_DIR / "profiles"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k == "metrics":                      # a pillar's metric list is replaced, not merged
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def list_profiles() -> list[dict]:
    out = []
    for p in sorted(PROFILE_DIR.glob("*.yaml")):
        prof = load_profile(p.stem)
        out.append({"id": prof["id"], "name": prof["name"], "description": prof.get("description", "").strip()})
    return out


def load_profile(ref: str | Path | dict) -> dict:
    """Load by id ('us_equity_style'), by path, or from an already-parsed dict."""
    if isinstance(ref, dict):
        raw = copy.deepcopy(ref)
    else:
        p = Path(ref)
        if not p.suffix:
            p = PROFILE_DIR / f"{ref}.yaml"
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    prof = _merge(load_profile(parent), raw) if parent else raw
    return normalise(prof)


def normalise(prof: dict) -> dict:
    prof.setdefault("overrides", {})
    prof.setdefault("screens", {})
    prof.setdefault("lenses", [])
    prof.setdefault("display_metrics", [])
    prof.setdefault("relative_metrics_basis", "standard_index")
    rc = prof.setdefault("recency", {})
    rc.setdefault("mode", "off")
    rc.setdefault("surge_flag", 35)
    rc.setdefault("theme_gap", 5.0)
    rc.setdefault("theme_skew", 15.0)
    sc = prof.setdefault("scoring", {})
    sc.setdefault("min_funds_per_metric", 5)
    sc.setdefault("shrinkage_k", 1.0)
    sc.setdefault("breadth_threshold", 60)
    sc.setdefault("stress_year", 2020)
    sc.setdefault("calendar_years", 10)
    sc.setdefault("low_evidence_coverage", 0.25)
    hw = prof.setdefault("horizon_weights", {h: 0.25 for h in SCORING_HORIZONS})
    prof["horizon_weights"] = {h: float(hw.get(h, 0) or 0) for h in SCORING_HORIZONS}
    for pid, p in prof["pillars"].items():
        p.setdefault("label", pid.replace("_", " ").title())
        p["weight"] = float(p.get("weight", 0) or 0)
        p["metrics"] = {m: float(w) for m, w in (p.get("metrics") or {}).items()}
    return prof


def profile_hash(prof: dict) -> str:
    return hashlib.sha256(json.dumps(prof, sort_keys=True, default=str).encode()).hexdigest()[:12]


def to_yaml(prof: dict) -> str:
    return yaml.safe_dump(prof, sort_keys=False, allow_unicode=True, width=110)

"""OPTIONS VAULT Phase 04 — free-data reproductions of headline claims.

Each module is a pure compute function over pandas series plus a thin fetch
layer (data.py). Outputs are data/vault/repro/<name>.json, cited elsewhere as
``repro:<name>``:

  vrp_spx          VIX vs next-21-day realized SPX vol, 1990-, by IV-rank bucket
  cboe_short_vol   Cboe PUT/BXM/BXMD/WPUT/CNDR/BFLY vs S&P 500 TR (re-checks
                   Phase 03 computed_stats)
  cboe_tail_hedge  Cboe PPUT/CLL vs S&P 500 TR (re-checks Phase 03)
  strangle_mgmt    model-priced 16-delta SPX strangle (Black-Scholes off VIX)
                   under hold / 50% / 21 DTE / both / both + 2x-credit stop
  earnings_drift   continuation after outlier earnings moves, 2001-, by size
                   and period, plus zenith's own PEAD tracker hit rate
  expected_move    how often SPX ends / stays inside the VIX-implied 1-sigma
  vix_spikes       forward VIX, SPX and realized vol after VIX spikes

Claims come from the user's seed hypotheses (00_master_brief.md) and from
data/vault/literature.json. compute/build leave ``tasty_value`` null with
``tasty_note: "pending Phase 02"``; save() then applies the Phase 02 fills in
data/vault/repro_tasty_fills.json (tastylive's value and sources, matched by
claim text), so a rebuild keeps them. Edit that file, not the outputs.

Run (needs network; raw downloads cached in data/vault/cache/repro/):
  .venv/Scripts/python -m zenith.vault.repro.run [--only name ...] [--refresh]

No scipy (commit 1da6c83): normal CDF from zenith.beta.hedge._ncdf, inverse
from statistics.NormalDist. Decision-support, not investment advice.
"""

from __future__ import annotations

import json
from datetime import date

from ...config import VAULT_DIR, VAULT_FILES

REPRO_DIR = VAULT_FILES["repro_dir"]
TASTY_NOTE = "pending Phase 02"
# Phase 02 fills (tastylive's value + sources per claim), matched by claim text and re-applied on
# every save so a rebuild keeps them. A claim whose wording changes falls back to TASTY_NOTE.
TASTY_FILLS_PATH = VAULT_DIR / "repro_tasty_fills.json"


def claim(text: str, source: str, ours: str, verdict: str, literature: list[str] | None = None,
          **extra) -> dict:
    """One tested claim. ``source`` is where the claim came from
    ('seed' = the user's brief, 'lit:<topic>' = literature.json agreement_detail)."""
    return {"text": text, "claim_source": source, "tasty_value": None, "tasty_sources": [],
            "tasty_note": TASTY_NOTE, "our_value": ours, "verdict": verdict,
            "literature": literature or [], **extra}


def artefact(name: str, *, claim_text: str, claims: list[dict], our_value: str, data: str,
             period: str, method: str, caveats: list[str], verdict: str, chart_data: dict,
             literature: list[str] | None = None, **extra) -> dict:
    return {"name": name, "ref": f"repro:{name}", "built": date.today().isoformat(),
            "claim": claim_text, "tasty_value": None, "tasty_sources": [],
            "tasty_note": TASTY_NOTE, "our_value": our_value, "data": data, "period": period,
            "method": method, "caveats": caveats, "verdict": verdict, "claims": claims,
            "literature": literature or [], "chart_data": chart_data, **extra}


def _jsonable(o):
    """json default: numpy scalars -> Python, timestamps -> ISO dates."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()[:10]
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def load_fills(path=None) -> dict[tuple[str, str], dict]:
    """{(repro name, claim text): fill} from the committed fills file ({} if absent)."""
    p = path or TASTY_FILLS_PATH
    if not p.exists():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8")).get("fills", [])
    return {(r["repro"], r["claim"]): r for r in rows}


def apply_tasty(name: str, obj: dict, fills: dict | None = None) -> dict:
    """Write tasty_value / tasty_sources / tasty_note into each matching claims[] row and mirror the
    row that matches the top-level claim. Rows without a fill keep the pending note."""
    fills = load_fills() if fills is None else fills
    for c in obj.get("claims", []):
        f = fills.get((name, c.get("text")))
        if f:
            c["tasty_value"], c["tasty_sources"], c["tasty_note"] = \
                f["tasty_value"], list(f["tasty_sources"]), f["tasty_note"]
    top = next((c for c in obj.get("claims", []) if c.get("text") == obj.get("claim")), None)
    if top:
        obj["tasty_value"], obj["tasty_sources"], obj["tasty_note"] = \
            top["tasty_value"], list(top["tasty_sources"]), top["tasty_note"]
    return obj


def save(name: str, obj: dict) -> None:
    apply_tasty(name, obj)
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    (REPRO_DIR / f"{name}.json").write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def load(name: str, default=None):
    p = REPRO_DIR / f"{name}.json"
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def rnd(x, nd: int = 4):
    """Round for JSON; NaN/None -> None."""
    try:
        if x is None or x != x:
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def verdict_share(lo: float, hi: float, threshold: float = 0.5) -> str:
    """agrees if the whole CI is above the threshold, contradicts if below, else mixed."""
    if lo > threshold:
        return "agrees"
    if hi < threshold:
        return "contradicts"
    return "mixed"

"""Plain-English tooltips for OPTIONS VAULT, fed by data/vault/glossary.json.

The glossary is the single source: the GLOSSARY section lists it and these lookups surface the same
definitions as "?" hover badges (ui_theme.help_badge) and native ``help=`` args, the pattern of
zenith/cas/help_text.py. Lookups never raise; an unknown term returns "".
"""

from __future__ import annotations

from functools import lru_cache

from . import load

# Short names used in the UI -> glossary term
ALIASES = {
    "bpr": "Buying power reduction (BPR)", "dte": "Duration (days to expiration)",
    "pop": "POP (probability of profit)", "iv rank": "IV rank", "theta": "Theta", "delta": "Delta",
    "undefined risk": "Undefined risk", "iv crush": "IV crush", "0dte": "0DTE",
}

# Vault-specific labels that are not glossary terms
EXTRA = {
    "tier": ("Evidence tier. A = replicated in peer-reviewed or major practitioner research across "
             "periods and markets; B = documented but sensitive to period, costs or implementation; "
             "C = tastylive-only, small sample, period-specific or contradicted."),
    "kind": ("EDGE = a source of expected return (the variance risk premium). CONVENTION = a "
             "management rule that reshapes the P&L distribution without creating return. "
             "MECHANIC = how something works (margin, assignment, settlement)."),
    "thin": ("Thin evidence: the card's numbers come mostly from general pages or other strategies, "
             "with no study or reproduction of this exact setup."),
    "draft": ("Draft: Phase 05 output accepted at the gate but not yet reviewed line by line; check "
              "the linked source before acting on a number."),
}


@lru_cache(maxsize=1)
def _terms() -> dict[str, str]:
    try:
        return {t["term"].lower(): t["definition"] for t in load("glossary", {"terms": []})["terms"]}
    except Exception:
        return {}


def help_for(name: str) -> str:
    key = (name or "").lower()
    if key in EXTRA:
        return EXTRA[key]
    terms = _terms()
    return terms.get(ALIASES.get(key, key).lower(), "")

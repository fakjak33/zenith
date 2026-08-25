"""IDEAS unusualness: how EXTREME/rare a setup is, kept deliberately separate
from conviction (spec section 17). A "good company at a fair price" scores
high conviction but LOW unusualness -- it is exactly what a plain long-only
screener would already surface, not the point of this feature.

Unusualness = the average absolute extremity of the covered signal groups
(how far each is pushed from neutral), with a hard discount for broad-beta
instruments (config.IDEAS_OBVIOUS_TICKERS) so "Buy SPY" cannot rank highly
without a genuinely extreme reading of its own (spec section 2).
"""

from __future__ import annotations

from ..config import IDEAS_OBVIOUS_TICKERS

OBVIOUS_DISCOUNT = 0.4     # multiplicative penalty applied to obvious tickers


def compute(group_scores: dict, ticker: str) -> dict:
    covered = {g: v["score"] for g, v in group_scores.items() if v.get("coverage")}
    if not covered:
        return {"unusual": 0.0, "extremity": 0.0, "n_groups": 0,
                "obvious_discount": False, "note": "no signal-group coverage"}
    extremity = sum(abs(s) for s in covered.values()) / len(covered)
    unusual = 100.0 * extremity
    is_obvious = ticker in IDEAS_OBVIOUS_TICKERS
    if is_obvious:
        unusual *= OBVIOUS_DISCOUNT
    return {"unusual": round(max(0.0, min(100.0, unusual)), 2),
            "extremity": round(extremity, 4), "n_groups": len(covered),
            "obvious_discount": is_obvious}

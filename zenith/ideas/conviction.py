"""IDEAS conviction: the weighted blend of the eight group scores into a
signed net score, and its 0-100 magnitude.

Coverage-aware exactly like mom.engine._weighted: a security missing (say)
options data is renormalized over the groups that ARE covered, never dragged
toward zero for a hole in the data. Weights come from config.IDEAS_WEIGHTS
(set a priori from each input's own evidence tier -- never fitted) and are
tilted by the current market regime via config.IDEAS_REGIME_TILTS (spec
section 23, three coarse states only).
"""

from __future__ import annotations

from ..config import IDEAS_WEIGHTS, IDEAS_REGIME_TILTS


def tilted_weights(regime_label: str | None) -> dict[str, float]:
    tilts = IDEAS_REGIME_TILTS.get(regime_label or "neutral / transition", {})
    return {k: w * tilts.get(k, 1.0) for k, w in IDEAS_WEIGHTS.items()}


def compute(group_scores: dict, regime_label: str | None = None) -> dict:
    """Returns {net_score [-1,1], coverage_n, components}. `net_score`'s SIGN
    is the idea's direction (bullish/bearish); its magnitude, once mapped by
    `magnitude()`, is the displayed 0-100 conviction. Direction and magnitude
    are kept in one number here because selection (select.py) needs the sign
    to route a security to the BUY or SELL pool; the 0-100 display transform
    happens only at the very end."""
    weights = tilted_weights(regime_label)
    total = total_w = 0.0
    components = {}
    for g, w in weights.items():
        gs = group_scores.get(g, {})
        if not gs.get("coverage"):
            continue
        total += w * gs["score"]
        total_w += w
        components[g] = {"score": gs["score"], "weight": round(w, 4),
                         "contribution": round(w * gs["score"], 4)}
    if total_w <= 0:
        return {"net_score": 0.0, "coverage_n": 0, "components": {},
                "note": "no signal-group coverage for this security"}
    net = max(-1.0, min(1.0, total / total_w))
    return {"net_score": round(net, 4), "coverage_n": len(components), "components": components}


def magnitude(net_score: float) -> float:
    """0-100 conviction display: 50 = no edge, 100 = maximum edge in whatever
    direction net_score points. The direction itself is a separate field
    (select.py sets side='long'/'short' from sign(net_score))."""
    return round(50.0 + 50.0 * min(1.0, abs(net_score)), 2)

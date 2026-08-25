"""IDEAS confluence: how many independent signal groups agree (spec section 16).

Per-group arrow (bullish/bearish/neutral/no_data) for every one of the eight
groups, plus a simple agree/disagree count relative to the idea's own overall
direction (conviction.py's net_score sign). Shown as "n/8 signals agree" in
the UI -- the spec's illustrative example uses 10 categories; this repo has
eight (see ideas/__init__.py's SIGNAL_GROUPS), so the denominator is honest
about what is actually being counted rather than padded to match the example.
"""

from __future__ import annotations

from . import SIGNAL_GROUPS

_NEUTRAL_BAND = 0.05


def _arrow(score: float) -> str:
    if score > _NEUTRAL_BAND:
        return "bullish"
    if score < -_NEUTRAL_BAND:
        return "bearish"
    return "neutral"


def compute(group_scores: dict, net_score: float) -> dict:
    direction = "bullish" if net_score > 0 else ("bearish" if net_score < 0 else "neutral")
    per_group = {}
    agree = disagree = neutral = no_data = 0
    for g in SIGNAL_GROUPS:
        gs = group_scores.get(g, {})
        if not gs.get("coverage"):
            per_group[g] = "no_data"
            no_data += 1
            continue
        arrow = _arrow(gs["score"])
        per_group[g] = arrow
        if arrow == "neutral":
            neutral += 1
        elif arrow == direction:
            agree += 1
        else:
            disagree += 1
    n_covered = len(SIGNAL_GROUPS) - no_data
    return {"direction": direction, "agree": agree, "disagree": disagree,
            "neutral": neutral, "no_data": no_data, "n_covered": n_covered,
            "n_total": len(SIGNAL_GROUPS), "per_group": per_group,
            "label": f"{agree}/{len(SIGNAL_GROUPS)} signals {direction}"}

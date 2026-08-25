"""IDEAS divergence: price vs fundamentals vs sentiment disagreement.

The spec calls this out as the mechanism most likely to generate genuinely
interesting ideas: "look specifically for situations where Price +
Fundamentals + Sentiment disagree" (spec section on sentiment). Three
concrete, checkable disagreements are flagged; anything not one of these is
left unflagged rather than guessed at.
"""

from __future__ import annotations

_THRESH = 0.15


def compute(group_scores: dict) -> dict:
    tech = group_scores.get("technicals", {})
    fund = group_scores.get("fundamentals", {})
    val = group_scores.get("valuation", {})
    sent = group_scores.get("sentiment", {})
    pos = group_scores.get("positioning", {})

    flags = []

    if tech.get("coverage") and fund.get("coverage"):
        t, f = tech["score"], fund["score"]
        if t <= -_THRESH and f >= _THRESH:
            flags.append({"type": "price_below_fundamentals",
                         "detail": "Price momentum is negative while fundamentals are improving "
                                   "-- the market may be pricing a decline the data does not support."})
        elif t >= _THRESH and f <= -_THRESH:
            flags.append({"type": "price_above_fundamentals",
                         "detail": "Price momentum is positive while fundamentals are deteriorating "
                                   "-- possible momentum without underlying support."})

    if tech.get("coverage") and sent.get("coverage"):
        t, s = tech["score"], sent["score"]
        if t * s < 0 and abs(t) >= _THRESH and abs(s) >= _THRESH:
            flags.append({"type": "price_vs_sentiment",
                         "detail": "Price momentum and analyst/options sentiment point in opposite "
                                   "directions."})

    if pos.get("coverage") and tech.get("coverage"):
        p, t = pos["score"], tech["score"]
        if p <= -_THRESH and t >= _THRESH:
            flags.append({"type": "crowded_short_vs_rising_price",
                         "detail": "Heavily shorted while price momentum turns positive -- a "
                                   "short-squeeze setup, not necessarily a fundamentals-driven move."})

    if val.get("coverage") and tech.get("coverage"):
        v, t = val["score"], tech["score"]
        if v >= _THRESH and t <= -_THRESH:
            flags.append({"type": "cheap_and_falling",
                         "detail": "Valuation screens cheap while price momentum is still negative "
                                   "-- classic value-trap risk unless a catalyst is present."})

    return {"flags": flags, "has_divergence": bool(flags), "n_flags": len(flags)}

"""IDEAS classify: assigns exactly one Opportunity Type + horizon from the
signal groups that actually fired for an idea (spec section 4).

A deterministic rule tree, evaluated top-to-bottom, first match wins -- never
a free-text label. Every rule references a real field in group_scores /
divergence, so a card's "Opportunity: Contrarian Dip Buy" is always traceable
to the numbers that produced it (spec section 29).
"""

from __future__ import annotations

from . import OPPORTUNITY_TYPES

_STRONG = 0.20


def _catalyst_days_out(catalyst_explain: dict) -> int | None:
    """Best-effort trading days to the next scheduled announcement, from the
    catalyst group's own explain payload (upcoming_report_date isn't a day
    count -- this is a light approximation the UI/narrative can refine)."""
    return None   # report_date -> day-count needs "today"; computed by the caller when available


def classify(side: str, group_scores: dict, divergence: dict,
            security_type: str = "stock", catalyst_days_out: int | None = None) -> dict:
    flags = {f["type"] for f in divergence.get("flags", [])}
    g = {k: v.get("score", 0.0) if v.get("coverage") else None for k, v in group_scores.items()}
    tech, val, fund, sent, pos, cat, macro_s = (
        g.get("technicals"), g.get("valuation"), g.get("fundamentals"),
        g.get("sentiment"), g.get("positioning"), g.get("catalyst"), g.get("macro"))

    def s(x):
        return x if x is not None else 0.0

    opp = None
    if side == "long":
        if "crowded_short_vs_rising_price" in flags:
            opp = "Sentiment Reversal"
        elif "price_below_fundamentals" in flags:
            opp = "Mispricing" if s(val) >= _STRONG else "Fundamental/Technical Divergence"
        elif cat is not None and cat >= _STRONG and catalyst_days_out is not None and catalyst_days_out <= 25:
            opp = "Catalyst"
        elif cat is not None and cat >= _STRONG:
            opp = "Earnings Inflection"
        elif s(tech) >= _STRONG and s(val) <= -_STRONG:
            opp = "Momentum"
        elif s(val) >= _STRONG and s(tech) <= -_STRONG / 2:
            opp = "Contrarian" if s(sent) is not None and s(sent) < 0 else "Value"
        elif fund is not None and fund >= _STRONG and val is not None and val >= 0:
            opp = "Quality" if fund >= 0.35 else "Growth"
        elif s(macro_s) >= _STRONG and abs(s(tech)) < _STRONG:
            opp = "Macro Opportunity"
        elif s(val) >= _STRONG and s(fund) >= 0:
            opp = "Long-Term Compounder" if fund is not None and fund >= _STRONG else "Value"
        else:
            opp = "Relative Value"
    else:  # short
        if "price_above_fundamentals" in flags:
            opp = "Fundamental/Technical Divergence"
        elif cat is not None and cat <= -_STRONG:
            opp = "Event-Driven"
        elif pos is not None and pos <= -_STRONG and s(tech) <= 0:
            opp = "Short Thesis"
        elif fund is not None and fund <= -_STRONG and s(tech) <= 0:
            opp = "Broken Thesis / Avoid"
        elif s(macro_s) <= -_STRONG and abs(s(tech)) < _STRONG:
            opp = "Risk-Off Hedge"
        else:
            opp = "Short Thesis"

    assert opp in OPPORTUNITY_TYPES, f"unregistered opportunity type: {opp}"

    if opp in ("Catalyst", "Event-Driven", "Earnings Inflection") and catalyst_days_out is not None and catalyst_days_out <= 25:
        horizon = "weeks"
    elif opp in ("Momentum", "Sentiment Reversal", "Short Thesis"):
        horizon = "months"
    elif opp in ("Value", "Quality", "Long-Term Compounder", "Secular Growth", "Mispricing", "Contrarian"):
        horizon = "6_18m"
    else:
        horizon = "months"

    return {"opportunity_type": opp, "horizon": horizon, "security_type": security_type}

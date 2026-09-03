"""IDEAS narrative: deterministic thesis prose.

Every sentence below is assembled from a field that is ALSO shown numerically
elsewhere on the same idea card -- there is no LLM call and no free-text
generation. If a group has no coverage, its sentence is simply omitted, never
replaced with a placeholder or an invented number (spec section 29, and the
user's explicit decision to keep this deterministic).

Seven fields per idea (spec sections 13 and 27):
  thesis, why_now, idiosyncratic_risk, market_view, zenith_view,
  bull_case, bear_case, change_my_mind
"""

from __future__ import annotations


def _pct(x, digits=1) -> str:
    return f"{x * 100:+.{digits}f}%" if x is not None else "n/a"


def _num(x, digits=1) -> str:
    return f"{x:.{digits}f}" if x is not None else "n/a"


def build(ticker: str, side: str, opp_type: str, group_scores: dict,
         divergence: dict, confluence: dict, riskreward_detail: dict | None = None) -> dict:
    verb_dir = "bullish" if side == "long" else "bearish"
    flags = divergence.get("flags", [])
    g = group_scores

    # --- thesis --------------------------------------------------------
    lead_bits = []
    tech = g.get("technicals", {})
    if tech.get("coverage"):
        # `.get(key, default)` only falls back when the KEY IS ABSENT -- and it
        # never is. Both producers of this dict store `state` explicitly as
        # None when there is no MOMENTUM state to report: groups.technicals()
        # passes `tech.get("state")` straight through, and panel.py's
        # price-only ETF fallback hardcodes `"state": None`. So the default
        # never fired and `.lower()` hit None instead, which is what took the
        # nightly IDEAS run down. `or` handles both the absent key and the
        # present-but-None value. `_num` likewise keeps a None composite from
        # rendering as the literal string "None" in the thesis.
        te = tech.get("explain") or {}
        lead_bits.append(f"a {(te.get('state') or 'mixed').lower()} technical read "
                         f"(composite {_num(te.get('composite'))})")
    val = g.get("valuation", {})
    if val.get("coverage"):
        cross = (val.get("explain") or {}).get("cross_sectional") or {}
        if cross.get("universe_pctile") is not None:
            lead_bits.append(f"valuation in the {cross['universe_pctile']:.0f}th percentile "
                             f"of the scan universe")
    sent = g.get("sentiment", {})
    if sent.get("coverage"):
        e = sent["explain"]
        if e.get("est_rev_pct") is not None:
            lead_bits.append(f"analyst EPS estimates moving {_pct(e['est_rev_pct'])} over the "
                             f"trailing quarter")
    thesis = f"{ticker} is a {opp_type} {('BUY' if side == 'long' else 'SELL/SHORT')} idea: " \
             + ("; ".join(lead_bits) + "." if lead_bits else "a confluence-driven setup.")

    # --- why now ---------------------------------------------------------
    why_bits = []
    cat = g.get("catalyst", {})
    if cat.get("coverage"):
        e = cat["explain"]
        if e.get("recent_report_date"):
            why_bits.append(f"the {e['recent_side']} reaction to its {e['recent_report_date']} "
                            f"earnings report (composite {e.get('recent_composite')})")
        if e.get("upcoming_report_date"):
            why_bits.append(f"a scheduled announcement on {e['upcoming_report_date']}")
    pos = g.get("positioning", {})
    if pos.get("coverage") and pos["explain"].get("squeeze_risk"):
        why_bits.append("a heavily-shorted position now showing rising price momentum "
                        "(squeeze risk)")
    why_now = ("Catalyst: " + "; ".join(why_bits) + ".") if why_bits else \
        "No scheduled near-term catalyst identified -- this is a signal-confluence setup, " \
        "not an event-driven one."

    # --- idiosyncratic risk ------------------------------------------------
    risk_bits = []
    if flags:
        risk_bits.append(flags[0]["detail"])
    lot = riskreward_detail or {}
    rr_explain = g.get("risk_reward", {}).get("explain", {}).get("parts", {})
    if rr_explain.get("lottery_penalty", 0) < -0.05:
        risk_bits.append("this name also carries an elevated idiosyncratic-volatility "
                         "(lottery/MAXbeta) profile, which historically predicts underperformance "
                         "independent of direction.")
    if not risk_bits:
        risk_bits.append("no specific disagreement between price, fundamentals and sentiment was "
                         "detected -- the main risk is that the composite signal itself proves wrong, "
                         "not a known structural conflict.")
    idiosyncratic_risk = " ".join(risk_bits)

    # --- market view vs zenith view ----------------------------------------
    if flags:
        market_view = "The market's current pricing is consistent with the negative/positive " \
                      "signal that created this divergence in the first place."
        zenith_view = flags[0]["detail"]
    else:
        market_view = f"Consensus pricing appears broadly aligned with the {verb_dir} read across " \
                      f"this idea's covered signal groups ({confluence.get('label', 'n/a')})."
        zenith_view = f"Zenith's fused read agrees with the {verb_dir} direction and ranks it by " \
                      f"how unusual/extreme the configuration is, not just how positive it looks."

    # --- bull / bear ---------------------------------------------------------
    bull_parts, bear_parts = [], []
    for name, gv in g.items():
        if not gv.get("coverage"):
            continue
        if gv["score"] > 0.10:
            bull_parts.append(name)
        elif gv["score"] < -0.10:
            bear_parts.append(name)
    bull_case = (f"Supportive signal groups: {', '.join(bull_parts)}." if bull_parts
                else "No individual group scores strongly positive; this idea rests on the overall blend.")
    bear_case = (f"Opposing signal groups: {', '.join(bear_parts)}." if bear_parts
                else "No individual group scores strongly negative for this idea today.")

    # --- what would change my mind --------------------------------------------
    change = []
    if riskreward_detail and riskreward_detail.get("available"):
        change.append(f"A daily close through {riskreward_detail['stop']:.2f} "
                      f"({'below' if side == 'long' else 'above'} the current level) invalidates the setup.")
    if cat.get("coverage") and cat["explain"].get("upcoming_report_date"):
        change.append(f"The {cat['explain']['upcoming_report_date']} earnings report resolves a key "
                      f"open question in this thesis.")
    if val.get("coverage"):
        change.append("A material re-rating in valuation percentile (up or down) would change the "
                      "valuation group's contribution meaningfully.")
    if not change:
        change.append("A reversal in the technical state or a meaningful change in analyst sentiment "
                      "would be the first signs this thesis is weakening.")

    return {
        "thesis": thesis, "why_now": why_now, "idiosyncratic_risk": idiosyncratic_risk,
        "market_view": market_view, "zenith_view": zenith_view,
        "bull_case": bull_case, "bear_case": bear_case, "change_my_mind": change,
    }

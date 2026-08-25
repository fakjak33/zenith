"""IDEAS — a discretionary-systematic opportunity engine for Zenith.

Every other Zenith package answers one narrow question: MOMENTUM asks "what has
trend?", EDGE asks "what is heavily shorted / has moving estimates / expensive
options / a lottery profile?", PEAD asks "what beat and reacted?", FMOM asks
"which style is in favour?", CAS asks "what regime are we in?". Each commits a
ranked artifact under data/ for (most of) the Russell 1000, every trading day.
Nothing in the app reads more than one of those artifacts at a time.

IDEAS is the FUSION layer, not a new data pipeline: it reads the artifacts
above (see panel.py), combines them into a per-security panel, and asks a
different question — "what is genuinely interesting, not just high-scoring?"
Per Howard Marks' "uncomfortable" framing, a good company at a fair price
should rank BELOW an ordinary-looking situation the market may be mispricing.
Three numbers are kept deliberately separate rather than blended into one:

  * Conviction (0-100)   — is this a good idea, all signals weighted (conviction.py)
  * Unusualness (0-100)  — how EXTREME/rare is this setup (unusual.py)
  * Confluence (n/10)    — how many independent signal groups agree (confluence.py)

A "Divergence" flag (divergence.py) surfaces the specific case the spec calls
out as most interesting: price, fundamentals and sentiment disagreeing with
each other.

Every idea's thesis prose (narrative.py) is assembled deterministically from
the SAME numbers shown on its card — no LLM, no API key, nothing that could
state a fact the payload doesn't itself carry (config.py's data-honesty rule,
spec §29: "never silently fabricate a metric").

ANTI-OVERFITTING (spec §21), stated once here rather than scattered:
  1. Group weights (config.IDEAS_WEIGHTS) are set from each input's OWN
     documented evidence tier, a priori. There is no fitting/optimization loop
     anywhere in this package.
  2. The engine accumulates its OWN out-of-sample IC/hit-rate in
     diagnostics.json (mom/compute.py's _diagnostics is the template) and the
     UI shows that, never an in-sample backtest presented as evidence.
  3. Factor-pair correlations are computed and flagged >0.85, exactly as
     mom.engine.correlations already does for the momentum factors.
  4. Regime conditioning (config.IDEAS_REGIME_TILTS) is three coarse states,
     never a fitted regime model.
  5. HONEST DAY-ONE EVIDENCE TIER: C+. The individual inputs it fuses are
     B/B+; this composite itself is novel with zero out-of-sample record and
     earns promotion from its own accumulated diagnostics or not at all.

Universe: Russell 1000 (pretom.universe.russell1000, same source every other
Zenith package uses) + the CAS tagged ETF set (cas.universe.frm_universe,
~335 names). See config.IDEAS_UNIVERSE_SCOPE.
"""

from __future__ import annotations

import json

from ..config import IDEAS_FILES

DISCLAIMER = ("IDEAS fuses MOMENTUM/EDGE/PEAD/FMOM/CAS signals (free/best-effort data) into "
              "a daily BUY/SELL idea list with conviction, unusualness and confluence scores. "
              "This composite is novel and carries NO out-of-sample track record yet — see its "
              "evidence-strength badge and the accumulating diagnostics on the Diagnostics tab. "
              "Decision-support and a research prompt, not investment advice, and not a promise "
              "of forward returns.")

SIGNAL_GROUPS = ("technicals", "sentiment", "positioning", "valuation",
                  "fundamentals", "catalyst", "risk_reward", "macro")

GROUP_LABELS = {
    "technicals": "Technicals", "sentiment": "Sentiment", "positioning": "Positioning",
    "valuation": "Valuation", "fundamentals": "Fundamentals", "catalyst": "Catalyst",
    "risk_reward": "Risk/Reward", "macro": "Macro",
}

# Opportunity types (spec §4) — a rule tree in classify.py assigns exactly one
# per idea from the signal groups that actually fired for it.
OPPORTUNITY_TYPES = (
    "Dip Buy", "Contrarian", "Momentum", "Mean Reversion", "Value", "Growth",
    "Quality", "Turnaround", "Earnings Inflection", "Catalyst", "Secular Growth",
    "Cyclical Opportunity", "Defensive Opportunity", "Mispricing",
    "Sentiment Reversal", "Fundamental/Technical Divergence", "Factor Reversal",
    "Macro Opportunity", "Industry Rotation", "Relative Value",
    "Long-Term Compounder", "Short Thesis", "Risk-Off Hedge", "Event-Driven",
    "Broken Thesis / Avoid", "Speculative Asymmetric Opportunity",
)

# Thesis-status states (spec §9) — set by tracker.py as an idea's underlying
# signals move after it was generated.
THESIS_STATES = ("Strong", "Improving", "Intact", "Neutral", "Weakening", "Broken")

HORIZON_LABELS = {
    "weeks": "Several weeks", "months": "Several months",
    "6_18m": "6-18+ months", "long_term": "Long-term position build",
}

SECURITY_TYPES = ("stock", "etf")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(IDEAS_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write an IDEAS artefact. Large per-universe artefacts (candidates,
    universe_scores) should pass indent=None — matching every other Zenith
    package's size-budget convention (see mom/__init__.py's note)."""
    IDEAS_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    IDEAS_FILES[name].write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

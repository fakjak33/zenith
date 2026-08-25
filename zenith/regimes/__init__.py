"""REGIMES — macro regime intelligence & early-warning system for Zenith.

Every other Zenith feature answers a security-level question. CAS's own
regime layer answers a market-wide question but with a SINGLE scalar
(cas/signals/regime.py blends VIX/HY-OAS percentiles, the 10y-2y curve and
SPY's trend into one risk-on/neutral/risk-off number — one of six CAS
segments, and IDEAS's smallest-weighted input at 0.05). That is too thin to
answer what this package is built to answer:

    "What regime are we in, what regime may we be transitioning toward, what
    evidence suggests a transition is occurring, and what historically
    benefits or suffers in that environment?"

METHODOLOGY — a classic two-axis growth/inflation quadrant (the S&P-style
framework: many indicators per axis, a persistence requirement before a
quadrant is DECLARED rather than flickering on one noisy release) as the
TOP-LEVEL regime, running alongside six independently-scored secondary
dimensions (monetary, liquidity, credit, financial conditions, dollar,
volatility) that are not folded into the quadrant call — see series.py for
the ~45-indicator registry and classify.py for exactly how "rising" and
"falling" are defined (indicator-level 3-month momentum + breadth of
agreement, NOT the raw level — a quadrant is about DIRECTION, matching every
institutional framework surveyed, not "is inflation high").

POINT-IN-TIME DISCIPLINE (spec section 42, "avoid look-ahead bias"): every
series carries its own real publication lag (series.py) and is shifted by it
before use, so no historical month's classification uses data that had not
actually been published as of that month. This removes TIMING look-ahead.
It does NOT remove REVISION look-ahead — FRED revises payrolls/GDP/PCE for
years after first print — which is a separate, harder problem this package
does not silently paper over: vintage.py is an optional, LOCAL-ONLY audit
(needs a free FRED_API_KEY, never required by the nightly Action) that
measures how much revisions would have changed historical labels on a
handful of headline drivers, and publishes that residual as an explicit
calibration caveat rather than pretending the lag-shift alone made history
leak-free.

EXPLAINABILITY (spec section 43): the headline is never bare
"Regime: Stagflation". Every classification carries "growth: 7 of 12
indicators rising" / "inflation: 6 of 11 indicators rising" with every
indicator listed, because a regime call the user cannot audit is not
decision-support.

ANTI-OVERFITTING (spec section 42), stated once here rather than scattered,
same posture the IDEAS package already ships under:
  1. Dimension composition and direction signs (series.py) are set from each
     indicator's own economic meaning and publication quality, documented
     inline in config.py — never fitted to historical regime outcomes.
  2. Transition probabilities (Phase 2) are EMPIRICAL BASE RATES read off the
     reconstructed timeline with `n` always shown — no Markov/HMM fitting.
  3. A quadrant must PERSIST (config.REGIMES_PERSISTENCE_MONTHS) before it is
     declared current, so one noisy release cannot flip the headline.
  4. Calibration (Phase 2) is measured against NBER's own USREC recession
     dating — an external label this engine cannot influence.
  5. HONEST DAY-ONE EVIDENCE TIER: C+. Growth/inflation regime frameworks are
     well-established institutionally, but THIS implementation has zero
     out-of-sample record. It earns promotion from its own accuracy tracking
     (Phase 2) or not at all.

Historical reconstruction runs on a MONTHLY grid from config.REGIMES_HISTORY_START
(1990-01) forward — bounded so every reconstructed month has a defensible
indicator count rather than reaching for a "complete" history built on 2-3
series (see config.py's inline rationale, and the plan's verification
section: 1990-92 slowdown, 1994-95 tightening, 1998 LTCM, 2000-02 dotcom,
2008-09 GFC, 2020 COVID shock and 2021-22 inflation are the canonical
checkable cases a correct classifier must reproduce).
"""

from __future__ import annotations

import json

from ..config import REGIMES_FILES

DISCLAIMER = ("REGIMES classifies growth/inflation and six secondary macro dimensions from "
              "free FRED data (no API key). This is a NOVEL composite with no out-of-sample "
              "track record yet — see its evidence-strength badge and (Phase 2) the "
              "accumulating calibration-vs-NBER tracker. A regime call is a probabilistic read "
              "of many indicators, never a certainty, and never investment advice.")

# The four top-level quadrants, in the exact growth/inflation combination the
# spec's own table specifies (section 2).
REGIME_LABELS = {
    (True, False): "Goldilocks / Reflation",   # growth rising, inflation falling
    (True, True): "Overheating",               # growth rising, inflation rising
    (False, True): "Stagflation",              # growth falling, inflation rising
    (False, False): "Deflation / Slowdown",    # growth falling, inflation falling
}

# Eight dimensions this package scores. "growth" and "inflation" drive the
# quadrant; the other six are independently-labelled secondary regimes
# (spec section 12) that run alongside it, never folded into the quadrant.
DIMENSIONS = ("growth", "inflation", "monetary", "liquidity", "credit",
             "financial_conditions", "dollar", "volatility")

DIMENSION_LABELS = {
    "growth": "Growth", "inflation": "Inflation", "monetary": "Monetary Policy",
    "liquidity": "Liquidity", "credit": "Credit", "financial_conditions": "Financial Conditions",
    "dollar": "Dollar", "volatility": "Volatility",
}

# Secondary-dimension state labels: (positive-pole label, negative-pole label).
# "Positive" = the direction-adjusted composite z is ABOVE the neutral band —
# see series.py's `direction` field for what "+1 z" means per dimension
# (e.g. dollar: stronger dollar; volatility: higher volatility — directional
# conventions, not "good/bad" economic judgments the way growth/inflation are).
DIMENSION_STATES = {
    "monetary": ("Accommodative", "Restrictive"),
    "liquidity": ("Expanding", "Contracting"),
    "credit": ("Easy / Expansionary", "Tight / Contracting"),
    "financial_conditions": ("Loose", "Stressed"),
    "dollar": ("Strong / Appreciating", "Weak / Depreciating"),
    "volatility": ("Elevated / Expansion", "Compressed"),
}
DIMENSION_NEUTRAL_BAND = 0.5   # |z| below this -> "Neutral"


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(REGIMES_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write a REGIMES artefact. `macro_raw` (the committed warm-start cache,
    ~45 series x decades of daily/monthly points) should pass indent=None —
    same reasoning as mom/__init__.py's save(): pretty-printing a large,
    rarely-hand-read JSON artefact just adds git history for no benefit."""
    REGIMES_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    REGIMES_FILES[name].write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

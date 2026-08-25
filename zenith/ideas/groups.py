"""IDEAS group scorers: the eight signal groups, each -> ([-1,1] score, coverage, explain).

Every scorer takes one panel row (panel.py's per-security dict) and returns a
score, a coverage flag (real data found or not -- never a fabricated
neutral), and an `explain` dict whose fields narrative.py cites directly, so
every sentence the thesis prose writes traces to a number that is actually in
the payload (spec section 29).

None of these scorers is fitted to historical returns -- see ideas/__init__.py
for the anti-overfitting note this whole package follows.
"""

from __future__ import annotations

from ..edge.common import finite
from . import riskreward
from . import valuation as val

_NEUTRAL = (0.0, False, {})


# ------------------------------------------------------------------ technicals
def technicals(row: dict) -> tuple[float, bool, dict]:
    tech = row.get("technicals")
    if not tech or tech.get("composite") is None:
        return _NEUTRAL
    score = max(-1.0, min(1.0, float(tech["composite"]) / 20.0))
    return score, True, {"composite": tech["composite"], "state": tech.get("state"),
                         "pctile": tech.get("pctile"), "source": tech.get("source", "mom")}


# ------------------------------------------------------------------- sentiment
def sentiment(row: dict) -> tuple[float, bool, dict]:
    s = row.get("sentiment")
    if not s:
        return _NEUTRAL
    parts, weights, explain = {}, {}, {}
    rev = s.get("revisions")
    if rev and finite(rev.get("pctile")):
        parts["revisions"] = (rev["pctile"] - 50.0) / 50.0
        weights["revisions"] = 0.65
        explain["revisions_pctile"] = rev["pctile"]
        explain["est_rev_pct"] = rev.get("est_rev_pct")
        explain["up_frac"] = rev.get("up_frac")
    ivs = s.get("ivspread")
    if ivs and finite(ivs.get("pctile")):
        parts["ivspread"] = (ivs["pctile"] - 50.0) / 50.0
        weights["ivspread"] = 0.35
        explain["iv_spread_bp"] = ivs.get("iv_spread_bp")
        explain["borrow_flag"] = ivs.get("borrow_flag")
    if not parts:
        return _NEUTRAL
    tw = sum(weights[k] for k in parts)
    score = sum(parts[k] * weights[k] for k in parts) / tw
    return max(-1.0, min(1.0, score)), True, explain


# ----------------------------------------------------------------- positioning
def positioning(row: dict) -> tuple[float, bool, dict]:
    p = row.get("positioning")
    if not p:
        return _NEUTRAL
    si = p.get("shortint")
    if not si or not finite(si.get("si_pctile")):
        return _NEUTRAL
    # shortint's own composite ranks MORE-shorted names higher (higher_is_long=False
    # in edge/shortint.py) -- so a high percentile here is a BEARISH read on price
    score = -((si["si_pctile"] - 50.0) / 50.0)
    return max(-1.0, min(1.0, score)), True, {
        "si_float": si.get("si_float"), "dtc": si.get("dtc"),
        "si_pctile": si.get("si_pctile"), "squeeze_risk": si.get("squeeze_risk"),
        "held_pct_institutions": p.get("held_pct_institutions"),
    }


# ------------------------------------------------------------------- valuation
def valuation(row: dict) -> tuple[float, bool, dict]:
    v = row.get("valuation")
    if not v:
        return _NEUTRAL
    score, detail = val.blend(v.get("cross_sectional"), v.get("own_history"))
    if not detail.get("coverage"):
        return _NEUTRAL
    return score, True, {"cross_sectional": v.get("cross_sectional"),
                         "own_history": v.get("own_history"), "blend": detail}


# ----------------------------------------------------------------- fundamentals
# Absolute-threshold quality/growth/leverage overlay (not cross-sectionally
# ranked, unlike valuation -- .info fields alone don't carry the batch needed
# for that in every run, and quality thresholds are reasonably universal
# across a mega-cap-to-mid-cap Russell 1000 sample). Centers chosen from
# textbook "healthy company" ranges, not fitted to any return sample.
_FUND_CENTERS = {
    "returnOnEquity": (0.12, 0.15),      # (center, scale) for tanh((x-center)/scale)
    "profitMargins": (0.08, 0.10),
    "revenueGrowth": (0.05, 0.15),
    "earningsGrowth": (0.05, 0.20),
    "debtToEquity": (100.0, -80.0),      # inverted: lower D/E is better (yfinance reports as a %, e.g. 60 = 0.6x)
    "currentRatio": (1.3, 0.6),
}
_FUND_WEIGHTS = {"returnOnEquity": 0.25, "profitMargins": 0.20, "revenueGrowth": 0.20,
                 "earningsGrowth": 0.15, "debtToEquity": 0.10, "currentRatio": 0.10}


def fundamentals(row: dict) -> tuple[float, bool, dict]:
    f = row.get("fundamentals")
    if not f:
        return _NEUTRAL
    import math
    parts, explain = {}, {}
    for field, (center, scale) in _FUND_CENTERS.items():
        x = f.get(field)
        if not finite(x):
            continue
        parts[field] = math.tanh((float(x) - center) / scale)
        explain[field] = x
    if not parts:
        return _NEUTRAL
    tw = sum(_FUND_WEIGHTS[k] for k in parts)
    score = sum(parts[k] * _FUND_WEIGHTS[k] for k in parts) / tw
    fcf = f.get("freeCashflow")
    mc = f.get("marketCap")
    if finite(fcf) and finite(mc) and mc:
        explain["fcf_yield"] = round(float(fcf) / float(mc), 4)
    return max(-1.0, min(1.0, score)), True, explain


# --------------------------------------------------------------------- catalyst
def catalyst(row: dict) -> tuple[float, bool, dict]:
    c = row.get("catalyst")
    if not c:
        return _NEUTRAL
    parts, weights, explain = {}, {}, {}
    rec = c.get("recent")
    if rec and finite(rec.get("composite")) and rec.get("side") in ("long", "short"):
        # PEAD's composite is already direction-signed: 50=neutral, 100=strong
        # in the direction of `side`. Flip back to a signed [-1,1] read here.
        mag = (rec["composite"] - 50.0) / 50.0
        parts["recent_reaction"] = mag if rec["side"] == "long" else -mag
        weights["recent_reaction"] = 0.7
        explain["recent_report_date"] = rec.get("report_date")
        explain["recent_side"] = rec.get("side")
        explain["recent_composite"] = rec.get("composite")
    up = c.get("upcoming")
    if up and finite(up.get("past_avg_excess")):
        import math
        parts["upcoming_prior"] = math.tanh(up["past_avg_excess"] / 0.03)
        weights["upcoming_prior"] = 0.3
        explain["upcoming_report_date"] = up.get("report_date")
        explain["upcoming_past_avg_excess"] = up.get("past_avg_excess")
        explain["upcoming_past_n"] = up.get("past_n")
    if not parts:
        return _NEUTRAL
    tw = sum(weights[k] for k in parts)
    score = sum(parts[k] * weights[k] for k in parts) / tw
    return max(-1.0, min(1.0, score)), True, explain


# ------------------------------------------------------------------ risk_reward
def risk_reward(row: dict) -> tuple[float, bool, dict]:
    score, detail = riskreward.proxy_score(row)
    if not detail.get("coverage"):
        return _NEUTRAL
    return score, True, detail


# ----------------------------------------------------------------------- macro
def macro(row: dict, regime_summary: dict) -> tuple[float, bool, dict]:
    from . import regime as ideas_regime
    beta = None
    fund = row.get("fundamentals")
    if fund and finite(fund.get("beta")):
        beta = fund["beta"]
    elif row.get("lottery") and finite(row["lottery"].get("beta")):
        beta = row["lottery"]["beta"]
    score, explain = ideas_regime.macro_score(regime_summary, beta)
    return score, True, explain


SCORERS = {
    "technicals": lambda row, regime_summary: technicals(row),
    "sentiment": lambda row, regime_summary: sentiment(row),
    "positioning": lambda row, regime_summary: positioning(row),
    "valuation": lambda row, regime_summary: valuation(row),
    "fundamentals": lambda row, regime_summary: fundamentals(row),
    "catalyst": lambda row, regime_summary: catalyst(row),
    "risk_reward": lambda row, regime_summary: risk_reward(row),
    "macro": lambda row, regime_summary: macro(row, regime_summary),
}


def score_all(row: dict, regime_summary: dict) -> dict[str, dict]:
    """Every group's (score, coverage, explain) for one security."""
    out = {}
    for name, fn in SCORERS.items():
        score, cov, explain = fn(row, regime_summary)
        out[name] = {"score": round(score, 4), "coverage": cov, "explain": explain}
    return out

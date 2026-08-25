"""Offline unit tests for zenith.ideas — pure-function logic only, no network."""

from __future__ import annotations

import math

from zenith.ideas import (conviction, unusual, confluence, divergence, classify,
                          riskreward, valuation, groups)
from zenith.config import IDEAS_WEIGHTS


# ---------------------------------------------------------------- conviction
def test_conviction_weights_sum_to_one():
    assert abs(sum(IDEAS_WEIGHTS.values()) - 1.0) < 1e-9


def test_conviction_coverage_aware_renormalization():
    # only technicals covered -> net_score should equal technicals' own score,
    # not be dragged toward zero by the uncovered groups
    gs = {"technicals": {"score": 0.5, "coverage": True},
         "sentiment": {"score": 0.0, "coverage": False}}
    result = conviction.compute(gs, "neutral / transition")
    assert result["coverage_n"] == 1
    assert abs(result["net_score"] - 0.5) < 1e-9


def test_conviction_no_coverage_returns_neutral():
    gs = {k: {"score": 0.0, "coverage": False} for k in IDEAS_WEIGHTS}
    result = conviction.compute(gs, "neutral / transition")
    assert result["net_score"] == 0.0
    assert result["coverage_n"] == 0


def test_conviction_magnitude_bounds():
    assert conviction.magnitude(0.0) == 50.0
    assert conviction.magnitude(1.0) == 100.0
    assert conviction.magnitude(-1.0) == 100.0
    assert conviction.magnitude(0.5) == 75.0


def test_conviction_regime_tilt_changes_weight():
    gs = {"technicals": {"score": 0.5, "coverage": True}}
    neutral = conviction.tilted_weights("neutral / transition")
    risk_on = conviction.tilted_weights("risk-on")
    assert neutral["technicals"] != risk_on["technicals"]


# ------------------------------------------------------------------ unusual
def test_unusual_requires_coverage():
    out = unusual.compute({}, "AAPL")
    assert out["unusual"] == 0.0
    assert out["n_groups"] == 0


def test_unusual_extreme_scores_score_higher():
    mild = {"technicals": {"score": 0.1, "coverage": True}}
    extreme = {"technicals": {"score": 0.9, "coverage": True}}
    assert unusual.compute(extreme, "XYZ")["unusual"] > unusual.compute(mild, "XYZ")["unusual"]


def test_unusual_obvious_ticker_discounted():
    gs = {"technicals": {"score": 0.5, "coverage": True}}
    plain = unusual.compute(gs, "ZZZZ")
    obvious = unusual.compute(gs, "SPY")
    assert obvious["obvious_discount"] is True
    assert obvious["unusual"] < plain["unusual"]


# --------------------------------------------------------------- confluence
def test_confluence_counts_agreement():
    gs = {
        "technicals": {"score": 0.3, "coverage": True},
        "sentiment": {"score": 0.2, "coverage": True},
        "positioning": {"score": -0.3, "coverage": True},
        "valuation": {"score": 0.0, "coverage": False},
    }
    out = confluence.compute(gs, net_score=0.4)
    assert out["direction"] == "bullish"
    assert out["agree"] == 2          # technicals + sentiment
    assert out["disagree"] == 1       # positioning
    assert out["no_data"] == 5        # valuation + the 4 groups not in gs at all
    assert out["n_total"] == 8


def test_confluence_label_format():
    gs = {"technicals": {"score": 0.3, "coverage": True}}
    out = confluence.compute(gs, net_score=0.3)
    assert out["label"].endswith("/8 signals bullish")


# --------------------------------------------------------------- divergence
def test_divergence_price_below_fundamentals():
    gs = {"technicals": {"score": -0.3, "coverage": True},
         "fundamentals": {"score": 0.3, "coverage": True}}
    out = divergence.compute(gs)
    assert out["has_divergence"] is True
    assert out["flags"][0]["type"] == "price_below_fundamentals"


def test_divergence_none_when_aligned():
    gs = {"technicals": {"score": 0.3, "coverage": True},
         "fundamentals": {"score": 0.3, "coverage": True}}
    out = divergence.compute(gs)
    assert out["has_divergence"] is False


def test_divergence_requires_both_groups_covered():
    gs = {"technicals": {"score": -0.9, "coverage": True},
         "fundamentals": {"score": 0.0, "coverage": False}}
    out = divergence.compute(gs)
    assert out["has_divergence"] is False


# ----------------------------------------------------------------- classify
def test_classify_long_returns_registered_type():
    gs = {"technicals": {"score": 0.3, "coverage": True}}
    out = classify.classify("long", gs, {"flags": []})
    from zenith.ideas import OPPORTUNITY_TYPES
    assert out["opportunity_type"] in OPPORTUNITY_TYPES
    assert out["horizon"] in ("weeks", "months", "6_18m", "long_term")


def test_classify_short_returns_registered_type():
    gs = {"technicals": {"score": -0.3, "coverage": True}}
    out = classify.classify("short", gs, {"flags": []})
    from zenith.ideas import OPPORTUNITY_TYPES
    assert out["opportunity_type"] in OPPORTUNITY_TYPES


def test_classify_catalyst_near_term_is_weeks_horizon():
    gs = {"catalyst": {"score": 0.5, "coverage": True}}
    out = classify.classify("long", gs, {"flags": []}, catalyst_days_out=5)
    assert out["opportunity_type"] == "Catalyst"
    assert out["horizon"] == "weeks"


# --------------------------------------------------------------- riskreward
def test_riskreward_proxy_no_coverage():
    score, detail = riskreward.proxy_score({})
    assert score == 0.0
    assert detail["coverage"] is False


def test_riskreward_proxy_favorable_structure_is_positive():
    row = {"technicals": {"detail": {"last_close": 100.0},
                          "breakout_grid": {"12m": {"high": 140.0, "low": 95.0}}}}
    score, detail = riskreward.proxy_score(row)
    assert detail["coverage"] is True
    assert score > 0        # upside (40) far exceeds downside (5) -> favorable R:R


def test_riskreward_build_insufficient_history_returns_unavailable():
    out = riskreward.build("XYZ", "long", None)
    assert out["available"] is False


def test_riskreward_build_long_stop_never_above_entry_near_own_low():
    # regression: a stock sitting right at its own 63d low (structural support
    # ~= entry) must not collapse the stop toward/above the entry price, which
    # would blow out the R/R ratio (caught live by scripts/screen.py on WSO/AVGO)
    import numpy as np
    import pandas as pd
    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(100, 60, n - 5).tolist() + [61, 62, 63, 64, 65], index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1_000_000}, index=idx)
    out = riskreward.build("TEST", "long", df, state="NEUTRAL")
    assert out["available"] is True
    assert out["stop"] < out["last_close"]
    assert out["risk_per_share"] > 0
    assert out["rr_ratio"] < 20.0


def test_riskreward_build_short_stop_never_below_entry_near_own_high():
    import numpy as np
    import pandas as pd
    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(60, 100, n - 5).tolist() + [99, 98, 97, 96, 95], index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1_000_000}, index=idx)
    out = riskreward.build("TEST", "short", df, state="NEUTRAL")
    assert out["available"] is True
    assert out["stop"] > out["last_close"]
    assert out["risk_per_share"] > 0
    assert out["rr_ratio"] < 20.0


# ---------------------------------------------------------------- valuation
def test_valuation_multiples_of_handles_missing_fields():
    out = valuation.multiples_of({"trailingPE": None, "priceToBook": 2.0})
    assert out["ep"] is None
    assert out["book_yield"] == 0.5


def test_valuation_blend_no_coverage():
    score, detail = valuation.blend(None, None)
    assert score == 0.0
    assert detail["coverage"] is False


def test_valuation_blend_cross_sectional_only():
    cross = {"universe_pctile": 80.0, "sector_pctile": None}
    score, detail = valuation.blend(cross, None)
    assert detail["coverage"] is True
    assert score > 0


def test_valuation_own_history_building_state_below_min_n():
    out = valuation.own_history_percentile("AAPL", {"ep": 0.05}, hist={"AAPL": [{"month": "2026-01", "ep": 0.04}]})
    assert out["state"] == "building"
    assert out["pctile"] is None


# ------------------------------------------------------------------- groups
def test_groups_technicals_no_data_neutral():
    score, cov, explain = groups.technicals({})
    assert score == 0.0 and cov is False


def test_groups_technicals_scales_composite():
    score, cov, explain = groups.technicals({"technicals": {"composite": 10.0, "state": "BULLISH"}})
    assert cov is True
    assert abs(score - 0.5) < 1e-9


def test_groups_score_all_returns_all_eight():
    from zenith.ideas import SIGNAL_GROUPS
    out = groups.score_all({}, {"label": "neutral / transition", "risk_score": 0.0})
    assert set(out.keys()) == set(SIGNAL_GROUPS)
    for g in SIGNAL_GROUPS:
        assert "score" in out[g] and "coverage" in out[g] and "explain" in out[g]
        assert not math.isnan(out[g]["score"])

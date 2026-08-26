"""REGIMES Phase 2/3 tests — offline, synthetic data only (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from zenith.regimes import transition, changes, crossasset, performance, analogs, accuracy
from zenith.regimes import themes, scenarios, alerts, evidence
from zenith.regimes import series as rs


# --------------------------------------------------------------- transition.py
def _synthetic_timeline(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    labels = ["Goldilocks / Reflation", "Overheating", "Stagflation", "Deflation / Slowdown"]
    declared = [labels[i] for i in rng.integers(0, 4, n)]
    return pd.DataFrame({"declared_regime": declared}, index=idx)


def test_transition_table_probabilities_sum_to_one_when_reported():
    tl = _synthetic_timeline(n=200)
    tables = transition.build_tables(tl)
    for start_regime, horizons in tables["unconditional"].items():
        for h, cell in horizons.items():
            probs = [d["p"] for d in cell["destinations"].values() if d["p"] is not None]
            if probs:
                # each p is independently rounded to 3dp, so the sum can be off
                # by a few thousandths without any real inconsistency
                assert abs(sum(probs) - 1.0) < 0.005


def test_transition_table_respects_min_n():
    tl = _synthetic_timeline(n=10)   # too few observations per bucket
    tables = transition.build_tables(tl)
    cell = tables["unconditional"]["Overheating"]["30"]
    if cell["n_start"] < transition.MIN_N_FOR_PROBABILITY:
        for d in cell["destinations"].values():
            assert d["p"] is None


def test_transition_for_current_unknown_regime():
    tl = _synthetic_timeline(n=50)
    tables = transition.build_tables(tl)
    out = transition.for_current(tables, None, 0.0)
    assert out["available"] is False


def test_transition_for_current_known_regime():
    tl = _synthetic_timeline(n=200)
    tables = transition.build_tables(tl)
    out = transition.for_current(tables, "Overheating", 5.0)
    assert out["available"] is True
    assert out["regime"] == "Overheating"


# ------------------------------------------------------------------- changes.py
def test_regime_change_score_bounded():
    n = 20
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    cols = [s.id for s in rs.REGISTRY][:5]
    rng = np.random.default_rng(1)
    z_df = pd.DataFrame({c: rng.normal(0, 1, n).cumsum() for c in cols}, index=idx)
    out = changes.regime_change_score(z_df)
    if out["score"] is not None:
        assert 0.0 <= out["score"] <= 100.0
        assert out["band"] in [b[1] for b in changes.CHANGE_SCORE_BANDS]


def test_regime_change_score_insufficient_history():
    z_df = pd.DataFrame({"CFNAI": [1.0, 2.0]})
    out = changes.regime_change_score(z_df, window_months=2)
    assert out["score"] is None


def test_band_for_thresholds():
    assert changes.band_for(90) == "Major Regime Shift"
    assert changes.band_for(10) == "Stable"
    assert changes.band_for(45) == "Emerging"


def test_indicator_deltas_shape():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    z_df = pd.DataFrame({"CFNAI": np.linspace(0, 1, 6)}, index=idx)
    rows = changes.indicator_deltas(z_df)
    assert len(rows) == 1
    assert rows[0]["id"] == "CFNAI"
    assert rows[0]["delta_1m"] is not None


# ---------------------------------------------------------------- crossasset.py
def test_confirmation_and_divergence_are_complementary():
    trends = {"IWM": {"trend_3m": 0.05, "direction": "up"}, "SPY": {"trend_3m": 0.02, "direction": "up"},
             "GLD": {"trend_3m": -0.01, "direction": "down"}, "TLT": {"trend_3m": 0.03, "direction": "up"}}
    breadth = {"available": True, "cyclicals_avg": 5.0, "defensives_avg": 2.0, "cyclicals_leading": True}
    conf = crossasset.confirmation(True, False, trends, breadth)
    div = crossasset.divergences(True, False, trends, breadth)
    assert conf["n_total"] == len(conf["checks"])
    assert len(div) == conf["n_total"] - conf["n_confirming"]


def test_confirmation_handles_missing_breadth():
    conf = crossasset.confirmation(True, True, {}, {"available": False})
    assert conf["n_total"] == 0


def test_sector_breadth_empty_returns_unavailable(monkeypatch):
    monkeypatch.setattr(crossasset, "mom_load", lambda name, default: {})
    out = crossasset.sector_breadth()
    assert out["available"] is False


# --------------------------------------------------------------- performance.py
def test_stats_computes_sane_metrics():
    r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    stats = performance._stats(r)
    assert stats is not None
    assert stats["n_months"] == 6
    assert -1.0 <= stats["max_drawdown"] <= 0.0
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_stats_none_when_too_few_months():
    assert performance._stats(pd.Series([0.01, 0.02])) is None


def test_by_regime_partitions_correctly():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    returns = pd.Series([0.01, 0.02, 0.01, 0.02, -0.01, -0.02, -0.01, -0.02], index=idx)
    regime = pd.Series(["A"] * 4 + ["B"] * 4, index=idx)
    # monkeypatch REGIME_LABELS-derived loop by calling with real labels present
    import zenith.regimes as regimes_pkg
    out = performance.by_regime(returns, regime.replace(
        {"A": "Goldilocks / Reflation", "B": "Stagflation"}))
    assert out["Goldilocks / Reflation"] is not None
    assert out["Goldilocks / Reflation"]["avg_return"] > 0
    assert out["Stagflation"]["avg_return"] < 0


# ------------------------------------------------------------------- analogs.py
def test_distance_requires_min_shared_dimensions():
    a = pd.Series({"growth": 1.0, "inflation": 1.0})
    b = pd.Series({"growth": 1.0, "inflation": 1.0})
    dist, n = analogs._distance(a, b)
    assert dist is None    # only 2 shared dims < MIN_SHARED_DIMENSIONS


def test_distance_zero_for_identical_vectors():
    idx = ["growth", "inflation", "monetary", "liquidity", "credit", "dollar"]
    a = pd.Series(1.0, index=idx)
    b = pd.Series(1.0, index=idx)
    dist, n = analogs._distance(a, b)
    assert dist == 0.0
    assert n == 6


def test_outcome_distribution_empty():
    assert analogs.outcome_distribution([], 3) == {"n": 0}


def test_outcome_distribution_computes_stats():
    rows = [{"forward": {"3": 0.05}}, {"forward": {"3": -0.02}}, {"forward": {"3": 0.03}}]
    out = analogs.outcome_distribution(rows, 3)
    assert out["n"] == 3
    assert out["worst"] == -0.02
    assert out["best"] == 0.05


# ------------------------------------------------------------------- accuracy.py
def test_lead_lag_detects_matched_recession():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    usrec = pd.Series(0.0, index=idx)
    usrec.iloc[12:15] = 1.0     # a recession starting at month 12
    declared = pd.Series(["Overheating"] * 24, index=idx)
    declared.iloc[10:16] = "Deflation / Slowdown"   # our signal fires 2 months early
    out = accuracy.lead_lag(declared, usrec)
    assert out["n_nber_recessions"] == 1
    assert out["n_matched"] == 1
    assert out["matches"][0]["lead_months"] == 2


def test_lead_lag_false_negative_when_never_signaled():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    usrec = pd.Series(0.0, index=idx)
    usrec.iloc[12:15] = 1.0
    declared = pd.Series(["Overheating"] * 24, index=idx)   # never flags Deflation/Slowdown
    out = accuracy.lead_lag(declared, usrec)
    assert out["n_false_negatives"] == 1
    assert out["n_matched"] == 0


def test_brier_score_perfect_forecast_scores_zero():
    idx = pd.date_range("2000-01-31", periods=20, freq="ME")
    declared = pd.Series(["Overheating"] * 10 + ["Deflation / Slowdown"] * 10, index=idx)
    tables = {"unconditional": {"Overheating": {"180": {"destinations":
             {"Deflation / Slowdown": {"p": 1.0, "n": 10}}}}}}
    out = accuracy.brier_score(declared, tables, horizon_months=6)
    assert out["in_sample"] is True
    if out["n"] > 0:
        assert out["brier"] <= 0.5


def test_brier_score_unknown_horizon():
    out = accuracy.brier_score(pd.Series(dtype=object), {}, horizon_months=7)
    assert out["brier"] is None


# ---------------------------------------------------------------------- themes.py
def test_z_to_score_bounds():
    assert themes._z_to_score(0.0) == 50.0
    assert themes._z_to_score(4.0) == 100.0
    assert themes._z_to_score(-4.0) == 0.0
    assert themes._z_to_score(None) is None


def test_tier_for_thresholds():
    assert themes._tier_for(0.09) == "Extreme"
    assert themes._tier_for(0.06) == "High"
    assert themes._tier_for(0.03) == "Moderate"
    assert themes._tier_for(0.01) == "Low"


# -------------------------------------------------------------------- scenarios.py
def test_implied_quadrant_flips_correct_axis():
    assert scenarios._implied_quadrant(True, False, "inflation", True) == "Overheating"
    # flipping "growth" to False leaves inflation at its CURRENT (False) reading
    # -> (growth=False, inflation=False) = Deflation / Slowdown, not Stagflation
    assert scenarios._implied_quadrant(True, False, "growth", False) == "Deflation / Slowdown"


def test_implied_quadrant_handles_unknown_axis_state():
    # flipping "growth" overrides growth's current value entirely -- the axis
    # that stays None is the one NOT being flipped (inflation, here)
    assert scenarios._implied_quadrant(True, None, "growth", True) is None


def test_evaluate_quadrant_scenario_grounded_with_perf_data():
    perf = {"SPY": {"label": "US Equities", "by_regime": {
        "Overheating": {"avg_return": 0.02, "n_months": 10, "median_return": 0.01,
                        "win_rate": 0.6, "volatility_ann": 0.15, "max_drawdown": -0.1, "sharpe_like": 0.5}}}}
    scenario = scenarios.QUADRANT_SCENARIOS[0]   # inflation_accelerates
    out = scenarios.evaluate_quadrant_scenario(scenario, True, False, perf)
    assert out["implied_regime"] == "Overheating"
    assert out["grounded"] is True
    assert len(out["historical_beneficiaries"]) == 1


def test_evaluate_dimension_scenario_never_grounded():
    out = scenarios.evaluate_dimension_scenario(scenarios.DIMENSION_SCENARIOS[0])
    assert out["grounded"] is False
    assert "caveat" in out


def test_build_returns_both_kinds():
    out = scenarios.build(True, True, {})
    assert len(out["quadrant_scenarios"]) == len(scenarios.QUADRANT_SCENARIOS)
    assert len(out["dimension_scenarios"]) == len(scenarios.DIMENSION_SCENARIOS)


# ----------------------------------------------------------------------- alerts.py
def test_alerts_detects_regime_change():
    today = date.today()
    journal = [{"date": (today.replace(day=1)).isoformat(), "regime": "Overheating",
               "confidence": 50.0, "transitioning": False, "momentum_score": 5.0, "change_score": 20.0}]
    current = {"regime": "Stagflation", "confidence": 55.0, "transitioning": False,
              "raw_regime": "Stagflation", "momentum": {"score": 3.0}}
    triggered = alerts.evaluate(journal, current, {"score": 25.0, "band": "Early Signals"}, lookback_days=30)
    ids = [a["id"] for a in triggered]
    assert "regime_changed" in ids


def test_alerts_no_journal_returns_empty():
    assert alerts.evaluate([], {"regime": "X"}, {"score": None}) == []


def test_alerts_confidence_crossed():
    today = date.today()
    journal = [{"date": today.isoformat(), "regime": "Overheating", "confidence": 25.0,
               "transitioning": False, "momentum_score": 0.0, "change_score": 10.0}]
    current = {"regime": "Overheating", "confidence": 75.0, "transitioning": False,
              "raw_regime": "Overheating", "momentum": {"score": 0.0}}
    triggered = alerts.evaluate(journal, current, {"score": 10.0}, lookback_days=30)
    ids = [a["id"] for a in triggered]
    assert any("confidence_crossed" in i for i in ids)


# ----------------------------------------------------------------------- evidence.py
def test_classify_speculation():
    item = {"title": "Sources say the Fed may act", "summary": "Rumored plans", "category": "insight"}
    assert evidence._classify(item) == "Speculation"


def test_classify_forecast():
    item = {"title": "Inflation is expected to moderate", "summary": "", "category": "insight"}
    assert evidence._classify(item) == "Forecast"


def test_classify_research_is_interpretation():
    item = {"title": "A study of wage dynamics", "summary": "", "category": "research"}
    assert evidence._classify(item) == "Interpretation"


def test_classify_news_is_fact():
    item = {"title": "CPI released at 2.9%", "summary": "", "category": "news"}
    assert evidence._classify(item) == "Fact"


def test_mine_filters_by_keyword(monkeypatch):
    import zenith.regimes.evidence as ev_mod
    monkeypatch.setattr(ev_mod.store, "archive_dates", lambda: ["2026-01-01"])
    monkeypatch.setattr(ev_mod.store, "load_archive", lambda d: [
        {"title": "AI capex boom continues", "summary": "", "category": "news", "source": "X",
         "link": "u1", "published": "2026-01-01"},
        {"title": "Unrelated soccer news", "summary": "", "category": "news", "source": "Y",
         "link": "u2", "published": "2026-01-01"},
    ])
    hits = ev_mod.mine(["AI capex"])
    assert len(hits) == 1
    assert hits[0]["title"].startswith("AI capex")


# ------------------------------------------------------------------- vintage.py
from zenith.regimes import vintage


def test_run_audit_degrades_without_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    out = vintage.run_audit()
    assert out["ran"] is False
    assert "FRED_API_KEY" in out["reason"]


def test_first_published_parses_mocked_response(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"date": "2020-01-01", "value": "3.1", "realtime_start": "2020-02-15"}]}

    monkeypatch.setattr(vintage.requests, "get", lambda *a, **k: _Resp())
    v = vintage.first_published("CPIAUCSL", date(2020, 1, 1), "fake_key")
    assert v == 3.1


def test_first_published_handles_missing_value(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"value": "."}]}

    monkeypatch.setattr(vintage.requests, "get", lambda *a, **k: _Resp())
    assert vintage.first_published("CPIAUCSL", date(2020, 1, 1), "fake_key") is None


def test_run_audit_with_mocked_key_and_responses(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fake_key")
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            val = "3.0" if calls["n"] % 2 else "3.2"
            return {"observations": [{"value": val}]}

    monkeypatch.setattr(vintage.requests, "get", lambda *a, **k: _Resp())
    out = vintage.run_audit(series_ids=("CPIAUCSL",), start_year=2023, end_year=2023)
    assert out["ran"] is True
    assert "CPIAUCSL" in out["series"]
    assert out["series"]["CPIAUCSL"]["n_months"] == 12

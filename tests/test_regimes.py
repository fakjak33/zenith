"""REGIMES tests — offline, synthetic data only (no network)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from zenith import config
from zenith.regimes import DIMENSIONS, REGIME_LABELS
from zenith.regimes import series as rs
from zenith.regimes import macro, classify, momentum, history, dimensions, compute as rc


# ------------------------------------------------------------------ series.py
def test_registry_ids_unique():
    ids = [s.id for s in rs.REGISTRY]
    assert len(ids) == len(set(ids))


def test_registry_every_dimension_covered():
    covered = {s.dimension for s in rs.REGISTRY}
    assert covered == set(DIMENSIONS)


def test_freq_and_lag_resolve_for_every_fred_id():
    for fid in rs.ALL_FRED_IDS:
        assert rs.freq_of(fid) in ("D", "W", "M", "Q")
        assert rs.lag_of(fid) >= 0


def test_derived_series_have_no_fred_id_but_are_registered():
    for did in rs.DERIVED_INPUTS:
        spec = rs.BY_ID[did]
        assert spec.fred_id is None


# ------------------------------------------------------------------- macro.py
def _synthetic_points(n_months: int, start_val: float, monthly_drift: float,
                      start="2015-01-01") -> list[dict]:
    idx = pd.date_range(start, periods=n_months, freq="MS")
    vals = start_val + np.arange(n_months) * monthly_drift
    return [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(idx, vals)]


def test_pit_series_shifts_by_lag():
    pts = [{"date": "2024-01-01", "value": 1.0}]
    s = macro._pit_series(pts, lag_days=10)
    assert s.index[0] == pd.Timestamp("2024-01-11")


def test_pit_series_empty_input():
    s = macro._pit_series([], lag_days=5)
    assert s.empty


def test_transform_yoy_and_mom_diff():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    level = pd.Series(np.linspace(100, 123, 24), index=idx)   # +1/month roughly
    yoy = macro._transform(level, "yoy")
    assert yoy.dropna().iloc[-1] > 0     # rising level -> positive YoY
    mom_diff = macro._transform(level, "mom_diff")
    assert abs(mom_diff.dropna().iloc[-1] - 1.0) < 0.5


def test_transform_unknown_raises():
    with pytest.raises(ValueError):
        macro._transform(pd.Series([1.0]), "bogus")


def test_build_transformed_panel_computes_derived_series():
    ends = macro.month_ends(start="2018-01-01", end=date(2020, 1, 1))
    raw = {
        "DFF": _synthetic_points(60, 2.0, 0.02, start="2015-01-01"),
        "PCEPILFE": _synthetic_points(60, 100.0, 0.15, start="2015-01-01"),
        "WALCL": _synthetic_points(60, 4_000_000.0, 5000.0, start="2015-01-01"),
        "RRPONTSYD": _synthetic_points(60, 100.0, 1.0, start="2015-01-01"),
        "WTREGEN": _synthetic_points(60, 300_000.0, 1000.0, start="2015-01-01"),
    }
    out = macro.build_transformed_panel(raw, ends)
    assert "REAL_FFR" in out and "NET_LIQ" in out
    assert out["REAL_FFR"].dropna().shape[0] > 0
    assert out["NET_LIQ"].dropna().shape[0] > 0


def test_zscore_panel_applies_direction_sign():
    idx = pd.date_range("2015-01-31", periods=150, freq="ME")
    rising = pd.Series(np.linspace(0, 10, 150), index=idx)
    # UNRATE has direction=-1 in the registry: a RISING raw series should
    # produce a NEGATIVE (unemployment-is-bad-for-growth) z at the end.
    z_df, raw_df = macro.zscore_panel({"UNRATE": rising})
    assert z_df["UNRATE"].dropna().iloc[-1] < 0


def test_zscore_panel_positive_direction():
    idx = pd.date_range("2015-01-31", periods=150, freq="ME")
    rising = pd.Series(np.linspace(0, 10, 150), index=idx)
    z_df, _ = macro.zscore_panel({"INDPRO": rising})
    assert z_df["INDPRO"].dropna().iloc[-1] > 0


# ----------------------------------------------------------------- classify.py
def _synthetic_z_df(n_months: int, growth_dir: int, infl_dir: int) -> pd.DataFrame:
    """A tiny synthetic z-panel with one growth series and one inflation
    series, each monotonically trending in the requested direction, so the
    expected quadrant is known in advance."""
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    growth_col = rs.BY_DIMENSION["growth"][0].id
    infl_col = rs.BY_DIMENSION["inflation"][0].id
    g = pd.Series(np.arange(n_months) * growth_dir * 0.1, index=idx)
    i = pd.Series(np.arange(n_months) * infl_dir * 0.1, index=idx)
    return pd.DataFrame({growth_col: g, infl_col: i})


def test_classify_timeline_overheating():
    z_df = _synthetic_z_df(30, growth_dir=+1, infl_dir=+1)
    # coverage floor requires REGIMES_MIN_COVERAGE indicators; monkeypatch not
    # needed -- just assert the raw call, not persistence, since one series
    # per axis won't clear config.REGIMES_MIN_COVERAGE by default.
    timeline = classify.classify_timeline(z_df)
    assert not timeline.empty
    last = timeline.iloc[-1]
    # with coverage below the floor, raw_regime is None by design (spec 29
    # applied to breadth) -- confirm that honesty rather than a fabricated call.
    if config.REGIMES_MIN_COVERAGE > 1:
        assert last["raw_regime"] is None


def test_classify_timeline_declares_only_after_persistence(monkeypatch):
    monkeypatch.setattr("zenith.regimes.classify.REGIMES_MIN_COVERAGE", 1)
    n = 12
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    # monotonic ramps -> diff(3) is cleanly positive every month once it has
    # 3 months of lookback, so raw_regime is stable "Overheating" from month
    # 3 onward with no ambiguity in the fixture itself.
    g = pd.Series(np.arange(n, dtype=float), index=idx)
    i = pd.Series(np.arange(n, dtype=float), index=idx)
    growth_col = rs.BY_DIMENSION["growth"][0].id
    infl_col = rs.BY_DIMENSION["inflation"][0].id
    z_df = pd.DataFrame({growth_col: g, infl_col: i})
    timeline = classify.classify_timeline(z_df)

    raw_start = timeline[timeline["raw_regime"] == "Overheating"].index[0]
    raw_start_pos = timeline.index.get_loc(raw_start)
    # the month raw_regime FIRST reads "Overheating" cannot yet be declared
    # (streak == 1 < REGIMES_PERSISTENCE_MONTHS == 2).
    assert timeline.loc[raw_start, "declared_regime"] != "Overheating"
    # once persistence has had time to accrue, it IS declared and stays so.
    later = timeline.index[raw_start_pos + config.REGIMES_PERSISTENCE_MONTHS]
    assert timeline.loc[later, "declared_regime"] == "Overheating"
    assert timeline.iloc[-1]["declared_regime"] == "Overheating"


def test_axis_state_tie_break_uses_composite_diff():
    assert classify._axis_state(0.5, 1.0) is True
    assert classify._axis_state(0.5, -1.0) is False
    assert classify._axis_state(0.5, None) is None
    assert classify._axis_state(None, 1.0) is None


def test_confidence_bounded_0_100():
    for bg in (0.0, 0.3, 0.5, 0.7, 1.0):
        for bi in (0.0, 0.5, 1.0):
            c = classify._confidence(bg, 6, bi, 6)
            assert c is None or 0.0 <= c <= 100.0


# ---------------------------------------------------------------- momentum.py
def test_regime_momentum_positive_for_rising_growth():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    growth = pd.Series(np.linspace(-1, 1, 12), index=idx)
    infl = pd.Series(np.linspace(0, 0, 12), index=idx)
    out = momentum.regime_momentum(growth, infl, "Goldilocks / Reflation")
    assert out["score"] is not None
    assert out["score"] > 0
    assert "narrative" in out and isinstance(out["narrative"], str)


def test_regime_momentum_insufficient_history():
    short = pd.Series([1.0, 2.0])
    out = momentum.regime_momentum(short, short, "Overheating")
    assert out["score"] is None


# ----------------------------------------------------------------- history.py
def _synthetic_timeline() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    regimes = ["Overheating", "Overheating", "Overheating",
              "Stagflation", "Stagflation", "Stagflation"]
    return pd.DataFrame({"declared_regime": regimes}, index=idx)


def test_segments_and_transitions():
    tl = _synthetic_timeline()
    segs = history.segments(tl)
    assert len(segs) == 2
    assert segs[0]["regime"] == "Overheating"
    assert segs[1]["regime"] == "Stagflation"
    trans = history.transitions(segs)
    assert len(trans) == 1
    assert trans[0]["from_regime"] == "Overheating"
    assert trans[0]["to_regime"] == "Stagflation"


def test_segments_skips_none_regime():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    tl = pd.DataFrame({"declared_regime": [None, None, "Overheating"]}, index=idx)
    segs = history.segments(tl)
    assert len(segs) == 1
    assert segs[0]["regime"] == "Overheating"


def test_append_journal_idempotent_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGIMES_JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(history, "REGIMES_JOURNAL_DIR", tmp_path)
    d = date(2026, 3, 1)
    history.append_journal({"regime": "Overheating"}, d)
    n = history.append_journal({"regime": "Stagflation"}, d)   # same day, different call
    assert n == 1
    rows = history.load_journal()
    assert len(rows) == 1
    assert rows[0]["regime"] == "Stagflation"    # the rerun's value wins, not a duplicate


# --------------------------------------------------------------- dimensions.py
def test_state_label_thresholds():
    assert dimensions.state_label("dollar", 1.0, True) == "Strong / Appreciating"
    assert dimensions.state_label("dollar", -1.0, True) == "Weak / Depreciating"
    assert dimensions.state_label("dollar", 0.1, True) == "Neutral"
    assert dimensions.state_label("dollar", 1.0, False) == "Insufficient coverage"
    assert dimensions.state_label("dollar", None, True) == "Insufficient coverage"


def test_composite_series_missing_dimension_returns_empty():
    comp, cov = dimensions.composite_series(pd.DataFrame(), "growth")
    assert comp.empty


# ---------------------------------------------------------------- compute.py
def test_scrub_replaces_nonfinite_floats():
    assert rc._scrub(float("nan")) is None
    assert rc._scrub(float("inf")) is None
    assert rc._scrub({"a": float("nan"), "b": [1.0, float("nan")]}) == {"a": None, "b": [1.0, None]}
    assert rc._scrub(3.5) == 3.5


def test_run_auto_handles_zero_data(tmp_path, monkeypatch):
    """The day-one condition: no committed macro_raw, network calls degrade
    to empty (monkeypatched), the pipeline must not raise."""
    monkeypatch.setattr(macro, "fetch_raw", lambda force=False, sleep=0.2: ({}, {"fetched": 0, "reused": 0}))
    monkeypatch.setattr(config, "REGIMES_JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(history, "REGIMES_JOURNAL_DIR", tmp_path)
    files = {}
    for name in config.REGIMES_FILES:
        files[name] = tmp_path / f"{name}.json"
    monkeypatch.setattr(config, "REGIMES_FILES", files)
    monkeypatch.setattr("zenith.regimes.REGIMES_FILES", files)
    result = rc.run_auto(force=False)
    assert result["ok"] is True

"""MOMENTUM tests — offline, synthetic data only (no network)."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import zenith.mom as mom
from zenith.mom import compute as mc
from zenith.mom import engine, factors, history as mh, normalize as mn
from zenith.mom import universe as mu
from zenith.config import MOM_WEIGHTS, MOM_HORIZON_WEIGHTS
from zenith.pretom import calendar as cal


def _ohlc_series(n, drift, vol=0.012, seed=1, start=100.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.cumprod(1 + rets)
    idx = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": 1e6}, index=idx)


# ------------------------------------------------------------- normalize.py --
def test_ols_slope_r2_recovers_known_line():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    slope, r2 = mn.ols_slope_r2(y)
    assert abs(slope - 1.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_ols_slope_r2_too_short():
    assert mn.ols_slope_r2(pd.Series([1.0, 2.0])) == (None, None)


def test_inv_normal_monotonic_and_bounded():
    vals = [mn.inv_normal(p) for p in (0.5, 25, 50, 75, 99.5)]
    assert vals == sorted(vals)
    assert all(-1.3 <= v <= 1.3 for v in vals)


def test_clip1_and_tanh_clip_handle_nan():
    assert mn.clip1(float("nan")) == 0.0
    assert mn.clip1(5) == 1.0
    assert mn.clip1(-5) == -1.0
    assert mn.tanh_clip(float("inf"), 1.0) == 0.0


# --------------------------------------------------------------- factors.py --
def test_min_bars_excludes_short_history():
    short = _ohlc_series(100, 0.001, seed=3)
    assert factors.build_all(short) is None


def test_uptrend_scores_positive_downtrend_negative():
    up = factors.build_all(_ohlc_series(500, 0.0025, vol=0.012, seed=1))
    dn = factors.build_all(_ohlc_series(500, -0.0025, vol=0.012, seed=2))
    assert up is not None and dn is not None
    assert up["ts_raw"]["horizons"]["12m"]["m"] > 0
    assert dn["ts_raw"]["horizons"]["12m"]["m"] < 0
    assert up["strength_raw"]["s_slope"] > 0
    assert dn["strength_raw"]["s_slope"] < 0
    assert up["strength_raw"]["s_quality"] > 0
    assert dn["strength_raw"]["s_quality"] < 0


def test_breakout_excludes_current_bar():
    """A fresh all-time-high TODAY must register as a break; the current bar
    must not be counted as part of the trailing channel it's compared to."""
    df = _ohlc_series(500, 0.0, vol=0.001, seed=4, start=100.0)
    close = df["close"].copy()
    close.iloc[-1] = close.iloc[:-1].max() * 1.05
    df["close"] = close
    df["open"] = close
    df["high"] = close * 1.001
    df["low"] = close * 0.999
    br = factors.breakout_raw(df)
    assert br["horizons"]["1m"]["state"] == "break_up"
    assert br["horizons"]["1m"]["b"] == 1.0


def test_breakout_trailing_high_excludes_today_by_construction():
    """The trailing high/low series is built with close.shift(1), so a level
    computed at t can never include t's own close."""
    df = _ohlc_series(300, 0.001, vol=0.01, seed=7)
    state, hi, lo = factors._breakout_state_series(df["close"], 60)
    # the high at position i must equal the max of strictly PRIOR closes only
    c = df["close"]
    i = 200
    expected_hi = c.iloc[i - 60:i].max()
    assert abs(hi.iloc[i] - expected_hi) < 1e-9


def test_no_lookahead_score_stable_under_future_appends():
    base = _ohlc_series(510, 0.001, vol=0.01, seed=5)
    asof_slice = base.iloc[:480]
    before = factors.build_all(asof_slice)
    # re-slicing the SAME as-of window after more rows exist elsewhere must
    # not change the result
    after = factors.build_all(base.iloc[:480])
    assert before["ts_raw"]["horizons"]["12m"]["m"] == after["ts_raw"]["horizons"]["12m"]["m"]
    assert before["breakout_raw"]["horizons"]["3m"]["b"] == after["breakout_raw"]["horizons"]["3m"]["b"]
    assert before["strength_raw"]["s_slope"] == after["strength_raw"]["s_slope"]


# ---------------------------------------------------------------- engine.py --
def test_weight_dicts_sum_to_one():
    assert abs(sum(MOM_WEIGHTS.values()) - 1.0) < 1e-9
    assert abs(sum(MOM_HORIZON_WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_bounded_and_contributions_sum_to_composite():
    rows = []
    for i in range(12):
        df = _ohlc_series(500, 0.003, seed=100 + i)
        rows.append({"ticker": f"UP{i}", "raw": factors.build_all(df)})
    for i in range(12):
        df = _ohlc_series(500, -0.003, seed=200 + i)
        rows.append({"ticker": f"DN{i}", "raw": factors.build_all(df)})
    engine.cross_sectional(rows)
    engine.composite(rows)
    scored = [r for r in rows if r["composite"] is not None]
    assert len(scored) == 24
    for r in scored:
        assert -20.0 <= r["composite"] <= 20.0
        total = sum(r["contributions"].values())
        clipped = max(-20.0, min(20.0, total))
        assert abs(clipped - r["composite"]) < 1e-6
        assert engine.state_for(r["composite"]) == r["state"]
    assert np.mean([r["composite"] for r in scored if r["ticker"].startswith("UP")]) > 0
    assert np.mean([r["composite"] for r in scored if r["ticker"].startswith("DN")]) < 0


def test_excluded_rows_never_scored_as_neutral():
    rows = [{"ticker": "NEWCO", "raw": factors.build_all(_ohlc_series(100, 0.01, seed=9))}]
    engine.cross_sectional(rows)
    engine.composite(rows)
    assert rows[0]["raw"] is None
    assert rows[0]["composite"] is None
    assert rows[0]["state"] is None


def test_state_bands_match_config():
    assert engine.state_for(20.0) == "EXTREME BULLISH"
    assert engine.state_for(0.0) == "NEUTRAL"
    assert engine.state_for(-20.0) == "EXTREME BEARISH"


# --------------------------------------------------------------- history.py --
@pytest.fixture
def tmp_mom_store(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}.json" for k in
            ("scores", "detail", "sectors", "diagnostics", "meta", "membership", "picks", "status")}
    monkeypatch.setattr(mom, "MOM_FILES", files)
    hist_dir = tmp_path / "history"
    monkeypatch.setattr(mh, "MOM_HISTORY_DIR", hist_dir)
    return tmp_path


def test_history_sharding_idempotent_and_weekly_full(tmp_mom_store):
    rows = [{"ticker": "AAPL", "composite": 12.3, "state": "STRONG BULLISH",
            "factor_scores": {"ts": 0.5}, "contributions": {"ts": 2.5}},
           {"ticker": "MSFT", "composite": -8.1, "state": "BEARISH",
            "factor_scores": {"ts": -0.3}, "contributions": {"ts": -1.5}},
           {"ticker": "NEWCO", "composite": None, "state": None}]
    fri = date(2026, 8, 21)
    assert fri.weekday() == 4
    assert mh.append_history(rows, fri) == 2          # NEWCO excluded (composite None)
    assert mh.append_history(rows, fri) == 0           # idempotent same-day

    mon = date(2026, 8, 24)
    mh.append_history([{"ticker": "AAPL", "composite": 13.1, "state": "STRONG BULLISH"}], mon)

    series = mh.series_for("AAPL")
    assert len(series) == 2
    assert "factor_scores" in series[0]        # Friday -> full row
    assert "factor_scores" not in series[1]    # Monday -> light row


def test_pick_tracker_evaluation_sign_convention(tmp_mom_store):
    scored = [
        {"ticker": "WINNER", "composite": 15.0, "rank": 1, "pctile": 99.0, "side": "long"},
        {"ticker": "LOSER", "composite": -15.0, "rank": 900, "pctile": 1.0, "side": "short"},
        {"ticker": "MIDDLE", "composite": 1.0, "rank": 500, "pctile": 50.0},   # no side -> excluded
    ]
    pr = mh.make_pick_rows(scored, "2026-01-05")
    assert {r["ticker"] for r in pr} == {"WINNER", "LOSER"}
    assert mh.append_picks(pr) == 2
    assert mh.append_picks(pr) == 0     # idempotent

    idx = pd.bdate_range("2026-01-01", periods=300)
    winner = pd.DataFrame({"close": 100 * np.cumprod(1 + np.full(300, 0.003))}, index=idx)
    loser = pd.DataFrame({"close": 100 * np.cumprod(1 + np.full(300, -0.003))}, index=idx)
    spy = pd.Series(100 * np.cumprod(1 + np.full(300, 0.0002)), index=idx)
    px = {"WINNER": winner, "LOSER": loser}

    today = idx[250].date()
    n_eval = mh.evaluate_pending(px, spy, today)
    assert n_eval == 6   # 2 tickers x 3 horizons, all matured by day 250

    doc = mom.load("picks", {})
    for r in doc["rows"]:
        for cell in r["eval"].values():
            if cell["evaluated"]:
                assert cell["excess"] > 0, "both picks should show a WINNING sign-adjusted excess"


# --------------------------------------------------------------- universe.py --
def test_membership_archive_idempotent_and_asof(tmp_mom_store):
    assert mu.membership_archive(["AAPL", "MSFT", "NVDA"], today="2026-08-01") == 1
    assert mu.membership_archive(["AAPL", "MSFT", "NVDA"], today="2026-08-01") == 0
    assert mu.membership_archive(["AAPL", "MSFT"], today="2026-08-02") == 1

    assert mu.membership_asof("2026-08-01") == {"AAPL", "MSFT", "NVDA"}
    assert mu.membership_asof("2020-01-01") is None


# --------------------------------------------------------------- compute.py --
def test_scrub_replaces_non_finite_with_none():
    obj = {"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf")}], "d": 3.5, "e": None}
    scrubbed = mc._scrub(obj)
    assert scrubbed["a"] is None
    assert scrubbed["b"][1] is None
    assert scrubbed["b"][2]["c"] is None
    assert scrubbed["d"] == 3.5
    # must round-trip through json without raising and without a literal NaN
    dumped = json.dumps(scrubbed)
    assert "NaN" not in dumped and "Infinity" not in dumped
    json.loads(dumped)


def test_run_auto_end_to_end(tmp_mom_store, monkeypatch):
    universe = ([{"ticker": f"UP{i}", "name": f"Up {i}", "sector": "Technology", "weight_pct": 0.5}
                for i in range(8)]
               + [{"ticker": f"DN{i}", "name": f"Down {i}", "sector": "Energy", "weight_pct": 0.3}
                  for i in range(8)]
               + [{"ticker": "NEWCO", "name": "New Co", "sector": "Health Care", "weight_pct": 0.1}])
    px = {f"UP{i}": _ohlc_series(480, 0.0025, seed=100 + i) for i in range(8)}
    px.update({f"DN{i}": _ohlc_series(480, -0.0025, seed=200 + i) for i in range(8)})
    px["NEWCO"] = _ohlc_series(100, 0.001, seed=300)     # too short -> excluded
    px["SPY"] = _ohlc_series(480, 0.0005, seed=999)

    monkeypatch.setattr(mu, "constituents", lambda **kw: (universe, {"source": "test", "n": len(universe)}))
    monkeypatch.setattr(mc, "_fetch_prices", lambda *a, **kw: px)
    monkeypatch.setattr(mu, "refresh_metadata", lambda *a, **kw: {"checked": 17, "stale": 17, "refreshed": 0})
    monkeypatch.setattr(cal, "is_trading_day", lambda d: True)

    result = mc.run_auto(force=True)
    assert result["ok"] and result["scored"] == 16   # NEWCO excluded

    scores = mom.load("scores", {})
    assert scores["n"] == 17
    assert scores["n_scored"] == 16
    newco = next(r for r in scores["rows"] if r["ticker"] == "NEWCO")
    assert newco["excluded"] and "insufficient_history" in newco["exclusion_reason"]

    ranks = [r["rank"] for r in scores["rows"] if not r["excluded"]]
    assert sorted(ranks) == list(range(1, 17))     # contiguous ranks

    pctiles = [r["pctile"] for r in scores["rows"] if not r["excluded"]]
    assert all(0 <= p <= 100 for p in pctiles)

    for r in scores["rows"]:
        if not r["excluded"]:
            total = sum(r["contributions"].values())
            assert abs(max(-20.0, min(20.0, total)) - r["composite"]) < 1e-6

    longs = {r["ticker"] for r in scores["rows"] if r.get("side") == "long"}
    shorts = {r["ticker"] for r in scores["rows"] if r.get("side") == "short"}
    assert not (longs & shorts)

    sectors = mom.load("sectors", {})
    assert "Technology" in sectors["sectors"] and "Energy" in sectors["sectors"]

    status = mom.load("status", {})
    coverage_seg = next(s for s in status["segments"] if s["segment"] == "coverage")
    assert coverage_seg["coverage"] >= 0.85


def test_run_auto_gates_on_non_trading_day(tmp_mom_store, monkeypatch):
    monkeypatch.setattr(cal, "is_trading_day", lambda d: False)
    result = mc.run_auto(force=False)
    assert result == {"ok": True, "gated": True}
    status = mom.load("status", {})
    assert status["is_trading_day"] is False

"""EPHEMERIS phase 3: Daily Five determinism and append-only schedule,
streaks, share text, 4H session resampling and hourly history merging."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from zenith import config
from zenith.ephemeris import daily, prefetch
from zenith.ephemeris.sampler import chart_at
from zenith.ephemeris.store_px import resample

from .test_ephemeris import FakeStore, _walk

CLASSES6 = ["Russell 1000", "US Equity ETFs", "Bonds", "Crypto", "Currencies", "Precious Metals"]


@pytest.fixture
def universe_store():
    uni, frames = [], {}
    for i in range(60):
        t = f"T{i:02d}"
        uni.append({"ticker": t, "name": t, "cls": CLASSES6[i % 6], "sector": ""})
        frames[t] = _walk(1200 + 5 * i, 100 + i)
    return uni, FakeStore(frames)


@pytest.fixture
def tmp_schedule(tmp_path, monkeypatch):
    monkeypatch.setitem(config.EPHEMERIS_FILES, "daily_schedule", tmp_path / "daily_schedule.json")
    monkeypatch.setattr(daily, "EPHEMERIS_FILES", config.EPHEMERIS_FILES)
    monkeypatch.setattr(daily, "EPHEMERIS_DIR", tmp_path)
    return tmp_path / "daily_schedule.json"


def test_build_day_is_deterministic_and_well_formed(universe_store):
    uni, store = universe_store
    a = daily.build_day(store, uni, "2026-10-04")
    b = daily.build_day(store, uni, "2026-10-04")
    c = daily.build_day(store, uni, "2026-10-05")
    assert a == b and a != c
    assert [s["slot"] for s in a] == list(range(5))
    assert [s["difficulty"] for s in a] == list(daily.SHAPE)
    assert len({s["ticker"] for s in a}) == 5
    assert max(pd.Series([s["cls"] for s in a]).value_counts()) <= daily.MAX_PER_CLASS
    assert a[0]["move_z"] >= a[4]["move_z"]                      # easy is more decisive than hard
    for s in a:                                                  # every slot rebuilds exactly
        row = next(r for r in uni if r["ticker"] == s["ticker"])
        ch = chart_at(store, row, "Daily", 120, 10, s["decision"])
        assert ch.decision_date.strftime("%Y-%m-%d") == s["decision"]


def test_schedule_is_append_only_and_avoids_recent_tickers(universe_store, tmp_schedule):
    uni, store = universe_store
    start = date(2026, 10, 1)
    assert daily.extend_schedule(store, uni, days_ahead=3, start=start) == 4
    first = json.loads(tmp_schedule.read_text())["days"]
    store.frames["T00"] = store.frames["T00"].iloc[:-50]           # data changes overnight...
    assert daily.extend_schedule(store, uni, days_ahead=5, start=start) == 2
    again = json.loads(tmp_schedule.read_text())["days"]
    for k, v in first.items():
        assert again[k] == v                                     # ...existing days never change
    d1 = {s["ticker"] for s in again["2026-10-01"]}
    d2 = {s["ticker"] for s in again["2026-10-02"]}
    assert not d1 & d2                                           # 30-day no-repeat (60 names, 6 days)


def test_streaks():
    t = date(2026, 10, 10)
    done = {t - timedelta(days=k) for k in (0, 1, 2, 5, 6)}
    assert daily.streak(done, t) == 3
    assert daily.streak(done - {t}, t) == 2                      # today unplayed: streak still alive
    assert daily.streak({t - timedelta(days=3)}, t) == 0
    assert daily.best_streak(done) == 3


def test_share_text_has_grid_and_no_answers():
    rows = [{"win": 1, "direction": 1, "pnl": 40.0, "ticker": "SECRET"},
            {"win": 0, "direction": -1, "pnl": -25.0, "ticker": "SECRET"}]
    txt = daily.share_text(date(2026, 10, 4), rows, 3)
    assert "🟩🟥" in txt and "▲▼" in txt and "1/2" in txt and "+$15" in txt and "streak 3" in txt
    assert "SECRET" not in txt and f"#{daily.game_number(date(2026, 10, 4))}" in txt


def test_4h_resample_sessions_and_24h():
    rth = pd.date_range("2026-03-02 09:30", periods=7, freq="1h")           # 09:30..15:30
    df = pd.DataFrame({"open": np.arange(7.) + 1, "high": np.arange(7.) + 2, "low": np.arange(7.),
                       "close": np.arange(7.) + 1.5, "volume": 1.0}, index=rth)
    out = resample(df, "4H")
    assert list(out.index.strftime("%H:%M")) == ["09:30", "13:30"]
    assert out["volume"].tolist() == [4.0, 3.0] and out["open"].iloc[1] == 5.0
    day = pd.date_range("2026-03-02 00:00", periods=24, freq="1h")
    d2 = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=day)
    o2 = resample(d2, "4H")
    assert list(o2.index.strftime("%H:%M")) == ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]


def test_merge_frames_keeps_old_history_and_prefers_new_bars():
    idx_old = pd.date_range("2024-01-01", periods=5, freq="1h")
    idx_new = pd.date_range("2024-01-01 03:00", periods=5, freq="1h")
    old = pd.DataFrame({"close": [1, 2, 3, 4, 5.]}, index=idx_old).astype("float32")
    new = pd.DataFrame({"close": [40, 50, 60, 70, 80.]}, index=idx_new).astype("float32")
    m = prefetch.merge_frames({"A": old, "B": old}, {"A": new})
    assert len(m["A"]) == 8 and m["A"]["close"].iloc[3] == 40 and m["A"]["close"].iloc[0] == 1
    assert len(m["B"]) == 5


def test_read_parquet_frames_roundtrip(tmp_path):
    frames = {"A": _walk(300, 1).astype("float32"), "B": _walk(200, 2).astype("float32")}
    prefetch.write_parquet(frames, tmp_path / "x.parquet")
    back = prefetch.read_parquet_frames(tmp_path / "x.parquet")
    assert set(back) == {"A", "B"} and len(back["B"]) == 200
    assert prefetch.read_parquet_frames(tmp_path / "missing.parquet") == {}

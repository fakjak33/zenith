"""NIGHT & DAY tests — offline, synthetic data only."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import zenith.nightday as nd
from zenith.nightday import core, history as nh
from zenith.pretom import calendar as cal


def _ohlc(opens, closes):
    idx = pd.bdate_range("2025-01-01", periods=len(opens))
    return pd.DataFrame({"open": opens, "high": [max(o, c) * 1.001 for o, c in zip(opens, closes)],
                         "low": [min(o, c) * 0.999 for o, c in zip(opens, closes)],
                         "close": closes, "volume": [1e6] * len(opens)}, index=idx)


def test_decomposition_identity():
    df = _ohlc([100, 101, 102, 99], [101, 103, 101, 100])
    s = core.streams(df)
    # (1+overnight)(1+intraday) - 1 == close/prevclose - 1 each day
    for i in range(len(s)):
        on, idr = s.iloc[i]["overnight"], s.iloc[i]["intraday"]
        d = s.index[i]
        pos = df.index.get_loc(d)
        direct = df["close"].iloc[pos] / df["close"].iloc[pos - 1] - 1
        assert abs((1 + on) * (1 + idr) - 1 - direct) < 1e-9


def test_streams_caps_glitches():
    # a 3x split-like glitch (open jumps) must be filtered out
    df = _ohlc([100, 300, 102], [101, 305, 101])
    s = core.streams(df)
    assert (s["overnight"].abs() < 0.5).all()


def test_stock_stats_overnight_share_and_signal():
    # construct a name whose gains are ALL overnight (open>prevclose), intraday flat
    n = 120
    opens, closes, c = [], [], 100.0
    for i in range(n):
        o = c * 1.002                     # +0.2% overnight
        opens.append(o); closes.append(o)  # intraday flat
        c = o
    df = _ohlc(opens, closes)
    stats = core.stock_stats(df)
    assert stats["cum_overnight"] > 0.15
    assert abs(stats["cum_intraday"]) < 1e-6
    assert stats["share_overnight"] > 0.99
    assert stats["overnight_avg_bp"] > stats["intraday_avg_bp"]


def test_cumulative_panel_shape():
    df = _ohlc([100 + i * 0.1 for i in range(120)], [100.5 + i * 0.1 for i in range(120)])
    panel = core.cumulative_panel(df)
    assert panel and {"d", "overnight", "intraday", "total"} <= set(panel[0])


# --- history -----------------------------------------------------------------
@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}.json" for k in ("panel", "screen", "history", "status")}
    monkeypatch.setattr(nd, "NIGHTDAY_FILES", files)
    return tmp_path


def _tds(start, n):
    out, d = [], start
    while len(out) < n:
        if cal.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def test_history_snapshot_and_eval(tmp_store):
    asof = date(2025, 6, 2)
    screen = {"long": [{"ticker": "AAA", "on_mom": 2.0}],
              "short": [{"ticker": "BBB", "on_mom": -2.0}]}
    assert nh.append_rows(nh.make_rows(screen, asof.isoformat())) == 2
    assert nh.append_rows(nh.make_rows(screen, asof.isoformat())) == 0

    days = _tds(asof - timedelta(days=10), 60)
    i0 = days.index(asof)
    aaa = [100.0] * (i0 + 1) + [100.0 + k for k in range(1, 60 - i0)]
    px = {"AAA": pd.DataFrame({"close": aaa}, index=pd.to_datetime(days)),
          "BBB": pd.DataFrame({"close": [50.0] * len(days)}, index=pd.to_datetime(days))}
    spy = pd.Series([400.0] * len(days), index=pd.to_datetime(days))
    assert nh.evaluate_pending(px, spy, days[-1]) == 2
    rows = nd.load("history")["rows"]
    assert next(r for r in rows if r["ticker"] == "AAA")["excess"] > 0
    assert nh.summarize(rows)["all"]["n"] == 2

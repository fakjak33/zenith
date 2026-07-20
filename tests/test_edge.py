"""EDGE screener tests — offline, synthetic data only."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import zenith.edge as edge
from zenith.edge import common, ivspread, revisions, shortint, lottery, history as eh
from zenith.pretom import calendar as cal


# --- common ------------------------------------------------------------------
def test_pct_ranks_monotone_and_ties():
    r = common.pct_ranks([1.0, 2.0, 2.0, 3.0])
    assert r[0] < r[1] and r[1] == r[2] and r[3] == max(r)


def test_assemble_sides_and_direction():
    rows = [{"ticker": f"T{i}", "x": float(i)} for i in range(30)]
    hi = common.assemble([dict(r) for r in rows], "x", higher_is_long=True)
    assert hi["long"][0]["x"] > hi["short"][0]["x"]      # top decile = long
    assert hi["ranked"][0]["rank"] == 1
    lo = common.assemble([dict(r) for r in rows], "x", higher_is_long=False)
    assert lo["short"][0]["x"] > lo["long"][0]["x"]      # top decile = short


# --- IV spread ---------------------------------------------------------------
def test_ivspread_long_short_and_borrow_flag():
    rows = [{"ticker": f"T{i}", "iv_spread": (i - 15) * 0.001, "n_pairs": 5}
            for i in range(30)]
    res = ivspread.build(rows, hard_to_borrow={"T0", "T1", "T2"})
    assert res["long"][0]["iv_spread"] > res["short"][0]["iv_spread"]
    # a negative-spread (short-side) name that is hard to borrow is flagged
    assert any(r["borrow_flag"] for r in res["short"])
    assert res["horizon_td"] == 5


# --- revisions ---------------------------------------------------------------
def test_revisions_composite_and_missing_components():
    rows = [{"ticker": f"T{i}", "est_rev_pct": (i - 15) * 0.01,
             "up_frac": (i / 30 if i % 2 else None),
             "net_reco": ((i - 15) / 30)} for i in range(30)]
    res = revisions.build(rows)
    assert res["long"][0]["composite"] > res["short"][0]["composite"]
    # a row missing up_frac still ranks (neutral 50 for that component)
    assert all("composite" in r for r in res["ranked"])


def test_revisions_requires_anchor():
    assert revisions.build([{"ticker": "X", "up_frac": 0.9}])["n"] == 0


# --- short interest ----------------------------------------------------------
def test_shortint_avoid_side_and_squeeze():
    rows = [{"ticker": f"T{i}", "si_float": float(i), "dtc": i * 0.4,
             "mom_1m": 0.1 if i >= 27 else -0.05} for i in range(30)]
    res = shortint.build(rows, si_float_history=[5, 6, 7, 8, 9, 10])
    # most-shorted names are the SHORT/avoid side
    assert res["short"][0]["si_float"] > res["long"][0]["si_float"]
    # high-SI + rising = squeeze risk
    assert any(r.get("squeeze_risk") for r in res["short"])
    assert res["aggregate"]["z_vs_history"] is not None


# --- lottery / MAX-beta ------------------------------------------------------
def _series(beta, mkt, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    r = beta * mkt + rng.normal(0, noise, len(mkt))
    return 100 * np.cumprod(1 + r)


def test_lottery_beta_purge_lowers_max_for_high_beta():
    idx = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(3)
    mkt = rng.normal(0, 0.01, 120)
    mkt[-3] = 0.06                       # a big MARKET up-day inside the MAX window
    spy = pd.Series(100 * np.cumprod(1 + mkt), index=idx)
    px = {"HI": pd.DataFrame({"close": _series(2.5, mkt, seed=1)}, index=idx),
          "LO": pd.DataFrame({"close": _series(0.3, mkt, seed=2)}, index=idx)}
    meta = {"HI": {"name": "High", "sector": "X"}, "LO": {"name": "Low", "sector": "Y"}}
    res = lottery.build(px, spy, meta)
    hi = next(r for r in res["ranked"] if r["ticker"] == "HI")
    # the market jump inflates raw MAX; the beta purge strips it out
    assert hi["max_beta"] < hi["max_raw"]
    assert hi["beta"] > 1.5
    assert res["horizon_td"] == 20


# --- history -----------------------------------------------------------------
@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}.json" for k in
             ("ivspread", "revisions", "shortint", "lottery", "history", "status")}
    monkeypatch.setattr(edge, "EDGE_FILES", files)
    import zenith.config as cfg
    monkeypatch.setattr(cfg, "EDGE_FILES", files, raising=False)
    return tmp_path


def _tds(start, n):
    out, d = [], start
    while len(out) < n:
        if cal.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def test_history_append_idempotent_and_eval(tmp_store):
    asof = date(2025, 6, 2)
    result = {"long": [{"ticker": "AAA", "side": "long", "rank": 1}],
              "short": [{"ticker": "BBB", "side": "short", "rank": 30}]}
    rows = eh.make_rows("lottery", result, asof.isoformat(), 20)
    assert eh.append_rows(rows) == 2
    assert eh.append_rows(rows) == 0                  # idempotent

    days = _tds(asof - timedelta(days=10), 60)
    i0 = days.index(asof)
    # AAA rises vs flat SPY (long works); BBB also rises (short fails)
    aaa = [100.0] * (i0 + 1) + [100.0 + k for k in range(1, 60 - i0)]
    bbb = [50.0] * (i0 + 1) + [50.0 + k for k in range(1, 60 - i0)]
    px = {"AAA": pd.DataFrame({"close": aaa}, index=pd.to_datetime(days)),
          "BBB": pd.DataFrame({"close": bbb}, index=pd.to_datetime(days))}
    spy = pd.Series([400.0] * len(days), index=pd.to_datetime(days))
    n = eh.evaluate_pending(px, spy, days[-1])
    assert n == 2
    rows2 = edge.load("history")["rows"]
    aaa_row = next(r for r in rows2 if r["ticker"] == "AAA")
    bbb_row = next(r for r in rows2 if r["ticker"] == "BBB")
    assert aaa_row["excess"] > 0                     # long that rose vs SPY worked
    assert bbb_row["excess"] < 0                     # short that rose vs SPY failed
    summ = eh.summarize(rows2)
    assert summ["lottery"]["all"]["n"] == 2

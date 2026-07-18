"""PEAD tests — offline, synthetic data only (no network)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import zenith.pead as pead
from zenith.pead import history as ph
from zenith.pead import signals as sig
from zenith.pretom import calendar as cal


# --- synthetic data helpers --------------------------------------------------

def _tds(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if cal.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _frame(days: list[date], close, volume=None, open_=None, high=None) -> pd.DataFrame:
    n = len(days)
    close = list(close) if hasattr(close, "__len__") else [close] * n
    df = pd.DataFrame({
        "open": open_ if open_ is not None else close,
        "high": high if high is not None else [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close],
        "close": close,
        "volume": volume if volume is not None else [1e6] * n,
    }, index=pd.to_datetime(days))
    return df


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}.json" for k in ("signals", "history", "status")}
    monkeypatch.setattr(pead, "PEAD_FILES", files)
    monkeypatch.setattr(pead, "PEAD_ARCHIVE_DIR", tmp_path / "archive")
    return tmp_path


# --- parse_eps ---------------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("$1.23", 1.23), ("($0.45)", -0.45), ("(0.45)", -0.45), ("2", 2.0),
    ("-$0.12", -0.12), ("", None), ("N/A", None), (None, None),
    ("--", None), (1.5, 1.5), ("$1,234.50", 1234.5),
])
def test_parse_eps(raw, expect):
    assert sig.parse_eps(raw) == expect


# --- reaction day ------------------------------------------------------------

def test_reaction_day_pre_market_same_day():
    tue = date(2026, 7, 14)
    assert sig.reaction_day(tue, "time-pre-market") == tue


def test_reaction_day_after_hours_next_day():
    tue = date(2026, 7, 14)
    assert sig.reaction_day(tue, "time-after-hours") == date(2026, 7, 15)


def test_reaction_day_friday_ah_to_monday():
    fri = date(2026, 7, 10)
    assert sig.reaction_day(fri, "time-after-hours") == date(2026, 7, 13)


def test_reaction_day_unknown_treated_after_hours():
    assert sig.reaction_day(date(2026, 7, 14), "time-not-supplied") == date(2026, 7, 15)


def test_reaction_day_pre_market_on_holiday_rolls():
    # 2026-07-03 observed Independence Day (Fri): pre-market rolls to Monday
    assert sig.reaction_day(date(2026, 7, 3), "time-pre-market") == date(2026, 7, 6)


# --- component raw values ----------------------------------------------------

def test_sue_raw_price_scaled_no_blowup():
    assert sig.sue_raw(1.42, 1.31, 50.0) == pytest.approx(0.0022)
    assert sig.sue_raw(0.05, -0.02, 10.0) == pytest.approx(0.007)   # negative consensus fine
    assert sig.sue_raw(None, 1.0, 50.0) is None
    assert sig.sue_raw(1.0, 1.0, 0.0) is None


def test_ear_exact_value():
    days = _tds(date(2026, 1, 5), 10)
    r_day = days[5]
    closes = [100.0] * 5 + [104.0] + [104.0] * 4
    px = _frame(days, closes)
    spy = _frame(days, [500.0] * 5 + [505.0] + [505.0] * 4)
    got = sig.ear(px["close"], spy["close"], r_day)
    assert got == pytest.approx(0.04 - 0.01)


def test_ear_missing_reaction_bar():
    days = _tds(date(2026, 1, 5), 5)
    px = _frame(days, 100.0)
    spy = _frame(days, 500.0)
    assert sig.ear(px["close"], spy["close"], cal.next_trading_day(days[-1])) is None


def test_rue_needs_eight_quarters():
    q = pd.date_range("2024-03-31", periods=12, freq="QE")
    steady = pd.Series([100 + i for i in range(12)], index=q, dtype=float)
    assert sig.rue(steady.iloc[:7]) is None
    accel = steady.copy()
    accel.iloc[-1] = accel.iloc[-5] * 1.5          # revenue growth acceleration
    assert sig.rue(accel) > 0


def test_abnormal_volume_spike():
    days = _tds(date(2025, 1, 6), 80)
    vol = [1e6] * 79 + [4e6]
    px = _frame(days, 50.0, volume=vol)
    got = sig.abnormal_volume(px["volume"], days[-1])
    assert got == pytest.approx(1.386, abs=0.01)   # log(4)
    assert sig.abnormal_volume(px["volume"], cal.next_trading_day(days[-1])) is None


def test_nearness_and_gap_and_momentum():
    days = _tds(date(2024, 6, 3), 300)
    closes = [50.0 + i * 0.1 for i in range(300)]  # steady uptrend, at the high
    px = _frame(days, closes, open_=[c - 0.5 for c in closes])
    pre = days[-2]
    near = sig.nearness_52w(px, pre)
    assert near is not None and near > 0.97
    assert sig.gap_held(px, days[-1], "long") is True
    assert sig.gap_held(px, days[-1], "short") is False
    assert sig.momentum_12_1(px["close"], pre) > 0


# --- direction + gates -------------------------------------------------------

def test_side_of_two_confirmations():
    assert sig.side_of(0.01, 0.02) == "long"
    assert sig.side_of(-0.01, -0.02) == "short"
    assert sig.side_of(0.01, -0.02) == "mixed"
    assert sig.side_of(None, 0.02) == "mixed"


def _row(**over):
    base = {"eps_actual": 1.0, "eps_consensus": 0.9, "side": "long",
            "raw": {"sue": 0.002, "ear": 0.03},
            "context": {"close_pre": 50.0, "adv_usd": 2e7}}
    for k, v in over.items():
        if k in ("raw", "context"):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def test_gates_each_reason():
    assert sig.gates(_row()) is None
    assert sig.gates(_row(eps_actual=None)) == "no_eps"
    assert sig.gates(_row(eps_consensus=None)) == "no_consensus"
    assert sig.gates(_row(raw={"ear": None})) == "no_prices"
    assert sig.gates(_row(raw={"sue": 0})) == "zero_surprise"
    assert sig.gates(_row(context={"close_pre": 4.0})) == "penny"
    assert sig.gates(_row(context={"adv_usd": 1e6})) == "illiquid"
    assert sig.gates(_row(side="mixed")) == "mixed"
    assert sig.gates(_row(raw={"ear": 0.001})) == "weak_reaction"


# --- ranks + composite -------------------------------------------------------

def test_pool_rank_endpoints_and_monotonic():
    pool = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, mid, hi = (sig.pool_rank(v, pool) for v in (0.0, 3.0, 9.0))
    assert lo < mid < hi
    assert sig.pool_rank(9.9, []) == sig.NEUTRAL_RANK


def test_component_ranks_flip_for_short():
    pool = {"sue": [-0.01, 0.0, 0.01], "ear": [-0.05, 0.0, 0.05],
            "rue": [-1, 0, 1], "avol": [0.0, 0.7, 1.4], "nearness": [0.5, 0.8, 1.0]}
    raw = {"sue": -0.02, "ear": -0.06, "rue": -2, "avol": 1.4, "nearness": 0.5}
    long_r, _ = sig.component_ranks(raw, "long", pool)
    short_r, _ = sig.component_ranks(raw, "short", pool)
    assert short_r["sue"] > long_r["sue"]          # bad news = good short
    assert short_r["avol"] == long_r["avol"]        # volume never flips
    assert short_r["anchor"] == pytest.approx(100 - long_r["anchor"], abs=0.1)


def test_composite_weights_and_bounds():
    assert sig.composite({k: 100.0 for k in sig.WEIGHTS}) == 100.0
    assert sig.composite({k: 0.0 for k in sig.WEIGHTS}) == 0.0
    assert sig.composite({}) == sig.NEUTRAL_RANK


# --- sheet end-to-end --------------------------------------------------------

def test_build_signal_sheet_end_to_end():
    days = _tds(date(2025, 4, 1), 320)
    r_day = days[-1]
    rep_day = sig.prev_trading_day(r_day)

    up = [50.0 + i * 0.05 for i in range(319)] + [0]
    up[-1] = up[-2] * 1.05                          # +5% reaction
    dn = [80.0 - i * 0.02 for i in range(319)] + [0]
    dn[-1] = dn[-2] * 0.93                          # -7% reaction
    flat = [30.0] * 320
    spy = _frame(days, [500.0] * 319 + [500.0])
    vol = [2e6] * 319 + [8e6]
    px = {"AAA": _frame(days, up, volume=vol),
          "BBB": _frame(days, dn, volume=vol),
          "CCC": _frame(days, flat, volume=[2e6] * 320)}

    reporters = [
        {"ticker": "AAA", "name": "Alpha", "report_date": rep_day.isoformat(),
         "time": "time-after-hours", "eps_actual": 1.2, "eps_consensus": 1.0,
         "n_estimates": 5, "mktcap": 30e9, "sue_source": "nasdaq"},
        {"ticker": "BBB", "name": "Beta", "report_date": rep_day.isoformat(),
         "time": "time-after-hours", "eps_actual": 0.5, "eps_consensus": 0.8,
         "n_estimates": 4, "mktcap": 8e9, "sue_source": "nasdaq"},
        {"ticker": "CCC", "name": "Gamma", "report_date": rep_day.isoformat(),
         "time": "time-after-hours", "eps_actual": 1.0, "eps_consensus": 0.9,
         "n_estimates": 2, "mktcap": 3e9, "sue_source": "nasdaq"},
    ]
    rows = sig.build_signal_sheet(reporters, px, spy["AAA" in px and "close"],
                                  {}, {}, r_day) if False else \
        sig.build_signal_sheet(reporters, px, spy["close"], {}, {}, r_day)

    by = {r["ticker"]: r for r in rows}
    assert by["AAA"]["side"] == "long" and by["AAA"]["excluded_reason"] is None
    assert by["BBB"]["side"] == "short" and by["BBB"]["excluded_reason"] is None
    assert by["CCC"]["side"] == "mixed"             # beat but flat tape
    assert by["CCC"]["excluded_reason"] == "mixed"
    assert 0 <= by["AAA"]["composite"] <= 100
    assert by["AAA"]["context"]["cap_tier"] == "large"
    assert by["BBB"]["context"]["cap_tier"] == "mid"
    assert by["AAA"]["flags"]["gap_held"] in (True, False)
    assert rows[0]["composite"] is not None         # scored rows sort first


def test_update_pool_trims_window():
    pool = {"days": {}}
    days = _tds(date(2025, 1, 6), sig.POOL_WINDOW_TD + 10)
    for d in days:
        rows = [{"raw": {"sue": 0.001, "ear": 0.01, "rue": None,
                         "avol": 0.5, "nearness": 0.9}}]
        pool = sig.update_pool(pool, rows, d)
    assert len(pool["days"]) == sig.POOL_WINDOW_TD
    assert len(pool["sue"]) == sig.POOL_WINDOW_TD
    assert pool["window_start"] == days[10].isoformat()


# --- history / tracker -------------------------------------------------------

def _sig_row(ticker="AAA", r_day=None, side="long", composite=72.0):
    r_day = r_day or date(2025, 6, 2)
    return {"ticker": ticker, "report_date": sig.prev_trading_day(r_day).isoformat(),
            "reaction_day": r_day.isoformat(), "side": side, "composite": composite,
            "sue_source": "nasdaq",
            "context": {"cap_tier": "mid", "next_earn_date": None}}


def test_append_rows_idempotent(tmp_store):
    rows = [ph.make_row(_sig_row()), ph.make_row(_sig_row(ticker="BBB"))]
    assert ph.append_rows(rows) == 2
    assert ph.append_rows(rows) == 0
    assert len(pead.load("history")["rows"]) == 2


def test_evaluate_pending_fills_horizons(tmp_store):
    r_day = date(2025, 6, 2)
    days = _tds(r_day - timedelta(days=30), 90)
    i0 = days.index(r_day)
    closes = [100.0] * (i0 + 1) + [100.0 + 2 * k for k in range(1, 90 - i0)]
    px = {"AAA": _frame(days, closes, open_=[c - 0.5 for c in closes])}
    spy = _frame(days, 500.0)

    ph.append_rows([ph.make_row(_sig_row(r_day=r_day))])
    today = days[i0 + 25]
    filled = ph.evaluate_pending(px, spy["close"], today)
    row = pead.load("history")["rows"][0]
    assert row["entry_date"] == days[i0 + 1].isoformat()
    assert row["entry_open"] == pytest.approx(closes[i0 + 1] - 0.5)
    assert row["horizons"]["h5"]["evaluated"]
    assert row["horizons"]["h20"]["evaluated"]
    assert not row["horizons"]["h60"]["evaluated"]
    assert row["horizons"]["h5"]["excess_ret"] > 0     # rose vs flat SPY
    assert row["active"] is True and "running" in row
    assert filled == 2


def test_next_earn_exit_day_before_next_reaction(tmp_store):
    r_day = date(2025, 6, 2)
    days = _tds(r_day - timedelta(days=30), 90)
    i0 = days.index(r_day)
    px = {"AAA": _frame(days, 100.0)}
    spy = _frame(days, 500.0)
    row = ph.make_row(_sig_row(r_day=r_day))
    nxt = days[i0 + 10]
    row["next_earn_date"] = nxt.isoformat()
    ph.append_rows([row])
    ph.evaluate_pending(px, spy["close"], days[i0 + 20])
    got = pead.load("history")["rows"][0]
    # AH assumption: next reaction = next td after report; exit = the report day close
    assert got["horizons"]["next_earn"]["evaluated"]
    assert got["horizons"]["next_earn"]["exit_date"] == nxt.isoformat()
    assert got["active"] is False


def test_short_side_excess_sign(tmp_store):
    r_day = date(2025, 6, 2)
    days = _tds(r_day - timedelta(days=30), 60)
    i0 = days.index(r_day)
    closes = [100.0] * (i0 + 1) + [100.0 - 1.5 * k for k in range(1, 60 - i0)]
    px = {"BBB": _frame(days, closes)}
    spy = _frame(days, 500.0)
    ph.append_rows([ph.make_row(_sig_row(ticker="BBB", r_day=r_day, side="short"))])
    ph.evaluate_pending(px, spy["close"], days[i0 + 10])
    got = pead.load("history")["rows"][0]
    assert got["horizons"]["h5"]["ret"] < 0            # stock fell...
    assert got["horizons"]["h5"]["excess_ret"] > 0     # ...so the short worked


def test_summarize_and_drift_curve(tmp_store):
    r_day = date(2025, 6, 2)
    days = _tds(r_day - timedelta(days=30), 90)
    i0 = days.index(r_day)
    closes = [100.0] * (i0 + 1) + [100.0 + k for k in range(1, 90 - i0)]
    px = {"AAA": _frame(days, closes)}
    spy = _frame(days, 500.0)
    ph.append_rows([ph.make_row(_sig_row(r_day=r_day))])
    ph.evaluate_pending(px, spy["close"], days[-1])
    rows = pead.load("history")["rows"]
    summ = ph.summarize(rows)
    key = "h5|long|all"
    assert summ["cells"][key]["n"] == 1
    assert summ["cells"][key]["win_rate"] == 1.0
    curve = ph.drift_curve(rows, px, spy["close"])
    assert curve and curve[0]["td"] == 1
    assert curve[-1]["long_avg_excess"] > curve[0]["long_avg_excess"]


def test_archive_roundtrip(tmp_store):
    pead.archive_day("2026-07-16", {"day": "2026-07-16", "signals": []})
    assert pead.archive_days() == ["2026-07-16"]
    assert pead.load_day("2026-07-16")["day"] == "2026-07-16"


def test_pending_reaction_days_backscan(tmp_store):
    from zenith.pead import compute as pc
    today = date(2026, 7, 16)                          # Thursday
    got = pc._pending_reaction_days(today)
    assert len(got) == pc.BACKSCAN_TD
    assert got[-1] == today
    assert all(cal.is_trading_day(d) for d in got)
    pead.archive_day("2026-07-15", {"day": "2026-07-15"})
    got2 = pc._pending_reaction_days(today)
    assert date(2026, 7, 15) not in got2 and today in got2

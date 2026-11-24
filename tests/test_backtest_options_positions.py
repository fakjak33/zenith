"""Multi-position engine: entry schedules, max open, roll at DTE, whole contracts, $ sizing, account."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.options import engine as E


@pytest.fixture(scope="module")
def mkt():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2016-01-01", periods=700)
    S = pd.Series(450 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx)))), index=idx)
    vix = pd.Series(np.clip(16 + np.cumsum(rng.normal(0, 0.6, len(idx))) * 0.3, 10, 45), index=idx)
    term = pd.DataFrame({"VIX9D": vix * 0.95, "VIX": vix, "VIX3M": vix * 1.06, "VIX6M": vix * 1.1})
    return S, term, pd.Series(125.0, index=idx), pd.Series(0.02, index=idx)


def run(strat, mkt):
    S, term, skew, rf = mkt
    return E.simulate(strat, S, term, skew, rf)


TAIL = E.OptionStrategy("tail", [E.Leg("put", 1, "delta", 0.05)], dte=60, roll="dte", sizing="premium_usd",
                        size_usd=500.0, capital=5_000.0, contracts="whole")


def test_monthly_ladder_respects_max_open(mkt):
    res = run(replace(TAIL, entry="monthly", entry_day=1, max_open=3), mkt)
    assert res.positions.max() <= 3 and res.positions.max() >= 2        # 60-DTE monthly entries overlap
    opens = pd.to_datetime(res.trades["open"])
    assert opens.dt.to_period("M").is_unique                               # one entry per month


def test_whole_contracts_never_fractional(mkt):
    res = run(replace(TAIL, entry="weekly", entry_weekday=2, max_open=10), mkt)
    c = res.trades["contracts"]
    assert len(c) > 10 and np.allclose(c, np.round(c)) and (c >= 1).all()
    assert np.allclose(res.contracts, np.round(res.contracts))


def test_under_one_contract_is_skipped(mkt):
    res = run(replace(TAIL, sizing="premium_usd", size_usd=1.0), mkt)    # $1 can't buy a 100x contract
    assert res.trades.empty and res.skipped.get("under one contract", 0) > 0


def test_fixed_contracts_and_dollar_pnl(mkt):
    res = run(replace(TAIL, sizing="contracts", size_contracts=2, entry="monthly", max_open=4), mkt)
    assert (res.trades["contracts"] == 2).all()
    closed = res.trades.dropna(subset=["pnl_$"])
    assert len(closed) and (closed["pnl_$"] >= -closed["premium_$"] * 1.2).all()   # a long put loses ≤ premium (+ spread)
    assert abs(res.equity.iloc[0] - 5_000) < 100                          # equity is in dollars


def test_cash_check_limits_debits(mkt):
    res = run(replace(TAIL, sizing="premium_usd", size_usd=50_000.0, entry="weekly", max_open=50), mkt)
    assert (res.trades["premium_$"] <= res.equity.reindex(pd.to_datetime(res.trades["open"])).values * 1.01).all()


def test_roll_reopens_same_day(mkt):
    res = run(replace(TAIL, roll_dte=21, entry="when_flat", max_open=1), mkt)
    tr = res.trades
    rolled = tr[tr["reason"] == "roll"]
    assert len(rolled) > 3
    opens = set(pd.to_datetime(tr["open"]).dt.date)
    assert all(d in opens for d in rolled["close"])                       # each roll reopened that day


def test_every_n_and_weekly_schedules():
    idx = pd.bdate_range("2020-01-01", periods=60)
    s = E.entry_schedule(E.OptionStrategy("x", entry="every_n", entry_every=5), idx)
    assert s.sum() == 12
    w = E.entry_schedule(E.OptionStrategy("x", entry="weekly", entry_weekday=4), idx)
    assert set(idx[w].weekday) == {4}
    m = E.entry_schedule(E.OptionStrategy("x", entry="monthly", entry_day=3), idx)
    assert m.sum() == 3


def test_default_strategy_unchanged_shape(mkt):
    res = run(E.TEMPLATES["Short put (PUT-style)"], mkt)
    assert res.positions.max() == 1 and "contracts" not in res.trades
    assert abs(res.equity.iloc[0] - 1.0) < 0.05


def test_account_equity_sums(mkt):
    S = mkt[0]
    res = run(replace(TAIL, entry="monthly", max_open=3), mkt)
    acct = E.account_equity(res.equity, {"SPY": S}, {"SPY": 25_000.0}, 1_000.0, mkt[3])
    assert np.allclose(acct["total"], acct["holdings"] + acct["options"] + acct["cash"])
    assert abs(acct["holdings"].iloc[0] - 25_000) < 1e-6 and acct["cash"].iloc[-1] > 1_000


def test_topup_is_contribution_not_return(mkt):
    base = replace(TAIL, entry="monthly", max_open=3)
    a = run(base, mkt)
    b = run(replace(base, topup_usd=300.0), mkt)
    assert b.contrib.sum() > 0 and a.contrib.sum() == 0
    # with no trading at all the top-up adds dollars but no return
    idle = replace(base, sizing="contracts", size_contracts=0)
    c = run(replace(idle, topup_usd=300.0), mkt)
    assert c.trades.empty and c.equity.iloc[-1] > 5_000 + c.contrib.sum() * 0.99
    assert (c.returns < 0.01).all()                                         # only T-bill interest
    acct = E.account_equity(c.equity, {}, {}, 10_000.0, mkt[3], c.contrib)
    assert abs(acct["total"].iloc[-1] - acct["options"].iloc[-1] - acct["cash"].iloc[-1]) < 1e-6
    assert acct["cash"].iloc[-1] < 10_000                                   # top-ups came out of idle cash

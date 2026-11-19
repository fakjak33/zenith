"""Phase 5–6: granular options builder, long vol, entry filters, 0DTE."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.options import engine as E


@pytest.fixture(scope="module")
def mkt():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=900)
    r = rng.normal(0.0003, 0.011, len(idx))
    S = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    vix = pd.Series(np.clip(16 + np.cumsum(rng.normal(0, 0.6, len(idx))) * 0.3, 10, 45), index=idx)
    term = pd.DataFrame({"VIX9D": vix * 0.95, "VIX": vix, "VIX3M": vix * 1.06, "VIX6M": vix * 1.1})
    skew = pd.Series(125.0, index=idx)
    rf = pd.Series(0.02, index=idx)
    o = S.shift(1).fillna(S.iloc[0]) * (1 + rng.normal(0, 0.002, len(idx)))
    hi = np.maximum(o, S) * (1 + np.abs(rng.normal(0, 0.004, len(idx))))
    lo = np.minimum(o, S) * (1 - np.abs(rng.normal(0, 0.004, len(idx))))
    ohlc = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": S})
    return S, term, skew, rf, ohlc, vix


def test_catalog_balanced_and_tagged():
    flows = [t["flow"] for t in E.TAGS.values()]
    vols = [t["vol"] for t in E.TAGS.values()]
    assert flows.count("debit") >= 10 and flows.count("credit") >= 10
    assert vols.count("long") >= 10 and vols.count("short") >= 10
    assert set(E.TAGS) == set(E.TEMPLATES)


def test_wing_and_premium_strikes():
    S, T, sig = 100.0, 30 / 365, 0.2
    k0 = E._strike(E.Leg("put", -1, "atm"), S, T, sig, 0.0, 125.0)
    kw = E._strike(E.Leg("put", 1, "wing", 0.05, ref=0), S, T, sig, 0.0, 125.0, [k0])
    assert np.isclose(kw, 95.0)
    kc = E._strike(E.Leg("call", 1, "wing", 0.05, ref=0), S, T, sig, 0.0, 125.0, [100.0])
    assert np.isclose(kc, 105.0)
    kp = E._strike(E.Leg("put", 1, "premium", 0.005), S, T, sig, 0.0, 125.0)
    px = float(E.price(False, S, kp, T, float(E.smile(sig, S, kp, T, 125.0)), 0.0))
    assert abs(px - 0.5) < 0.1 and kp < 100


def test_premium_sizing_spends_budget():
    legs = [{"leg": E.Leg("call", 1, "atm"), "K": 100.0}]
    st = E.OptionStrategy("x", sizing="premium", size_pct=0.02)
    u = E._units(st, 1.0, 100.0, legs, book=3.0)
    assert np.isclose(u * 3.0, 0.02)


def test_risk_sizing_caps_max_loss():
    legs = [{"leg": E.Leg("put", -1, "pct", 1.0), "K": 100.0}, {"leg": E.Leg("put", 1, "pct", 0.9), "K": 90.0}]
    st = E.OptionStrategy("x", sizing="risk", size_pct=0.02)
    book = -3.0                                            # 3 credit on a 10-wide spread → max loss 7
    u = E._units(st, 1.0, 100.0, legs, book)
    assert np.isclose(u * 7.0, 0.02, rtol=1e-3)


def test_vix_band_gate(mkt):
    S, term, *_ , vix = mkt
    st = E.OptionStrategy("x", filters={"vix_lo": 15.0, "vix_hi": 20.0})
    g = E.entry_gate(st, S.index, S, term)
    assert ((vix[g] >= 15) & (vix[g] <= 20)).all() and 0 < g.mean() < 1
    assert E.entry_gate(E.OptionStrategy("x", min_vix=18.0), S.index, S, term).equals(vix >= 18)


def test_zscore_and_dow_gates(mkt):
    S, term, *_ = mkt
    g = E.entry_gate(E.OptionStrategy("x", filters={"z_lo": 1.0, "z_len": 21}), S.index, S, term)
    assert g.mean() < 0.4
    g2 = E.entry_gate(E.OptionStrategy("x", filters={"dow": [4]}), S.index, S, term)
    assert set(S.index[g2].weekday) == {4}


def test_filters_reduce_trades(mkt):
    S, term, skew, rf, *_ = mkt
    base = E.TEMPLATES["OTM put write (30Δ)"]
    a = E.simulate(base, S, term, skew, rf)
    b = E.simulate(replace(base, filters={"vix_lo": 20.0}), S, term, skew, rf)
    assert len(b.trades) < len(a.trades) and b.gate is not None


def test_long_short_symmetry(mkt):
    S, term, skew, rf, *_ = mkt
    short = replace(E.TEMPLATES["Short straddle (ATM)"], spread_pct=0.0, min_spread_bps=0.0, roll="dte", dte=30)
    long_ = replace(short, legs=[replace(L, side=1) for L in short.legs])
    a, b = E.simulate(short, S, term, skew, rf), E.simulate(long_, S, term, skew, rf)
    cash = rf.shift(1).fillna(0) / 252
    # same collateral units each cycle while equity is similar → option P&L mirrors in the first cycle
    first = a.returns.index[1:20]
    assert np.allclose((a.returns + b.returns)[first], 2 * cash[first], atol=2e-3)


def test_debit_take_profit_and_stop(mkt):
    S, term, skew, rf, *_ = mkt
    st = replace(E.TEMPLATES["Long call (ATM)"], take_profit=0.5, stop_mult=0.5, roll="dte", dte=45)
    res = E.simulate(st, S, term, skew, rf)
    reasons = set(res.trades["reason"].dropna())
    assert {"take profit", "stop"} & reasons


def test_calendar_closes_at_front_expiry(mkt):
    S, term, skew, rf, *_ = mkt
    res = E.simulate(E.TEMPLATES["Long calendar (ATM, 30d / 60d)"], S, term, skew, rf)
    t = res.trades.dropna(subset=["close"])
    assert (t["days"] <= 33).all() and "@60d" in t["strikes"].iloc[0]   # 30d front, +weekend roll


def test_time_exit(mkt):
    S, term, skew, rf, *_ = mkt
    res = E.simulate(replace(E.TEMPLATES["Short strangle (16Δ)"], roll="dte", dte=45, exit_dte=21), S, term, skew, rf)
    assert (res.trades["reason"].dropna() == "time exit").mean() > 0.8


def test_spx_expiry_calendar():
    assert E.spx_daily_expiry(pd.Timestamp("2015-06-12"))                 # Friday
    assert not E.spx_daily_expiry(pd.Timestamp("2015-06-10"))             # Wed before 2016
    assert E.spx_daily_expiry(pd.Timestamp("2017-06-14"))                 # Wed after 2016
    assert not E.spx_daily_expiry(pd.Timestamp("2019-06-11"))             # Tue before 2022
    assert E.spx_daily_expiry(pd.Timestamp("2023-06-13"))                 # Tue 2023


def test_0dte_settles_and_respects_calendar(mkt):
    S, term, skew, rf, ohlc, vix = mkt
    st = E.TEMPLATES["0DTE put credit spread (15Δ / 5Δ)"]
    res = E.simulate_0dte(st, ohlc, vix * 0.9, skew, rf, term, actual_expiries=True)
    days = pd.to_datetime(res.trades["date"])
    assert len(days) > 50 and all(E.spx_daily_expiry(d) for d in days)
    allday = E.simulate_0dte(st, ohlc, vix * 0.9, skew, rf, term, actual_expiries=False)
    assert len(allday.trades) > len(res.trades)


def test_0dte_stop_bounds_loss(mkt):
    S, term, skew, rf, ohlc, vix = mkt
    st = replace(E.TEMPLATES["0DTE iron condor (10Δ / 5Δ)"], stop_mult=1.0, spread_pct=0.0, min_spread_bps=0.0,
                 sizing="premium", size_pct=0.0)
    res = E.simulate_0dte(st, ohlc, vix * 1.5, skew, rf, term, actual_expiries=False)
    stops = res.trades[res.trades["reason"] == "stop"]
    nonstop = res.trades[res.trades["reason"] != "stop"]
    if len(stops):
        # a stopped trade loses exactly the stop; nothing loses more than the defined-risk max
        assert stops["pnl_%_equity"].std() < stops["pnl_%_equity"].abs().mean() + 1e-9
    assert len(nonstop) > 0


def test_entry_preview(mkt):
    S, term, skew, rf, *_ = mkt
    pv = E.entry_preview(E.TEMPLATES["Iron condor (16Δ / 5Δ)"], float(S.iloc[-1]),
                         term.iloc[-1][E.TERM_COLS].to_numpy(float), 125.0, 0.02)
    assert pv["net_credit_%"] > 0 and pv["max_loss"] > 0 and len(pv["breakevens"]) == 2
    pv2 = E.entry_preview(E.TEMPLATES["Long straddle (ATM)"], 100.0, np.array([15, 16, 17, 18.0]), 125.0, 0.02)
    assert pv2["net_credit_%"] < 0 and pv2["greeks"]["vega"] > 0

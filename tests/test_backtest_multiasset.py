"""Phase 4: futures sizing, portfolio combination, synthetic options engine."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.config import FUTURES, BacktestConfig
from zenith.backtest.data.synthetic import generate
from zenith.backtest.engine.engine import buffer_and_round, run_single
from zenith.backtest.options import engine as E
from zenith.backtest.portfolio import combine
from zenith.backtest.rules import REGISTRY
from zenith.backtest.spec import StrategySpec


def test_whole_contracts_and_fees():
    df = generate("trending", n_days=1500, seed=4) * 1.0
    df[["open", "high", "low", "close"]] *= 40          # ~ES-like price level
    fc = REGISTRY["ema"]().forecast(df)
    m = FUTURES["ES=F"][1]
    a = run_single(fc, df, BacktestConfig(multiplier=m, fee_per_contract=0.0))
    b = run_single(fc, df, BacktestConfig(multiplier=m, fee_per_contract=5.0))
    contracts = a.position / m
    assert np.allclose(contracts, contracts.round())
    assert b.costs.sum() > a.costs.sum()


def test_buffer_cuts_turnover():
    df = generate("gbm", n_days=1500, seed=6)
    fc = REGISTRY["ema"](fast=4, slow=16).forecast(df)
    raw = run_single(fc, df, BacktestConfig())
    buf = run_single(fc, df, BacktestConfig(buffer=0.2))
    assert buf.position.diff().abs().sum() < raw.position.diff().abs().sum()


def test_buffer_is_causal():
    p = pd.Series([0, 100, 100, 300, 50.0])
    a = buffer_and_round(p, pd.Series(20.0, index=p.index), None)
    b = buffer_and_round(pd.concat([p.iloc[:3], pd.Series([-999.0, 999.0], index=[3, 4])]),
                         pd.Series(20.0, index=p.index), None)
    assert a.iloc[:3].equals(b.iloc[:3])


def test_spec_uses_contract_table():
    s = StrategySpec(name="ES=F")
    assert s.bt().multiplier == 50 and s.bt().fee_per_contract == 2.5
    assert StrategySpec(name="SPY").bt().multiplier is None
    assert StrategySpec(name="ES=F", src="Synthetic").bt().multiplier is None


def test_portfolio_diversifies_uncorrelated():
    idx = pd.bdate_range("2010-01-01", periods=2000)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0004, 0.0125, (2000, 4)), index=idx, columns=list("abcd"))
    out = combine(rets, "equal", 252)
    assert out["idm"].iloc[-1] == pytest.approx(2.0, abs=0.25)        # 1/sqrt(1/4) for uncorrelated
    port_vol = out["returns"].iloc[300:].std()
    assert port_vol == pytest.approx(rets.std().mean(), rel=0.15)     # IDM restores the vol target
    assert out["risk_contrib"].sum() == pytest.approx(1.0)


def test_portfolio_weights_are_causal():
    idx = pd.bdate_range("2010-01-01", periods=600)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 2)), index=idx, columns=["a", "b"])
    w1 = combine(rets, "inverse_vol", 100)["weights"]
    shocked = rets.copy()
    shocked.iloc[400:, 0] *= 10
    w2 = combine(shocked, "inverse_vol", 100)["weights"]
    pd.testing.assert_frame_equal(w1.iloc[:401], w2.iloc[:401])


# ------------------------------------------------------------------ options
@pytest.fixture(scope="module")
def market():
    idx = pd.bdate_range("2012-01-02", periods=900)
    rng = np.random.default_rng(2)
    S = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, len(idx)))), index=idx)
    term = pd.DataFrame({"VIX9D": 15.0, "VIX": 16.0, "VIX3M": 17.5, "VIX6M": 18.5}, index=idx)
    return S, term, pd.Series(125.0, index=idx), pd.Series(0.02, index=idx)


def test_term_structure_interpolates_in_variance():
    row = np.array([15.0, 16.0, 17.5, 18.5])
    assert E.atm_vol(row, 30 / 365) == pytest.approx(0.16)
    v60 = E.atm_vol(row, 60 / 365)
    assert 0.16 < v60 < 0.175
    assert E.atm_vol(np.array([np.nan, 20.0, np.nan, np.nan]), 0.5) == pytest.approx(0.20)


def test_smile_puts_richer_than_calls():
    iv = E.smile(0.16, 100.0, np.array([90.0, 100.0, 110.0]), 30 / 365, 130.0)
    assert iv[0] > iv[1] > iv[2]


def test_third_fridays():
    f = E.third_fridays("2024-01-01", "2024-03-31")
    assert [d.day for d in f] == [19, 16, 15] and all(d.weekday() == 4 for d in f)


def test_short_put_collects_premium_in_calm_market(market):
    S, term, skew, rf = market
    calm = pd.Series(np.linspace(2000, 2100, len(S)), index=S.index)
    res = E.simulate(E.TEMPLATES["Short put (PUT-style)"], calm, term, skew, rf)
    assert res.returns.sum() > 0
    assert len(res.trades) >= 30
    assert (res.exposure["delta"] > 0).mean() > 0.9          # short put = long delta


def test_costs_reduce_option_returns(market):
    S, term, skew, rf = market
    from dataclasses import replace
    t = E.TEMPLATES["Short strangle (16Δ)"]
    a = E.simulate(replace(t, spread_pct=0.0, min_spread_bps=0.0), S, term, skew, rf)
    b = E.simulate(replace(t, spread_pct=0.10), S, term, skew, rf)
    assert b.equity.iloc[-1] < a.equity.iloc[-1]


def test_iron_condor_bounded_loss(market):
    S, term, skew, rf = market
    crash = S.copy()
    crash.iloc[450:] *= 0.7
    res = E.simulate(E.TEMPLATES["Iron condor (16Δ / 5Δ)"], crash, term, skew, rf)
    assert res.returns.min() > -0.2                          # wings cap the loss on a 30% gap


def test_calibration_metrics():
    idx = pd.bdate_range("2015-01-01", periods=800)
    r = pd.Series(np.random.default_rng(3).normal(0.0003, 0.008, len(idx)), index=idx)
    level = (1 + r).cumprod() * 100
    cal = E.calibrate(r, level)
    assert cal["corr"] > 0.99 and cal["tracking_error"] < 1e-6
    assert E.calibration_tier(cal)[0] == "A"

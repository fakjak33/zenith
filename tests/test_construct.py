"""CONSTRUCT: the portfolio-construction maths, checked against closed forms and simulation."""

import numpy as np
import pandas as pd
import pytest

from zenith.construct import engine as K


def _returns(n=2500, k=3, rho=0.3, seed=0, mu=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    C = np.full((k, k), rho) + np.eye(k) * (1 - rho)
    L = np.linalg.cholesky(C)
    X = rng.normal(0, 1, (n, k)) @ L.T * vol + mu
    return pd.DataFrame(X, index=pd.bdate_range("2010-01-01", periods=n), columns=[f"A{i}" for i in range(k)])


def test_ew_vol_floor():
    assert np.isclose(K.ew_portfolio_vol(1, 0.2, 0.3), 0.2)
    assert np.isclose(K.ew_portfolio_vol(1e9, 0.2, 0.25), 0.2 * 0.5, atol=1e-6)


def test_idm_two_uncorrelated():
    assert np.isclose(K.diversification_multiplier([0.5, 0.5], np.eye(2)), np.sqrt(2))
    # negative correlations are floored at 0 (Carver), and the cap holds
    assert np.isclose(K.diversification_multiplier([0.5, 0.5], [[1, -0.9], [-0.9, 1]]), np.sqrt(2))
    assert K.diversification_multiplier(np.full(50, 0.02), np.eye(50)) == 2.5


def test_vol_drag():
    assert np.isclose(K.geometric_from_arithmetic(0.10, 0.20), 0.08)


def test_shannon_closed_form_and_simulation():
    # +100% / −50% coin: the asset alone has zero expected log growth; a 50/50 rebalanced mix grows
    assert np.isclose(K.constant_mix_growth(1.0, 1.0, 0.5), 0.0)
    g = float(K.constant_mix_growth(0.5, 1.0, 0.5))
    assert np.isclose(g, 0.5 * (np.log(1.5) + np.log(0.75)))
    assert g > 0
    sim = K.shannon_paths(n_paths=400, n_steps=200, w=0.5, seed=1)
    sim_g = np.log(sim["mix"][:, -1]).mean() / 200
    assert abs(sim_g - g) < 0.01
    assert np.median(sim["asset"][:, -1]) < np.median(sim["mix"][:, -1])
    assert np.isclose(K.kelly_fraction(1.0, 0.5), 0.5)


def test_rebalancing_premium_formula():
    p = K.rebalancing_premium([0.5, 0.5], [0.2, 0.2], [[1, 0], [0, 1]])
    assert np.isclose(p, 0.5 * (0.04 - 0.02))


def test_add_test_equivalence():
    for rho in (-0.2, 0.3, 0.8):
        R = _returns(k=2, rho=rho, seed=int(10 + rho * 10))
        R["A0"] += 0.0002
        d = K.add_test_returns(R["A0"], R["A1"])
        assert d["adds"] == d["adds_beta_form"] == (d["alpha"] > 0)
    assert K.add_test(0.55, 0.70, 0.8)["adds"] is False and K.add_test(0.57, 0.70, 0.8)["adds"] is True


def test_optimal_mix_bounds():
    assert K.optimal_mix(0.5, 0.5, 0.0) == pytest.approx(0.5)
    assert K.optimal_mix(0.1, 0.8, 0.9) == 0.0                 # fails the hurdle → none of it


def test_weight_schemes_sum_to_one_and_erc_equalises():
    R = _returns(k=4, rho=0.2)
    R["A3"] *= 3                                               # one riskier asset
    cov = np.cov(R.to_numpy().T)
    for s in K.SCHEMES:
        w = K.weights_for(s, R)
        assert np.isclose(w.sum(), 1.0, atol=1e-6) and (w >= -1e-9).all(), s
    rc = K.risk_contributions(K.risk_parity(cov), cov)
    assert np.allclose(rc, 0.25, atol=0.01)
    assert K.weights_for("inverse_vol", R)[3] < 0.15


def test_ledoit_wolf_is_shrunk():
    X = _returns(n=60, k=10).to_numpy()
    S, LW = np.cov(X.T, bias=True), K.ledoit_wolf(X)
    assert np.linalg.cond(LW) < np.linalg.cond(S)


def test_run_mix_rebalance_rules():
    R = _returns(k=2, rho=0.0)
    w = np.array([0.6, 0.4])
    none = K.run_mix(R, w, "none")
    q = K.run_mix(R, w, "quarterly")
    band = K.run_mix(R, w, "band", band=0.02)
    assert none["rebalances"] == 0 and 35 <= q["rebalances"] <= 45 and band["rebalances"] > 0
    vt = K.run_mix(R, w, "monthly", vol_target=0.10)
    assert abs(vt["returns"].iloc[200:].std() * np.sqrt(252) - 0.10) < 0.03


def test_leverage_and_drawdown_odds():
    df = K.growth_vs_leverage(0.08, 0.2, 0.0)
    assert np.isclose(df.loc[df["growth"].idxmax(), "L"], 2.0, atol=0.03)          # Kelly = μ/σ²
    assert K.prob_ever_drawdown(-0.01, 0.2, 0.5) == 1.0
    assert 0 < K.prob_ever_drawdown(0.05, 0.2, 0.5) < 1


def test_crisis_table_and_trend_sleeve():
    R = _returns(k=3, rho=0.5)
    t = K.crisis_table(R, "A0")
    assert list(t.index) == ["A1", "A2"] and t["corr_all"].between(-1, 1).all()
    px = (1 + R).cumprod() * 100
    s = K.trend_sleeve(px)
    assert len(s) == len(px) and np.isfinite(s).all()


# ------------------------------------------------------------------ page --
_STUB = '''
import numpy as np, pandas as pd
import zenith.construct.view as V
def _fake(tickers, start="2005-01-01"):
    rng = np.random.default_rng(len(tickers))
    idx = pd.bdate_range("2008-01-01", periods=2600)
    X = rng.normal(0.0003, 0.011, (len(idx), len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(X, axis=0)), index=idx, columns=list(tickers))
V._prices = _fake
V._library_returns = lambda: {}
import streamlit as st
st.session_state.setdefault("cn_mode", MODE)
st.session_state.setdefault("cn_ch", CH)
V.render()
'''


@pytest.mark.parametrize("mode,ch", [("GUIDE", c) for c in
                                     ["01 DIVERSIFY", "02 VOL DRAG", "03 SHANNON'S DEMON", "04 SHOULD I ADD IT?",
                                      "05 WEIGHTING", "06 REBALANCING", "07 LEVERAGE", "08 CRISES"]]
                         + [("SANDBOX", "01 DIVERSIFY"), ("SOURCES", "01 DIVERSIFY")])
def test_construct_page_renders(mode, ch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(_STUB.replace("MODE", repr(mode)).replace("CH", repr(ch)), default_timeout=300)
    at.run()
    assert not at.exception, at.exception

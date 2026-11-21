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


# ------------------------------------------------- drawdown budget & tails --
_DL = [-0.10, -0.15, -0.25, -0.35, 0.50, 0.35, 0.25, 0.15, 0.10]


def test_napkin_60_40():
    assert np.isclose(K.napkin_equity_weight(0.30, 0.50, 0.0), 0.60)
    # a cushion that rallies lets you hold more equity; one that falls (2022) forces less
    assert K.napkin_equity_weight(0.30, 0.50, 0.05) > 0.60 > K.napkin_equity_weight(0.30, 0.50, -0.17)


def test_bs_delta_strike_round_trip_and_spy_example():
    from scipy.stats import norm
    T, s = 60 / 365, 0.25
    for d in (-0.10, -0.25, 0.25, 0.10):
        k = K.bs_delta_strike(s, T, d)
        d1 = (np.log(1 / k) + 0.5 * s * s * T) / (s * np.sqrt(T))
        assert np.isclose(norm.cdf(d1) - (1 if d < 0 else 0), d, atol=1e-9)
    # 2-month SPY, 10-delta put 29.8% / call 14.4% IV: about -14% vs +8% -> ~1.7x, not 2x
    put = K.bs_delta_strike(0.298, T, -0.10) - 1
    call = K.bs_delta_strike(0.144, T, 0.10) - 1
    assert -0.15 < put < -0.12 and 0.07 < call < 0.09


def test_rn_density_flat_smile_is_lognormal():
    T, s = 0.25, 0.20
    x, p = K.rn_density([s] * 9, _DL, T)
    assert np.isclose(p.sum(), 1.0) and abs((x * p).sum()) < 1e-6           # a martingale
    sd = np.sqrt(((np.log1p(x) - (np.log1p(x) * p).sum()) ** 2 * p).sum())
    assert np.isclose(sd, s * np.sqrt(T), rtol=0.03)
    t = K.tail_measures(x, p, 0.10)
    assert t["skew"] > 0 and t["ratio"] > 1                                   # lognormal: right-skewed


def test_implied_tails_put_skew_lowers_ratio():
    flat = K.implied_tails([0.18] * 9, _DL, 60 / 365)
    skewed = K.implied_tails([0.30, 0.27, 0.23, 0.21, 0.19, 0.17, 0.16, 0.15, 0.145], _DL, 60 / 365)
    assert skewed["ratio"] < flat["ratio"] and skewed["skew"] < 0
    assert skewed["put10_dist"] < 0 < skewed["call10_dist"]
    assert K.implied_tails([np.nan] * 9, _DL, 0.25) == {}


def test_clean_smile_drops_bad_wing_prints():
    eem = [0.434, 0.389, 0.391, 0.308, 0.308, 0.314, 0.286, 0.271, 0.733]      # 10C at 73% next to 27%
    ivs, n = K.clean_smile(eem, _DL)
    assert n == 1 and np.isnan(ivs[-1]) and ivs[:8] == eem[:8]
    spy = [0.298, 0.2668, 0.231, 0.2087, 0.1876, 0.1663, 0.156, 0.1472, 0.1437]
    assert K.clean_smile(spy, _DL) == (spy, 0)                                  # a real put skew survives
    t = K.implied_tails(eem, _DL, 60 / 365)
    assert t["dropped"] == 1 and np.isnan(t["call10_dist"]) and t["etg"] < 0.25


def test_tail_measures_normal():
    from scipy.stats import norm
    x = np.random.default_rng(0).normal(0, 1, 400_000)
    t = K.tail_measures(x, None, 0.10)
    exact = norm.pdf(norm.ppf(0.10)) / 0.10                                  # E[X | X < q10] for N(0,1)
    assert np.isclose(-t["etl"], exact, rtol=0.01) and np.isclose(t["etg"], exact, rtol=0.01)
    assert np.isclose(t["ratio"], 1, atol=0.01) and abs(t["skew"]) < 0.02


def test_crisis_returns_and_lenses():
    idx = pd.bdate_range("2007-01-01", "2010-12-31")
    a = pd.Series(100.0, index=idx)
    a[(idx >= "2007-10-09")] = 50.0                                          # -50% across the GFC window
    b = pd.Series(100.0, index=idx)
    b[(idx >= "2007-10-09")] = 110.0                                         # +10% cushion
    cr = K.crisis_returns(pd.DataFrame({"EQ": a, "BD": b}), {"GFC": ("2007-10-09", "2009-03-09")})
    assert np.isclose(cr.loc["GFC", "EQ"], 0.0) and np.isclose(cr.loc["GFC", "BD"], 0.0)
    cr = K.crisis_returns(pd.DataFrame({"EQ": a, "BD": b}), {"GFC": ("2007-10-01", "2009-03-09")})
    assert np.isclose(cr.loc["GFC", "EQ"], -0.5) and np.isclose(cr.loc["GFC", "BD"], 0.1)
    assert cr.loc[:, :].shape == (1, 2)
    L = K.stress_lenses(np.array([0.6, 0.4]), crisis=cr, implied_etl=[-0.2, -0.08], crisis_sign=[-0.5, 0.1])
    assert np.isclose(L["crisis"], 0.26) and np.isclose(L["implied"], 0.12)   # bond: no charge, no credit
    assert L["binding"] == "crisis" and np.isclose(L["worst"], 0.26)
    assert np.isclose(K.headroom(0.30, 0.26)["scale"], 0.30 / 0.26)


def test_max_convexity_and_budget_fit():
    rng = np.random.default_rng(4)
    n = 3000
    eq = rng.standard_t(4, n) * 0.06 - np.where(rng.random(n) < 0.05, 0.15, 0)   # crash-prone
    hedge = -0.5 * eq + rng.normal(0, 0.03, n)                                       # convex payoff vs eq
    noise = rng.normal(0.0, 0.05, n)
    H = pd.DataFrame({"EQ": eq, "HEDGE": hedge, "N": noise})
    w = K.max_convexity(H, 0.10, max_w=0.8)
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all() and (w <= 0.8 + 1e-6).all()
    assert K.portfolio_tails(w, H)["ratio"] >= K.portfolio_tails(np.full(3, 1 / 3), H)["ratio"] - 1e-9
    assert w[1] > w[0]                                                                # prefers the convex leg
    lens = lambda ww: K.stress_lenses(ww, implied_etl=[-0.4, -0.1, -0.1])            # noqa: E731
    s, L = K.fit_to_budget(np.array([1.0, 0, 0]), 0.20, lens)
    assert np.isclose(s, 0.5, atol=1e-3) and L["worst"] <= 0.20 + 1e-9
    assert K.fit_to_budget(np.array([0, 1.0, 0]), 0.20, lens)[0] == 1.0             # already inside budget


def test_implied_tail_breach_and_zscore():
    idx = pd.bdate_range("2010-01-01", periods=800)
    px = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 800))), index=idx)
    df = K.implied_tail_breach(px, pd.Series(16.0, index=idx), 63, 0.10)
    assert len(df) == 800 - 63 and (df["tail"] < 0).all()
    assert 0.0 <= df["breach"].mean() < 0.25
    z = K.skew_zscore(pd.Series(np.r_[np.full(300, 120.0) + np.random.default_rng(2).normal(0, 1, 300), 140.0],
                                index=pd.bdate_range("2015-01-01", periods=301)))
    assert z.iloc[-1] > 5


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
def _cboe(name):
    idx = pd.bdate_range("2008-01-01", periods=2600)
    return pd.Series(20 + np.random.default_rng(3).normal(0, 1, len(idx)).cumsum() * 0.1 + (100 if name == "SKEW" else 0),
                     index=idx)
V._cboe = _cboe
V._surf_hist = lambda t: pd.DataFrame()
import streamlit as st
st.session_state.setdefault("cn_mode", MODE)
st.session_state.setdefault("cn_ch", CH)
V.render()
'''


@pytest.mark.parametrize("mode,ch", [("GUIDE", c) for c in
                                     ["01 DIVERSIFY", "02 VOL DRAG", "03 SHANNON'S DEMON", "04 SHOULD I ADD IT?",
                                      "05 WEIGHTING", "06 REBALANCING", "07 LEVERAGE", "08 CRISES",
                                      "09 DRAWDOWN BUDGET", "10 TAILS & CONVEXITY"]]
                         + [("SANDBOX", "01 DIVERSIFY"), ("SOURCES", "01 DIVERSIFY")])
def test_construct_page_renders(mode, ch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(_STUB.replace("MODE", repr(mode)).replace("CH", repr(ch)), default_timeout=300)
    at.run()
    assert not at.exception, at.exception

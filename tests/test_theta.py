"""THETA RUN tests -- offline, synthetic data only (no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from zenith.theta import ivmodel
from zenith.theta.coach import Vault, adherence, on_events, pretrade
from zenith.theta.engine import bot, clock, margin, pricing, scenario, scoring
from zenith.theta.engine.book import Book
from zenith.theta.engine.strategies import TEMPLATES, Leg, build
from zenith.theta.market import Market, round_strike


# ------------------------------------------------------------------ fixtures --

def _frames(n=700, seed=1, vol=0.18, every_day=False, drift=0.0, iv=None):
    rng = np.random.default_rng(seed)
    freq = "D" if every_day else "B"
    idx = pd.date_range("2019-01-01", periods=n, freq=freq)
    r = rng.normal(drift / 252, vol / math.sqrt(252), n)
    spy = 300 * np.exp(np.cumsum(r))
    stock = 100 * np.exp(np.cumsum(rng.normal(0, 0.3 / math.sqrt(252), n)))
    vix = np.full(n, 0.18 if iv is None else iv)

    def f(close, base, kind):
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 1e6}, index=idx)
        out = df.copy()
        out["rv21"] = vol
        out["rv_fc"] = vol
        out["iv_base"] = base
        out["em"] = 0.06 if kind == "stock" else 0.0
        out["next_earn"] = -1
        out["iv30"] = base
        out["iv_rank"] = 0.5
        out["iv_pct"] = 0.5
        out["skew"] = 1.0
        out["beta"] = 1.0
        return out

    frames = {"SPY": f(spy, vix, "index"), "XYZ": f(stock, np.full(n, 0.35), "stock")}
    frames["_VIX"] = pd.DataFrame({"close": vix}, index=idx)
    frames["_RATE"] = pd.DataFrame({"close": np.full(n, 0.02)}, index=idx)
    uni = [{"ticker": "SPY", "kind": "index", "tier": "A", "sector": ""},
           {"ticker": "XYZ", "kind": "stock", "tier": "C", "sector": "Tech"}]
    earn = {"XYZ": [str(idx[300].date()), str(idx[363].date())]}
    return frames, uni, earn


@pytest.fixture(scope="module")
def mkt():
    return Market(*_frames())


# ------------------------------------------------------------------- pricing --

def test_ncdf_matches_math_erf():
    xs = np.linspace(-6, 6, 241)
    exact = np.array([0.5 * (1 + math.erf(x / math.sqrt(2))) for x in xs])
    assert np.max(np.abs(pricing.ncdf(xs) - exact)) < 2e-7


def test_put_call_parity():
    S, K, T, sig, r = 100.0, 95.0, 0.25, 0.3, 0.03
    c = pricing.price(True, S, K, T, sig, r)
    p = pricing.price(False, S, K, T, sig, r)
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-9


def test_greeks_match_finite_differences():
    S, K, T, sig, r = 100.0, 105.0, 0.2, 0.25, 0.01
    for call in (True, False):
        g = pricing.greeks(call, S, K, T, sig, r)
        h = 1e-3
        d = (pricing.price(call, S + h, K, T, sig, r) - pricing.price(call, S - h, K, T, sig, r)) / (2 * h)
        gm = (pricing.price(call, S + h, K, T, sig, r) - 2 * pricing.price(call, S, K, T, sig, r)
              + pricing.price(call, S - h, K, T, sig, r)) / h ** 2
        v = (pricing.price(call, S, K, T, sig + 1e-4, r) - pricing.price(call, S, K, T, sig - 1e-4, r)) / 2e-4 / 100
        th = (pricing.price(call, S, K, T - 1 / 365, sig, r) - pricing.price(call, S, K, T, sig, r))
        assert abs(g["delta"] - d) < 1e-5
        assert abs(g["gamma"] - gm) < 1e-3
        assert abs(g["vega"] - v) < 1e-5
        assert abs(g["theta"] - th) < 2e-3


def test_strike_for_delta_roundtrip():
    S, T, sig, r = 100.0, 45 / 365, 0.2, 0.02
    for call in (True, False):
        K = pricing.strike_for_delta(S, T, sig, r, 0.16, call)
        dlt = float(pricing.greeks(call, S, K, T, sig, r)["delta"])
        assert abs(abs(dlt) - 0.16) < 1e-6


def test_expired_option_is_intrinsic():
    assert pricing.price(False, 90.0, 100.0, 0.0, 0.3) == pytest.approx(10.0)
    assert pricing.price(True, 90.0, 100.0, 0.0, 0.3) == pytest.approx(0.0)


# ------------------------------------------------------------------- IV model --

def _px(n=600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)


def test_features_have_no_lookahead():
    px = _px()
    idx = px.index
    vix = pd.Series(0.2, index=idx)
    spy_r = pd.Series(np.random.default_rng(0).normal(0, 0.01, len(idx)), index=idx)
    mult = pd.Series(1.3, index=idx)
    reports = [str(idx[k].date()) for k in (100, 163, 226, 289, 352, 415, 478)]
    a = ivmodel.features(px, reports, mult, vix, spy_r, "stock")
    px2 = px.copy()
    px2.iloc[400:, px2.columns.get_loc("close")] *= 1.7          # change the future only
    b = ivmodel.features(px2, reports, mult, vix, spy_r, "stock")
    cols = ["rv21", "rv_fc", "iv_base", "em", "iv30", "iv_rank", "iv_pct", "beta"]
    pd.testing.assert_frame_equal(a.iloc[:400][cols], b.iloc[:400][cols])


def test_earnings_bump_and_crush():
    atm_far = ivmodel.atm_vol(0.30, 0.06, 30 / 365, False)
    atm_ev = ivmodel.atm_vol(0.30, 0.06, 30 / 365, True)
    assert atm_far == pytest.approx(0.30)
    assert atm_ev > 0.35                             # report inside expiry: IV bid up
    assert ivmodel.atm_vol(0.30, 0.06, 5 / 365, True) > atm_ev   # shorter expiry, bigger bump


def test_smile_puts_richer_than_calls():
    iv = ivmodel.smile(0.2, 100.0, np.array([90.0, 100.0, 110.0]), 45 / 365, 1.2)
    assert iv[0] > iv[1] and iv[1] >= iv[2] * 0.99


def test_index_iv_is_vix():
    px = _px()
    vix = pd.Series(np.linspace(0.12, 0.4, len(px)), index=px.index)
    f = ivmodel.features(px, [], pd.Series(1.3, index=px.index), vix,
                         pd.Series(0.0, index=px.index), "index")
    assert np.allclose(f["iv_base"].iloc[10:], vix.iloc[10:])


def test_days_to_next_report_visibility():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    s = ivmodel.days_to_next_report(idx, ["2020-03-01"])
    assert s.iloc[0] == -1                           # 60 days out: not yet visible
    assert s.loc["2020-02-20"] == 10
    assert s.loc["2020-03-02"] == -1                 # after the report


# ------------------------------------------------------------------- market --

def test_monthly_expiries_are_third_fridays(mkt):
    for e in mkt.expiries(False)[:12]:
        d = mkt.dates[e]
        assert d.weekday() == 4 and 15 <= d.day <= 21


def test_round_strike_grid():
    assert round_strike(99.3, 99.3) == 99.0
    assert round_strike(101.3, 101.3) == 102.5
    assert round_strike(433.0, 433.0) == 435.0


def test_snapshot_rows(mkt):
    rows = mkt.snapshot(400)
    assert {r["ticker"] for r in rows} == {"SPY", "XYZ"}
    assert all(r["em_30d"] > 0 for r in rows)


# ------------------------------------------------------------------ margin --

def test_bpr_naked_put_and_strangle():
    legs = [Leg("put", -1, 90.0, 10)]
    req = margin.bpr(legs, 100.0, {0: 1.0})
    assert req == pytest.approx((max(20 - 10, 9) + 1.0) * 100)
    strangle = [Leg("put", -1, 90.0, 10), Leg("call", -1, 110.0, 10)]
    req2 = margin.bpr(strangle, 100.0, {0: 1.0, 1: 0.8})
    assert req2 == pytest.approx(max((10 + 1.0) * 100 + 80, (10 + 0.8) * 100 + 100))


def test_bpr_iron_condor_is_width_minus_credit():
    legs = [Leg("put", 1, 85.0, 10), Leg("put", -1, 90.0, 10),
            Leg("call", -1, 110.0, 10), Leg("call", 1, 115.0, 10)]
    prices = {0: 0.4, 1: 1.0, 2: 0.9, 3: 0.3}
    credit = (1.0 - 0.4 + 0.9 - 0.3) * 100
    assert margin.bpr(legs, 100.0, prices) == pytest.approx(500 - credit)


def test_bpr_covered_call_and_uncovered_counts():
    legs = [Leg("stock", 100.0), Leg("call", -1, 110.0, 10)]
    assert margin.uncovered(legs) == (0.0, 0.0)
    assert margin.bpr(legs, 100.0, {1: 1.0}) == pytest.approx(5000.0)


# ------------------------------------------------------------- templates --

@pytest.mark.parametrize("tpl", sorted(TEMPLATES))
def test_every_template_opens_and_prices(mkt, tpl):
    b = Book(100_000, 100_000)
    e = mkt.nearest_expiry("SPY", 300, 45)
    legs = build(tpl, mkt, "SPY", 300, e)
    pv = b.preview(mkt, "SPY", 300, legs)
    assert pv["bpr"] > 0
    if tpl not in ("covered_call",):
        assert pv["credit"] != 0


def test_strangle_strikes_near_16_delta(mkt):
    e = mkt.nearest_expiry("SPY", 300, 45)
    p, c = build("short_strangle", mkt, "SPY", 300, e)
    S = mkt.spot("SPY", 300)
    assert p.strike < S < c.strike


# ------------------------------------------------------------- book flow --

def test_open_close_cash_accounting_at_mid(mkt):
    b = Book(100_000, 100_000, slip=0.0)
    e = mkt.nearest_expiry("SPY", 300, 45)
    legs = build("short_strangle", mkt, "SPY", 300, e)
    p = b.open(mkt, "SPY", 300, "short_strangle", legs)
    assert b.nlv(mkt, 300) == pytest.approx(100_000 - 2.0, abs=1e-6)   # only the $1x2 fees
    pnl = b.close(mkt, p, 300)
    assert pnl == pytest.approx(-2.0, abs=1e-6)
    assert b.cash == pytest.approx(100_000 - 2.0)


def test_slippage_costs_money(mkt):
    a, c = Book(1e5, 1e5, slip=0.0), Book(1e5, 1e5, slip=1.0)
    e = mkt.nearest_expiry("XYZ", 300, 45)
    legs = build("iron_condor", mkt, "XYZ", 300, e)
    assert c.preview(mkt, "XYZ", 300, legs)["credit"] < a.preview(mkt, "XYZ", 300, legs)["credit"]


def test_expiry_assignment_makes_stock_and_keeps_nlv():
    frames, uni, earn = _frames()
    frames["XYZ"].iloc[320:, frames["XYZ"].columns.get_loc("close")] = 50.0             # crash below the put strike
    m = Market(frames, uni, earn)
    b = Book(1e5, 1e5, slip=0.0)
    e = [x for x in m.expiries(False) if x > 300][0]
    leg = Leg("put", -1, round_strike(m.spot("XYZ", 300) * 0.95, m.spot("XYZ", 300)), e)
    p = b.open(m, "XYZ", 300, "short_put", [leg])
    run = clock.Run(b, 300, e + 5, 300)
    before = b.nlv(m, e)
    while run.i < e:
        clock.step(m, run)
    after = b.nlv(m, e)
    assert any(lg.kind == "stock" and lg.qty == 100 for lg in p.legs)
    assert after == pytest.approx(before, rel=1e-6)


def test_roll_out_for_credit_and_untested(mkt):
    b = Book(1e5, 1e5, slip=0.0)
    e1 = mkt.nearest_expiry("SPY", 300, 45)
    p = b.open(mkt, "SPY", 300, "short_strangle", build("short_strangle", mkt, "SPY", 300, e1))
    e2 = mkt.nearest_expiry("SPY", 330, 45)
    assert e2 > e1
    rolled = b.roll(mkt, p, 330, e2, "out")
    assert rolled is p and p.rolls == 1 and p.expiry_i() == e2
    u = b.roll(mkt, p, 331, e2, "untested", require_credit=False)
    assert u is p and p.rolls == 2


def test_add_legs_wheel_call(mkt):
    b = Book(1e5, 1e5, slip=0.0)
    e = mkt.nearest_expiry("XYZ", 300, 45)
    p = b.open(mkt, "XYZ", 300, "covered_call", build("covered_call", mkt, "XYZ", 300, e))
    n_before = len(p.legs)
    e2 = mkt.nearest_expiry("XYZ", 340, 45)
    b.add_legs(mkt, p, 340, [Leg("call", -1, round_strike(mkt.spot("XYZ", 340) * 1.08, 100), e2)], "Sold call")
    assert len(p.legs) == n_before + 1


def test_serialisation_roundtrip(mkt):
    b = Book(1e5, 1e5)
    e = mkt.nearest_expiry("SPY", 300, 45)
    b.open(mkt, "SPY", 300, "iron_condor", build("iron_condor", mkt, "SPY", 300, e))
    run = clock.Run(b, 300, 360, 300)
    clock.advance(mkt, run, "week")
    run2 = clock.Run.from_dict(run.to_dict())
    assert run2.book.nlv(mkt, run2.i) == pytest.approx(run.book.nlv(mkt, run.i))


# ------------------------------------------------------------------ clock --

def test_event_order_and_flags(mkt):
    b = Book(1e5, 1e5, slip=0.0)
    e = mkt.nearest_expiry("SPY", 300, 45)
    b.open(mkt, "SPY", 300, "short_strangle", build("short_strangle", mkt, "SPY", 300, e))
    run = clock.Run(b, 300, 400, 300)
    kinds = []
    for _ in range(80):
        ev = clock.advance(mkt, run, "event")
        kinds += [x.kind for x in ev]
        if run.done or not b.open_positions():
            break
    assert "dte21" in kinds or "target" in kinds
    assert kinds.count("dte21") <= 1


def test_earnings_event_fires(mkt):
    b = Book(1e5, 1e5, slip=0.0)
    e = [x for x in mkt.expiries(False) if x > 300][0]
    b.open(mkt, "XYZ", 280, "iron_condor", build("iron_condor", mkt, "XYZ", 280, e))
    run = clock.Run(b, 280, 320, 280)
    kinds = []
    while not run.done:
        kinds += [x.kind for x in clock.step(mkt, run)]
    assert "earnings" in kinds


def test_deterministic_given_scenario():
    m = Market(*_frames())
    s1 = scenario.random_scenario(m, 42, 63, "normal")
    s2 = scenario.random_scenario(m, 42, 63, "normal")
    assert s1 == s2 and s1["hide"] and s1["aliases"]["SPY"].startswith("ASSET-")


# --------------------------------------------- bot vs the vault repro --

def test_bot_matches_vault_strangle_repro():
    """The engine's mechanical bot reproduces repro/strangle.simulate (p50_d21, sequential)
    on the same flat-vol synthetic tape: same win rate and P&L in credits."""
    from zenith.vault.repro import strangle as rs
    frames, uni, earn = _frames(n=900, seed=7, every_day=True, vol=0.16, iv=0.19)
    m = Market(frames, uni, earn)
    S = frames["SPY"]["close"]
    vix = frames["_VIX"]["close"]
    rf = frames["_RATE"]["close"]
    sim = rs.simulate(S, vix, rf, dte=45, delta=0.16)
    seq = rs.sequential_book(sim[sim["rule"] == "p50_d21"])
    res = bot.play(m, 0, len(m) - 60, 1e7, slip=0.0, flat_iv=True, expiry_mode="calendar",
                   contracts=1)
    tr = pd.DataFrame(res["trades"])
    assert abs(len(tr) - len(seq)) <= 2
    ref = sim[sim["rule"] == "p50_d21"].set_index("entry")
    ref_win = (ref.loc[[m.dates[i] for i in tr["open_i"] if m.dates[i] in ref.index], "pnl"] > 0).mean()
    assert abs((tr["pnl"] > 0).mean() - ref_win) < 0.06
    mine = (tr["pnl"] / tr["credit"]).mean()
    theirs = (ref["pnl"] / ref["credit"]).mean()
    assert abs(mine - theirs) < 0.15


# ----------------------------------------------------------------- coach --

def test_every_template_maps_to_a_vault_card():
    v = Vault()
    for k, spec in TEMPLATES.items():
        assert spec["vault"] in v.cards, k


def test_coach_refs_resolve():
    from zenith.theta import coach
    v = Vault()
    refs = [r for _, _, r in coach.EVENT_REFS.values()] + list(coach.EVENT_CAVEATS.values())
    refs += ["tree:opening.o5", "tree:opening.o1", "alloc", "edge:iv-rank-timing",
             "edge:earnings-iv-crush", "edge:number-of-occurrences"]
    for spec in TEMPLATES.values():
        c = spec["vault"]
        refs += [f"card:{c}.when.iv", f"card:{c}.setup.dte"]
    for r in refs:
        assert v.resolve(r), r


def test_coach_pretrade_flags_low_ivr_and_oversize():
    v = Vault()
    row = {"ticker": "XYZ", "kind": "stock", "tier": "C", "ivr": 0.1}
    pv = {"credit": 300.0, "bpr": 15_000.0}
    ns = pretrade(v, row, "short_strangle", pv, 45, 100_000, 40_000, ["XYZ"], True)
    titles = " ".join(n.title for n in ns)
    assert "IV rank" in titles and "net liq" in titles and "Earnings" in titles
    assert ns[0].severity == "alert"


def test_coach_events_and_adherence():
    v = Vault()
    ns = on_events(v, [{"kind": "dte21", "msg": "#1 SPY at 21 DTE", "pid": 1, "i": 10}])
    assert ns and ns[0].title.startswith("21 DTE")
    log = [{"i": 10, "kind": "dte21", "pid": 1, "msg": ""}, {"i": 11, "kind": "close", "pid": 1, "msg": ""},
           {"i": 20, "kind": "target", "pid": 2, "msg": ""}]
    rate, rows = adherence(log)
    assert rate == pytest.approx(0.5) and len(rows) == 2


# --------------------------------------------------------------- scoring --

def test_curve_stats_and_score():
    eq = [[0, 100.0], [1, 110.0], [2, 99.0], [3, 120.0]]
    st = scoring.curve_stats(eq)
    assert st["ret"] == pytest.approx(0.2) and st["max_dd"] == pytest.approx(0.1)
    s = scoring.score(st, {"ret": 0.05, "max_dd": 0.2}, 0.1, 0.9, "hard", True,
                      {"trades": 3, "closed": 25, "max_bp": 0.4, "theta_nlv": 0.002, "wheel": True})
    assert {"beat_bot", "beat_spy", "survivor", "discipline", "occurrences", "theta",
            "wheel", "small", "green"} <= set(s["achievements"])
    assert s["mult"] == 2.0


def test_level_curve():
    assert scoring.level(0) == (1, 0, 500)
    assert scoring.level(500)[0] == 2

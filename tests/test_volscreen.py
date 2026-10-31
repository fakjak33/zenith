"""VOL SCREENER — offline: fixtures are trimmed real Cboe chains (2026-10-30)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zenith.theta.engine import pricing
from zenith.volscreen import (checklist, cboe, clusters, garch, history, iv, liquidity, realized, surface,
                              vrp)

FIX = Path(__file__).parent / "fixtures" / "volscreen"


def _chain(t):
    js = json.loads((FIX / f"{t}.json").read_text(encoding="utf-8"))
    u = cboe.underlying(js)
    px = iv.price_chain(cboe.chain_frame(js, t), u["spot"], cboe.quote_time(js), 0.04)
    return js, u, px


# ------------------------------------------------------------------ parsing --

def test_parse_occ():
    assert cboe.parse_occ("SPXW261116C06000000") == ("SPXW", pd.Timestamp("2026-11-16").date(), "C", 6000.0)
    assert cboe.parse_occ("AAPL270115P00180000")[1:] == (pd.Timestamp("2027-01-15").date(), "P", 180.0)


def test_chain_frame_roots_and_symbols():
    js = json.loads((FIX / "SPX.json").read_text(encoding="utf-8"))
    df = cboe.chain_frame(js, "SPX")
    assert set(df["root"]) <= {"SPX", "SPXW"} and len(df) > 100
    js2 = {"data": {"options": [{"option": "AAPL1270115P00180000", "bid": 1, "ask": 2, "iv": 0.3,
                                 "open_interest": 1, "volume": 1, "delta": -0.1}]}}
    assert cboe.chain_frame(js2, "AAPL").empty                     # adjusted root dropped
    assert cboe.cboe_symbol("SPX") == "_SPX" and cboe.cboe_symbol("AAPL") == "AAPL"


# ------------------------------------------------------------------ IV / surface --

def test_implied_vol_round_trip():
    rng = np.random.default_rng(3)
    n = 400
    call = rng.random(n) > 0.5
    F = np.full(n, 100.0)
    K = rng.uniform(55, 160, n)
    T = rng.uniform(0.01, 2.0, n)
    sig = rng.uniform(0.08, 1.2, n)
    r = 0.04
    px = pricing.price(call, F * np.exp(-r * T), K, T, sig, r)
    got = iv.implied_vol(call, F, K, T, px, r)
    ok = np.isfinite(got)
    assert ok.mean() > 0.95
    assert np.nanmax(np.abs(got[ok] - sig[ok])) < 1e-4


def test_implied_forward_recovers_carry():
    S, T, r, q = 100.0, 0.5, 0.04, 0.02
    F = S * math.exp((r - q) * T)
    ks = np.arange(80, 121, 5.0)
    rows = []
    for k in ks:
        for cp in ("C", "P"):
            p = float(pricing.price(cp == "C", F * math.exp(-r * T), k, T, 0.25, r))
            rows.append({"strike": k, "cp": cp, "mid": p, "quoted": True})
    f = iv.implied_forward(pd.DataFrame(rows), S, T, r)
    assert abs(f - F) < 1e-3


def test_fixture_term_matches_cboe_iv30():
    for t in ("SPY", "SPX", "AAPL"):
        _, u, px = _chain(t)
        tm, _ = surface.term(surface.slices(px))
        assert abs(tm[30] - u["iv30_cboe"]) < 0.015, (t, tm[30], u["iv30_cboe"])


def test_constant_maturity_total_variance():
    v, ext = surface.cm_interp([0.1, 0.5], [0.2, 0.2], 0.3)
    assert abs(v - 0.2) < 1e-12 and not ext                       # flat vol stays flat
    v, _ = surface.cm_interp([0.1, 0.3], [0.30, 0.20], 0.2)
    w = 0.30 ** 2 * 0.1 + (0.20 ** 2 * 0.3 - 0.30 ** 2 * 0.1) * 0.5
    assert abs(v - math.sqrt(w / 0.2)) < 1e-12
    assert surface.cm_interp([0.1, 0.3], [0.2, 0.2], 0.05)[1]     # extrapolation flagged
    sl = [{"T": 0.1, "atm": 0.30}, {"T": 0.2, "atm": 0.15}]
    assert surface.calendar_violations(sl) == 1


def test_event_variance_recovers_move():
    sig, e = 0.25, 0.05
    sl = []
    for i, d in enumerate((10, 38, 66)):
        T = d / 365
        w = sig ** 2 * T + (e ** 2 if d > 20 else 0)
        sl.append({"expiry": (pd.Timestamp("2026-11-01") + pd.Timedelta(days=d)).date().isoformat(), "dte": d,
                   "T": T, "atm": math.sqrt(w / T)})
    ev = surface.event_variance(sl, pd.Timestamp("2026-11-20").date())
    assert abs(math.sqrt(ev["e2"]) - e) < 1e-6 and ev["method"] == "pre-event expiry"
    ev2 = surface.event_variance(sl[1:], pd.Timestamp("2026-11-20").date())
    assert abs(math.sqrt(ev2["e2"]) - e) < 1e-6 and ev2["method"] == "forward variance"
    ex = surface.ex_event_term({38: sl[1]["atm"], 10: sl[0]["atm"]}, ev, 19 / 365)
    assert abs(ex[38] - sig) < 1e-6 and ex[10] == sl[0]["atm"]      # pre-event tenor untouched


# ------------------------------------------------------------------ realized vol --

def _gbm_ohlc(n=2500, sig=0.30, seed=0, steps=390):
    rng = np.random.default_rng(seed)
    dt = 1 / 252 / steps
    rows, p = [], 100.0
    for _ in range(n):
        path = p * np.exp(np.cumsum(-0.5 * sig ** 2 * dt + sig * math.sqrt(dt) * rng.standard_normal(steps)))
        rows.append((p, max(p, path.max()), min(p, path.min()), path[-1], 1e6))
        p = path[-1]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.bdate_range("2000-01-03", periods=n))


def test_range_estimators_near_true_vol():
    df = _gbm_ohlc()
    n = len(df) - 1
    for f in (realized.parkinson, realized.garman_klass, realized.rogers_satchell, realized.yang_zhang,
              realized.close_to_close):
        v = realized.last(f(df, n))
        # discrete sampling biases range estimators slightly low; 8% is generous but real
        assert abs(v / 0.30 - 1) < 0.08, (f.__name__, v)


def test_yang_zhang_hand_value():
    df = pd.DataFrame({"open": [100, 101, 99, 102, 103, 101], "high": [102, 103, 101, 104, 105, 103],
                       "low": [99, 100, 98, 101, 102, 100], "close": [101, 100, 101, 103, 102, 102],
                       "volume": [1] * 6}, dtype=float)
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    n = 5
    on = np.log(o / c.shift(1)).iloc[1:]
    oc = np.log(c / o).iloc[1:]
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).iloc[1:]
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    want = math.sqrt((on.var(ddof=1) + k * oc.var(ddof=1) + (1 - k) * rs.mean()) * 252)
    got = realized.yang_zhang(df, n).iloc[-1]
    assert abs(got - want) < 1e-12


# ------------------------------------------------------------------ GARCH --

def test_nelder_mead_rosenbrock():
    x, fx, _ = garch.nelder_mead(lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, [-1.2, 1.0],
                                 maxiter=3000)
    assert np.allclose(x, [1, 1], atol=1e-4) and fx < 1e-8


def test_garch_recovers_parameters():
    r = garch.simulate(4000, 0.02, 0.08, 0.90, seed=1) / 100
    f = garch.fit(r, "garch", n_max=4000)
    assert f.ok and abs(f.alpha - 0.08) < 0.03 and abs(f.beta - 0.90) < 0.03 and abs(f.rho - 0.98) < 0.02


def test_gjr_recovers_asymmetry():
    r = garch.simulate(4000, 0.02, 0.03, 0.88, gamma=0.12, seed=2) / 100
    f = garch.fit(r, "gjr", n_max=4000)
    assert f.ok and f.gamma > 0.06 and abs(f.beta - 0.88) < 0.04


def test_term_vol_mean_reverts():
    r = garch.simulate(3000, 0.02, 0.08, 0.90, seed=4) / 100
    f = garch.fit(r, "garch", n_max=3000)
    lr = math.sqrt(f.var_lr * 252) / 100
    assert abs(garch.term_vol(f, 5000) - lr) < 0.01 * lr + 1e-4
    hi = garch.GarchFit("garch", f.omega, f.alpha, 0.0, f.beta, f.var_lr, f.var_lr * 4, 0, 1, True, [])
    assert garch.term_vol(hi, 1) > garch.term_vol(hi, 21) > garch.term_vol(hi, 252) > lr


# ------------------------------------------------------------------ VRP / state --

def test_vrp_snapshot_and_slope():
    s = vrp.snapshot({30: 0.25, 90: 0.24}, {"yz20": 0.20, "cc252": 0.22, "har30": 0.21}, {30: 0.20, 90: 0.21})
    assert abs(s["vrp30"] - 0.05) < 1e-12 and abs(s["iv_rv"] - 1.25) < 1e-12
    assert abs(vrp.ols_slope([1, 2, 3, 4, 5]) - 1.0) < 1e-12
    rank, pct, n = vrp.iv_rank(pd.Series(np.r_[np.linspace(0.1, 0.3, 100), 0.2]))
    assert abs(rank - 0.5) < 1e-9 and n == 101


@pytest.mark.parametrize("series,want", [
    ([0.03] * 30, "ALL-CLEAR"),                                   # steady premium
    ([-0.01] * 30, "CONFIRMED"),                                  # realized above implied
    ([0.04 + 0.0005 * np.sin(i) for i in range(60)] + [0.035, 0.028, 0.020, 0.012, 0.006], "ALARM"),
])
def test_state_machine_cases(series, want):
    assert checklist.vrp_states(series)[-1] == want


def test_state_machine_recovers_then_clears():
    base = [0.04 + 0.0005 * np.sin(i) for i in range(60)]
    crash = [0.02, 0.0, -0.02, -0.03]
    heal = [-0.025, -0.01, 0.005, 0.015, 0.02, 0.025, 0.03, 0.035]
    st = checklist.vrp_states(base + crash + heal)
    assert "CONFIRMED" in st and "RECOVERING" in st and st[-1] == "ALL-CLEAR"
    assert st.index("RECOVERING") > st.index("CONFIRMED")


def test_checklist_score_bounds():
    good = {"stars": 4, "vrp30": 0.05, "iv_rv": 1.3, "state": "ALL-CLEAR", "ivr": 0.6, "ivp": 0.7,
            "ratio_30_90": 0.9, "kink_explained": False, "yz10": 0.15, "yz60": 0.2, "earn_days": None,
            "skew_pct": 0.8, "vrp_t": {"30": 0.04, "60": 0.03}}
    bad = {"stars": 1, "vrp30": -0.05, "iv_rv": 0.8, "state": "CONFIRMED", "ivr": 0.05, "ivp": 0.05,
           "ratio_30_90": 1.2, "kink_explained": False, "yz10": 0.4, "yz60": 0.2, "earn_days": 3,
           "skew_pct": 0.1, "vrp_t": {"30": -0.04, "60": -0.03}}
    g, b = checklist.evaluate(good), checklist.evaluate(bad)
    assert g["sell"]["score"] == 100.0 and g["sell"]["grade"] == "A"
    assert b["sell"]["score"] == 0.0 and b["sell"]["grade"] == "F"
    assert b["buy"]["score"] > g["buy"]["score"]
    for r in (g, b):
        for m in ("sell", "buy"):
            assert 0 <= r[m]["score"] <= 100


# ------------------------------------------------------------------ liquidity --

def test_liquidity_from_fixture_and_classes():
    _, u, px = _chain("SPY")
    m = liquidity.metrics(px, u["spot"], None)
    assert m["opt_vol"] > 0 and m["atm_spread"] < 0.05 and m["strikes"] > 10
    m["stars"] = liquidity.stars(m)
    assert m["stars"] >= 3
    thin = {"opt_vol": 50, "oi": 300, "atm_spread": 0.3}
    assert liquidity.stars(thin) == 1 and liquidity.classify(thin) == "THIN"
    steady = liquidity.with_history({**m}, [m["opt_vol"]] * 20)
    steady["stars"] = 4
    assert liquidity.classify(steady) == "CORE"
    surge = liquidity.with_history({**m, "opt_vol": m["opt_vol"] * 5}, [m["opt_vol"]] * 20)
    surge["stars"] = 4
    assert liquidity.classify(surge) == "SURGE"


# ------------------------------------------------------------------ clusters --

def test_clusters_recover_blocks():
    rng = np.random.default_rng(0)
    T = 250
    f = rng.standard_normal((T, 3))
    closes = {}
    for b in range(3):
        for i in range(8):
            r = 0.01 * (0.84 * f[:, b] + 0.55 * rng.standard_normal(T))
            closes[f"B{b}_{i}"] = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2025-01-01", periods=T))
    scores = {t: rng.random() for t in closes}
    out = clusters.build(closes, scores, set(closes), window=250, k=3)
    groups = sorted(sorted(c["members"]) for c in out["clusters"])
    assert groups == sorted(sorted(f"B{b}_{i}" for i in range(8)) for b in range(3))
    assert len(out["basket"]) == 3 and len({t[:2] for t in out["basket"]}) == 3
    C = np.array(out["corr"])
    assert np.linalg.eigvalsh(C).min() > -1e-6                    # shrunk matrix stays PSD
    assert 2.5 < out["eff_universe"] < 8


# ------------------------------------------------------------------ history --

def test_history_append_and_guard(tmp_path):
    h = history.empty_term()
    row = pd.DataFrame([{"ticker": "SPY", "date": pd.Timestamp("2026-10-30"), "src": 0, "iv_30": 0.15}])
    seed = pd.DataFrame([{"ticker": "SPY", "date": pd.Timestamp("2026-10-30"), "src": 1, "iv_30": 0.99},
                         {"ticker": "SPY", "date": pd.Timestamp("2026-10-29"), "src": 1, "iv_30": 0.14}])
    a = history.append(history.append(h, seed), row)
    assert len(a) == 2 and float(a.loc[a["date"] == "2026-10-30", "iv_30"].iloc[0]) == 0.15
    assert len(history.append(a, row)) == 2                        # idempotent
    p = tmp_path / "t.parquet"
    history.write(a, p, 0, history.TERM_COLS)
    back = history.read_ticker(p, "SPY")
    assert len(back) == 2
    with pytest.raises(RuntimeError):
        history.write(a.iloc[:1], p, 2, history.TERM_COLS)


# ------------------------------------------------------------------ end to end --

def test_offline_compute_writes_clean_json(tmp_path, monkeypatch):
    from zenith import config
    from zenith.volscreen import compute
    for k in ("universe", "screen", "surfaces", "baskets", "garch", "status", "earnings"):
        monkeypatch.setitem(config.VOLSCREEN_FILES, k, tmp_path / f"{k}.json")
    st = compute.run("force", tickers=["SPY", "SPX", "AAPL", "KSS"], offline=str(FIX), write_history=False)
    assert st["n_ok"] == 4 and not st["failures"]
    sc = json.loads((tmp_path / "screen.json").read_text(encoding="utf-8"))
    text = (tmp_path / "screen.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    rows = {r["ticker"]: r for r in sc["rows"]}
    assert set(rows) == {"SPY", "SPX", "AAPL", "KSS"}
    for r in rows.values():
        assert 0.03 < r["iv"]["30"] < 2.0
        assert r["check"]["sell"]["grade"] in "ABCDF"
        assert r["liq"]["stars"] in (1, 2, 3, 4)
    surf = json.loads((tmp_path / "surfaces.json").read_text(encoding="utf-8"))
    assert len(surf["surf"]["SPY"]["delta"]) == 7 and len(surf["surf"]["SPY"]["delta"][0]) == 9

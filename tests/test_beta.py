"""CLEAN BETA tests — fully offline, seeded synthetic returns, no network.

Conventions mirror tests/test_trend.py: a fixture redirects BETA_FILES (in
place, never config itself) and every network touch in compute.py is a
monkeypatchable seam (_fetch_prices, _load_universe, _fundamentals,
_earnings, _optionable, _chains).
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from zenith import config
from zenith.beta import basket as bk
from zenith.beta import compute as bc
from zenith.beta import estimators as est
from zenith.beta import hedge as hg
from zenith.beta import quality as q
from zenith.beta import screen as sc


# ------------------------------------------------------------------ helpers --
def _idx(n: int, end: str = "2026-09-30") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _market(n=500, seed=0, vol=0.01):
    return np.random.default_rng(seed).normal(0.0003, vol, n)


def _stock(rm, beta, idio, seed):
    return beta * rm + np.random.default_rng(seed).normal(0, idio, len(rm))


def _frame(rets: np.ndarray, idx, start=100.0, volume=2e6) -> pd.DataFrame:
    close = start * np.exp(np.cumsum(np.r_[0.0, rets[1:]]))
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)


def _welch_reference(ri, rm, delta=3.0, lam=2.0):
    """Line-by-line port of Welch's procedure (loops, no vectorization):
    1. for each day, winsorize r_i into [min(lo,hi), max(lo,hi)] with
       lo = (1-δ)·r_m and hi = (1+δ)·r_m;
    2. weight_t = exp(-λ · (T-1-t)/252);
    3. weighted least squares slope with intercept."""
    n = len(ri)
    y, w = [], []
    for t in range(n):
        lo, hi = (1 - delta) * rm[t], (1 + delta) * rm[t]
        a, b = min(lo, hi), max(lo, hi)
        y.append(min(max(ri[t], a), b))
        w.append(math.exp(-lam * (n - 1 - t) / 252.0))
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, rm)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    num = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, rm, y))
    den = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, rm))
    return num / den


@pytest.fixture
def tmp_beta_store(tmp_path, monkeypatch):
    for k, p in list(config.BETA_FILES.items()):
        monkeypatch.setitem(config.BETA_FILES, k, tmp_path / p.name)
    monkeypatch.setattr(bc, "BETA_EXPORT_DIR", tmp_path / "exports")
    return tmp_path


# --------------------------------------------------------------- estimators --
@pytest.mark.parametrize("beta,idio,seed", [(0.6, 0.012, 1), (1.0, 0.01, 2), (1.6, 0.008, 3),
                                            (2.2, 0.02, 4), (-0.3, 0.015, 5)])
def test_bswa_matches_welch_reference(beta, idio, seed):
    rm = _market(seed=seed + 100)
    ri = _stock(rm, beta, idio, seed)
    assert est.bswa(ri, rm) == pytest.approx(_welch_reference(ri, rm), abs=1e-9)


def test_bswa_recovers_beta_on_clean_low_noise_data():
    rm = _market(n=750)
    ri = _stock(rm, 1.4, 0.002, 7)
    assert est.bswa(ri, rm) == pytest.approx(1.4, abs=0.05)


def test_bswa_resists_a_one_day_event_ols_does_not():
    rm = _market(n=500, seed=11)
    ri = _stock(rm, 1.2, 0.006, 12)
    k = 450
    rm[k] = 0.012                                   # an ordinary up day...
    shocked = ri.copy()
    shocked[k] += 0.40                              # ...with a +40% takeover pop
    w = slice(-252, None)                           # OLS on the screen's own 252-day window
    d_ols = abs(est.ols_stats(shocked[w], rm[w])["beta"] - est.ols_stats(ri[w], rm[w])["beta"])
    d_bswa = abs(est.bswa(shocked, rm) - est.bswa(ri, rm))
    assert d_ols > 0.1
    assert d_bswa < d_ols / 4


def test_winsorization_band_bounds_implied_daily_beta():
    rm = np.array([0.01, -0.02, 0.005, 0.0, 0.03] * 10)
    ri = np.array([0.20, 0.20, -0.30, 0.05, 0.01] * 10)
    a, b = (1 - 3) * rm, (1 + 3) * rm
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    clipped = np.clip(ri, lo, hi)
    nz = rm != 0
    assert np.all(clipped[nz] / rm[nz] <= 4 + 1e-12) and np.all(clipped[nz] / rm[nz] >= -2 - 1e-12)
    assert np.all(clipped[~nz] == 0.0)              # zero-market days carry no slope information


def test_decay_weights_halve_in_about_four_months():
    # exp(-2 · age) = 0.5 at age = ln2/2 years ≈ 87 trading days ≈ 4 months
    assert math.log(2) / 2 * 252 == pytest.approx(87.3, abs=0.1)
    # an old regime matters far less than a recent one
    n = 504
    rm = _market(n=n, seed=21)
    ri = np.r_[_stock(rm[:252], 0.5, 0.004, 22), _stock(rm[252:], 1.5, 0.004, 23)]
    assert est.bswa(ri, rm) > 1.2
    assert est.ols_stats(ri, rm)["beta"] == pytest.approx(1.0, abs=0.1)


def test_ols_stats_rho_r2_ivol_and_jumps():
    rm = _market(n=252, seed=31)
    ri = _stock(rm, 1.0, 0.01, 32)
    s = est.ols_stats(ri, rm)
    assert s["rho"] == pytest.approx(np.corrcoef(rm, ri)[0, 1], abs=1e-12)
    assert s["r2"] == pytest.approx(s["rho"] ** 2)
    assert s["ivol"] == pytest.approx(0.01 * math.sqrt(252), rel=0.12)
    assert s["jumps"] <= 1
    ri2 = ri.copy()
    ri2[[50, 120, 200]] += 0.15
    assert est.ols_stats(ri2, rm)["jumps"] >= 3


def test_effective_bets_extremes():
    n = 10
    w = np.full(n, 1 / n)
    assert est.effective_bets(w, np.ones((n, n))) == pytest.approx(1.0, abs=1e-6)
    assert est.effective_bets(w, np.eye(n)) == pytest.approx(n, abs=1e-6)
    assert est.effective_dimension(np.ones((n, n))) == pytest.approx(1.0, abs=1e-6)
    assert est.effective_dimension(np.eye(n)) == pytest.approx(n, abs=1e-6)


def test_vasicek_shrinks_toward_mean():
    betas = {"A": 2.0, "B": 1.0, "C": 0.5, "D": 1.5}
    out = est.vasicek(betas, {"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.0})
    mu = np.mean(list(betas.values()))
    assert abs(out["A"] - mu) < abs(2.0 - mu) and out["B"] == pytest.approx(1.0)


# ------------------------------------------------------------------ quality --
def test_quality_components_direction_and_imputation():
    good = {"returnOnAssets": 0.15, "grossMargins": 0.6, "freeCashflow": 20, "totalRevenue": 100,
            "profitMargins": 0.2, "operatingCashflow": 30, "debtToEquity": 20}
    bad = {"returnOnAssets": -0.05, "grossMargins": 0.1, "freeCashflow": -10, "totalRevenue": 100,
           "profitMargins": 0.1, "operatingCashflow": 2, "debtToEquity": 400}
    raw = {"G": q.raw_components(good, 0.3, None), "B": q.raw_components(bad, -0.2, None),
           "M": q.raw_components(None, None, None)}
    s = q.scores(raw)
    assert s["G"]["quality"] > s["B"]["quality"]
    assert s["M"]["quality_imputed"] and s["M"]["quality"] == 50.0


def test_issuance_needs_twelve_months_of_own_snapshots():
    doc = {}
    q.snapshot_shares(doc, {"X": {"sharesOutstanding": 100.0}}, date(2025, 10, 1))
    assert q.issuance(doc, "X", date(2026, 9, 1)) is None
    q.snapshot_shares(doc, {"X": {"sharesOutstanding": 110.0}}, date(2026, 10, 2))
    assert q.issuance(doc, "X", date(2026, 10, 2)) == pytest.approx(math.log(1.1))


# ------------------------------------------------------------------- screen --
def _universe(n=60, seed=0):
    idx = _idx(500)
    rm = _market(n=500, seed=seed)
    px = {"SPY": _frame(rm, idx)}
    members = []
    rng = np.random.default_rng(seed + 1)
    sectors = ["Tech", "Fin", "Energy", "Health", "Ind", "Cons"]
    for i in range(n):
        t = f"T{i:02d}"
        beta = 0.4 + 2.0 * i / n
        idio = 0.004 + 0.02 * rng.random()
        px[t] = _frame(_stock(rm, beta, idio, 1000 + i), idx)
        members.append({"ticker": t, "name": t, "sector": sectors[i % 6],
                        "mktcap": [5e9, 20e9, 80e9, 300e9][i % 4]})
    return members, px


def _rows(members, px, incumbents=frozenset(), earn=None, fund=None):
    bench = px["SPY"]["close"].pct_change().dropna()
    return sc.build_rows(members, px, bench, fund or {}, earn or {}, {}, set(incumbents), date(2026, 9, 30))


def test_screen_flags_and_score():
    members, px = _universe()
    rows = _rows(members, px)
    liq = [x for x in rows if x["f_liquid"]]
    assert len(liq) == 60
    hi = [x for x in liq if x["f_beta"]]
    assert 10 <= len(hi) <= 14                       # top quintile of 60
    for x in liq:
        assert 0 <= x["score"] <= 100
        assert x["passes"] == all(x[f] for f in sc.FILTERS)
        if not x["passes"]:
            assert x["fail_reasons"]
    ranks = [x["rank"] for x in rows if x["rank"]]
    assert ranks == sorted(ranks)


def test_screen_liquidity_and_event_filters():
    members, px = _universe()
    members[0]["mktcap"] = 1e9
    px["T01"] = px["T01"].assign(volume=10.0)
    rows = {x["ticker"]: x for x in _rows(members, px, earn={"T59": "2026-10-05"})}
    assert not rows["T00"]["f_liquid"] and "mkt cap" in rows["T00"]["excluded_reason"]
    assert not rows["T01"]["f_liquid"] and "ADV" in rows["T01"]["excluded_reason"]
    assert not rows["T59"]["f_events"] and "earnings" in rows["T59"]["fail_reasons"]


def test_buffer_band_keeps_incumbent_between_60th_and_80th():
    members, px = _universe()
    rows = _rows(members, px)
    mid = next(x for x in rows if x.get("beta_pct") is not None and 62 <= x["beta_pct"] < 78)
    assert not mid["f_beta"]
    rows2 = {x["ticker"]: x for x in _rows(members, px, incumbents={mid["ticker"]})}
    assert rows2[mid["ticker"]]["f_beta"]
    low = next(x for x in rows if x.get("beta_pct") is not None and x["beta_pct"] < 55)
    rows3 = {x["ticker"]: x for x in _rows(members, px, incumbents={low["ticker"]})}
    assert not rows3[low["ticker"]]["f_beta"]


# ------------------------------------------------------------------- basket --
def _cands(n=80, sectors=6, seed=3):
    idx = _idx(260)
    rm = _market(n=260, seed=seed)
    rows, cols = [], {}
    for i in range(n):
        t = f"C{i:02d}"
        cols[t] = _stock(rm, 1.5, 0.01, 500 + i)
        rows.append({"ticker": t, "sector": f"S{i % sectors}", "size": ["Mega", "Large", "Mid", "Small"][i % 4],
                     "passes": True, "score": 100 - i * 0.5, "bswa": 1.5, "rho": 0.7, "ivol": 0.2,
                     "incumbent": False})
    return rows, pd.DataFrame(cols, index=idx)


def test_basket_respects_caps_and_size():
    rows, rets = _cands()
    b = bk.select(rows, rets)
    assert config.BETA_BASKET_MIN <= b["n"] <= config.BETA_BASKET_MAX
    cap = math.floor(config.BETA_SECTOR_CAP * b["target"] + 1e-9)
    per_sector = pd.Series([m["sector"] for m in b["members"]]).value_counts()
    assert per_sector.max() <= cap
    assert all(m["weight"] <= config.BETA_NAME_CAP + 1e-12 for m in b["members"])
    assert b["beta"] == pytest.approx(1.5 * b["invested"], abs=1e-3)
    assert 1.0 <= b["enb"] <= b["n"] and 1.0 <= b["eff_dim"] <= b["n"]


def test_basket_seats_passing_incumbents_first_and_reports_short():
    rows, rets = _cands()
    rows[-1]["incumbent"] = True
    b = bk.select(rows, rets)
    assert rows[-1]["ticker"] in {m["ticker"] for m in b["members"]}
    few = [dict(x, passes=i < 12) for i, x in enumerate(rows)]
    b2 = bk.select(few, rets)
    assert b2["short"] and b2["n"] <= 12 and b2["invested"] < 1.0


def test_short_basket_still_respects_sector_weight_cap():
    rows, rets = _cands(sectors=2)
    few = [dict(x, passes=i < 20) for i, x in enumerate(rows)]       # 20 names, 2 sectors
    b = bk.select(few, rets)
    assert all(w <= config.BETA_SECTOR_CAP + 1e-9 for w in b["sectors"].values())


# -------------------------------------------------------------------- hedge --
def test_trend_gate_states():
    up = pd.Series(np.exp(np.linspace(0, 0.5, 300)), index=_idx(300))
    dn = pd.Series(np.exp(np.linspace(0.5, 0, 300)), index=_idx(300))
    assert hg.trend_gate(up)["state"] == "ON"
    assert hg.trend_gate(dn)["state"] == "REDUCED"
    assert hg.trend_gate(dn)["exposure"] == config.BETA_TREND_EXPOSURE["REDUCED"]


def test_vol_target_scale():
    r = pd.Series(np.random.default_rng(4).normal(0, 0.02, 100))     # ~32% annualized
    vt = hg.vol_target(r)
    assert vt["scale"] == pytest.approx(config.BETA_VOL_TARGET / vt["forecast"], abs=1e-3)
    calm = hg.vol_target(pd.Series(np.random.default_rng(5).normal(0, 0.003, 100)))
    assert calm["scale"] == config.BETA_VOL_SCALE_CAP


def test_vrp_math_and_budget_direction():
    idx = _idx(400)
    spy = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(6).normal(0, 0.01, 400))), index=idx)
    fc = hg.har_forecast(spy.pct_change())
    vix_hi = pd.Series(np.linspace(15, 15, 400), index=idx)
    vix_hi.iloc[-1] = 40
    v = hg.vrp(vix_hi, spy)
    assert v["vrp"] == pytest.approx(0.40 - fc.iloc[-1], abs=1e-4)
    assert v["vrp_pct"] > 95 and v["vix_pct"] > 95 and v["harvest_window"]
    assert v["put_budget"] == pytest.approx(config.BETA_PUT_BUDGET[0], abs=1e-3)   # expensive -> spend less


def test_put_ladder_from_synthetic_chain():
    spot = 600.0

    def chain(dte):
        T = dte / 365
        ks = np.arange(400, 620, 5.0)
        iv = 0.18 + (spot - ks) / spot * 0.4
        prices = []
        for k, s in zip(ks, iv):
            d1 = (math.log(spot / k) + (0.04 + s * s / 2) * T) / (s * math.sqrt(T))
            d2 = d1 - s * math.sqrt(T)
            nd = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))           # noqa: E731
            prices.append(k * math.exp(-0.04 * T) * nd(-d2) - spot * nd(-d1))
        p = np.array(prices)
        return pd.DataFrame({"strike": ks, "impliedVolatility": iv, "bid": p * 0.98, "ask": p * 1.02,
                             "lastPrice": p})
    chains = [{"expiry": f"2026-{m:02d}-15", "dte": d, "puts": chain(d)}
              for m, d in ((12, 70), (1, 100), (2, 130), (3, 160))]
    lad = hg.put_ladder(spot, chains, beta=1.6, budget=0.01)
    assert len(lad["rungs"]) == 3
    for x in lad["rungs"]:
        assert x["long_strike"] > x["short_strike"]
        assert x["long_delta"] == pytest.approx(-0.25, abs=0.04)
        assert x["short_delta"] == pytest.approx(-0.10, abs=0.04)
        assert x["max_multiple"] > 1
    assert 0 < lad["coverage"] <= 1


# ---------------------------------------------------------- compute (offline) --
def test_compute_end_to_end_offline(tmp_beta_store, monkeypatch):
    members, px = _universe(n=120, seed=9)
    del px["T05"]                                    # a failed download must not be fatal
    hedge_px = {"SPY": px["SPY"], "^VIX": pd.DataFrame({"close": 18.0}, index=px["SPY"].index),
                "BTAL": px["T00"], "DBMF": px["T01"], "KMLM": px["T02"]}

    def fake_fetch(tickers, period, status, label):
        src = {**px, **hedge_px}
        got = {t: src[t] for t in tickers if t in src}
        status.append({"segment": label, "ok": True, "n": len(got), "requested": len(tickers)})
        return got
    monkeypatch.setattr(bc, "_fetch_prices", fake_fetch)
    monkeypatch.setattr(bc, "_load_universe", lambda: (members, {"source": "test"}))
    monkeypatch.setattr(bc, "_fundamentals", lambda t: {})
    monkeypatch.setattr(bc, "_earnings", lambda u: {})
    monkeypatch.setattr(bc, "_optionable", lambda t, d: {})
    monkeypatch.setattr(bc, "_chains", lambda t, d: (None, []))
    out = bc.run(action="rebalance", today=date(2026, 10, 1))
    assert out["ok"]
    latest = json.loads(config.BETA_FILES["latest"].read_text(encoding="utf-8"))
    assert latest["n"] == 120 and latest["n_priced"] == 119
    t05 = next(x for x in latest["rows"] if x["ticker"] == "T05")
    assert t05["f_liquid"] is False and t05["excluded_reason"] == "no price data"
    basket = json.loads(config.BETA_FILES["basket"].read_text(encoding="utf-8"))
    assert basket["quarter"] == "2026-10" and basket["n"] > 0
    hedge = json.loads(config.BETA_FILES["hedge"].read_text(encoding="utf-8"))
    assert hedge["trend"]["state"] in ("ON", "PARTIAL", "REDUCED")
    assert hedge["puts"]["rungs"] == []
    assert (tmp_beta_store / "exports" / f"{latest['as_of']}.csv").exists()
    assert "NaN" not in config.BETA_FILES["latest"].read_text(encoding="utf-8")


def test_scheduling_keys():
    assert bc.quarter_key(date(2026, 10, 2)) == "2026-10"
    assert bc.quarter_key(date(2026, 12, 15)) == "2026-10"
    assert bc.quarter_key(date(2027, 2, 1)) == "2027-01"
    assert bc.needs_screen({"computed": "2026-09-30"}, date(2026, 10, 1))
    assert not bc.needs_screen({"computed": "2026-10-01"}, date(2026, 10, 20))
    assert bc.needs_rebalance({"quarter": "2026-07"}, date(2026, 10, 1))
    assert not bc.needs_rebalance({"quarter": "2026-10"}, date(2026, 11, 1))

"""OPTIONS VAULT Phase 04 reproduction tests — offline, synthetic series only."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from zenith.beta.hedge import bs_put_delta
from zenith.vault import repro
from zenith.vault.repro import cboe, earnings, expected_move, strangle, vix_spikes, vrp

SCHEMA = {"claim", "tasty_value", "tasty_sources", "tasty_note", "our_value", "data", "period",
          "method", "caveats", "verdict", "chart_data", "claims"}


# --- synthetic data helpers --------------------------------------------------

def _bdays(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _gbm(n: int, vol: float, seed: int = 1, start: str = "2000-01-03", drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift / 252, vol / math.sqrt(252), n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=_bdays(n, start))


def _check_schema(obj: dict) -> None:
    assert SCHEMA <= set(obj)
    assert obj["tasty_value"] is None and obj["tasty_note"] == "pending Phase 02"
    assert obj["verdict"] in ("agrees", "mixed", "contradicts")
    for c in obj["claims"]:
        assert c["tasty_value"] is None and c["verdict"] in ("agrees", "mixed", "contradicts")


# --- shared helpers --------------------------------------------------------------

def test_verdict_share_and_rnd():
    assert repro.verdict_share(0.52, 0.6) == "agrees"
    assert repro.verdict_share(0.40, 0.48) == "contradicts"
    assert repro.verdict_share(0.47, 0.53) == "mixed"
    assert repro.rnd(float("nan")) is None and repro.rnd(1.23456, 2) == 1.23


# --- vrp ---------------------------------------------------------------------------

def test_forward_realized_matches_true_vol():
    s = _gbm(3000, 0.20)
    rv = vrp.forward_realized(s).dropna()
    assert abs(rv.mean() - 0.20) < 0.01
    assert vrp.forward_realized(s).iloc[-21:].isna().all()


def test_iv_rank_bounds():
    iv = pd.Series(np.r_[np.linspace(0.1, 0.3, 300), [0.2]], index=_bdays(301))
    ivr = vrp.iv_rank(iv).dropna()
    assert ivr.between(0, 100).all() and ivr.iloc[-2] == pytest.approx(100.0)


def test_vrp_compute_positive_premium():
    s = _gbm(2000, 0.15, seed=3)
    vix = pd.Series(0.20, index=s.index) + np.random.default_rng(4).normal(0, 0.01, len(s))
    res = vrp.compute(vix, s)
    assert res["overall"]["share_iv_gt_rv"] > 0.9
    assert 3.0 < res["overall"]["spread"] < 7.0
    out = vrp.build(res)
    _check_schema(out)
    assert out["claims"][0]["verdict"] == "agrees"


# --- cboe --------------------------------------------------------------------------

def test_cagr_and_drawdown():
    idx = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"])
    s = pd.Series([100.0, 50.0, 200.0], index=idx)
    assert cboe.cagr(s) == pytest.approx(1.0, rel=1e-2)
    assert cboe.max_dd(s) == pytest.approx(-0.5)


def test_window_stats_beta_and_sharpe_sign():
    b = _gbm(1500, 0.16, seed=5, drift=0.08)
    lev = (1 + 2 * b.pct_change().fillna(0)).cumprod() * 100
    rf = pd.Series(0.0, index=b.index)
    st = cboe.window_stats(lev, b, rf)
    assert st["beta"] == pytest.approx(2.0, abs=0.05)
    assert st["vol"] > 0.25


def test_year_return_and_recovery():
    idx = pd.bdate_range("2019-12-02", "2021-06-30")
    v = np.where(idx < "2020-02-20", 100.0, np.where(idx < "2020-03-23", 70.0, 0))
    ramp = np.linspace(70, 120, int((idx >= "2020-03-23").sum()))
    v[idx >= "2020-03-23"] = ramp
    s = pd.Series(v, index=idx)
    rec = cboe.recovery_date(s)
    assert rec is not None and rec > "2020-03-23" and s.loc[rec] >= 100
    assert cboe.year_return(s, 2020) == pytest.approx(s.loc[:"2020-12-31"].iloc[-1] / 100 - 1)


def test_cash_index_compounds():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    c = cboe.cash_index(pd.Series(0.05, index=idx), idx)
    assert c.iloc[-1] == pytest.approx(math.exp(0.05 * 365 / 365), rel=0.01)


def test_phase03_check_and_builders():
    idx = pd.bdate_range("1988-01-04", "2026-10-23")
    b = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0004, 0.01, len(idx)))), index=idx)
    ix = {n: b * (1 + 0.0) for n in cboe.SHORT_VOL + cboe.TAIL}
    rf = pd.Series(0.02, index=idx)
    res = cboe.compute(ix, b, rf, cboe.SHORT_VOL + cboe.TAIL)
    chk = cboe.phase03_check(res, cboe.SHORT_VOL)
    assert chk["n"] > 0 and all(r["match"] in (True, False, None) for r in chk["rows"])
    # identical series -> identical stats for index and benchmark
    assert res["stats"][("2007", "BXM")]["cagr"] == pytest.approx(res["stats"][("2007", "SPTR")]["cagr"])
    mix = cboe.matched_mix(res["stats"][("1988", "PPUT")]["cagr"], b, rf)
    assert mix["w_stock"] == pytest.approx(1.0, abs=0.02)
    _check_schema(cboe.build_short_vol(res))
    _check_schema(cboe.build_tail_hedge(res, mix))


# --- strangle ----------------------------------------------------------------------

def test_strike_for_delta_roundtrip():
    S, T, sig, r = 4000.0, 45 / 365, 0.2, 0.03
    Kp = strangle.strike_for_delta(S, T, sig, r, 0.16, call=False)
    Kc = strangle.strike_for_delta(S, T, sig, r, 0.16, call=True)
    assert Kp < S < Kc
    assert bs_put_delta(S, Kp, T, sig, r) == pytest.approx(-0.16, abs=1e-3)


def test_bs_put_call_parity_and_intrinsic():
    S, K, T, sig, r = 100.0, 95.0, 0.5, 0.25, 0.04
    c = strangle.bs_price(S, K, T, sig, r, call=True)
    p = strangle.bs_price(S, K, T, sig, r, call=False)
    assert float(c - p) == pytest.approx(S - K * math.exp(-r * T), rel=1e-6)
    assert float(strangle.bs_price(80.0, K, 0.0, sig, r, call=False)) == pytest.approx(15.0)


def _flat_market(n=400, level=100.0, iv=0.2):
    idx = _bdays(n, "2010-01-04")
    return (pd.Series(level, index=idx), pd.Series(iv, index=idx), pd.Series(0.0, index=idx))


def test_simulate_flat_market_all_win_and_p50_exits_early():
    S, iv, rf = _flat_market()
    df = strangle.simulate(S, iv, rf, dte=45, step=20)
    assert (df["pnl"] > 0).all()
    hold = df[df["rule"] == "hold"]
    p50 = df[df["rule"] == "p50"]
    assert (hold["pnl"] / hold["credit"]).round(6).eq(1.0).all()      # keeps the full credit
    assert (p50["days"].to_numpy() < hold["days"].to_numpy()).all()
    d21 = df[df["rule"] == "d21"]
    assert (d21["days"] <= 45 - 21 + 4).all()


def test_simulate_crash_stop_limits_loss():
    S, iv, rf = _flat_market(200)
    S = S.copy()
    S.iloc[10:] = 75.0                                                  # -25% gap after entry 0
    df = strangle.simulate(S, iv, rf, dte=45, step=1000)
    first = df.set_index("rule")
    assert first.loc["hold", "pnl"] < 0
    assert first.loc["p50_d21_stop2x", "exit"] <= first.loc["p50_d21", "exit"]


def test_rule_stats_and_book():
    S, iv, rf = _flat_market(500)
    df = strangle.simulate(S, iv, rf, dte=45, step=1)
    sub = df[df["rule"] == "p50_d21"]
    st = strangle.rule_stats(sub)
    assert st["win_rate"] == 1.0 and st["cvar5_pct"] > 0
    bk = strangle.sequential_book(sub)
    assert (bk["entry"].iloc[1:].to_numpy() >= bk["exit"].iloc[:-1].to_numpy()).all()


def test_regt_bpr_floor():
    assert strangle.regt_bpr(100.0, 50.0, 150.0, 0.0, 0.0) == pytest.approx(10.0)


def test_strangle_compute_and_build():
    s = _gbm(900, 0.15, seed=11, start="2007-01-02", drift=0.05)
    vix = pd.Series(0.19, index=s.index)
    rf = pd.Series(0.01, index=s.index)
    res = strangle.compute(s, vix, rf, step=5)
    assert set(res["by_rule"]) == set(strangle.RULES)
    _check_schema(strangle.build(res))


# --- earnings ----------------------------------------------------------------------

def _earn_series(n=1400, jumps=(), drift_after=0.0, seed=2):
    idx = _bdays(n, "2004-01-02")
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    for p, j in jumps:
        r[p] += j
        r[p + 2:p + 23] += drift_after * np.sign(j) / 21
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return pd.DataFrame({"close": c, "volume": 1e6}), pd.Series(100.0, index=idx)


def test_ticker_events_detects_outlier_and_continuation():
    pos = list(range(100, 1300, 63))
    jumps = [(p, 0.01) for p in pos[:-1]] + [(pos[-1], 0.20)]
    px, spy = _earn_series(jumps=jumps, drift_after=0.25)
    dates = [px.index[p].date().isoformat() for p in pos]
    ev = earnings.ticker_events(px, spy, dates, "T")
    assert len(ev) == len(pos)
    last = ev[-1]
    assert last["reaction"] > 0.15 and last["drift_21"] > 0
    df = earnings.classify(pd.DataFrame(ev))
    assert bool(df["out_A"].iloc[-1]) and bool(df["out_C"].iloc[-1])
    assert not df["out_A"].iloc[:earnings.MIN_PRIOR].any()     # no own history yet


def test_cont_stats_and_summarize():
    rows = []
    for i in range(300):
        rx = 0.1 if i % 2 else -0.1
        rows.append({"ticker": f"T{i % 10}", "report_date": f"{2003 + i % 20}-05-01", "year": 2003 + i % 20,
                     "reaction": rx, "dollar_vol": float(i), "drift_5": rx * (1 if i % 3 else -1),
                     "drift_21": rx * (1 if i % 4 else -1), "mean_abs8": 0.02, "mean8": 0.0, "sd8": 0.02})
    res = earnings.summarize(pd.DataFrame(rows))
    a = res["variants"]["A"]["h21"]
    assert a["n"] == 300 and a["share_cont"] == pytest.approx(0.75, abs=0.01)
    assert set(res["variants"]["A"]["by_period"]) == {"2001-2005", "2006-2015", "2016-2026"}
    tr = earnings.tracker_stats([{"side": "long", "reaction_day": "2024-05-01",
                                  "horizons": {"h20": {"evaluated": True, "excess_ret": 0.02},
                                               "h5": {"evaluated": True, "excess_ret": -0.01}}},
                                 {"side": "short", "reaction_day": "2024-06-01",      # stored side-adjusted
                                  "horizons": {"h20": {"evaluated": True, "excess_ret": 0.03}}}])
    assert tr["h20"]["share_cont"] == 1.0 and tr["h5"]["n"] == 1
    _check_schema(earnings.build(res, tr))


def test_sample_universe_stratified():
    rows = [{"ticker": f"T{i}", "weight_pct": 1000 - i} for i in range(900)]
    s = earnings.sample_universe(rows, 90, seed=1)
    assert len(s) == 90
    ranks = sorted(int(t[1:]) for t in s)
    assert sum(r < 300 for r in ranks) == 30 and sum(r >= 600 for r in ranks) == 30


# --- expected move ----------------------------------------------------------------

def test_expected_move_flat_and_jump():
    S, iv, _ = _flat_market(300)
    res = expected_move.compute(S, iv)
    assert res["overall"]["inside"] == 1.0
    S2 = S.copy()
    S2.iloc[150:] = 80.0
    rows = expected_move.compute_rows(S2, iv)
    assert rows["down_break"].any() and not rows["up_break"].any()
    _check_schema(expected_move.build(expected_move.compute(_gbm(800, 0.2), pd.Series(0.2, index=_gbm(800, 0.2).index))))


# --- vix spikes ----------------------------------------------------------------------

def test_vix_events_declustered():
    idx = _bdays(200)
    v = pd.Series(0.15, index=idx)
    v.iloc[50:55] = 0.35
    v.iloc[56] = 0.15
    v.iloc[57:60] = 0.40          # re-cross within 21 days: not a new event
    v.iloc[120] = 0.32
    ev = vix_spikes.events(v)
    assert [idx.get_loc(d) for d in ev["cross30"]] == [50, 120]
    assert idx.get_loc(ev["jump50"][0]) == 50


def test_vix_spikes_compute_and_build():
    s = _gbm(1500, 0.18, seed=21)
    v = pd.Series(0.18, index=s.index)
    for p in (300, 700, 1100):
        v.iloc[p:p + 10] = 0.40
    res = vix_spikes.compute(s, v)
    assert res["events"]["cross30"]["n"] == 3
    _check_schema(vix_spikes.build(res))


# --- charts ----------------------------------------------------------------------------

def test_charts_build_from_chart_data():
    from zenith.vault.repro import charts
    s = _gbm(900, 0.15, seed=11, start="2007-01-02", drift=0.05)
    vix = pd.Series(0.19, index=s.index)
    objs = {"vrp_spx": vrp.build(vrp.compute(vix, s)),
            "strangle_mgmt": strangle.build(strangle.compute(s, vix, pd.Series(0.01, index=s.index), step=5)),
            "expected_move": expected_move.build(expected_move.compute(s, vix))}
    for name, obj in objs.items():
        spec = charts.chart(name, obj).to_dict()
        assert spec.get("data") or spec.get("datasets") or spec.get("hconcat")


# --- Phase 02 tasty fills ---------------------------------------------------------

def test_save_reapplies_tasty_fills(tmp_path, monkeypatch):
    fills = tmp_path / "fills.json"
    fills.write_text(json.dumps({"fills": [
        {"repro": "demo", "claim": "B is true", "tasty_value": "tasty says B",
         "tasty_sources": ["page:concepts-strangle"], "tasty_note": ""},
        {"repro": "other", "claim": "A is true", "tasty_value": "wrong file", "tasty_sources": [], "tasty_note": ""}]}),
        encoding="utf-8")
    monkeypatch.setattr(repro, "TASTY_FILLS_PATH", fills)
    monkeypatch.setattr(repro, "REPRO_DIR", tmp_path / "out")
    claims = [repro.claim("A is true", "seed", "ours A", "mixed"), repro.claim("B is true", "seed", "ours B", "agrees")]
    obj = repro.artefact("demo", claim_text="B is true", claims=claims, our_value="x", data="d", period="p",
                         method="m", caveats=[], verdict="agrees", chart_data={})
    _check_schema(obj)                                   # build output stays pending
    repro.save("demo", obj)
    out = json.loads((tmp_path / "out" / "demo.json").read_text(encoding="utf-8"))
    a, b = out["claims"]
    assert a["tasty_value"] is None and a["tasty_note"] == repro.TASTY_NOTE      # no fill for this file
    assert b["tasty_value"] == "tasty says B" and b["tasty_sources"] == ["page:concepts-strangle"]
    assert out["tasty_value"] == "tasty says B"                                   # top-level mirrors its row


def test_committed_fills_match_current_claims():
    if not repro.TASTY_FILLS_PATH.exists():
        pytest.skip("no committed fills file")
    fills = repro.load_fills()
    assert len(fills) == 21
    for (name, text), f in fills.items():
        obj = repro.load(name, {})
        assert text in [c["text"] for c in obj.get("claims", [])], (name, text)
        assert isinstance(f["tasty_sources"], list) and f["tasty_note"]

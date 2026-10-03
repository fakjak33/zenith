"""EPHEMERIS stats engine vs hand-computed fixtures, the Read's guardrails,
and a seeded ~200-guess profile rendering the STATS / READ views."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from zenith.ephemeris import report as R
from zenith.ephemeris import seed
from zenith.ephemeris import stats as S
from zenith.ephemeris.repo import SqliteRepo


def _df(rows):
    base = {"id": 0, "mode": "practice", "ts": "2026-01-01T00:00:00Z", "timeframe": "Daily", "horizon": 10,
            "asset_class": "Bonds", "sector": "", "conviction": "Medium", "direction": 1, "rule_call": 0,
            "indicators_json": "[]", "regime_json": "{}", "exit_reason": "horizon", "ambiguous": 0,
            "r_mult": None, "atr_ret": None, "mfe": 0.0, "mae": 0.0, "mfe_full": 0.0, "mae_full": 0.0,
            "candles_held": 10, "stake": 1000.0, "base_rate": 0.5, "rule_ret": None}
    out = []
    for i, r in enumerate(rows):
        x = dict(base, id=i, ts=(pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i)).isoformat())
        x.update(r)
        x.setdefault("trade_ret", x["direction"] * x["market_ret"])
        x.setdefault("pnl", x["trade_ret"] * x["stake"])
        x.setdefault("win", int(x["trade_ret"] > 0))
        out.append(x)
    return S.prepare(pd.DataFrame(out))


def test_wilson_matches_hand_computation():
    p, lo, hi = S.wilson(7, 10)
    # z=1.959964: centre=(0.7+0.192)/1.384=0.6445, half=1.96*sqrt(0.021+0.00096)/1.384=0.2346
    assert p == 0.7 and lo == pytest.approx(0.3968, abs=5e-4) and hi == pytest.approx(0.8922, abs=5e-4)
    assert S.wilson(0, 0)[0] != S.wilson(0, 0)[0]                      # nan for n=0


def test_binomial_and_normal_tail():
    assert S.binom_two_sided(5, 10) == pytest.approx(1.0)
    assert S.binom_two_sided(9, 10) == pytest.approx(22 / 1024)        # P(<=1)+P(>=9) = 2*11/1024
    assert S.norm_sf(1.959964) == pytest.approx(0.025, abs=1e-6)


def test_edge_vs_base_hand_fixture():
    wins = np.array([1, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.6, 0.6])
    e = S.edge_vs_base(wins, p)
    # diff = 3 - 2.2 = 0.8 ; var = .25+.25+.24+.24 = .98 ; z = .8/sqrt(.98)
    assert e["edge"] == pytest.approx(0.2) and e["base"] == pytest.approx(0.55)
    assert e["z"] == pytest.approx(0.8 / math.sqrt(0.98))
    assert e["lo"] == pytest.approx(0.2 - 1.959964 * math.sqrt(0.98) / 4)


def test_headline_money_metrics():
    d = _df([{"market_ret": 0.02}, {"market_ret": -0.01}, {"market_ret": 0.03}, {"market_ret": -0.04}])
    h = S.headline(d)
    assert h["n"] == 4 and h["wins"] == 2 and h["hit"] == 0.5
    assert h["expectancy"] == pytest.approx(0.0)                      # (20-10+30-40)/4
    assert h["profit_factor"] == pytest.approx(50 / 50)
    assert h["win_loss"] == pytest.approx(25 / 25)
    assert h["longest_win"] == 1 and h["longest_loss"] == 1
    assert h["always_long_hit"] == 0.5


def test_equity_curve_drawdown_and_resets():
    d = _df([{"market_ret": 0.1}, {"market_ret": -0.2}, {"market_ret": 0.05}, {"market_ret": 0.3}])
    eq = S.equity_curve(d, [], start=1000)
    assert eq["equity"].tolist() == pytest.approx([1100, 900, 950, 1250])
    dd = S.drawdown_stats(eq)
    assert dd["max_dd"] == pytest.approx(-200) and dd["max_dd_pct"] == pytest.approx(-200 / 1100)
    assert dd["longest_dd"] == 2
    eq2 = S.equity_curve(d, [{"ts": "2026-01-01T01:30:00Z"}], start=1000)
    assert eq2["equity"].tolist() == pytest.approx([1100, 900, 1050, 1350]) and eq2["reset"].sum() == 1


def test_paired_rule_edge_and_calibration():
    d = _df([{"market_ret": 0.02, "rule_call": 1}, {"market_ret": 0.01, "rule_call": -1},
             {"market_ret": -0.03, "rule_call": 0}])
    # player always long: [.02,.01,-.03]; rule: [.02,-.01,0] -> diffs [0,.02,-.03]
    pr = S.paired_edge(d["trade_ret"], d["rule_ret"])
    assert pr["mean"] == pytest.approx(-0.01 / 3)
    rows = [{"market_ret": 0.01, "conviction": "High"}] * 30 + [{"market_ret": -0.01, "conviction": "Low"}] * 30
    cal = S.calibration(_df(rows))
    assert cal["monotonic"] is True
    assert cal["brier"] == pytest.approx(((0.75 - 1) ** 2 + (0.55 - 0) ** 2) / 2)


def test_breakdown_flags_small_n_and_report_makes_no_small_claims():
    d = _df([{"market_ret": 0.01, "asset_class": "Crypto"}] * 10)    # 10/10 wins, but n < 30
    t = S.breakdown(d, "asset_class")
    assert bool(t["small_n"].iloc[0]) and t["n"].iloc[0] == 10
    rep = R.build(d)
    assert not rep["strengths"] and not rep["weaknesses"]
    assert any("fewer than 30" in x["text"] for x in rep["notes"])


def test_stopped_then_reversed_and_hindsight():
    d = _df([{"market_ret": 0.05, "trade_ret": -0.02, "exit_reason": "stop", "atr_ret": -1.0, "mae_full": -0.03},
             {"market_ret": -0.05, "trade_ret": -0.02, "exit_reason": "stop", "atr_ret": -1.0, "mae_full": -0.06}])
    st = S.stops_analysis(d)
    assert st["stopped_n"] == 2 and st["stopped_then_reversed"] == 0.5
    hs = st["hindsight"]
    one = hs[hs["stop"] == "1×ATR"].iloc[0]                           # atr_pct = 0.02, both MAEs reach 2%
    assert one["exp_ret"] == pytest.approx(-0.02)
    none = hs[hs["stop"] == "none"].iloc[0]
    assert none["exp_ret"] == pytest.approx(0.0)


def test_seeded_profile_read_finds_planted_structure():
    repo = SqliteRepo(":memory:")
    pid = seed.seed(repo, "demo", 200)
    d = S.prepare(repo.guesses_df(pid))
    assert len(d) == 200 and set(d["mode"]) == {"practice", "daily"}
    rep = R.build(d)
    text = " ".join(x["text"] for x in rep["strengths"])
    assert "calls in Crypto" in text                               # planted 65% skill, specific label
    assert any("calls in Bonds" in x["text"] for x in rep["weaknesses"])   # planted 38%
    assert rep["drills"] and rep["drills"][0]["text"].startswith("Next drill:")
    for k in ("strengths", "weaknesses"):
        for x in rep[k]:
            assert "n=" in x["text"] and "CI" in x["text"]           # never a claim without n and CI
    html = R.to_html(rep, "demo")
    assert "<h1>EPHEMERIS" in html and "Strengths" in html


@pytest.mark.parametrize("sub,needle", [("Stats", "equity curve"), ("Read", "summary")])
def test_stats_and_read_views_render_with_seeded_profile(tmp_path, monkeypatch, sub, needle):
    db = tmp_path / "eph.sqlite3"
    seed.seed(SqliteRepo(db), "demo-seed", 200)
    monkeypatch.setenv("EPHEMERIS_SQLITE", str(db))
    at = AppTest.from_string("\n".join([
        "import streamlit as st",
        "from zenith.ephemeris import view",
        "st.session_state['eph_player'] = 'demo-seed'",
        f"st.session_state['eph_sub'] = {sub!r}",
        "view.render()",
    ]), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(str(m.value) for m in at.markdown).lower()
    if "no price data yet" in " ".join(str(i.value) for i in at.info).lower():
        pytest.skip("no local price store")
    assert needle in text

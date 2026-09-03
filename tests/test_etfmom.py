"""ETF MOMENTUM tests — fully offline, synthetic, no network.

Mirrors tests/test_mom.py's conventions: seeded random-walk OHLC frames, a
fixture that monkeypatches the package's path dicts (never config), and
monkeypatchable seams (`ec._fetch_prices`, `ec._load_mvt`, `eu.constituents`)
so no test ever makes a ~900-ticker call or touches committed data.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import zenith.etfmom as em
from zenith.etfmom import compute as ec
from zenith.etfmom import history as eh
from zenith.etfmom import mvt_link
from zenith.etfmom import universe as eu
import zenith.mom as mom
from zenith.mom import history as mh
from zenith.config import MOM_WEIGHTS
from zenith.pretom import calendar as cal


def _ohlc_series(n: int, drift: float, vol: float = 0.012, seed: int = 1,
                 start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(r))
    idx = pd.bdate_range(end=date.today(), periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": np.full(n, 1_000_000.0)}, index=idx)


# =========================================================== taxonomy (pure) ==
def test_normalize_category_strips_source_prefix_and_is_idempotent():
    # The exact bug this exists to fix: the committed Morningstar catalog and
    # the yfinance metadata cache spell the same bucket two different ways.
    assert eu.normalize_category("US Fund Large Blend") == "Large Blend"
    assert eu.normalize_category("Large Blend") == "Large Blend"
    assert eu.normalize_category("US Fund Large Blend") == eu.normalize_category("Large Blend")
    once = eu.normalize_category("US Fund Mid-Cap Value")
    assert eu.normalize_category(once) == once
    assert eu.normalize_category(None) == ""


@pytest.mark.parametrize("category,expected", [
    ("Large Blend", "Equity"),
    ("Foreign Large Value", "Equity"),
    ("Technology", "Equity"),
    ("Intermediate Core Bond", "Fixed Income"),
    ("High Yield Muni", "Fixed Income"),
    ("Muni National Interm", "Fixed Income"),
    ("Short Government", "Fixed Income"),
    ("Long Government", "Fixed Income"),
    ("Government Mortgage-Backed Bond", "Fixed Income"),
    ("Ultrashort Bond", "Fixed Income"),
    ("Bank Loan", "Fixed Income"),
    ("Preferred Stock", "Fixed Income"),
    ("Commodities Broad Basket", "Commodity"),
    ("Commodities Focused", "Commodity"),
    ("Real Estate", "Real Estate"),
    ("Global Real Estate", "Real Estate"),
    ("Moderate Allocation", "Allocation"),
    ("Tactical Allocation", "Allocation"),
    ("Single Currency", "Currency"),
    ("Digital Assets", "Digital Assets"),
    ("Equity Market Neutral", "Alternative"),
    ("Derivative Income", "Alternative"),
    ("", "Unknown"),
])
def test_asset_class_rollup(category, expected):
    assert eu.asset_class_of(category) == expected


def test_asset_class_rollup_keeps_equity_sector_funds_out_of_commodity():
    """The three cases most likely to be "simplified" into a bug later.
    These are equity funds holding miners/operators/crypto-adjacent companies,
    not the underlying asset -- Morningstar's own `Equity ` prefix says so."""
    assert eu.asset_class_of("Equity Precious Metals") == "Equity"
    assert eu.asset_class_of("Equity Energy") == "Equity"
    assert eu.asset_class_of("Natural Resources") == "Equity"
    assert eu.asset_class_of("Infrastructure") == "Equity"
    assert eu.asset_class_of("Energy Limited Partnership") == "Equity"
    # Exact match, not substring: "Equity Digital Assets" holds equities.
    assert eu.asset_class_of("Equity Digital Assets") == "Equity"
    assert eu.asset_class_of("Digital Assets") == "Digital Assets"


def test_asset_class_is_total_and_never_raises():
    for junk in ("", "   ", "Some Brand New Morningstar Category", "???"):
        assert eu.asset_class_of(junk) in eu.ASSET_CLASSES


# ================================================================= mvt_link ==
def _mvt_doc(as_of: date, rows, exclusions=None):
    return {"as_of": as_of.isoformat(),
            "rows": [{"ticker": t, "normalized_score": v} for t, v in rows],
            "empirical_exclusions": exclusions or {}}


def test_mvt_direct_and_inherited(monkeypatch):
    today = date(2026, 9, 2)
    doc = _mvt_doc(today, [("SPY", 4.5), ("AGG", -2.0)],
                   {"VOO": "near_duplicate_of:SPY", "IVV": "near_duplicate_of:SPY"})
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: doc)
    scores, status = mvt_link.scores(today=today)
    assert scores["SPY"] == {"score": 4.5, "source": "SPY"}
    # Inherited across the near-duplicate cluster, and LABELLED as inherited --
    # never laundered into looking like it was measured on VOO.
    assert scores["VOO"] == {"score": 4.5, "source": "SPY"}
    assert scores["IVV"]["source"] == "SPY"
    assert status["direct"] == 2 and status["inherited"] == 2
    assert status["mvt_stale_days"] == 0


def test_mvt_inheritance_does_not_dangle_when_keeper_is_unscored(monkeypatch):
    today = date(2026, 9, 2)
    doc = _mvt_doc(today, [("AGG", -2.0)], {"VOO": "near_duplicate_of:SPY"})
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: doc)
    scores, status = mvt_link.scores(today=today)
    assert "VOO" not in scores          # nothing to inherit -> five-factor row
    assert status["inherited"] == 0


def test_mvt_ignores_non_duplicate_exclusion_reasons(monkeypatch):
    today = date(2026, 9, 2)
    doc = _mvt_doc(today, [("SPY", 4.5)],
                   {"TQQQ": "empirical_leverage_vs_SPY(corr=0.99,vol_ratio=3.0x)"})
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: doc)
    scores, _ = mvt_link.scores(today=today)
    assert "TQQQ" not in scores


def test_mvt_used_when_recent_and_dropped_when_too_stale(monkeypatch):
    today = date(2026, 9, 2)
    fresh = _mvt_doc(today - timedelta(days=3), [("SPY", 4.5)])
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: fresh)
    scores, status = mvt_link.scores(today=today, max_stale_days=10)
    assert scores and status["mvt_stale_days"] == 3 and "error" not in status

    stale = _mvt_doc(today - timedelta(days=40), [("SPY", 4.5)])
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: stale)
    scores, status = mvt_link.scores(today=today, max_stale_days=10)
    # Dropped for EVERY row rather than blended: a stale 6th factor mixed into
    # five fresh ones is a silent discontinuity.
    assert scores == {} and "too_stale" in status["error"]


def test_mvt_missing_artefact_is_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(mvt_link, "mvt_load", lambda name, default=None: {})
    scores, status = mvt_link.scores(today=date(2026, 9, 2))
    assert scores == {} and status["error"] == "mvt_etf_artefact_missing_or_empty"


# ============================================================ leverage gate ==
def test_leverage_gate_excludes_geared_funds_but_keeps_near_duplicates():
    """The decision-2 implementation under test: a 3x clone must go, an exact
    near-duplicate must stay (mvt drops those; a ranked list must not)."""
    base = _ohlc_series(400, 0.0004, seed=7)
    px = {"SPY": base, "VOO": base.copy()}
    # A geared clone: same path, 3x the daily move.
    r = base["close"].pct_change().fillna(0.0)
    lev_close = 100 * (1 + 3 * r).cumprod()
    px["TQQQ"] = pd.DataFrame({"open": lev_close, "high": lev_close * 1.005,
                               "low": lev_close * 0.995, "close": lev_close,
                               "volume": np.full(len(lev_close), 1e6)}, index=base.index)
    for i in range(12):   # filler so the panel is wide enough to correlate on
        px[f"F{i}"] = _ohlc_series(400, 0.0002, seed=100 + i)

    lev = ec._leverage_exclusions(px)
    assert "TQQQ" in lev and lev["TQQQ"].startswith("empirical_leverage")
    assert "VOO" not in lev and "SPY" not in lev
    assert not any(r.startswith("near_duplicate_of") for r in lev.values())


# ==================================================================== store ==
@pytest.fixture
def tmp_etfmom_store(tmp_path, monkeypatch):
    """Redirects the ETF store to tmp_path. Mirrors test_mom.py's
    tmp_mom_store, including why: these are module-level path dicts read at
    call time, so patching them (not config) is what actually redirects."""
    files = {k: tmp_path / f"{k}.json" for k in
             ("scores", "detail", "categories", "diagnostics", "picks", "status")}
    hist = tmp_path / "history"
    monkeypatch.setattr(em, "ETFMOM_FILES", files)
    monkeypatch.setattr(eh, "ETFMOM_HISTORY_DIR", hist)
    return {"files": files, "history": hist}


def test_history_sharding_and_picks_round_trip(tmp_etfmom_store):
    friday = date(2026, 8, 28)
    rows = [{"ticker": "SPY", "composite": 6.0, "state": "BULLISH",
             "factor_scores": {"ts": 0.3}, "contributions": {"ts": 1.2},
             "side": "long", "rank": 1, "pctile": 99.0},
            {"ticker": "AGG", "composite": -6.0, "state": "BEARISH",
             "factor_scores": {"ts": -0.3}, "contributions": {"ts": -1.2},
             "side": "short", "rank": 2, "pctile": 1.0}]
    assert eh.append_history(rows, friday) == 2
    assert eh.append_history(rows, friday) == 0          # idempotent per (date, ticker)
    shard = json.loads((tmp_etfmom_store["history"] / "2026.json").read_text())
    assert {r["ticker"] for r in shard["rows"]} == {"SPY", "AGG"}
    assert all("factor_scores" in r for r in shard["rows"])   # Friday => full row
    assert eh.append_picks(eh.make_pick_rows(rows, friday.isoformat())) == 2
    assert eh.series_for("SPY")[0]["composite"] == 6.0


def test_etf_history_writes_do_not_touch_the_mom_store(tmp_etfmom_store):
    """The whole point of the mom/history.py parameterization: binding the ETF
    store must not leak into the Russell 1000 one."""
    before = sorted(p.name for p in mh.MOM_HISTORY_DIR.glob("*.json")) \
        if mh.MOM_HISTORY_DIR.exists() else []
    eh.append_history([{"ticker": "SPY", "composite": 1.0, "state": "NEUTRAL"}],
                      date(2026, 8, 28))
    after = sorted(p.name for p in mh.MOM_HISTORY_DIR.glob("*.json")) \
        if mh.MOM_HISTORY_DIR.exists() else []
    assert before == after
    assert (tmp_etfmom_store["history"] / "2026.json").exists()


def test_scrub_replaces_non_finite_with_none():
    out = ec._scrub({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.5}})
    assert out == {"a": None, "b": [1.0, None], "c": {"d": 2.5}}
    json.dumps(out)   # would emit a bare NaN token and corrupt the file otherwise


# ================================================================ end-to-end ==
def _fake_universe():
    classes = [("Equity", "Large Blend"), ("Fixed Income", "Intermediate Core Bond"),
               ("Commodity", "Commodities Focused")]
    uni, px = [], {}
    for i in range(24):
        t = f"E{i:02d}"
        ac, cat = classes[i % 3]
        uni.append({"ticker": t, "name": f"Fund {t}", "category": cat, "asset_class": ac,
                    "region": "", "aum_m": 100.0 + i, "er": 0.10, "included": True})
        px[t] = _ohlc_series(700, 0.0006 if i < 12 else -0.0005, seed=i + 1)
    # A fund younger than MIN_BARS -- must be excluded with a stated reason,
    # never scored on a truncated window.
    uni.append({"ticker": "YOUNG", "name": "New Fund", "category": "Large Blend",
                "asset_class": "Equity", "region": "", "aum_m": None, "er": None,
                "included": True})
    px["YOUNG"] = _ohlc_series(120, 0.0005, seed=99)
    # A name-gated exclusion must never enter the scored set.
    uni.append({"ticker": "LEVX", "name": "3x Something", "category": "Large Blend",
                "asset_class": "Equity", "region": "", "aum_m": None, "er": None,
                "included": False, "exclusion_reason": "leveraged_or_inverse_by_name"})
    return uni, px


def _wire(monkeypatch, uni, px, mvt_scores, mvt_status):
    monkeypatch.setattr(eu, "constituents", lambda: (uni, {"n_raw": len(uni)}))
    monkeypatch.setattr(ec, "_fetch_prices",
                        lambda tk, p, st, lb: (st.append({"segment": lb, "ok": True,
                                                          "n": len(px)}), px)[1])
    monkeypatch.setattr(ec, "_load_mvt", lambda today: (mvt_scores, mvt_status))
    monkeypatch.setattr(cal, "is_trading_day", lambda d: True)


def test_run_auto_end_to_end(tmp_etfmom_store, monkeypatch):
    uni, px = _fake_universe()
    mvt = {"E00": {"score": 9.0, "source": "E00"},
           "E01": {"score": -7.0, "source": "E00"}}    # E01 inherits from E00
    _wire(monkeypatch, uni, px, mvt,
          {"mvt_as_of": date.today().isoformat(), "mvt_stale_days": 0,
           "direct": 1, "inherited": 1, "n_rows": 1})

    res = ec.run_auto(force=True)
    assert res["ok"] and res["scored"] == 24

    doc = json.loads(tmp_etfmom_store["files"]["scores"].read_text())
    rows = doc["rows"]
    scored = [r for r in rows if not r["excluded"]]
    assert doc["n"] == 25 and doc["n_scored"] == 24     # LEVX never entered at all
    assert {r["ticker"] for r in rows if r["excluded"]} == {"YOUNG"}
    assert "insufficient_history" in next(r for r in rows if r["excluded"])["exclusion_reason"]
    assert "LEVX" not in {r["ticker"] for r in rows}

    assert sorted(r["rank"] for r in scored) == list(range(1, len(scored) + 1))
    assert all(0 <= r["pctile"] <= 100 for r in scored)
    assert all(-20 <= r["composite"] <= 20 for r in scored)
    longs = {r["ticker"] for r in scored if r["side"] == "long"}
    shorts = {r["ticker"] for r in scored if r["side"] == "short"}
    assert longs and shorts and not (longs & shorts)

    # THE renormalization invariant. It must hold for the five-factor rows too
    # -- that is the whole reason a missing mvt is dropped rather than zeroed.
    for r in scored:
        assert abs(sum(r["contributions"].values()) - r["composite"]) < 1e-4
        assert r["n_factors"] == len(r["factor_scores"])
        assert r["n_factors"] in (5, 6)
    assert {r["ticker"] for r in scored if r["n_factors"] == 6} == {"E00", "E01"}
    assert next(r for r in scored if r["ticker"] == "E01")["mvt_source"] == "E00"
    assert next(r for r in scored if r["ticker"] == "E00")["mvt_source"] is None

    # Both taxonomy levels must PARTITION the scored set exactly.
    cats = json.loads(tmp_etfmom_store["files"]["categories"].read_text())
    assert sum(v["n"] for v in cats["asset_classes"].values()) == doc["n_scored"]
    assert sum(v["n"] for v in cats["categories"].values()) == doc["n_scored"]
    assert set(cats["asset_classes"]) <= set(eu.ASSET_CLASSES)

    status = json.loads(tmp_etfmom_store["files"]["status"].read_text())
    segs = {s["segment"]: s for s in status["segments"]}
    assert segs["coverage"]["coverage"] >= 0.85
    assert segs["mvt"]["mvt_as_of"] == date.today().isoformat()
    assert segs["history"]["benchmark"] == "equal_weight_universe"
    assert json.loads(tmp_etfmom_store["files"]["picks"].read_text())["rows"]


def test_run_auto_survives_a_missing_mvt_artefact(tmp_etfmom_store, monkeypatch):
    """A bad mvt night must degrade to a uniform five-factor cross-section, not
    take the tab down and not impute a neutral zero."""
    uni, px = _fake_universe()
    _wire(monkeypatch, uni, px, {}, {"mvt_as_of": None, "mvt_stale_days": None,
                                     "direct": 0, "inherited": 0, "n_rows": 0,
                                     "error": "mvt_etf_artefact_missing_or_empty"})
    res = ec.run_auto(force=True)
    assert res["ok"]
    scored = [r for r in json.loads(tmp_etfmom_store["files"]["scores"].read_text())["rows"]
              if not r["excluded"]]
    assert scored
    for r in scored:
        assert r["n_factors"] == 5
        assert "mvt" not in r["factor_scores"]          # absent, never 0.0
        assert abs(sum(r["contributions"].values()) - r["composite"]) < 1e-4


def test_run_auto_gates_on_non_trading_day(tmp_etfmom_store, monkeypatch):
    monkeypatch.setattr(cal, "is_trading_day", lambda d: False)
    assert ec.run_auto(force=False) == {"ok": True, "gated": True}
    status = json.loads(tmp_etfmom_store["files"]["status"].read_text())
    assert status["is_trading_day"] is False


def test_universe_benchmark_is_equal_weight_not_a_single_ticker():
    """Regression guard on the reason this exists: measured against SPY, a long
    bond-fund pick would fail through every bull market regardless of signal
    skill, and the IC would be reporting asset-class beta as evidence."""
    px = {f"E{i:02d}": _ohlc_series(400, 0.0005 if i < 15 else -0.0005, seed=i + 1)
          for i in range(30)}
    bench = ec.universe_benchmark(px, list(px))
    assert bench is not None and len(bench) > 60
    # An equal-weight blend of up- and down-trending funds must sit between the
    # two extremes, i.e. it is not just tracking one member.
    finals = sorted((px[t]["close"].iloc[-1] / px[t]["close"].iloc[0]) for t in px)
    ratio = float(bench.iloc[-1] / bench.iloc[0])
    assert finals[0] < ratio < finals[-1]
    assert ec.universe_benchmark(px, list(px)[:3]) is None    # too few to average


# =============================================================== invariants ==
def test_weights_are_momentums_not_a_fork():
    """If someone later forks the weights, the two tabs stop being comparable,
    which is the main thing having both is for. Assert the shared object."""
    import inspect
    src = inspect.getsource(ec.run_auto)
    assert "weights=MOM_WEIGHTS" in src
    assert abs(sum(MOM_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(MOM_WEIGHTS) == set(em.FACTORS)


def test_vocabulary_is_reexported_from_mom_not_redefined():
    assert em.FACTORS is mom.FACTORS
    assert em.FACTOR_LABELS is mom.FACTOR_LABELS
    assert em.HORIZONS is mom.HORIZONS

"""Offline tests for the FMOM feature (synthetic data, no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from zenith.fmom import core, history
from zenith.fmom.catalog import ETF_PROXIES
from zenith.fmom.families import etf_proxy


# ------------------------------------------------------------------ helpers --
def synth_panel(n_factors: int = 6, n_months: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    data = rng.normal(0.002, 0.02, size=(n_months, n_factors))
    return pd.DataFrame(data, index=idx,
                        columns=[f"F{i}" for i in range(n_factors)])


# --------------------------------------------------------------------- core --
def test_tsfm_clip_bounds():
    panel = synth_panel()
    panel.iloc[-1, 0] = 5.0     # absurd +500% month
    panel.iloc[-1, 1] = -5.0
    s = core.tsfm_signals(panel).iloc[-1]
    assert s["F0"] == pytest.approx(core.CLIP)
    assert s["F1"] == pytest.approx(-core.CLIP)
    assert (s.dropna().abs() <= core.CLIP + 1e-12).all()


def test_signals_need_vol_history():
    panel = synth_panel(n_months=core.VOL_MIN_PERIODS)   # vol lags one month
    assert core.tsfm_signals(panel).dropna(how="all").empty


def test_leg_weights_sum_to_unit_legs():
    s = pd.Series({"A": 1.5, "B": 0.5, "C": -0.3, "D": -0.9, "E": 0.0})
    w = core.leg_weights(s)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)
    assert w["E"] == 0.0
    assert w["A"] == pytest.approx(1.5 / 2.0)
    assert w["D"] == pytest.approx(-0.9 / 1.2)


def test_leg_weights_degenerate_one_sided():
    w = core.leg_weights(pd.Series({"A": 0.4, "B": 1.1}))
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()          # empty short leg stays empty


def test_csfm_zero_when_cross_section_identical():
    panel = synth_panel()
    panel.iloc[-1] = 0.03          # every factor identical this month
    s = core.csfm_signals(panel).iloc[-1]
    assert np.allclose(s.dropna(), 0.0)


def test_csfm_equals_tsfm_of_demeaned_numerator():
    panel = synth_panel()
    vol = core.ann_vol(panel)
    expected = (panel.sub(panel.mean(axis=1), axis=0) / vol).clip(-2, 2)
    pd.testing.assert_frame_equal(core.csfm_signals(panel), expected)


def test_no_lookahead():
    panel = synth_panel()
    bumped = panel.copy()
    bumped.iloc[-1] = 0.5          # rewrite the final month
    a = core.tsfm_signals(panel).iloc[:-1]
    b = core.tsfm_signals(bumped).iloc[:-1]
    pd.testing.assert_frame_equal(a, b)   # earlier signals unmoved


def test_decile_flags():
    s = pd.Series(np.linspace(-1, 1, 20), index=[f"F{i}" for i in range(20)])
    flags = core.decile_flags(s)
    assert (flags == "top").sum() == 2
    assert (flags == "bottom").sum() == 2
    assert flags[s.idxmax()] == "top"
    assert flags[s.idxmin()] == "bottom"


def test_backtest_alignment_and_stats():
    panel = synth_panel(n_months=72)
    bt = core.backtest(panel)
    series = bt["series"]
    # earliest model return needs VOL_MIN_PERIODS of history plus one holding month
    assert series["tsfm"].dropna().index[0] >= panel.index[core.VOL_MIN_PERIODS]
    # independent recomputation of the final month (alignment check)
    t, t1 = panel.index[-2], panel.index[-1]
    s = core.tsfm_signals(panel).loc[t].dropna()
    expected = core.model_return(core.leg_weights(s), panel.loc[t1])
    assert series.loc[t1, "tsfm"] == pytest.approx(expected)
    stats = bt["stats"]["tsfm"]
    assert stats["n_months"] == series["tsfm"].dropna().shape[0]
    assert -1.0 <= stats["max_dd"] <= 0.0


def test_backtest_ew_benchmark_is_mean_of_factors():
    panel = synth_panel()
    bt = core.backtest(panel)
    t1 = panel.index[-1]
    assert bt["series"].loc[t1, "ew"] == pytest.approx(float(panel.loc[t1].mean()))


def test_perf_stats_short_series_degrades():
    assert core.perf_stats(pd.Series([0.01] * 5)) == {"n_months": 5}


def test_winsorize_caps_extremes():
    x = pd.Series(np.arange(100, dtype=float))
    w = core.winsorize(x, 0.05)
    assert w.max() == pytest.approx(x.quantile(0.95))
    assert w.min() == pytest.approx(x.quantile(0.05))


def test_build_factor_bins_paper_construction():
    rng = np.random.default_rng(3)
    n = 200
    chars = pd.Series(rng.normal(size=n), index=[f"T{i}" for i in range(n)])
    caps = pd.Series(rng.lognormal(10, 1, size=n), index=chars.index)
    out = core.build_factor_bins(chars, caps)
    bins = out["bins"]
    assert set(bins) == {"LL", "LM", "LH", "SL", "SM", "SH"}
    # 30/40/30 split within each size half (quantile ties make this approximate)
    for size in ("L", "S"):
        counts = {c: len(bins[size + c]) for c in ("L", "M", "H")}
        half = sum(counts.values())
        assert half == pytest.approx(n / 2, abs=1)
        assert counts["L"] == pytest.approx(0.30 * half, abs=3)
        assert counts["M"] == pytest.approx(0.40 * half, abs=3)
        assert counts["H"] == pytest.approx(0.30 * half, abs=3)
    for key, members in bins.items():
        assert sum(m["w"] for m in members) == pytest.approx(1.0, abs=1e-3)
    # every Large member out-caps the split, every Small member is under it
    large = {m["ticker"] for c in "LMH" for m in bins["L" + c]}
    assert all(caps[t] >= out["size_split"] for t in large)


def test_build_factor_bins_tiny_universe_degrades():
    out = core.build_factor_bins(pd.Series({"A": 1.0}), pd.Series({"A": 10.0}))
    assert out["bins"] == {}


def test_monthly_returns_resample():
    days = pd.date_range("2024-01-01", "2024-03-29", freq="B")
    close = pd.Series(np.linspace(100, 120, len(days)), index=days)
    m = core.monthly_returns(close)
    assert len(m) == 2                       # first month seeds pct_change
    assert (m > 0).all()


# ------------------------------------------------------------------ history --
def _fake_history(tmp_path, monkeypatch):
    """Point the FMOM artefact store at a temp dir."""
    from zenith import config
    files = {k: tmp_path / v.name for k, v in config.FMOM_FILES.items()}
    monkeypatch.setattr("zenith.fmom.FMOM_FILES", files)
    return files


def test_history_append_idempotent(tmp_path, monkeypatch):
    _fake_history(tmp_path, monkeypatch)
    w = pd.Series({"A": 0.6, "B": 0.4, "C": -1.0})
    row = history.make_row("2026-06", "etf_tsfm", w)
    assert history.append_rows([row]) == 1
    assert history.append_rows([row]) == 0           # duplicate skipped
    rows = history.load("history")["rows"]
    assert len(rows) == 1
    assert rows[0]["picks"]["long"] == [{"factor": "A", "w": 0.6},
                                        {"factor": "B", "w": 0.4}]
    assert rows[0]["picks"]["short"] == [{"factor": "C", "w": -1.0}]
    assert not rows[0]["degenerate"]


def test_history_degenerate_flag(tmp_path, monkeypatch):
    _fake_history(tmp_path, monkeypatch)
    row = history.make_row("2026-06", "etf_tsfm", pd.Series({"A": 0.5, "B": 0.5}))
    assert row["degenerate"]


def test_history_evaluation_and_summary(tmp_path, monkeypatch):
    _fake_history(tmp_path, monkeypatch)
    w = pd.Series({"A": 1.0, "B": -1.0})
    history.append_rows([history.make_row("2026-05", "etf_tsfm", w),
                         history.make_row("2026-06", "etf_tsfm", w)])
    panel = pd.DataFrame(
        {"A": [0.02], "B": [-0.01]},
        index=pd.to_datetime(["2026-06-30"]))
    filled = history.evaluate_pending({"etf": panel})
    assert filled == 1                     # 2026-06 stays pending (no 2026-07 yet)
    rows = history.load("history")["rows"]
    done = next(r for r in rows if r["month"] == "2026-05")
    # long A: +2%; short B: -(-1%) = +1% -> model +3%
    assert done["realized"]["model_ret"] == pytest.approx(0.03)
    assert done["realized"]["long_ret"] == pytest.approx(0.02)
    assert done["realized"]["short_ret"] == pytest.approx(-0.01)  # return OF the short side
    assert done["realized"]["hit"] is True
    pending = next(r for r in rows if r["month"] == "2026-06")
    assert not pending["evaluated"]

    summary = history.summarize(rows)
    m = summary["models"]["etf_tsfm"]
    assert m["n_months"] == 1
    assert m["hit_rate"] == 1.0
    assert m["cum_ret"] == pytest.approx(0.03)
    assert summary["pending_months"] == ["2026-06"]


def test_history_next_month_rollover():
    assert history._next_month("2026-12") == "2027-01"
    assert history._next_month("2026-01") == "2026-02"


# ------------------------------------------------------------------ catalog --
def test_catalog_unique_and_complete():
    tickers = [p["ticker"] for p in ETF_PROXIES]
    factors = [p["factor"] for p in ETF_PROXIES]
    assert len(set(tickers)) == len(tickers)
    assert len(set(factors)) == len(factors)
    assert all(p["ticker"].isupper() and p["since"] for p in ETF_PROXIES)


# --------------------------------------------------------------- man styles --
def _toy_catalog() -> dict:
    def e(t, groups, cat="US Fund Large Blend", aum=100.0, sheets=None):
        return {"ticker": t, "name": t, "category": cat, "groups": groups,
                "aum_m": aum, "er": 0.2, "sheets": sheets or ["VALUE"]}
    return {"etfs": [
        e("VA1", ["Value"], aum=500), e("VA2", ["Value"]), e("VA3", ["Value"]),
        e("MO1", ["Momentum"]), e("MO2", ["Momentum"]), e("MO3", ["Momentum"]),
        e("MF1", ["Momentum", "Quality", "Value"]),          # multifactor, 3 core tags
        e("MF2", ["Value"], sheets=["MULTIFACTOR 1"]),
        e("MF3", ["Momentum", "Value", "Volatility", "Quality"]),
        e("FX1", ["Value"], cat="US Fund Foreign Large Blend"),   # excluded: region
        e("FI1", ["Value", "FixedIncome"]),                       # excluded: group
    ]}


def test_man_members_rules():
    from zenith.fmom.families import man_styles
    mem = man_styles.members(_toy_catalog())
    assert set(mem) == set(__import__("zenith.fmom.catalog",
                                      fromlist=["MAN_STYLES"]).MAN_STYLES)
    value = [e["ticker"] for e in mem["Value"]]
    assert value == ["VA1", "VA2", "VA3"]        # AUM-sorted, no FX1/FI1/MF*
    momentum = [e["ticker"] for e in mem["Momentum"]]
    assert set(momentum) == {"MO1", "MO2", "MO3"}
    multi = [e["ticker"] for e in mem["Multifactor"]]
    assert set(multi) == {"MF1", "MF2", "MF3"}


def test_man_members_real_catalog_counts():
    """Against the committed catalog every style must have enough candidates."""
    import json
    from pathlib import Path
    from zenith.fmom.families import man_styles
    from zenith.fmom.catalog import MAN_MIN_MEMBERS
    path = Path(__file__).resolve().parent.parent / "data" / "fmom" / "etf_catalog.json"
    cat = json.loads(path.read_text(encoding="utf-8"))
    mem = man_styles.members(cat)
    assert len(mem) == 13
    for style, etfs in mem.items():
        assert len(etfs) >= MAN_MIN_MEMBERS, f"{style} has only {len(etfs)} members"


def test_man_panel_composite_math():
    from zenith.fmom.families import man_styles
    cat = _toy_catalog()
    tks = man_styles.tickers(cat)
    px = _fake_px(tks)
    panel, meta = man_styles.build_panel(px, cat)
    assert meta["ok"]
    assert "Value" in panel.columns and "Momentum" in panel.columns
    # composite = EW mean of members minus SPY
    spy_m = core.monthly_returns(px["SPY"]["close"])
    t = panel.index[-1]
    expect = (sum(core.monthly_returns(px[k]["close"])[t]
                  for k in ("VA1", "VA2", "VA3")) / 3) - spy_m[t]
    assert panel.loc[t, "Value"] == pytest.approx(expect)
    # styles with too few live members stay out
    assert "Yield" not in panel.columns


# ---------------------------------------------------------------- aqr family --
def test_aqr65_table_integrity():
    from zenith.fmom.catalog import AQR65, AQR_SUPPLEMENTAL
    assert len(AQR65) == 65
    abbrs = [r["abbr"] for r in AQR65]
    assert len(set(abbrs)) == 65
    for r in AQR65 + AQR_SUPPLEMENTAL:
        if r["live_source"] == "sort":
            assert r["ds"] and r["long_col"] and r["short_col"], r["abbr"]
        elif r["live_source"] == "factor":
            assert r["ds"], r["abbr"]
        elif r["live_source"] == "aqr":
            assert r["aqr_key"], r["abbr"]
        if r["yf_char"]:
            assert r["long_side"] in ("high", "low"), r["abbr"]


def test_aqr_panel_from_stubbed_loaders(monkeypatch):
    from zenith.fmom.families import aqr_published
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(5)

    def fake_series(*a, **k):
        return pd.Series(rng.normal(0.003, 0.02, 60), index=idx)
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_french_sorts",
                        fake_series)
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_french_factor",
                        fake_series)
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_aqr_series",
                        fake_series)
    panel, meta = aqr_published.build_panel()
    assert meta["ok"]
    n_live = len(aqr_published.live_rows())
    assert meta["n_factors"] == n_live
    assert not meta["missing"]
    cov = meta["coverage"]
    assert cov["defined"] == 65 and cov["live"] >= 14 and cov["replicable"] >= 15


def test_aqr_panel_missing_sources_degrade(monkeypatch):
    from zenith.fmom.families import aqr_published
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_french_sorts",
                        lambda *a, **k: None)
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_french_factor",
                        lambda *a, **k: None)
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_aqr_series",
                        lambda *a, **k: None)
    panel, meta = aqr_published.build_panel()
    assert panel.empty and not meta["ok"]
    assert len(meta["missing"]) == len(aqr_published.live_rows())


# -------------------------------------------------------------- replication --
def test_replication_holdings_synthetic():
    from zenith.fmom.families import replication
    rng = np.random.default_rng(9)
    n = 120
    tickers = [f"T{i}" for i in range(n)]
    universe = [{"ticker": t, "name": f"Co {t}"} for t in tickers]
    fund = {t: {"marketCap": float(rng.lognormal(23, 1)),
                "priceToBook": float(rng.lognormal(1, 0.5)),
                "trailingPE": float(rng.lognormal(3, 0.4)),
                "priceToSalesTrailing12Months": float(rng.lognormal(1, 0.5)),
                "operatingCashflow": float(rng.lognormal(21, 1)),
                "dividendYield": float(abs(rng.normal(0.02, 0.01))),
                "returnOnEquity": float(rng.normal(0.15, 0.1)),
                "returnOnAssets": float(rng.normal(0.07, 0.05)),
                "profitMargins": float(rng.normal(0.1, 0.08)),
                "revenueGrowth": float(rng.normal(0.08, 0.1)),
                "debtToEquity": float(abs(rng.normal(80, 40))),
                "enterpriseToEbitda": float(rng.lognormal(2.5, 0.4)),
                "totalCash": float(rng.lognormal(21, 1)),
                "freeCashflow": float(rng.lognormal(20.5, 1))}
            for t in tickers}
    px = _fake_px(tickers + ["SPY"], months=30)
    out = replication.build_holdings(universe, px, fund)
    assert out["n_factors"] >= 15
    bm = out["factors"]["BM"]
    assert bm["long_bins"] == ["LH", "SH"]
    assert 0.5 <= bm["coverage"] <= 1.0
    # value factor: every LONG (high B/M) name has char >= every SHORT name's
    long_chars = [m["char"] for k in bm["long_bins"] for m in bm["bins"][k]]
    short_chars = [m["char"] for k in bm["short_bins"] for m in bm["bins"][k]]
    assert min(long_chars) >= max(short_chars)
    smb = out["factors"]["SMB"]
    assert smb["long_bins"] == ["LL", "SL"]       # size: long the small side
    # price-based factor present too
    assert "MOM" in out["factors"]


def test_replication_price_chars_beta_of_spy_clone():
    from zenith.fmom.families import replication
    px = _fake_px(["SPY"], months=30)
    clone = {"CLONE": px["SPY"].copy(), "SPY": px["SPY"]}
    chars = replication._price_chars(clone, px["SPY"])
    assert chars["beta"]["CLONE"] == pytest.approx(1.0, abs=1e-6)
    assert chars["ivol"]["CLONE"] == pytest.approx(0.0, abs=1e-9)
    assert chars["wh52"]["CLONE"] <= 1.0


# -------------------------------------------------------------- osap family --
def _stub_osap_series(n_signals=8, end="2024-12-31", months=400, seed=13):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=months, freq="ME")
    return {f"Sig{i}": pd.Series(rng.normal(0.004, 0.03, months), index=idx)
            for i in range(n_signals)}


def test_osap_panel_stale_months_accepted(monkeypatch):
    from zenith.fmom.families import osap
    monkeypatch.setattr("zenith.cas.backtest.factor_data.load_osap",
                        lambda **k: _stub_osap_series())
    panel, meta = osap.build_panel()
    assert meta["ok"] and meta["annual"] and meta["max_lag"] is None
    assert meta["last_month"] == "2024-12"


def test_model_block_max_lag_none_vs_3():
    from zenith.fmom.compute import _model_block
    idx = pd.date_range(end="2024-12-31", periods=120, freq="ME")
    rng = np.random.default_rng(2)
    panel = pd.DataFrame(rng.normal(0.003, 0.02, (120, 5)), index=idx,
                         columns=list("ABCDE"))
    stale = _model_block(panel, "osap", "tsfm", max_lag=None)
    assert stale is not None and stale["n_factors"] == 5
    assert all(r["source_month"] == "2024-12" for r in stale["rows"])
    # a factor 6 months staler than the others is dropped at max_lag=3
    panel2 = panel.copy()
    panel2.loc[panel2.index[-6]:, "E"] = np.nan
    tight = _model_block(panel2, "aqr", "tsfm", max_lag=3)
    assert "E" not in {r["factor"] for r in tight["rows"]}
    loose = _model_block(panel2, "osap", "tsfm", max_lag=None)
    assert "E" in {r["factor"] for r in loose["rows"]}


def test_osap_excluded_from_backfill(monkeypatch, tmp_path):
    from zenith.fmom import compute
    _fake_history(tmp_path, monkeypatch)
    panel = synth_panel(n_factors=4, n_months=40)
    fake = {"etf": (panel, {"ok": True, "n_factors": 4, "last_month": "x"}),
            "osap": (panel.copy(), {"ok": True, "n_factors": 4,
                                    "annual": True, "max_lag": None,
                                    "last_month": "x"})}
    monkeypatch.setattr(compute, "_build_panels", lambda status: fake)
    compute.run_backfill()
    from zenith.fmom import history as hist_mod
    rows = hist_mod.load("history")["rows"]
    fams = {r["model"].split("_")[0] for r in rows}
    assert "etf" in fams and "osap" not in fams


def test_osap_char_map_integrity():
    from zenith.fmom.catalog import OSAP_CHAR_MAP, CHAR_META
    for sig, char in OSAP_CHAR_MAP.items():
        assert char in CHAR_META, f"{sig} maps to unknown char {char}"


# ------------------------------------------------------- definitions/screens --
def test_definitions_complete():
    from zenith.fmom.catalog import (AQR65, AQR_DEFS, AQR_SUPPLEMENTAL,
                                     ETF_PROXIES, MAN_STYLES)
    for r in AQR65 + AQR_SUPPLEMENTAL:
        assert r["abbr"] in AQR_DEFS, r["abbr"]
        d, rec = AQR_DEFS[r["abbr"]]
        assert d and rec
    for p in ETF_PROXIES:
        assert p["definition"] and p["recipe"], p["factor"]
    for s, rule in MAN_STYLES.items():
        assert rule["definition"] and rule["recipe"], s


def test_aqr_yf_chars_are_known():
    from zenith.fmom.catalog import AQR65, CHAR_META
    for r in AQR65:
        if r["yf_char"]:
            assert r["yf_char"] in CHAR_META, r["abbr"]


def test_build_screens_ordering():
    from zenith.fmom.families import replication
    rng = np.random.default_rng(4)
    n = 100
    tickers = [f"T{i}" for i in range(n)]
    universe = [{"ticker": t, "name": t} for t in tickers]
    chars = {"mktcap": pd.Series(rng.lognormal(23, 1, n), index=tickers),
             "mom_12_2": pd.Series(rng.normal(0.1, 0.3, n), index=tickers)}
    out = replication.build_screens(universe, chars, n=10)
    mom = out["screens"]["mom_12_2"]
    assert len(mom["long"]) == 10 and len(mom["short"]) == 10
    # momentum: high_is long -> long list descends from the max
    assert mom["long"][0]["value"] == pytest.approx(chars["mom_12_2"].max())
    assert mom["short"][0]["value"] == pytest.approx(chars["mom_12_2"].min())
    size = out["screens"]["mktcap"]      # size premium: long the SMALLEST
    assert size["long"][0]["value"] == pytest.approx(chars["mktcap"].min())


def test_new_price_chars_present():
    from zenith.fmom.families import replication
    px = _fake_px(["AAA", "SPY"], months=30)
    chars = replication._price_chars(px, px["SPY"])
    for key in ("mom_6m", "price", "realized_vol"):
        assert key in chars and "AAA" in chars[key].index


def test_trim_rows():
    from zenith.fmom.view import _trim_rows, TRIM_N
    rows = [{"factor": f"F{i}", "s": (i - 100) / 50} for i in range(212)]
    trimmed, was = _trim_rows(rows, show_all=False)
    assert was and len(trimmed) == 2 * TRIM_N
    svals = sorted(r["s"] for r in trimmed)
    assert svals[0] == min(r["s"] for r in rows)
    assert svals[-1] == max(r["s"] for r in rows)
    full, was2 = _trim_rows(rows, show_all=True)
    assert not was2 and len(full) == 212
    small, was3 = _trim_rows(rows[:20], show_all=False)
    assert not was3 and len(small) == 20


def test_factor_info_per_family():
    from zenith.fmom.view import _factor_info
    etf = _factor_info("etf", "Momentum", {})
    assert etf["recipe"] and etf["char"] == "mom_12_2" and etf["proxy"] == "MTUM"
    man = _factor_info("man", "Value", {})
    assert man["recipe"] and man["char"] == "book_to_market"
    aqr = _factor_info("aqr", "ACC", {})
    assert "cash flow" in aqr["recipe"].lower()
    osap_cat = {"signals": {"BM": {"name": "Book to market", "sign": 1.0,
                                   "definition": "Book equity over market equity",
                                   "category": "valuation", "authors": "X",
                                   "year": 1985, "sample_start": 1963,
                                   "sample_end": 1990}}}
    osp = _factor_info("osap", "BM", osap_cat)
    assert osp["char"] == "book_to_market"
    assert "HIGHER returns" in osp["recipe"]


# ---------------------------------------------------------------- etf panel --
def _fake_px(tickers: list[str], months: int = 40) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    days = pd.date_range(end=pd.Timestamp.today().normalize(),
                         periods=21 * months, freq="B")
    out = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + 0.0001 * i
        close = pd.Series(100 * np.exp(np.cumsum(
            rng.normal(drift, 0.01, len(days)))), index=days)
        out[t] = pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1e6})
    return out


def test_etf_panel_excess_vs_spy_and_partial_month_dropped():
    px = _fake_px(etf_proxy.tickers())
    panel, meta = etf_proxy.build_panel(px)
    assert meta["ok"] and meta["n_factors"] == len(ETF_PROXIES)
    # no still-forming month
    cur = pd.Timestamp.today().replace(day=1).normalize()
    assert (panel.index < cur).all()
    # excess return = proxy monthly return minus SPY monthly return
    spy_m = core.monthly_returns(px["SPY"]["close"])
    vlue_m = core.monthly_returns(px["VLUE"]["close"])
    t = panel.index[-1]
    assert panel.loc[t, "Value"] == pytest.approx(vlue_m[t] - spy_m[t])


def test_etf_panel_missing_benchmark():
    px = _fake_px([p["ticker"] for p in ETF_PROXIES])   # no SPY
    panel, meta = etf_proxy.build_panel(px)
    assert panel.empty and not meta["ok"]

"""CAS unit tests — signal math, schema, aggregation, registry. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenith.cas import schema, consensus, overlap
from zenith.cas.signals import indicators as ind
from zenith.cas.signals import strategies, strategies151, factor_rotation as frot
from zenith.cas.backtest import factor_momentum as fm


def _ramp(n=300, start=100.0, step=0.5):
    """A steadily rising price series -> bullish trend."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(start + step * np.arange(n), index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": pd.Series(1e6, index=idx)})


# --- schema ---------------------------------------------------------------
def test_make_signal_clamps_and_states():
    s = schema.make_signal("SPY", "strategies", "momentum", 5.0)
    assert s["signal"] == 1.0 and s["state"] == "buy"
    assert schema.make_signal("X", "strategies", "f", -5.0)["state"] == "sell"
    assert schema.make_signal("X", "strategies", "f", 0.0)["state"] == "neutral"
    assert schema.validate(s)


def test_state_band():
    assert schema.state_from_signal(0.2) == "buy"
    assert schema.state_from_signal(-0.2) == "sell"
    assert schema.state_from_signal(0.05) == "neutral"


def test_dynamic_confidence():
    assert schema.confidence_of(0.7) == "high"
    assert schema.confidence_of(0.4) == "medium"
    assert schema.confidence_of(0.1) == "low"
    assert schema.confidence_of(0.4, percentile=0.95) == "high"   # extreme percentile bumps
    # make_signal derives confidence from strength when not given
    assert schema.make_signal("X", "strategies", "f", 0.8)["confidence"] == "high"
    assert schema.make_signal("X", "strategies", "f", 0.05)["confidence"] == "low"


def test_themes_emits_multiple_families():
    from zenith.cas.signals import themes
    data = {"SPY": _ramp(step=0.3), "XLK": _ramp(step=0.6), "XLE": _ramp(step=0.1)}
    fams = {s["family"] for s in themes.compute(data)}
    assert {"relative_strength", "rs_short", "rs_long", "trend_vs_200dma"} <= fams


# --- indicators / strategies ----------------------------------------------
def test_uptrend_is_bullish():
    df = _ramp()
    assert strategies._momentum(df["close"]) > 0.3
    assert strategies._ma_cross(df["close"], 50, 200) > 0
    assert strategies._trend_following(df["close"]) > 0.5


def test_donchian_breakout_at_top_of_channel():
    df = _ramp()
    # rising series sits at the top of its Donchian channel -> positive
    assert strategies._donchian(df, 20) > 0.5


def test_mean_reversion_opposes_spike():
    df = _ramp()
    df.loc[df.index[-1], "close"] *= 1.5      # sharp spike up
    assert strategies._mean_reversion(df["close"]) < 0


def test_zscore_and_clip():
    s = pd.Series(np.arange(50, dtype=float))
    assert not np.isnan(ind.zscore(s, 20).iloc[-1])
    assert ind.clip1(5) == 1.0 and ind.clip1(-5) == -1.0 and ind.clip1(float("nan")) == 0.0


# --- aggregation ----------------------------------------------------------
def test_overlap_counts_aligned_buys():
    sigs = [
        schema.make_signal("AAA", "strategies", "momentum", 0.8),
        schema.make_signal("AAA", "themes", "relative_strength", 0.6),
        schema.make_signal("AAA", "flows", "cot_leveraged_positioning", 0.4),
        schema.make_signal("BBB", "strategies", "momentum", -0.7),
    ]
    out = overlap.build(sigs)
    aaa = next(r for r in out["ranked"] if r["asset"] == "AAA")
    assert aaa["buy_count"] == 3 and aaa["n_segments"] == 3 and aaa["net"] > 0


def test_consensus_variety_and_state():
    sigs = [schema.make_signal("AAA", "strategies", f"fam{i}", 0.5) for i in range(6)]
    cons = consensus.build(sigs)
    rec = next(r for r in cons if r["asset"] == "AAA")
    assert rec["state"] == "buy"
    assert rec["signal_variety"] == 6
    assert 0.0 <= rec["entropy"] <= 1.0


def test_strategies151_emits_families_for_uptrend():
    df = _ramp()
    sigs = strategies151.compute({"SPY": df}, category_of=lambda t: "broad")
    fams = {s["family"] for s in sigs}
    assert all(f.startswith("s151_") for f in fams)
    assert "s151_sma_cross_50_200" in fams and "s151_tsmom" in fams
    # an uptrend should make the trend families bullish
    trend = next(s for s in sigs if s["family"] == "s151_tsmom")
    assert trend["signal"] > 0.5 and trend["segment"] == "strategies"
    assert all(schema.validate(s) for s in sigs)


def test_master_etf_list_is_sane():
    from zenith.cas.etf_master import MASTER_ETFS, master_tickers
    from zenith.cas.universe import master_etfs
    assert len(MASTER_ETFS) > 300
    assert "SPY" in MASTER_ETFS and "SMH" in MASTER_ETFS
    # master_etfs() unions the core universe with the master list, no dupes
    mt = master_etfs()
    assert len(mt) == len(set(mt)) and len(mt) >= len(master_tickers())


def test_consensus_entropy_high_when_split():
    sigs = ([schema.make_signal("Z", "strategies", f"b{i}", 0.9) for i in range(3)]
            + [schema.make_signal("Z", "strategies", f"s{i}", -0.9) for i in range(3)])
    rec = next(r for r in consensus.build(sigs) if r["asset"] == "Z")
    assert rec["entropy"] > 0.4        # disagreement -> high entropy


# --- factor rotation momentum model ---------------------------------------
def test_frm_ts_momentum_positive_on_uptrend():
    # three US-region style ETFs, all rising -> TS momentum bullish, segment set
    prices = {t: _ramp(step=0.5) for t in ("VLUE", "MTUM", "QUAL")}
    sigs = frot.compute(prices=prices)
    assert sigs and all(schema.validate(s) for s in sigs)
    ts = [s for s in sigs if s["family"] == "frm_ts_mom"]
    assert ts and all(s["signal"] > 0 for s in ts)
    assert all(s["segment"] == "factor_rotation" for s in sigs)
    assert {"frm_ts_mom", "frm_cs_region", "frm_cs_peer",
            "frm_composite"} <= {s["family"] for s in sigs}


def test_frm_cross_section_is_monotonic():
    # different slopes -> different trailing returns -> ranked within the US style peers
    prices = {"VLUE": _ramp(step=1.0), "MTUM": _ramp(step=0.5), "QUAL": _ramp(step=0.1)}
    sigs = frot.compute(prices=prices)
    csr = {s["asset"]: s["signal"] for s in sigs if s["family"] == "frm_cs_region"}
    assert csr["VLUE"] > csr["MTUM"] > csr["QUAL"]
    assert csr["VLUE"] > 0 and csr["QUAL"] < 0


def test_frm_cross_section_by_group_segregates_peers():
    # an industry ETF and a style ETF in the same region must NOT be ranked together
    prices = {"VLUE": _ramp(step=1.0), "MTUM": _ramp(step=0.1), "ITA": _ramp(step=0.5)}
    sigs = frot.compute(prices=prices)
    csr = {s["asset"]: s["signal"] for s in sigs if s["family"] == "frm_cs_region"}
    # ITA is the only 'industry' member -> no peer cross-section -> 0
    assert csr["ITA"] == 0.0
    # the two styles still rank against each other
    assert csr["VLUE"] > csr["MTUM"]


def test_frm_universe_tags_sane():
    from zenith.cas.universe import (frm_tickers, master_etfs, frm_tag, style_of, region_of)
    tickers = frm_tickers()
    assert len(tickers) == len(set(tickers))            # unique
    assert {"MTUM", "IVLU", "EMGF", "ITA", "SMH", "EUFN"} <= set(tickers)
    assert set(tickers) <= set(master_etfs())           # all price-pulled
    assert style_of("IVLU") == "Value" and region_of("IVLU") == "DEV"
    assert frm_tag("ITA")["group"] == "industry"
    assert frm_tag("EUFN")["group"] == "region_sector" and frm_tag("EUFN")["region"] == "EU"
    assert "factor_rotation" in schema.SEGMENTS


def test_logo_vector_well_formed():
    import xml.dom.minidom as md
    from zenith.ui_theme import _logo_vector, THEME
    svg = _logo_vector(84)
    md.parseString(svg)                       # raises if not well-formed XML
    assert svg.startswith("<svg")
    assert "clip-path" in svg and "<animate " in svg          # clipped sphere + sun pulse
    assert "<rect" not in svg                                 # transparent (no bg rect)
    palette = [THEME.coral, THEME.orange, THEME.mustard, THEME.mint, THEME.teal, THEME.navy]
    assert sum(c in svg for c in palette) >= 5                # uses the palette


def test_help_badge_and_section_render():
    from zenith.ui_theme import help_badge, section
    from zenith.cas.help_text import HELP
    b = help_badge(HELP["frm"])
    assert "z-help" in b and "z-tip" in b
    assert help_badge("") == ""
    sec = section("Title", 0, help="hint <x>")        # html-escaped, no raw <
    assert "z-help" in sec and "&lt;x&gt;" in sec


def test_hitrate_math_on_trending_series():
    from zenith.cas.analytics import history
    from zenith.cas.signals import factor_rotation as frm
    # a steadily rising series: TS signal positive AND forward returns positive -> high hit-rate
    df = _ramp(n=600, step=0.5)
    hits = history._series_hits(df["close"], frm._ts_momentum)
    h1 = hits["1m"]
    assert h1[1] > 0 and h1[0] / h1[1] > 0.9


def test_beta_etfs_integrated():
    from zenith.cas.beta_etfs import BETA_ETFS
    from zenith.cas.universe import frm_universe, frm_tickers, frm_tag, master_etfs, label_of
    assert len(BETA_ETFS) > 150
    uni = frm_universe()
    beta_only = [t for t in BETA_ETFS if (frm_tag(t) or {}).get("group") == "beta"]
    assert beta_only                                  # at least some new beta names
    t = beta_only[0]
    assert frm_tag(t)["region"] == "US" and frm_tag(t)["label"]
    assert label_of(t) != t                           # resolves to a real name
    assert set(frm_tickers()) <= set(master_etfs())   # all price-pulled
    assert sum(1 for v in uni.values() if v["group"] == "beta") > 100


def test_price_panel_build_and_read(tmp_path, monkeypatch):
    from zenith.cas.analytics import history
    from zenith.cas import store_cas, view
    px = {"SPY": _ramp(n=300), "QQQ": _ramp(n=300, step=1.0)}
    panel = history.build_price_panel(px)
    assert "SPY" in panel and panel["SPY"]["d"] and panel["SPY"]["c"]
    monkeypatch.setitem(store_cas.CAS_FILES, "price_panel", tmp_path / "panel.json")
    store_cas.save("price_panel", panel)
    s = view._price_series("SPY")
    assert s is not None and len(s) > 3


def test_build_history_has_frm_families():
    from zenith.cas.analytics import history
    from zenith.cas.universe import frm_tickers
    closes = {t: _ramp(n=400, step=0.3)["close"] for t in frm_tickers()[:6]}
    rows = history.build_history(closes)
    fams = {r["family"] for r in rows}
    assert {"frm_ts_mom", "frm_cs_region", "frm_composite"} <= fams
    assert all({"date", "asset", "family", "signal"} <= set(r) for r in rows)


def test_aqr_header_finder():
    from zenith.cas.backtest import factor_data
    raw = pd.DataFrame([["AQR disclaimer"], ["more text"], ["DATE"], ["1990-01-31"]])
    assert factor_data._find_header_row(raw) == 2
    assert factor_data._find_header_row(pd.DataFrame([["x"], ["y"]])) is None


# --- academic-factor backtest math (no network) ---------------------------
def _ar1_series(phi, n=400, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    idx = pd.period_range("1990-01", periods=n, freq="M")
    return pd.Series(r, index=idx)


def test_ar1_detects_positive_persistence():
    b, t = fm.ar1(_ar1_series(0.3))
    assert b > 0 and abs(t) > 1.96            # significant positive autocorrelation


def test_ar1_detects_mean_reversion():
    b, _ = fm.ar1(_ar1_series(-0.3))
    assert b < 0                              # negative autocorrelation


def test_ts_strategy_and_sharpe_run():
    s = _ar1_series(0.3)
    strat = fm.ts_strategy(s)
    assert len(strat) > 0
    assert isinstance(fm.sharpe(strat), float)

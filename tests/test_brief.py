"""Weekly Brief unit tests — perf math, GEX gamma, talking points, registry notes.
No network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenith.brief import sources as src
from zenith.brief import compute as bc
from zenith.cas import registry


def _series(n=300, start=100.0, step=0.5):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series(start + step * np.arange(n), index=idx)


def test_perf_basic():
    c = _series()
    p = src._perf(c)
    assert p["last"] == round(float(c.iloc[-1]), 2)
    assert p["w1"] is not None and p["m1"] is not None
    assert isinstance(p["spark"], list) and len(p["spark"]) > 0
    assert p["w1"] > 0                      # rising series -> positive 1w


def test_market_overview_grouped():
    px = {"SPY": pd.DataFrame({"close": _series()}),
          "QQQ": pd.DataFrame({"close": _series(step=1.0)}),
          "GLD": pd.DataFrame({"close": _series(step=0.2)}),
          "TLT": pd.DataFrame({"close": _series(step=-0.1)})}
    ov = src.market_overview(px)
    assert set(ov) >= {"equity", "commodity", "bond", "fx"}     # grouped by asset class
    eq = {r["ticker"] for r in ov["equity"]}
    assert {"SPY", "QQQ"} <= eq
    assert {"GLD"} <= {r["ticker"] for r in ov["commodity"]}
    assert all("spark" in r and "series" in r for r in ov["equity"])


def test_series_is_daily_dicts():
    s = src._series(_series(n=400))
    assert s and isinstance(s[0], dict) and {"d", "c"} <= set(s[0])
    assert len(s) <= 252


def test_perf_has_long_horizons():
    p = src._perf(_series(n=900))           # enough history for 3y (756)
    assert {"m6", "y1", "y3"} <= set(p)
    assert p["y3"] is not None and p["m6"] is not None


def test_mktcap_parse():
    assert src._mktcap_num("$1,234,000,000") == 1234000000.0
    assert src._mktcap_num("N/A") is None and src._mktcap_num(None) is None


def test_grad_css_diverging():
    from zenith.brief import view as bv
    up = bv._grad_css(0.10)
    dn = bv._grad_css(-0.10)
    assert "background-color" in up and "background-color" in dn
    assert up != dn and bv._grad_css(None) == ""


def test_bs_gamma_peaks_atm():
    atm = src._bs_gamma(100, 100, 0.1, 0.2)
    otm = src._bs_gamma(100, 150, 0.1, 0.2)
    assert atm > 0 and atm > otm
    assert src._bs_gamma(100, 100, 0, 0.2) == 0.0      # guards


def test_talking_points_generation():
    ov = [{"ticker": "SPY", "label": "S&P 500", "last": 500, "w1": 0.012, "m1": 0.03},
          {"ticker": "QQQ", "label": "Nasdaq", "last": 430, "w1": 0.02, "m1": 0.05},
          {"ticker": "IWM", "label": "Small", "last": 200, "w1": -0.01, "m1": 0.0},
          {"ticker": "GLD", "label": "Gold", "last": 200, "w1": 0.005, "m1": 0.02},
          {"ticker": "USO", "label": "Oil", "last": 70, "w1": -0.02, "m1": -0.01},
          {"ticker": "UUP", "label": "USD", "last": 28, "w1": 0.001, "m1": 0.0},
          {"ticker": "^VIX", "label": "VIX", "last": 14.0, "w1": -0.05, "m1": 0.0}]
    sectors = [{"label": "Tech", "w1": 0.03}, {"label": "Utilities", "w1": -0.01}]
    rt = {"curve": {"DGS10": {"value": 4.2}, "T10Y2Y": {"value": 0.3}},
          "fed_funds": {"target_lower": 4.0, "target_upper": 4.25, "implied_12m_change_bps": -50}}
    gx = [{"ticker": "SPY", "net_gex_bn": 1.5, "regime": "positive (vol-dampening)"}]
    heat = {"breadth": {"pct_above_50dma": 0.62, "pct_above_200dma": 0.55}}
    pts = bc.talking_points(ov, sectors, rt, gx, heat)
    assert len(pts) >= 5
    blob = " ".join(pts)
    assert "S&P 500" in blob and "Tech led" in blob and "VIX" in blob


def test_registry_note_records_source_and_status(tmp_path, monkeypatch):
    from zenith.cas import store_cas
    # redirect the registry artefact to a temp file so we don't touch committed data
    monkeypatch.setitem(store_cas.CAS_FILES, "registry", tmp_path / "registry.json")
    note = registry.add_note("frm_ts_mom", "Test paper", "", source="http://x", status="pending-review")
    assert note["source"] == "http://x" and note["status"] == "pending-review"
    reg = registry.load()
    assert reg["notes"][0]["title"] == "Test paper"

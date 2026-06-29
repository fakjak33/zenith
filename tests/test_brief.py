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


def test_market_overview_from_prices():
    px = {"SPY": pd.DataFrame({"close": _series()}),
          "QQQ": pd.DataFrame({"close": _series(step=1.0)})}
    ov = src.market_overview(px)
    tickers = {r["ticker"] for r in ov}
    assert {"SPY", "QQQ"} <= tickers
    assert all("spark" in r for r in ov)


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

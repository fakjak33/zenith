"""EPHEMERIS indicators, benchmarks and regime tags.

The critical one is causality: indicators are computed once on the full
fetched series (warm-up + window + horizon) and then sliced. That is only
legitimate if truncating the series at the decision candle changes NO value
at or before it -- tested here for every registered indicator at many
decision points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.ephemeris import indicators as ind
from zenith.ephemeris.benchmarks import base_rate, build_base_rates, trend_rule_call, up_counts
from zenith.ephemeris.regime import tags
from zenith.ephemeris.store_px import resample


def _ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(1e5, 1e6, n).astype(float)
    return o, h, l, c, v


ALL_SPECS = [{"id": k, "params": {}} for k in ind.REGISTRY]


def _flatten(res):
    out = {}
    for o in res["overlays"]:
        out["ov:" + o["name"]] = np.asarray(o["values"], float)
    for p in res["panes"]:
        for s in p["series"]:
            out[f"pn:{p['name']}:{s['name']}"] = np.asarray(s["values"], float)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_indicator_is_causal(seed):
    o, h, l, c, v = _ohlcv(seed=seed)
    anchor = 250
    # panes are capped at 4 per chart, so check in two batches to cover every pane
    batches = [ALL_SPECS[:12], ALL_SPECS[:8] + ALL_SPECS[12:]]
    for specs in batches:
        full = _flatten(ind.compute(specs, o, h, l, c, v, anchor))
        assert full, "no series computed"
        for t in (anchor, 300, 377, 450, 520, 599):
            cut = _flatten(ind.compute(specs, o[:t + 1], h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1], anchor))
            for name, arr in full.items():
                np.testing.assert_allclose(cut[name], arr[:t + 1], rtol=1e-9, atol=1e-9, equal_nan=True,
                                           err_msg=f"{name} changed when truncated at {t}")


def test_all_registry_ids_produce_output():
    o, h, l, c, v = _ohlcv()
    for spec in ALL_SPECS:
        res = ind.compute([spec], o, h, l, c, v, 100)
        assert res["overlays"] or res["panes"], spec["id"]
        for arr in _flatten(res).values():
            assert len(arr) == len(c) and np.isfinite(arr[-1]), spec["id"]


def test_reference_values_match_vela_fixtures():
    np.testing.assert_allclose(ind.sma(np.array([1, 2, 3, 4, 5.]), 3), [np.nan, np.nan, 2, 3, 4], equal_nan=True)
    e = ind.ema(np.array([1, 2, 3, 4, 5.]), 3)
    assert np.isnan(e[1]) and e[2] == 2 and e[3] == pytest.approx(3) and e[4] == pytest.approx(4)
    assert ind.rolling_std(np.array([1, 2, 3, 4.]), 4)[3] == pytest.approx(np.sqrt(1.25))
    up = np.arange(1, 40, dtype=float)
    assert ind.rsi(up, 14)[-1] == 100 and ind.rsi(up[::-1], 14)[-1] == 0
    r = ind.rsi(np.tile([1.0, 2.0], 30), 14)
    assert np.isnan(r[13]) and np.isfinite(r[14])
    line, sig, hist = ind.macd(np.full(80, 5.0))
    assert line[-1] == pytest.approx(0) and hist[-1] == pytest.approx(0)
    u, m, lo = ind.bollinger(np.full(30, 4.0))
    assert u[-1] == m[-1] == lo[-1] == 4


def test_swing_levels_only_known_after_confirmation():
    h = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1, 1, 1.0])
    hi, _ = ind.swing_levels(h, h, k=2)
    assert np.isnan(hi[4]) and hi[5] == 9              # pivot at 3 confirmed at 3+2


def test_trend_rule_and_base_rate_counts():
    up = np.linspace(50, 100, 120)
    assert trend_rule_call(up, 119, 50, 10) == 1
    assert trend_rule_call(up[::-1], 119, 50, 10) == -1
    assert trend_rule_call(up, 30, 50, 10) == 0         # SMA not formed yet
    assert up_counts(np.array([1, 2, 3, 2, 3.]), 1) == (3, 4)
    assert up_counts(np.array([1, 2, 3, 2, 3.]), 2) == (1, 2)             # 1->3 up, 3->3 flat


def test_build_base_rates_and_lookup():
    idx = pd.bdate_range("2010-01-01", periods=400)
    rising = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(10, 20, 400),
                           "volume": 1.0}, index=idx)
    falling = rising.assign(close=np.linspace(20, 10, 400))
    t = build_base_rates({"UP": rising, "DN": falling}, {"UP": "Bonds", "DN": "Crypto"}, resample)
    assert t["Daily"]["Bonds"]["10"]["p"] == 1.0 and t["Daily"]["Crypto"]["10"]["p"] == 0.0
    assert t["Daily"]["ALL"]["10"]["p"] == pytest.approx(0.5, abs=0.02)
    assert base_rate("Bonds", "Daily", 12, t)["h"] == 10              # nearest tabulated horizon
    assert base_rate("Nope", "Daily", 10, t)["p"] == t["Daily"]["ALL"]["10"]["p"]


def test_regime_tags_use_only_past():
    o, h, l, c, v = _ohlcv(800, 3)
    a = tags(o, h, l, c, 500)
    o2 = o.copy(); h2 = h.copy(); l2 = l.copy(); c2 = c.copy()
    c2[501:] *= 3; h2[501:] *= 3; l2[501:] *= 3; o2[501:] *= 3        # rewrite the future
    assert tags(o2, h2, l2, c2, 500) == a
    assert {"trend", "vol", "dist_high", "rsi"} <= set(a)

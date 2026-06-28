"""CAS unit tests — signal math, schema, aggregation, registry. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenith.cas import schema, consensus, overlap
from zenith.cas.signals import indicators as ind
from zenith.cas.signals import strategies, strategies151


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

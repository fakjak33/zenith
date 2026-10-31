"""STRATEGY INDICES — offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenith.stratidx import fetch, registry, signals


def test_csv_reader_handles_both_layouts():
    two = "DATE,CNDR\n06/20/1986,100.0\n06/23/1986,100.5\n"
    five = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n01/03/1990,18.0,18.5,17.0,18.19\n"
    a = fetch.parse_csv(two, "CNDR")
    b = fetch.parse_csv(five, "VIX")
    assert list(a.values) == [100.0, 100.5]
    assert list(b.values) == [17.24, 18.19]                        # CLOSE, not OPEN
    assert b.index[0] == pd.Timestamp("1990-01-02")


def _series(drift, n=400, seed=0):
    rng = np.random.default_rng(seed)
    r = drift + 0.01 * rng.standard_normal(n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2024-01-01", periods=n))


def test_tsmom_and_donchian_signs():
    up, down = _series(0.002), _series(-0.002)
    assert signals.tsmom(up)["sign"] == 1 and signals.tsmom(down)["sign"] == -1
    s = pd.Series(np.r_[np.full(100, 100.0), 101.0], index=pd.bdate_range("2024-01-01", periods=101))
    assert signals.donchian(s)["state"] == "BREAKOUT"
    s2 = pd.Series(np.r_[np.full(100, 100.0), 99.0], index=pd.bdate_range("2024-01-01", periods=101))
    assert signals.donchian(s2)["state"] == "BREAKDOWN"


def test_xs_ranks_order():
    lv = {"A": _series(0.003, seed=1), "B": _series(0.0, seed=2), "C": _series(-0.003, seed=3)}
    xs = signals.xs_ranks(lv)
    assert xs["A"]["r12"] == 1.0 and xs["C"]["r12"] == 0.0


def test_performance_fields():
    p = signals.performance(_series(0.001))
    for k in ("r1m", "r3m", "r1y", "maxdd", "dd", "sharpe1y"):
        assert k in p
    assert p["maxdd"] <= p["dd"] <= 0


def test_registry_has_methodology_for_every_index():
    for t in registry.INDICES:
        url = registry.methodology_url(t)
        assert url.startswith("https://cdn.cboe.com/resources/indices/governance/") and url.endswith(".pdf")
    fams = {v[0] for v in registry.INDICES.values()}
    assert fams == set(registry.FAMILIES)
    assert "CNDR" in registry.INDICES and "0.20" in registry.INDICES["CNDR"][2]

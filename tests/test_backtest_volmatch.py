"""Shared vol-matched toggle: scaling only when on, Sharpe unchanged, never part of a spec hash."""

import numpy as np
import pandas as pd

from zenith.backtest.engine import diversify as dv
from zenith.backtest.spec import StrategySpec
from zenith.backtest.views import common as C


def _rets(seed, vol):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    return pd.Series(rng.normal(0.0003, vol, len(idx)), index=idx)


def test_match_vol_on_off():
    r, b = _rets(1, 0.005), _rets(2, 0.02)
    on, off = C.match_vol(b, r, True), C.match_vol(b, r, False)
    assert abs(on.std() - r.std()) < 1e-12
    assert off.equals(b.reindex(r.index).fillna(0.0))
    sr = lambda x: x.mean() / x.std()  # noqa: E731
    assert abs(sr(on) - sr(off)) < 1e-12
    assert C.bench_label("SPY", True) == "SPY (vol-matched)" and C.bench_label("SPY", False) == "SPY"


def test_frontier_flag_passes_through():
    r, b = _rets(3, 0.005), _rets(4, 0.02)
    a1 = dv.analyse(r, b, 252, vol_match=True)
    a0 = dv.analyse(r, b, 252, vol_match=False)
    assert a1["ok"] and a0["ok"] and not np.isclose(a1["frontier"]["Vol"].iloc[-1], a0["frontier"]["Vol"].iloc[-1])


def test_toggle_not_in_spec():
    assert "vol" not in " ".join(StrategySpec().hash_dict().keys()).replace("target_vol", "")

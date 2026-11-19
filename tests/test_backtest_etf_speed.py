"""Phase 3: ETF LAB speed — same answers, fewer network calls."""

import numpy as np
import pandas as pd

from zenith.backtest.data import providers
from zenith.backtest.etflab import factors


def _panel(n=300, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)), index=idx,
                      columns=[f"T{i}" for i in range(k)])
    px.iloc[:40, 2] = np.nan                      # a late lister
    return px


def test_vectorized_beta_matches_pandas():
    px = _panel()
    mkt = px.mean(axis=1)
    got = factors.beta(px, px.index[-1], mkt, 252)
    sub = px.iloc[-253:]
    m = mkt.reindex(sub.index).pct_change()
    want = sub.pct_change().apply(lambda c: c.cov(m) / m.var())
    # pairwise-complete variance differs slightly for the late lister; others match exactly
    assert np.allclose(got.drop("T2"), want.drop("T2"), atol=1e-10)
    assert np.isfinite(got["T2"])


def test_dividend_growth_is_point_in_time(monkeypatch):
    d = pd.Series([1.0] * 4 + [2.0] * 4 + [4.0] * 4,
                  index=pd.to_datetime([f"{y}-0{q}-15" for y in (2018, 2019, 2020) for q in (1, 4, 7, 9)]))
    monkeypatch.setattr(factors, "_dividends", lambda ts: {t: d for t in ts})
    early = factors.dividend_growth(["X"], years=1, asof="2020-06-01")["X"]     # 2018 → 2019
    late = factors.dividend_growth(["X"], years=1, asof="2021-06-01")["X"]      # 2019 → 2020
    assert np.isclose(early, 1.0) and np.isclose(late, 1.0)
    assert np.isnan(factors.dividend_growth(["X"], years=1, asof="2019-06-01")["X"])  # only 2018 known


def test_get_panel_uses_store_and_never_refetches(monkeypatch):
    px = _panel(k=4)
    store = px[["T0", "T1"]]
    calls = {"prewarm": [], "download": []}
    monkeypatch.setattr(providers, "_store_panel", lambda ts, f: store)

    def fake_prewarm(ts, force_refresh=False):
        calls["prewarm"].append(list(ts))
        return {"T2", "T3"}

    def fake_get_prices(t, start=None, end=None, force_refresh=False, download=True):
        calls["download"].append((t, download))
        if t == "T3":
            raise ValueError("gone")
        return pd.DataFrame({"close": px[t]})

    monkeypatch.setattr(providers, "prewarm", fake_prewarm)
    monkeypatch.setattr(providers, "get_prices", fake_get_prices)
    out = providers.get_panel(["T0", "T1", "T2", "T3"], start="2015-03-01")
    assert calls["prewarm"] == [["T2", "T3"]]                       # one batch, store names excluded
    assert all(dl is False for _, dl in calls["download"])          # no serial second download
    assert list(out.columns) == ["T0", "T1", "T2"] and "T3" in out.attrs["errors"]
    assert out.index.min() >= pd.Timestamp("2015-03-01") and out.attrs["from_store"] == 2

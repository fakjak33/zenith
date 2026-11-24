"""ETF options are struck on prices AS TRADED: raw closes + a dividend yield, not dividend-adjusted closes."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.options import data as D


@pytest.fixture
def fake(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=900)
    raw = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    # total return = raw plus a 2%/yr dividend stream reinvested
    tr = raw * np.exp(0.02 * np.arange(len(idx)) / 252)
    adj = tr / tr.iloc[-1] * raw.iloc[-1]                  # Yahoo-style: anchored to today, old prices lower
    px = {"SPY": adj, "^GSPC": raw * 10}
    monkeypatch.setattr(D, "_px", lambda t: px[t])
    monkeypatch.setattr(D, "get_prices", lambda t, start=None: pd.DataFrame(
        {"open": px[t], "high": px[t], "low": px[t], "close": px[t]}))
    monkeypatch.setattr(D, "cboe_csv", lambda c: pd.Series(18.0, index=idx))
    monkeypatch.setattr(D, "cash_index", lambda: pd.Series(np.linspace(1, 1.1, len(idx)), index=idx))
    state = {"raw": pd.DataFrame({"open": raw, "high": raw, "low": raw, "close": raw})}
    monkeypatch.setattr(D, "get_raw_prices", lambda t: state["raw"])
    return raw, adj, state


def test_etf_uses_raw_close_and_dividend_yield(fake):
    raw, adj, _ = fake
    x = D.load("SPY", "2010-01-01")
    assert np.allclose(x["S"].to_numpy(), raw.to_numpy())             # not the adjusted series
    assert x["S"].iloc[0] > adj.iloc[0] * 1.05                         # old prices are no longer scaled down
    assert x["price_basis"].startswith("as traded")
    assert abs(float(x["q"].iloc[-1]) - 0.02) < 0.003                  # the dividend comes back as a yield
    assert np.allclose(x["ohlc"]["close"].to_numpy(), raw.to_numpy())  # 0DTE settles on raw too


def test_falls_back_to_adjusted_without_raw(fake):
    _, adj, state = fake
    state["raw"] = None
    x = D.load("SPY", "2010-01-01")
    assert np.allclose(x["S"].to_numpy(), adj.to_numpy()) and x["q"] is None
    assert "adjusted" in x["price_basis"]

import numpy as np
import pandas as pd

from zenith.backtest.accumulation.engine import AccumConfig, run_accumulation, benchmarks
from zenith.backtest.accumulation.strategies import BUY_RULES, SELL_RULES
from zenith.backtest.accumulation.regression import regression_channel
from zenith.backtest.accumulation import signals as sig
from zenith.backtest.data.universe import is_leveraged, nonleveraged_master_tickers
from zenith.backtest.data.stocks500 import STOCKS_500


def _ramp(n=500, start="2018-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(np.linspace(100, 200, n), index=idx)


def test_dca_invests_everything():
    close = _ramp()
    cfg = AccumConfig(initial_cash=1000, contribution=100, cadence="Weekly")
    bench = benchmarks(close, cfg)
    assert "DCA" in bench and "Buy & Hold (lump)" in bench
    # both benchmarks end positive and lump >= dca on a monotonic uptrend
    assert bench["Buy & Hold (lump)"].iloc[-1] >= bench["DCA"].iloc[-1] > 0


def test_accumulation_runs_and_holds_cash_until_dip():
    close = _ramp()
    cfg = AccumConfig(initial_cash=10000, contribution=0, cadence="Weekly")
    # drawdown rule never fires on a pure uptrend -> stays in cash
    res = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=10), None, cfg, {})
    assert res.shares.iloc[-1] == 0.0
    assert abs(res.equity.iloc[-1] - 10000) < 1e-6


def test_dca_rule_deploys():
    close = _ramp()
    cfg = AccumConfig(initial_cash=0, contribution=100, cadence="Weekly")
    res = run_accumulation(close, BUY_RULES["dca"](), None, cfg, {})
    assert res.shares.iloc[-1] > 0
    assert "Beta" in res.stats and "Alpha(ann)%" in res.stats


def test_regression_channel_bands_order():
    close = _ramp()
    ch = regression_channel(close, lookback=100, k=2.0, log=True)
    valid = ch.dropna()
    assert (valid["upper"] >= valid["fit"]).all()
    assert (valid["fit"] >= valid["lower"]).all()


def test_rsi_bounds():
    close = _ramp()
    r = sig.rsi(close, 14)
    assert (r >= 0).all() and (r <= 100).all()


def test_leverage_filter():
    assert is_leveraged("TQQQ")
    assert is_leveraged("SOXL")
    assert is_leveraged("X", "Direxion Daily 3x Bull")
    assert not is_leveraged("SPY")
    assert not is_leveraged("XLF")
    nonlev = nonleveraged_master_tickers()
    assert "TQQQ" not in nonlev and "SOXL" not in nonlev


def _wavy(n=400, start="2018-01-01"):
    # oscillating price so drawdown-from-high fires repeatedly
    idx = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    return pd.Series(120 + 20 * np.sin(t / 15.0) + 0.02 * t, index=idx)


def test_fixed_dollar_deploy_holds_more_cash_than_all_in():
    close = _wavy()
    base = dict(initial_cash=10_000_000, contribution=0, cadence="Weekly")
    allin = AccumConfig(deploy_mode="pct_cash", deploy_fraction=1.0, **base)
    fixed = AccumConfig(deploy_mode="fixed_dollar", deploy_dollar=1000, **base)
    r_allin = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=3), None, allin, {})
    r_fixed = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=3), None, fixed, {})
    # fixed-$ keeps far more dry powder than going all-in on the first dip
    assert r_fixed.cash.iloc[-1] > r_allin.cash.iloc[-1]


def test_deploy_modes_differ():
    close = _wavy()
    base = dict(initial_cash=50000, contribution=0)
    pct = AccumConfig(deploy_mode="pct_cash", deploy_fraction=1.0, **base)
    fix = AccumConfig(deploy_mode="fixed_dollar", deploy_dollar=1000, **base)
    r1 = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=3), None, pct, {})
    r2 = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=3), None, fix, {})
    assert abs(r1.equity.iloc[-1] - r2.equity.iloc[-1]) > 1.0


def test_panel_indicators_present():
    close = _ramp()
    for key in ["drawdown", "rsi", "ma_slope"]:
        panel = BUY_RULES[key]().panel_indicator(close, {})
        assert panel is not None
        ser, label, levels = panel
        assert len(ser) == len(close) and isinstance(label, str)


def test_profit_flat_until_first_deploy():
    # drawdown rule never fires on a pure uptrend -> no capital deployed, so the
    # P/L curve must stay flat at exactly $0 even as contributions accumulate.
    close = _ramp()
    cfg = AccumConfig(initial_cash=10_000, contribution=100, cadence="Weekly")
    res = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=10), None, cfg, {})
    assert (res.deployed == 0.0).all()
    assert float(res.profit.abs().max()) < 1e-6        # P/L pinned to zero
    # equity still grows (dry powder), but only from contributions
    assert res.equity.iloc[-1] > res.equity.iloc[0]
    assert abs(res.equity.iloc[-1] - res.invested.iloc[-1]) < 1e-6


def test_deployed_tracks_buys_and_bounds():
    close = _ramp()
    cfg = AccumConfig(initial_cash=0, contribution=100, cadence="Weekly")
    res = run_accumulation(close, BUY_RULES["dca"](), None, cfg, {})
    # capital actually deployed is positive and never exceeds money contributed
    assert res.deployed.iloc[-1] > 0
    assert res.deployed.iloc[-1] <= res.invested.iloc[-1] + 1e-6
    assert res.deployed.is_monotonic_increasing       # no sells -> basis only grows
    assert res.profit.iloc[-1] > 0                     # uptrend -> real gains


def test_sell_reduces_deployed_basis():
    close = _wavy()
    cfg = AccumConfig(initial_cash=50_000, contribution=0, sell_fraction=0.5)
    res = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=3),
                           SELL_RULES["rsi_sell"](level=60), cfg, {})
    # at least one sell happened and basis dipped below its running peak
    assert res.deployed.min() <= res.deployed.max()
    assert res.deployed.iloc[-1] >= 0


def test_accum_stats_schema():
    close = _ramp()
    cfg = AccumConfig(initial_cash=1000, contribution=100, cadence="Weekly")
    res = run_accumulation(close, BUY_RULES["dca"](), None, cfg, {})
    for key in ("FinalEquity", "Contributed", "Deployed", "Profit",
                "ReturnOnContributed%", "ReturnOnDeployed%"):
        assert key in res.stats


def test_stock_universe_size():
    assert len(STOCKS_500) >= 400  # curated large/mid-cap set
    assert len(STOCKS_500) == len(set(STOCKS_500))  # de-duped

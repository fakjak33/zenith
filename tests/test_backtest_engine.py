import numpy as np
import pandas as pd

from zenith.backtest.data.synthetic import generate
from zenith.backtest.rules import REGISTRY
from zenith.backtest.engine.engine import run_strategy
from zenith.backtest.engine.spectrum import run_spectrum
from zenith.backtest.engine import metrics


def test_backtest_runs_and_has_metrics():
    df = generate("trending", n_days=600, seed=5)
    res = run_strategy(REGISTRY["ema"](fast=16, slow=64), df)
    assert len(res.equity) == len(df)
    assert set(["Sharpe", "CAGR", "MaxDD", "DSR"]).issubset(res.stats)
    assert np.isfinite(res.returns).all()


def test_trend_follower_beats_meanrev_on_trending_data():
    """Sanity: a trend rule should out-Sharpe mean reversion on trending data."""
    df = generate("trending", n_days=1200, seed=11)
    trend = run_strategy(REGISTRY["tsmom"](lookback=90), df)
    mr = run_strategy(REGISTRY["meanrev"](lookback=20), df)
    assert trend.stats["Sharpe"] > mr.stats["Sharpe"]


def test_meanrev_beats_trend_on_mean_reverting_data():
    df = generate("mean_reverting", n_days=1200, seed=12)
    trend = run_strategy(REGISTRY["tsmom"](lookback=90), df)
    mr = run_strategy(REGISTRY["meanrev"](lookback=15), df)
    assert mr.stats["Sharpe"] > trend.stats["Sharpe"]


def test_spectrum_produces_table():
    df = generate("trending", n_days=800, seed=7)
    results, table = run_spectrum(REGISTRY["ema"], df, param="fast",
                                  values=[2, 4, 8, 16, 32])
    assert len(results) == len(table) == 5
    assert "Sharpe" in table.columns
    # deflated Sharpe should be present and <= PSR (stricter bar)
    assert (table["DSR"] <= table["PSR"] + 1e-9).all()


def test_direction_filter():
    from zenith.backtest.config import BacktestConfig
    df = generate("gbm", n_days=600, seed=9)
    lo = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(direction="long"))
    so = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(direction="short"))
    assert lo.position.min() >= -1e-9
    assert so.position.max() <= 1e-9


def test_vol_target_toggle_changes_pl():
    from zenith.backtest.config import BacktestConfig
    df = generate("trending", n_days=800, seed=9)
    on = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(use_vol_target=True))
    off = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig(use_vol_target=False))
    assert abs(on.equity.iloc[-1] - off.equity.iloc[-1]) > 1.0


def test_pl_changes_with_seed():
    e1 = run_strategy(REGISTRY["ema"](), generate("trending", n_days=800, seed=1)).equity.iloc[-1]
    e2 = run_strategy(REGISTRY["ema"](), generate("trending", n_days=800, seed=2)).equity.iloc[-1]
    assert abs(e1 - e2) > 1.0


def test_beta_correlation():
    from zenith.backtest.engine.diagnostics import beta_and_correlation
    df = generate("gbm", n_days=600, seed=4)
    res = run_strategy(REGISTRY["ema"](), df)
    bc = beta_and_correlation(res.returns, df["close"])
    assert set(bc) == {"beta", "correlation"}
    assert -1.0 <= bc["correlation"] <= 1.0


def test_extract_trades_has_accurate_prices():
    from zenith.backtest.engine.trades import extract_trades
    df = generate("trending", n_days=800, seed=3)
    res = run_strategy(REGISTRY["ema"](), df)
    led = extract_trades(res.position, df["close"])
    assert not led.empty
    assert led["entry_price"].notna().all() and led["exit_price"].notna().all()
    assert (led["entry_price"] > 0).all() and (led["exit_price"] > 0).all()
    assert set(led["side"]).issubset({"LONG", "SHORT"})


def test_indicator_lines_for_ma_strategies():
    df = generate("gbm", n_days=400, seed=2)
    for k in ["ema", "sma", "gmma", "crossover", "breakout", "meanrev"]:
        lines = REGISTRY[k]().indicator_lines(df)
        assert len(lines) >= 1
        for s in lines.values():
            assert len(s) == len(df)


def test_spectrum_2d_grid():
    from zenith.backtest.engine.spectrum import run_spectrum_2d, make_spectrum
    df = generate("trending", n_days=600, seed=4)
    vx = make_spectrum(2, 64, 4, True)
    vy = make_spectrum(10, 200, 4, True)
    grid = run_spectrum_2d(REGISTRY["ema"], df, "fast", vx, "slow", vy)
    assert grid.shape == (len(vy), len(vx))
    assert grid.notna().sum().sum() > 0


def test_manual_sharpe_matches():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    manual = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(metrics.sharpe(r) - manual) < 1e-9

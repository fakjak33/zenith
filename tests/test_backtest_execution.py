import numpy as np
import pandas as pd

from zenith.backtest.engine.execution import apply_delay
from zenith.backtest.config import BacktestConfig
from zenith.backtest.data.synthetic import generate
from zenith.backtest.rules import REGISTRY
from zenith.backtest.engine.engine import run_strategy


def test_delay_noop_when_zero():
    pos = pd.Series([0, 1, 1, -1, 0], dtype=float)
    out = apply_delay(pos, 0, 0)
    assert out.equals(pos)


def test_entry_delay_holds_position():
    # target opens at index 1; entry_delay=2 means it applies 2 bars later
    pos = pd.Series([0, 1, 1, 1, 1], dtype=float)
    out = apply_delay(pos, entry_delay=2, exit_delay=0)
    assert out.iloc[1] == 0.0  # not yet
    assert out.iloc[3] == 1.0  # applied after 2 bars persisting


def test_exit_delay_holds_position():
    pos = pd.Series([1, 1, 0, 0, 0], dtype=float)
    out = apply_delay(pos, entry_delay=0, exit_delay=2)
    assert out.iloc[2] == 1.0  # exit delayed
    assert out.iloc[4] == 0.0  # eventually closes


def test_delay_changes_backtest():
    df = generate("trending", n_days=800, seed=5)
    base = run_strategy(REGISTRY["ema"](), df, bt=BacktestConfig())
    delayed = run_strategy(REGISTRY["ema"](), df,
                           bt=BacktestConfig(use_delay=True, entry_delay=5, exit_delay=2))
    assert abs(base.equity.iloc[-1] - delayed.equity.iloc[-1]) > 1.0

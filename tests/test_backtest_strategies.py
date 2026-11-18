import numpy as np
import pandas as pd
import pytest

from zenith.backtest import rules as strategies
from zenith.backtest.rules import REGISTRY
from zenith.backtest.data.synthetic import generate
from zenith.backtest.config import FORECAST_CAP


EXPECTED = {"ema", "sma", "gmma", "crossover", "tsmom", "xsmom", "breakout", "meanrev", "carry",
            "turtle", "strev", "tom", "lowvol"}
CROSS_SECTIONAL = {"xsmom", "lowvol"}


def test_all_strategies_registered():
    assert EXPECTED.issubset(set(REGISTRY))


@pytest.mark.parametrize("key", sorted(EXPECTED - CROSS_SECTIONAL))
def test_single_instrument_forecast_shape_and_cap(key):
    df = generate("gbm", n_days=400, seed=1)
    strat = REGISTRY[key]()
    fc = strat.forecast(df)
    assert len(fc) == len(df)
    assert np.isfinite(fc).all()
    assert fc.abs().max() <= FORECAST_CAP + 1e-9


def test_xsmom_panel():
    from zenith.backtest.data.synthetic import generate_panel
    panel = generate_panel(n_assets=5, kind="trending", n_days=400, seed=3)
    strat = REGISTRY["xsmom"]()
    fcs = strat.forecast_panel(panel)
    assert fcs.shape[1] == 5
    assert np.isfinite(fcs.fillna(0)).all().all()


# --- explainer coverage -----------------------------------------------------
@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_every_strategy_is_explained(key):
    """The Strategy Guide is registry-driven, so a rule with no explainer
    silently renders an empty card."""
    x = REGISTRY[key].explain()
    assert x["plain"], f"{key} has no plain-English description"
    assert x["analogy"], f"{key} has no analogy"
    assert len(x["how_it_works"]) >= 3, f"{key} needs at least 3 how-it-works steps"
    assert x["works_when"] and x["fails_when"], f"{key} must say when it fails"
    assert x["evidence"], f"{key} has no stated evidence"


# --- turtle -----------------------------------------------------------------
def _turtle_frame(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": 1_000.0}, index=idx)


def test_turtle_goes_long_on_a_fresh_high():
    # flat for 30 bars, then a sustained climb -> must end up long
    df = _turtle_frame([100.0] * 30 + [100.0 + i for i in range(1, 41)])
    strat = REGISTRY["turtle"](entry_lb=20, exit_lb=10, atr_period=5)
    units = strat._units(df)
    assert units.iloc[-1] > 0, "turtle should be long after a sustained breakout"
    assert (units.iloc[:30] == 0).all(), "no position before any breakout"


def test_turtle_goes_short_on_a_fresh_low():
    df = _turtle_frame([100.0] * 30 + [100.0 - i for i in range(1, 41)])
    strat = REGISTRY["turtle"](entry_lb=20, exit_lb=10, atr_period=5)
    assert strat._units(df).iloc[-1] < 0


def test_turtle_pyramids_but_respects_max_units():
    df = _turtle_frame([100.0] * 30 + [100.0 + 2 * i for i in range(1, 81)])
    strat = REGISTRY["turtle"](entry_lb=20, exit_lb=10, atr_period=5,
                               add_atr=0.5, max_units=4)
    units = strat._units(df)
    assert units.max() > 1, "a strong sustained trend should add units"
    assert units.abs().max() <= 4


def test_turtle_exits_on_the_opposite_channel():
    # climb, then collapse -> the long must be closed out
    up = [100.0] * 25 + [100.0 + i for i in range(1, 31)]
    down = [up[-1] - 3 * i for i in range(1, 21)]
    df = _turtle_frame(up + down)
    strat = REGISTRY["turtle"](entry_lb=20, exit_lb=10, atr_period=5)
    units = strat._units(df)
    assert units.max() > 0
    assert units.iloc[-1] <= 0, "long should be gone after the collapse"


def test_turtle_forecast_preserves_the_unit_ladder():
    """Turtle bypasses scale_forecast on purpose — check the ladder survives."""
    df = _turtle_frame([100.0] * 30 + [100.0 + 2 * i for i in range(1, 81)])
    strat = REGISTRY["turtle"](entry_lb=20, exit_lb=10, atr_period=5, max_units=4)
    fc = strat.forecast(df)
    # more than one distinct non-zero magnitude => the pyramid is still visible
    assert len(set(np.round(fc[fc != 0].abs().unique(), 6))) > 1
    assert fc.abs().max() <= FORECAST_CAP + 1e-9


def test_turtle_channel_does_not_peek_at_the_current_bar():
    """The entry channel is shifted, so a bar cannot trigger on its own high."""
    df = _turtle_frame([100.0] * 40)
    strat = REGISTRY["turtle"]()
    assert (strat._units(df) == 0).all(), "a flat series must never trigger"


# --- short-term reversal ----------------------------------------------------
def test_strev_fades_the_recent_move():
    df = generate("gbm", n_days=500, seed=7)
    fc = REGISTRY["strev"](lookback=5, smooth=1).forecast(df)
    past = df["close"] / df["close"].shift(5) - 1.0
    aligned = pd.DataFrame({"fc": fc, "past": past}).dropna()
    assert aligned["fc"].corr(aligned["past"]) < 0, "reversal must lean against the recent move"


# --- turn-of-month ----------------------------------------------------------
def test_tom_is_flat_mid_month_and_invested_at_the_boundary():
    df = generate("gbm", n_days=400, seed=11)
    fc = REGISTRY["tom"](pre=3, post=3).forecast(df)
    assert (fc >= 0).all(), "turn-of-month is long-only"
    assert (fc == 0).any() and (fc > 0).any(), "must be in some days and out on others"
    # roughly 6 trading days a month out of ~21
    assert 0.15 < (fc > 0).mean() < 0.50


def test_tom_needs_no_price_data():
    """It is a calendar rule — a totally flat price must still produce signals."""
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    flat = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 1.0}, index=idx)
    assert (REGISTRY["tom"]().forecast(flat) > 0).any()


# --- low volatility ---------------------------------------------------------
def test_lowvol_prefers_the_calmer_asset():
    from zenith.backtest.data.synthetic import generate_panel
    panel = generate_panel(n_assets=4, kind="gbm", n_days=600, seed=13)
    fcs = REGISTRY["lowvol"](lookback=126).forecast_panel(panel)
    assert fcs.shape[1] == 4
    assert np.isfinite(fcs.fillna(0)).all().all()

    realized = {t: df["close"].pct_change().std() for t, df in panel.items()}
    calmest = min(realized, key=realized.get)
    wildest = max(realized, key=realized.get)
    tail = fcs.iloc[-1]
    assert tail[calmest] > tail[wildest], "calmest asset must score above the wildest"

"""Date range + date-based lock: the lock is counted back from the END, so moving the start never moves it,
and new fields at their defaults leave old spec hashes untouched."""

import pandas as pd

from zenith.backtest.data.synthetic import generate
from zenith.backtest.spec import StrategySpec, lock_date, split_holdout


def test_lock_years_independent_of_start():
    df = generate("gbm", n_days=3000)
    end = df.index[-1]
    locks = set()
    for k in (0, 500, 1500):
        _, at = split_holdout(df.iloc[k:], 0, 2.0)
        locks.add(at)
    assert len(locks) == 1
    at = locks.pop()
    assert end - pd.DateOffset(months=25) < at <= end - pd.DateOffset(months=24) + pd.Timedelta(days=5)


def test_lock_years_in_sample_excludes_locked():
    df = generate("gbm", n_days=2000)
    ins, at = split_holdout(df, 0, 1.5)
    assert ins.index[-1] < at and len(ins) + (df.index >= at).sum() == len(df)


def test_lock_zero_means_none_and_pct_legacy():
    df = generate("gbm", n_days=1000)
    assert lock_date(df.index, 0, 0) is None
    assert split_holdout(df, 30)[1] == df.index[700]


def test_new_fields_keep_old_hash():
    old = StrategySpec()
    assert old.hash() == StrategySpec(start_date="", end_date="", lock_years=0.0).hash()
    assert old.hash() != StrategySpec(lock_years=2.0).hash()
    assert old.family() != StrategySpec(end_date="2020-01-01").family()


def test_spec_split_uses_lock_years():
    df = generate("gbm", n_days=2000)
    s = StrategySpec(holdout=30, lock_years=1.0)
    _, at = s.split(df)
    assert at == lock_date(df.index, 0, 1.0)
    assert s.lock_label() == "last 1y"


def test_start_end_ts():
    s = StrategySpec(start_year=2010, start_date="2011-03-04", end_date="2015-01-01")
    assert s.start_ts() == pd.Timestamp("2011-03-04") and s.end_ts() == pd.Timestamp("2015-01-01")
    assert StrategySpec(start_year=2010).start_ts() == pd.Timestamp("2010-01-01")
    assert StrategySpec().end_ts() is None


def test_atr_period_late_default_and_plumbing():
    assert StrategySpec().hash() == StrategySpec(atr_period=14).hash()
    s = StrategySpec(atr_period=30, use_atr_stop=True)
    assert s.hash() != StrategySpec(use_atr_stop=True).hash()
    assert s.bt().atr_period == 30 and s.risk().atr_period == 30

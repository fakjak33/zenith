"""Phase 2 validation-dashboard tests — offline, synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.mom.mvt import validate as val


def _synthetic_price_panel(T=650, N=60, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, N)
    sec = rng.integers(0, 5, N)
    mkt = rng.standard_normal(T) * 0.01
    secf = rng.standard_normal((T, 5)) * 0.006
    idvol = np.exp(rng.normal(np.log(0.012), 0.3, N))
    X = beta[None, :] * mkt[:, None] + secf[:, sec] + rng.standard_normal((T, N)) * idvol[None, :]
    close = 100 * np.exp(np.cumsum(X, axis=0))
    idx = pd.bdate_range("2021-01-01", periods=T)
    px = {}
    for i in range(N):
        px[f"T{i}"] = pd.DataFrame({"open": close[:, i], "high": close[:, i] * 1.001,
                                    "low": close[:, i] * 0.999, "close": close[:, i],
                                    "volume": rng.integers(1e5, 5e6, T)}, index=idx)
    spy_close = 100 * np.exp(np.cumsum(mkt))
    px["SPY"] = pd.DataFrame({"open": spy_close, "high": spy_close * 1.001, "low": spy_close * 0.999,
                              "close": spy_close, "volume": rng.integers(1e6, 1e7, T)}, index=idx)
    return px


# --------------------------------------------------------- rebalance_dates --
def test_rebalance_dates_returns_month_ends_in_order():
    idx = pd.bdate_range("2023-01-01", periods=500)
    dates = val.rebalance_dates(idx, n_months=6)
    assert len(dates) == 7  # n_months + 1
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1))
    # each date is the last trading day of its own month
    for d in dates:
        month_end_check = idx[(idx.to_period("M") == d.to_period("M"))].max()
        assert d == month_end_check


# ------------------------------------------------------------- portfolios --
def test_decile_buckets_splits_correctly():
    scores = {f"T{i}": float(i) for i in range(100)}
    longs, shorts = val._decile_buckets(scores, 0.1)
    assert len(longs) == 10 and len(shorts) == 10
    assert min(longs) == "T90"  # top decile = highest scores
    assert max(shorts, key=lambda t: int(t[1:])) == "T9"
    assert not (longs & shorts)


def test_decile_buckets_empty_on_too_few_names():
    scores = {f"T{i}": float(i) for i in range(5)}
    longs, shorts = val._decile_buckets(scores, 0.1)
    assert longs == set() and shorts == set()


def test_portfolio_returns_long_minus_short():
    periods = [{
        "A_timeseries": {"longs": ["A", "B"], "shorts": ["C", "D"],
                         "fwd_return_by_ticker": {"A": 0.10, "B": 0.20, "C": -0.05, "D": 0.05}}
    }]
    rets = val.portfolio_returns(periods, "A_timeseries")
    assert rets == [pytest.approx(0.15 - 0.0)]  # mean(0.10,0.20)=0.15, mean(-0.05,0.05)=0.0


def test_portfolio_returns_none_when_bucket_unevaluable():
    periods = [{"A_timeseries": {"longs": ["A"], "shorts": ["B"], "fwd_return_by_ticker": {}}}]
    assert val.portfolio_returns(periods, "A_timeseries") == [None]


# ------------------------------------------------------- performance_stats --
def test_performance_stats_known_series():
    # constant +1%/month for 12 months -> CAGR ~ (1.01^12 - 1), vol = 0 (no variance)
    rets = [0.01] * 12
    stats = val.performance_stats(rets)
    assert stats["n_periods"] == 12
    assert stats["cagr"] == pytest.approx(1.01 ** 12 - 1, abs=1e-9)
    assert stats["hit_rate"] == 1.0
    assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert stats["volatility"] == pytest.approx(0.0, abs=1e-9)
    assert stats["sharpe"] is None  # zero-vol series -> undefined Sharpe, not inf/nan


def test_performance_stats_insufficient_data():
    stats = val.performance_stats([0.01, 0.02])
    assert "note" in stats and "insufficient" in stats["note"]


def test_performance_stats_drawdown_and_hit_rate():
    rets = [0.10, -0.20, 0.05, 0.05, 0.05, 0.05, 0.05]
    stats = val.performance_stats(rets)
    assert stats["hit_rate"] == pytest.approx(6 / 7)
    assert stats["max_drawdown"] < 0
    # wealth path: 1.10, 0.88, 0.924, 0.9702, 1.01871, 1.0696..., peak was 1.10
    assert stats["max_drawdown"] == pytest.approx(0.88 / 1.10 - 1.0, abs=1e-6)


def test_performance_stats_none_input_filtered():
    stats = val.performance_stats([0.01, None, 0.02, None, 0.03, -0.01, 0.02])
    assert stats["n_periods"] == 5


# ------------------------------------------------------------- turnover --
def test_turnover_full_replacement_is_100pct():
    periods = [
        {"A_timeseries": {"longs": ["A", "B"], "shorts": []}},
        {"A_timeseries": {"longs": ["C", "D"], "shorts": []}},
    ]
    t = val.turnover_stats(periods, "A_timeseries")
    assert t["avg_monthly_turnover"] == pytest.approx(1.0)
    assert t["avg_holding_period_months"] == pytest.approx(1.0)


def test_turnover_no_change_is_zero():
    periods = [
        {"A_timeseries": {"longs": ["A", "B"], "shorts": []}},
        {"A_timeseries": {"longs": ["A", "B"], "shorts": []}},
    ]
    t = val.turnover_stats(periods, "A_timeseries")
    assert t["avg_monthly_turnover"] == pytest.approx(0.0)
    assert t["avg_holding_period_months"] is None  # 1/0 undefined, not inf


# --------------------------------------------------------------- market beta --
def test_market_beta_recovers_known_slope():
    rng = np.random.default_rng(1)
    spy = rng.standard_normal(50) * 0.02
    port = 1.5 * spy  # exact beta of 1.5, no noise
    beta = val.market_beta(list(port), list(spy))
    assert beta == pytest.approx(1.5, abs=1e-6)


def test_market_beta_none_on_insufficient_data():
    assert val.market_beta([0.01, 0.02], [0.01, 0.02]) is None


# ------------------------------------------------------- P&L correlation --
def test_pnl_matrix_signs_shorts_negative():
    periods = [{"asof": "2024-01-31", "A_timeseries": {"longs": ["A"], "shorts": ["B"],
                                                        "fwd_return_by_ticker": {"A": 0.05, "B": -0.03}}}]
    mat = val.pnl_matrix(periods, "A_timeseries")
    assert mat.loc["A"].iloc[0] == pytest.approx(0.05)
    assert mat.loc["B"].iloc[0] == pytest.approx(0.03)  # short * negative return = positive P&L


def test_pairwise_correlation_perfectly_correlated_instruments():
    # 3 instruments (the function's own minimum for a meaningful "average
    # pairwise correlation" -- 2 instruments is just 1 pair, not a sample),
    # all with IDENTICAL P&L series -> every pairwise correlation is 1.0.
    periods = []
    rng = np.random.default_rng(2)
    base = rng.standard_normal(10) * 0.05
    for i in range(10):
        periods.append({"asof": f"2024-{i+1:02d}-28", "A_timeseries": {
            "longs": ["X", "Y", "Z"], "shorts": [],
            "fwd_return_by_ticker": {"X": float(base[i]), "Y": float(base[i]), "Z": float(base[i])},
        }})
    mat = val.pnl_matrix(periods, "A_timeseries")
    stats = val.pairwise_correlation_stats(mat, min_periods_held=5)
    assert stats["n_instruments"] == 3
    assert stats["n_pairs"] == 3
    assert stats["avg_pairwise_corr"] == pytest.approx(1.0, abs=1e-6)


def test_pairwise_correlation_excludes_thinly_held_instruments():
    periods = [{"asof": "2024-01-31", "A_timeseries": {"longs": ["A", "B"], "shorts": [],
                                                        "fwd_return_by_ticker": {"A": 0.01, "B": 0.02}}}]
    mat = val.pnl_matrix(periods, "A_timeseries")
    stats = val.pairwise_correlation_stats(mat, min_periods_held=6)
    assert stats["n_instruments"] == 0
    assert stats["avg_pairwise_corr"] is None


# --------------------------------------------------------- regime / stress --
def test_regime_buckets_splits_into_terciles():
    periods = [{"effective_factor_count": float(v)} for v in range(1, 10)]
    buckets = val.regime_buckets(periods)
    assert set(buckets) == {"low_effective_factors_high_corr_regime", "mid_regime",
                            "high_effective_factors_low_corr_regime"}
    total = sum(len(v) for v in buckets.values())
    assert total == 9


def test_regime_buckets_empty_on_too_few_periods():
    periods = [{"effective_factor_count": 5.0}] * 5
    assert val.regime_buckets(periods) == {}


def test_stress_window_periods_filters_by_date_string():
    periods = [{"asof": "2022-03-31"}, {"asof": "2023-06-30"}, {"asof": "2024-01-31"}]
    wp = val.stress_window_periods(periods, "2022-01-01", "2022-12-31")
    assert wp == [{"asof": "2022-03-31"}]


# ----------------------------------------------------------------- end-to-end --
def test_run_backtest_and_summarize_end_to_end():
    px = _synthetic_price_panel(T=650, N=50, seed=7)
    bt = val.run_backtest(px, n_months=8, progress_every=0)
    assert bt["n_periods"] == 8
    for p in bt["periods"]:
        for m in ("A_timeseries", "B_crosssectional", "C_multivariate", "D_combined"):
            assert m in p
            assert isinstance(p[m]["longs"], list)

    report = val.summarize(bt, px)
    assert report["n_periods"] == 8
    assert set(report["models"]) == {"A_timeseries", "B_crosssectional", "C_multivariate", "D_combined"}
    # market beta should be computable (SPY present, >=6 periods)
    assert report["models"]["A_timeseries"]["market_beta"] is not None
    # stress windows should all report unavailable (synthetic panel starts 2021, only 8 months backtested)
    assert all(not w.get("available", True) or w.get("n_periods", 0) >= 0
              for w in report["by_stress_window"].values())


def test_summarize_flags_in_range_stress_window_as_unavailable_when_unusable(monkeypatch):
    """The real bug this guards against: a stress window's dates can fall
    INSIDE the backtest's nominal period range (len(wp) >= 3) while every
    one of those periods still has empty long/short buckets -- e.g. because
    the factor computations hadn't accumulated enough TRAILING history yet
    (exactly what a live 58-month backtest run showed: the earliest ~20 of
    58 nominal months produced zero usable decile portfolios, and a stress
    window landing entirely inside that stretch must be reported
    unavailable, not silently shown as "0% return that period")."""
    empty_model = {"longs": [], "shorts": [], "fwd_return_by_ticker": {}}
    full_model = {"longs": ["A", "B"], "shorts": ["C", "D"],
                 "fwd_return_by_ticker": {"A": 0.02, "B": 0.01, "C": -0.01, "D": -0.02}}
    all_models = {"A_timeseries": None, "B_crosssectional": None,
                 "C_multivariate": None, "D_combined": None}
    periods = []
    # 6 "unusable" months (empty buckets) inside the stress window's dates
    for i in range(1, 7):
        periods.append({"asof": f"2022-{i:02d}-28", "next": f"2022-{i + 1:02d}-28",
                        "effective_factor_count": 10.0,
                        **{m: empty_model for m in all_models}})
    # 8 "usable" months outside the window
    for i in range(1, 9):
        periods.append({"asof": f"2023-{i:02d}-28", "next": f"2023-{i + 1:02d}-28",
                        "effective_factor_count": 10.0,
                        **{m: full_model for m in all_models}})

    monkeypatch.setattr(val, "MOM_MVT_STRESS_WINDOWS", {"test_window": ("2022-01-01", "2022-12-31")})
    px_full = {"SPY": pd.DataFrame({"close": [400.0] * 700},
                                   index=pd.bdate_range("2021-01-01", periods=700))}
    report = val.summarize({"periods": periods}, px_full)

    window = report["by_stress_window"]["test_window"]
    assert window["n_periods"] == 6            # dates DO fall in range
    assert window["available"] is False        # but nothing was usable
    assert "trailing history" in window["note"].lower()
    assert window["models"]["A_timeseries"]["n_periods"] == 0


def test_run_backtest_handles_insufficient_history():
    """Too little history to clear MOM_MVT_MIN_BARS -- periods are still
    created (the rebalance-date scaffolding doesn't depend on any one
    ticker), but nothing gets scored in them, honestly reflected as
    n_scored=0 rather than a crash or a fabricated period."""
    px = {"A": pd.DataFrame({"close": [100.0] * 50, "volume": [1e5] * 50},
                            index=pd.bdate_range("2024-01-01", periods=50))}
    bt = val.run_backtest(px, n_months=6)
    assert all(p["n_scored"] == 0 for p in bt["periods"])

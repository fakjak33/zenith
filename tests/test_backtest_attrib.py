"""Phase 3: attribution maths, benchmark suggestions, verdict grading."""

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.attrib import core as A
from zenith.backtest.attrib import verdict as V
from zenith.backtest.attrib.benchmarks import sixty_forty, suggest, to_returns


@pytest.fixture(scope="module")
def mkt():
    idx = pd.bdate_range("2005-01-03", periods=3000)
    return pd.Series(np.random.default_rng(1).normal(0.0004, 0.011, len(idx)), index=idx)


def test_pure_beta_has_no_alpha(mkt):
    ab = A.alpha_beta(mkt, mkt, 252)
    assert ab["beta"] == pytest.approx(1.0, abs=1e-9)
    assert ab["alpha_ann"] == pytest.approx(0.0, abs=1e-9)
    assert ab["beta_share"] == pytest.approx(1.0, abs=1e-6)


def test_recovers_alpha_and_beta(mkt):
    rng = np.random.default_rng(2)
    r = 0.5 * mkt + 0.0004 + pd.Series(rng.normal(0, 0.003, len(mkt)), index=mkt.index)
    ab = A.alpha_beta(r, mkt, 252)
    assert ab["beta"] == pytest.approx(0.5, abs=0.03)
    assert ab["alpha_ann"] == pytest.approx(0.0004 * 252, abs=0.03)
    assert ab["alpha_t"] > 2
    assert ab["beta_share"] < 0.5


def test_capture_and_convexity(mkt):
    cap = A.capture(mkt, mkt)
    assert cap["up_capture"] == pytest.approx(1.0) and cap["down_capture"] == pytest.approx(1.0)
    m = A.monthly(mkt)
    # a straddle-like payoff on monthly moves is convex: γ > 0
    daily_abs = mkt.abs() - mkt.abs().mean()
    cx = A.capture(daily_abs, mkt)
    assert cx["gamma"] > 0 and len(m) > 100


def test_legs_partition_pnl(mkt):
    pos = pd.Series(np.where(np.arange(len(mkt)) % 3 == 0, 1.0, np.where(np.arange(len(mkt)) % 3 == 1, -1.0, 0.0)),
                    index=mkt.index)
    lt = A.legs(mkt, pos, 252)
    assert lt["time"].sum() == pytest.approx(1.0)
    assert lt["share_of_pnl"].sum() == pytest.approx(1.0)


def test_regimes_cover_history(mkt):
    labels = A.regime_labels(mkt, 252)
    assert {"benchmark trend", "benchmark vol", "benchmark state"} <= set(labels)
    tbl = A.by_regime(mkt, labels, 252)
    assert set(tbl["dimension"]) == {"benchmark trend", "benchmark vol", "benchmark state"}


def test_factor_regression_recovers_loadings():
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(3)
    f = {f"US:{c}": pd.Series(rng.normal(0.005, 0.04, len(idx)), index=idx) for c in A.FACTOR_COLS}
    m = 0.8 * f["US:Mkt-RF"] + 0.3 * f["US:WML"] + rng.normal(0, 0.005, len(idx))
    daily_idx = pd.bdate_range(idx[0] - pd.offsets.MonthBegin(1), idx[-1])
    # spread each month's return over its first trading day so monthly compounding reproduces it
    r = pd.Series(0.0, index=daily_idx)
    for d, v in m.items():
        first = daily_idx[(daily_idx.year == d.year) & (daily_idx.month == d.month)][0]
        r[first] = v
    out = A.factor_regression(r, f, "US")
    co = dict(zip(out["names"], out["coef"]))
    assert co["Mkt-RF"] == pytest.approx(0.8, abs=0.05)
    assert co["WML"] == pytest.approx(0.3, abs=0.05)
    assert out["r2"] > 0.9


def test_sixty_forty_bounds():
    idx = pd.bdate_range("2020-01-01", periods=300)
    spy = pd.Series(np.linspace(100, 130, 300), index=idx)
    ief = pd.Series(np.linspace(100, 103, 300), index=idx)
    r = sixty_forty(spy, ief, idx)
    a, b = to_returns(spy, idx), to_returns(ief, idx)
    assert ((r <= np.maximum(a, b) + 1e-12) & (r >= np.minimum(a, b) - 1e-12)).all()


def test_benchmark_suggestions():
    assert suggest("Beat buy-and-hold", "SPY", "both", "Market")[0].key == "bh"
    assert suggest("Absolute return", "SPY", "both", "Market")[0].key == "cash"
    assert suggest("Crisis alpha / diversifier", "TLT", "both", "Market")[0].key == "SPY"
    keys = [s.key for s in suggest("Absolute return", "GLD", "long", "Market")]
    assert keys[1] == "bh" and len(keys) == len(set(keys))


def _facts(**kw):
    base = {"hypothesis": True, "trades": 80, "years": 12, "dsr": 0.95, "trials": 10, "n_filters": 0,
            "sharpe_2x": 0.6, "attr": {"alpha_ann": 0.05, "alpha_t": 2.6, "beta": 0.2, "beta_share": 0.2,
                                       "bench": "SPY"},
            "sensitivity": {"robustness": 0.85, "share_pos": 0.9}, "perm_p": 0.01, "noise_kept": 0.8,
            "wf_eff": 0.7, "oos": {"kept": 0.7, "oos_sharpe": 0.6, "is_sharpe": 0.85}}
    base.update(kw)
    return base


def test_verdict_all_good_is_A():
    tier, _ = V.grade(V.build(_facts()))
    assert tier == "A"


def test_verdict_failed_permutation_caps_at_C():
    tier, note = V.grade(V.build(_facts(perm_p=0.4)))
    assert tier == "C" and "Permutation" in note


def test_verdict_cannot_be_A_without_oos():
    f = _facts()
    f.pop("oos")
    tier, _ = V.grade(V.build(f))
    assert tier == "B"


def test_verdict_contaminated_oos_fails():
    checks = V.build(_facts(oos={"contaminated": True}))
    assert next(c for c in checks if c.name == "Out-of-sample").status == V.FAIL


def test_verdict_too_few_checks_is_C():
    f = _facts()
    for k in ("attr", "sensitivity", "perm_p", "noise_kept", "wf_eff", "oos"):
        f.pop(k)
    tier, note = V.grade(V.build(f))
    assert tier == "C" and "not enough testing" in note

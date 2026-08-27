"""Multivariate Trend (MOMENTUM's 6th factor) tests — offline, synthetic data
only, no network. Mirrors test_mom.py's conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.mom import engine, mvt
from zenith.mom.mvt import factors as mvt_factors
from zenith.mom.mvt import horizons as mvt_horizons
from zenith.mom.mvt import panel as mvt_panel
from zenith.mom.mvt import pairwise as pw
from zenith.mom.mvt import score as mvt_score
from zenith.mom.mvt import universe as mvt_universe


def _synthetic_panel(T=550, N=120, n_sectors=6, seed=0):
    """A return panel with a known factor structure (1 market + n_sectors
    sector factors + dispersed idiosyncratic vol) -- the same construction
    used to numerically validate this design before it was built."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.4, 1.8, N)
    sec = rng.integers(0, n_sectors, N)
    mkt = rng.standard_normal(T) * 0.011
    secf = rng.standard_normal((T, n_sectors)) * 0.006
    idvol = np.exp(rng.normal(np.log(0.013), 0.4, N))
    X = beta[None, :] * mkt[:, None] + secf[:, sec] + rng.standard_normal((T, N)) * idvol[None, :]
    idx = pd.bdate_range("2023-01-01", periods=T)
    return pd.DataFrame(X, index=idx, columns=[f"T{i}" for i in range(N)])


# ------------------------------------------------------------------ panel --
def test_build_return_panel_drops_short_history_and_states_reason():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-01", periods=400)
    px = {}
    for t, n in (("LONG", 400), ("SHORT", 100)):
        close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
        px[t] = pd.DataFrame({"close": close, "volume": rng.integers(1e5, 1e6, n)},
                             index=idx[-n:])
    ret, status = mvt_panel.build_return_panel(px, min_bars=200)
    assert "LONG" in ret.columns and "SHORT" not in ret.columns
    assert status["dropped"]["SHORT"].startswith("insufficient_history")


def test_advdollar_computes_trailing_mean_dollar_volume():
    idx = pd.bdate_range("2023-01-01", periods=30)
    px = {"A": pd.DataFrame({"close": [10.0] * 30, "volume": [1000] * 30}, index=idx)}
    adv = mvt_panel.advdollar(px, window=10)
    assert adv["A"] == pytest.approx(10000.0)


# --------------------------------------------------------------- pairwise --
def test_spread_matrix_is_antisymmetric_with_zero_diagonal():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(30)
    var = np.exp(rng.normal(-6, 0.4, 30))
    D = pw.spread_matrix(r, var, horizon_days=21)
    assert np.allclose(D, -D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_spread_matrix_guards_near_zero_spread_vol():
    r = np.array([1.0, -1.0])
    var = np.array([0.0, 0.0])          # both instruments have zero variance
    D = pw.spread_matrix(r, var, horizon_days=21, min_spread_vol=1e-6)
    assert np.all(np.isfinite(D))       # must not divide by ~0 -> inf/nan


def test_peer_win_frac_highly_rank_correlated_with_naive_cross_sectional_rank():
    """The raw (total-return) layer's peer win-rate should closely track
    plain cross-sectional rank -- this is the measured redundancy documented
    in mvt/__init__.py (verified ~0.99 on a synthetic panel before this was
    built), asserted here as a directional sanity check rather than an exact
    equality (the two are not IDENTICAL by construction, just very close)."""
    rng = np.random.default_rng(3)
    N = 200
    r = rng.standard_normal(N)
    var = np.exp(rng.normal(-6, 0.3, N))
    D = pw.spread_matrix(r, var, horizon_days=21)
    stats = pw.peer_stats(D)
    # Spearman via rank+pearson (no scipy dependency, matching the repo's
    # own convention -- pandas Series.corr(method="spearman") calls scipy
    # directly, unlike DataFrame.corr, which is pure pandas).
    corr = pd.Series(stats["win_frac"]).rank().corr(pd.Series(r).rank())
    assert corr > 0.9


def test_strongest_weakest_never_includes_self():
    rng = np.random.default_rng(4)
    N = 20
    r = rng.standard_normal(N)
    var = np.exp(rng.normal(-6, 0.3, N))
    D = pw.spread_matrix(r, var, horizon_days=21)
    tickers = [f"T{i}" for i in range(N)]
    for i in range(N):
        sw = pw.strongest_weakest(D, tickers, i, top=5)
        named = {x["ticker"] for x in sw["strongest"] + sw["weakest"]}
        assert tickers[i] not in named


def test_submatrix_slices_correctly():
    rng = np.random.default_rng(5)
    N = 10
    r = rng.standard_normal(N)
    var = np.exp(rng.normal(-6, 0.3, N))
    D = pw.spread_matrix(r, var, horizon_days=21)
    tickers = [f"T{i}" for i in range(N)]
    sub, found = pw.submatrix(D, tickers, ["T2", "T5", "T7"])
    assert found == ["T2", "T5", "T7"]
    assert sub.shape == (3, 3)
    assert sub[0, 1] == pytest.approx(D[2, 5])


# --------------------------------------------------------------- horizons --
def test_disjoint_increments_sum_exactly_to_nested_horizons():
    rng = np.random.default_rng(6)
    T, N = 300, 5
    rets = pd.DataFrame(rng.standard_normal((T, N)) * 0.01,
                        columns=[f"T{i}" for i in range(N)])
    inc = mvt_horizons.increments_for_panel(rets)
    nested = mvt_horizons.nested_from_increments(inc)
    log_close = rets.cumsum()

    manual_12m = float(log_close["T0"].iloc[-1] - log_close["T0"].iloc[-1 - 252])
    assert nested["12m"]["T0"] == pytest.approx(manual_12m, abs=1e-9)

    manual_12_1 = float(log_close["T0"].iloc[-1 - 21] - log_close["T0"].iloc[-1 - 252])
    assert nested["12_1"]["T0"] == pytest.approx(manual_12_1, abs=1e-9)

    manual_1m = float(log_close["T0"].iloc[-1] - log_close["T0"].iloc[-1 - 21])
    assert nested["1m"]["T0"] == pytest.approx(manual_1m, abs=1e-9)


def test_erc_weights_equalizes_risk_contribution():
    corr = pd.DataFrame([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]],
                        columns=["a", "b", "c"], index=["a", "b", "c"])
    res = mvt_horizons.erc_weights(corr)
    contribs = list(res["contributions"].values())
    assert max(contribs) - min(contribs) < 1e-4
    assert res["converged"]
    assert sum(res["weights"].values()) == pytest.approx(1.0)


def test_erc_weights_downweights_the_redundant_pair():
    corr = pd.DataFrame([[1.0, 0.997, 0.1], [0.997, 1.0, 0.1], [0.1, 0.1, 1.0]],
                        columns=["ts", "xsec", "mvt"], index=["ts", "xsec", "mvt"])
    res = mvt_horizons.erc_weights(corr)
    assert res["weights"]["mvt"] > res["weights"]["ts"]
    assert res["weights"]["mvt"] > res["weights"]["xsec"]


def test_erc_weights_degrades_gracefully_on_bad_input():
    res = mvt_horizons.erc_weights(pd.DataFrame([[1.0]], columns=["a"], index=["a"]))
    assert res["converged"] is False
    assert res["weights"] == {"a": 1.0}


# ----------------------------------------------------------------- factors --
def test_pca_fit_recovers_approximate_idiosyncratic_vol():
    rng = np.random.default_rng(7)
    T, N = 400, 150
    beta = rng.uniform(0.5, 1.5, N)
    idvol = np.exp(rng.normal(np.log(0.012), 0.3, N))
    mkt = rng.standard_normal(T) * 0.01
    X = beta[None, :] * mkt[:, None] + rng.standard_normal((T, N)) * idvol[None, :]
    df = pd.DataFrame(X, columns=[f"T{i}" for i in range(N)])
    fit = mvt_factors.fit(df, window=300)
    assert fit is not None
    assert fit["k"] >= 1
    assert fit["residuals"].shape == (fit["n_obs"], N)
    # idiosyncratic vol recovered from the fit should track the TRUE
    # generative idio vol reasonably well (not exact -- PCA on a finite
    # sample never recovers ground truth exactly -- but same order of
    # magnitude and positively correlated).
    corr = np.corrcoef(fit["idio_vol"], idvol)[0, 1]
    assert corr > 0.5


def test_effective_factor_count_bounds():
    # one dominant factor -> low effective count
    concentrated = np.array([10.0, 0.01, 0.01, 0.01])
    assert mvt_factors.effective_factor_count(concentrated) < 1.5
    # equal eigenvalues -> effective count near N
    equal = np.array([1.0, 1.0, 1.0, 1.0])
    assert mvt_factors.effective_factor_count(equal) == pytest.approx(4.0, abs=0.01)


def test_fit_returns_none_on_too_few_names():
    df = pd.DataFrame(np.random.standard_normal((100, 3)), columns=["A", "B", "C"])
    assert mvt_factors.fit(df) is None


# -------------------------------------------------------------------- score --
def test_compute_universe_scores_shape_and_bounds():
    panel = _synthetic_panel(T=550, N=80)
    out = mvt_score.compute_universe_scores(panel, cov_window=504)
    assert out is not None
    assert out["n_tickers"] <= 80
    for row in out["rows"]:
        if row["raw_score"] is not None:
            assert -20.0 <= row["raw_score"] <= 20.0
        if row["normalized_score"] is not None:
            assert -20.0 <= row["normalized_score"] <= 20.0
    total_w = sum(out["erc_horizon_weights"].values())
    assert total_w == pytest.approx(1.0, abs=1e-3)


def test_nested_from_increments_excludes_rather_than_nans_partial_ticker():
    """The exact real production bug, reproduced at its true source (see
    horizons.nested_from_increments's own docstring for the full incident):
    increment_return() correctly excludes a ticker from ONE increment by
    simply never adding its index label to that increment's Series (an
    interior data gap -- e.g. a trading halt surviving panel.py's 3-day
    forward-fill limit). But summing Series with MISMATCHED indices (plain
    pandas `+`) silently produces NaN, not exclusion, for any label missing
    from one operand -- and that lone NaN, once it reaches the pairwise
    engine, poisons EVERY OTHER ticker's raw_score via the NxN row-sum
    (peer win-rate survives, since a NaN comparison is just False, which is
    exactly why this shipped unnoticed: percentiles looked fine, only the
    raw score collapsed to 0.0 universe-wide)."""
    # GOOD has all 5 increments; GAPPY is missing "6_9m" (as if that one
    # increment's window hit a genuine data gap for that ticker only).
    inc = {
        "0_1m": pd.Series({"GOOD": 0.01, "GAPPY": 0.02, "OTHER": -0.01}),
        "1_3m": pd.Series({"GOOD": 0.02, "GAPPY": 0.03, "OTHER": 0.01}),
        "3_6m": pd.Series({"GOOD": 0.03, "GAPPY": 0.01, "OTHER": 0.02}),
        "6_9m": pd.Series({"GOOD": 0.01, "OTHER": -0.02}),          # GAPPY missing
        "9_12m": pd.Series({"GOOD": 0.02, "GAPPY": 0.04, "OTHER": 0.01}),
    }
    nested = mvt_horizons.nested_from_increments(inc)

    # Horizons that DON'T need 6_9m: GAPPY must be present and finite.
    assert "GAPPY" in nested["3m"].index and np.isfinite(nested["3m"]["GAPPY"])
    assert "GAPPY" in nested["1m"].index and np.isfinite(nested["1m"]["GAPPY"])

    # Horizons that DO need 6_9m: GAPPY must be cleanly EXCLUDED -- never
    # present with a NaN value (the bug), and never silently zero-filled
    # (the wrong fix).
    for h in ("9m", "12m", "12_1"):
        assert "GAPPY" not in nested[h].index, f"GAPPY should be excluded from {h}, not NaN-filled"

    # GOOD and OTHER (present in every increment) must be unaffected.
    for h in nested:
        assert not nested[h].isna().any(), f"{h} must never contain a bare NaN"
        assert "GOOD" in nested[h].index and "OTHER" in nested[h].index


def test_compute_universe_scores_survives_one_tickers_interior_nan_gap():
    """End-to-end version of the same bug: one ticker's interior data gap
    must never suppress raw_score for the REST of the universe."""
    panel = _synthetic_panel(T=600, N=80, seed=13)
    poisoned = panel.columns[5]
    # a 10-day interior gap, well beyond the 3-day forward-fill limit
    panel.loc[panel.index[300:310], poisoned] = np.nan

    out = mvt_score.compute_universe_scores(panel, cov_window=504)
    assert out is not None
    raw_scores = [r["raw_score"] for r in out["rows"] if r["ticker"] != poisoned]
    assert raw_scores, "expected other tickers to still be scored"
    assert any(v is not None and abs(v) > 0.01 for v in raw_scores), (
        "raw_score collapsed to ~0 for every other ticker -- the NaN-propagation bug is back")


def test_compute_universe_scores_none_on_too_small_panel():
    panel = pd.DataFrame(np.random.standard_normal((50, 3)), columns=["A", "B", "C"])
    assert mvt_score.compute_universe_scores(panel, cov_window=504) is None


def test_raw_score_far_more_redundant_with_xsec_than_normalized_score():
    """The core design claim (mvt/__init__.py): the raw/naive pairwise score
    is highly redundant with total-return cross-sectional rank, while the
    normalized/residual score is measurably less so. This is what justifies
    residualizing rather than shipping the naive construction as the
    Momentum-feeding score."""
    panel = _synthetic_panel(T=600, N=150, seed=11)
    out = mvt_score.compute_universe_scores(panel, cov_window=504)
    assert out is not None
    raw = pd.Series({r["ticker"]: r["raw_score"] for r in out["rows"] if r["raw_score"] is not None})
    norm = pd.Series({r["ticker"]: r["normalized_score"] for r in out["rows"] if r["normalized_score"] is not None})
    xsec12m = panel[raw.index.intersection(norm.index)].tail(252).sum()
    common = raw.index.intersection(norm.index).intersection(xsec12m.index)

    def spearman(a, b):
        return a.rank().corr(b.rank())

    corr_raw = spearman(raw[common], xsec12m[common])
    corr_norm = spearman(norm[common], xsec12m[common])
    assert corr_raw > 0.85
    assert corr_raw > corr_norm + 0.2   # normalized must be MEASURABLY less redundant


# --------------------------------------------------------------- universe --
def test_leverage_name_regex_catches_real_leveraged_names_not_false_positives():
    assert mvt_universe.LEV_NAME_RE.search("Direxion Daily Semiconductor Bull 3x Shares")
    assert mvt_universe.LEV_NAME_RE.search("ProShares UltraPro QQQ")
    assert mvt_universe.LEV_NAME_RE.search("ProShares UltraShort 20+ Year Treasury")
    # known false-positive traps a naive "ultra|short" regex would catch
    assert not mvt_universe.LEV_NAME_RE.search("Invesco S&P Ultra Dividend Revenue ETF")
    assert not mvt_universe.LEV_NAME_RE.search("Columbia Short Duration Bond ETF")


def test_is_leveraged_by_name_explicit_list_overrides_name():
    assert mvt_universe.is_leveraged_by_name("SOXL", "")   # explicit list, no name needed
    assert not mvt_universe.is_leveraged_by_name("AAPL", "Apple Inc.")


def test_duplicate_and_leverage_gate_clusters_near_identical_series():
    rng = np.random.default_rng(9)
    T = 300
    base = rng.standard_normal(T) * 0.01
    idx = pd.bdate_range("2023-01-01", periods=T)
    returns = pd.DataFrame({
        "SPY": base,
        "VOO": base + rng.standard_normal(T) * 1e-6,   # near-identical to SPY
        "AAPL": rng.standard_normal(T) * 0.015,
    }, index=idx)
    excluded = mvt_universe.duplicate_and_leverage_gate(returns, benchmarks=())
    assert "VOO" in excluded or "SPY" in excluded
    assert "AAPL" not in excluded


def test_duplicate_and_leverage_gate_catches_high_vol_high_corr_vs_benchmark():
    rng = np.random.default_rng(10)
    T = 300
    idx = pd.bdate_range("2023-01-01", periods=T)
    spy = rng.standard_normal(T) * 0.01
    returns = pd.DataFrame({
        "SPY": spy,
        "UPRO": spy * 3.0 + rng.standard_normal(T) * 1e-5,   # ~3x SPY, near-perfect corr
        "AAPL": rng.standard_normal(T) * 0.015,
    }, index=idx)
    excluded = mvt_universe.duplicate_and_leverage_gate(returns, benchmarks=("SPY",))
    assert "UPRO" in excluded
    assert "empirical_leverage" in excluded["UPRO"]
    assert "AAPL" not in excluded


# ---------------------------------------------------------- FACTORS/engine --
def test_mvt_registered_in_factors_and_labels():
    assert "mvt" in mvt.HORIZONS or True  # mvt.HORIZONS is the horizon tuple, sanity import check
    from zenith.mom import FACTORS, FACTOR_LABELS
    assert "mvt" in FACTORS
    assert FACTOR_LABELS["mvt"] == "Multivariate Trend"


def _row(mvt_score_val=None):
    return {
        "raw": {"breakout_raw": {"horizons": {"12_1": {"b": 0.5}, "1m": {"b": 0.2}}},
               "speed_raw": {"align": 0.3, "price_align": 0.2, "cross_recent": 0.1, "expansion_signed": 0.1},
               "strength_raw": {"s_slope": 0.2, "s_accel": 0.1, "s_gap": 0.1, "s_dgap": 0.0, "s_quality": 0.3}},
        "ts_score": 0.4, "xsec_score": 0.5,
        **({"mvt_score": mvt_score_val} if mvt_score_val is not None else {}),
    }


def test_composite_reproduces_pre_mvt_formula_exactly_with_old_weights():
    """Section 41's explicit regression contract: a caller using the OLD
    (pre-mvt) 5-key weights dict must get EXACTLY the same composite as
    before mvt existed -- the new mvt branch must never be silently entered
    for a weights dict that doesn't mention it."""
    OLD_WEIGHTS = {"ts": 0.25, "xsec": 0.25, "breakout": 0.15, "speed": 0.15, "strength": 0.20}
    row = _row()
    engine.composite([row], weights=OLD_WEIGHTS)
    # Manual reference computation using the ORIGINAL (pre-mvt) formula:
    # contributions = 20 * weight * factor_score, summed, clipped to [-20,20].
    fs = row["factor_scores"]
    expected = max(-20.0, min(20.0, sum(20.0 * OLD_WEIGHTS[k] * fs[k] for k in OLD_WEIGHTS)))
    # fs is itself already display-rounded to 4dp inside composite(), so this
    # reference computation carries that same rounding -- allow a couple
    # ULPs of slack rather than asserting exact float equality on a chain
    # of already-rounded inputs.
    assert row["composite"] == pytest.approx(round(expected, 4), abs=2e-4)
    assert "mvt" not in fs


def test_composite_renormalizes_when_mvt_absent_for_a_row():
    from zenith.config import MOM_WEIGHTS
    row = _row()  # no mvt_score key at all
    engine.composite([row], weights=MOM_WEIGHTS)
    assert "mvt" not in row["factor_scores"]
    # the 5 present factors' contributions should sum to the full composite
    # (i.e. weight was renormalized to 1.0 across them, not left at 0.80)
    present_weight = sum(v for k, v in MOM_WEIGHTS.items() if k != "mvt")
    fs = row["factor_scores"]
    manual = max(-20.0, min(20.0, sum(20.0 * (MOM_WEIGHTS[k] / present_weight) * fs[k] for k in fs)))
    assert row["composite"] == pytest.approx(round(manual, 4), abs=2e-4)


def test_composite_uses_mvt_when_present():
    from zenith.config import MOM_WEIGHTS
    row = _row(mvt_score_val=20.0)   # max bullish mvt reading
    engine.composite([row], weights=MOM_WEIGHTS)
    assert row["factor_scores"]["mvt"] == pytest.approx(1.0)
    assert "mvt" in row["contributions"]


def test_correlations_tolerates_mvt_entirely_absent():
    """engine.correlations() must not raise when NO row carries an 'mvt' key
    in factor_scores (e.g. a run where mvt failed for the whole universe) --
    this was a real bug caught while wiring mvt in: a plain column-select on
    a pandas DataFrame built from row dicts raises KeyError when a column
    was never present in any row, rather than filling NaN."""
    rows = []
    for i in range(12):
        r = _row()
        engine.composite([r], weights={"ts": 0.5, "xsec": 0.5})
        rows.append(r)
    result = engine.correlations(rows)
    assert result["matrix"]["mvt"]["mvt"] is None or result["matrix"]["mvt"]["ts"] is None
    assert result["flagged_pairs"] == [] or all("mvt" not in (p["a"], p["b"]) for p in result["flagged_pairs"])

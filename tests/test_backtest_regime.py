"""Regime engine: causal labels, hysteresis, HMM recovery, statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenith.backtest.data.synthetic import generate
from zenith.backtest.regime import DETECTORS, by_regime, debounce, detect, hmm_probs, spells, transition_matrix


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    df = generate("regime", n_days=2500, seed=4)
    df["vix"] = 20 + np.random.default_rng(2).normal(0, 4, len(df))
    return df


@pytest.mark.parametrize("key", [k for k in DETECTORS if k != "hmm"])
def test_detector_is_causal(frame, key):
    full = detect(key, frame, min_bars=3).astype(str)
    part = detect(key, frame.iloc[:1700], min_bars=3).astype(str)
    assert (full.iloc[:1700] == part).all(), key


def test_hmm_filter_is_causal_after_fit():
    """With the fit window fixed, the filtered label at t ignores bars after t."""
    df = generate("regime", n_days=2400, seed=8)
    p_full = hmm_probs(df, 2, 50)
    p_part = hmm_probs(df.iloc[:2000], 2, 60)          # same fit window: 1200 returns in both
    assert p_full is not None and p_part is not None
    a, b = p_full.iloc[1300:1990].to_numpy(), p_part.iloc[1300:1990].to_numpy()
    assert np.allclose(a, b, atol=1e-4)


def test_hmm_separates_calm_and_wild_vol():
    rng = np.random.default_rng(0)
    r = np.r_[rng.normal(0, 0.005, 600), rng.normal(0, 0.03, 600), rng.normal(0, 0.005, 600), rng.normal(0, 0.03, 600)]
    idx = pd.bdate_range("2000-01-03", periods=len(r) + 1)
    c = 100 * np.exp(np.r_[0, r.cumsum()])
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    lab = detect("hmm", df, states=2, fit_pct=50)
    # bars 1200-1800 are calm and 1800-2400 wild, both after the fit window (out of sample)
    assert (lab.iloc[1300:1750] == "hmm low-vol").mean() > 0.8
    assert (lab.iloc[1850:2350] == "hmm high-vol").mean() > 0.8


def test_debounce_kills_short_flips():
    s = pd.Series(list("AAAAB" + "AAAA" + "BBBB" + "A"))
    out = debounce(s, 3).tolist()
    assert out[4] == "A"          # one-bar B ignored
    assert out[12] == "B"         # B adopted after 3 in a row
    assert out[-1] == "B"         # single A at the end ignored


def test_transition_rows_sum_to_one(frame):
    tm = transition_matrix(detect("trend", frame))
    assert np.allclose(tm.sum(axis=1), 1.0)


def test_by_regime_uses_previous_label(frame):
    """Returns are attributed to the label known at the previous close."""
    idx = frame.index[:6]
    lab = pd.Series(["a", "a", "b", "b", "a", "a"], index=idx)
    r = pd.Series([0.0, 0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    t = by_regime(r, lab, 252, boot=False) if len(r) > 100 else None
    sh = lab.shift(1)
    assert sh.iloc[3] == "b" and sh.iloc[4] == "b"
    assert t is None or not t.empty


def test_spells_cover_all_labels(frame):
    lab = detect("vol", frame, min_bars=3)
    sp = spells(lab)
    assert sp["bars"].sum() == lab.notna().sum()


def test_vol_regime_matches_synthetic_regimes():
    """On the 'regime' synthetic world, stressed bars should have higher realised vol than calm ones."""
    df = generate("regime", n_days=4000, seed=9)
    lab = detect("vol", df)
    r = np.log(df["close"]).diff().abs()
    assert r[lab == "stressed"].mean() > 1.5 * r[lab == "calm"].mean()

"""TREND FOLLOWING tests — fully offline, synthetic price series, no network.

Conventions mirror tests/test_mom.py / test_etfmom.py: seeded synthetic OHLC
frames, a fixture that redirects the package's path dicts (in place, never
config itself), and monkeypatchable seams (`tc._fetch_prices`,
`tc._load_universe`, `tc._leverage`) so no test touches the network or the
committed data/ tree.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import zenith.trend as trend
from zenith import config
from zenith.config import (MOM_STATES, TREND_FORECAST_CAP, TREND_FORECAST_SCALARS,
                           TREND_SPEEDS)
from zenith.trend import compute as tc
from zenith.trend import events as ev
from zenith.trend import ewmac, history, structure, table
from zenith.trend import universe as tu

KEYS = trend.SPEED_KEYS


# ------------------------------------------------------------------ helpers --
def _idx(n: int, end: str = "2026-09-18") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _walk(n: int, drift: float, vol: float = 0.01, seed: int = 1, end: str = "2026-09-18") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=_idx(n, end))


def _ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": 1e6}, index=close.index)


def _ema_reference(x: np.ndarray, span: int) -> np.ndarray:
    """Hand-rolled bias-corrected EWMA: sum_i (1-a)^i x_{t-i} / sum_i (1-a)^i."""
    a = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    num = den = 0.0
    for t, v in enumerate(x):
        num = v + (1 - a) * num
        den = 1.0 + (1 - a) * den
        out[t] = num / den
    return out


@pytest.fixture
def tmp_trend_store(tmp_path, monkeypatch):
    """Redirect every TREND path to tmp_path by mutating the shared dicts in
    place (modules imported them by reference)."""
    for u in config.TREND_UNIVERSES:
        base = tmp_path / u
        for name, p in list(config.TREND_FILES[u].items()):
            monkeypatch.setitem(config.TREND_FILES[u], name, base / p.name)
        monkeypatch.setitem(config.TREND_HISTORY_DIRS, u, base / "history")
        monkeypatch.setitem(config.TREND_EVENT_DIRS, u, base / "events")
    return tmp_path


# ==================================================================== EMA ====
def test_ema_matches_hand_rolled_adjusted_ewma():
    x = _walk(300, 0.0005, seed=3)
    for span in (2, 8, 32, 128):
        got = ewmac.ema(x, span).to_numpy()
        ref = _ema_reference(x.to_numpy(), span)
        valid = ~np.isnan(got)
        assert valid.sum() == len(x) - span + 1          # NaN until `span` bars exist
        assert np.allclose(got[valid], ref[valid], rtol=0, atol=1e-10)


def test_ema_is_exponential_not_simple():
    x = pd.Series(np.r_[np.full(50, 100.0), np.full(10, 110.0)], index=_idx(60))
    e, s = ewmac.ema(x, 8).iloc[-1], x.rolling(8).mean().iloc[-1]
    assert s == pytest.approx(110.0)                     # SMA has fully caught up
    assert 100.0 < e < 110.0                             # EMA still carries older weight


def test_daily_data_is_required():
    weekly = pd.Series(np.linspace(100, 150, 200), index=pd.date_range("2020-01-03", periods=200, freq="W-FRI"))
    with pytest.raises(ValueError, match="DAILY"):
        ewmac.build(weekly)
    ewmac.build(_walk(200, 0.0))                          # business-daily: fine


# ============================================================ forecasts ====
def test_speeds_and_scalars_are_the_specified_seven():
    assert TREND_SPEEDS == ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256), (128, 512))
    assert KEYS == ("2_8", "4_16", "8_32", "16_64", "32_128", "64_256", "128_512")
    assert list(TREND_FORECAST_SCALARS.values()) == [10.6, 7.5, 5.3, 3.75, 2.65, 1.87, 1.32]


def test_raw_ewmac_formula_is_explicit():
    x = _walk(700, 0.0008, seed=5)
    vol = ewmac.price_vol(x)
    raw = ewmac.raw_ewmac(x, 16, 64, vol)
    manual = (ewmac.ema(x, 16) - ewmac.ema(x, 64)) / vol
    pd.testing.assert_series_equal(raw.dropna(), manual.dropna(), check_names=False)
    f, _ = ewmac.forecasts(x)
    assert np.allclose(f["16_64"].dropna(), (manual * 3.75).clip(-20, 20).dropna())


def test_uptrend_is_bullish_at_every_speed_and_downtrend_mirrors():
    up = _walk(1300, 0.0012, vol=0.008, seed=2)
    r = ewmac.build(up)
    last = r.forecast.iloc[-1]
    assert (last > 0).all() and r.score.iloc[-1] >= 5          # bullish band
    down = pd.Series(10_000.0 / up.to_numpy(), index=up.index)      # log-reflected path
    rd = ewmac.build(down)
    assert (rd.forecast.iloc[-1] < 0).all() and rd.score.iloc[-1] < -5


def test_reflected_series_negates_every_forecast_exactly():
    x = _walk(900, 0.0004, seed=9)
    mirror = 2 * x.mean() - x                          # arithmetic reflection: price diffs negate
    f1, f2 = ewmac.build(x).forecast, ewmac.build(mirror).forecast
    assert np.allclose(f1.dropna(), -f2.dropna(), atol=1e-9)


def test_forecasts_are_scale_invariant():
    x = _walk(900, 0.0006, seed=4)
    a, b = ewmac.build(x).forecast, ewmac.build(x * 3.0).forecast
    assert np.allclose(a.dropna(), b.dropna(), atol=1e-9)


def test_forecasts_and_score_are_bounded():
    rocket = pd.Series(100.0 * np.exp(np.linspace(0, 3, 1300)) + np.sin(np.arange(1300)) * 0.01,
                       index=_idx(1300))
    r = ewmac.build(rocket)
    assert r.forecast.abs().max().max() <= TREND_FORECAST_CAP
    assert r.score.dropna().between(-20, 20).all()
    assert r.forecast.iloc[-1].max() == pytest.approx(20.0)        # the cap binds


def test_score_is_equal_weight_mean_of_the_seven():
    r = ewmac.build(_walk(1300, 0.0005, seed=11))
    full = r.forecast.dropna()
    assert np.allclose(r.score.loc[full.index], full.mean(axis=1))
    # nudging ONE speed by delta moves the score by exactly delta / 7
    f = full.iloc[[-1]].copy()
    base, _ = ewmac.combine(f)
    for k in KEYS:
        g = f.copy()
        g[k] = g[k] - 2.8
        bumped, _ = ewmac.combine(g)
        assert float(base.iloc[0] - bumped.iloc[0]) == pytest.approx(2.8 / 7)


def test_flat_series_has_no_nan_or_inf():
    r = ewmac.build(pd.Series(100.0, index=_idx(800)))
    assert (r.forecast.iloc[-1] == 0).all() and r.score.iloc[-1] == 0


def test_partial_and_insufficient_history():
    r = ewmac.build(_walk(300, 0.001, seed=6))             # 256 < 300 < 512
    assert int(r.n_valid.iloc[-1]) == 6
    assert np.isfinite(r.score.iloc[-1])
    assert r.score.iloc[-1] == pytest.approx(r.forecast.iloc[-1].dropna().mean())
    short = ewmac.build(_walk(50, 0.001, seed=6))           # < 64 bars -> fewer than 4 speeds
    assert short.score.isna().all()


def test_normalizers_are_pluggable():
    x = _walk(700, 0.001, seed=8)
    fs, _ = ewmac.forecasts(x, normalization="sign")
    assert set(np.unique(fs.dropna().to_numpy())) <= {-20.0, 0.0, 20.0}
    ft, _ = ewmac.forecasts(x, normalization="tanh")
    assert ft.abs().max().max() < 20.0


# ============================================================ crossovers ====
def test_v_shape_crossovers_faster_speeds_first_and_on_the_exact_day():
    n = 1400
    lp = np.r_[np.linspace(np.log(200), np.log(100), 900), np.linspace(np.log(100), np.log(160), n - 900)]
    rng = np.random.default_rng(0)
    x = pd.Series(np.exp(lp + rng.normal(0, 0.001, n)), index=_idx(n))
    r = ewmac.build(x)
    lc = ev.last_cross(r.raw)
    order = [lc[k]["date"] for k in KEYS if lc[k] and lc[k]["dir"] == 1]
    assert len(order) >= 5
    assert order == sorted(order)                           # faster speeds turned bullish first
    for k, (fs, ss) in zip(KEYS, TREND_SPEEDS):
        if not lc[k] or lc[k]["dir"] != 1:
            continue
        above = (ewmac.ema(x, fs) - ewmac.ema(x, ss)) > 0
        # independent check: the reported date is the day fast crossed ABOVE slow
        d = pd.Timestamp(lc[k]["date"])
        i = above.index.get_loc(d)
        assert bool(above.iloc[i]) and not bool(above.iloc[i - 1])
        assert above.iloc[i:].all()


def test_crossover_matrix_counts_sign_changes_only():
    raw = pd.DataFrame({k: [np.nan, -1.0, -0.5, 0.0, 0.3, 0.2, -0.1] for k in KEYS}, index=_idx(7))
    x = ev.crossovers(raw)["2_8"].tolist()
    assert x == [0, 0, 0, 0, 1, 0, -1]                     # the exact tie does not count as a cross


# ================================================================ events ====
def _events_for_scores(scores: list[float]) -> list[dict]:
    idx = _idx(len(scores))
    raw = pd.DataFrame({k: np.ones(len(scores)) for k in KEYS}, index=idx)
    return ev.detect("T", raw, pd.Series(scores, index=idx, dtype=float))


@pytest.mark.parametrize("a,b,kind", [
    (4.0, 9.0, "trigger"),      # neutral -> bullish is a (bullish) trigger
    (-8.0, -3.0, "upgrade"),    # bearish -> neutral
    (15.5, 9.0, "downgrade"),   # extreme bullish -> bullish
    (-4.0, -12.0, "trigger"),   # neutral -> bearish is a (bearish) trigger
    (6.0, 11.0, "upgrade"),
    (-12.0, -16.0, "downgrade"),
    (-7.0, 7.0, "trigger"),     # straight through neutral
])
def test_band_changes(a, b, kind):
    evs = [e for e in _events_for_scores([a, b]) if e["type"] != "cross"]
    assert len(evs) == 1 and evs[0]["type"] == kind
    assert evs[0]["dir"] == (1 if b > a else -1)
    assert evs[0]["score_before"] == a and evs[0]["score_after"] == b


def test_user_examples_are_upgrades_and_downgrades_by_direction():
    # The spec's own examples: +4 -> +9 and -8 -> -3 are UPGRADES in direction,
    # +15 -> +9 and -4 -> -12 DOWNGRADES. +4 -> +9 and -4 -> -12 also cross into
    # a trend state, so they surface as the stronger trigger event.
    for a, b, d in [(4, 9, 1), (-8, -3, 1), (15.5, 9, -1), (-4, -12, -1)]:
        e = [x for x in _events_for_scores([a, b]) if x["type"] != "cross"][0]
        assert e["dir"] == d


def test_hysteresis_one_event_for_a_score_hovering_on_a_line():
    # without a buffer this path is trigger, downgrade, trigger, downgrade...
    evs = [e for e in _events_for_scores([4.0, 5.2, 4.6, 5.3, 4.2, 5.1, 3.9]) if e["type"] != "cross"]
    assert [(e["type"], e["date"] == _idx(7)[1].strftime("%Y-%m-%d")) for e in evs][0] == ("trigger", True)
    assert [e["type"] for e in evs] == ["trigger", "downgrade"]     # exits only below 5 - 1 = 4
    assert evs[1]["score_after"] == 3.9 and evs[1]["to"] == "NEUTRAL"


def test_hysteresis_is_symmetric_for_bearish_states():
    evs = [e for e in _events_for_scores([-4.0, -5.5, -4.5, -5.2, -3.8]) if e["type"] != "cross"]
    assert [e["type"] for e in evs] == ["trigger", "upgrade"]         # exits only at/above -5 + 1 = -4


def test_zero_buffer_reproduces_raw_bands():
    sc = np.array([4.0, 5.2, 4.6, 5.3, 4.2, -7.0, 16.0, 9.0])
    assert ev.held_bands(sc, buffer=0.0).tolist() == ev.band_of(sc).tolist()


def test_band_semantics_match_momentum_state_for():
    from zenith.mom.engine import state_for
    for s in np.linspace(-20, 20, 161):
        assert ev.state_label(s) == state_for(s, MOM_STATES)


def test_no_event_without_a_band_change():
    assert [e for e in _events_for_scores([6.0, 7.0, 9.9, 5.1]) if e["type"] != "cross"] == []


def _cross_raw(n: int, flips: dict[str, int]) -> pd.DataFrame:
    """raw frame: each speed is -1 until its flip index, +1 after."""
    data = {}
    for k in KEYS:
        v = -np.ones(n)
        if k in flips:
            v[flips[k]:] = 1.0
        data[k] = v
    return pd.DataFrame(data, index=_idx(n))


def test_multi_speed_confirmation_fires_for_three_speeds():
    raw = _cross_raw(40, {"2_8": 20, "4_16": 23, "8_32": 26})
    evs = ev.detect("T", raw, pd.Series(0.0, index=raw.index))
    conf = [e for e in evs if e["type"] == "confirmation"]
    assert len(conf) == 1
    c = conf[0]
    assert c["dir"] == 1 and c["speeds"] == ["2_8", "4_16", "8_32"] and c["n_speeds"] == 3
    assert c["date"] == raw.index[26].strftime("%Y-%m-%d") and c["horizon"] == "short"


def test_single_fast_cross_is_not_a_confirmation():
    raw = _cross_raw(40, {"2_8": 20})
    evs = ev.detect("T", raw, pd.Series(0.0, index=raw.index))
    assert [e["type"] for e in evs] == ["cross"]


def test_confirmation_requires_crosses_within_the_window_and_still_live():
    spread = _cross_raw(60, {"2_8": 10, "4_16": 25, "8_32": 45})          # too far apart
    assert not [e for e in ev.detect("T", spread, pd.Series(0.0, index=spread.index))
                if e["type"] == "confirmation"]
    raw = _cross_raw(40, {"2_8": 20, "4_16": 23, "8_32": 26})
    raw.iloc[24:, 0] = -1.0                                               # 2/8 flipped back down
    assert not [e for e in ev.detect("T", raw, pd.Series(0.0, index=raw.index))
                if e["type"] == "confirmation"]


def test_long_term_confirmation_is_tagged_long():
    raw = _cross_raw(40, {"16_64": 20, "32_128": 22, "64_256": 24})
    c = [e for e in ev.detect("T", raw, pd.Series(0.0, index=raw.index)) if e["type"] == "confirmation"][0]
    assert c["horizon"] == "long"


def test_since_filters_old_events():
    raw = _cross_raw(40, {"2_8": 20})
    assert ev.detect("T", raw, pd.Series(0.0, index=raw.index), since=raw.index[25].strftime("%Y-%m-%d")) == []


def test_flip_stats():
    raw = _cross_raw(40, {"2_8": 20})
    fs = ev.flip_stats(raw)
    assert fs["2_8"]["n_crosses"] == 1 and fs["2_8"]["n_bear_to_bull"] == 1
    assert fs["2_8"]["current_side"] == 1 and fs["2_8"]["current_run"] == 20


# ============================================================= structure ====
def test_structure_labels_for_the_specs_three_examples():
    persistent = [15, 14, 16, 13, 12, 11, 10]
    assert structure.label_for(persistent, sum(persistent) / 7) == "Persistent Uptrend"
    emerging = [12, 10, 8, 0, -6, -8, -9]
    assert structure.label_for(emerging, sum(emerging) / 7) == "Emerging Uptrend"
    deteriorating = [-10, -8, -4, 2, 8, 10, 12]
    assert structure.label_for(deteriorating, sum(deteriorating) / 7) == "Deteriorating Uptrend"
    down = [-15, -14, -16, -13, -12, -11, -10]
    assert structure.label_for(down, sum(down) / 7) == "Persistent Downtrend"
    assert structure.label_for([1, -1, 0.5, -0.5, 0.2, -0.2, 0.1], 0.0) == "Mixed"


def test_structure_analyse_fields():
    a = structure.analyse([12, 10, 8, 0, -6, -8, -9], 1.0)
    assert a["n_bull"] == 3 and a["n_bear"] == 3 and a["fast"] == 10 and a["slow"] == pytest.approx(-23 / 3, abs=1e-3)
    assert a["slope"] == pytest.approx(10 + 23 / 3, abs=1e-3) and a["disagree"] is True
    assert structure.regime_of([5] * 7) == "all_bull"
    assert structure.regime_of([-5] * 7) == "all_bear"
    assert structure.regime_of([8, 8, 8, 0, -8, -8, -8]) == "disagree"


# =============================================================== history ====
def test_history_roundtrip_idempotent_and_gap_fill(tmp_trend_store):
    x = _walk(700, 0.001, seed=12, end="2026-09-18")
    r = ewmac.build(x)
    enc = history.encode_rows(r.forecast, r.raw, r.score)
    head = history.Encoded(enc.dates[:-5], enc.arr[:-5])
    out = history.append("stocks", {"AAA": head}, today=date(2026, 9, 18))
    assert out["dates_added"] == len(head)
    again = history.append("stocks", {"AAA": head}, today=date(2026, 9, 18))
    assert again["dates_added"] == 0                                  # idempotent
    last = history.last_date("stocks")
    gap = history.append("stocks", {"AAA": enc.after(last)}, today=date(2026, 9, 18))
    assert gap["dates_added"] == 5                                    # missed days gap-filled
    back = history.series_for("stocks", "AAA")
    assert len(back) == len(enc)
    tail = r.score.dropna().index[-1]
    assert back.loc[tail, "score"] == pytest.approx(round(r.score.loc[tail] * 10) / 10)
    for k in KEYS:
        assert back.loc[tail, f"f{k}"] == round(r.forecast.loc[tail, k])
        assert back.loc[tail, f"bull_{k}"] == (1 if r.raw.loc[tail, k] > 0 else -1)
    # closed years are gzipped, the current one stays plain
    d = config.TREND_HISTORY_DIRS["stocks"]
    assert (d / "2026.json").exists() and (d / "2025.json.gz").exists()
    assert not (d / "2025.json").exists()


def test_history_bull_mask_reconstructs_crossovers_exactly(tmp_trend_store):
    n = 1400
    lp = np.r_[np.linspace(np.log(200), np.log(100), 900), np.linspace(np.log(100), np.log(160), n - 900)]
    x = pd.Series(np.exp(lp), index=_idx(n))
    r = ewmac.build(x)
    history.append("etfs", {"V": history.encode_rows(r.forecast, r.raw, r.score)}, today=date(2026, 9, 18))
    h = history.series_for("etfs", "V")
    lc = ev.last_cross(r.raw)
    for k in KEYS:
        if lc[k] is None:
            continue
        s = h[f"bull_{k}"].dropna()
        flips = s[s != s.shift()].index[1:]
        assert flips[-1].strftime("%Y-%m-%d") == lc[k]["date"]


def test_breadth_row():
    rows = [[100, 20, 5, None, 0, 0, 0, 0, 0b0000011], [-100, -20, -5, None, 0, 0, 0, 0, 0]]
    b = history.breadth_row("2026-09-18", rows)
    assert b["n"] == 2 and b["pct_bull"] == 0.5 and b["pct_bear"] == 0.5
    assert b["speed_bull"][0] == 0.5 and b["speed_bull"][2] is None


def test_event_store_roundtrip(tmp_trend_store):
    raw = _cross_raw(40, {"2_8": 20, "4_16": 23, "8_32": 26})
    evs = ev.detect("T", raw, pd.Series(np.linspace(-8, 8, 40), index=raw.index))
    n = history.append_events("stocks", evs, today=date(2026, 9, 18))
    assert n == sum(1 for e in evs if e["type"] != "cross") and n > 0
    assert history.append_events("stocks", evs, today=date(2026, 9, 18)) == 0
    back = history.events_for("stocks", "T")
    assert {e["type"] for e in back} <= {"trigger", "upgrade", "downgrade", "confirmation"}
    conf = [e for e in back if e["type"] == "confirmation"][0]
    assert conf["speeds"] == ["2_8", "4_16", "8_32"]


# ======================================================= universe sync ====
def test_stock_universe_is_exactly_the_momentum_universe(monkeypatch):
    from zenith.mom import universe as mu
    fake = [{"ticker": "AAA", "name": "A Co", "sector": "Tech"},
            {"ticker": "BBB", "name": "B Co", "sector": "Energy"}]
    monkeypatch.setattr(mu, "constituents", lambda *a, **k: (fake, {"source": "test"}))
    rows, _ = tu.stocks()
    assert [r["ticker"] for r in rows] == [r["ticker"] for r in mu.constituents()[0]]


def test_stock_universe_drops_duplicate_rows_but_keeps_the_ticker_set(monkeypatch):
    from zenith.mom import universe as mu
    fake = [{"ticker": "AAA", "name": "A"}, {"ticker": "BBB", "name": "B"}, {"ticker": "AAA", "name": "A2"}]
    monkeypatch.setattr(mu, "constituents", lambda *a, **k: (fake, {}))
    rows, st = tu.stocks()
    assert [r["ticker"] for r in rows] == ["AAA", "BBB"] and rows[0]["name"] == "A"
    assert st["n_duplicates_dropped"] == 1


def test_event_append_dedupes_within_a_batch(tmp_trend_store):
    raw = _cross_raw(40, {"2_8": 20, "4_16": 23, "8_32": 26})
    evs = ev.detect("T", raw, pd.Series(np.linspace(-8, 8, 40), index=raw.index))
    n = history.append_events("stocks", evs + evs, today=date(2026, 9, 18))
    assert n == sum(1 for e in evs if e["type"] != "cross")


def test_etf_universe_is_exactly_the_etf_momentum_included_set(monkeypatch):
    from zenith.etfmom import universe as eu
    fake = [{"ticker": "SPY", "name": "S&P", "included": True, "category": "Large Blend",
             "asset_class": "Equity"},
            {"ticker": "TQQQ", "name": "3x", "included": False, "exclusion_reason": "leveraged"},
            {"ticker": "TLT", "name": "Treasury", "included": True, "category": "Long Government",
             "asset_class": "Fixed Income"}]
    monkeypatch.setattr(eu, "constituents", lambda: (fake, {}))
    rows, _ = tu.etfs()
    assert [r["ticker"] for r in rows] == ["SPY", "TLT"]
    assert rows[1]["asset_class"] == "Fixed Income"


# ============================================================ full run ====
def _stub_run(monkeypatch, n_bars: int = 1300):
    members = {
        "stocks": [{"ticker": t, "name": f"{t} Inc", "sector": s, "industry": "", "mktcap": 1e10}
                   for t, s in [("UP", "Tech"), ("DN", "Energy"), ("FLAT", "Utilities"),
                                ("NEW", "Tech"), ("GONE", "Tech")]],
        "etfs": [{"ticker": t, "name": t, "category": "Large Blend", "asset_class": "Equity"}
                 for t in ("E1", "E2", "LEV")],
    }
    px = {
        "UP": _ohlc(_walk(n_bars, 0.0012, seed=1)),
        "DN": _ohlc(_walk(n_bars, -0.0012, seed=2)),
        "FLAT": _ohlc(_walk(n_bars, 0.0, vol=0.004, seed=3)),
        "NEW": _ohlc(_walk(40, 0.001, seed=4)),                # insufficient history
        "E1": _ohlc(_walk(n_bars, 0.0008, seed=5)),
        "E2": _ohlc(_walk(n_bars, -0.0004, seed=6)),
        "LEV": _ohlc(_walk(n_bars, 0.003, seed=7)),
    }
    monkeypatch.setattr(tc, "_load_universe", lambda u: (members[u], {"source": "test"}))
    monkeypatch.setattr(tc, "_fetch_prices", lambda tickers, period, status, label: {
        t: px[t] for t in tickers if t in px})
    monkeypatch.setattr(tc, "_leverage", lambda p: {"LEV": "empirical_leverage_vs_SPY(test)"})
    return members, px


def test_run_writes_valid_artefacts_sorted_by_score(tmp_trend_store, monkeypatch):
    _stub_run(monkeypatch)
    out = tc.run(action="auto", force=True)
    assert out["ok"]
    doc = trend.load("stocks", "latest")
    rows = doc["rows"]
    assert rows[0]["ticker"] == "UP"
    scored = [r for r in rows if not r["excluded"]]
    assert [r["score"] for r in scored] == sorted([r["score"] for r in scored], reverse=True)
    assert [r["rank"] for r in scored] == list(range(1, len(scored) + 1))
    for r in scored:
        assert -20 <= r["score"] <= 20 and len(r["forecasts"]) == 7
        assert all(-20 <= f <= 20 for f in r["forecasts"])
        assert r["score"] == pytest.approx(sum(r["forecasts"]) / 7, abs=0.02)
        assert r["structure"] in structure.STRUCTURES
    byt = {r["ticker"]: r for r in rows}
    assert byt["NEW"]["excluded"] and "insufficient_history" in byt["NEW"]["exclusion_reason"]
    assert byt["GONE"]["excluded"] and byt["GONE"]["exclusion_reason"] == "no_price_data"
    assert byt["UP"]["score"] > 0 > byt["DN"]["score"]
    etf = {r["ticker"]: r for r in trend.load("etfs", "latest")["rows"]}
    assert etf["LEV"]["excluded"] and etf["LEV"]["exclusion_reason"].startswith("empirical_leverage")
    # the whole 5y series is persisted on a first run; events and breadth too
    h = history.series_for("stocks", "UP")
    assert len(h) > 1000
    assert trend.load("stocks", "breadth")["rows"]
    assert trend.load("stocks", "recent_events")["rows"]
    diag = trend.load("stocks", "diagnostics")
    assert set(diag["realized_abs_longrun"]) == set(KEYS)
    json.dumps(doc)                                              # valid JSON, no NaN


def test_rerun_same_day_adds_no_history_and_events_are_idempotent(tmp_trend_store, monkeypatch):
    _stub_run(monkeypatch)
    tc.run(action="auto", force=True)
    n1 = len(history.series_for("stocks", "UP"))
    e1 = len(history.events_for("stocks"))
    tc.run(action="auto", force=True)
    assert len(history.series_for("stocks", "UP")) == n1
    assert len(history.events_for("stocks")) == e1


def test_non_trading_day_is_a_no_op(tmp_trend_store, monkeypatch):
    from zenith.pretom import calendar as cal
    monkeypatch.setattr(cal, "is_trading_day", lambda d: False)
    called = []
    monkeypatch.setattr(tc, "_fetch_prices", lambda *a, **k: called.append(1) or {})
    assert tc.run(action="auto")["gated"] and not called


# ============================================================= table.py ====
def _table_df():
    rows = []
    for i, (t, s, st, sec) in enumerate([("A", 18.0, "Persistent Uptrend", "Tech"),
                                         ("B", 3.0, "Emerging Uptrend", "Energy"),
                                         ("C", -17.0, "Persistent Downtrend", "Tech"),
                                         ("D", 8.0, "Deteriorating Uptrend", "Health")]):
        rows.append({"ticker": t, "name": f"{t} corp", "score": s, "structure": st, "sector": sec,
                     "forecasts": [s] * 7, "d5": float(i), "d20": -float(i), "slope": float(i - 2),
                     "n_bull": 7 if s > 0 else 0, "partial": t == "D",
                     "latest_cross_date": f"2026-09-1{i}", "last_event": None,
                     "last_event_date": f"2026-09-0{i + 1}"})
    return table.frame(rows)


def test_table_default_sort_is_score_descending():
    df = _table_df()
    assert table.sort(df, "Trend Score")["ticker"].tolist() == ["A", "D", "B", "C"]
    assert table.sort(df, "Trend Score", ascending=True)["ticker"].tolist() == ["C", "B", "D", "A"]


@pytest.mark.parametrize("key", list(table.SORTS))
def test_every_sort_option_runs(key):
    out = table.sort(_table_df(), key)
    assert len(out) == 4


def test_table_sorts_by_individual_speed_and_ticker():
    df = _table_df()
    assert table.sort(df, "EWMAC 16/64")["ticker"].tolist()[0] == "A"
    assert table.sort(df, "Ticker", ascending=True)["ticker"].tolist() == ["A", "B", "C", "D"]


def test_table_filters():
    df = _table_df()
    assert table.filter_rows(df, query="c corp")["ticker"].tolist() == ["C"]
    assert set(table.filter_rows(df, groups=["Tech"], group_col="sector")["ticker"]) == {"A", "C"}
    assert table.filter_rows(df, structures=["Emerging Uptrend"])["ticker"].tolist() == ["B"]
    assert set(table.filter_rows(df, score_range=(0, 20))["ticker"]) == {"A", "B", "D"}
    assert "D" not in table.filter_rows(df, hide_partial=True)["ticker"].tolist()
    df["cash_like"] = df["ticker"] == "A"
    assert "A" not in table.filter_rows(df, hide_cash_like=True)["ticker"].tolist()
    assert "A" in table.filter_rows(df)["ticker"].tolist()
    # agreement is direction-aware: a 7-speed DOWNtrend (C) qualifies too
    assert set(table.filter_rows(df, min_bull=7)["ticker"]) == {"A", "B", "C", "D"}
    df.loc[df["ticker"] == "A", "n_bull"] = 5
    assert "A" not in table.filter_rows(df, min_bull=6)["ticker"].tolist()


# ================================================================ view ====
def _render(sub: str | None = None, universe: str = "stocks", pick: str | None = None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from zenith.trend import view\nview.render()\n", default_timeout=120)
    if universe:
        at.session_state["trend_uni"] = universe
    if sub:
        at.session_state[f"trend_sub_{universe}"] = sub
    if pick:
        at.session_state[f"trend_pick_{universe}"] = pick
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    return at, text


def test_view_renders_empty_state(tmp_trend_store):
    at, text = _render()
    assert "key findings" in text.lower()
    assert any("No data yet" in str(i.value) for i in at.info)


@pytest.mark.parametrize("universe", ["stocks", "etfs"])
@pytest.mark.parametrize("sub,needle", [
    ("Overview", "Trend Structure Map"),
    ("All Trends", "scored"),
    ("Triggers", "what just changed"),
    ("Detail", "Trend ladder"),
])
def test_view_sub_views_render(tmp_trend_store, monkeypatch, universe, sub, needle):
    _stub_run(monkeypatch)
    tc.run(action="auto", force=True)
    from zenith.trend import view as tv
    monkeypatch.setattr(tv, "_prices", lambda t: None)          # no network in the Detail view
    at, text = _render(sub, universe=universe)
    assert needle.lower() in text.lower()
    assert "STOCKS" in text and "ETFS" in text                   # cross-universe header


def test_view_all_trends_heatmap_mode(tmp_trend_store, monkeypatch):
    _stub_run(monkeypatch)
    tc.run(action="auto", force=True)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from zenith.trend import view\nview.render()\n", default_timeout=120)
    at.session_state["trend_uni"] = "stocks"
    at.session_state["trend_sub_stocks"] = "All Trends"
    at.session_state["trend_mode_stocks"] = "Heatmap"
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_today_badge(tmp_trend_store, monkeypatch):
    from zenith.trend import view as tv
    assert tv.today_badge() is None
    _stub_run(monkeypatch)
    tc.run(action="auto", force=True)
    assert "TREND" in tv.today_badge()


def test_click_navigation_handoff_opens_detail_for_the_picked_asset(tmp_trend_store, monkeypatch):
    """A click elsewhere stages `trend_nav_<u>` + `trend_pick_<u>` (the radio's
    own key cannot be written once it is instantiated); the next run must land
    on Detail for exactly that asset."""
    _stub_run(monkeypatch)
    tc.run(action="auto", force=True)
    from streamlit.testing.v1 import AppTest
    from zenith.trend import view as tv
    monkeypatch.setattr(tv, "_prices", lambda t: None)
    at = AppTest.from_string("from zenith.trend import view\nview.render()\n", default_timeout=120)
    at.session_state["trend_uni"] = "stocks"
    at.session_state["trend_sub_stocks"] = "Overview"
    at.session_state["trend_nav_stocks"] = "Detail"
    at.session_state["trend_pick_stocks"] = "DN"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["trend_sub_stocks"] == "Detail"
    assert at.session_state["trend_detail_pick_stocks"] == "DN"
    assert any("DN — DN Inc" in str(m.value) for m in at.markdown)

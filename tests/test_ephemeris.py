"""EPHEMERIS — offline tests: scoring rules, sampler bounds, the anti-leak
payload rule, repository round-trips and the per-ticker Parquet store.

Synthetic, seeded data only (np.random.default_rng), no network.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest

from zenith.ephemeris import chart as board
from zenith.ephemeris import prefetch
from zenith.ephemeris.repo import SqliteRepo, ddl, valid_handle
from zenith.ephemeris.sampler import Chart, NoChartError, draw, window_is_clean
from zenith.ephemeris.scoring import InvalidLevels, level_from_spec, score_trade, validate_levels
from zenith.ephemeris.store_px import PxStore


# ================================================================ scoring ====
def _bars(rows):
    o, h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return dict(o=o, h=h, l=l, c=c)


def test_no_stop_long_and_short():
    f = _bars([(100, 103, 99, 102), (102, 106, 101, 105)])
    r = score_trade(direction=1, entry=100.0, stake=1000, **f)
    assert r["market_ret"] == pytest.approx(0.05) and r["outcome"] == "UP"
    assert r["pnl"] == pytest.approx(50.0) and r["win"]
    assert r["exit_reason"] == "horizon" and r["candles_held"] == 2
    assert r["mfe"] == pytest.approx(0.06) and r["mae"] == pytest.approx(-0.01)
    s = score_trade(direction=-1, entry=100.0, stake=500, **f)
    assert s["pnl"] == pytest.approx(-25.0) and not s["win"]
    assert s["mfe"] == pytest.approx(0.01) and s["mae"] == pytest.approx(-0.06)


def test_flat_counts_as_loss():
    r = score_trade(direction=1, entry=100.0, stake=1000, **_bars([(100, 101, 99, 100)]))
    assert r["outcome"] == "FLAT" and not r["win"] and r["pnl"] == 0


def test_stop_then_target_exits_at_stop_level():
    f = _bars([(100, 101, 96, 97), (97, 112, 97, 111)])          # stop bar 0, target bar 1
    r = score_trade(direction=1, entry=100.0, stake=1000, sl=97.5, tp=110.0, **f)
    assert r["exit_reason"] == "stop" and r["exit_price"] == 97.5
    assert r["exit_index"] == 0 and r["r_mult"] == pytest.approx(-1.0)
    assert r["market_ret"] == pytest.approx(0.11)                 # market still records the horizon move
    assert r["mfe_full"] == pytest.approx(0.12)                   # left on the table


def test_same_candle_touch_is_stop_and_ambiguous():
    f = _bars([(100, 111, 95, 104)])
    r = score_trade(direction=1, entry=100.0, stake=1000, sl=96.0, tp=110.0, **f)
    assert r["exit_reason"] == "stop" and r["ambiguous"] and r["exit_price"] == 96.0
    s = score_trade(direction=-1, entry=100.0, stake=1000, sl=104.0, tp=90.0,
                    **_bars([(100, 105, 89, 95)]))
    assert s["exit_reason"] == "stop" and s["ambiguous"] and s["exit_price"] == 104.0


def test_gap_through_stop_fills_at_open():
    f = _bars([(100, 101, 99, 100), (93, 94, 90, 92)])            # gaps below a 97 stop
    r = score_trade(direction=1, entry=100.0, stake=1000, sl=97.0, **f)
    assert r["exit_reason"] == "stop" and r["gap_fill"] and r["exit_price"] == 93.0
    assert r["r_mult"] == pytest.approx(-7 / 3)                   # worse than -1R
    assert not r["ambiguous"]


def test_gap_through_target_fills_at_open_short():
    f = _bars([(100, 101, 99, 100), (88, 89, 86, 87)])
    r = score_trade(direction=-1, entry=100.0, stake=1000, sl=104.0, tp=92.0, **f)
    assert r["exit_reason"] == "target" and r["gap_fill"] and r["exit_price"] == 88.0
    assert r["trade_ret"] == pytest.approx(0.12)


def test_target_hit_and_untouched_horizon():
    f = _bars([(100, 102, 99, 101), (101, 108, 100, 107), (107, 109, 106, 108)])
    r = score_trade(direction=1, entry=100.0, stake=1000, tp=105.0, **f)
    assert r["exit_reason"] == "target" and r["exit_index"] == 1 and r["exit_price"] == 105.0
    n = score_trade(direction=1, entry=100.0, stake=1000, sl=90.0, tp=120.0, **f)
    assert n["exit_reason"] == "horizon" and n["exit_price"] == 108.0


def test_level_validation_and_specs():
    with pytest.raises(InvalidLevels):
        validate_levels(1, 100, 101, None)
    with pytest.raises(InvalidLevels):
        validate_levels(-1, 100, None, 101)
    validate_levels(-1, 100, 103, 95)
    assert level_from_spec(1, 100, "sl", "%", 2, None) == pytest.approx(98)
    assert level_from_spec(-1, 100, "tp", "ATR", 1.5, 2.0) == pytest.approx(97)
    with pytest.raises(InvalidLevels):
        level_from_spec(1, 100, "sl", "ATR", 1, None)


# ================================================================ sampler ====
def _walk(n, seed, start="2000-01-03", vol=0.01, freq="B"):
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
    o = c * (1 + rng.normal(0, vol / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = rng.integers(200_000, 900_000, n).astype(float)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def counts(self, tf="Daily"):
        return {t: len(df) for t, df in self.frames.items()}

    def bars(self, t, tf="Daily"):
        return self.frames.get(t)


UNI = [{"ticker": "AAA", "name": "Alpha", "cls": "Russell 1000", "sector": "Tech"},
       {"ticker": "BBB", "name": "Beta Gold", "cls": "Precious Metals", "sector": ""},
       {"ticker": "CCC", "name": "Short", "cls": "Bonds", "sector": ""}]


@pytest.fixture
def store():
    return FakeStore({"AAA": _walk(3000, 1), "BBB": _walk(1500, 2), "CCC": _walk(100, 3)})


def test_draw_respects_bounds(store):
    rng = random.Random(7)
    for _ in range(200):
        ch = draw(store, UNI, classes=("Russell 1000", "Precious Metals", "Bonds"), lookback=250,
                  horizon=50, rng=rng)
        assert ch.ticker in ("AAA", "BBB")                       # CCC is too short
        assert ch.vis0 >= 0 and ch.t + ch.horizon < len(ch.c)
        assert len(ch.future()["c"]) == 50
        full = store.frames[ch.ticker]
        assert full["close"].loc[ch.decision_date] == pytest.approx(ch.entry)


def test_draw_is_class_balanced(store):
    rng = random.Random(3)
    picks = [draw(store, UNI, classes=("Russell 1000", "Precious Metals"), lookback=60, horizon=5,
                  rng=rng).cls for _ in range(400)]
    share = picks.count("Precious Metals") / len(picks)
    assert 0.4 < share < 0.6


def test_draw_skips_seen_and_raises_when_nothing_fits(store):
    rng = random.Random(1)
    ch = draw(store, UNI, classes=("Precious Metals",), lookback=60, horizon=5, rng=rng)
    seen = {("BBB", "Daily"): [ch.decision_date]}
    for _ in range(50):
        c2 = draw(store, UNI, classes=("Precious Metals",), lookback=60, horizon=5, rng=rng, seen=seen)
        assert abs((c2.decision_date - ch.decision_date).days) > 60
    with pytest.raises(NoChartError):
        draw(store, UNI, classes=("Bonds",), lookback=250, horizon=10, rng=rng)


def test_era_filter(store):
    rng = random.Random(5)
    for _ in range(30):
        ch = draw(store, UNI, classes=("Russell 1000",), lookback=60, horizon=5, rng=rng, crisis_only=True)
        d = ch.decision_date
        assert (pd.Timestamp("2000-03-01") <= d <= pd.Timestamp("2002-10-31")
                or pd.Timestamp("2007-10-01") <= d <= pd.Timestamp("2009-03-31"))
        ch = draw(store, UNI, classes=("Russell 1000",), lookback=60, horizon=5, rng=rng, min_year=2008)
        assert ch.decision_date.year >= 2008


def test_window_is_clean_rejects_bad_data():
    df = _walk(200, 9)
    a = [df[k].to_numpy() for k in ("open", "high", "low", "close", "volume")] + [df.index.values]
    assert window_is_clean(*a, "Russell 1000", "Daily")
    c = a[3].copy(); c[100] *= 0.5                                # unadjusted 2:1 split
    assert not window_is_clean(a[0], a[1], a[2], c, a[4], a[5], "Russell 1000", "Daily")
    v = a[4].copy(); v[50:60] = 0                                 # zero-volume run
    assert not window_is_clean(a[0], a[1], a[2], a[3], v, a[5], "Russell 1000", "Daily")
    assert window_is_clean(a[0], a[1], a[2], a[3], v, a[5], "Currencies", "Daily")
    c = a[3].copy(); c[20:30] = c[20]                             # stale prints
    assert not window_is_clean(a[0], a[1], a[2], c, a[4], a[5], "Russell 1000", "Daily")
    ts = a[5].copy(); ts[150:] = ts[150:] + np.timedelta64(30, "D")   # 30-day hole
    assert not window_is_clean(*a[:5], ts, "Russell 1000", "Daily")


# ================================================================ payload ====
def _chart(store) -> Chart:
    return draw(store, UNI, classes=("Russell 1000",), lookback=120, horizon=10, rng=random.Random(11))


def test_decision_payload_leaks_nothing(store):
    ch = _chart(store)
    p = board.payload(ch)
    blob = json.dumps(p)
    assert "future" not in p and "result" not in p and "unblind" not in p and "dates" not in p
    assert len(p["bars"]) == 120
    assert ch.ticker not in blob and ch.name not in blob
    assert not any(pd.Timestamp(x).strftime("%Y-%m-%d") in blob for x in ch.ts)
    assert p["bars"][0][3] == pytest.approx(100.0)               # rebased to 100
    assert p["bars"][-1][3] == pytest.approx(100.0 * ch.entry / ch.c[ch.vis0], rel=1e-6)
    assert not any(b[3] == pytest.approx(ch.entry, rel=1e-9) for b in p["bars"][1:])


def test_unblinded_options_and_reveal_payload(store):
    ch = _chart(store)
    p = board.payload(ch, rebase=False, hide_dates=False, hide_ticker=False)
    assert ch.ticker in p["header"] and len(p["dates"]) == 120
    assert p["bars"][-1][3] == pytest.approx(ch.entry, rel=1e-6)
    fut = ch.future()
    res = score_trade(direction=1, entry=ch.entry, stake=500, **fut)
    r = board.payload(ch, direction=1, stake=500, result=res)
    assert len(r["future"]) == 10 and r["unblind"]["ticker"] == ch.ticker
    assert r["result"]["pnl"] == pytest.approx(res["pnl"])
    html = board.board_html(r)
    assert "__PAYLOAD__" not in html and "lightweight-charts" in html


# ================================================================ repo ====
def test_repo_roundtrip_and_low_friction_profiles():
    repo = SqliteRepo(":memory:")
    a = repo.get_or_create_player("Tape Reader")
    assert repo.get_or_create_player("  tape   READER ")["id"] == a["id"]     # same handle, any case
    b = repo.get_or_create_player("other")
    assert {p["display"] for p in repo.players()} == {"Tape Reader", "other"}
    assert repo.check_pin(a, "")                                   # no PIN -> open
    repo.set_pin(a["id"], "1234")
    a = repo.player("tape reader")
    assert repo.check_pin(a, "1234") and not repo.check_pin(a, "0000")

    gid = repo.log_guess({"player_id": a["id"], "mode": "practice", "ticker": "AAA", "timeframe": "Daily",
                          "horizon": 10, "lookback": 120, "decision_date": "2010-05-03T00:00:00",
                          "chart_key": "AAA|Daily|2010-05-03 00:00", "direction": 1, "pnl": 12.5,
                          "win": True, "blinding_json": {"rebase": True}})
    assert isinstance(gid, int)
    df = repo.guesses_df(a["id"])
    assert len(df) == 1 and df["win"].iloc[0] == 1 and json.loads(df["blinding_json"].iloc[0])["rebase"]
    assert repo.guesses_df(b["id"]).empty
    assert ("AAA", "Daily") in repo.seen_charts(a["id"])
    assert repo.save_daily(a["id"], "2026-10-03", 0, gid) and not repo.save_daily(a["id"], "2026-10-03", 0, gid)
    repo.reset_account(a["id"], "practice")
    assert len(repo.resets(a["id"], "practice")) == 1
    repo.save_preset(a["id"], "params", "Swing", {"tf": "4H", "h": 20})
    repo.save_preset(a["id"], "params", "Swing", {"tf": "4H", "h": 25})
    assert repo.presets(a["id"], "params") == {"Swing": {"tf": "4H", "h": 25}}


def test_handle_rules_and_postgres_ddl():
    assert valid_handle("jo") is None and valid_handle("a") and valid_handle("x" * 25)
    assert valid_handle("bad;drop") is not None
    pg = ddl("postgres")
    assert "BIGSERIAL" in pg and "ENABLE ROW LEVEL SECURITY" in pg and "AUTOINCREMENT" not in pg


# ================================================================ store ====
def test_parquet_store_reads_one_ticker(tmp_path):
    frames = {t: _walk(600 + i * 50, i).astype("float32") for i, t in enumerate(["AAA", "BBB", "ZZZ"])}
    n = prefetch.write_parquet(frames, tmp_path / "daily.parquet")
    assert n == 3
    st = PxStore(root=tmp_path)
    assert st.available("Daily") and st.counts("Daily") == {"AAA": 600, "BBB": 650, "ZZZ": 700}
    b = st.bars("BBB")
    assert len(b) == 650 and b["close"].iloc[-1] == pytest.approx(float(frames["BBB"]["close"].iloc[-1]))
    w = st.bars("BBB", "Weekly")
    assert 120 <= len(w) <= 140 and w["high"].max() == pytest.approx(b["high"].max())
    assert st.bars("NOPE") is None


def test_normalize_repairs_and_drops():
    df = _walk(200, 4)
    df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] * 0.9           # inverted bar
    df.iloc[6, df.columns.get_loc("close")] = np.nan
    out = prefetch._normalize(df.rename(columns=str.title))
    assert len(out) == 199 and (out["high"] >= out[["open", "close", "low"]].max(axis=1) - 1e-6).all()


def test_indicator_payload_is_sliced_and_normalized(store):
    from zenith.ephemeris import indicators as ind
    ch = _chart(store)
    data = ind.compute([{"id": "sma", "params": {"lengths": "20,200"}}, {"id": "volume", "params": {}},
                        {"id": "rsi", "params": {}}], ch.o, ch.h, ch.l, ch.c, ch.v, ch.vis0)
    p = board.payload(ch, ind=data)
    assert all(len(o["values"]) == 120 for o in p["overlays"])          # no future in decision state
    assert all(len(s["values"]) == 120 for pn in p["panes"] for s in pn["series"])
    vol = [x for x in p["panes"][0]["series"][0]["values"] if x is not None]
    assert np.median(vol) == pytest.approx(1.0, rel=1e-6)               # raw share counts never shipped
    sma20 = p["overlays"][0]["values"]
    assert sma20[-1] == pytest.approx(100 * ind.sma(ch.c, 20)[ch.t] / ch.c[ch.vis0], rel=1e-6)   # rebased
    res = score_trade(direction=1, entry=ch.entry, stake=500, **ch.future())
    r = board.payload(ch, direction=1, stake=500, result=res, ind=data)
    assert all(len(o["values"]) == 130 for o in r["overlays"]) and len(r["up"]) == 130
    assert board.total_height(r) == board.HEIGHT + 2 * board.PANE_HEIGHT


def test_volume_pane_dropped_when_instrument_has_no_volume(store):
    from zenith.ephemeris import indicators as ind
    ch = _chart(store)
    ch.v[:] = 0.0                                                       # spot FX: no volume field
    data = ind.compute([{"id": "volume", "params": {}}, {"id": "rsi", "params": {}}],
                       ch.o, ch.h, ch.l, ch.c, ch.v, ch.vis0)
    p = board.payload(ch, ind=data)
    assert [pn["name"] for pn in p["panes"]] == ["RSI 14"]

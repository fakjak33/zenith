"""PRETOM tests — offline, synthetic data only (no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import zenith.pretom as pretom
from zenith.pretom import analytics as an
from zenith.pretom import calendar as cal
from zenith.pretom.universe import parse_vone_items


# ---------------------------------------------------------------- calendar --
def test_nyse_holidays_known_dates():
    h26 = cal.nyse_holidays(2026)
    assert date(2026, 1, 1) in h26
    assert date(2026, 1, 19) in h26          # MLK
    assert date(2026, 4, 3) in h26           # Good Friday (Easter Apr 5)
    assert date(2026, 6, 19) in h26          # Juneteenth (Friday)
    assert date(2026, 7, 3) in h26           # July 4 is Saturday -> Friday
    assert date(2026, 11, 26) in h26         # Thanksgiving
    assert date(2026, 12, 25) in h26
    # July 4, 2021 was a Sunday -> observed Monday July 5
    assert date(2021, 7, 5) in cal.nyse_holidays(2021)
    # Jan 1, 2022 was a Saturday -> NYSE did not observe it
    assert date(2022, 1, 1) not in cal.nyse_holidays(2022)
    assert cal.is_trading_day(date(2021, 12, 31))


def test_month_schedule_july_2026():
    s = cal.month_schedule(2026, 7)
    assert s["tau"] == date(2026, 7, 31)
    assert s["lock_by"] == date(2026, 7, 17)          # tau-10
    assert s["win_start"] == date(2026, 7, 20)        # tau-9
    assert s["win_end_classic"] == date(2026, 7, 27)  # tau-4
    assert s["win_end_t1"] == date(2026, 7, 28)       # tau-3
    assert s["post_start"] == date(2026, 7, 28)
    assert s["post_end"] == date(2026, 8, 5)          # tau+3 next month


def test_month_schedule_tom_keys():
    s = cal.month_schedule(2026, 7)
    assert s["tom_start"] == s["tau"] == date(2026, 7, 31)
    assert s["tom_end"] == s["post_end"] == date(2026, 8, 5)   # tau+3
    iso = cal.schedule_iso(s)
    assert iso["tom_start"] == "2026-07-31" and iso["tom_end"] == "2026-08-05"


def test_month_schedule_skips_holiday_at_month_end():
    # December 2026: Christmas (Fri Dec 25) sits inside the window span.
    s = cal.month_schedule(2026, 12)
    assert s["tau"] == date(2026, 12, 31)
    assert s["win_end_classic"] == date(2026, 12, 24)  # tau-4 skips Dec 25
    assert s["win_end_t1"] == date(2026, 12, 28)


def test_post_window_spans_month_boundary():
    st = cal.state_for(date(2026, 8, 3))
    assert st["state"] == "POST"
    assert st["month"] == "2026-07"
    assert st["tau_offset"] == 1                       # tau+1


def test_state_machine_walk_july_2026():
    def s(d):
        return cal.state_for(d)

    assert s(date(2026, 7, 1))["state"] == "POST"      # June's tau+1
    assert s(date(2026, 7, 1))["month"] == "2026-06"
    assert s(date(2026, 7, 7))["state"] == "FORMING"
    st15 = s(date(2026, 7, 15))
    assert st15["state"] == "FORMING" and st15["days_to_open"] == 3
    assert s(date(2026, 7, 17))["state"] == "LOCKED"   # tau-10
    st20 = s(date(2026, 7, 20))
    assert st20["state"] == "WINDOW_OPEN" and st20["tau_offset"] == -9
    assert st20["days_to_close_classic"] == 5
    st28 = s(date(2026, 7, 28))
    assert st28["state"] == "WINDOW_OPEN" and st28["t1_tail"]
    assert s(date(2026, 7, 29))["state"] == "POST"
    assert s(date(2026, 7, 31))["state"] == "POST"
    # weekend evaluates at the next trading day but is flagged
    sat = s(date(2026, 7, 18))
    assert sat["state"] == "WINDOW_OPEN" and not sat["is_trading_day"]
    assert cal.tau_offset(date(2026, 7, 18)) is None


def test_completed_months_and_verify():
    months = cal.completed_months(24, date(2026, 7, 15))
    assert len(months) == 24
    # June 2026's post window ends Jul 6 < Jul 15, so June is complete.
    assert months[-1] == (2026, 6)
    assert months[0] == (2024, 7)
    assert months == sorted(months)

    idx = pd.DatetimeIndex(cal.trading_days(date(2026, 6, 1), date(2026, 6, 30)))
    assert cal.verify_against(idx) == []
    missing = cal.verify_against(idx.delete(3))
    assert len(missing) == 1 and "unexpected close" in missing[0]


# --------------------------------------------------------------- analytics --
def _px(dates, closes, volume=1_000_000.0):
    idx = pd.DatetimeIndex(dates)
    c = pd.Series(list(closes), index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": float(volume)}, index=idx)


def _flat_then_drop(dates, level, drop_to, n_tail):
    n = len(dates)
    return [level] * (n - n_tail) + [drop_to] * n_tail


def _mk_universe(specs):
    """specs: {ticker: (drop_frac, volume, weight, div_payer)} on 300 bars."""
    days = cal.trading_days(date(2025, 4, 1), date(2026, 7, 17))[-300:]
    px, uni, divs = {}, [], {}
    for t, (drop, vol, w, payer) in specs.items():
        closes = _flat_then_drop(days, 100.0, 100.0 * (1 - drop), 40)
        px[t] = _px(days, closes, vol)
        uni.append({"ticker": t, "name": t, "sector": "Test", "weight_pct": w})
        divs[t] = payer
    return px, uni, divs, days[-1]


def test_decile_and_composite_ordering():
    specs = {"DEEP": (0.60, 5e7, 3.0, False),   # far below high, liquid, big
             "SHAL": (0.30, 1e5, 0.01, True)}   # the other decile member
    for i in range(18):                          # 18 near-high fillers
        specs[f"F{i:02d}"] = (0.01 + i * 0.001, 1e6, 0.1, True)
    px, uni, divs, lock = _mk_universe(specs)
    basket = an.build_basket(px, uni, lock, divs)
    assert len(basket) == 2                      # bottom decile of 20 names
    assert [b["ticker"] for b in basket] == ["DEEP", "SHAL"]
    assert basket[0]["rank"] == 1
    assert all(0.0 <= b["score"] <= 1.0 for b in basket)
    assert basket[0]["pct_below_high"] == pytest.approx(0.60)
    assert basket[0]["mom_12_2"] is not None and basket[0]["mom_12_2"] < 0


def test_liquidity_ranks_up():
    # identical drop/weight; only ADV differs -> higher ADV must rank better
    specs = {"LIQ": (0.50, 9e8, 0.05, True), "ILL": (0.50, 1e4, 0.05, True)}
    for i in range(18):
        specs[f"F{i:02d}"] = (0.01, 1e6, 0.1, True)
    px, uni, divs, lock = _mk_universe(specs)
    basket = an.build_basket(px, uni, lock, divs)
    ranks = {b["ticker"]: b["rank"] for b in basket}
    assert ranks["LIQ"] < ranks["ILL"]


def _window_setup():
    sched = cal.month_schedule(2026, 7)
    days = cal.trading_days(date(2026, 6, 1), date(2026, 8, 10))
    return sched, days


def test_window_stats_directional():
    sched, days = _window_setup()
    fall = [100.0 * (0.99 ** i) for i in range(len(days))]
    # V-shape: drops 3% mid-window, recovers above base by the window end
    base_i = days.index(sched["win_start"])
    vshape = [100.0] * len(days)
    for j, off in enumerate(range(base_i, len(days))):
        vshape[off] = 97.0 if j in (1, 2) else 100.5 if j >= 3 else 100.0
    px = {"DOWN": _px(days, fall), "VEE": _px(days, vshape)}
    names = [{"ticker": "DOWN", "weight_pct": 1.0}, {"ticker": "VEE", "weight_pct": 1.0}]
    spy = _px(days, [100.0] * len(days))["close"]
    s = an.window_stats(names, px, sched["win_start"], sched["win_end_classic"], spy)
    assert s["n_priced"] == 2
    assert s["ew_ret"] < 0 and s["ew_excess"] < 0
    assert s["pct_negative_end"] == 0.5          # DOWN yes, VEE recovered
    assert s["pct_negative_any"] == 1.0          # both dipped below base
    assert len(s["daily"]) == 6                  # six classic window days


def test_cap_vs_equal_weight():
    sched, days = _window_setup()
    fall = [100.0 * (0.99 ** i) for i in range(len(days))]
    rise = [100.0 * (1.01 ** i) for i in range(len(days))]
    px = {"BIG": _px(days, fall), "TINY": _px(days, rise)}
    names = [{"ticker": "BIG", "weight_pct": 5.0}, {"ticker": "TINY", "weight_pct": 0.01}]
    spy = _px(days, [100.0] * len(days))["close"]
    s = an.window_stats(names, px, sched["win_start"], sched["win_end_classic"], spy)
    assert s["cw_ret"] < 0 < abs(s["ew_ret"]) or s["cw_ret"] < s["ew_ret"]
    assert s["cw_ret"] < s["ew_ret"]


def test_reversal_ratio():
    sched, days = _window_setup()
    closes = []
    for d in days:
        if d < sched["win_start"]:
            closes.append(100.0)
        elif d <= sched["win_end_t1"]:
            closes.append(90.0)                  # -10% during the window
        elif d <= sched["post_end"]:
            closes.append(97.0)                  # recovers most of it after
        else:
            closes.append(97.0)
    px = {"X": _px(days, closes)}
    names = [{"ticker": "X", "weight_pct": 1.0}]
    spy = _px(days, [100.0] * len(days))["close"]
    classic = an.window_stats(names, px, sched["win_start"], sched["win_end_classic"], spy)
    post = an.post_stats(names, px, sched["post_start"], sched["post_end"], spy,
                         classic["ew_excess"])
    assert classic["ew_excess"] == pytest.approx(-0.10)
    # post base is the tau-4 close (90): 97/90-1 = +7.78%, ~78% retrace
    assert post["reversal_ratio"] == pytest.approx(0.778, abs=0.01)


def test_per_name_marks_and_panel():
    sched, days = _window_setup()
    fall = [100.0 * (0.99 ** i) for i in range(len(days))]
    px = {"A": _px(days, fall)}
    names = [{"ticker": "A", "weight_pct": 1.0}]
    marks = an.per_name_marks(names, px, sched)
    assert marks["A"]["classic_ret"] < 0
    assert marks["A"]["t1_ret"] <= marks["A"]["classic_ret"]  # keeps falling
    assert marks["A"]["min_ret_in_window"] <= marks["A"]["t1_ret"]
    panel = an.build_panel(px, ["A", "MISSING"], sched["win_start"], sched["post_end"])
    assert "A" in panel and "MISSING" not in panel
    assert len(panel["A"]["d"]) == len(panel["A"]["c"]) > 0


# ------------------------------------------------------------------- TOM ----
def test_tom_market_stats_directional():
    sched, days = _window_setup()
    tom_days = set(cal.trading_days(sched["tom_start"], sched["tom_end"]))
    closes, level = [], 100.0
    for d in days:
        if d in tom_days:
            level *= 1.01                          # +1%/day inside [tau, tau+3]
        closes.append(level)
    spy = _px(days, closes)["close"]
    s = an.tom_market_stats(spy, sched)
    assert s["n_days"] == 4
    assert s["spy_ret"] == pytest.approx(1.01 ** 4 - 1.0, abs=1e-6)
    assert len(s["daily"]) == 4


def _tom_series(start, end, bumps):
    """Synthetic close series: +bump on each month's TOM days, flat elsewhere.
    bumps: {'YYYY-MM': daily_return}."""
    days = cal.trading_days(start, end)
    months = sorted({(d.year, d.month) for d in days})
    tom_map = {}
    for (y, m) in months:
        sched = cal.month_schedule(y, m)
        for d in cal.trading_days(sched["tom_start"], sched["tom_end"]):
            tom_map[d] = f"{y:04d}-{m:02d}"
    closes, level = [], 100.0
    for d in days:
        mkey = tom_map.get(d)
        if mkey is not None:
            level *= 1.0 + bumps.get(mkey, 0.0)
        closes.append(level)
    return _px(days, closes)["close"]


def test_tom_longrun_split_and_summary():
    bumps = {"2026-01": 0.010, "2026-02": 0.011,
             "2026-03": 0.012, "2026-04": 0.013}
    spy = _tom_series(date(2026, 1, 2), date(2026, 5, 15), bumps)
    rows = an.tom_longrun(spy)
    got = {r["month"]: r for r in rows}
    # Jan-Apr have complete windows inside the series; Dec/May do not.
    assert set(got) == set(bumps)
    for m, b in bumps.items():
        assert got[m]["n_tom_days"] == 4
        assert got[m]["tom_ret"] == pytest.approx((1 + b) ** 4 - 1, abs=1e-5)
        assert got[m]["rest_ret"] == pytest.approx(0.0, abs=1e-9)
        assert got[m]["n_rest_days"] >= 5

    s = an.summarize_tom(rows)
    assert s["overall"]["n_months"] == 4
    assert s["overall"]["tom_hit"] == 1.0
    assert s["overall"]["rest_avg"] == pytest.approx(0.0, abs=1e-6)
    assert s["overall"]["tom_avg"] > 0.04
    assert s["overall"]["t_stat"] is not None and s["overall"]["t_stat"] > 5
    assert set(s["by_decade"]) == {"2020s"}
    assert s["since_2010"]["n_months"] == 4
    assert s["trailing_24m"]["n_months"] == 4


def test_summarize_tom_empty():
    s = an.summarize_tom([])
    assert s["overall"] is None and s["by_decade"] == {}


# ------------------------------------------------------- universe / store ---
def test_vone_parser():
    items = [
        {"ticker": "NVDA", "longName": "NVIDIA Corp.", "percentWeight": "6.72"},
        {"ticker": "BRK.B", "longName": "Berkshire", "percentWeight": "1.1"},
        {"ticker": "-", "longName": "US Dollar", "percentWeight": "0.1"},
        {"ticker": "", "longName": "Blank", "percentWeight": "0.0"},
        {"ticker": "ABC", "longName": "Alphabet Co", "percentWeight": "n/a"},
    ]
    rows = parse_vone_items(items)
    tickers = [r["ticker"] for r in rows]
    assert tickers == ["NVDA", "BRK-B", "ABC"]
    assert rows[0]["weight_pct"] == pytest.approx(6.72)
    assert rows[2]["weight_pct"] is None


def test_archive_roundtrip_and_history_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(pretom, "PRETOM_ARCHIVE_DIR", tmp_path)
    obj = {"month": "2026-06", "names": [{"ticker": "X"}],
           "stats": {"classic": {"ew_ret": -0.01}, "t1": {"ew_ret": -0.012},
                     "post": {"ew_ret": 0.008},
                     "tom_market": {"spy_ret": 0.012, "n_days": 4},
                     "tom_basket": {"ew_ret": 0.02}}}
    pretom.archive_month("2026-06", obj)
    assert pretom.archive_months() == ["2026-06"]
    back = pretom.load_month("2026-06")
    assert back["stats"]["classic"]["ew_ret"] == -0.01
    assert "t1" in back["stats"] and "post" in back["stats"]
    assert back["stats"]["tom_market"]["spy_ret"] == 0.012


def test_update_stats_tolerates_old_schedule(monkeypatch, tmp_path):
    """Archives written before the TOM keys existed must still reconcile
    (tom_start/tom_end fall back to tau/post_end)."""
    from zenith.pretom import compute as pc
    sched, days = _window_setup()
    old_iso = {k: v.isoformat() for k, v in sched.items()
               if k not in ("tom_start", "tom_end")}
    closes = [100.0] * len(days)
    px = {"A": _px(days, closes)}
    obj = {"month": "2026-07", "locked_at": sched["lock_by"].isoformat(),
           "schedule": old_iso, "names": [{"ticker": "A", "weight_pct": 1.0}]}
    spy = _px(days, closes)["close"]
    pc._update_stats(obj, px, spy, date(2026, 8, 10))
    assert obj["stats"]["tom_market"]["n_days"] == 4
    assert obj["stats"]["tom_basket"]["n_priced"] == 1
    assert obj["stats"]["final"] is True

"""MOMENTUM factor builders — pure functions, no I/O, no network.

Every builder takes a single stock's OHLCV DataFrame (columns
[open, high, low, close, volume], DatetimeIndex, as returned by
`zenith.cas.sources.prices.get_history`) and returns RAW components. Two of
the five factors (time-series, cross-sectional) share the same per-horizon
vol-adjusted return and need the WHOLE universe's distribution to finish
normalizing (winsorization, percentile rank) — that step lives in
`zenith.mom.engine`, which is the only place a cross-sectional panel exists.
The other three (breakout, trend speed, momentum strength) are entirely
single-stock and are returned already combined into a factor score.

Prices are assumed adjusted close (splits + dividends) — the only
self-consistent choice for both return math and trailing high/low breakout
levels, since an unadjusted series jumps at every split.

The current bar is EXCLUDED from every trailing high/low/window computation
that it would otherwise trivially satisfy, and every slope/quality window
ends at the current bar — no look-ahead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .normalize import clip1, tanh_clip, ols_slope_r2

# 400-day MA needs 400 bars just to exist at the latest point; the momentum-
# strength "change in slope" needs a second slope reading 21 days earlier,
# which itself needs a 400-day MA ending 21 days earlier -> 400 + 21 + slack.
MIN_BARS = 460

# (lookback_days, skip_days). "12_1" = the classic UMD spec: the return from
# ~12 months ago through ~1 month ago (skips the most recent month, which is
# known to mean-revert rather than continue -- Jegadeesh 1990).
HORIZON_SPEC = {
    "12_1": (252, 21),
    "12m": (252, 0),
    "9m": (189, 0),
    "6m": (126, 0),
    "3m": (63, 0),
    "1m": (21, 0),
}

MA_PERIODS = (9, 21, 50, 100, 200, 250, 400)
# Momentum-strength slopes deliberately exclude the fastest (9D) average --
# §12 of the spec asks for 21D..400D slopes; 9D is too noisy to call a "trend".
SLOPE_PERIODS = (21, 50, 100, 200, 250, 400)


def daily_vol(close: pd.Series, window: int = 252, min_obs: int = 60) -> float | None:
    """Trailing daily-return stdev (NOT annualized) over `window` days. This is
    the vol used to convert a raw horizon return into a t-stat-like measure:
    dividing by sigma_d * sqrt(holding_days) answers "how many standard
    deviations was this move, given how noisy this stock normally is"."""
    r = close.pct_change(fill_method=None).dropna().tail(window)
    if len(r) < min_obs:
        return None
    v = float(r.std())
    return v if math.isfinite(v) and v > 0 else None


def trailing_return(close: pd.Series, lookback: int, skip: int = 0) -> float | None:
    """(close[t-skip] / close[t-lookback]) - 1. skip=0 -> ordinary trailing
    return; skip=21 with lookback=252 -> the 12-1 spec."""
    n = len(close)
    if n < lookback + 1:
        return None
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - lookback]
    if not (math.isfinite(end) and math.isfinite(start)) or start == 0:
        return None
    return float(end / start - 1.0)


def moving_averages(close: pd.Series, periods=MA_PERIODS) -> dict[int, pd.Series]:
    return {p: close.rolling(p).mean() for p in periods}


# --------------------------------------------------------------------- F1 ---
def time_series_raw(df: pd.DataFrame, horizons: dict = HORIZON_SPEC) -> dict:
    """Per-horizon raw return + vol-adjusted t-stat-like measure. NOT yet
    winsorized or squashed -- that needs the cross-sectional panel (engine.py)."""
    close = df["close"]
    sigma = daily_vol(close, 252)
    out = {"vol_d": sigma, "horizons": {}}
    for name, (lb, skip) in horizons.items():
        r = trailing_return(close, lb, skip)
        n_h = lb - skip
        m = (r / (sigma * math.sqrt(n_h))) if (r is not None and sigma) else None
        out["horizons"][name] = {"ret": r, "m": m, "n_days": n_h}
    return out


# --------------------------------------------------------------------- F2 ---
def _breakout_state_series(close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Trailing high/low computed over [t-lookback, t-1] -- the CURRENT bar is
    excluded via shift(1), so today's close can only break a level set by
    prior days, never by itself."""
    hi = close.shift(1).rolling(lookback).max()
    lo = close.shift(1).rolling(lookback).min()
    state = pd.Series(0, index=close.index, dtype=int)
    state[close > hi] = 1
    state[close < lo] = -1
    return state, hi, lo


def breakout_raw(df: pd.DataFrame, horizons: dict = HORIZON_SPEC,
                  decay_window: int = 10, decay_weight: float = 0.6,
                  inside_weight: float = 0.4) -> dict:
    """Daily-close breakout signal per horizon. +1 = closed above the trailing
    high, -1 = closed below the trailing low, else a weak (|.| <= 0.4)
    position-in-channel read. A break that fired within the last
    `decay_window` trading days (but not today) still counts at `decay_weight`
    strength -- whichever of "today" or "recent break" is larger wins.
    `days_since_break`/`confirmed` are reported for the UI, not scored."""
    close = df["close"]
    out = {"horizons": {}}
    for name, (lb, _skip) in horizons.items():
        if len(close) < lb + 2:
            out["horizons"][name] = {"b": 0.0, "state": "n/a", "days_since_break": None,
                                      "confirmed": False, "high": None, "low": None}
            continue
        state, hi, lo = _breakout_state_series(close, lb)
        last_state = int(state.iloc[-1])
        today_hi, today_lo, today_close = hi.iloc[-1], lo.iloc[-1], close.iloc[-1]

        if last_state == 1:
            b_today = 1.0
        elif last_state == -1:
            b_today = -1.0
        elif pd.notna(today_hi) and pd.notna(today_lo) and today_hi != today_lo:
            pos = (today_close - today_lo) / (today_hi - today_lo)
            b_today = inside_weight * (2.0 * pos - 1.0)
        else:
            b_today = 0.0

        decay_b = 0.0
        if last_state == 0 and len(state) > decay_window:
            window = state.iloc[-decay_window:-1]     # the decay_window-1 days before today
            nz = window[window != 0]
            if len(nz):
                decay_b = decay_weight * float(nz.iloc[-1])
        b_final = b_today if abs(b_today) >= abs(decay_b) else decay_b

        nz_all = state[state != 0]
        days_since_break = (len(state) - 1 - state.index.get_loc(nz_all.index[-1])
                            if len(nz_all) else None)
        confirmed = bool(last_state != 0 and len(state) >= 2 and state.iloc[-2] == last_state)

        out["horizons"][name] = {
            "b": round(float(clip1(b_final)), 4),
            "state": "break_up" if last_state == 1 else "break_down" if last_state == -1 else "inside",
            "days_since_break": days_since_break, "confirmed": confirmed,
            "high": float(today_hi) if pd.notna(today_hi) else None,
            "low": float(today_lo) if pd.notna(today_lo) else None,
        }
    return out


# --------------------------------------------------------------------- F4 ---
def trend_speed_raw(df: pd.DataFrame, periods=MA_PERIODS, cross_window: int = 21,
                    width_window: int = 21, width_vol_window: int = 252) -> dict:
    """GMMA-style trend-speed reading: how the MOVING-AVERAGE STRUCTURE is
    behaving right now, distinct from the long-horizon return itself.
      align         -- bullish-minus-bearish share of the 6 ordered MA pairs
      price_align    -- price above vs below each of the 7 MAs
      cross_recent   -- net fresh crossovers in the last `cross_window` days
                        (this is the "speed" term)
      expansion_signed -- change in MA spread (width), z-scored against its
                        own history, signed by the current alignment so
                        expanding-while-bearish scores negative
    """
    close = df["close"]
    mas = moving_averages(close, periods)
    last_close = float(close.iloc[-1])
    last_mas = {p: (float(mas[p].iloc[-1]) if pd.notna(mas[p].iloc[-1]) else None) for p in periods}
    pairs = list(zip(periods, periods[1:]))

    align_signs = [1 if last_mas[f] > last_mas[s] else (-1 if last_mas[f] < last_mas[s] else 0)
                   for f, s in pairs if last_mas[f] is not None and last_mas[s] is not None]
    align = float(np.mean(align_signs)) if align_signs else 0.0

    price_signs = [1 if last_close > m else (-1 if last_close < m else 0)
                   for m in last_mas.values() if m is not None]
    price_align = float(np.mean(price_signs)) if price_signs else 0.0

    cross_bull = cross_bear = 0
    for f, s in pairs:
        diff = (mas[f] - mas[s])
        sign = np.sign(diff).dropna()
        recent = sign.tail(cross_window + 1).to_numpy()
        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            if prev <= 0 and curr > 0:
                cross_bull += 1
            elif prev >= 0 and curr < 0:
                cross_bear += 1
    cross_recent = clip1((cross_bull - cross_bear) / max(1, len(pairs)))

    ma_df = pd.DataFrame({p: mas[p] for p in periods})
    width = (ma_df.std(axis=1) / close).replace([np.inf, -np.inf], np.nan)
    width_now = float(width.iloc[-1]) if pd.notna(width.iloc[-1]) else None
    dwidth = width.diff(width_window)
    dwidth_now = float(dwidth.iloc[-1]) if pd.notna(dwidth.iloc[-1]) else None
    hist = dwidth.dropna().tail(width_vol_window)
    sigma_dw = float(hist.std()) if len(hist) >= 60 else None
    z = (dwidth_now / sigma_dw) if (dwidth_now is not None and sigma_dw) else None
    align_sign = 1.0 if align > 0 else (-1.0 if align < 0 else 0.0)
    expansion_signed = clip1((z / 2.0) * align_sign) if z is not None else 0.0

    return {
        "align": round(align, 4), "price_align": round(price_align, 4),
        "cross_recent": round(cross_recent, 4), "cross_bull_n": cross_bull, "cross_bear_n": cross_bear,
        "expansion_signed": round(expansion_signed, 4),
        "width": width_now, "width_change_21d": dwidth_now,
        "ma_values": last_mas,
    }


# --------------------------------------------------------------------- F5 ---
def momentum_strength_raw(df: pd.DataFrame, ma_periods=MA_PERIODS, slope_periods=SLOPE_PERIODS,
                          slope_window: int = 21, gap_window: int = 21,
                          quality_window: int = 126, slope_cap: float = 1.5,
                          gap_scale: float = 0.10, dgap_scale: float = 0.05) -> dict:
    """Trend QUALITY, not just direction: are moving-average slopes rising,
    accelerating, and is the MA structure expanding, and is the price path
    itself a smooth drift (high R^2) rather than one discrete jump (Da,
    Gurun & Warachka 2014 "Frog in the Pan")."""
    close = df["close"]
    sigma = daily_vol(close, 252)
    mas = moving_averages(close, ma_periods)

    def _slope_at(series: pd.Series, offset: int) -> float | None:
        i_now, i_prev = -1 - offset, -1 - offset - slope_window
        if len(series) < -i_prev:
            return None
        a, b = series.iloc[i_now], series.iloc[i_prev]
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return float(a / b - 1.0)

    slopes, accels = {}, {}
    for p in slope_periods:
        s = mas[p]
        sl_now = _slope_at(s, 0)
        sl_prev = _slope_at(s, slope_window)
        slopes[p] = sl_now
        accels[p] = (sl_now - sl_prev) if (sl_now is not None and sl_prev is not None) else None

    slope_z = {p: (v / (sigma * math.sqrt(slope_window)) if (v is not None and sigma) else None)
               for p, v in slopes.items()}
    accel_z = {p: (v / (sigma * math.sqrt(slope_window)) if (v is not None and sigma) else None)
               for p, v in accels.items()}
    valid_sz = [v for v in slope_z.values() if v is not None]
    valid_az = [v for v in accel_z.values() if v is not None]
    s_slope = float(np.mean([clip1(v / slope_cap) for v in valid_sz])) if valid_sz else 0.0
    s_accel = float(np.mean([clip1(v / slope_cap) for v in valid_az])) if valid_az else 0.0

    pairs = list(zip(ma_periods, ma_periods[1:]))
    last_mas = {p: (float(mas[p].iloc[-1]) if pd.notna(mas[p].iloc[-1]) else None) for p in ma_periods}
    gaps_now, gaps_prev, dgaps = {}, {}, {}
    for f, s in pairs:
        mf_now, ms_now = last_mas[f], last_mas[s]
        g_now = (mf_now / ms_now - 1.0) if (mf_now is not None and ms_now and ms_now != 0) else None
        gaps_now[(f, s)] = g_now
        if len(mas[f]) > gap_window and len(mas[s]) > gap_window:
            mf_p, ms_p = mas[f].iloc[-1 - gap_window], mas[s].iloc[-1 - gap_window]
            g_prev = (float(mf_p) / float(ms_p) - 1.0) if (pd.notna(mf_p) and pd.notna(ms_p) and ms_p != 0) else None
        else:
            g_prev = None
        gaps_prev[(f, s)] = g_prev
        dgaps[(f, s)] = (g_now - g_prev) if (g_now is not None and g_prev is not None) else None

    valid_g = [v for v in gaps_now.values() if v is not None]
    valid_dg = [v for v in dgaps.values() if v is not None]
    s_gap = float(np.mean([tanh_clip(v, gap_scale) for v in valid_g])) if valid_g else 0.0
    s_dgap = float(np.mean([tanh_clip(v, dgap_scale) for v in valid_dg])) if valid_dg else 0.0

    log_close = close.where(close > 0).apply(math.log) if (close > 0).any() else close * float("nan")
    slope, r2 = ols_slope_r2(log_close, window=quality_window)
    if slope is not None and r2 is not None:
        s_quality = clip1((1.0 if slope >= 0 else -1.0) * r2)
    else:
        s_quality = 0.0

    return {
        "s_slope": round(s_slope, 4), "s_accel": round(s_accel, 4),
        "s_gap": round(s_gap, 4), "s_dgap": round(s_dgap, 4), "s_quality": round(s_quality, 4),
        "quality_r2": r2, "quality_slope": slope,
        "slopes": slopes, "accels": accels,
        "gaps": {f"{f}_{s}": g for (f, s), g in gaps_now.items()},
        "gaps_change": {f"{f}_{s}": g for (f, s), g in dgaps.items()},
    }


def build_all(df: pd.DataFrame, min_bars: int = MIN_BARS) -> dict | None:
    """Run all four single-stock builders (TS raw, breakout, speed, strength).
    Returns None if the stock has insufficient history -- callers must treat
    that as "excluded, insufficient history", never as a zero score."""
    if df is None or df.empty or len(df) < min_bars:
        return None
    return {
        "bars": len(df),
        "ts_raw": time_series_raw(df),
        "breakout_raw": breakout_raw(df),
        "speed_raw": trend_speed_raw(df),
        "strength_raw": momentum_strength_raw(df),
        "last_close": float(df["close"].iloc[-1]),
        "asof": df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None,
    }

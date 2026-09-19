"""Signal-change detection — "what just changed?". Pure, no I/O.

Five event types, all derived from one asset's full daily EWMAC history, so a
missed nightly run is healed automatically: the next run recomputes the series
and emits every event dated after the last one stored.

  cross         a single speed's fast EMA crossed its slow EMA (a sign change
                of the UNROUNDED raw EWMAC -- never inferred from a rounded
                forecast). dir=+1 bullish crossover, -1 bearish.
  trigger       the Trend Score's state entered BULLISH (>= +5) or BEARISH
                (< -5) from neutral or from the opposite side -- a new trend
                at the composite level (bullish / bearish trigger).
  upgrade /     the score crossed one or more of MOMENTUM's state bands
  downgrade     (+/-5, +/-10, +/-15; config.MOM_STATES) without being a
                trigger. +4 -> +9 and -8 -> -3 are upgrades; +15 -> +9 and
                -4 -> -12 are downgrades. The same bands as the MOMENTUM tabs,
                so "STRONG BULLISH" means one thing everywhere in Zenith.

  Triggers and band changes run on a HYSTERETIC band state
  (config.TREND_BAND_HYSTERESIS, see held_bands()): a move to a band further
  from neutral fires at the threshold itself, a move back toward neutral
  only once the score has cleared the band edge by the buffer. A score
  hovering on a line therefore produces one event, not a daily pair.
  confirmation  a multi-speed confirmation: >= TREND_CONFIRM_MIN_SPEEDS distinct
                speeds crossed in the SAME direction within TREND_CONFIRM_WINDOW
                trading days and are all still on that side today. This is what
                separates a broad trend change from one noisy 2/8 flip.

Every event carries the score before/after, the state labels, and a horizon
tag (short / medium / long -- see trend.SPEED_HORIZON) so the Triggers view can
filter short-term from long-term changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (MOM_STATES, TREND_BAND_HYSTERESIS, TREND_CONFIRM_MIN_SPEEDS,
                      TREND_CONFIRM_WINDOW)
from . import SPEED_HORIZON, SPEED_KEYS

# MOM_STATES is (threshold, label) descending; a score belongs to the first
# band whose threshold it meets (mom.engine.state_for). As an ascending ordinal:
# 0 = EXTREME BEARISH ... 3 = NEUTRAL ... 6 = EXTREME BULLISH.
_ASC_THRESH = np.array(sorted(t for t, _ in MOM_STATES)[1:], dtype=float)   # drop the -20 floor
BAND_LABELS = tuple(lbl for _, lbl in reversed(MOM_STATES))
_NEUTRAL = BAND_LABELS.index("NEUTRAL")

EVENT_TYPES = ("cross", "trigger", "upgrade", "downgrade", "confirmation")


def band_of(score) -> np.ndarray:
    """Band ordinal(s) for score(s); identical semantics to mom.engine.state_for."""
    return np.searchsorted(_ASC_THRESH, np.asarray(score, dtype=float), side="right")


def state_label(score: float | None) -> str | None:
    if score is None or not np.isfinite(score):
        return None
    return BAND_LABELS[int(band_of(score))]


def side_of_band(band: int) -> int:
    """+1 bullish state, -1 bearish state, 0 neutral."""
    return 1 if band > _NEUTRAL else (-1 if band < _NEUTRAL else 0)


def held_bands(sc: np.ndarray, buffer: float = TREND_BAND_HYSTERESIS) -> np.ndarray:
    """The hysteretic band state per day (-1 where the score is invalid).

    From held band c, with the day's raw band r = band_of(score):
      * r further from neutral than c (a stronger trend, or leaving neutral):
        move to r immediately -- entering a band happens at its threshold;
      * r closer to (or through) neutral: move to r only if the score has
        cleared c's edge by `buffer` -- below c's lower edge minus buffer
        when falling, at/above c's upper edge plus buffer when rising.
    buffer=0 reproduces the raw day-over-day band exactly."""
    held = np.full(len(sc), -1, dtype=int)
    c = -1
    for i, s in enumerate(sc):
        if not np.isfinite(s):
            held[i] = -1 if c < 0 else c
            continue
        r = int(band_of(s))
        if c < 0 or r == c:
            c = r if c < 0 else c
        elif (c >= _NEUTRAL and r > c) or (c <= _NEUTRAL and r < c):
            c = r                                              # strengthening: at the line
        elif r < c and s < _ASC_THRESH[c - 1] - buffer:        # weakening from above
            c = r
        elif r > c and s >= _ASC_THRESH[c] + buffer:           # weakening from below
            c = r
        held[i] = c
    return held


def _r(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def crossovers(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-speed crossover matrix: +1 on a bullish crossover day, -1 on a
    bearish one, 0 otherwise. An exact tie (raw == 0) holds the prior side
    rather than counting as a cross of its own."""
    side = np.sign(raw).replace(0.0, np.nan).ffill()
    valid = raw.notna()
    prev = side.shift(1)
    crossed = (side != prev) & side.notna() & prev.notna() & valid & valid.shift(1, fill_value=False)
    return side.where(crossed, 0.0).fillna(0.0).astype(int)


def detect(ticker: str, raw: pd.DataFrame, score: pd.Series, since: str | None = None,
           confirm_min: int = TREND_CONFIRM_MIN_SPEEDS,
           confirm_window: int = TREND_CONFIRM_WINDOW) -> list[dict]:
    """All events for one asset dated strictly after `since` (ISO date).
    `raw`/`score` must share one daily index; pass enough history before
    `since` (>= confirm_window + 1 rows) for the confirmation look-back."""
    if raw.empty:
        return []
    dates = [d.strftime("%Y-%m-%d") for d in raw.index]
    sc = score.reindex(raw.index).to_numpy(dtype=float)
    out: list[dict] = []

    def _emit(i: int, **kw) -> None:
        if since is not None and dates[i] <= since:
            return
        prev = sc[i - 1] if i > 0 else np.nan
        out.append({"date": dates[i], "ticker": ticker,
                    "score_before": _r(prev), "score_after": _r(sc[i]), **kw})

    # --- single-speed crossovers ------------------------------------------
    xm = crossovers(raw)
    xv = xm.to_numpy()
    for j, key in enumerate(SPEED_KEYS):
        if key not in xm.columns:
            continue
        for i in np.nonzero(xv[:, j])[0]:
            _emit(int(i), type="cross", dir=int(xv[i, j]), speed=key, speeds=[key],
                  n_speeds=1, horizon=SPEED_HORIZON[key])

    # --- score state: triggers and band upgrades/downgrades ----------------
    valid = np.isfinite(sc)
    bands = held_bands(sc)
    changed = np.nonzero(valid[1:] & (bands[:-1] >= 0) & (bands[1:] != bands[:-1]))[0] + 1
    for i in changed:
        i = int(i)
        b0, b1 = int(bands[i - 1]), int(bands[i])
        s0, s1 = side_of_band(b0), side_of_band(b1)
        common = {"from": BAND_LABELS[b0], "to": BAND_LABELS[b1], "speed": None, "speeds": [],
                  "n_speeds": 0, "horizon": "composite"}
        if s1 != 0 and s1 != s0:
            _emit(i, type="trigger", dir=s1, **common)
        else:
            _emit(i, type="upgrade" if b1 > b0 else "downgrade", dir=1 if b1 > b0 else -1, **common)

    # --- multi-speed confirmation -----------------------------------------
    # Vectorized: a speed "counts" for direction d on day i if it crossed in
    # direction d within the trailing window AND is still on that side on day
    # i (a 2/8 that flipped up and straight back down does not confirm
    # anything). A confirmation fires on a day where the count reaches the
    # threshold, that day itself contributed a qualifying cross, and no
    # same-direction confirmation fired within the window already.
    keys = [k for k in SPEED_KEYS if k in xm.columns]
    side_now = np.sign(raw[keys]).replace(0.0, np.nan).ffill().to_numpy()
    for d in (1, -1):
        hit_x = (xm[keys] == d).astype(float)
        recent = hit_x.rolling(confirm_window, min_periods=1).max().to_numpy() > 0
        live = recent & (side_now == d)
        count = live.sum(axis=1)
        today = (xv[:, [SPEED_KEYS.index(k) for k in keys]] == d) & live
        cand = np.nonzero((count >= confirm_min) & today.any(axis=1))[0]
        last_fire = -10**9
        for i in cand:
            if i - last_fire < confirm_window:
                continue
            last_fire = int(i)
            hit = [k for j, k in enumerate(keys) if live[i, j]]
            slowest = max(SPEED_KEYS.index(k) for k in hit)
            _emit(int(i), type="confirmation", dir=d, speed=None, speeds=hit, n_speeds=len(hit),
                  horizon=SPEED_HORIZON[SPEED_KEYS[slowest]])
    out.sort(key=lambda e: (e["date"], EVENT_TYPES.index(e["type"]), e.get("speed") or ""))
    return out


def last_cross(raw: pd.DataFrame) -> dict:
    """{speed: {"date", "dir", "bars_ago"}} for each speed's most recent
    crossover, or None if it has never crossed within the available history."""
    xm = crossovers(raw)
    out = {}
    n = len(xm)
    for key in SPEED_KEYS:
        if key not in xm.columns:
            out[key] = None
            continue
        nz = np.nonzero(xm[key].to_numpy())[0]
        if not len(nz):
            out[key] = None
            continue
        i = int(nz[-1])
        out[key] = {"date": xm.index[i].strftime("%Y-%m-%d"), "dir": int(xm[key].iloc[i]),
                    "bars_ago": n - 1 - i}
    return out


def flip_stats(raw: pd.DataFrame, bars_per_year: int = 252) -> dict:
    """Per speed: crossovers per year, average run length (bars between
    crosses) and the current run's length and side. Answers "how often does
    this asset flip between bullish and bearish at each speed?"."""
    xm = crossovers(raw)
    side = np.sign(raw).replace(0.0, np.nan).ffill()
    out = {}
    for key in SPEED_KEYS:
        if key not in xm.columns or raw[key].notna().sum() < 2:
            out[key] = None
            continue
        valid_n = int(raw[key].notna().sum())
        nz = np.nonzero(xm[key].to_numpy())[0]
        n_x = len(nz)
        runs = np.diff(nz) if n_x > 1 else np.array([])
        cur_start = int(nz[-1]) if n_x else int(np.argmax(raw[key].notna().to_numpy()))
        out[key] = {
            "n_crosses": n_x,
            "n_bull_to_bear": int((xm[key] == -1).sum()),
            "n_bear_to_bull": int((xm[key] == 1).sum()),
            "per_year": round(n_x / valid_n * bars_per_year, 2) if valid_n else None,
            "avg_run": round(float(runs.mean()), 1) if len(runs) else None,
            "current_run": len(raw) - cur_start,
            "current_side": int(side[key].iloc[-1]) if pd.notna(side[key].iloc[-1]) else 0,
        }
    return out

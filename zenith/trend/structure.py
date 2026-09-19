"""Trend STRUCTURE — what the relationship between the seven speeds says.

Pure functions over one asset's seven forecasts (a list aligned with
SPEED_KEYS, None where a speed is not valid). The score says how much trend
there is; this says WHERE on the horizon ladder it lives:

  breadth        -- how many of the seven speeds are bullish / bearish
  fast / slow    -- mean forecast of the three fastest (2/8, 4/16, 8/32) and
                    three slowest (32/128, 64/256, 128/512) speeds; 16/64 is the
                    pivot and belongs to neither group
  slope          -- fast - slow. Positive: the short-term trend is running
                    AHEAD of the long-term one (accelerating up, or a downtrend
                    recovering). Negative: short-term is lagging (a rising
                    trend losing momentum, or a downtrend accelerating).
  label          -- one of STRUCTURES, by the explicit rule order in label_for()

The two labels the whole feature exists to separate:
  * "Persistent Uptrend": at least 6 of 7 speeds bullish and score >= +5 --
    trending across horizons.
  * "Emerging Uptrend": the fast group is bullish while the slow group is
    still bearish -- only the short-term trend has turned up.
and their mirror "Deteriorating Uptrend": the fast group has turned bearish
while the slow group is still bullish.
"""

from __future__ import annotations

from ..config import (MOM_STATES, TREND_DISAGREE_MIN, TREND_FAST_IDX,
                      TREND_PERSISTENT_MIN_SPEEDS, TREND_SLOW_IDX, THEME)

BULL_BAND = next(t for t, lbl in MOM_STATES if lbl == "BULLISH")        # +5
BEAR_BAND = next(t for t, lbl in MOM_STATES if lbl == "NEUTRAL")        # -5 (neutral floor)

STRUCTURES = ("Persistent Uptrend", "Uptrend", "Emerging Uptrend", "Mixed",
              "Deteriorating Uptrend", "Downtrend", "Persistent Downtrend")

STRUCTURE_COLORS = {
    "Persistent Uptrend": THEME.teal,
    "Uptrend": "#1f8f85",
    "Emerging Uptrend": THEME.mint,
    "Mixed": THEME.muted,
    "Deteriorating Uptrend": THEME.mustard,
    "Downtrend": "#b8452f",
    "Persistent Downtrend": THEME.coral,
}

STRUCTURE_HELP = {
    "Persistent Uptrend": f">= {TREND_PERSISTENT_MIN_SPEEDS} of 7 speeds bullish and score >= +{BULL_BAND:.0f}: "
                          "trending up across horizons.",
    "Uptrend": f"Score >= +{BULL_BAND:.0f} but not every horizon agrees.",
    "Emerging Uptrend": "Fast speeds bullish while slow speeds are still bearish: only the short-term "
                        "trend has turned up so far.",
    "Mixed": "No clear trend: score between the bullish and bearish bands without a fast/slow split.",
    "Deteriorating Uptrend": "Fast speeds have turned bearish while slow speeds are still bullish: a "
                             "rising trend losing momentum (or a new downtrend starting).",
    "Downtrend": f"Score <= {BEAR_BAND:.0f} but not every horizon agrees.",
    "Persistent Downtrend": f">= {TREND_PERSISTENT_MIN_SPEEDS} of 7 speeds bearish and score <= {BEAR_BAND:.0f}: "
                            "trending down across horizons.",
}


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def groups(f: list) -> tuple[float | None, float | None, float | None]:
    """(fast mean, mid = the 16/64 forecast, slow mean). None where no valid speed."""
    fast = _mean([f[i] for i in TREND_FAST_IDX])
    slow = _mean([f[i] for i in TREND_SLOW_IDX])
    mid = f[3] if len(f) > 3 else None
    return fast, mid, slow


def label_for(f: list, score: float | None) -> str:
    """Explicit rule order: persistent trends first (they are the strongest
    statement), then a genuine fast/slow split, then plain up/down by score."""
    if score is None:
        return "Mixed"
    n_bull = sum(1 for x in f if x is not None and x > 0)
    n_bear = sum(1 for x in f if x is not None and x < 0)
    if score >= BULL_BAND and n_bull >= TREND_PERSISTENT_MIN_SPEEDS:
        return "Persistent Uptrend"
    if score < BEAR_BAND and n_bear >= TREND_PERSISTENT_MIN_SPEEDS:
        return "Persistent Downtrend"
    fast, _, slow = groups(f)
    if fast is not None and slow is not None:
        if fast > TREND_DISAGREE_MIN and slow < -TREND_DISAGREE_MIN:
            return "Emerging Uptrend"
        if fast < -TREND_DISAGREE_MIN and slow > TREND_DISAGREE_MIN:
            return "Deteriorating Uptrend"
    if score >= BULL_BAND:
        return "Uptrend"
    if score < BEAR_BAND:
        return "Downtrend"
    return "Mixed"


def analyse(f: list, score: float | None) -> dict:
    """All structure fields for one asset on one day."""
    fast, mid, slow = groups(f)
    return {
        "n_bull": sum(1 for x in f if x is not None and x > 0),
        "n_bear": sum(1 for x in f if x is not None and x < 0),
        "n_valid": sum(1 for x in f if x is not None),
        "fast": None if fast is None else round(fast, 3),
        "mid": None if mid is None else round(mid, 3),
        "slow": None if slow is None else round(slow, 3),
        "slope": None if (fast is None or slow is None) else round(fast - slow, 3),
        "disagree": bool(fast is not None and slow is not None
                         and fast * slow < 0
                         and abs(fast) > TREND_DISAGREE_MIN and abs(slow) > TREND_DISAGREE_MIN),
        "structure": label_for(f, score),
    }


def regime_of(f: list) -> str:
    """Coarse day-level regime for the detail chart's background bands:
    all_bull / all_bear / disagree (fast vs slow split) / mixed."""
    valid = [x for x in f if x is not None]
    if not valid:
        return "mixed"
    if all(x > 0 for x in valid) and len(valid) >= TREND_PERSISTENT_MIN_SPEEDS:
        return "all_bull"
    if all(x < 0 for x in valid) and len(valid) >= TREND_PERSISTENT_MIN_SPEEDS:
        return "all_bear"
    fast, _, slow = groups(f)
    if (fast is not None and slow is not None and fast * slow < 0
            and abs(fast) > TREND_DISAGREE_MIN and abs(slow) > TREND_DISAGREE_MIN):
        return "disagree"
    return "mixed"

"""Benchmarks computed on the SAME charts the player sees.

  * Coin flip -- 50%, with a Wilson band at the player's sample size (stats.py).
  * Always-long base rate -- the share of UP outcomes for this asset class x
    timeframe x horizon, measured from the price store (base_rates.json,
    rebuilt nightly by prefetch). Beating 50% is not skill when the class
    drifts up 56% of the time; beating THIS is.
  * Trend rule -- long if close > SMA-N and the SMA's slope over m bars > 0,
    short if both are reversed, otherwise no trade. N and m are configurable.
"""

from __future__ import annotations

import json

import numpy as np

from ..config import EPHEMERIS_FILES
from . import HORIZONS
from .indicators import sma

TREND_RULE_DEFAULT = {"n": 50, "m": 10}
BASE_RATE_TFS = ("Daily", "Weekly", "Monthly")


# ------------------------------------------------------------ base rates ----
def up_counts(c: np.ndarray, h: int) -> tuple[int, int]:
    """Non-overlapping h-bar outcomes over a close series: (#up, #total)."""
    c = np.asarray(c, float)
    if len(c) <= h:
        return 0, 0
    pts = c[::h]
    r = pts[1:] / pts[:-1] - 1
    r = r[np.isfinite(r)]
    return int((r > 0).sum()), int(r.size)


DAILY_RULES = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}
HOURLY_RULES = {"1H": None, "4H": "4H"}


def build_base_rates(frames: dict, classes: dict[str, str], resample, rules: dict | None = None) -> dict:
    """{tf: {class: {h: {"p": share_up, "n": outcomes}}}} incl. class "ALL".
    frames: ticker -> OHLCV DataFrame at the source frequency; `rules` maps each
    timeframe to its resample rule (None = as-is). Defaults to Daily/Weekly/Monthly."""
    acc: dict = {}
    rules = DAILY_RULES if rules is None else rules
    for t, df in frames.items():
        cls = classes.get(t)
        if cls is None:
            continue
        for tf, rule in rules.items():
            c = (resample(df, rule) if rule else df)["close"].to_numpy(float)
            for h in HORIZONS:
                u, n = up_counts(c, h)
                if not n:
                    continue
                for key in (cls, "ALL"):
                    a = acc.setdefault(tf, {}).setdefault(key, {}).setdefault(str(h), [0, 0])
                    a[0] += u
                    a[1] += n
    return {tf: {cls: {h: {"p": round(u / n, 4), "n": n} for h, (u, n) in hs.items()}
                 for cls, hs in by.items()} for tf, by in acc.items()}


_CACHE: dict = {}


def base_rates() -> dict:
    p = EPHEMERIS_FILES["base_rates"]
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if _CACHE.get("mt") != mt:
        _CACHE.update(mt=mt, data=json.loads(p.read_text(encoding="utf-8")))
    return _CACHE["data"]


def base_rate(cls: str, tf: str, horizon: int, table: dict | None = None) -> dict | None:
    """{"p", "n", "h"} for the nearest tabulated horizon; class falls back to ALL."""
    table = base_rates() if table is None else table
    by = (table or {}).get(tf) or (table or {}).get("Daily")
    if not by:
        return None
    row = by.get(cls) or by.get("ALL")
    if not row:
        return None
    hs = sorted(int(h) for h in row)
    near = min(hs, key=lambda x: abs(x - horizon))
    out = dict(row[str(near)])
    out["h"] = near
    return out


# ------------------------------------------------------------ trend rule ----
def trend_rule_call(c: np.ndarray, t: int, n: int = 50, m: int = 10) -> int:
    """+1 / -1 / 0 at decision index t, using only bars 0..t."""
    s = sma(np.asarray(c, float)[:t + 1], n)
    if t - m < 0 or not np.isfinite(s[t]) or not np.isfinite(s[t - m]):
        return 0
    slope = s[t] - s[t - m]
    if c[t] > s[t] and slope > 0:
        return 1
    if c[t] < s[t] and slope < 0:
        return -1
    return 0


def trend_rule_label(call: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "NO TRADE"}[call]

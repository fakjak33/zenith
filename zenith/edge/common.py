"""Shared ranking / assembly helpers for the edge screeners (pure functions)."""

from __future__ import annotations

import math

import numpy as np


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def pct_ranks(values: list[float]) -> list[float]:
    """0-100 percentile rank of each value (average rank for ties)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = round(100.0 * avg / (n - 1), 1) if n > 1 else 50.0
        i = j + 1
    return ranks


def zscore(x: float, series: list[float]) -> float | None:
    s = [v for v in series if finite(v)]
    if len(s) < 5:
        return None
    mu, sd = float(np.mean(s)), float(np.std(s, ddof=1))
    if sd <= 0:
        return None
    return round((x - mu) / sd, 3)


def assemble(rows: list[dict], key: str, higher_is_long: bool,
             decile: float = 0.1) -> dict:
    """Rank rows by `rows[i][key]`, tag percentile / decile / side.

    higher_is_long=True: the TOP decile is the long idea, bottom is short.
    higher_is_long=False (lottery/short-interest): the TOP decile is the
    short/avoid idea; bottom decile is the benign long-ish side.
    Returns {n, long:[...], short:[...], ranked:[...]} sorted by the measure.
    """
    rows = [r for r in rows if finite(r.get(key))]
    if not rows:
        return {"n": 0, "long": [], "short": [], "ranked": []}
    vals = [float(r[key]) for r in rows]
    pr = pct_ranks(vals)
    for r, p in zip(rows, pr):
        r["pctile"] = p
    rows.sort(key=lambda r: r[key], reverse=True)   # high -> low
    n = len(rows)
    cut = max(1, round(n * decile))
    top, bottom = rows[:cut], rows[-cut:]
    long_side, short_side = (top, bottom) if higher_is_long else (bottom, top)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["decile"] = ("top" if i < cut else "bottom" if i >= n - cut else "mid")
    for r in long_side:
        r["side"] = "long"
    for r in short_side:
        r["side"] = "short"
    return {"n": n, "long": long_side, "short": short_side, "ranked": rows}

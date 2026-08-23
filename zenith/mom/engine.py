"""MOMENTUM composite engine — combines the five factor scores.

Two of the five factors (time-series, cross-sectional) are built together in
`cross_sectional()` because they share the same per-horizon vol-adjusted
input (`factors.time_series_raw`) and BOTH need the whole universe's
distribution to finish normalizing: time-series needs cross-sectional
winsorization before its tanh squash (one extreme mover shouldn't blow out
its own score), and cross-sectional momentum is a percentile rank by
definition. The other three factors (breakout, trend speed, momentum
strength) are entirely single-stock and live in `FACTOR_REGISTRY`.

Extensibility: a sixth single-stock factor is one `FACTOR_REGISTRY` entry
plus one `config.MOM_WEIGHTS` key — `composite()` picks it up automatically
without any other change to this module.
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from ..config import MOM_HORIZON_WEIGHTS, MOM_WEIGHTS, MOM_STATES
from ..edge.common import pct_ranks
from .normalize import clip1, inv_normal
from . import HORIZONS


def _weighted(values: dict, weights: dict) -> float:
    """Sum(w * v) over keys present in `values`, renormalized by the weight
    actually used — a stock missing one horizon (e.g. too little history for
    the full 12-1 leg) isn't silently dragged toward zero for it."""
    total_w = total = 0.0
    for k, w in weights.items():
        v = values.get(k)
        if v is None:
            continue
        total += w * v
        total_w += w
    return total / total_w if total_w > 0 else 0.0


# ---------------------------------------------------------- single-stock ---
def breakout_score(breakout_raw: dict, horizon_weights: dict = MOM_HORIZON_WEIGHTS) -> float:
    grid = breakout_raw.get("horizons", {})
    vals = {h: grid[h]["b"] for h in grid if grid[h].get("b") is not None}
    return clip1(_weighted(vals, horizon_weights))


def speed_score(speed_raw: dict) -> float:
    return clip1(0.40 * speed_raw.get("align", 0.0) + 0.15 * speed_raw.get("price_align", 0.0)
                 + 0.25 * speed_raw.get("cross_recent", 0.0) + 0.20 * speed_raw.get("expansion_signed", 0.0))


def strength_score(strength_raw: dict) -> float:
    return clip1(0.30 * strength_raw.get("s_slope", 0.0) + 0.25 * strength_raw.get("s_accel", 0.0)
                 + 0.15 * strength_raw.get("s_gap", 0.0) + 0.15 * strength_raw.get("s_dgap", 0.0)
                 + 0.15 * strength_raw.get("s_quality", 0.0))


# Maps a factor key to a callable(raw_dict) -> score in [-1, 1]. "ts" and
# "xsec" are deliberately absent: they are cross-sectional and set on each
# row by `cross_sectional()` before `composite()` runs.
FACTOR_REGISTRY: dict[str, Callable[[dict], float]] = {
    "breakout": lambda raw: breakout_score(raw["breakout_raw"]),
    "speed": lambda raw: speed_score(raw["speed_raw"]),
    "strength": lambda raw: strength_score(raw["strength_raw"]),
}


# --------------------------------------------------------- cross-sectional ---
def cross_sectional(rows: list[dict], horizon_weights: dict = MOM_HORIZON_WEIGHTS,
                    winsor_p: float = 0.01) -> None:
    """Mutates `rows` in place, adding `ts_score` / `xsec_score` / `ts_grid`.
    Each row must carry `raw` = factors.build_all(...) output (or None for
    insufficient history, in which case it's simply skipped here and excluded
    downstream — never silently scored as neutral)."""
    priced = [r for r in rows if r.get("raw") is not None]
    for h in HORIZONS:
        m_vals, idxs = [], []
        for i, r in enumerate(priced):
            m = r["raw"]["ts_raw"]["horizons"].get(h, {}).get("m")
            if m is not None and math.isfinite(m):
                m_vals.append(m)
                idxs.append(i)
        if len(m_vals) < 5:
            continue
        s = pd.Series(m_vals, dtype=float)
        lo, hi = s.quantile(winsor_p), s.quantile(1.0 - winsor_p)
        wz = s.clip(lo, hi)
        pr = pct_ranks(wz.tolist())
        for j, i in enumerate(idxs):
            priced[i].setdefault("_ts_grid", {})[h] = {
                "m_raw": round(m_vals[j], 4), "m_winsorized": round(float(wz.iloc[j]), 4),
                "ts_signal": round(math.tanh(float(wz.iloc[j]) / 2.0), 4),
                "pctile": pr[j], "xsec_signal": round(inv_normal(pr[j]), 4),
            }
    for r in priced:
        grid = r.get("_ts_grid", {})
        ts_vals = {h: v["ts_signal"] for h, v in grid.items()}
        xs_vals = {h: v["xsec_signal"] for h, v in grid.items()}
        r["ts_score"] = clip1(_weighted(ts_vals, horizon_weights))
        r["xsec_score"] = clip1(_weighted(xs_vals, horizon_weights))
        r["ts_grid"] = grid
        r.pop("_ts_grid", None)


# -------------------------------------------------------------- composite ---
def state_for(score: float, states=MOM_STATES) -> str:
    for threshold, label in states:
        if score >= threshold:
            return label
    return states[-1][1]


def composite(rows: list[dict], weights: dict = MOM_WEIGHTS) -> None:
    """Mutates `rows` in place. Requires `cross_sectional()` to have already
    run (for ts_score/xsec_score). Rows with raw=None are left unscored
    (composite=None) so callers can exclude them cleanly rather than treat a
    missing score as a neutral one."""
    eq_w = {k: 1.0 / len(weights) for k in weights}
    for r in rows:
        raw = r.get("raw")
        if raw is None:
            r["composite"] = None
            r["state"] = None
            continue
        factor_scores = {}
        for k in weights:
            if k in FACTOR_REGISTRY:
                factor_scores[k] = FACTOR_REGISTRY[k](raw)
            elif k == "ts":
                factor_scores[k] = r.get("ts_score", 0.0)
            elif k == "xsec":
                factor_scores[k] = r.get("xsec_score", 0.0)
            else:
                factor_scores[k] = 0.0
        r["factor_scores"] = {k: round(v, 4) for k, v in factor_scores.items()}
        contributions = {k: round(20.0 * weights[k] * v, 4) for k, v in factor_scores.items()}
        r["contributions"] = contributions
        comp = max(-20.0, min(20.0, float(sum(contributions.values()))))
        r["composite"] = round(comp, 4)
        r["state"] = state_for(comp)
        eq_comp = max(-20.0, min(20.0, sum(20.0 * eq_w[k] * v for k, v in factor_scores.items())))
        r["composite_equal_weight"] = round(eq_comp, 4)
        r["breakout_grid"] = raw["breakout_raw"]["horizons"]


def correlations(rows: list[dict], factors: tuple = (), flag_threshold: float = 0.85) -> dict:
    """Spearman correlation matrix of the factor scores across the priced
    universe, plus a plain list of pairs above `flag_threshold`. This is the
    CHECK on the factor-weight redundancy assumption, not an assertion of
    it — the app renders it and flags any surprise."""
    from . import FACTORS
    factors = factors or FACTORS
    scored = [r["factor_scores"] for r in rows if r.get("factor_scores")]
    if len(scored) < 10:
        return {"n": len(scored), "matrix": {}, "flagged_pairs": []}
    df = pd.DataFrame(scored)[list(factors)]
    corr = df.corr(method="spearman")
    matrix = {a: {b: (None if pd.isna(corr.loc[a, b]) else round(float(corr.loc[a, b]), 3))
                  for b in factors} for a in factors}
    flagged = []
    for i, a in enumerate(factors):
        for b in factors[i + 1:]:
            v = matrix[a][b]
            if v is not None and abs(v) >= flag_threshold:
                flagged.append({"a": a, "b": b, "corr": v})
    return {"n": len(scored), "matrix": matrix, "flagged_pairs": flagged}

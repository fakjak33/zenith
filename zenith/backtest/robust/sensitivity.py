"""Parameter sensitivity: sweep 1 or 2 parameters and find the plateau.

The point (Carver; the 'gradient' view of Sharpe across parameters) is not to
pick the best cell but to see the *shape*: a broad high plateau is evidence of
a real effect, an isolated spike is evidence of fitting noise. The plateau
detector scores each cell by the mean of its neighbourhood and recommends the
centre of the best neighbourhood — not the raw peak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..spec import StrategySpec, run_spec
from .runner import pmap

METRICS = ("Sharpe", "CAGR", "MaxDD", "Calmar", "Sortino", "Turnover")


def default_values(spec: StrategySpec, param: str, n: int = 9) -> list:
    """Geometric spread around the current value, bounded by the rule's limits."""
    default, lo, hi, step = spec.strategy().__class__.params[param]
    cur = spec.params.get(param, default)
    a, b = max(lo, cur / 4), min(hi, cur * 4)
    if b <= a:
        a, b = lo, hi
    if isinstance(default, int) and isinstance(step, int):
        vals = np.unique(np.round(np.geomspace(max(a, 1), b, n)).astype(int))
        return [int(v) for v in vals]
    return [float(round(v, 4)) for v in np.linspace(a, b, n)]


def sweep(spec: StrategySpec, ohlcv: pd.DataFrame, px: str, xs: list, py: str | None = None,
          ys: list | None = None, aux: dict | None = None, cost_mult: float = 1.0,
          progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Run the grid; returns (long table with x[, y] + metrics, trial hashes)."""
    combos = [(x, y) for y in (ys or [None]) for x in xs]

    def one(c):
        x, y = c
        kw = {px: x}
        if py:
            kw[py] = y
        s = spec.with_params(**kw)
        try:
            st = run_spec(s, ohlcv, aux, cost_mult=cost_mult).stats
        except Exception:  # noqa: BLE001 — an invalid combo is a hole, not a crash
            st = {m: np.nan for m in METRICS}
        return {"x": x, "y": y, **{m: st.get(m, np.nan) for m in METRICS}}, s.hash()

    out = pmap(one, combos, progress)
    df = pd.DataFrame([o[0] for o in out])
    return df, [o[1] for o in out]


def to_grid(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df["y"].isna().all():
        return df.set_index("x")[[metric]].T
    return df.pivot(index="y", columns="x", values=metric)


def _neigh_mean(z: np.ndarray, radius: int = 1) -> np.ndarray:
    """NaN-aware mean over a (2r+1)^2 window (edges use what exists)."""
    z = np.atleast_2d(z).astype(float)
    out = np.full_like(z, np.nan)
    rows, cols = z.shape
    for i in range(rows):
        for j in range(cols):
            w = z[max(0, i - radius):i + radius + 1, max(0, j - radius):j + radius + 1]
            if np.isfinite(w).any():
                out[i, j] = np.nanmean(w)
    return out


def plateau(grid: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """Score the surface. Returns peak, plateau pick, robustness ratio and maps."""
    z = grid.to_numpy(dtype=float)
    zz = z if higher_is_better else -z
    sm = _neigh_mean(zz)
    if not np.isfinite(zz).any():
        return {"ok": False}
    pi = np.unravel_index(np.nanargmax(zz), zz.shape)
    qi = np.unravel_index(np.nanargmax(sm), sm.shape)
    peak, plat = float(zz[pi]), float(sm[qi])
    gy, gx = np.gradient(np.nan_to_num(zz, nan=np.nanmean(zz))) if min(zz.shape) > 1 else \
        (np.zeros_like(zz), np.gradient(np.nan_to_num(zz[0], nan=np.nanmean(zz)))[None, :])
    thresh = plat - 0.2 * abs(plat)
    region = sm >= thresh
    sign = 1 if higher_is_better else -1
    return {
        "ok": True,
        "peak": {"x": grid.columns[pi[1]], "y": grid.index[pi[0]], "value": sign * peak},
        "plateau": {"x": grid.columns[qi[1]], "y": grid.index[qi[0]], "value": sign * plat},
        # 1.0 = the neighbourhood is as good as the peak (robust); near 0 = lone spike
        "robustness": float(plat / peak) if peak > 1e-9 else 0.0,
        "smoothed": pd.DataFrame(sign * sm, index=grid.index, columns=grid.columns),
        "gradient": pd.DataFrame(np.hypot(gx, gy), index=grid.index, columns=grid.columns),
        "region": pd.DataFrame(region, index=grid.index, columns=grid.columns),
        "share_positive": float(np.mean(z[np.isfinite(z)] > 0)) if higher_is_better else np.nan,
    }

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

from dataclasses import replace

from ..spec import FILTERS, StrategySpec, run_spec
from .runner import pmap

METRICS = ("Sharpe", "CAGR", "MaxDD", "Calmar", "Sortino", "Turnover")

# Spec-level knobs beyond the rule's own parameters: (default, lo, hi, step), label.
# Names are "s:<field>" (a StrategySpec field) or "f:<filter>:<param>" (an enabled filter's parameter).
SPEC_KNOBS = {
    "s:target_vol": ((0.20, 0.05, 0.60, 0.01), "target vol"),
    "s:fixed_pct": ((1.0, 0.05, 3.0, 0.05), "notional % (fixed)"),
    "s:risk_pct": ((0.01, 0.001, 0.20, 0.001), "risk % per trade"),
    "s:atr_stop_mult": ((3.0, 0.5, 10.0, 0.5), "stop x ATR"),
    "s:atr_tp_mult": ((6.0, 0.5, 20.0, 0.5), "target x ATR"),
    "s:buffer": ((0.10, 0.0, 0.5, 0.05), "no-trade buffer"),
    "s:max_leverage": ((2.0, 0.5, 10.0, 0.5), "max leverage"),
}


def knob_label(name: str) -> str:
    if name in SPEC_KNOBS:
        return SPEC_KNOBS[name][1]
    if name.startswith("f:"):
        _, fk, pn = name.split(":")
        return f"{FILTERS[fk]['label'].lower()} · {pn}"
    return name


def knob_choices(spec: StrategySpec) -> list[str]:
    """Every sweepable knob for this spec: rule params, then the parameters of enabled filters,
    then the sizing/exit knobs that the chosen sizing mode and exits actually use."""
    out = list(spec.strategy().__class__.params)
    for f in spec.filters:
        if f.get("on", True):
            out += [f"f:{f['key']}:{pn}" for pn in FILTERS[f["key"]]["params"]]
    mode = spec.sizing_mode
    out += {"vol_target": ["s:target_vol"], "fixed_pct": ["s:fixed_pct"],
            "fixed_risk": ["s:risk_pct"]}.get(mode, [])
    if spec.use_atr_stop or mode == "fixed_risk":
        out.append("s:atr_stop_mult")
    if spec.use_atr_tp:
        out.append("s:atr_tp_mult")
    out += ["s:buffer"]
    if spec.max_leverage:
        out.append("s:max_leverage")
    return out


def knob_spec(spec: StrategySpec, name: str) -> tuple:
    """(default, lo, hi, step) and current value for a knob."""
    if name in SPEC_KNOBS:
        t = SPEC_KNOBS[name][0]
        return t, getattr(spec, name[2:])
    if name.startswith("f:"):
        _, fk, pn = name.split(":")
        t = FILTERS[fk]["params"][pn]
        cur = next((f.get("params", {}).get(pn, t[0]) for f in spec.filters if f["key"] == fk), t[0])
        return t, cur
    t = spec.strategy().__class__.params[name]
    return t, spec.params.get(name, t[0])


def set_knob(spec: StrategySpec, name: str, value) -> StrategySpec:
    if name in SPEC_KNOBS:
        return replace(spec, **{name[2:]: type(SPEC_KNOBS[name][0][0])(value)})
    if name.startswith("f:"):
        _, fk, pn = name.split(":")
        fl = [({**f, "params": {**f.get("params", {}), pn: value}} if f["key"] == fk else f) for f in spec.filters]
        return spec.with_filters(fl)
    return spec.with_params(**{name: value})


def default_values(spec: StrategySpec, param: str, n: int = 9) -> list:
    """Geometric spread around the current value, bounded by the knob's limits."""
    (default, lo, hi, step), cur = knob_spec(spec, param)
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
        s = set_knob(spec, px, x)
        if py:
            s = set_knob(s, py, y)
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

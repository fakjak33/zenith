"""Filter ablation: does each filter earn its place?

Runs the strategy with no filters, each filter alone, all filters, and all
filters minus each one. For every variant the Sharpe difference against the
no-filter base gets a paired block-bootstrap confidence interval (same resampled
dates for both legs), so a filter that only looks good by chance is flagged
'adds no value' when its interval spans zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..spec import StrategySpec, run_spec
from .runner import pmap


def _sharpe(x: np.ndarray, periods: int) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(periods)) if s > 1e-12 else 0.0


def paired_sharpe_ci(a: pd.Series, b: pd.Series, periods: int, n: int = 400, block: int = 20,
                     seed: int = 5, level: float = 0.9) -> tuple[float, float, float]:
    """(delta, lo, hi) for Sharpe(a) − Sharpe(b) via a circular block bootstrap."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    x, y = df["a"].to_numpy(), df["b"].to_numpy()
    m = len(x)
    if m < 2 * block:
        d = _sharpe(x, periods) - _sharpe(y, periods)
        return d, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / block))
    deltas = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m] % m
        deltas[i] = _sharpe(x[idx], periods) - _sharpe(y[idx], periods)
    q = (1 - level) / 2
    return _sharpe(x, periods) - _sharpe(y, periods), float(np.quantile(deltas, q)), float(np.quantile(deltas, 1 - q))


def ablation(spec: StrategySpec, ohlcv: pd.DataFrame, aux: dict | None = None, progress=None) -> tuple[pd.DataFrame, list]:
    fs = [dict(f, on=True) for f in spec.filters]
    labels = {f["key"]: f["key"] for f in fs}
    variants = [("no filters", [])]
    variants += [(f"only {labels[f['key']]}", [f]) for f in fs]
    if len(fs) > 1:
        variants += [("all filters", fs)]
        variants += [(f"all − {labels[f['key']]}", [g for g in fs if g is not f]) for f in fs]

    def one(v):
        name, flist = v
        s = spec.with_filters(flist)
        return name, s.hash(), run_spec(s, ohlcv, aux)

    out = pmap(one, variants, progress)
    base = out[0][2]
    rows = []
    for name, h, res in out:
        d, lo, hi = paired_sharpe_ci(res.returns, base.returns, res.periods)
        exposure = float((res.position != 0).mean())
        verdict = "base" if name == "no filters" else (
            "adds value" if lo > 0 else "hurts" if hi < 0 else "no clear value")
        rows.append({"variant": name, "Sharpe": res.stats["Sharpe"], "CAGR": res.stats["CAGR"],
                     "MaxDD": res.stats["MaxDD"], "exposure": exposure, "ΔSharpe": d, "CI lo": lo, "CI hi": hi,
                     "verdict": verdict})
    return pd.DataFrame(rows), [h for _, h, _ in out]

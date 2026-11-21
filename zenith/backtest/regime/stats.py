"""Regime statistics: performance by regime, transitions, spell lengths.

Alignment rule: a strategy's return on bar t was earned by the position chosen
at the close of t−1, so it is attributed to the regime label *known at t−1*
(``labels.shift(1)``). That makes the table answer the tradeable question
"how did I do while the label said X".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(x: np.ndarray, periods: int) -> float:
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else np.nan


def _boot_ci(x: np.ndarray, periods: int, n: int = 400, block: int = 5, seed: int = 7) -> tuple[float, float]:
    """95% CI of the Sharpe from a moving-block bootstrap (blocks keep short-run dependence)."""
    if len(x) < 20:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = max(1, min(block, len(x) // 4))
    starts = np.arange(len(x) - b + 1)
    k = int(np.ceil(len(x) / b))
    out = np.empty(n)
    for i in range(n):
        idx = (rng.choice(starts, k)[:, None] + np.arange(b)).ravel()[: len(x)]
        out[i] = _sharpe(x[idx], periods)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def spells(labels: pd.Series) -> pd.DataFrame:
    """Contiguous runs of one label: start, end, label, bars."""
    lab = labels.dropna()
    if lab.empty:
        return pd.DataFrame(columns=["start", "end", "label", "bars"])
    grp = (lab != lab.shift()).cumsum()
    g = lab.groupby(grp)
    return pd.DataFrame({"start": g.apply(lambda s: s.index[0]), "end": g.apply(lambda s: s.index[-1]),
                         "label": g.first(), "bars": g.size()}).reset_index(drop=True)


def by_regime(r: pd.Series, labels: pd.Series, periods: int, bench: pd.Series | None = None,
              order: tuple = (), boot: bool = True) -> pd.DataFrame:
    lab = labels.shift(1).reindex(r.index)
    sp = spells(labels)
    rows = []
    keys = [k for k in order if k in set(lab.dropna())] + sorted(set(lab.dropna()) - set(order))
    total = lab.notna().sum()
    for k in keys:
        m = (lab == k).to_numpy()
        x = r.to_numpy()[m]
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue
        lo, hi = _boot_ci(x, periods) if boot else (np.nan, np.nan)
        row = {"regime": k, "time": m.sum() / max(total, 1), "bars": int(m.sum()),
               "ann_return": float(x.mean() * periods), "vol": float(x.std(ddof=1) * np.sqrt(periods)),
               "sharpe": _sharpe(x, periods), "ci_lo": lo, "ci_hi": hi,
               "hit": float((x > 0).sum() / max((x != 0).sum(), 1)),
               "spells": int((sp["label"] == k).sum()),
               "avg_bars": float(sp.loc[sp["label"] == k, "bars"].mean()) if (sp["label"] == k).any() else np.nan}
        if bench is not None:
            bx = bench.reindex(r.index).to_numpy()[m]
            row["bench_sharpe"] = _sharpe(bx[np.isfinite(bx)], periods)
        rows.append(row)
    return pd.DataFrame(rows)


def transition_matrix(labels: pd.Series, order: tuple = ()) -> pd.DataFrame:
    """P(label tomorrow | label today), rows sum to 1."""
    lab = labels.dropna()
    if len(lab) < 2:
        return pd.DataFrame()
    ct = pd.crosstab(lab.iloc[:-1].to_numpy(), lab.iloc[1:].to_numpy())
    keys = [k for k in order if k in ct.index] + [k for k in ct.index if k not in order]
    ct = ct.reindex(index=keys, columns=keys, fill_value=0)
    return ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)


def two_way(r: pd.Series, a: pd.Series, b: pd.Series, periods: int) -> pd.DataFrame:
    """Sharpe by (regime A, regime B) — a 2-D regime heatmap."""
    la, lb = a.shift(1).reindex(r.index), b.shift(1).reindex(r.index)
    df = pd.DataFrame({"r": r, "a": la, "b": lb}).dropna()
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["a", "b"])["r"].apply(lambda x: _sharpe(x.to_numpy(), periods) if len(x) >= 20 else np.nan) \
        .unstack()


def current(labels: pd.Series) -> tuple[str | None, int]:
    """Latest label and how many bars it has lasted."""
    lab = labels.dropna()
    if lab.empty:
        return None, 0
    sp = spells(lab)
    return str(sp["label"].iloc[-1]), int(sp["bars"].iloc[-1])

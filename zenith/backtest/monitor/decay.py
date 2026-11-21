"""DECAY: when to stop a strategy (SharpeEdge) — noise, bug or real decay?

Reference = the period the strategy was researched on; monitored = what came
after (the revealed out-of-sample, or a pretend go-live date inside the
in-sample for practice). Every tool asks the same question a different way:
is the monitored record inconsistent with the reference record?

  CUSUM            Page (1954): cumulative evidence of a drop in the mean
  bootstrap cone   where the reference returns say equity should be by now
  month rank       each live month's percentile among reference months
  DD breaker       the drawdown that is a 1-in-20 event over this horizon
  rolling alpha    alpha vs buy & hold with HAC errors, through time
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.montecarlo import auto_block_length


def _sums(r: pd.Series, agg: int) -> pd.Series:
    n = len(r) // agg * agg
    x = r.iloc[len(r) - n:] if n else r.iloc[:0]
    return x.groupby(np.arange(len(x)) // agg).sum().set_axis(x.index[agg - 1::agg]) if n else x


def cusum(ref: pd.Series, live: pd.Series, k: float = 0.25, h: float = 8.0, agg: int = 5) -> dict:
    """One-sided CUSUM for a downward shift in the mean (Page 1954), on non-overlapping ``agg``-bar sums in
    reference-σ units. Weekly sums with k = 0.25, h = 8 keep false alarms rare (several years between them for
    a strategy that still works) while a mean drop of ~½σ a week is flagged within months."""
    rs, ls = _sums(ref.dropna(), agg), _sums(live.fillna(0.0), agg)
    mu, sd = rs.mean(), rs.std()
    if not np.isfinite(sd) or sd <= 0 or ls.empty:
        return {"s": pd.Series(dtype=float), "alarm": None, "h": h}
    live = ls
    z = ((mu - live) / sd).to_numpy()
    s = np.zeros(len(z))
    for i, x in enumerate(z):
        s[i] = max(0.0, (s[i - 1] if i else 0.0) + x - k)
    s = pd.Series(s, index=live.index)
    hit = s[s > h]
    return {"s": s, "alarm": hit.index[0] if len(hit) else None, "h": h}


def _paths(x: np.ndarray, n_paths: int, length: int, seed: int = 0) -> np.ndarray:
    """Stationary-bootstrap paths (Politis-White block length)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / max(1.0, auto_block_length(x))
    T = len(x)
    jump = rng.random((n_paths, length)) < p
    starts = rng.integers(0, T, (n_paths, length))
    idx = np.empty((n_paths, length), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, length):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return x[idx]


def cone(ref: pd.Series, live: pd.Series, n_paths: int = 1000, seed: int = 0) -> pd.DataFrame:
    eq = np.cumprod(1 + _paths(ref.dropna().to_numpy(), n_paths, len(live), seed), axis=1)
    q = np.percentile(eq, [5, 25, 50, 75, 95], axis=0).T
    out = pd.DataFrame(q, index=live.index, columns=["p5", "p25", "p50", "p75", "p95"])
    out["actual"] = (1 + live.fillna(0)).cumprod().to_numpy()
    return out


def dd_breaker(ref: pd.Series, horizon: int, q: float = 5.0, n_paths: int = 1000, seed: int = 1) -> float:
    eq = np.cumprod(1 + _paths(ref.dropna().to_numpy(), n_paths, max(horizon, 2), seed), axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_paths, 1)), eq], axis=1), axis=1)[:, 1:]
    return float(np.percentile((eq / peak - 1).min(axis=1), q))


def month_ranks(ref: pd.Series, live: pd.Series) -> pd.DataFrame:
    rm = (1 + ref).resample("ME").prod() - 1
    lm = (1 + live).resample("ME").prod() - 1
    return pd.DataFrame({"return": lm.to_numpy(), "percentile": [float((rm <= v).mean()) for v in lm]},
                        index=lm.index)


def rolling_alpha(r: pd.Series, bench: pd.Series, window: int = 252, step: int = 21, periods: int = 252,
                  lags: int = 5) -> pd.DataFrame:
    """Annualised alpha (with a 95% HAC band) and beta of r on bench over a rolling window."""
    import statsmodels.api as sm
    d = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    rows = []
    for end in range(window, len(d) + 1, step):
        w = d.iloc[end - window: end]
        if w["b"].std() <= 0:
            continue
        fit = sm.OLS(w["r"], sm.add_constant(w["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        a, se = fit.params["const"], fit.bse["const"]
        rows.append({"date": w.index[-1], "alpha": a * periods, "lo": (a - 1.96 * se) * periods,
                     "hi": (a + 1.96 * se) * periods, "beta": fit.params["b"]})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def verdict(cs: dict, cn: pd.DataFrame, breaker: float, live: pd.Series, ranks: pd.DataFrame) -> dict:
    eq = (1 + live.fillna(0)).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    flags = {
        "CUSUM alarm": cs.get("alarm") is not None,
        "below the 5% cone": bool(len(cn) and cn["actual"].iloc[-1] < cn["p5"].iloc[-1]),
        "drawdown past the breaker": dd < breaker,
        "3+ bottom-decile months": bool(len(ranks) and int((ranks["percentile"] < 0.1).sum()) >= 3),
    }
    n = sum(flags.values())
    return {"flags": flags, "state": "WORKING" if n == 0 else "WATCH" if n == 1 else "LIKELY BROKEN", "live_dd": dd}

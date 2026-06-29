"""Replicate Gupta & Kelly (2019) on public academic factor returns.

  python -m zenith.cas.backtest.factor_momentum

For each factor return series we (1) fit an AR(1) to measure own-return
persistence (the paper's "59 positive / 49 significant" result) and (2) build a
time-series momentum strategy that goes long/short the factor on the sign of its
trailing 12-month return. We combine the per-factor strategies at equal
volatility into a TS factor-momentum portfolio and report its Sharpe (the paper's
headline was ~0.84). Results are written to data/cas/backtest_factor_momentum.json
for the UI and to seed a registry research note.

The AR(1) and Sharpe maths are dependency-free (numpy only) so they're unit-
testable without network access; only data loading touches pandas-datareader.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from . import factor_data
from .. import store_cas, registry

T_CRIT = 1.96            # ~5% two-sided significance


def ar1(series) -> tuple[float, float]:
    """OLS AR(1): r_t = a + b·r_{t-1}. Returns (b, t_stat); (nan, nan) if too short."""
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 24:
        return (float("nan"), float("nan"))
    x = s.iloc[:-1].to_numpy()
    y = s.iloc[1:].to_numpy()
    n = len(y)
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx == 0.0:
        return (0.0, 0.0)
    b = float(((x - xbar) * (y - ybar)).sum() / sxx)
    a = ybar - b * xbar
    resid = y - (a + b * x)
    dof = n - 2
    if dof <= 0:
        return (b, float("nan"))
    s2 = float((resid ** 2).sum()) / dof
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    t = b / se if se > 0 else float("nan")
    return (b, t)


def ts_strategy(series, lookback: int = 12) -> pd.Series:
    """Time-series momentum on a monthly factor series: position = sign of the
    trailing ``lookback``-month return, lagged one month (no look-ahead)."""
    s = pd.Series(series).dropna().astype(float)
    formation = s.rolling(lookback).sum().shift(1)
    pos = np.sign(formation)
    return (pos * s).dropna()


def sharpe(returns, periods_per_year: int = 12) -> float:
    r = pd.Series(returns).dropna().astype(float)
    sd = r.std()
    if len(r) < 12 or sd == 0 or math.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def _combine_equal_vol(strat: pd.DataFrame) -> pd.Series:
    """Average the per-factor strategy returns after scaling each to unit vol."""
    strat = strat.dropna(how="all")
    if strat.empty:
        return pd.Series(dtype=float)
    vols = strat.std().replace(0, np.nan)
    weights = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = strat.mul(weights, axis=1)
    return scaled.mean(axis=1, skipna=True).dropna()


def run() -> dict:
    """Load factors, compute AR(1) + TS-momentum stats, persist JSON + seed note."""
    factors = factor_data.load_all()
    per: dict[str, dict] = {}
    by_region: dict[str, list[float]] = defaultdict(list)
    strat = pd.DataFrame()

    for label, s in factors.items():
        b, t = ar1(s)
        sret = ts_strategy(s)
        sh = sharpe(sret)
        positive = bool(b > 0) if not math.isnan(b) else False
        significant = bool(positive and not math.isnan(t) and abs(t) > T_CRIT)
        region = label.split(":", 1)[0]
        per[label] = {
            "region": region,
            "ar1": None if math.isnan(b) else round(b, 4),
            "t_stat": None if math.isnan(t) else round(t, 3),
            "positive": positive,
            "significant": significant,
            "ts_sharpe": None if math.isnan(sh) else round(sh, 3),
            "n_months": int(pd.Series(s).dropna().shape[0]),
        }
        if not math.isnan(sh):
            by_region[region].append(sh)
        if len(sret):
            strat[label] = sret

    combo = _combine_equal_vol(strat)
    out = {
        "as_of": date.today().isoformat(),
        "source": "Ken French Data Library (+ AQR if cached)",
        "n_factors": len(per),
        "n_positive": sum(v["positive"] for v in per.values()),
        "n_significant": sum(v["significant"] for v in per.values()),
        "combined_ts_sharpe": None if math.isnan(sharpe(combo)) else round(sharpe(combo), 3),
        "by_region_mean_sharpe": {r: round(float(np.mean(v)), 3) for r, v in by_region.items() if v},
        "factors": per,
    }
    store_cas.save("backtest", out)
    _seed_note(out)
    return out


def _seed_note(out: dict) -> None:
    """Log a research note against the style families so the result shows in the
    Models & notes UI (best-effort; never breaks the compute run)."""
    try:
        if not out.get("n_factors"):
            return
        abstract = (
            f"Factor-momentum backtest ({out['as_of']}, {out['source']}): "
            f"{out['n_positive']}/{out['n_factors']} factors show positive AR(1) "
            f"autocorrelation, {out['n_significant']} significant (|t|>1.96). "
            f"Combined TS factor-momentum Sharpe = {out.get('combined_ts_sharpe')}. "
            "Replicates Gupta & Kelly (2019), 'Factor Momentum Everywhere'."
        )
        registry.add_note("style_ts_mom", "Factor Momentum Everywhere — replication", abstract)
    except Exception:
        pass


def main() -> None:
    out = run()
    print(f"[backtest] factors={out['n_factors']} positive={out['n_positive']} "
          f"significant={out['n_significant']} combined_ts_sharpe={out['combined_ts_sharpe']}")
    for label, v in sorted(out["factors"].items()):
        print(f"  {label:>10}: AR1={v['ar1']} t={v['t_stat']} "
              f"{'sig' if v['significant'] else '   '} sharpe={v['ts_sharpe']} n={v['n_months']}")


if __name__ == "__main__":
    main()

"""How diversifying is a strategy against what you already hold?

Everything is computed on aligned per-bar returns of the strategy (``s``) and an
underlying or portfolio (``u``):

- correlation and beta, overall, rolling, and split into up / down markets;
- **crisis correlation**: over the underlying's worst decile of months, which is
  when diversification is needed and when correlations usually rise;
- the **add-to-portfolio test**. Adding a little of ``s`` to ``u`` raises the
  Sharpe ratio exactly when SR_s > ρ · SR_u, or equivalently when
  μ_s > β_s|u · μ_u: the strategy has positive alpha against what you hold
  (Elton & Gruber; Multi-Strat Mark's napkin rule);
- a **blend frontier**: Sharpe, CAGR and max drawdown of w·s + (1−w)·u for
  w from 0 to 1, rebalanced every bar, with s scaled to u's volatility first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(s: pd.Series, u: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"s": s, "u": u}).replace([np.inf, -np.inf], np.nan).dropna()


def _sharpe(x: pd.Series, periods: int) -> float:
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else 0.0


def _maxdd(x: pd.Series) -> float:
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def _cagr(x: pd.Series, periods: int) -> float:
    if len(x) < 2:
        return 0.0
    g = float((1 + x).prod())
    return g ** (periods / len(x)) - 1 if g > 0 else -1.0


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 5 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def blend_frontier(s: pd.Series, u: pd.Series, periods: int, steps: int = 21,
                   vol_match: bool = True) -> pd.DataFrame:
    """Stats of w·s + (1−w)·u for w in [0, 1]. With ``vol_match`` the strategy is first scaled to the
    underlying's volatility, so w is a share of *risk* rather than a share of dollars (otherwise a
    low-vol strategy 'improves' the blend just by de-risking it)."""
    df = _align(s, u)
    if vol_match and df["s"].std() > 1e-12:
        df["s"] = df["s"] * (df["u"].std() / df["s"].std())
    rows = []
    for w in np.linspace(0, 1, steps):
        x = w * df["s"] + (1 - w) * df["u"]
        rows.append({"w": float(w), "Sharpe": _sharpe(x, periods), "CAGR": _cagr(x, periods),
                     "Vol": float(x.std() * np.sqrt(periods)), "MaxDD": _maxdd(x)})
    return pd.DataFrame(rows)


def rolling_corr(s: pd.Series, u: pd.Series, window: int) -> pd.Series:
    df = _align(s, u)
    return df["s"].rolling(window, min_periods=max(10, window // 2)).corr(df["u"])


def analyse(s: pd.Series, u: pd.Series, periods: int) -> dict:
    """Headline diversification numbers. ``periods`` = bars per year."""
    df = _align(s, u)
    if len(df) < 20:
        return {"ok": False, "n": len(df)}
    a, b = df["s"], df["u"]
    var_u = float(b.var())
    beta = float(a.cov(b) / var_u) if var_u > 1e-16 else float("nan")
    rho = _corr(a, b)
    up, dn = b > 0, b < 0
    sr_s, sr_u = _sharpe(a, periods), _sharpe(b, periods)

    # crisis: worst decile of the underlying's months
    m = (1 + df).resample("ME").prod() - 1
    crisis = {"corr": float("nan"), "s_mean": float("nan"), "u_mean": float("nan"), "months": 0}
    if len(m) >= 20:
        cut = m["u"].quantile(0.10)
        worst = m[m["u"] <= cut]
        crisis = {"corr": _corr(worst["s"], worst["u"]) if len(worst) >= 5 else float("nan"),
                  "s_mean": float(worst["s"].mean()), "u_mean": float(worst["u"].mean()),
                  "months": int(len(worst)), "hit": float((worst["s"] > 0).mean())}

    alpha = float((a.mean() - beta * b.mean()) * periods) if np.isfinite(beta) else float("nan")
    hurdle = rho * sr_u
    fr = blend_frontier(a, b, periods)
    best = fr.loc[fr["Sharpe"].idxmax()]
    return {
        "ok": True, "n": len(df), "corr": rho, "beta": beta,
        "corr_up": _corr(a[up], b[up]), "corr_down": _corr(a[dn], b[dn]),
        "crisis": crisis, "sharpe_s": sr_s, "sharpe_u": sr_u,
        "hurdle": hurdle, "adds": bool(sr_s > hurdle), "alpha": alpha,
        "frontier": fr, "best_w": float(best["w"]), "best_sharpe": float(best["Sharpe"]),
    }

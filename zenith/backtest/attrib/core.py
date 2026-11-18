"""Return attribution: where did the strategy's returns come from?

- **alpha / beta** vs the chosen benchmark (OLS, Newey-West t-stats), the share
  of the return that beta alone explains, and the beta-hedged residual stream;
- **capture**: monthly up- vs down-market behaviour and convexity
  (Treynor-Mazuy γ: positive = pays off in big moves either way, the
  trend-follower 'smile');
- **regimes**: returns split by benchmark trend, vol tercile, drawdown state
  and VIX bucket (descriptive, ex-post — not a trading signal);
- **legs**: long vs short contribution; **yearly** returns;
- **factors**: Fama-French 5 + momentum regression on monthly returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ OLS ----
def ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with Newey-West (HAC) standard errors. X must include the constant."""
    n, k = X.shape
    if n <= k + 2:
        return {"coef": np.full(k, np.nan), "t": np.full(k, np.nan), "se": np.full(k, np.nan), "r2": np.nan, "n": n}
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, b / se, np.nan)
    return {"coef": b, "t": t, "se": se, "r2": float(r2), "n": n, "resid": e}


def _align(r: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()


# ------------------------------------------------------------ alpha/beta ----
def alpha_beta(r: pd.Series, b: pd.Series, periods: int, rf: pd.Series | None = None) -> dict:
    """Regress strategy on benchmark (both in excess of cash when ``rf`` given)."""
    df = _align(r, b)
    if rf is not None:
        rfa = rf.reindex(df.index).fillna(0.0)
        df = df.assign(r=df["r"] - rfa, b=df["b"] - rfa)
    X = np.column_stack([np.ones(len(df)), df["b"].to_numpy()])
    fit = ols(df["r"].to_numpy(), X)
    a, beta = fit["coef"]
    ta, tb = fit["t"]
    mean_r, mean_b = df["r"].mean(), df["b"].mean()
    beta_part = beta * mean_b
    active = df["r"] - df["b"]
    te = active.std(ddof=1)
    return {
        "alpha_ann": float(a * periods), "alpha_t": float(ta), "beta": float(beta), "beta_t": float(tb),
        "r2": fit["r2"], "corr": float(df["r"].corr(df["b"])) if len(df) > 2 else np.nan,
        # share of the strategy's average return that beta × benchmark explains (can exceed 1 or go negative)
        "beta_share": float(beta_part / mean_r) if abs(mean_r) > 1e-12 else np.nan,
        "info_ratio": float(active.mean() / te * np.sqrt(periods)) if te > 1e-12 else np.nan,
        "beta_ret": df["b"] * beta,
        "hedged": df["r"] - df["b"] * beta,
        "n": len(df),
    }


def rolling_beta(r: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = _align(r, b)
    cov = df["r"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    return (cov / var).rename("beta")


# --------------------------------------------------------------- capture ----
def monthly(r: pd.Series) -> pd.Series:
    return (1 + r).resample("ME").prod() - 1


def capture(r: pd.Series, b: pd.Series) -> dict:
    m = _align(monthly(r), monthly(b))
    up, dn = m[m["b"] > 0], m[m["b"] < 0]
    X = np.column_stack([np.ones(len(m)), m["b"], m["b"] ** 2])
    tm = ols(m["r"].to_numpy(), X, lags=3)
    return {
        "months": m,
        "up_capture": float(up["r"].mean() / up["b"].mean()) if len(up) and up["b"].mean() else np.nan,
        "down_capture": float(dn["r"].mean() / dn["b"].mean()) if len(dn) and dn["b"].mean() else np.nan,
        "up_hit": float((up["r"] > 0).mean()) if len(up) else np.nan,
        "down_hit": float((dn["r"] > 0).mean()) if len(dn) else np.nan,
        "up_mean": float(up["r"].mean()) if len(up) else np.nan,
        "down_mean": float(dn["r"].mean()) if len(dn) else np.nan,
        "gamma": float(tm["coef"][2]), "gamma_t": float(tm["t"][2]),
        "tm_coef": tm["coef"],
    }


# --------------------------------------------------------------- regimes ----
def regime_labels(bench_ret: pd.Series, periods: int, vix: pd.Series | None = None) -> pd.DataFrame:
    """Ex-post regime labels from the benchmark (and VIX when available)."""
    px = (1 + bench_ret.fillna(0)).cumprod()
    n_trend = max(10, int(round(periods * 200 / 252)))
    sma = px.rolling(n_trend, min_periods=n_trend).mean()
    trend = np.where(px > sma, "uptrend", "downtrend")
    trend = pd.Series(trend, index=px.index).where(sma.notna())
    w = max(5, int(round(periods * 63 / 252)))
    vol = bench_ret.rolling(w, min_periods=w).std()
    terc = pd.qcut(vol.rank(method="first"), 3, labels=["calm", "normal", "stressed"]) if vol.notna().sum() > 30 \
        else pd.Series(index=px.index, dtype=object)
    dd = px / px.cummax() - 1
    dds = pd.Series(np.where(dd < -0.10, "drawdown >10%", "near highs"), index=px.index)
    out = pd.DataFrame({"benchmark trend": trend, "benchmark vol": terc.astype(object), "benchmark state": dds})
    if vix is not None and not vix.empty:
        v = vix.reindex(px.index).ffill()
        out["VIX"] = pd.cut(v, [0, 15, 20, 30, 1000], labels=["<15", "15–20", "20–30", "30+"]).astype(object)
    return out


REGIME_ORDER = ("downtrend", "uptrend", "calm", "normal", "stressed", "near highs", "drawdown >10%",
                "<15", "15–20", "20–30", "30+")


def by_regime(r: pd.Series, labels: pd.DataFrame, periods: int) -> pd.DataFrame:
    rows = []
    rr = r.reindex(labels.index)
    total = rr.sum()
    for col in labels.columns:
        groups = dict(list(rr.groupby(labels[col])))
        order = [k for k in REGIME_ORDER if k in groups] + [k for k in groups if k not in REGIME_ORDER]
        for lab in order:
            g = groups[lab]
            if len(g) < 5:
                continue
            sd = g.std(ddof=1)
            rows.append({"dimension": col, "regime": str(lab), "time": len(g) / len(rr.dropna()),
                         "ann_return": g.mean() * periods,
                         "sharpe": g.mean() / sd * np.sqrt(periods) if sd > 1e-12 else np.nan,
                         "share_of_pnl": g.sum() / total if abs(total) > 1e-12 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ legs ----
def legs(returns: pd.Series, position: pd.Series, periods: int) -> pd.DataFrame:
    """Split P&L by the side held over each bar (position is lagged like the engine)."""
    side = np.sign(position.shift(1).fillna(0.0)).reindex(returns.index)
    total = returns.sum()
    rows = []
    for lab, s in (("long", 1), ("short", -1), ("flat", 0)):
        g = returns[side == s]
        rows.append({"leg": lab, "time": len(g) / len(returns) if len(returns) else np.nan,
                     "ann_contrib": g.sum() / (len(returns) / periods) if len(returns) else np.nan,
                     "share_of_pnl": g.sum() / total if abs(total) > 1e-12 else np.nan,
                     "hit_rate": float((g > 0).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def yearly(r: pd.Series, b: pd.Series) -> pd.DataFrame:
    y = pd.DataFrame({"strategy": (1 + r).resample("YE").prod() - 1, "benchmark": (1 + b).resample("YE").prod() - 1})
    y.index = y.index.year
    return y.dropna(how="all")


# --------------------------------------------------------------- factors ----
FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML")


def factor_regression(r: pd.Series, factors: dict[str, pd.Series], region: str = "US") -> dict:
    """Monthly regression on Fama-French 5 + momentum (``{region}:{col}`` keys)."""
    cols = {c: factors.get(f"{region}:{c}") for c in FACTOR_COLS}
    cols = {c: s for c, s in cols.items() if s is not None and not s.empty}
    if not cols:
        return {"ok": False}
    f = pd.DataFrame(cols)
    f.index = pd.to_datetime(f.index).to_period("M")
    m = monthly(r)
    m.index = m.index.to_period("M")
    df = pd.concat([m.rename("r"), f], axis=1).dropna()
    if len(df) < 24:
        return {"ok": False, "n": len(df)}
    names = list(cols)
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in names])
    fit = ols(df["r"].to_numpy(), X, lags=3)
    return {"ok": True, "names": ["alpha"] + names, "coef": fit["coef"], "t": fit["t"], "se": fit["se"],
            "r2": fit["r2"], "n": len(df), "alpha_ann": float(fit["coef"][0] * 12),
            "start": str(df.index[0]), "end": str(df.index[-1])}

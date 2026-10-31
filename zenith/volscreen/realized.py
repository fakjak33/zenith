"""Realized-volatility estimators on daily OHLC, annualised (x252).

  close-to-close   sqrt(mean r^2)                                      (zero-mean)
  Parkinson 1980   (1 / 4 ln2) * mean ln(H/L)^2                        range-based, ~5x efficient
  Garman-Klass 1980  mean[ 0.5 ln(H/L)^2 - (2 ln2 - 1) ln(C/O)^2 ]      adds open/close
  Rogers-Satchell 1991  mean[ ln(H/C) ln(H/O) + ln(L/C) ln(L/O) ]       drift-robust
  Yang-Zhang 2000  var(overnight) + k var(open->close) + (1-k) RS,
                   k = 0.34 / (1.34 + (n+1)/(n-1))                     gap-aware, min variance
  EWMA (RiskMetrics) sigma^2_t = lam sigma^2_{t-1} + (1-lam) r^2_{t-1}, lam = 0.94
  HAR-RV (Corsi 2009) direct h-day regression of future realized variance on
                   daily / weekly / monthly realized variance; OLS via lstsq.

Range estimators ignore the overnight gap; Yang-Zhang is the one to quote for
single stocks. Earnings reaction days can be masked out so the realized side
matches the ex-earnings implied side of the VRP.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TD = 252
LN2 = math.log(2.0)


def _ohlc(df: pd.DataFrame) -> tuple[pd.Series, ...]:
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    return o.where(ok), h.where(ok), l.where(ok), c.where(ok)


def _masked(x: pd.Series, mask: pd.Series | None) -> pd.Series:
    return x if mask is None else x.where(~mask.reindex(x.index, fill_value=False))


def _roll_mean(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=max(3, int(n * 0.6))).mean()


def close_to_close(df, n: int, mask=None) -> pd.Series:
    r = np.log(df["close"].astype(float)).diff()
    return np.sqrt(_roll_mean(_masked(r * r, mask), n) * TD)


def parkinson(df, n: int, mask=None) -> pd.Series:
    _, h, l, _ = _ohlc(df)
    x = np.log(h / l) ** 2 / (4.0 * LN2)
    return np.sqrt(_roll_mean(_masked(x, mask), n) * TD)


def garman_klass(df, n: int, mask=None) -> pd.Series:
    o, h, l, c = _ohlc(df)
    x = 0.5 * np.log(h / l) ** 2 - (2.0 * LN2 - 1.0) * np.log(c / o) ** 2
    return np.sqrt(_roll_mean(_masked(x, mask), n).clip(lower=0) * TD)


def rs_daily(df) -> pd.Series:
    o, h, l, c = _ohlc(df)
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def rogers_satchell(df, n: int, mask=None) -> pd.Series:
    return np.sqrt(_roll_mean(_masked(rs_daily(df), mask), n).clip(lower=0) * TD)


def yang_zhang(df, n: int, mask=None) -> pd.Series:
    o, h, l, c = _ohlc(df)
    on = _masked(np.log(o / c.shift(1)), mask)
    oc = _masked(np.log(c / o), mask)
    rs = _masked(rs_daily(df), mask)
    mp = max(3, int(n * 0.6))
    var_o = on.rolling(n, min_periods=mp).var(ddof=1)
    var_c = oc.rolling(n, min_periods=mp).var(ddof=1)
    var_rs = rs.rolling(n, min_periods=mp).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((var_o + k * var_c + (1 - k) * var_rs).clip(lower=0) * TD)


def ewma(df, lam: float = 0.94, mask=None) -> pd.Series:
    r = _masked(np.log(df["close"].astype(float)).diff(), mask).fillna(0.0)
    v = (r * r).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v * TD)


def daily_variance(df) -> pd.Series:
    """Per-day variance proxy for HAR: overnight gap^2 + Rogers-Satchell intraday."""
    o, _, _, c = _ohlc(df)
    gap = np.log(o / c.shift(1)) ** 2
    return (gap.fillna(0) + rs_daily(df).clip(lower=0)).replace([np.inf, -np.inf], np.nan)


def har_forecast(df, h: int = 21, mask=None, fit_n: int = 750) -> float:
    """Annualised vol forecast over the next h trading days (HAR-RV, direct
    regression); NaN when there is too little history."""
    v = _masked(daily_variance(df), mask).dropna()
    if len(v) < 120:
        return float("nan")
    d = v
    w = v.rolling(5).mean()
    m = v.rolling(22).mean()
    fut = v[::-1].rolling(h).mean()[::-1].shift(-1)          # mean of the next h days
    X = pd.concat([d, w, m], axis=1).dropna()
    y = fut.reindex(X.index)
    fit = pd.concat([X, y], axis=1).dropna().tail(fit_n)
    if len(fit) < 60:
        return float("nan")
    A = np.column_stack([np.ones(len(fit)), fit.iloc[:, :3].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, fit.iloc[:, 3].to_numpy(), rcond=None)
    x_now = np.r_[1.0, X.iloc[-1].to_numpy()]
    f = float(x_now @ beta)
    floor = float(m.iloc[-1]) * 0.25 if np.isfinite(m.iloc[-1]) else 0.0
    return math.sqrt(max(f, floor) * TD)


def last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def estimator_set(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """Latest value of every estimator the screener shows."""
    if df is None or len(df) < 30:
        return {}
    out = {
        "cc10": last(close_to_close(df, 10, mask)), "cc20": last(close_to_close(df, 20, mask)),
        "cc60": last(close_to_close(df, 60, mask)), "cc252": last(close_to_close(df, 252, None)),
        "pk20": last(parkinson(df, 20, mask)), "gk20": last(garman_klass(df, 20, mask)),
        "rs20": last(rogers_satchell(df, 20, mask)),
        "yz10": last(yang_zhang(df, 10, mask)), "yz20": last(yang_zhang(df, 20, mask)),
        "yz60": last(yang_zhang(df, 60, mask)), "ewma": last(ewma(df, 0.94, mask)),
        "har30": har_forecast(df, 21, mask),
    }
    return out

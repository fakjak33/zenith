"""FMOM signal math — pure functions, no I/O, no network.

Implements Gupta & Kelly (2019) Equation 1 for the 1-1 case and the paper's
portfolio construction:

  s_it   = clip( r_it / sigma_it , -2, +2 )          (TSFM signal)
  sigma  = trailing 36-month std of monthly factor returns, annualized
  CSFM   = same, but r_it is de-meaned across factors at t first
  legs   = long: w_i = s_i / sum(s>0);  short: w_i = s_i / sum(|s<=0|)
  model  = sum(w_i * r_{i,t+1})                       ($1 long / $1 short)

All functions take a "monthly panel": DataFrame indexed by month-end
Timestamp, one column per factor, values = monthly returns (decimals).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

VOL_WINDOW = 36          # months of trailing vol (paper: previous three years)
VOL_MIN_PERIODS = 24     # accept a signal once 2y of history exists
CLIP = 2.0               # z-score cap (paper: +/- 2)


# ------------------------------------------------------------------ signals --
def ann_vol(monthly: pd.DataFrame) -> pd.DataFrame:
    """Trailing annualized vol of the PREVIOUS 36 months, excluding the
    formation month itself (the paper scales the formation return by vol 'over
    the previous three years'; excluding month t keeps a huge formation move
    from deflating its own z-score)."""
    roll = monthly.rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).std()
    return roll.shift(1) * math.sqrt(12)


def tsfm_signals(monthly: pd.DataFrame) -> pd.DataFrame:
    """1-1 TSFM signal formed at the END of each month (trade the next month)."""
    vol = ann_vol(monthly).replace(0.0, np.nan)
    return (monthly / vol).clip(-CLIP, CLIP)


def csfm_signals(monthly: pd.DataFrame) -> pd.DataFrame:
    """1-1 CSFM: de-mean the formation return across factors, then scale by each
    factor's OWN trailing vol (the paper: 'otherwise follows the same
    construction')."""
    vol = ann_vol(monthly).replace(0.0, np.nan)
    demeaned = monthly.sub(monthly.mean(axis=1), axis=0)
    return (demeaned / vol).clip(-CLIP, CLIP)


# --------------------------------------------------------------- portfolio ---
def leg_weights(s: pd.Series) -> pd.Series:
    """$1-long / $1-short signal weighting. Positive signals get weights summing
    to +1, non-positive signals weights summing to -1 (zeros stay zero). An
    all-one-sign month leaves the empty leg at 0 — callers flag it."""
    s = s.dropna().astype(float)
    w = pd.Series(0.0, index=s.index)
    pos, neg = s[s > 0], s[s <= 0]
    if len(pos) and pos.sum() > 0:
        w[pos.index] = pos / pos.sum()
    if len(neg) and neg.abs().sum() > 0:
        w[neg.index] = neg / neg.abs().sum()
    return w


def model_return(weights: pd.Series, realized_next: pd.Series) -> float:
    """Realized model return: sum of weight x next-month factor return over the
    factors that have a realized value."""
    common = weights.index.intersection(realized_next.dropna().index)
    if not len(common):
        return float("nan")
    return float((weights[common] * realized_next[common]).sum())


def decile_flags(s: pd.Series) -> pd.Series:
    """'top' / 'bottom' / '' — top and bottom decile of factors by signal value
    (at least one name per side once there are >= 2 factors)."""
    s = s.dropna()
    flags = pd.Series("", index=s.index)
    n = len(s)
    if n < 2:
        return flags
    k = max(1, int(math.ceil(n / 10)))
    order = s.sort_values(ascending=False)
    flags[order.index[:k]] = "top"
    flags[order.index[-k:]] = "bottom"
    return flags


# ----------------------------------------------------------------- backtest --
def perf_stats(returns: pd.Series) -> dict:
    """Annualized stats for a monthly return series."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 12:
        return {"n_months": int(len(r))}
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    sd = float(r.std())
    return {
        "n_months": int(len(r)),
        "start": r.index[0].strftime("%Y-%m"),
        "end": r.index[-1].strftime("%Y-%m"),
        "ann_ret": round(float(r.mean()) * 12, 4),
        "ann_vol": round(sd * math.sqrt(12), 4),
        "sharpe": round(float(r.mean()) / sd * math.sqrt(12), 3) if sd > 0 else None,
        "max_dd": round(dd, 4),
        "hit_rate": round(float((r > 0).mean()), 4),
    }


def backtest(monthly: pd.DataFrame) -> dict:
    """Walk the panel month by month: signals at t, realized at t+1 — for both
    lenses plus an equal-weight-all-factors benchmark. No look-ahead: the
    weights applied to month t+1 only use data through t."""
    monthly = monthly.sort_index()
    sig = {"tsfm": tsfm_signals(monthly), "csfm": csfm_signals(monthly)}
    rows: dict[pd.Timestamp, dict] = {}
    idx = monthly.index
    for i in range(len(idx) - 1):
        t, t1 = idx[i], idx[i + 1]
        realized = monthly.loc[t1]
        rec: dict[str, float] = {}
        for lens, s_all in sig.items():
            s = s_all.loc[t].dropna()
            if len(s) < 3:
                continue
            rec[lens] = model_return(leg_weights(s), realized)
        avail = monthly.loc[t].dropna().index      # factors live at formation
        if len(avail):
            rec["ew"] = float(realized[avail].dropna().mean())
        if rec:
            rows[t1] = rec
    series = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return {
        "series": series,
        "stats": {c: perf_stats(series[c]) for c in series.columns},
    }


# ------------------------------------------- replication factor construction --
def winsorize(x: pd.Series, p: float = 0.01) -> pd.Series:
    """Cross-sectional winsorization at the p / 1-p quantiles (paper: 1%)."""
    x = x.astype(float)
    lo, hi = x.quantile(p), x.quantile(1.0 - p)
    return x.clip(lo, hi)


def build_factor_bins(chars: pd.Series, mktcap: pd.Series) -> dict:
    """Paper's six-bin construction on one characteristic snapshot.

    chars/mktcap: indexed by ticker. Winsorize the characteristic at 1%, split
    the universe at the median market cap (Large/Small), split each size bin at
    the 30th/70th characteristic percentiles (Low/Med/High, 30-40-30), then
    value-weight within each bin. Factor = 0.5*(LH+SH) - 0.5*(LL+SL).

    Returns {"bins": {"LH": [{"ticker","w","char"}...], ...}, "n": int,
             "size_split": float, "breakpoints": {"L": [lo,hi], "S": [lo,hi]}}.
    """
    df = pd.DataFrame({"char": chars, "cap": mktcap}).dropna()
    df = df[df["cap"] > 0]
    if len(df) < 20:
        return {"bins": {}, "n": int(len(df)), "size_split": None, "breakpoints": {}}
    df["char"] = winsorize(df["char"])
    size_split = float(df["cap"].median())
    out: dict[str, list[dict]] = {}
    bps: dict[str, list[float]] = {}
    for size_code, part in (("L", df[df["cap"] >= size_split]),
                            ("S", df[df["cap"] < size_split])):
        lo, hi = part["char"].quantile(0.30), part["char"].quantile(0.70)
        bps[size_code] = [round(float(lo), 6), round(float(hi), 6)]
        for char_code, sub in (("L", part[part["char"] <= lo]),
                               ("M", part[(part["char"] > lo) & (part["char"] < hi)]),
                               ("H", part[part["char"] >= hi])):
            total = float(sub["cap"].sum())
            out[size_code + char_code] = [
                {"ticker": t, "w": round(float(r["cap"]) / total, 6),
                 "char": round(float(r["char"]), 6)}
                for t, r in sub.sort_values("cap", ascending=False).iterrows()
            ] if total > 0 else []
    # rename to paper-style keys: first letter size, second characteristic level
    bins = {"LL": out["LL"], "LM": out["LM"], "LH": out["LH"],
            "SL": out["SL"], "SM": out["SM"], "SH": out["SH"]}
    return {"bins": bins, "n": int(len(df)), "size_split": size_split,
            "breakpoints": bps}


# ------------------------------------------------------------------ helpers --
def monthly_returns(close_daily: pd.Series) -> pd.Series:
    """Daily close series -> month-end monthly returns (drops the seed NaN)."""
    m = close_daily.dropna().resample("ME").last()
    return m.pct_change().dropna()

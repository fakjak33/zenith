"""Modeled implied volatility for THETA RUN (pure numpy/pandas, no scipy).

All rolling statistics are trailing, so a feature at date t never sees prices
after t. Earnings report DATES may lie in the future (they are scheduled and
public weeks ahead); earnings MOVES only come from reports already completed.
See the package docstring for the model and its citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252
HAR_W = (0.2, 0.4, 0.4)              # weights on 5 / 21 / 63-day realized vol
RV_FLOOR = 0.04
MULT_CLIP = (0.75, 2.5)              # VIX / SPY forecast ratio bounds
STOCK_VRP_SHARE = 0.6                # single stocks keep 60% of the index premium
EM_LOOKBACK = 8                      # reports averaged for the expected earnings move
EM_PREMIUM = 1.10                    # implied earnings moves price slightly above realized
EARN_VISIBLE_DAYS = 45               # a report shows up this many days ahead
SKEW_BASE = {"index": 1.6, "etf": 1.2, "stock": 0.7}
IV30_T = 30 / 365.0


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def reaction_mask(index: pd.DatetimeIndex, report_dates) -> pd.Series:
    """True on the two trading days that absorb each report (pre- or post-market)."""
    mask = np.zeros(len(index), dtype=bool)
    for d in report_dates:
        i = int(index.searchsorted(pd.Timestamp(d), side="left"))
        for j in (i, i + 1):
            if 0 <= j < len(index):
                mask[j] = True
    return pd.Series(mask, index=index)


def realized(r: pd.Series, n: int, mask: pd.Series | None = None) -> pd.Series:
    """Annualized trailing realized vol over n days, skipping masked days."""
    r2 = r.pow(2)
    if mask is not None:
        r2 = r2.where(~mask)
    return np.sqrt(r2.rolling(n, min_periods=max(3, n // 2)).mean() * TD)


def har_forecast(r: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    w5, w21, w63 = HAR_W
    fc = (w5 * realized(r, 5, mask) + w21 * realized(r, 21, mask)
          + w63 * realized(r, 63, mask))
    return fc.ffill().clip(lower=RV_FLOOR)


def earnings_moves(close: pd.Series, report_dates) -> pd.Series:
    """|two-day reaction| per completed report, stamped on the day it is known."""
    idx = close.index
    out = pd.Series(np.nan, index=idx)
    c = close.to_numpy(float)
    for d in report_dates:
        i = int(idx.searchsorted(pd.Timestamp(d), side="left"))
        before, after = i - 1, i + 1
        if before < 0 or after >= len(idx):
            continue
        if c[before] > 0 and np.isfinite(c[after]):
            out.iloc[after] = abs(c[after] / c[before] - 1.0)
    return out


def expected_move(close: pd.Series, report_dates, rv: pd.Series) -> pd.Series:
    """Expected one-event move: mean of the last EM_LOOKBACK completed reactions
    (x EM_PREMIUM); before any history, ~2.5 daily standard deviations."""
    moves = earnings_moves(close, report_dates)
    hist = moves.dropna()
    em = pd.Series(np.nan, index=close.index)
    if len(hist):
        roll = hist.rolling(EM_LOOKBACK, min_periods=1).mean() * EM_PREMIUM
        em.loc[roll.index] = roll.to_numpy()
        em = em.ffill()
    fallback = 2.5 * rv / np.sqrt(TD)
    return em.fillna(fallback)


def days_to_next_report(index: pd.DatetimeIndex, report_dates) -> pd.Series:
    """Calendar days until the next report strictly after each date (-1 = none
    scheduled within EARN_VISIBLE_DAYS)."""
    ds = np.array(sorted(pd.Timestamp(d).value for d in report_dates), dtype="int64")
    out = np.full(len(index), -1, dtype=np.int32)
    if len(ds):
        vals = index.asi8
        pos = np.searchsorted(ds, vals, side="right")
        ok = pos < len(ds)
        nd = np.where(ok, ds[np.minimum(pos, len(ds) - 1)], 0)
        days = (nd - vals) // (86400 * 10**9)
        vis = ok & (days <= EARN_VISIBLE_DAYS)
        out[vis] = days[vis]
    return pd.Series(out, index=index)


def vrp_multiplier(vix: pd.Series, spy_fc: pd.Series) -> pd.Series:
    """Daily index volatility-risk-premium multiplier, smoothed (trailing EWM)."""
    m = (vix / spy_fc).clip(*MULT_CLIP)
    return m.ewm(span=5, adjust=False).mean()


def rolling_rank(x: pd.Series, n: int = TD) -> tuple[pd.Series, pd.Series]:
    """IV rank (position in the trailing min-max range) and IV percentile."""
    lo = x.rolling(n, min_periods=20).min()
    hi = x.rolling(n, min_periods=20).max()
    rank = ((x - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)
    pct = x.rolling(n, min_periods=20).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
    return rank, pct


def skew_strength(kind: str, vix: pd.Series) -> pd.Series:
    return SKEW_BASE[kind] * (1.0 + (vix - 0.16) / 0.30).clip(0.7, 2.0)


def features(px: pd.DataFrame, report_dates, mult: pd.Series, vix: pd.Series,
             spy_ret: pd.Series, kind: str) -> pd.DataFrame:
    """Per-day feature table for one ticker.

    px: OHLCV indexed by date. mult/vix/spy_ret on the master calendar (reindexed
    here). kind: 'index' (IV = VIX), 'etf' or 'stock'.
    """
    close = px["close"].astype(float)
    idx = close.index
    r = log_returns(close)
    mask = reaction_mask(idx, report_dates) if kind == "stock" else None
    fc = har_forecast(r, mask)
    rv21 = realized(r, 21)
    m = mult.reindex(idx).ffill()
    v = vix.reindex(idx).ffill()
    if kind == "index":
        iv_base = v.copy()
    elif kind == "etf":
        iv_base = fc * m
    else:
        iv_base = fc * (1.0 + STOCK_VRP_SHARE * (m - 1.0))
    iv_base = iv_base.clip(lower=0.05)

    if kind == "stock" and len(report_dates):
        em = expected_move(close, report_dates, rv21.fillna(fc))
        nxt = days_to_next_report(idx, report_dates)
    else:
        em = pd.Series(0.0, index=idx)
        nxt = pd.Series(-1, index=idx, dtype=np.int32)
    in30 = (nxt >= 0) & (nxt <= 30)
    iv30 = np.sqrt(iv_base.pow(2) + np.where(in30, em.pow(2) / IV30_T, 0.0))
    ivr, ivp = rolling_rank(iv30)

    sr = spy_ret.reindex(idx)
    cov = r.rolling(TD, min_periods=60).cov(sr)
    var = sr.rolling(TD, min_periods=60).var()
    beta = (cov / var).clip(-1, 4).fillna(1.0)

    return pd.DataFrame({
        "open": px["open"].astype(float), "high": px["high"].astype(float),
        "low": px["low"].astype(float), "close": close,
        "volume": px["volume"].astype(float),
        "rv21": rv21, "rv_fc": fc, "iv_base": iv_base, "em": em, "next_earn": nxt,
        "iv30": iv30, "iv_rank": ivr, "iv_pct": ivp,
        "skew": skew_strength(kind, v), "beta": beta,
    }, index=idx)


def atm_vol(iv_base: float, em: float, T: float, earn_before_expiry: bool) -> float:
    """ATM vol for an expiry T years out: base vol plus the earnings variance
    em^2 spread over T when a report falls before expiry."""
    T = max(T, 1.0 / 365.0)
    return float(np.sqrt(iv_base ** 2 + (em ** 2 / T if earn_before_expiry else 0.0)))


def smile(atm: float, S: float, K, T: float, skew: float):
    """Strike-level IV from the ATM vol: puts (K < S) richer, a mild smile."""
    T = max(T, 1.0 / 365.0)
    m = np.log(np.asarray(K, dtype=float) / S) / (atm * np.sqrt(T))
    return atm * np.clip(1.0 - skew * 0.10 * m + 0.03 * m * m, 0.6, 2.5)

"""Strategies segment — a battery of buy/neutral/sell signals per asset.

Combines two traditions:
  * Classic systematic strategies (momentum, moving-average single & crossover,
    multi-MA, mean-reversion, Donchian/channel breakout, RSI, realized-vol
    regime, multi-asset trend) computed per series.
  * 101-formulaic-alpha-style cross-sectional expressions (Kakushadze 2015):
    rank/ts_rank/delta/decay_linear/correlation operators over OHLCV, ranked
    across the universe.

Every model returns continuous signals in [-1, 1] via schema.make_signal so the
overlap/consensus layers can compare them. Parameters come from the registry so
the user can tune them over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal
from ..universe import asset_class_of

SRC = "yfinance"


# ---- per-series classic families -----------------------------------------
def _momentum(close: pd.Series, lookback: int = 126) -> float:
    if len(close) < lookback + 5:
        return 0.0
    r = close.iloc[-1] / close.iloc[-lookback] - 1.0
    return ind.clip1(np.tanh(3 * r))


def _ma_single(close: pd.Series, d: int = 200) -> float:
    m = ind.sma(close, d)
    if pd.isna(m.iloc[-1]):
        return 0.0
    return ind.clip1((close.iloc[-1] / m.iloc[-1] - 1.0) * 8)


def _ma_cross(close: pd.Series, fast: int = 50, slow: int = 200) -> float:
    f, s = ind.sma(close, fast), ind.sma(close, slow)
    if pd.isna(f.iloc[-1]) or pd.isna(s.iloc[-1]):
        return 0.0
    return ind.clip1((f.iloc[-1] / s.iloc[-1] - 1.0) * 12)


def _multi_ma(close: pd.Series, windows=(20, 50, 100, 200)) -> float:
    px = close.iloc[-1]
    above = [1 if px > ind.sma(close, w).iloc[-1] else -1
             for w in windows if not pd.isna(ind.sma(close, w).iloc[-1])]
    return ind.clip1(np.mean(above)) if above else 0.0


def _mean_reversion(close: pd.Series, d: int = 20) -> float:
    z = ind.zscore(close, d)
    return ind.clip1(-z.iloc[-1] / 2.0) if not pd.isna(z.iloc[-1]) else 0.0


def _donchian(df: pd.DataFrame, d: int = 20) -> float:
    hi, lo = ind.donchian(df["high"], df["low"], d)
    px = df["close"].iloc[-1]
    if pd.isna(hi.iloc[-1]) or pd.isna(lo.iloc[-1]) or hi.iloc[-1] == lo.iloc[-1]:
        return 0.0
    pos = (px - lo.iloc[-1]) / (hi.iloc[-1] - lo.iloc[-1])     # 0..1 in channel
    return ind.clip1((pos - 0.5) * 2)


def _rsi_mr(close: pd.Series, d: int = 14) -> float:
    r = ind.rsi(close, d).iloc[-1]
    if pd.isna(r):
        return 0.0
    return ind.clip1((50 - r) / 30.0)         # >70 sell, <30 buy (contrarian)


def _vol_regime(close: pd.Series) -> float:
    rv = ind.realized_vol(close, 20)
    base = ind.realized_vol(close, 100)
    if pd.isna(rv.iloc[-1]) or pd.isna(base.iloc[-1]) or base.iloc[-1] == 0:
        return 0.0
    # rising short-vol vs long-vol = risk-off tilt (negative)
    return ind.clip1(-(rv.iloc[-1] / base.iloc[-1] - 1.0))


def _trend_following(close: pd.Series) -> float:
    """Multi-asset-style trend: blend of 1m/3m/6m/12m momentum signs."""
    looks = (21, 63, 126, 252)
    vals = []
    for lb in looks:
        if len(close) > lb:
            vals.append(np.sign(close.iloc[-1] / close.iloc[-lb] - 1.0))
    return ind.clip1(np.mean(vals)) if vals else 0.0


_PER_SERIES = {
    "momentum": lambda df: _momentum(df["close"]),
    "ma_single_200": lambda df: _ma_single(df["close"], 200),
    "ma_cross_50_200": lambda df: _ma_cross(df["close"], 50, 200),
    "multi_ma": lambda df: _multi_ma(df["close"]),
    "mean_reversion": lambda df: _mean_reversion(df["close"]),
    "donchian_breakout": lambda df: _donchian(df, 20),
    "rsi_reversion": lambda df: _rsi_mr(df["close"]),
    "vol_regime": lambda df: _vol_regime(df["close"]),
    "trend_following": lambda df: _trend_following(df["close"]),
}

_HORIZON = {
    "momentum": "months", "ma_single_200": "months", "ma_cross_50_200": "months",
    "multi_ma": "weeks", "mean_reversion": "days", "donchian_breakout": "weeks",
    "rsi_reversion": "days", "vol_regime": "weeks", "trend_following": "months",
}


# ---- cross-sectional 101-alpha-style expressions --------------------------
def _alpha_cs(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """A couple of representative cross-sectional formulaic alphas, ranked across
    the universe. Returns ticker -> signal in [-1, 1]."""
    closes = pd.DataFrame({t: d["close"] for t, d in data.items()}).dropna(how="all")
    vols = pd.DataFrame({t: d["volume"] for t, d in data.items()}).reindex(closes.index)
    rets = closes.pct_change()

    # Alpha-A: short-term reversal — rank(-returns over 5d)
    a = ind.rank_cs(-(closes.pct_change(5))) * 2 - 1
    # Alpha-B: volume-confirmed momentum — rank(20d return) * rank(volume trend)
    vol_tr = vols / vols.rolling(20).mean()
    b = ((ind.rank_cs(closes.pct_change(20)) * 2 - 1)
         * (ind.rank_cs(vol_tr).clip(lower=0.2)))
    # Alpha-C: decayed correlation of close & volume (crowding proxy)
    c = ind.rank_cs(pd.DataFrame(
        {t: -ind.correlation(closes[t], vols[t], 10) for t in closes.columns})) * 2 - 1

    blend = (a.add(b, fill_value=0).add(c, fill_value=0)) / 3.0
    out = {}
    if len(blend):
        row = blend.iloc[-1]
        for t in closes.columns:
            v = row.get(t, np.nan)
            out[t] = ind.clip1(v) if not pd.isna(v) else 0.0
    return out


def compute(data: dict[str, pd.DataFrame], params: dict | None = None) -> list[dict]:
    """Run the full strategies battery over the price universe."""
    params = params or {}
    out: list[dict] = []
    alpha = _alpha_cs(data) if len(data) >= 3 else {}

    for t, df in data.items():
        ac = asset_class_of(t)
        for family, fn in _PER_SERIES.items():
            try:
                sig = fn(df)
            except Exception:
                sig = 0.0
            out.append(make_signal(
                t, "strategies", family, sig, asset_class=ac,
                horizon=_HORIZON[family], source=SRC,
                rationale=f"{family} on daily OHLCV"))
        if t in alpha:
            out.append(make_signal(
                t, "strategies", "formulaic_alpha_combo", alpha[t], asset_class=ac,
                horizon="days", source=SRC,
                rationale="cross-sectional 101-alpha-style blend (reversal + "
                          "vol-confirmed momentum + crowding)"))
    return out

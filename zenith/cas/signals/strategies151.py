"""'151 strategies' CAS model — the signal-generating subset, scanned across the
master ETF list.

Inspired by the Kakushadze & Serur "151 Trading Strategies" catalog, which spans
many asset classes. Most of the catalog entries are *payoff structures* (covered
calls, spreads, straddles, ladders, structured/convertible/tax strategies) that
are NOT directional signals and cannot be evaluated from free OHLCV — those are
out of scope here, treated as reference only.

This module implements the catalog's directional, technical/statistical strategy
*types* as our own indicator code (momentum/trend, mean-reversion, breakout,
oscillator, volatility, channel), producing buy/neutral/sell signals per ETF.
Families are ``s151_*`` prefixed so the CAS UI shows them as a distinct model.
All maths reuse zenith.cas.signals.indicators; nothing is copied from the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal

SRC = "yfinance"


# --- trend / momentum -----------------------------------------------------
def _sma_cross(c: pd.Series, fast: int, slow: int) -> float:
    f, s = ind.sma(c, fast), ind.sma(c, slow)
    if pd.isna(f.iloc[-1]) or pd.isna(s.iloc[-1]) or s.iloc[-1] == 0:
        return 0.0
    return ind.clip1((f.iloc[-1] / s.iloc[-1] - 1.0) * 12)


def _ema_cross(c: pd.Series, fast: int, slow: int) -> float:
    f, s = ind.ema(c, fast), ind.ema(c, slow)
    if pd.isna(f.iloc[-1]) or pd.isna(s.iloc[-1]) or s.iloc[-1] == 0:
        return 0.0
    return ind.clip1((f.iloc[-1] / s.iloc[-1] - 1.0) * 12)


def _macd(c: pd.Series) -> float:
    macd = ind.ema(c, 12) - ind.ema(c, 26)
    sig = macd.ewm(span=9, adjust=False).mean()
    d = (macd - sig).iloc[-1]
    scale = c.iloc[-1] * 0.01 + 1e-9
    return ind.clip1(d / scale)


def _roc(c: pd.Series, d: int) -> float:
    if len(c) < d + 2:
        return 0.0
    return ind.clip1(np.tanh(3 * (c.iloc[-1] / c.iloc[-d] - 1.0)))


def _tsmom(c: pd.Series) -> float:
    looks = (21, 63, 126, 252)
    vals = [np.sign(c.iloc[-1] / c.iloc[-lb] - 1.0) for lb in looks if len(c) > lb]
    return ind.clip1(np.mean(vals)) if vals else 0.0


def _price_vs_sma(c: pd.Series, d: int) -> float:
    m = ind.sma(c, d)
    if pd.isna(m.iloc[-1]) or m.iloc[-1] == 0:
        return 0.0
    return ind.clip1((c.iloc[-1] / m.iloc[-1] - 1.0) * 8)


# --- mean reversion / oscillators -----------------------------------------
def _zscore_rev(c: pd.Series, d: int) -> float:
    z = ind.zscore(c, d).iloc[-1]
    return ind.clip1(-z / 2.0) if not pd.isna(z) else 0.0


def _rsi_rev(c: pd.Series, d: int = 14) -> float:
    r = ind.rsi(c, d).iloc[-1]
    return ind.clip1((50 - r) / 30.0) if not pd.isna(r) else 0.0


def _bollinger_pctb(c: pd.Series, d: int = 20, k: float = 2.0) -> float:
    ma, sd = ind.sma(c, d), ind.stddev(c, d)
    if pd.isna(ma.iloc[-1]) or pd.isna(sd.iloc[-1]) or sd.iloc[-1] == 0:
        return 0.0
    upper, lower = ma + k * sd, ma - k * sd
    pctb = (c.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)
    return ind.clip1((0.5 - pctb) * 2)          # near lower band -> buy


def _stochastic(df: pd.DataFrame, d: int = 14) -> float:
    hi, lo = df["high"].rolling(d).max(), df["low"].rolling(d).min()
    if pd.isna(hi.iloc[-1]) or hi.iloc[-1] == lo.iloc[-1]:
        return 0.0
    kpct = (df["close"].iloc[-1] - lo.iloc[-1]) / (hi.iloc[-1] - lo.iloc[-1]) * 100
    return ind.clip1((50 - kpct) / 35.0)        # oscillator reversion


# --- breakout / channel ---------------------------------------------------
def _donchian(df: pd.DataFrame, d: int = 20) -> float:
    hi, lo = ind.donchian(df["high"], df["low"], d)
    px = df["close"].iloc[-1]
    if pd.isna(hi.iloc[-1]) or hi.iloc[-1] == lo.iloc[-1]:
        return 0.0
    pos = (px - lo.iloc[-1]) / (hi.iloc[-1] - lo.iloc[-1])
    return ind.clip1((pos - 0.5) * 2)


def _keltner(df: pd.DataFrame, d: int = 20) -> float:
    mid = ind.ema(df["close"], d)
    rng = ind.atr(df, d) * 2
    if pd.isna(mid.iloc[-1]) or pd.isna(rng.iloc[-1]) or rng.iloc[-1] == 0:
        return 0.0
    return ind.clip1((df["close"].iloc[-1] - mid.iloc[-1]) / rng.iloc[-1])


# --- volatility -----------------------------------------------------------
def _vol_regime(c: pd.Series) -> float:
    rv, base = ind.realized_vol(c, 20), ind.realized_vol(c, 100)
    if pd.isna(rv.iloc[-1]) or pd.isna(base.iloc[-1]) or base.iloc[-1] == 0:
        return 0.0
    return ind.clip1(-(rv.iloc[-1] / base.iloc[-1] - 1.0))


# family -> (callable, horizon). Callables take the OHLCV df.
_FAMILIES = {
    "s151_sma_cross_20_50": (lambda d: _sma_cross(d["close"], 20, 50), "weeks"),
    "s151_sma_cross_50_200": (lambda d: _sma_cross(d["close"], 50, 200), "months"),
    "s151_ema_cross_12_26": (lambda d: _ema_cross(d["close"], 12, 26), "weeks"),
    "s151_macd": (lambda d: _macd(d["close"]), "weeks"),
    "s151_roc_63": (lambda d: _roc(d["close"], 63), "months"),
    "s151_tsmom": (lambda d: _tsmom(d["close"]), "months"),
    "s151_price_vs_sma200": (lambda d: _price_vs_sma(d["close"], 200), "months"),
    "s151_zscore_reversion": (lambda d: _zscore_rev(d["close"], 20), "days"),
    "s151_rsi_reversion": (lambda d: _rsi_rev(d["close"]), "days"),
    "s151_bollinger_pctb": (lambda d: _bollinger_pctb(d["close"]), "days"),
    "s151_stochastic": (lambda d: _stochastic(d), "days"),
    "s151_donchian_breakout": (lambda d: _donchian(d), "weeks"),
    "s151_keltner": (lambda d: _keltner(d), "weeks"),
    "s151_vol_regime": (lambda d: _vol_regime(d["close"]), "weeks"),
}


def compute(data: dict[str, pd.DataFrame], category_of=None) -> list[dict]:
    """Run the 151-model technical battery over the price universe (master list)."""
    out: list[dict] = []
    for t, df in data.items():
        ac = category_of(t) if category_of else "equity"
        for family, (fn, horizon) in _FAMILIES.items():
            try:
                sig = fn(df)
            except Exception:
                sig = 0.0
            out.append(make_signal(
                t, "strategies", family, sig, asset_class=ac, horizon=horizon,
                source=SRC, confidence="low",
                rationale=f"{family.replace('s151_', '')} on daily OHLCV (151-model)"))
    return out

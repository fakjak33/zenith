"""Momentum / oscillator family: bounded readings of recent strength."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import ema, ind, true_range, wilder

OSC_EVID = ("Short-horizon oscillators mostly capture short-term reversal (Jegadeesh 1990; Lehmann 1990), which "
            "is strongest in volatile, illiquid markets (Nagel 2012). Single-rule evidence is weak once costs and "
            "data-snooping are accounted for (Sullivan, Timmermann & White 1999; Park & Irwin 2007).")


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.where(dn > 0))


@ind("rsi", "RSI", "momentum", {"length": (14, 2, 100, 1, "Bars in the average gain / loss. 14 is Wilder's.")},
     signal="band", lo=30, hi=70,
     hover="Relative Strength Index, 0-100: how much of the recent movement was up-moves. Below 30 = oversold.",
     how="100 − 100/(1 + average gain ÷ average loss), using Wilder's smoothing.",
     use="Mean reversion: buy under 30, sell over 70. In strong trends it can stay overbought for months.",
     evidence="Wilder (1978). " + OSC_EVID)
def _rsi_ind(df, length):
    return _rsi(df["close"], int(length))


@ind("rsi2", "RSI(2) — Connors", "momentum", {"length": (2, 2, 10, 1, "Very short RSI length.")},
     signal="band", lo=10, hi=90,
     hover="A 2-bar RSI: an extreme short-term pullback gauge. Under 10 = a sharp dip.",
     how="RSI with a 2-bar window, so a couple of down days pushes it near 0.",
     use="Connors: buy dips (RSI2 < 10) only while price is above its 200-day average; exit on a bounce.",
     evidence="Connors & Alvarez, Short Term Trading Strategies That Work (2009). " + OSC_EVID)
def _rsi2(df, length):
    return _rsi(df["close"], int(length))


@ind("stoch", "Stochastic %K", "momentum",
     {"length": (14, 3, 100, 1, "Lookback for the high-low range."), "smooth": (3, 1, 20, 1, "Smoothing of %K.")},
     signal="band", lo=20, hi=80,
     hover="Where the close sits inside the recent high-low range, 0-100. Near 0 = closing at the lows.",
     how="100 × (close − lowest low) ÷ (highest high − lowest low), averaged over a few bars.",
     use="Below 20 oversold, above 80 overbought.", evidence="Lane (1950s). " + OSC_EVID)
def _stoch(df, length, smooth):
    n = int(length)
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).where(hi > lo)
    return k.rolling(int(smooth), min_periods=int(smooth)).mean()


@ind("stochrsi", "Stochastic RSI", "momentum",
     {"length": (14, 3, 100, 1, "RSI length and the window it is ranked over.")}, signal="band", lo=20, hi=80,
     hover="The stochastic of the RSI: where today's RSI sits within its own recent range, 0-100.",
     how="Stochastic formula applied to the RSI series instead of price.",
     use="A faster, twitchier oversold/overbought gauge.", evidence="Chande & Kroll (1994). " + OSC_EVID)
def _stochrsi(df, length):
    n = int(length)
    r = _rsi(df["close"], n)
    lo, hi = r.rolling(n, min_periods=n).min(), r.rolling(n, min_periods=n).max()
    return 100 * (r - lo) / (hi - lo).where(hi > lo)


@ind("willr", "Williams %R", "momentum", {"length": (14, 3, 100, 1, "Lookback for the high-low range.")},
     signal="band", lo=-80, hi=-20,
     hover="The stochastic flipped upside-down: 0 at the top of the range, −100 at the bottom.",
     how="−100 × (highest high − close) ÷ (highest high − lowest low).",
     use="Below −80 oversold, above −20 overbought.", evidence="Williams (1973). " + OSC_EVID)
def _willr(df, length):
    n = int(length)
    lo, hi = df["low"].rolling(n, min_periods=n).min(), df["high"].rolling(n, min_periods=n).max()
    return -100 * (hi - df["close"]) / (hi - lo).where(hi > lo)


@ind("cci", "Commodity Channel Index", "momentum", {"length": (20, 5, 200, 1, "Lookback.")},
     signal="band", lo=-100, hi=100,
     hover="How far the typical price is from its average, in units of its typical deviation. ±100 = stretched.",
     how="(typical price − SMA) ÷ (0.015 × mean absolute deviation).",
     use="Fade beyond ±100 in ranges, or follow it in breakouts.", evidence="Lambert (1980). " + OSC_EVID)
def _cci(df, length):
    n = int(length)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.where(md > 0))


@ind("roc", "Rate of change", "momentum", {"length": (21, 1, 500, 1, "Bars back.")}, signal="level",
     hover="Percent change in price over the last N bars. The simplest momentum measure.",
     how="close ÷ close N bars ago − 1.",
     use="Positive → momentum up. Short N tends to revert, 3-12 months tends to persist.",
     evidence="Jegadeesh & Titman (1993) momentum; Jegadeesh (1990) short-term reversal.")
def _roc(df, length):
    return df["close"] / df["close"].shift(int(length)) - 1


@ind("tsi", "True strength index", "momentum",
     {"long": (25, 5, 100, 1, "First smoothing."), "short": (13, 2, 50, 1, "Second smoothing.")},
     signal="level",
     hover="Double-smoothed momentum, −100 to +100. Smoother than RSI, keeps the direction.",
     how="100 × EMA(EMA(Δclose)) ÷ EMA(EMA(|Δclose|)).", use="Above 0 → bullish momentum.",
     evidence="Blau (1991).")
def _tsi(df, long, short):
    d = df["close"].diff()
    num = ema(ema(d, long), short)
    den = ema(ema(d.abs(), long), short)
    return 100 * num / den.where(den > 0)


@ind("ultosc", "Ultimate oscillator", "momentum",
     {"short": (7, 2, 30, 1, "Short window."), "mid": (14, 5, 60, 1, "Middle window."),
      "long": (28, 10, 120, 1, "Long window.")}, signal="band", lo=30, hi=70,
     hover="Buying pressure averaged over three horizons at once, 0-100, to cut false signals.",
     how="Weighted mix (4:2:1) of buying pressure ÷ true range over short, middle and long windows.",
     use="Under 30 oversold, over 70 overbought.", evidence="Williams (1985). " + OSC_EVID)
def _ultosc(df, short, mid, long):
    pc = df["close"].shift(1)
    bp = df["close"] - pd.concat([df["low"], pc], axis=1).min(axis=1)
    tr = true_range(df)
    avg = lambda n: bp.rolling(int(n)).sum() / tr.rolling(int(n)).sum()  # noqa: E731
    return 100 * (4 * avg(short) + 2 * avg(mid) + avg(long)) / 7


@ind("mfi", "Money flow index", "momentum", {"length": (14, 3, 100, 1, "Lookback.")}, signal="band",
     lo=20, hi=80, needs=("volume",),
     hover="A volume-weighted RSI: did money flow in (up days on big volume) or out?",
     how="100 − 100/(1 + positive money flow ÷ negative money flow), money flow = typical price × volume.",
     use="Under 20 oversold, over 80 overbought.", evidence="Quong & Soudack (1989). " + OSC_EVID)
def _mfi(df, length):
    n = int(length)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.where(neg > 0))

"""Price-structure family: the shape of individual bars and swings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import atr, ind


@ind("gap", "Opening gap (ATR units)", "structure", {}, signal="level",
     hover="How far today opened from yesterday's close, measured in average ranges. Big gaps often partly fill.",
     how="(open − previous close) ÷ ATR(14).",
     use="Fade large gaps (gap fill) or follow them (breakaway gap) — test both with 'side'.",
     evidence="Gap behaviour around news: Berkman, Koch, Tuttle & Zhang (2012).")
def _gap(df):
    return (df["open"] - df["close"].shift(1)) / atr(df, 14).shift(1)


@ind("body", "Candle body ÷ range", "structure", {}, signal="level",
     hover="How much of today's range was a real move: +1 = closed at the high from the low (strong up bar), "
           "−1 = the reverse, 0 = indecision (doji).",
     how="(close − open) ÷ (high − low).", use="SharpeEdge's meta-labeling features include candle structure.",
     evidence="Candle-shape evidence is weak (Marshall, Young & Rose 2006); useful as an ML feature.",
     video="SharpeEdge · Does Machine Learning Actually Improve a Trading Strategy?")
def _body(df):
    rng = df["high"] - df["low"]
    return (df["close"] - df["open"]) / rng.where(rng > 0)


@ind("range_exp", "Range expansion", "structure", {"length": (20, 5, 200, 1, "Average-range window.")},
     signal="state", lo=0.7, hi=1.5,
     hover="Today's high-low range versus the average range. Above 1.5 = an unusually wide bar.",
     how="(high − low) ÷ average (high − low) over the window.",
     use="Wide bars mark news and regime shifts; narrow bars (NR7) precede breakouts.", evidence="Crabel (1990).")
def _rexp(df, length):
    n = int(length)
    rng = df["high"] - df["low"]
    return rng / rng.rolling(n, min_periods=n).mean()


@ind("hh_ll", "Higher highs / lower lows", "structure", {"length": (20, 3, 200, 1, "Swing window.")},
     signal="level",
     hover="+1 when today set a new N-bar high, −1 for a new N-bar low, 0 otherwise.",
     how="Compares today's high/low with the highest high / lowest low of the previous N bars.",
     use="A pure breakout trigger.", evidence="Breakout rules: Hurst, Ooi & Pedersen (2017); Turtle system.")
def _hhll(df, length):
    n = int(length)
    hh = df["high"] > df["high"].shift(1).rolling(n, min_periods=n).max()
    ll = df["low"] < df["low"].shift(1).rolling(n, min_periods=n).min()
    out = pd.Series(np.where(hh, 1.0, np.where(ll, -1.0, 0.0)), index=df.index)
    return out.where(df["high"].shift(1).rolling(n, min_periods=n).max().notna())


@ind("dist_high", "Distance from N-bar high", "structure", {"length": (252, 20, 1000, 1, "Window.")},
     signal="level",
     hover="How close price is to its N-bar high (0 = at the high, −0.2 = 20% below).",
     how="close ÷ highest close of the last N bars − 1.",
     use="Stocks near their 52-week high tend to keep outperforming.",
     evidence="George & Hwang (2004) J. Finance — the 52-week-high momentum effect.")
def _dhigh(df, length):
    n = int(length)
    return df["close"] / df["close"].rolling(n, min_periods=n).max() - 1

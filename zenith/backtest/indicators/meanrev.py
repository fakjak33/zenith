"""Mean-reversion family: how stretched price is from 'normal'."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import ind, zscore

MR_EVID = ("Short-term reversal: Jegadeesh (1990), Lehmann (1990); returns to reversal rise with volatility "
           "(Nagel 2012). Variance-ratio tests: Lo & MacKinlay (1988).")
MR_VID = "MemLabs · How to create a Mean Reversion strategy"


@ind("zscore", "Price z-score", "meanrev", {"length": (20, 3, 500, 1, "Window for the average and spread.")},
     signal="band", lo=-2, hi=2,
     hover="How many standard deviations today's price is from its recent average. −2 = unusually low.",
     how="(close − N-bar average) ÷ N-bar standard deviation.",
     use="Buy below −2, sell above +2, exit near 0. MemLabs warns this is a heuristic, not proof of an edge — "
         "check the next-return table in EXPLORE first.",
     evidence=MR_EVID, video=MR_VID)
def _z(df, length):
    return zscore(df["close"], int(length))


@ind("bb_pctb", "Bollinger %B", "meanrev",
     {"length": (20, 5, 200, 1, "Window of the middle band."), "mult": (2.0, 0.5, 4.0, 0.25, "Band width in σ.")},
     signal="band", lo=0.0, hi=1.0,
     hover="Where price sits inside the Bollinger bands: 0 = lower band, 1 = upper band, 0.5 = the middle.",
     how="(close − lower band) ÷ (upper − lower), bands = SMA ± mult × σ.",
     use="Below 0 = outside the lower band (stretched down).", evidence="Bollinger (2001). " + MR_EVID)
def _pctb(df, length, mult):
    n = int(length)
    m, s = df["close"].rolling(n, min_periods=n).mean(), df["close"].rolling(n, min_periods=n).std()
    lo, hi = m - mult * s, m + mult * s
    return (df["close"] - lo) / (hi - lo).where(hi > lo)


@ind("bb_width", "Bollinger bandwidth", "meanrev", {"length": (20, 5, 200, 1, "Window.")}, signal="state",
     hover="How wide the Bollinger bands are relative to price. A 'squeeze' (very narrow) often precedes a big move.",
     how="(upper − lower) ÷ middle band.",
     use="Context, not direction: use it to time breakouts or to stand aside.", evidence="Bollinger (2001).")
def _bbw(df, length):
    n = int(length)
    m, s = df["close"].rolling(n, min_periods=n).mean(), df["close"].rolling(n, min_periods=n).std()
    return 4 * s / m


@ind("ibs", "Internal bar strength", "meanrev", {}, signal="band", lo=0.2, hi=0.8,
     hover="Where the close sits within today's high-low range: 0 = closed at the low, 1 = at the high.",
     how="(close − low) ÷ (high − low) for the current bar.",
     use="Equity indices tend to bounce after closing near the low (IBS < 0.2).",
     evidence="Pagonidis (2014); Levy & Lieberman, 'Overreaction of country ETFs' — next-day reversal after weak "
              "closes.")
def _ibs(df):
    rng = (df["high"] - df["low"])
    return (df["close"] - df["low"]) / rng.where(rng > 0)


@ind("ma_dist", "Distance from moving average", "meanrev", {"length": (50, 5, 400, 1, "Average length.")},
     signal="zscore",
     hover="How far price is above or below its moving average, in percent.",
     how="close ÷ SMA(N) − 1, then z-scored against its own past for the signal.",
     use="Large positive = stretched up (fade it) or strong trend (follow it) — test both with 'side'.",
     evidence=MR_EVID)
def _ma_dist(df, length):
    n = int(length)
    return df["close"] / df["close"].rolling(n, min_periods=n).mean() - 1


@ind("lag1", "Last bar's return (lag-1)", "meanrev", {}, signal="level",
     hover="Yesterday's return. MemLabs' first test: does the next bar tend to repeat it (momentum) or "
           "reverse it (mean reversion)?",
     how="log(close_t ÷ close_t−1). Used with side = −1 it bets on reversal, +1 on continuation.",
     use="Check EXPLORE › next return by lag-1 sign: if up bars are followed by down bars in BOTH halves, fade it.",
     evidence="Lo & MacKinlay (1988, 1990): serial correlation; Lehmann (1990).", video=MR_VID)
def _lag1(df):
    return np.log(df["close"]).diff()


@ind("streak", "Up/down streak", "meanrev", {}, signal="level",
     hover="How many bars in a row price has closed up (+) or down (−).",
     how="Counts consecutive closes in the same direction; resets on a change.",
     use="Long streaks of down days in indices often bounce (Connors' streak RSI).",
     evidence="Connors, Alvarez & Radtke (2012) ConnorsRSI.")
def _streak(df):
    d = np.sign(df["close"].diff()).fillna(0.0).to_numpy()
    out = np.zeros(len(d))
    for i in range(1, len(d)):
        out[i] = (out[i - 1] + d[i]) if d[i] != 0 and np.sign(out[i - 1]) in (0, d[i]) else d[i]
    return pd.Series(out, index=df.index)

"""Volume family. Needs a real volume column (market data, not futures rolls)."""

from __future__ import annotations

import numpy as np

from .catalog import ind, zscore

V_EVID = ("Volume carries information about future returns and reversals (Gervais, Kaniel & Mingelgrin 2001 "
          "high-volume return premium; Llorente, Michaely, Saar & Wang 2002). Single-indicator evidence is weak.")


@ind("obv", "On-balance volume trend", "volume", {"length": (50, 5, 400, 1, "EMA used as OBV's trend line.")},
     signal="level", needs=("volume",),
     hover="Running total of volume, added on up days and subtracted on down days. Rising OBV = buying pressure.",
     how="OBV = Σ sign(Δclose) × volume; the signal is OBV minus its EMA, scaled by average volume.",
     use="OBV rising faster than its average while price is flat can lead price.",
     evidence="Granville (1963). " + V_EVID)
def _obv(df, length):
    obv = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()
    n = int(length)
    return (obv - obv.ewm(span=n, adjust=False, min_periods=n).mean()) / df["volume"].rolling(n, min_periods=n).mean()


@ind("cmf", "Chaikin money flow", "volume", {"length": (20, 5, 200, 1, "Window.")}, signal="level",
     needs=("volume",),
     hover="Were closes near the highs on heavy volume (positive) or near the lows (negative)? −1 to +1.",
     how="Σ(money-flow multiplier × volume) ÷ Σ volume, multiplier = ((C−L) − (H−C)) ÷ (H−L).",
     use="Above 0 → accumulation.", evidence="Chaikin (1980s). " + V_EVID)
def _cmf(df, length):
    n = int(length)
    rng = (df["high"] - df["low"])
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng.where(rng > 0)
    return (mfm * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()


@ind("adl", "Accumulation / distribution trend", "volume", {"length": (50, 5, 400, 1, "Trend EMA length.")},
     signal="level", needs=("volume",),
     hover="A running total of money flow. Rising = accumulation, falling = distribution.",
     how="ADL = Σ(money-flow multiplier × volume); signal = ADL minus its EMA ÷ average volume.",
     use="Divergences with price.", evidence="Chaikin. " + V_EVID)
def _adl(df, length):
    n = int(length)
    rng = (df["high"] - df["low"])
    mfm = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng.where(rng > 0)).fillna(0)
    adl = (mfm * df["volume"]).cumsum()
    return (adl - adl.ewm(span=n, adjust=False, min_periods=n).mean()) / df["volume"].rolling(n, min_periods=n).mean()


@ind("vwap_dist", "Distance from rolling VWAP", "volume", {"length": (20, 5, 252, 1, "Window.")},
     signal="cross", overlay=False, needs=("volume",),
     hover="How far price is from the volume-weighted average price of the last N bars, in percent.",
     how="close ÷ (Σ typical price × volume ÷ Σ volume) − 1.",
     use="Institutions benchmark fills to VWAP; price far above it is 'expensive' short term.",
     evidence="Berkowitz, Logue & Noser (1988) — VWAP as the execution benchmark.")
def _vwap(df, length):
    n = int(length)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vw = (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()
    return df["close"] / vw - 1


@ind("vol_z", "Volume z-score", "volume", {"length": (50, 10, 400, 1, "Window.")}, signal="state", lo=-1, hi=2,
     needs=("volume",),
     hover="How unusual today's volume is versus its recent average, in standard deviations.",
     how="z-score of log volume over the window.",
     use="Context: big-volume days carry more information (high-volume return premium).", evidence=V_EVID)
def _volz(df, length):
    return zscore(np.log(df["volume"].where(df["volume"] > 0)), int(length))

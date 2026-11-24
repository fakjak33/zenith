"""Cross-asset family: signals from another ticker (``ref_close``) — the
MemLabs 'cross returns' idea that another asset's past can predict this one."""

from __future__ import annotations

import numpy as np

from .catalog import ind, zscore

CROSS_VID = "MemLabs · Improve Trading with Cross Returns"
CROSS_EVID = ("Lead-lag across assets: Lo & MacKinlay (1990) big stocks lead small; Hou (2007) industry leaders "
              "lead followers; Cohen & Frazzini (2008) customer-supplier links.")


@ind("rel_strength", "Relative strength vs reference", "cross",
     {"lookback": (126, 5, 750, 1, "Window for the return comparison.")}, signal="level", needs=("ref_close",),
     hover="Has this asset beaten the reference ticker over the last N bars? Positive = outperforming.",
     how="N-bar log return of this asset minus N-bar log return of the reference.",
     use="Own the stronger of two related assets (e.g. QQQ vs SPY), or rotate.",
     evidence="Relative momentum: Jegadeesh & Titman (1993); Antonacci, Dual Momentum (2014).", video=CROSS_VID)
def _rs(df, lookback):
    n = int(lookback)
    return np.log(df["close"] / df["close"].shift(n)) - np.log(df["ref_close"] / df["ref_close"].shift(n))


@ind("ref_lag", "Reference ticker's last return", "cross", {"lag": (1, 1, 20, 1, "Which lag.")}, signal="level",
     needs=("ref_close",),
     hover="The reference ticker's return k bars ago. If it leads this asset, it predicts the next move.",
     how="log return of the reference, shifted by k bars (a 'distributed lag').",
     use="MemLabs found another coin's lag predicted ETH better than ETH's own past. Check it in EXPLORE first.",
     evidence=CROSS_EVID, video=CROSS_VID)
def _reflag(df, lag):
    return np.log(df["ref_close"]).diff().shift(int(lag) - 1)


@ind("spread_z", "Spread z-score vs reference", "cross",
     {"length": (60, 10, 500, 1, "Window for the spread's average and spread."),
      "beta_len": (252, 30, 1000, 10, "Window for the hedge ratio.")}, signal="band", lo=-2, hi=2,
     needs=("ref_close",),
     hover="How stretched the price ratio between this asset and the reference is. Far from 0 = the pair has "
           "drifted apart.",
     how="Rolling hedge ratio β from log prices; spread = log P − β log P_ref; z-score over the window.",
     use="Pairs trading: buy the laggard when the spread is very negative, expecting it to close.",
     evidence="Gatev, Goetzmann & Rouwenhorst (2006) Rev. Fin. Studies — pairs trading earned excess returns that "
              "faded after 2000.", video="MemLabs · Feature Engineering for Quant Trading (spread lag)")
def _spreadz(df, length, beta_len):
    a, b = np.log(df["close"]), np.log(df["ref_close"])
    m = int(beta_len)
    beta = a.rolling(m, min_periods=m).cov(b) / b.rolling(m, min_periods=m).var()
    return zscore(a - beta * b, int(length))


@ind("ref_corr", "Correlation with reference", "cross", {"length": (63, 10, 500, 1, "Window.")}, signal="state",
     lo=0.3, hi=0.8, needs=("ref_close",),
     hover="Rolling correlation of daily returns with the reference ticker. Near 1 = they move together.",
     how="Pearson correlation of log returns over the window.",
     use="Context: diversification disappears when correlations jump in a crisis.",
     evidence="Longin & Solnik (2001): correlations rise in bear markets.")
def _refcorr(df, length):
    n = int(length)
    return np.log(df["close"]).diff().rolling(n, min_periods=n).corr(np.log(df["ref_close"]).diff())

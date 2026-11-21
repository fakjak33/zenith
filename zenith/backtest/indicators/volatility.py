"""Volatility & market-state family. Mostly *context* (``state``) variables
that the regime tools and the explorer use to condition other signals."""

from __future__ import annotations

import numpy as np

from ...volscreen import realized as rv
from .catalog import atr, ind, true_range

REG_VID = "SharpeEdge · Market Regime Detection for Algo Trading"
VOL_EVID = ("Volatility clusters and is forecastable (Engle 1982; Bollerslev 1986); scaling exposure down when vol "
            "is high improves Sharpe for many factors (Moreira & Muir 2017).")
ANN = np.sqrt(252)


@ind("atr", "Average true range", "volatility", {"length": (14, 2, 200, 1, "Wilder smoothing length.")},
     signal="state",
     hover="The average daily trading range, including gaps, in price units. Used for stops and sizing.",
     how="Wilder average of max(high−low, |high−prev close|, |low−prev close|).",
     use="Stops at k × ATR, position size ∝ 1/ATR.", evidence="Wilder (1978).")
def _atr(df, length):
    return atr(df, int(length))


@ind("natr", "Normalised ATR %", "volatility", {"length": (14, 2, 200, 1, "Wilder smoothing length.")},
     signal="state",
     hover="ATR as a percent of price, so it can be compared across assets and over time.",
     how="100 × ATR ÷ close.", use="A quick volatility regime gauge.", evidence="Wilder (1978).")
def _natr(df, length):
    return 100 * atr(df, int(length)) / df["close"]


@ind("rv_cc", "Realised vol · close-to-close", "volatility", {"length": (21, 5, 252, 1, "Window in bars.")},
     signal="state", hover="Annualised standard deviation of daily returns over the window.",
     how="std(log returns) × √252.", use="The textbook vol measure; noisy on short windows.", evidence=VOL_EVID)
def _rvcc(df, length):
    return rv.close_to_close(df, int(length))


@ind("rv_park", "Realised vol · Parkinson", "volatility", {"length": (21, 5, 252, 1, "Window in bars.")},
     signal="state", hover="Volatility estimated from each bar's high-low range: about 5× more efficient than "
                           "close-to-close.",
     how="√(mean(ln(H/L)²) ÷ 4ln2) × √252.", use="Better vol with fewer bars.", evidence="Parkinson (1980).")
def _rvpk(df, length):
    return rv.parkinson(df, int(length))


@ind("rv_gk", "Realised vol · Garman-Klass", "volatility", {"length": (21, 5, 252, 1, "Window in bars.")},
     signal="state", hover="Volatility from open, high, low and close together.",
     how="Garman-Klass OHLC estimator, annualised.", use="Efficient when there are few overnight gaps.",
     evidence="Garman & Klass (1980).")
def _rvgk(df, length):
    return rv.garman_klass(df, int(length))


@ind("rv_yz", "Realised vol · Yang-Zhang", "volatility", {"length": (21, 5, 252, 1, "Window in bars.")},
     signal="state", hover="Volatility that handles overnight gaps and drift: the best all-round OHLC estimator.",
     how="Overnight variance + k × open-to-close variance + (1−k) × Rogers-Satchell.",
     use="Default choice for daily equity data.", evidence="Yang & Zhang (2000).")
def _rvyz(df, length):
    return rv.yang_zhang(df, int(length))


@ind("rv_ewma", "Realised vol · EWMA (RiskMetrics)", "volatility",
     {"lam": (0.94, 0.80, 0.99, 0.01, "Decay: 0.94 is RiskMetrics' daily value. Higher = slower.")},
     signal="state", hover="Exponentially weighted volatility: yesterday's estimate blended with today's move.",
     how="σ²_t = λσ²_t−1 + (1−λ)r²_t, annualised.", use="Reacts quickly to shocks.",
     evidence="J.P. Morgan RiskMetrics (1996).")
def _rvew(df, lam):
    return rv.ewma(df, float(lam))


@ind("volofvol", "Vol of vol", "volatility",
     {"length": (21, 5, 126, 1, "Window of the vol estimate."), "outer": (63, 10, 252, 1, "Window of its std.")},
     signal="state", hover="How unstable volatility itself is. Spikes when markets switch regime.",
     how="std of the rolling realised vol over the outer window, ÷ its mean.",
     use="High vol-of-vol = regime uncertainty; size down.", evidence=VOL_EVID)
def _vov(df, length, outer):
    v = rv.close_to_close(df, int(length))
    o = int(outer)
    return v.rolling(o, min_periods=o).std() / v.rolling(o, min_periods=o).mean()


@ind("chop", "Choppiness index", "volatility", {"length": (14, 5, 100, 1, "Window.")}, signal="state",
     lo=38.2, hi=61.8,
     hover="0-100: high (> 61.8) = sideways chop, low (< 38.2) = a clean trend. Direction-free.",
     how="100 × log10(Σ ATR(1) ÷ (highest high − lowest low)) ÷ log10(N).",
     use="Switch between trend and mean-reversion rules.", evidence="Dreiss (1990s), practitioner.")
def _chop(df, length):
    n = int(length)
    s = true_range(df).rolling(n, min_periods=n).sum()
    rng = df["high"].rolling(n, min_periods=n).max() - df["low"].rolling(n, min_periods=n).min()
    return 100 * np.log10(s / rng.where(rng > 0)) / np.log10(n)


@ind("er", "Efficiency ratio", "volatility", {"length": (20, 5, 252, 1, "Window.")}, signal="state",
     lo=0.2, hi=0.5,
     hover="0-1: how much of the path was a straight line. 1 = moved only one way, 0 = went nowhere.",
     how="|close − close N bars ago| ÷ sum of |bar-to-bar changes|.",
     use="SharpeEdge's 'directionality' axis for regimes: high ER → trend rules, low ER → mean reversion.",
     evidence="Kaufman (1995).", video=REG_VID)
def _er(df, length):
    n = int(length)
    c = df["close"]
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    return (c - c.shift(n)).abs() / path.where(path > 0)


@ind("hurst", "Hurst exponent", "volatility", {"length": (128, 32, 512, 16, "Window in bars.")}, signal="state",
     lo=0.45, hi=0.55,
     hover="Above 0.5 = trending (moves persist), below 0.5 = mean-reverting, 0.5 = random walk.",
     how="Slope of log(range of cumulative deviations) vs log(window) across sub-windows (rescaled range).",
     use="A regime gauge; noisy on short windows, so treat it as a tilt not a switch.",
     evidence="Hurst (1951); Lo (1991) shows R/S is biased by short-term memory.", video=REG_VID)
def _hurst(df, length):
    n = int(length)
    r = np.log(df["close"]).diff()
    sizes = [s for s in (8, 16, 32, 64, 128, 256) if s <= n // 2]

    def h(x):
        if len(sizes) < 2:
            return np.nan
        rs = []
        for s in sizes:
            k = len(x) // s
            seg = x[-k * s:].reshape(k, s)
            dev = (seg - seg.mean(1, keepdims=True)).cumsum(1)
            sd = seg.std(1)
            ok = sd > 0
            rs.append(np.mean((dev.max(1) - dev.min(1))[ok] / sd[ok]) if ok.any() else np.nan)
        rs = np.array(rs)
        if not np.all(np.isfinite(rs)):
            return np.nan
        return np.polyfit(np.log(sizes), np.log(rs), 1)[0]

    return r.rolling(n, min_periods=n).apply(h, raw=True)


@ind("varratio", "Variance ratio", "volatility",
     {"q": (5, 2, 60, 1, "Horizon q: compares q-bar variance with q × 1-bar variance."),
      "length": (252, 60, 1000, 10, "Window.")}, signal="state", lo=0.9, hi=1.1,
     hover="Above 1 = returns trend (q-bar moves bigger than chance), below 1 = they mean-revert.",
     how="Var(q-bar returns) ÷ (q × Var(1-bar returns)) over the window. Random walk = 1.",
     use="Lo-MacKinlay's test of the random walk; the regime video uses it as a directionality measure.",
     evidence="Lo & MacKinlay (1988) Rev. Fin. Studies.", video=REG_VID)
def _vr(df, q, length):
    q, n = int(q), int(length)
    r = np.log(df["close"]).diff()
    rq = r.rolling(q, min_periods=q).sum()
    return rq.rolling(n, min_periods=n).var() / (q * r.rolling(n, min_periods=n).var())


@ind("autocorr", "Rolling autocorrelation", "volatility",
     {"lag": (1, 1, 20, 1, "Which lag."), "length": (126, 30, 1000, 1, "Window.")}, signal="level",
     hover="Correlation of each bar's return with the return k bars earlier. Positive = momentum, negative = "
           "reversal.",
     how="Rolling Pearson correlation of r_t with r_t−k.",
     use="MemLabs: a negative lag-1 weight means mean reversion, positive means momentum.",
     evidence="Lo & MacKinlay (1990); Campbell, Lo & MacKinlay (1997).",
     video="MemLabs · Quant Trading Accelerator · Part 4: Time Series")
def _ac(df, lag, length):
    r = np.log(df["close"]).diff()
    return r.rolling(int(length), min_periods=int(length)).corr(r.shift(int(lag)))


@ind("drawdown", "Drawdown from high", "volatility", {}, signal="state",
     hover="How far price is below its highest close so far, in percent (0 = at a new high).",
     how="close ÷ running maximum of close − 1.",
     use="Context: many strategies behave differently near highs vs deep in a drawdown.",
     evidence="Drawdown-conditioned behaviour of momentum: Daniel & Moskowitz (2016).")
def _dd(df):
    c = df["close"]
    return c / c.cummax() - 1


@ind("vix", "VIX level", "volatility", {}, signal="state", lo=15, hi=25, needs=("vix",),
     hover="The market's expected 30-day S&P 500 volatility, from option prices. Above 25-30 = stress.",
     how="Cboe VIX close (market data only).",
     use="A fear gauge: many long strategies do worse when VIX is high; short-term reversal does better.",
     evidence="Whaley (2000); Nagel (2012) reversal returns rise with VIX.")
def _vix(df):
    return df["vix"]


@ind("vix_term", "VIX term structure (VIX ÷ VIX3M)", "volatility", {}, signal="state", lo=0.9, hi=1.0,
     needs=("vix", "vix3m"),
     hover="Short-term fear vs 3-month fear. Above 1 (inverted) = acute stress right now.",
     how="VIX ÷ VIX3M. Normally ~0.85-0.95 (contango).",
     use="Inversion flags panics; risk-off filters often key on it.",
     evidence="Johnson (2017) J. Fin. Quant. Anal. — the VIX term-structure slope predicts variance returns.")
def _vixts(df):
    return df["vix"] / df["vix3m"]


@ind("vol_pct", "Vol percentile (expanding)", "volatility",
     {"length": (21, 5, 126, 1, "Realised-vol window.")}, signal="state", lo=0.33, hi=0.67,
     hover="Where today's volatility ranks against all of its own history so far, 0-1. Never uses the future.",
     how="Expanding percentile rank of the rolling realised vol (warm-up 252 bars).",
     use="The causal way to say 'high vol' — thresholds learnt only from the past.",
     evidence=VOL_EVID, video=REG_VID)
def _volpct(df, length):
    v = rv.close_to_close(df, int(length))
    return v.expanding(min_periods=252).rank(pct=True)

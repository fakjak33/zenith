"""Trend family: moving averages, trend strength, channels and filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import atr, ema, ind, wilder

L = (lambda d, lo=2, hi=500, step=1, h="How many bars the average looks back. Longer = smoother but slower to react.":
     (d, lo, hi, step, h))

MA_EVID = ("Brock, Lakonishok & LeBaron (1992) J. Finance — moving-average rules had predictive power on the Dow "
           "1897-1986; Faber (2007) — a 10-month average halved equity drawdowns; Levine & Pedersen (2016) — all "
           "trend filters are weighted sums of past returns.")
TREND_VID = "SharpeEdge · Is the Kalman Filter Just a Fancy Moving Average?"


@ind("sma", "Simple moving average", "trend", {"length": L(50)}, signal="cross", overlay=True,
     hover="The average closing price over the last N bars. Price above it = uptrend, below = downtrend.",
     how="Add up the last N closes and divide by N. Every bar in the window counts equally.",
     use="Long while price is above the line, short (or flat) below it. 50, 100 and 200 days are the classics.",
     evidence=MA_EVID, video=TREND_VID)
def _sma(df, length):
    return df["close"].rolling(int(length), min_periods=int(length)).mean()


@ind("ema", "Exponential moving average", "trend", {"length": L(50)}, signal="cross", overlay=True,
     hover="A moving average that weights recent prices more, so it turns faster than a simple average.",
     how="Each bar: EMA = EMA_yesterday + k × (close − EMA_yesterday), with k = 2/(N+1).",
     use="Same as the SMA, with less lag. SharpeEdge shows a Kalman trend filter often collapses to an EMA.",
     evidence=MA_EVID, video=TREND_VID)
def _ema(df, length):
    return ema(df["close"], length)


@ind("wma", "Weighted moving average", "trend", {"length": L(30)}, signal="cross", overlay=True,
     hover="A moving average whose weights fall in a straight line: the newest bar counts most, the oldest least.",
     how="Weights N, N−1, …, 1 applied to the last N closes, divided by the sum of the weights.",
     use="A middle ground between SMA (equal weights) and EMA (exponential weights).", evidence=MA_EVID)
def _wma(df, length):
    n = int(length)
    w = np.arange(1, n + 1, dtype=float)
    return df["close"].rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


@ind("hma", "Hull moving average", "trend", {"length": L(30, 4)}, signal="cross", overlay=True,
     hover="A very fast moving average that cancels most of the lag by combining two weighted averages.",
     how="WMA( 2×WMA(N/2) − WMA(N), √N ). Fast and smooth, but it overshoots at turns.",
     use="For quicker trend flips. Expect more trades, so costs matter more.",
     evidence="Hull (2005). Practitioner indicator; no peer-reviewed edge beyond the general MA evidence.")
def _hma(df, length):
    n = int(length)

    def wma(s, k):
        k = max(1, int(k))
        w = np.arange(1, k + 1, dtype=float)
        return s.rolling(k, min_periods=k).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    c = df["close"]
    return wma(2 * wma(c, n / 2) - wma(c, n), np.sqrt(n))


@ind("dema", "Double EMA", "trend", {"length": L(30)}, signal="cross", overlay=True,
     hover="An EMA with its own lag subtracted out: reacts faster than a plain EMA of the same length.",
     how="2 × EMA(N) − EMA(EMA(N)).", use="A faster trend line.", evidence="Mulloy (1994) TASC.")
def _dema(df, length):
    e = ema(df["close"], length)
    return 2 * e - ema(e, length)


@ind("tema", "Triple EMA", "trend", {"length": L(30)}, signal="cross", overlay=True,
     hover="Like the double EMA, with one more layer of lag removed. Very responsive, quite noisy.",
     how="3×EMA − 3×EMA(EMA) + EMA(EMA(EMA)).", use="A very fast trend line.", evidence="Mulloy (1994) TASC.")
def _tema(df, length):
    e1 = ema(df["close"], length)
    e2 = ema(e1, length)
    return 3 * e1 - 3 * e2 + ema(e2, length)


@ind("kama", "Kaufman adaptive MA", "trend",
     {"length": L(10), "fast": (2, 2, 30, 1, "Speed when the market trends cleanly."),
      "slow": (30, 5, 200, 1, "Speed when the market is choppy.")},
     signal="cross", overlay=True,
     hover="A moving average that speeds up when price moves in a straight line and slows down in chop.",
     how="Uses the Efficiency Ratio (net move ÷ total path) to slide the smoothing between a fast and slow EMA.",
     use="Fewer whipsaws in sideways markets than a fixed-length average.",
     evidence="Kaufman, Trading Systems and Methods (1995).")
def _kama(df, length, fast, slow):
    c = df["close"]
    n = int(length)
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / vol.where(vol > 0)).fillna(0.0)
    fs, ss = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fs - ss) + ss) ** 2
    out = np.full(len(c), np.nan)
    cv, scv = c.to_numpy(), sc.to_numpy()
    for i in range(n, len(c)):
        prev = out[i - 1] if np.isfinite(out[i - 1]) else cv[i - 1]
        out[i] = prev + scv[i] * (cv[i] - prev)
    return pd.Series(out, index=c.index)


@ind("macd", "MACD histogram", "trend",
     {"fast": (12, 2, 100, 1, "Fast EMA length."), "slow": (26, 5, 300, 1, "Slow EMA length."),
      "signal": (9, 2, 50, 1, "EMA of the MACD line used as its trigger.")},
     signal="level",
     hover="The gap between a fast and a slow EMA, minus its own average. Positive = momentum is building upward.",
     how="MACD = EMA(fast) − EMA(slow); signal line = EMA(MACD); histogram = MACD − signal; here ÷ price.",
     use="Histogram above zero → long, below → short. Classic 12/26/9.",
     evidence="Appel (1979). Equivalent to an EMA-crossover trend rule (Levine & Pedersen 2016).")
def _macd(df, fast, slow, signal):
    c = df["close"]
    m = ema(c, fast) - ema(c, slow)
    return (m - ema(m, signal)) / c


@ind("ppo", "Percentage price oscillator", "trend",
     {"fast": (12, 2, 100, 1, "Fast EMA length."), "slow": (26, 5, 300, 1, "Slow EMA length.")},
     signal="level",
     hover="How far the fast EMA is above the slow EMA, in percent. A scale-free MACD.",
     how="(EMA(fast) − EMA(slow)) ÷ EMA(slow) × 100.",
     use="Positive → uptrend. Comparable across assets because it is in percent.", evidence=MA_EVID)
def _ppo(df, fast, slow):
    c = df["close"]
    s = ema(c, slow)
    return (ema(c, fast) - s) / s * 100


def _dmi(df, n):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * wilder(plus, n) / a
    mdi = 100 * wilder(minus, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where(pdi + mdi > 0)
    return pdi, mdi, wilder(dx.fillna(0.0), n)


@ind("adx", "ADX trend strength", "trend", {"length": L(14)}, signal="state", lo=20, hi=25,
     hover="How strongly the market is trending (0-100), regardless of direction. Above 25 = a real trend.",
     how="Wilder's average of |+DI − −DI| ÷ (+DI + −DI), where DI measures up-moves vs down-moves.",
     use="A filter: only take trend trades when ADX > 20-25. On its own it has no direction.",
     evidence="Wilder (1978). SharpeEdge uses ADX as one of the 'directionality' regime measures.",
     video="SharpeEdge · Market Regime Detection for Algo Trading")
def _adx(df, length):
    return _dmi(df, int(length))[2]


@ind("dmi", "Directional movement (+DI − −DI)", "trend", {"length": L(14)}, signal="level",
     hover="Up-move strength minus down-move strength. Positive = buyers in control.",
     how="+DI and −DI from Wilder: the share of the average range made by up-moves vs down-moves.",
     use="Long when +DI > −DI. Usually combined with ADX to avoid chop.", evidence="Wilder (1978).")
def _dmi_diff(df, length):
    p, m, _ = _dmi(df, int(length))
    return p - m


@ind("aroon", "Aroon oscillator", "trend", {"length": L(25)}, signal="level",
     hover="How recently the high vs the low was set. Recent highs = uptrend (+100), recent lows = downtrend.",
     how="Aroon up = 100 × (N − bars since N-bar high)/N; down likewise for the low; oscillator = up − down.",
     use="Above 0 → uptrend.", evidence="Chande (1995).")
def _aroon(df, length):
    n = int(length)
    hi = df["high"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - (n - np.argmax(x)), raw=True)
    lo = df["low"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - (n - np.argmin(x)), raw=True)
    return 100 * (hi - lo) / n


@ind("supertrend", "Supertrend", "trend",
     {"length": L(10), "mult": (3.0, 0.5, 10.0, 0.5, "Band width in ATRs. Wider = fewer flips.")},
     signal="cross", overlay=True,
     hover="A trailing stop line that sits below price in uptrends and flips above it in downtrends.",
     how="Bands at (high+low)/2 ± mult×ATR that only ratchet in the trend's direction; flips when price closes through.",
     use="Long while close is above the line. Popular with retail; a volatility-scaled breakout in disguise.",
     evidence="Olivier Seban (practitioner). Related evidence: ATR channel breakouts (Clenow 2013).")
def _supertrend(df, length, mult):
    a = atr(df, int(length)).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    up, dn = hl2 - mult * a, hl2 + mult * a
    out = np.full(len(c), np.nan)
    fu, fd, trend = np.nan, np.nan, 1
    for i in range(len(c)):
        if not np.isfinite(a[i]):
            continue
        fu = up[i] if not np.isfinite(fu) or c[i - 1] < fu else max(up[i], fu)
        fd = dn[i] if not np.isfinite(fd) or c[i - 1] > fd else min(dn[i], fd)
        if trend == 1 and c[i] < fu:
            trend = -1
        elif trend == -1 and c[i] > fd:
            trend = 1
        out[i] = fu if trend == 1 else fd
    return pd.Series(out, index=df.index)


@ind("psar", "Parabolic SAR", "trend",
     {"step": (0.02, 0.005, 0.1, 0.005, "Acceleration added each new extreme."),
      "max_step": (0.2, 0.05, 0.5, 0.05, "Cap on the acceleration.")},
     signal="cross", overlay=True,
     hover="A dotted stop that accelerates toward price as a trend matures, then flips when price touches it.",
     how="SAR moves toward the trend's extreme price by an acceleration factor that grows with each new extreme.",
     use="Long while price is above the SAR. Always in the market; whipsaws in ranges.", evidence="Wilder (1978).")
def _psar(df, step, max_step):
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(h)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=df.index)
    bull, af, ep, sar = True, step, h[0], lo[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar:
                bull, sar, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                bull, sar, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        out[i] = sar
    return pd.Series(out, index=df.index)


@ind("ichimoku", "Ichimoku cloud position", "trend",
     {"tenkan": (9, 2, 60, 1, "Conversion line length."), "kijun": (26, 5, 120, 1, "Base line length."),
      "senkou": (52, 10, 240, 1, "Leading span B length.")},
     signal="level",
     hover="Where price sits versus the Ichimoku 'cloud': above = +1 (uptrend), inside = 0, below = −1.",
     how="Cloud = the two leading spans (midpoints of recent highs/lows), as they were plotted for today.",
     use="Trade only in the direction price has broken out of the cloud.",
     evidence="Hosoda (1969). Practitioner; mixed academic evidence (e.g. Lim, Yanyali & Savidge 2016).")
def _ichimoku(df, tenkan, kijun, senkou):
    mid = lambda n: (df["high"].rolling(int(n)).max() + df["low"].rolling(int(n)).min()) / 2  # noqa: E731
    a = ((mid(tenkan) + mid(kijun)) / 2).shift(int(kijun))
    b = mid(senkou).shift(int(kijun))
    top, bot = pd.concat([a, b], axis=1).max(axis=1), pd.concat([a, b], axis=1).min(axis=1)
    c = df["close"]
    out = pd.Series(np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0)), index=df.index)
    return out.where(top.notna())


@ind("donchian", "Donchian channel position", "trend", {"length": L(55)}, signal="level",
     hover="Where today's close sits inside the N-bar high-low range: +1 at the top, −1 at the bottom.",
     how="(close − midpoint) ÷ (half the range), with the range from the last N bars.",
     use="The Turtle breakout family: buy new highs, sell new lows.",
     evidence="Donchian (1960s); Turtle traders; breakout rules in Hurst, Ooi & Pedersen (2017).")
def _donchian(df, length):
    n = int(length)
    hi, lo = df["high"].rolling(n, min_periods=n).max(), df["low"].rolling(n, min_periods=n).min()
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return (df["close"] - mid) / half.where(half > 0)


@ind("keltner", "Keltner channel position", "trend",
     {"length": L(20), "mult": (2.0, 0.5, 5.0, 0.25, "Channel half-width in ATRs.")}, signal="level",
     hover="How many channel-widths price is above or below its EMA, using ATR for the width.",
     how="(close − EMA(N)) ÷ (mult × ATR(N)). +1 = at the upper band.",
     use="Above +1 is a breakout (trend) or a stretch (reversion), depending on the market.",
     evidence="Keltner (1960); ATR version by Raschke.")
def _keltner(df, length, mult):
    return (df["close"] - ema(df["close"], length)) / (mult * atr(df, int(length)))


def _roll_reg(y: pd.Series, n: int):
    n = int(n)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    ym = y.rolling(n, min_periods=n).mean()
    sxy = y.rolling(n, min_periods=n).apply(lambda v: np.dot(x - xm, v), raw=True)
    slope = sxy / sxx
    syy = y.rolling(n, min_periods=n).var(ddof=0) * n
    resid = (syy - slope ** 2 * sxx).clip(lower=0)
    se = np.sqrt(resid / max(n - 2, 1) / sxx)
    r2 = 1 - resid / syy.where(syy > 0)
    return slope, se, r2, ym


@ind("reg_slope", "Regression slope t-stat", "trend", {"length": L(63, 10)}, signal="level",
     hover="Fits a straight line through the last N log-prices and asks how confident we are that it slopes.",
     how="OLS of log(close) on time over N bars; t-stat = slope ÷ its standard error.",
     use="Long when t > 0 (up-slope). SharpeEdge trades the slope t-stat of a Kalman trend the same way.",
     evidence="Rolling-regression trend filters: Levine & Pedersen (2016); Clenow, Stocks on the Move (2015).",
     video=TREND_VID)
def _reg_slope(df, length):
    s, se, _, _ = _roll_reg(np.log(df["close"]), length)
    return s / se.where(se > 0)


@ind("reg_r2", "Regression R² (trend quality)", "trend", {"length": L(63, 10)}, signal="state", lo=0.2, hi=0.6,
     hover="How straight the recent price path is (0 = random, 1 = a perfect line). Direction-free.",
     how="R² of the rolling regression of log(close) on time.",
     use="A trend-quality filter: trade trend rules only when R² is high.",
     evidence="SharpeEdge regime video lists regression slope and R² as directionality measures.",
     video="SharpeEdge · Market Regime Detection for Algo Trading")
def _reg_r2(df, length):
    return _roll_reg(np.log(df["close"]), length)[2]


@ind("kalman", "Kalman local trend", "trend",
     {"horizon": (40, 5, 400, 5, "Trend horizon in bars. Sets how much the filter trusts new prices.")},
     signal="cross", overlay=True,
     hover="A statistical filter that estimates the 'true' price level and slope, updating as each bar arrives.",
     how="Local-linear-trend Kalman filter on log price, noise ratio set from the horizon (no fitting). "
         "Filtered, never smoothed: smoothed estimates use the future.",
     use="Long while price is above the filtered level. SharpeEdge shows it behaves much like an EMA.",
     evidence="Harvey (1989) structural time series; Levine & Pedersen (2016) filter equivalence.",
     video=TREND_VID)
def _kalman(df, horizon):
    y = np.log(df["close"].to_numpy(dtype=float))
    n = len(y)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=df.index)
    lam = 2.0 / (float(horizon) + 1.0)
    q_level = (lam ** 2) / max(1 - lam, 1e-6)
    q_slope = q_level * lam ** 2
    rv = np.nanvar(np.diff(y[: min(n, 60)])) if n > 2 else 1e-4
    r = max(rv, 1e-8)
    x = np.array([y[0], 0.0])
    P = np.eye(2) * r
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level * r, q_slope * r])
    for t in range(n):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        if np.isfinite(y[t]):
            k = P[:, 0] / (P[0, 0] + r)
            x = x + k * (y[t] - x[0])
            P = P - np.outer(k, P[0, :])
        out[t] = x[0]
    s = pd.Series(np.exp(out), index=df.index)
    # the noise scale comes from the first 60 bars, so nothing before bar 60 is shown (no peeking)
    s.iloc[: min(max(int(horizon), 60), n)] = np.nan
    return s


@ind("tsmom", "Time-series momentum", "trend",
     {"lookback": L(252, 10, 750, 5, "How far back the return is measured. 12 months is the classic.")},
     signal="level",
     hover="Has this asset gone up over the last N bars? If yes, own it; if no, short it.",
     how="Past N-bar return divided by its volatility, so quiet and wild markets are comparable.",
     use="The core rule of managed-futures funds. Works across asset classes over 1-12 months.",
     evidence="Moskowitz, Ooi & Pedersen (2012) J. Fin. Econ.; Hurst, Ooi & Pedersen (2017) — 137 years.",
     video="MemLabs · How to create a Trend Following strategy")
def _tsmom(df, lookback):
    n = int(lookback)
    r = np.log(df["close"]).diff()
    return r.rolling(n, min_periods=n).sum() / (r.rolling(n, min_periods=n).std() * np.sqrt(n))

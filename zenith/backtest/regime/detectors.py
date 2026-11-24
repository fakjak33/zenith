"""Causal market-regime detectors.

Every detector returns a label Series on the bar index. The label at bar t
uses data up to and including t only: thresholds come from *expanding*
percentiles (what you could have known then), never from the full sample, and
the hidden-Markov model is fitted on a training window and then run forward as
a *filter* (smoothed probabilities would use the future). Bars before a
detector has enough history are NaN ("warm-up").

``debounce`` adds hysteresis: a new label is only adopted after it has held
for ``min_bars`` bars in a row, which kills one-bar whipsaws (SharpeEdge's
"minimum duration" fix).

Detectors follow SharpeEdge's market-regime video (volatility × directionality
grid, ER/ADX measures, HMM filtered-not-smoothed) and the academic regime
literature (Hamilton 1989; Ang & Bekaert 2002; Kritzman, Page & Turkington 2012).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..indicators.trend import _dmi


@dataclass
class Detector:
    key: str
    label: str
    fn: Callable[..., pd.Series]
    params: dict = field(default_factory=dict)       # name -> (default, lo, hi, step, help)
    order: tuple = ()                                # label order for tables / colours
    help: str = ""
    needs: tuple = ()                                # aux columns (vix, vix3m)
    cite: str = ""


DETECTORS: dict[str, Detector] = {}


def _det(key, label, params=None, order=(), help="", needs=(), cite=""):
    def wrap(fn):
        DETECTORS[key] = Detector(key, label, fn, params or {}, order, help, needs, cite)
        return fn
    return wrap


def debounce(labels: pd.Series, min_bars: int) -> pd.Series:
    """Adopt a new label only after it has persisted ``min_bars`` consecutive bars (causal)."""
    m = int(min_bars)
    if m <= 1 or labels.empty:
        return labels
    vals = labels.to_numpy(dtype=object)
    out = np.empty(len(vals), dtype=object)
    cur, cand, run = None, None, 0
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = cur if cur is not None else np.nan
            continue
        if cur is None:
            cur = v
        if v == cur:
            cand, run = None, 0
        elif v == cand:
            run += 1
            if run >= m:
                cur, cand, run = v, None, 0
        else:
            cand, run = v, 1
            if run >= m:
                cur, cand, run = v, None, 0
        out[i] = cur
    return pd.Series(out, index=labels.index)


def _expanding_rank(s: pd.Series, warmup: int = 252) -> pd.Series:
    return s.expanding(min_periods=warmup).rank(pct=True)


def _label(cond_list, labels, index) -> pd.Series:
    out = pd.Series(np.select([c.fillna(False).to_numpy() for c in cond_list], labels, default=None),
                    index=index, dtype=object)
    return out.where(out.notna(), np.nan)


# ----------------------------------------------------------------- trend ----
@_det("trend", "Trend (price vs moving average)",
      {"length": (200, 20, 400, 10, "Moving-average length in bars."),
       "band": (0.0, 0.0, 5.0, 0.5, "Hysteresis band in %: price must clear the average by this much to flip.")},
      order=("uptrend", "downtrend"),
      help="Uptrend while price is above its N-bar average, downtrend below. The simplest regime there is.",
      cite="Faber (2007); Daniel & Moskowitz (2016) — momentum crashes after downtrends.")
def trend(df, length=200, band=0.0, **_):
    c = df["close"]
    sma = c.rolling(int(length), min_periods=int(length)).mean()
    up, dn = c > sma * (1 + band / 100), c < sma * (1 - band / 100)
    raw = pd.Series(np.where(up, "uptrend", np.where(dn, "downtrend", None)), index=c.index, dtype=object)
    # inside the hysteresis band keep the previous label
    return raw.ffill().where(sma.notna(), np.nan)


# -------------------------------------------------------------- volatility ----
@_det("vol", "Volatility tercile (expanding)",
      {"length": (21, 5, 126, 1, "Realised-vol window in bars."),
       "lo": (33, 5, 50, 1, "Percentile below which vol is 'calm'."),
       "hi": (67, 50, 95, 1, "Percentile above which vol is 'stressed'.")},
      order=("calm", "normal", "stressed"),
      help="Today's realised volatility ranked against all of its own history so far.",
      cite="Moreira & Muir (2017) — volatility-managed portfolios.")
def vol(df, length=21, lo=33, hi=67, **_):
    r = np.log(df["close"]).diff()
    v = r.rolling(int(length), min_periods=int(length)).std()
    pct = _expanding_rank(v) * 100
    return _label([pct < lo, pct > hi, pct.notna()], ["calm", "stressed", "normal"], df.index)


# --------------------------------------------------------------------- VIX ----
@_det("vix", "VIX level", {"low": (15, 8, 25, 1, "Below = complacent."), "high": (25, 15, 45, 1, "Above = fear."),
                           "crisis": (35, 25, 80, 1, "Above = crisis.")},
      order=("vix low", "vix normal", "vix high", "vix crisis"), needs=("vix",),
      help="Buckets of the VIX close (implied 30-day S&P 500 volatility). Market data only.",
      cite="Whaley (2000); Nagel (2012).")
def vix(df, low=15, high=25, crisis=35, **_):
    if "vix" not in df:
        return pd.Series(np.nan, index=df.index, dtype=object)
    v = df["vix"]
    return _label([v < low, v >= crisis, v >= high, v.notna()], ["vix low", "vix crisis", "vix high", "vix normal"],
                  df.index)


# ----------------------------------------------------------- directionality ----
@_det("direction", "Directionality (efficiency ratio)",
      {"length": (20, 5, 252, 1, "Efficiency-ratio window."),
       "split": (50, 20, 80, 1, "Expanding percentile that separates trending from choppy.")},
      order=("trending", "choppy"),
      help="Is price moving in a straight line (trending) or going back and forth (choppy)? Uses Kaufman's "
           "efficiency ratio ranked against its own past.",
      cite="Kaufman (1995); SharpeEdge · Market Regime Detection.")
def direction(df, length=20, split=50, **_):
    c = df["close"]
    n = int(length)
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    er = (c - c.shift(n)).abs() / path.where(path > 0)
    pct = _expanding_rank(er) * 100
    return _label([pct >= split, pct.notna()], ["trending", "choppy"], df.index)


@_det("adx", "Trend strength (ADX)", {"length": (14, 5, 50, 1, "ADX length."),
                                      "level": (25, 10, 50, 1, "ADX above this = trending.")},
      order=("trending", "choppy"),
      help="Wilder's ADX above the level = a trending market, below = a range.", cite="Wilder (1978).")
def adx(df, length=14, level=25, **_):
    a = _dmi(df, int(length))[2]
    a = a.where(a > 0)
    return _label([a >= level, a.notna()], ["trending", "choppy"], df.index)


# ------------------------------------------------------- vol × direction grid ----
@_det("grid", "Volatility × direction grid",
      {"vol_len": (21, 5, 126, 1, "Realised-vol window."), "er_len": (20, 5, 252, 1, "Efficiency-ratio window.")},
      order=("calm trend", "calm range", "wild trend", "wild range"),
      help="SharpeEdge's 2 × 2 map: volatility (above / below its expanding median) × directionality (efficiency "
           "ratio above / below its expanding median). Trend rules like 'trend'; reversion likes 'range'.",
      cite="SharpeEdge · Market Regime Detection for Algo Trading; Kritzman, Page & Turkington (2012).")
def grid(df, vol_len=21, er_len=20, **_):
    r = np.log(df["close"]).diff()
    v = _expanding_rank(r.rolling(int(vol_len), min_periods=int(vol_len)).std())
    c = df["close"]
    n = int(er_len)
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    e = _expanding_rank((c - c.shift(n)).abs() / path.where(path > 0))
    ok = v.notna() & e.notna()
    hi_v, hi_e = v > 0.5, e > 0.5
    return _label([ok & ~hi_v & hi_e, ok & ~hi_v & ~hi_e, ok & hi_v & hi_e, ok & hi_v & ~hi_e],
                  ["calm trend", "calm range", "wild trend", "wild range"], df.index)


# ------------------------------------------------------------- drawdown ----
@_det("drawdown", "Drawdown state", {"depth": (10, 2, 50, 1, "Drawdown % that counts as 'in drawdown'.")},
      order=("near highs", "in drawdown"),
      help="Is the instrument within N% of its running high, or deeper in a drawdown?",
      cite="Daniel & Moskowitz (2016).")
def drawdown(df, depth=10, **_):
    c = df["close"]
    dd = c / c.cummax() - 1
    return _label([dd <= -depth / 100, dd.notna()], ["in drawdown", "near highs"], df.index)


# ----------------------------------------------------------------- HMM ----
@_det("hmm", "Hidden Markov model (filtered)",
      {"states": (2, 2, 3, 1, "Number of hidden states."),
       "fit_pct": (50, 20, 80, 5, "% of the history used to fit the model; it then runs forward unseen.")},
      order=("hmm low-vol", "hmm mid-vol", "hmm high-vol"),
      help="A statistical model that assumes the market switches between hidden states with different average "
           "return and volatility. Fitted on the first part of the history, then run forward one bar at a time "
           "(filtered probabilities, never smoothed). Labels before the fit cut-off are in-sample.",
      cite="Hamilton (1989) Econometrica; Ang & Bekaert (2002); Nystrup et al. (2018).")
def hmm(df, states=2, fit_pct=50, **_):
    probs = hmm_probs(df, int(states), float(fit_pct))
    if probs is None:
        return pd.Series(np.nan, index=df.index, dtype=object)
    names = ["hmm low-vol", "hmm high-vol"] if probs.shape[1] == 2 else ["hmm low-vol", "hmm mid-vol", "hmm high-vol"]
    ok = probs.notna().all(axis=1)
    lab = pd.Series(np.nan, index=df.index, dtype=object)
    lab[ok] = probs[ok].idxmax(axis=1).map(dict(enumerate(names)))
    return lab


def hmm_probs(df: pd.DataFrame, k: int = 2, fit_pct: float = 50.0) -> pd.DataFrame | None:
    """Filtered state probabilities (columns ordered low → high variance), or None if it cannot fit."""
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except Exception:  # noqa: BLE001
        return None
    r = (np.log(df["close"]).diff() * 100).dropna()
    if len(r) < 300:
        return None
    cut = int(len(r) * fit_pct / 100)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = MarkovRegression(r.iloc[:cut], k_regimes=k, trend="c", switching_variance=True).fit(
                disp=False, maxiter=200)
            full = MarkovRegression(r, k_regimes=k, trend="c", switching_variance=True)
            filt = full.filter(fit.params)
        p = pd.DataFrame(np.asarray(filt.filtered_marginal_probabilities), index=r.index)
        sig = [float(fit.params[f"sigma2[{i}]"]) for i in range(k)]
    except Exception:  # noqa: BLE001
        return None
    order = list(np.argsort(sig))
    p = p[order]
    p.columns = range(k)
    return p.reindex(df.index)


# ----------------------------------------------------------- macro (monthly) ----
@_det("macro", "Macro quadrant (REGIMES page)", {},
      order=("Goldilocks / Reflation", "Overheating", "Stagflation", "Deflation / Slowdown"),
      help="The monthly growth × inflation regime from Zenith's REGIMES page. Each month's label is applied from "
           "the first trading day after that month ends (it is only known then). US macro, so best for US assets.",
      cite="Zenith REGIMES; Ilmanen, Maloney & Ross (2014) — asset returns across growth/inflation regimes.")
def macro(df, **_):
    tl = macro_timeline()
    if tl is None or tl.empty:
        return pd.Series(np.nan, index=df.index, dtype=object)
    avail = tl.copy()
    avail.index = avail.index + pd.Timedelta(days=1)          # known only after month end
    s = avail.reindex(avail.index.union(df.index)).ffill().reindex(df.index)
    return s.astype(object).where(s.notna(), np.nan)


def macro_timeline() -> pd.Series | None:
    try:
        import json
        from ...config import REGIMES_FILES
        d = json.loads(REGIMES_FILES["timeline"].read_text(encoding="utf-8"))
        rows = [(m["month"], m.get("declared_regime")) for m in d.get("months", [])]
        s = pd.Series({pd.Timestamp(k): v for k, v in rows if v})
        return s.sort_index()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------- API ----
def detect(key: str, df: pd.DataFrame, min_bars: int = 1, **params) -> pd.Series:
    """Labels for detector ``key`` on ``df`` (needs 'close'; some need high/low or aux columns)."""
    d = DETECTORS[key]
    p = {k: v[0] for k, v in d.params.items()}
    p.update({k: v for k, v in params.items() if k in d.params})
    lab = d.fn(df, **p)
    return debounce(lab, min_bars) if min_bars > 1 else lab

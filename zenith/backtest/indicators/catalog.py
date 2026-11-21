"""Indicator catalogue: one declarative entry per indicator.

Each entry knows how to compute itself from an OHLCV frame (strictly causal:
the value at bar t uses bars <= t only), how to explain itself to someone with
no finance background (``hover`` for the info badge, ``how`` for the guide),
where the evidence for it comes from, and how it is usually turned into a
trading signal (``signal``) so the LAB can backtest it with no code.

Signal styles (see ``rules/indicator_rule.py``):
  - ``cross``   price above/below the indicator line → long/short, sized by the
                distance in ATR units (trend lines, bands' midline, VWAP …)
  - ``level``   the indicator is already a signed signal (MACD histogram,
                TSMOM return, slope t-stat …) — use it as-is
  - ``band``    an oscillator with oversold/overbought levels (RSI, %B …):
                below ``lo`` → long, above ``hi`` → short (mean reversion by
                default; flip ``side`` to trade the breakout instead)
  - ``zscore``  z-score of the indicator over its own recent past
  - ``state``   a context variable (volatility, calendar, regime): not a
                direction on its own — the LAB fades/follows it via ``side``
                and the explorer / regime tools use it to *condition* others.

Extra frame columns some indicators read (joined by ``run_spec`` from the aux
data the builder loads): ``ref_close`` (reference ticker), ``vix``, ``vix3m``.
Missing columns give an all-NaN indicator, which the rule turns into a flat
forecast and the LAB flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

FAMILIES = {
    "trend": "TREND",
    "momentum": "MOMENTUM / OSCILLATORS",
    "meanrev": "MEAN REVERSION",
    "volatility": "VOLATILITY & MARKET STATE",
    "volume": "VOLUME",
    "calendar": "CALENDAR",
    "cross": "CROSS-ASSET",
    "structure": "PRICE STRUCTURE",
}

SIGNALS = ("cross", "level", "band", "zscore", "state")


@dataclass
class Indicator:
    key: str
    label: str
    family: str
    compute: Callable[..., pd.Series]
    params: dict = field(default_factory=dict)      # name -> (default, lo, hi, step, help)
    signal: str = "level"
    overlay: bool = False                           # plotted on the price axis
    lo: float | None = None                         # band signal: oversold level
    hi: float | None = None                         # band signal: overbought level
    hover: str = ""                                 # one or two plain sentences (info badge)
    how: str = ""                                   # how it is calculated, in words
    use: str = ""                                   # how traders use it
    evidence: str = ""                              # citation(s)
    video: str = ""                                 # SharpeEdge / MemLabs video it relates to
    needs: tuple = ()                               # extra frame columns required

    def defaults(self) -> dict:
        return {k: v[0] for k, v in self.params.items()}

    def __call__(self, df: pd.DataFrame, **kw) -> pd.Series:
        p = {**self.defaults(), **{k: v for k, v in kw.items() if k in self.params}}
        if any(c not in df.columns for c in self.needs):
            return pd.Series(float("nan"), index=df.index)
        out = self.compute(df, **p)
        return out.astype(float).reindex(df.index)


CATALOG: dict[str, Indicator] = {}


def ind(key: str, label: str, family: str, params: dict | None = None, **meta):
    """Decorator: register ``fn(df, **params) -> Series`` as an Indicator."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")

    def wrap(fn):
        if key in CATALOG:
            raise ValueError(f"duplicate indicator {key!r}")
        if meta.get("signal", "level") not in SIGNALS:
            raise ValueError(f"{key}: bad signal style")
        CATALOG[key] = Indicator(key=key, label=label, family=family, compute=fn, params=params or {}, **meta)
        return fn

    return wrap


def by_family() -> dict[str, list[Indicator]]:
    out: dict[str, list[Indicator]] = {f: [] for f in FAMILIES}
    for i in CATALOG.values():
        out[i.family].append(i)
    return out


# ---------------------------------------------------------- shared maths ----
def ema(s: pd.Series, n: float) -> pd.Series:
    return s.ewm(span=max(1.0, float(n)), adjust=False, min_periods=max(1, int(n))).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / max(1, int(n)), adjust=False, min_periods=int(n)).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def zscore(s: pd.Series, n: int) -> pd.Series:
    n = int(n)
    sd = s.rolling(n, min_periods=n).std()
    return (s - s.rolling(n, min_periods=n).mean()) / sd.where(sd > 1e-12)


def log_ret(df: pd.DataFrame) -> pd.Series:
    import numpy as np
    return np.log(df["close"]).diff()

"""Factor Rotation Momentum (FRM) model — the CAS 'factor_rotation' segment.

Harvests trend + rotation across equity **style factors**, **industries/sub-
industries**, and the (sparse) **region-specific sector** ETFs defined in
``universe.frm_universe()`` — every name a US-listed ETF so the free yfinance
pull works unchanged.

Motivated by Gupta & Kelly (2019) "Factor Momentum Everywhere" (time-series
factor momentum: a factor's own recent return predicts its forward return) and
the Man Group style-trend strategy (cross-sectional rotation). Per ETF we emit:

  * ``frm_ts_mom``    — time-series momentum: vol-scaled blend of 1/3/6/12m
                        trailing returns (12m uses the classic 12-1 skip).
  * ``frm_cs_region`` — cross-sectional rank within the ETF's peer group *in its
                        region* (styles vs styles in US, industries vs industries…).
  * ``frm_cs_peer``   — cross-sectional rank of the ETF vs the same label *across
                        regions* (e.g. Value US vs Value Intl vs Value EM).
  * ``frm_composite`` — 0.6·TS + 0.3·CS-region + 0.1·CS-peer.

All signals are in [-1, 1] via schema.make_signal, segment ``factor_rotation``,
horizon ``months`` — so the overlap/consensus layers pick them up like any other
family. Maths reuse zenith.cas.signals.indicators.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal
from .. import universe as univ

SRC = "yfinance"

# (lookback days, skip days, weight) — longer horizons weighted more heavily;
# the 12m leg uses a 1m skip (the classic 12-1 momentum convention).
_TS_LOOKBACKS = ((21, 0, 0.15), (63, 0, 0.25), (126, 0, 0.25), (252, 21, 0.35))
_TS_SCALE = 0.6          # tanh slope on the blended risk-adjusted momentum
_MIN_BARS = 60           # need at least this much history to score a series


def _ts_momentum(close: pd.Series) -> float:
    """Vol-scaled time-series momentum: blend of trailing returns, each divided
    by its horizon's expected volatility (annualised realized vol, de-annualised
    to the horizon)."""
    vol = ind.realized_vol(close, 63).iloc[-1]      # annualised
    if pd.isna(vol) or vol <= 0:
        vol = 0.10                                   # floor so we never divide by ~0
    parts: list[float] = []
    wsum = 0.0
    for lb, skip, w in _TS_LOOKBACKS:
        if len(close) <= lb + 1:
            continue
        end = close.iloc[-1 - skip]
        start = close.iloc[-lb - 1]
        if start <= 0:
            continue
        r = end / start - 1.0
        n = lb - skip
        horizon_vol = vol * np.sqrt(max(n, 1) / 252.0)
        parts.append(w * (r / (horizon_vol + 1e-9)))
        wsum += w
    if not parts or wsum == 0:
        return 0.0
    blended = sum(parts) / wsum
    return ind.clip1(np.tanh(_TS_SCALE * blended))


def _trailing_return(close: pd.Series, lb: int = 252, skip: int = 21) -> float:
    """12-1m trailing return used for the cross-sectional ranking."""
    if len(close) < lb + 2:
        return float("nan")
    start = close.iloc[-lb - 1]
    if start <= 0:
        return float("nan")
    return float(close.iloc[-1 - skip] / start - 1.0)


def _cross_section(tr: dict[str, float], key_fn) -> dict[str, float]:
    """Percentile-rank trailing returns within peer groups, mapped to [-1, 1].

    ``key_fn(ticker) -> group key``; tickers sharing a key are ranked against each
    other. Groups with <2 valid members yield 0 (no meaningful cross-section)."""
    groups: dict[object, list[str]] = defaultdict(list)
    for t, v in tr.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        key = key_fn(t)
        if key is not None:
            groups[key].append(t)

    out: dict[str, float] = {}
    for members in groups.values():
        if len(members) < 2:
            for t in members:
                out[t] = 0.0
            continue
        ranks = pd.Series({t: tr[t] for t in members}).rank(pct=True)
        for t in members:
            out[t] = ind.clip1((float(ranks[t]) - 0.5) * 2.0)
    return out


def compute(prices: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    """Run the factor-rotation model over the tagged FRM universe.

    If ``prices`` is None the module pulls its own 5y daily history for just the
    FRM tickers (cheap, ~120 names) so the 12m look-back is robust. Tests pass a
    synthetic ``prices`` dict to stay network-free.
    """
    uni = univ.frm_universe()
    tickers = list(uni)
    if prices is None:
        from ..sources import prices as price_src
        data, _ = price_src.get_history(tickers, period="5y")
    else:
        data = prices

    closes: dict[str, pd.Series] = {}
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < _MIN_BARS or "close" not in df:
            continue
        closes[t] = df["close"]

    ts = {t: _ts_momentum(c) for t, c in closes.items()}
    tr = {t: _trailing_return(c) for t, c in closes.items()}
    cs_region = _cross_section(tr, lambda t: (uni[t]["group"], uni[t]["region"]))
    cs_peer = _cross_section(tr, lambda t: (uni[t]["group"], uni[t]["label"]))

    out: list[dict] = []
    for t in closes:
        tag = uni[t]
        group, label, region = tag["group"], tag["label"], tag["region"]
        rlabel = univ.REGION_LABEL.get(region, region)
        tsig = ts[t]
        csr = cs_region.get(t, 0.0)
        csp = cs_peer.get(t, 0.0)
        comp = ind.clip1(0.6 * tsig + 0.3 * csr + 0.1 * csp)
        base = dict(asset_class=group, horizon="months", source=SRC)
        out.append(make_signal(
            t, "factor_rotation", "frm_ts_mom", tsig,
            rationale=f"{label}/{rlabel} time-series momentum (vol-scaled 1/3/6/12m)", **base))
        out.append(make_signal(
            t, "factor_rotation", "frm_cs_region", csr,
            rationale=f"{label}/{rlabel} cross-sectional rank within {group} peers in {rlabel}", **base))
        out.append(make_signal(
            t, "factor_rotation", "frm_cs_peer", csp,
            rationale=f"{label}/{rlabel} cross-sectional rank vs same label across regions", **base))
        out.append(make_signal(
            t, "factor_rotation", "frm_composite", comp,
            rationale=f"{label}/{rlabel} composite = 0.6·TS + 0.3·CS(region) + 0.1·CS(peer)", **base))
    return out

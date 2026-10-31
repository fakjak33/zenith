"""Implied volatility from mid quotes, against the put-call-parity forward.

Why recompute instead of trusting the feed's IV: the parity forward
F = K + e^{rT}(C - P) folds dividends and borrow into the forward, so every
expiry's smile is centred on the market's own forward and the OTM IVs from
both wings join smoothly at k = ln(K/F) = 0. Black-76 on F:
    C = e^{-rT} [F N(d1) - K N(d2)],  d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T)
which is pricing.price() with S = F e^{-rT}. The solve is a vectorised
bisection (monotone in sigma, never diverges) finished by two Newton steps.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from ..theta.engine import pricing
from .cboe import AM_SETTLED

YEAR_S = 365.0 * 86400.0
MIN_T = 1.0 / (365.0 * 24.0)            # one hour: never divide by a zero tenor
IV_LO, IV_HI = 0.005, 5.0


def year_frac(qtime: datetime, expiry: pd.Series, roots: pd.Series) -> np.ndarray:
    """Years from the quote to settlement: 09:30 ET for AM-settled monthlies
    (SPX, NDX, RUT, VIX roots), 16:00 ET otherwise."""
    exp = pd.to_datetime(expiry)
    am = roots.isin(AM_SETTLED).to_numpy()
    settle = exp + pd.to_timedelta(np.where(am, 9.5, 16.0), unit="h")
    secs = (settle - pd.Timestamp(qtime)).dt.total_seconds().to_numpy(float)
    return np.maximum(secs / YEAR_S, MIN_T)


def implied_vol(call, F, K, T, px, r: float = 0.0, iters: int = 64) -> np.ndarray:
    call = np.asarray(call, dtype=bool)
    F, K, T, px = (np.asarray(x, dtype=float) for x in (F, K, T, px))
    disc = np.exp(-r * T)
    S = F * disc
    intrinsic = np.where(call, np.maximum(S - K * disc, 0.0), np.maximum(K * disc - S, 0.0))
    upper = np.where(call, S, K * disc)
    ok = np.isfinite(px) & (px > intrinsic + 1e-10) & (px < upper) & (T > 0) & (K > 0) & (F > 0)
    lo = np.full(px.shape, IV_LO)
    hi = np.full(px.shape, IV_HI)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = pricing.price(call, S, K, T, mid, r) > px
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    sig = 0.5 * (lo + hi)
    for _ in range(2):                                  # polish
        vega = pricing.greeks(call, S, K, T, sig, r)["vega"] * 100.0
        diff = pricing.price(call, S, K, T, sig, r) - px
        step = np.where(vega > 1e-8, diff / np.maximum(vega, 1e-8), 0.0)
        sig = np.clip(sig - np.clip(step, -0.05, 0.05), IV_LO, IV_HI)
    sig = np.where(ok, sig, np.nan)
    # pinned at the bracket edge = no solution inside it
    return np.where((sig <= IV_LO * 1.001) | (sig >= IV_HI * 0.999), np.nan, sig)


def bs_delta(call, F, K, T, sig, r: float = 0.0) -> np.ndarray:
    """Spot delta of the forward-based model (dividend-consistent)."""
    S = np.asarray(F, float) * np.exp(-r * np.asarray(T, float))
    return pricing.greeks(call, S, K, T, sig, r)["delta"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Mid, relative spread and a usable-quote flag. Never drops rows."""
    out = df.copy()
    bid, ask = out["bid"].fillna(0.0), out["ask"].fillna(0.0)
    out["mid"] = np.where((bid > 0) & (ask >= bid), 0.5 * (bid + ask), np.nan)
    out["spread_pct"] = np.where(out["mid"] > 0, (ask - bid) / out["mid"], np.nan)
    out["quoted"] = np.isfinite(out["mid"]) & (out["spread_pct"] < 1.5)
    return out


def implied_forward(g: pd.DataFrame, spot: float, T: float, r: float, n: int = 6) -> float:
    """Median parity forward over the n strikes nearest spot that have both
    a call and a put mid; spot * e^{rT} when no pair is quoted."""
    q = g[g["quoted"]]
    piv = q.pivot_table(index="strike", columns="cp", values="mid", aggfunc="first")
    if "C" not in piv or "P" not in piv:
        return spot * math.exp(r * T)
    piv = piv.dropna()
    if piv.empty:
        return spot * math.exp(r * T)
    piv = piv.iloc[np.argsort(np.abs(piv.index.to_numpy() - spot))[:n]]
    fwd = piv.index.to_numpy() + math.exp(r * T) * (piv["C"].to_numpy() - piv["P"].to_numpy())
    f = float(np.median(fwd))
    # a broken pair can throw the forward far away: keep it within a sane band
    return f if 0.7 * spot < f < 1.3 * spot else spot * math.exp(r * T)


def price_chain(df: pd.DataFrame, spot: float, qtime: datetime, r: float) -> pd.DataFrame:
    """Add T, dte, F, k, otm flag, iv (from mid; the feed's iv as a fallback)
    and model delta. Rows with no usable vol keep iv = NaN."""
    if df.empty or not spot:
        return df.assign(T=[], dte=[], F=[], k=[], otm=[], iv_mid=[], iv_use=[], d=[])
    out = clean(df)
    out["T"] = year_frac(qtime, out["expiry"], out["root"])
    out["dte"] = (out["expiry"] - pd.Timestamp(qtime).normalize()).dt.days
    fwd = {}
    for exp, g in out.groupby("expiry"):
        fwd[exp] = implied_forward(g, spot, float(g["T"].iloc[0]), r)
    out["F"] = out["expiry"].map(fwd).astype(float)
    out["k"] = np.log(out["strike"] / out["F"])
    call = (out["cp"] == "C").to_numpy()
    out["otm"] = np.where(call, out["strike"] >= out["F"], out["strike"] < out["F"])
    out["iv_mid"] = implied_vol(call, out["F"], out["strike"], out["T"], out["mid"], r)
    feed = out["iv"].where(out["iv"] > 0.005)
    out["iv_use"] = out["iv_mid"].where(np.isfinite(out["iv_mid"]), feed)
    sig = out["iv_use"].fillna(0.3).to_numpy(float)
    out["d"] = bs_delta(call, out["F"], out["strike"], out["T"], sig, r)
    out.loc[~np.isfinite(out["iv_use"]), "d"] = np.nan
    return out

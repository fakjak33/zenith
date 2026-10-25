"""Black-Scholes prices and greeks, vectorised (numpy only, no scipy).

The normal CDF uses the Abramowitz & Stegun 7.1.26 erf approximation
(|error| < 1.5e-7), fast on arrays; the inverse comes from statistics.NormalDist.
Greeks are per share: theta per calendar day, vega per 1 vol point.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_INV = NormalDist().inv_cdf
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def ncdf(x):
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + _P * z)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1d2(S, K, T, sig, r):
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return d1, d1 - v


def price(call, S, K, T, sig, r=0.0):
    """Option price per share; T in years (<= 0 -> intrinsic). `call` may be an array."""
    call = np.asarray(call, dtype=bool)
    S, K, T, sig, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sig, r)))
    intrinsic = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    live = T > 1e-9
    Tl = np.where(live, T, 1.0)
    d1, d2 = _d1d2(S, K, Tl, np.maximum(sig, 1e-6), r)
    disc = np.exp(-r * Tl)
    c = S * ncdf(d1) - K * disc * ncdf(d2)
    p = K * disc * ncdf(-d2) - S * ncdf(-d1)
    return np.where(live, np.where(call, c, p), intrinsic)


def greeks(call, S, K, T, sig, r=0.0) -> dict:
    """{delta, gamma, theta, vega} per share (theta per calendar day, vega per vol point)."""
    call = np.asarray(call, dtype=bool)
    S, K, T, sig, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sig, r)))
    live = T > 1e-9
    Tl = np.where(live, T, 1.0)
    s = np.maximum(sig, 1e-6)
    d1, d2 = _d1d2(S, K, Tl, s, r)
    disc = np.exp(-r * Tl)
    pdf = npdf(d1)
    delta = np.where(call, ncdf(d1), ncdf(d1) - 1.0)
    gamma = pdf / (S * s * np.sqrt(Tl))
    common = -S * pdf * s / (2.0 * np.sqrt(Tl))
    theta_y = np.where(call, common - r * K * disc * ncdf(d2), common + r * K * disc * ncdf(-d2))
    vega = S * pdf * np.sqrt(Tl) / 100.0
    itm = np.where(call, S > K, S < K).astype(float)
    dead_delta = np.where(call, itm, -itm)
    return {"delta": np.where(live, delta, dead_delta),
            "gamma": np.where(live, gamma, 0.0),
            "theta": np.where(live, theta_y / 365.0, 0.0),
            "vega": np.where(live, vega, 0.0)}


def strike_for_delta(S: float, T: float, sig: float, r: float, delta: float, call: bool) -> float:
    """Strike whose BS delta is +delta (call) or -delta (put); closed form."""
    d1 = _INV(delta) if call else _INV(1.0 - delta)
    return S * math.exp(-d1 * sig * math.sqrt(T) + (r + 0.5 * sig * sig) * T)

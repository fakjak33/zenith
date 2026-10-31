"""GARCH(1,1) and GJR-GARCH(1,1,1) by Gaussian quasi-maximum likelihood, numpy only.

    sigma^2_t = omega + alpha r^2_{t-1} + gamma r^2_{t-1} 1[r_{t-1} < 0] + beta sigma^2_{t-1}
    (gamma = 0 is Bollerslev 1986; gamma > 0 is Glosten-Jagannathan-Runkle 1993,
     the "leverage" asymmetry that equity vol shows)

The repo carries no scipy, so the optimiser is a plain Nelder-Mead simplex.
It searches an UNCONSTRAINED reparametrisation that keeps every candidate
stationary and positive:
    persistence  rho = 0.9995 * sigmoid(a)          (alpha + beta + gamma/2 < 1)
    shares       softmax(b, g, 0) -> alpha, gamma/2, beta of rho
    long-run     sigma_bar^2 = exp(c),  omega = sigma_bar^2 (1 - rho)
The variance recursion is linear in its inputs, so it is evaluated as one
geometric convolution (C speed) instead of a Python loop.

FORECAST: the h-day average variance mean-reverts geometrically,
    E[avg var over h] = sigma_bar^2 + (sigma^2_{t+1} - sigma_bar^2) (1 - rho^h) / ((1 - rho) h)
which gives a full realized-vol TERM STRUCTURE to set against the implied one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SCALE = 100.0           # fit percent returns: better conditioned
TD = 252


def nelder_mead(f, x0, step: float = 0.5, maxiter: int = 600, tol: float = 1e-8):
    """Minimise f from x0 (adaptive coefficients, Gao & Han 2012). Returns (x, fx, iters)."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    a, g, c, s = 1.0, 1.0 + 2.0 / n, 0.75 - 1.0 / (2 * n), 1.0 - 1.0 / n
    simplex = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    fs = [f(x) for x in simplex]
    it = 0
    for it in range(maxiter):
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        if abs(fs[-1] - fs[0]) <= tol * (1 + abs(fs[0])) and \
                max(np.max(np.abs(x - simplex[0])) for x in simplex[1:]) < 1e-6:
            break
        cen = np.mean(simplex[:-1], axis=0)
        xr = cen + a * (cen - simplex[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = cen + g * (xr - cen)
            fe = f(xe)
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            if fr < fs[-1]:
                xc = cen + c * (xr - cen)
            else:
                xc = cen + c * (simplex[-1] - cen)
            fc = f(xc)
            if fc < min(fr, fs[-1]):
                simplex[-1], fs[-1] = xc, fc
            else:
                best = simplex[0]
                simplex = [best] + [best + s * (x - best) for x in simplex[1:]]
                fs = [fs[0]] + [f(x) for x in simplex[1:]]
    i = int(np.argmin(fs))
    return simplex[i], float(fs[i]), it


@dataclass
class GarchFit:
    kind: str
    omega: float
    alpha: float
    gamma: float
    beta: float
    var_lr: float          # long-run daily variance (percent^2)
    var_next: float        # sigma^2_{t+1} (percent^2)
    ll: float
    n: int
    ok: bool
    theta: list

    @property
    def rho(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma

    def as_dict(self) -> dict:
        return asdict(self)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def unpack(theta, kind: str) -> tuple[float, float, float, float, float]:
    a, b, c = theta[0], theta[1], theta[2]
    rho = 0.9995 * _sig(a)
    if kind == "gjr":
        g = theta[3]
        e = np.exp(np.array([b, g, 0.0]) - max(b, g, 0.0))
        w = e / e.sum()
        alpha, gamma, beta = rho * w[0], 2.0 * rho * w[1], rho * w[2]
    else:
        alpha, gamma = rho * _sig(b), 0.0
        beta = rho - alpha
    var_lr = math.exp(max(-20.0, min(20.0, c)))
    return var_lr * (1.0 - rho), alpha, gamma, beta, var_lr


def variance_path(r: np.ndarray, omega, alpha, gamma, beta, s0: float) -> np.ndarray:
    """sigma^2_t for t = 0..n (the last entry is the one-step-ahead forecast)."""
    n = len(r)
    shock = alpha * r * r + gamma * r * r * (r < 0)
    x = np.empty(n + 1)
    x[0] = 0.0
    x[1:] = omega + shock                       # input arriving at t (from r_{t-1})
    kern = beta ** np.arange(n + 1)
    path = np.convolve(x, kern)[: n + 1] + s0 * kern
    path[0] = s0
    return np.maximum(path, 1e-10)


def nll(theta, r: np.ndarray, kind: str, s0: float) -> float:
    omega, alpha, gamma, beta, _ = unpack(theta, kind)
    v = variance_path(r, omega, alpha, gamma, beta, s0)[:-1]
    out = 0.5 * float(np.sum(np.log(v) + r * r / v))
    return out if np.isfinite(out) else 1e12


def _x0(r: np.ndarray, kind: str) -> list[float]:
    lv = math.log(max(float(np.var(r)), 1e-6))
    # rho ~ 0.97, alpha share ~ 0.08
    base = [math.log(0.97 / 0.0295), math.log(0.08 / 0.92), lv]
    return base + [0.0] if kind == "gjr" else base


def fit(returns, kind: str = "garch", x0=None, maxiter: int = 600, n_max: int = 1000) -> GarchFit:
    """Fit on (log) returns; the most recent n_max observations are used."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)][-n_max:] * SCALE
    n = len(r)
    if n < 150:
        return GarchFit(kind, float("nan"), float("nan"), 0.0, float("nan"), float("nan"),
                        float("nan"), float("nan"), n, False, [])
    r = r - r.mean()
    s0 = float(np.var(r[:20])) if n > 40 else float(np.var(r))
    start = list(x0) if x0 is not None and len(x0) == (4 if kind == "gjr" else 3) else _x0(r, kind)
    theta, f, _ = nelder_mead(lambda t: nll(t, r, kind, s0), start, maxiter=maxiter)
    omega, alpha, gamma, beta, var_lr = unpack(theta, kind)
    v = variance_path(r, omega, alpha, gamma, beta, s0)
    ok = bool(np.isfinite(f) and alpha + beta + 0.5 * gamma < 0.9995 and var_lr > 0)
    return GarchFit(kind, omega, alpha, gamma, beta, var_lr, float(v[-1]), -f, n, ok,
                    [float(t) for t in theta])


def term_vol(fit_: GarchFit, h_days: int) -> float:
    """Annualised average vol over the next h trading days."""
    if not fit_.ok or h_days < 1:
        return float("nan")
    rho = min(fit_.rho, 0.9995)
    lr, nxt = fit_.var_lr, fit_.var_next
    if rho >= 0.9995 - 1e-9:
        avg = nxt
    else:
        avg = lr + (nxt - lr) * (1.0 - rho ** h_days) / ((1.0 - rho) * h_days)
    return math.sqrt(max(avg, 1e-10) * TD) / SCALE


def term_structure(fit_: GarchFit, tenors_cal) -> dict:
    """{calendar_days: annualised forecast vol} (calendar -> trading days x252/365)."""
    return {int(t): term_vol(fit_, max(1, round(t * TD / 365))) for t in tenors_cal}


def simulate(n: int, omega: float, alpha: float, beta: float, gamma: float = 0.0,
             seed: int = 0) -> np.ndarray:
    """Synthetic GARCH/GJR returns (percent units) for tests."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    v = omega / max(1e-9, 1 - alpha - beta - 0.5 * gamma)
    for t in range(n):
        r[t] = math.sqrt(v) * z[t]
        v = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * v
    return r

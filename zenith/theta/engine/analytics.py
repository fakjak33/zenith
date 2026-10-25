"""Trade analytics for the ticket: payoff curves, POP, max P/L, breakevens.

POP integrates the expiry P&L over a lognormal terminal price with zero drift
and the ATM implied vol: the probability-of-profit the trading platforms quote.
OPTIONS VAULT notes that realized vol has averaged ~0.78x implied, so the
model-world POP is conservative for short premium on average -- and wrong
exactly in the tails that matter.
"""

from __future__ import annotations

import numpy as np

from ..market import Market
from . import pricing
from .strategies import Leg


def _value(m: Market, t: str, i: int, legs: list[Leg], S: np.ndarray, days_fwd: int,
           flat: bool) -> np.ndarray:
    """Position value on a price grid `days_fwd` calendar days ahead, IV held fixed."""
    out = np.zeros_like(S)
    for lg in legs:
        if lg.kind == "stock":
            out += lg.qty * S
            continue
        T = max(m.days_between(i, lg.expiry_i) - days_fwd, 0) / 365.0
        sig = float(m.iv(t, i, lg.strike, lg.expiry_i, flat))
        out += lg.qty * 100.0 * pricing.price(lg.kind == "call", S, lg.strike, T, sig, m.rate[i])
    return out


def analyse(m: Market, t: str, i: int, legs: list[Leg], cost: float, flat: bool = False,
            n: int = 241) -> dict:
    """cost = cash paid to open (negative for a credit). Returns curves + stats."""
    S0 = m.spot(t, i)
    exp = min(lg.expiry_i for lg in legs if lg.kind != "stock") if any(
        lg.kind != "stock" for lg in legs) else i + 21
    dte = max(m.days_between(i, exp), 1)
    atm = float(m.iv(t, i, S0, exp, flat))
    sd = atm * np.sqrt(dte / 365.0)
    lo, hi = S0 * np.exp(-3.2 * sd), S0 * np.exp(3.2 * sd)
    grid = np.linspace(lo, hi, n)
    expiry = _value(m, t, i, legs, grid, dte, flat) - cost
    now = _value(m, t, i, legs, grid, 0, flat) - cost
    half = _value(m, t, i, legs, grid, dte // 2, flat) - cost

    # lognormal terminal density, zero drift
    z = (np.log(grid / S0) + 0.5 * sd * sd) / sd
    dens = pricing.npdf(z) / (grid * sd)
    w = dens * np.gradient(grid)
    w = w / w.sum()
    pop = float(w[expiry > 0].sum())
    ev = float((w * expiry).sum())
    sign = np.sign(expiry)
    cross = np.where(np.diff(sign) != 0)[0]
    bes = [float(grid[k] - expiry[k] * (grid[k + 1] - grid[k]) / (expiry[k + 1] - expiry[k]))
           for k in cross]
    # true extremes at expiry (including far tails) for undefined-risk honesty
    wide = np.linspace(0.01, S0 * 3.0, 601)
    exp_wide = _value(m, t, i, legs, wide, dte, flat) - cost
    unlimited_up = exp_wide[-1] < exp_wide[-2] - 1e-6 and exp_wide[-1] < 0
    return {"grid": grid, "expiry": expiry, "now": now, "half": half, "half_days": dte // 2,
            "pop": pop, "ev": ev, "breakevens": bes, "dte": dte, "atm_iv": atm,
            "max_profit": float(exp_wide.max()), "max_loss": float(exp_wide.min()),
            "unlimited_up": bool(unlimited_up), "em": S0 * sd}

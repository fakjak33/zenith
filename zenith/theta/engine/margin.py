"""Approximate Reg-T buying-power requirement (BPR) for a position, in dollars.

Equity/ETF option rules (OPTIONS VAULT portfolio.json, tastytrade help center):
  naked short put   max(20% of S - OTM amount, 10% of K) + premium, per share
  naked short call  max(20% of S - OTM amount, 10% of S) + premium, per share
  both sides naked  the larger side's requirement + the other side's premium
  defined risk      the maximum loss at expiry (width - credit)
  long stock        50% of its value (margin account)
Legs are "covered" when long options on the same side offset the shorts; any
uncovered remainder is charged as naked on its most dangerous short strike.
"""

from __future__ import annotations

import numpy as np

from .strategies import Leg

NAKED_PCT, NAKED_MIN = 0.20, 0.10
STOCK_PCT = 0.50


def _payoff(legs: list[Leg], ST: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ST)
    for lg in legs:
        if lg.kind == "stock":
            out += lg.qty * ST
        elif lg.kind == "call":
            out += lg.qty * 100.0 * np.maximum(ST - lg.strike, 0.0)
        else:
            out += lg.qty * 100.0 * np.maximum(lg.strike - ST, 0.0)
    return out


def uncovered(legs: list[Leg]) -> tuple[float, float]:
    """(uncovered short put contracts, uncovered short call contracts)."""
    puts = sum(lg.qty for lg in legs if lg.kind == "put")
    calls = sum(lg.qty for lg in legs if lg.kind == "call")
    shares = sum(lg.qty for lg in legs if lg.kind == "stock")
    return max(0.0, -puts), max(0.0, -(calls + shares / 100.0))


def bpr(legs: list[Leg], S: float, prices: dict[int, float]) -> float:
    """prices: leg index -> current option mid per share."""
    if not legs:
        return 0.0
    stock_val = sum(lg.qty * S for lg in legs if lg.kind == "stock")
    opts = [(k, lg) for k, lg in enumerate(legs) if lg.kind != "stock"]
    if not opts:
        return STOCK_PCT * abs(stock_val)
    up, uc = uncovered(legs)
    put_prem = sum(-lg.qty * 100.0 * prices[k] for k, lg in opts if lg.kind == "put" and lg.qty < 0)
    call_prem = sum(-lg.qty * 100.0 * prices[k] for k, lg in opts if lg.kind == "call" and lg.qty < 0)

    req_put = req_call = 0.0
    if up > 0:
        K = max(lg.strike for _, lg in opts if lg.kind == "put" and lg.qty < 0)
        per = max(NAKED_PCT * S - max(S - K, 0.0), NAKED_MIN * K) * 100.0
        req_put = per * up + put_prem
    if uc > 0:
        K = min(lg.strike for _, lg in opts if lg.kind == "call" and lg.qty < 0)
        per = max(NAKED_PCT * S - max(K - S, 0.0), NAKED_MIN * S) * 100.0
        req_call = per * uc + call_prem

    stock_req = STOCK_PCT * abs(stock_val)
    if up == 0 and uc == 0:
        # defined risk (incl. covered calls): worst expiry loss vs today's value
        strikes = [lg.strike for _, lg in opts]
        grid = np.unique(np.r_[0.0, np.array(strikes), max(strikes + [S]) * 3.0,
                               np.linspace(0.0, max(strikes + [S]) * 3.0, 61)])
        value_now = stock_val + sum(lg.qty * 100.0 * prices[k] for k, lg in opts)
        if stock_val:
            return stock_req                 # covered call: the short call is covered
        worst = float(np.min(_payoff(legs, grid)))
        return max(0.0, value_now - worst)
    if up > 0 and uc > 0:
        return max(req_put + call_prem, req_call + put_prem) + stock_req
    return (req_put + call_prem if up > 0 else req_call + put_prem) + stock_req

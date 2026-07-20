"""IV-spread fetcher (Cremers-Weinbaum) — the only network access for that screen.

For each ticker: nearest standard expiry >= MIN_DTE days out, pull the option
chain, match call/put strikes within +/-NEAR of spot, and compute the
open-interest-weighted mean of (call IV - put IV). Cloned from brief.sources.gex
(per-ticker try/except, tk.option_chain). Cached per day (option data is slow).
"""

from __future__ import annotations

import time
from datetime import date, datetime

from ..cas import store_cas

_KEY = "edge_ivspread"
MIN_DTE = 7             # skip expiries inside a week (noisy, pinning)
NEAR = 0.05            # strikes within +/-5% of spot count as "near the money"
MIN_PAIRS = 3          # need at least this many matched strikes


def _nearest_expiry(expiries: list[str]) -> str | None:
    today = date.today()
    good = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        if (d - today).days >= MIN_DTE:
            good.append((d, e))
    return min(good)[1] if good else None


def _spread_for(tk, spot: float) -> dict | None:
    try:
        expiries = list(tk.options or [])
    except Exception:
        return None
    exp = _nearest_expiry(expiries)
    if not exp:
        return None
    try:
        ch = tk.option_chain(exp)
    except Exception:
        return None
    lo, hi = spot * (1 - NEAR), spot * (1 + NEAR)
    calls = {round(float(r["strike"]), 2): r for _, r in ch.calls.iterrows()
             if lo <= float(r["strike"]) <= hi}
    puts = {round(float(r["strike"]), 2): r for _, r in ch.puts.iterrows()
            if lo <= float(r["strike"]) <= hi}
    num = den = 0.0
    n = 0
    for k, c in calls.items():
        p = puts.get(k)
        if p is None:
            continue
        civ, piv = c.get("impliedVolatility"), p.get("impliedVolatility")
        if civ is None or piv is None or civ <= 0 or piv <= 0:
            continue
        w = float((c.get("openInterest") or 0) + (p.get("openInterest") or 0)) + 1.0
        num += w * (float(civ) - float(piv))
        den += w
        n += 1
    if n < MIN_PAIRS or den <= 0:
        return None
    return {"iv_spread": round(num / den, 6), "n_pairs": n, "expiry": exp}


def iv_spread_rows(tickers: list[str], spots: dict[str, float],
                   sleep: float = 0.1, max_age_hours: float = 20.0) -> dict[str, dict]:
    """ticker -> {iv_spread, n_pairs, expiry, spot}. Cached ~1 day."""
    cached = store_cas.cache_get(_KEY, max_age_hours) or {}
    todo = [t for t in tickers if t not in cached and t in spots]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        for i, t in enumerate(todo):
            spot = spots.get(t)
            row = None
            if spot and spot > 0:
                try:
                    row = _spread_for(yf.Ticker(t), spot)
                except Exception:
                    row = None
            cached[t] = {**row, "spot": round(spot, 2)} if row else {}
            if sleep:
                time.sleep(sleep)
            if i and i % 100 == 0:
                store_cas.cache_put(_KEY, cached)
                print(f"[edge] iv-spread {i}/{len(todo)}")
        store_cas.cache_put(_KEY, cached)
    return {t: v for t, v in cached.items() if v.get("iv_spread") is not None}

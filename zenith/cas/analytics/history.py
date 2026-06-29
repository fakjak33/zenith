"""Signal-history tracking + predictive hit-rate for the CAS monitor.

Two products, both saved as JSON for the viewer:

  * ``history.json`` — a time series of each asset's signal, so the UI can chart
    how a signal has evolved. Built from (a) the per-day signal archives that
    accrue every CAS run, plus (b) an immediate backfill of the Factor-Rotation
    time-series-momentum signal (deterministic from price, so we can recompute it
    at past month-ends without waiting for snapshots).

  * ``hitrate.json`` — how often the Factor-Rotation signal's direction matched
    the asset's ACTUAL forward return at 1 / 3 / 6 / 12 months. 50% = coin-flip;
    above 50% = the signal has had predictive value. Backfilled from price history
    so it's meaningful immediately. Past hit-rate is not a guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from .. import store_cas
from .. import universe as univ
from ..signals import factor_rotation as frm

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
_STEP = 21          # evaluate monthly
_MIN_FORM = 253     # bars of history needed before the first signal point
_BAND = 0.05        # only score directional calls (ignore ~neutral)


def _series_hits(close: pd.Series) -> dict[str, list[int]]:
    """For one price series, walk monthly: recompute the FRM time-series signal on
    history-to-date and check its sign vs the forward return at each horizon.
    Returns {horizon: [hits, total]}."""
    res = {h: [0, 0] for h in HORIZONS}
    n = len(close)
    for t in range(_MIN_FORM, n, _STEP):
        sig = frm._ts_momentum(close.iloc[:t + 1])
        if abs(sig) < _BAND:
            continue
        for h, days in HORIZONS.items():
            if t + days < n:
                base = close.iloc[t]
                if base <= 0:
                    continue
                fwd = close.iloc[t + days] / base - 1.0
                if fwd == 0:
                    continue
                res[h][1] += 1
                if (sig > 0) == (fwd > 0):
                    res[h][0] += 1
    return res


def _signal_path(close: pd.Series) -> list[dict]:
    """Backfilled monthly FRM time-series-momentum signal for one series."""
    out = []
    n = len(close)
    for t in range(_MIN_FORM, n, _STEP):
        out.append({"date": close.index[t].date().isoformat(),
                    "signal": round(frm._ts_momentum(close.iloc[:t + 1]), 4)})
    return out


def _frm_prices(prices: dict[str, pd.DataFrame] | None) -> dict[str, pd.Series]:
    tickers = univ.frm_tickers()
    if prices is None:
        from ..sources import prices as price_src
        prices, _ = price_src.get_history(tickers, period="5y")
    closes = {}
    for t in tickers:
        df = prices.get(t)
        if df is not None and "close" in df and len(df) >= _MIN_FORM:
            closes[t] = df["close"]
    return closes


def build_hitrate(prices: dict[str, pd.DataFrame] | None = None) -> dict:
    """Aggregate FRM directional hit-rate across the universe, overall and by group."""
    closes = _frm_prices(prices)
    overall = {h: [0, 0] for h in HORIZONS}
    by_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: {h: [0, 0] for h in HORIZONS})
    for t, close in closes.items():
        tag = univ.frm_tag(t) or {}
        grp = tag.get("group", "other")
        hits = _series_hits(close)
        for h in HORIZONS:
            overall[h][0] += hits[h][0]
            overall[h][1] += hits[h][1]
            by_group[grp][h][0] += hits[h][0]
            by_group[grp][h][1] += hits[h][1]

    def _fmt(d):
        return {h: {"hit_rate": round(v[0] / v[1], 3) if v[1] else None, "n": v[1]}
                for h, v in d.items()}

    return {
        "as_of": date.today().isoformat(),
        "model": "Factor Rotation Momentum (time-series)",
        "n_assets": len(closes),
        "by_horizon": _fmt(overall),
        "by_group": {g: _fmt(d) for g, d in by_group.items()},
        "note": "Backfilled from price history; directional calls only. Not investment advice.",
    }


def build_history(prices: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    """Long-form signal time series for charting. FRM time-series-momentum signal,
    backfilled monthly per asset, merged with any archived snapshot signals."""
    closes = _frm_prices(prices)
    out: list[dict] = []
    for t, close in closes.items():
        for pt in _signal_path(close):
            out.append({"date": pt["date"], "asset": t, "family": "frm_ts_mom",
                        "signal": pt["signal"]})

    # merge accrued snapshots (forward-accruing, any family) so live runs enrich it
    for day in store_cas.archive_dates():
        for s in store_cas.load_archive(day):
            if s.get("segment") == "factor_rotation" and s.get("family") == "frm_composite":
                out.append({"date": s.get("asof", day), "asset": s["asset"],
                            "family": "frm_composite", "signal": s["signal"]})
    return out


def run(prices: dict[str, pd.DataFrame] | None = None) -> dict:
    """Compute + persist history.json and hitrate.json."""
    closes = _frm_prices(prices)            # fetch once, reuse for both
    px = {t: pd.DataFrame({"close": c}) for t, c in closes.items()}
    hist = build_history(px)
    hr = build_hitrate(px)
    store_cas.save("history", hist)
    store_cas.save("hitrate", hr)
    return {"n_history": len(hist), "n_assets": hr["n_assets"]}

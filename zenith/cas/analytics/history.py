"""Signal-history tracking + predictive hit-rate for the CAS monitor.

Two products, both saved as JSON for the viewer and backfilled from price history
so they're meaningful immediately (no waiting for weekly snapshots):

  * ``history.json`` — a monthly time series of the Factor-Rotation signals
    (``frm_ts_mom`` = own-trend, ``frm_cs_region`` = cross-sectional rank,
    ``frm_composite`` = the blend) per asset, recomputed historically so the UI
    can chart how each evolved and overlay it on price.

  * ``hitrate.json`` — for several price-deterministic models, how often the
    signal's direction matched the asset's ACTUAL forward return at 1/3/6/12
    months. 50% = coin-flip; above 50% = predictive value. (COT/flows can't be
    backfilled and accrue forward only.) Past hit-rate is not a guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from .. import store_cas
from .. import universe as univ
from ..signals import factor_rotation as frm
from ..signals import strategies as strat
from ..signals.indicators import clip1

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
_STEP = 21          # evaluate monthly
_MIN_FORM = 253     # bars of history needed before the first signal point
_BAND = 0.05        # only score directional calls (ignore ~neutral)

# price-deterministic models we can backfill a hit-rate for: key -> (label, fn(close)->signal)
MODELS = {
    "frm_ts_mom": ("Factor-Rotation — time-series momentum", frm._ts_momentum),
    "strat_trend": ("Strategies — multi-horizon trend", strat._trend_following),
    "strat_momentum": ("Strategies — 6-month momentum", strat._momentum),
    "strat_ma_cross": ("Strategies — 50/200 MA cross", lambda c: strat._ma_cross(c, 50, 200)),
    "strat_ma_200": ("Strategies — price vs 200-day", lambda c: strat._ma_single(c, 200)),
}


def _series_hits(close: pd.Series, fn) -> dict[str, list[int]]:
    """Walk monthly: recompute ``fn`` on history-to-date, check its sign vs the
    forward return at each horizon. Returns {horizon: [hits, total]}."""
    res = {h: [0, 0] for h in HORIZONS}
    n = len(close)
    for t in range(_MIN_FORM, n, _STEP):
        try:
            sig = fn(close.iloc[:t + 1])
        except Exception:
            continue
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


def build_history(closes: dict[str, pd.Series]) -> list[dict]:
    """Historical monthly FRM signals (ts / cs_region / composite) per asset — the
    cross-section is recomputed at each month-end so cs/composite are faithful."""
    if not closes:
        return []
    uni = univ.frm_universe()
    panel = pd.DataFrame(closes).sort_index()
    idx = panel.index
    out: list[dict] = []
    for row in range(_MIN_FORM, len(idx), _STEP):
        d = idx[row].date().isoformat()
        sub = panel.iloc[:row + 1]
        ts, tr = {}, {}
        for t in panel.columns:
            c = sub[t].dropna()
            if len(c) < _MIN_FORM:
                continue
            ts[t] = frm._ts_momentum(c)
            tr[t] = frm._trailing_return(c)
        csr = frm._cross_section(tr, lambda t: (uni[t]["group"], uni[t]["region"]))
        csp = frm._cross_section(tr, lambda t: (uni[t]["group"], uni[t]["label"]))
        for t in ts:
            comp = clip1(0.6 * ts[t] + 0.3 * csr.get(t, 0.0) + 0.1 * csp.get(t, 0.0))
            out.append({"date": d, "asset": t, "family": "frm_ts_mom", "signal": round(ts[t], 4)})
            out.append({"date": d, "asset": t, "family": "frm_cs_region",
                        "signal": round(csr.get(t, 0.0), 4)})
            out.append({"date": d, "asset": t, "family": "frm_composite", "signal": round(comp, 4)})
    return out


def _agg(model_fn, closes: dict[str, pd.Series]) -> dict:
    overall = {h: [0, 0] for h in HORIZONS}
    by_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: {h: [0, 0] for h in HORIZONS})
    for t, close in closes.items():
        grp = (univ.frm_tag(t) or {}).get("group", "other")
        hits = _series_hits(close, model_fn)
        for h in HORIZONS:
            overall[h][0] += hits[h][0]
            overall[h][1] += hits[h][1]
            by_group[grp][h][0] += hits[h][0]
            by_group[grp][h][1] += hits[h][1]

    def _fmt(d):
        return {h: {"hit_rate": round(v[0] / v[1], 3) if v[1] else None, "n": v[1]}
                for h, v in d.items()}

    return {"by_horizon": _fmt(overall), "by_group": {g: _fmt(d) for g, d in by_group.items()}}


def build_hitrate(closes: dict[str, pd.Series]) -> dict:
    """Directional hit-rate for each backfillable model, overall and by group."""
    models = {}
    for key, (label, fn) in MODELS.items():
        agg = _agg(fn, closes)
        agg["label"] = label
        models[key] = agg
    return {
        "as_of": date.today().isoformat(),
        "n_assets": len(closes),
        "horizons": list(HORIZONS),
        "models": models,
        "note": "Backfilled from price history; directional calls only (|signal|>0.05). "
                "50% = coin-flip. COT/flows accrue forward-only. Not investment advice.",
    }


def run(prices: dict[str, pd.DataFrame] | None = None) -> dict:
    """Compute + persist history.json and hitrate.json."""
    closes = _frm_prices(prices)
    store_cas.save("history", build_history(closes))
    store_cas.save("hitrate", build_hitrate(closes))
    return {"n_assets": len(closes)}

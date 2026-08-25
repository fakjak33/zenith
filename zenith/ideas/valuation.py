"""IDEAS valuation: three honestly-separate percentile lenses.

Spec section 5 asks for valuation "vs its own history". There is no free
point-in-time fundamentals history (yfinance .info is a snapshot;
quarterly_income_stmt serves ~5 quarters -- PEAD already established this
limit for the whole repo). Rather than fabricate one blended "valuation
percentile", three lenses are computed, labelled, and shown separately --
never silently averaged into a number that looks more authoritative than it
is (config's data-honesty rule, spec section 29):

  * cross_sectional -- percentile of today's multiples vs the universe/sector/
    industry TODAY. Fully real, available immediately.
  * own_history      -- percentile vs this app's own accumulating monthly
    fundamentals archive (valuation_history.json). Empty on day one,
    meaningful after ~6-12 months of monthly snapshots. Reports its own `n`.
  * price_anchored   -- holds trailing EPS/sales/FCF fixed at today's value
    and walks the PRICE back 5y: "what would this multiple have been at each
    past price". This is a price percentile wearing a valuation costume, and
    it is labelled exactly that everywhere it is shown -- never presented as
    a true historical-fundamentals percentile.

Only cross_sectional and (once it has enough points) own_history feed the
conviction blend; price_anchored is shown for context, not scored, because it
mechanically just re-derives price momentum.
"""

from __future__ import annotations

import json
import math
from datetime import date

import pandas as pd

from ..edge.common import pct_ranks, finite
from . import load

_HIST_MIN_N = 8          # minimum own-history points before it is shown as a real percentile


def _safe_inv(x) -> float | None:
    try:
        x = float(x)
        return (1.0 / x) if x not in (0.0, None) and math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def multiples_of(fund_row: dict) -> dict:
    """The four yield-style multiples this module works with (higher = cheaper),
    from one fundamentals.py .info row. None where the underlying field is
    missing -- never fabricated."""
    mc = fund_row.get("marketCap")
    fcf = fund_row.get("freeCashflow")
    fcf_yield = (float(fcf) / float(mc)) if (finite(fcf) and finite(mc) and mc) else None
    return {
        "ep": _safe_inv(fund_row.get("trailingPE")),
        "book_yield": _safe_inv(fund_row.get("priceToBook")),
        "sales_yield": _safe_inv(fund_row.get("priceToSalesTrailing12Months")),
        "fcf_yield": fcf_yield,
    }


# ------------------------------------------------------------ cross-sectional
def cross_sectional(fund: dict[str, dict], sector_of: dict[str, str]) -> dict[str, dict]:
    """Batch percentile of each multiple vs the whole universe and vs sector,
    for every ticker with at least one computable multiple. Returns
    ticker -> {universe_pctile, sector_pctile, n_universe, n_sector,
    multiples}. Percentile is the average of the per-field percentile ranks
    that are actually available for that ticker (edge.common.pct_ranks,
    ties-averaged) -- a name missing FCF is ranked on the other three, not
    dragged toward the middle for it (mom.engine._weighted's same principle)."""
    rows = {t: multiples_of(r) for t, r in fund.items()}
    fields = ("ep", "book_yield", "sales_yield", "fcf_yield")

    def _pctile_map(universe: list[str]) -> dict[str, list[float]]:
        per_field_pct: dict[str, dict[str, float]] = {}
        for f in fields:
            vals, tks = [], []
            for t in universe:
                v = rows.get(t, {}).get(f)
                if finite(v):
                    vals.append(v)
                    tks.append(t)
            if len(vals) < 5:
                continue
            pr = pct_ranks(vals)
            per_field_pct[f] = dict(zip(tks, pr))
        out: dict[str, list[float]] = {}
        for t in universe:
            got = [per_field_pct[f][t] for f in fields if t in per_field_pct.get(f, {})]
            if got:
                out[t] = got
        return out

    uni_tickers = list(rows.keys())
    uni_pct = _pctile_map(uni_tickers)

    by_sector: dict[str, list[str]] = {}
    for t in uni_tickers:
        by_sector.setdefault(sector_of.get(t) or "Unknown", []).append(t)
    sec_pct: dict[str, float] = {}
    sec_n: dict[str, int] = {}
    for sec, tks in by_sector.items():
        m = _pctile_map(tks)
        for t, got in m.items():
            sec_pct[t] = round(sum(got) / len(got), 1)
            sec_n[t] = len(tks)

    out = {}
    for t in uni_tickers:
        if t not in uni_pct and t not in sec_pct:
            continue
        out[t] = {
            "universe_pctile": round(sum(uni_pct[t]) / len(uni_pct[t]), 1) if t in uni_pct else None,
            "n_universe": len(uni_tickers),
            "sector_pctile": sec_pct.get(t),
            "n_sector": sec_n.get(t),
            "multiples": rows.get(t, {}),
        }
    return out


# ------------------------------------------------------------------ own-history
def _history_path():
    from ..config import IDEAS_DIR
    return IDEAS_DIR / "valuation_history.json"


def _load_history() -> dict:
    p = _history_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_history(doc: dict) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=None, ensure_ascii=False), encoding="utf-8")


def append_monthly_snapshot(fund: dict[str, dict], today: date) -> int:
    """Append this month's multiples to the own-history archive, once per
    (ticker, YYYY-MM) -- idempotent, safe to call every run. Monthly cadence
    (not daily) keeps the archive small: ~1000 tickers x ~12 rows/yr x 4
    floats is well under 1MB/yr, matching the repo's committed-artifact
    size discipline."""
    month = today.strftime("%Y-%m")
    hist = _load_history()
    added = 0
    for t, row in fund.items():
        m = multiples_of(row)
        if not any(finite(v) for v in m.values()):
            continue
        series = hist.setdefault(t, [])
        if series and series[-1].get("month") == month:
            continue
        series.append({"month": month, **{k: (round(v, 6) if finite(v) else None) for k, v in m.items()}})
        added += 1
    if added:
        _save_history(hist)
    return added


def own_history_percentile(ticker: str, current: dict, hist: dict | None = None) -> dict:
    """Percentile of today's multiples vs this ticker's own accumulated
    monthly history. Returns {pctile, n, state}: state='building' below
    _HIST_MIN_N points (shown as such, never as a confident number)."""
    hist = hist if hist is not None else _load_history()
    series = hist.get(ticker, [])
    fields = ("ep", "book_yield", "sales_yield", "fcf_yield")
    pcts = []
    for f in fields:
        cur = current.get(f)
        if not finite(cur):
            continue
        past = [row[f] for row in series if finite(row.get(f))]
        if len(past) < _HIST_MIN_N:
            continue
        pcts.append(round(100.0 * sum(1 for p in past if p <= cur) / len(past), 1))
    n = max((len([r for r in series if finite(r.get(f))]) for f in fields), default=0)
    if not pcts:
        return {"pctile": None, "n": n, "state": "building"}
    return {"pctile": round(sum(pcts) / len(pcts), 1), "n": n, "state": "ready"}


# ---------------------------------------------------------------- price-anchored
def price_anchored_percentile(close: pd.Series, current_multiple_yield: float | None) -> dict:
    """Holds today's yield-style multiple's implied earnings/sales/fcf fixed
    and walks price back: since yield ~ 1/price when the fundamental is held
    constant, the historical "multiple" at each past price is simply
    current_multiple_yield * (price_today / price_past). Percentile of
    today's (=1.0x) ratio vs that walked-back distribution. This is
    mechanically a price percentile -- labelled as such wherever it is shown,
    never presented as a true historical-fundamentals read (see module
    docstring)."""
    if not finite(current_multiple_yield) or close is None or len(close.dropna()) < 260:
        return {"pctile": None, "n": 0, "state": "unavailable"}
    px = close.dropna().tail(1260)      # ~5y
    today_px = float(px.iloc[-1])
    if today_px <= 0:
        return {"pctile": None, "n": 0, "state": "unavailable"}
    implied_yield = current_multiple_yield * (today_px / px)     # today/price_past scaling
    pctile = round(100.0 * float((implied_yield <= current_multiple_yield).mean()), 1)
    return {"pctile": pctile, "n": int(len(px)), "state": "ready",
            "label": "price-anchored (today's fundamental held fixed, price walked back -- "
                     "a price percentile, not a true fundamentals history)"}


# --------------------------------------------------------------------- blend
def blend(cross: dict | None, own: dict | None) -> tuple[float, dict]:
    """[-1,1] valuation score from whichever lenses have real coverage.
    price_anchored NEVER enters this blend (see module docstring -- it is
    mechanically price, and would double-count with the technicals group)."""
    parts, weights = {}, {}
    if cross and finite(cross.get("universe_pctile")):
        parts["cross_sectional"] = (cross["universe_pctile"] - 50.0) / 50.0
        weights["cross_sectional"] = 0.65
    if cross and finite(cross.get("sector_pctile")):
        parts["sector"] = (cross["sector_pctile"] - 50.0) / 50.0
        weights["sector"] = 0.35
    if own and own.get("state") == "ready" and finite(own.get("pctile")):
        parts["own_history"] = (own["pctile"] - 50.0) / 50.0
        # own-history, once it exists, is treated as at least as informative
        # as the cross-sectional read and reweights toward it
        total_existing = sum(weights.values()) or 1.0
        weights = {k: v * 0.6 / total_existing for k, v in weights.items()}
        weights["own_history"] = 0.4
    if not parts:
        return 0.0, {"coverage": False, "parts": {}}
    tw = sum(weights.get(k, 0.0) for k in parts)
    score = sum(parts[k] * weights.get(k, 0.0) for k in parts) / tw if tw else 0.0
    return max(-1.0, min(1.0, score)), {"coverage": True, "parts": parts, "weights": weights}

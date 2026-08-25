"""IDEAS panel: the fusion layer.

Reads every artefact Zenith's other packages already commit and assembles one
per-security dict carrying everything downstream scoring needs, plus an
explicit `coverage` map of which signal groups actually have real data for
that security (spec section 29: never silently fabricate a missing metric --
a stock with no options data is renormalized over what IS available
downstream, never scored as if the missing piece were neutral).

Two security types, deliberately asymmetric in what is available for them:

  * stock (Russell 1000) -- the full eight-group panel: MOMENTUM technicals,
    EDGE sentiment/positioning/risk, PEAD catalysts, valuation/fundamentals
    from this package's own committed cache, macro/regime.
  * etf (CAS master ETF list, ~400 names -- cas.universe.master_etfs(), the
    union CAS itself scans: sector/theme/factor/macro core + the curated
    strategic-beta list + the full style/industry/region-sector grid) --
    technicals from CAS's Factor-Rotation composite signal (frm_composite)
    where the ETF is style/industry/beta-tagged; for the broad sector/theme/
    macro core that frm_composite does not cover (SPY, XLK, GLD, TLT, ...) a
    lighter price-only trend score is computed on demand (see
    `_etf_price_technical`), reusing mom.factors' own vol-adjusted-return and
    squash primitives rather than inventing a second methodology.
    EDGE, PEAD and per-security fundamentals are Russell-1000-only in this
    repo; rather than invent proxies, those groups are simply marked
    uncovered for ETFs (see groups.py, which renormalizes around this).

No network calls happen here for the STOCK path -- this module only reads
committed JSON (via each package's own `load()`). The ETF price-only trend
fallback needs one price pull for the ~400-name master ETF list; that fetch
is done by compute.py (mirroring mom/compute.py's own `_fetch_prices`) and
passed in via `etf_px`, keeping this module itself IO-free apart from that
single injected dict.
"""

from __future__ import annotations

import pandas as pd

from ..pretom import universe as pretom_universe
from ..cas import universe as cas_universe
from ..mom import load as mom_load
from ..mom import factors as mom_factors
from ..mom.normalize import tanh_clip
from ..edge import load as edge_load
from ..pead import load as pead_load
from ..cas import store_cas as cas_store
from . import fundamentals as ideas_fund
from . import valuation as val

# Horizons for the ETF price-only trend fallback -- deliberately the same
# medium/long-term tilt as MOM_HORIZON_WEIGHTS (12-1/6m/3m), not a day-trading
# read, matching the whole feature's stated investment horizon (spec section 3).
_ETF_TREND_HORIZONS = {"12_1": (252, 21), "6m": (126, 0), "3m": (63, 0)}
_ETF_TREND_WEIGHTS = {"12_1": 0.5, "6m": 0.3, "3m": 0.2}


# ------------------------------------------------------------------ universe
def build_universe() -> list[dict]:
    """Russell 1000 stocks + the CAS master ETF universe, one row each:
    {ticker, name, sector, security_type}. Thin pass-through to the two
    existing universe sources -- IDEAS does not maintain a third universe
    pipeline (mom.universe's own stated principle, reused verbatim here)."""
    rows = []
    r1000, _status = pretom_universe.russell1000()
    for u in r1000:
        rows.append({"ticker": u["ticker"], "name": u.get("name") or u["ticker"],
                     "sector": u.get("sector") or "", "security_type": "stock",
                     "weight_pct": u.get("weight_pct")})
    for tkr in cas_universe.master_etfs():
        tag = cas_universe.frm_tag(tkr)
        label = (tag.get("label") if tag else None) or cas_universe.asset_class_of(tkr)
        rows.append({"ticker": tkr, "name": cas_universe.label_of(tkr),
                     "sector": label, "security_type": "etf", "weight_pct": None})
    return rows


# --------------------------------------------------------------- source loads
def _index_by_ticker(rows: list[dict]) -> dict[str, dict]:
    return {r["ticker"]: r for r in rows if r.get("ticker")}


def load_momentum() -> tuple[dict[str, dict], dict[str, dict]]:
    """ticker -> mom scores row, ticker -> mom detail (speed/strength/last_close)."""
    scores = mom_load("scores", {})
    detail = mom_load("detail", {})
    return _index_by_ticker(scores.get("rows", [])), detail.get("stocks", {})


def load_edge() -> dict[str, dict[str, dict]]:
    """screen -> ticker -> row, for the four EDGE screens (ranked lists, so
    every priced ticker is present, not just the decile extremes)."""
    out = {}
    for screen in ("lottery", "shortint", "revisions", "ivspread"):
        doc = edge_load(screen, {})
        out[screen] = _index_by_ticker(doc.get("ranked", []))
    return out


def load_nightday() -> dict[str, dict]:
    from ..nightday import load as nightday_load
    doc = nightday_load("screen", {})
    return _index_by_ticker(doc.get("ranked", []))


def load_pead() -> tuple[dict[str, dict], dict[str, dict]]:
    """ticker -> most recent signal row (recent reporters only -- sparse),
    ticker -> upcoming scheduled announcement (the forward-looking catalyst)."""
    sig_doc = pead_load("signals", {})
    recent = _index_by_ticker(sig_doc.get("signals", []))
    eap_doc = pead_load("eap", {})
    upcoming = _index_by_ticker(eap_doc.get("upcoming", []))
    return recent, upcoming


def load_factor_rotation() -> dict[str, dict]:
    """ticker -> the frm_composite Factor-Rotation record (ETF technicals)."""
    rows = cas_store.load("factor_rotation", [])
    out = {}
    for r in rows:
        if r.get("family") == "frm_composite":
            out[r["asset"]] = r
    return out


def _etf_price_technical(df: pd.DataFrame) -> dict | None:
    """Price-only trend score for an ETF that frm_composite does not cover
    (the broad sector/theme/macro core -- SPY, XLK, GLD, TLT, ...). Reuses
    mom.factors.trailing_return / daily_vol (the SAME vol-adjusted-return
    primitive MOMENTUM uses for stocks) and mom.normalize.tanh_clip for the
    squash, so this fallback is methodologically the same read, just without
    the cross-sectional/breakout/GMMA machinery a whole-universe daily job
    would need. Returns None (never a fabricated 0) if there is not enough
    history."""
    if df is None or df.empty or len(df) < 260:
        return None
    close = df["close"]
    sigma = mom_factors.daily_vol(close, 252)
    if not sigma:
        return None
    parts = {}
    for name, (lb, skip) in _ETF_TREND_HORIZONS.items():
        r = mom_factors.trailing_return(close, lb, skip)
        if r is None:
            continue
        n_h = lb - skip
        m = r / (sigma * (n_h ** 0.5))
        parts[name] = tanh_clip(m, scale=2.0)
    if not parts:
        return None
    tw = sum(_ETF_TREND_WEIGHTS[k] for k in parts)
    score = sum(parts[k] * _ETF_TREND_WEIGHTS[k] for k in parts) / tw
    composite = round(20.0 * max(-1.0, min(1.0, score)), 2)
    return {"composite": composite, "state": None, "source": "ideas.panel(price_only_fallback)",
            "horizons": {k: round(v, 4) for k, v in parts.items()}}


# --------------------------------------------------------------------- panel
def build_panel(universe: list[dict] | None = None,
                etf_px: dict[str, pd.DataFrame] | None = None) -> dict[str, dict]:
    """The fused per-security panel. Returns ticker -> {meta, technicals,
    sentiment, positioning, valuation, fundamentals, catalyst, coverage}.
    `coverage[group]` is True only when real data was found for that
    security -- never inferred. `etf_px` (ticker -> OHLCV DataFrame, from
    compute.py) feeds the price-only trend fallback for ETFs frm_composite
    does not tag; omit it and those ETFs are simply left uncovered."""
    universe = universe if universe is not None else build_universe()
    tickers = [u["ticker"] for u in universe]
    etf_px = etf_px or {}

    mom_scores, mom_detail = load_momentum()
    edge = load_edge()
    nightday = load_nightday()
    pead_recent, pead_upcoming = load_pead()
    frm = load_factor_rotation()
    fund_cache = ideas_fund.get(tickers)
    val_hist = val._load_history()
    sector_of = {u["ticker"]: u["sector"] for u in universe}
    val_cross = val.cross_sectional(fund_cache, sector_of) if fund_cache else {}

    panel: dict[str, dict] = {}
    for u in universe:
        t = u["ticker"]
        is_stock = u["security_type"] == "stock"
        cov = {g: False for g in ("technicals", "sentiment", "positioning",
                                   "valuation", "fundamentals", "catalyst")}

        row: dict = {"meta": u}

        # --- technicals -----------------------------------------------
        if is_stock and t in mom_scores and mom_scores[t].get("composite") is not None:
            m = mom_scores[t]
            row["technicals"] = {
                "composite": m.get("composite"), "state": m.get("state"),
                "factor_scores": m.get("factor_scores"), "contributions": m.get("contributions"),
                "rank": m.get("rank"), "pctile": m.get("pctile"), "decile": m.get("decile"),
                "ts_grid": m.get("ts_grid"), "breakout_grid": m.get("breakout_grid"),
                "bars": m.get("bars"),
                "detail": mom_detail.get(t, {}),
                "nightday": nightday.get(t),
            }
            cov["technicals"] = True
            u["mktcap"] = m.get("mktcap")
        elif not is_stock and t in frm:
            f = frm[t]
            row["technicals"] = {"composite": round(20.0 * f.get("signal", 0.0), 2),
                                 "state": f.get("state"), "source": "cas.factor_rotation"}
            cov["technicals"] = True
        elif not is_stock and t in etf_px:
            fallback = _etf_price_technical(etf_px[t])
            if fallback is not None:
                row["technicals"] = fallback
                cov["technicals"] = True

        # --- sentiment (analyst revisions + option IV spread) ----------
        rev = edge["revisions"].get(t)
        ivs = edge["ivspread"].get(t)
        if rev or ivs:
            row["sentiment"] = {"revisions": rev, "ivspread": ivs}
            cov["sentiment"] = True

        # --- positioning (short interest + ownership) -------------------
        si = edge["shortint"].get(t)
        held_inst = fund_cache.get(t, {}).get("heldPercentInstitutions")
        if si or held_inst is not None:
            row["positioning"] = {"shortint": si, "held_pct_institutions": held_inst}
            cov["positioning"] = True

        # --- lottery / risk profile (feeds risk_reward, not positioning) --
        row["lottery"] = edge["lottery"].get(t)

        # --- valuation ----------------------------------------------------
        if t in fund_cache:
            mult = val.multiples_of(fund_cache[t])
            cross = val_cross.get(t)
            own = val.own_history_percentile(t, mult, val_hist)
            row["valuation"] = {"multiples": mult, "cross_sectional": cross, "own_history": own}
            cov["valuation"] = bool(cross or own.get("state") == "ready")

        # --- fundamentals (quality/growth/leverage) ------------------------
        if t in fund_cache:
            row["fundamentals"] = fund_cache[t]
            cov["fundamentals"] = True

        # --- catalyst (recent PEAD reaction + scheduled upcoming) ----------
        rec, up = pead_recent.get(t), pead_upcoming.get(t)
        if rec or up:
            row["catalyst"] = {"recent": rec, "upcoming": up}
            cov["catalyst"] = True

        row["coverage"] = cov
        panel[t] = row

    return panel

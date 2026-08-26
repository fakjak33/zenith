"""REGIMES cross-asset confirmation (spec section 28) + divergence detection
(spec section 29).

A suspected regime read is more trustworthy when independent MARKETS agree
with it, not just when the indicator composites agree with each other —
this is the "securities provide evidence about regimes" half of the
two-way relationship spec section 39 asks for (the other half, "regimes
influence securities", is IDEAS's macro tilt, updated separately).

Confirmation checks use a small set of cross-asset price trends (yfinance,
via cas.sources.prices — the same cached fetcher every other Zenith package
uses) PLUS this repo's own MOMENTUM sector aggregates (data/mom/sectors.json,
already committed nightly) for a cyclicals-vs-defensives breadth read that
needs no new data pull at all.

Divergences are FLAGGED, never auto-interpreted as bullish or bearish (spec
section 29 explicitly: "do not automatically interpret divergence as
bullish or bearish. Investigate it.") — each flag carries the two
disagreeing readings and lets the viewer draw their own conclusion.
"""

from __future__ import annotations

import pandas as pd

from ..cas.sources import prices as cas_prices
from ..mom import load as mom_load

CONTEXT_TICKERS = ("SPY", "TLT", "GLD", "DBC", "UUP", "IWM")

CYCLICAL_SECTORS = {"Technology", "Consumer Cyclical", "Industrials",
                    "Financial Services", "Basic Materials", "Energy"}
DEFENSIVE_SECTORS = {"Consumer Defensive", "Utilities", "Healthcare", "Real Estate"}


def context_trends(period: str = "1y") -> dict:
    """Trailing-3-month trend direction for a handful of cross-asset
    proxies — best-effort, never raises, missing tickers just drop out."""
    px, status = cas_prices.get_history(list(CONTEXT_TICKERS), period=period)
    out = {}
    for t in CONTEXT_TICKERS:
        df = px.get(t)
        if df is None or len(df) < 65:
            continue
        close = df["close"].dropna()
        chg3m = float(close.iloc[-1] / close.iloc[-63] - 1.0) if len(close) > 63 else None
        out[t] = {"trend_3m": None if chg3m is None else round(chg3m, 4),
                  "direction": None if chg3m is None else ("up" if chg3m > 0 else "down")}
    return out


def sector_breadth() -> dict:
    """Cyclicals-vs-defensives breadth from MOMENTUM's own daily sector
    aggregates — zero new data pull, reuses data/mom/sectors.json exactly as
    it's already committed."""
    doc = mom_load("sectors", {})
    sectors = doc.get("sectors", {})
    if not sectors:
        return {"available": False}
    cyc_means = [v["mean"] for k, v in sectors.items() if k in CYCLICAL_SECTORS and "mean" in v]
    def_means = [v["mean"] for k, v in sectors.items() if k in DEFENSIVE_SECTORS and "mean" in v]
    if not cyc_means or not def_means:
        return {"available": False}
    cyc_avg, def_avg = sum(cyc_means) / len(cyc_means), sum(def_means) / len(def_means)
    return {"available": True, "cyclicals_avg": round(cyc_avg, 3), "defensives_avg": round(def_avg, 3),
           "cyclicals_leading": bool(cyc_avg > def_avg), "as_of": doc.get("as_of")}


def confirmation(growth_rising: bool | None, infl_rising: bool | None,
                 trends: dict, breadth: dict) -> dict:
    """Independent evidence checks that either CONFIRM or CONTRADICT the
    classifier's growth/inflation read. Each check is deliberately narrow
    and named so a viewer can audit it, not a black-box "confidence" bump."""
    checks = []

    if growth_rising is not None and breadth.get("available"):
        agrees = breadth["cyclicals_leading"] == growth_rising
        checks.append({"check": "Cyclicals vs Defensives (MOMENTUM sector breadth)",
                       "expects": "cyclicals leading" if growth_rising else "defensives leading",
                       "observed": "cyclicals leading" if breadth["cyclicals_leading"] else "defensives leading",
                       "confirms": agrees})

    if growth_rising is not None and "IWM" in trends and "SPY" in trends:
        small_cap_outperforming = (trends["IWM"]["trend_3m"] or 0) > (trends["SPY"]["trend_3m"] or 0)
        checks.append({"check": "Small-cap vs Large-cap (risk appetite proxy)",
                       "expects": "small-caps outperforming" if growth_rising else "large-caps outperforming",
                       "observed": "small-caps outperforming" if small_cap_outperforming else "large-caps outperforming",
                       "confirms": small_cap_outperforming == growth_rising})

    if infl_rising is not None and "GLD" in trends:
        gold_up = trends["GLD"]["direction"] == "up"
        checks.append({"check": "Gold trend (inflation-hedge proxy)",
                       "expects": "rising" if infl_rising else "falling",
                       "observed": "rising" if gold_up else "falling",
                       "confirms": gold_up == infl_rising})

    if infl_rising is not None and "TLT" in trends:
        bonds_falling = trends["TLT"]["direction"] == "down"   # rising inflation -> yields up -> TLT down
        checks.append({"check": "Long Treasuries (rate-expectation proxy)",
                       "expects": "falling (yields up)" if infl_rising else "rising (yields down)",
                       "observed": "falling" if bonds_falling else "rising",
                       "confirms": bonds_falling == infl_rising})

    n = len(checks)
    n_confirm = sum(1 for c in checks if c["confirms"])
    return {"checks": checks, "n_confirming": n_confirm, "n_total": n,
           "confirmation_ratio": None if n == 0 else round(n_confirm / n, 3)}


def divergences(growth_rising: bool | None, infl_rising: bool | None,
                trends: dict, breadth: dict) -> list[dict]:
    """The confirmation checks that DISAGREE, surfaced as flags to
    investigate — never labelled bullish/bearish (spec section 29)."""
    conf = confirmation(growth_rising, infl_rising, trends, breadth)
    return [{"flag": c["check"], "classifier_expects": c["expects"], "market_shows": c["observed"],
            "note": "Independent evidence disagrees with the classifier's read here — worth "
                    "investigating why (market anticipation, a false signal, or a genuine "
                    "transition the slower-moving macro data hasn't caught up to yet)."}
           for c in conf["checks"] if not c["confirms"]]

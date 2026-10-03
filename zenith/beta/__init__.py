"""CLEAN BETA — a quality high-beta screener plus a convex-hedge monitor.

The thesis (Dennis line): high-beta stocks are historically OVERPRICED —
leverage-constrained investors overpay for beta (Frazzini & Pedersen 2014,
"Betting Against Beta") — UNLESS that beta comes from market correlation
rather than idiosyncratic noise. Liu, Stambaugh & Yuan (2018) show the beta
anomaly is really beta's correlation with idiosyncratic volatility: drop the
overpriced high-IVOL names and it goes away. So this tab screens for "clean
beta": high Welch bswa beta + high correlation + low IVOL + no event noise +
not fundamentally overpriced, then builds a diversified 30-50 name basket.

The hedge side follows the research too:
  * PRIMARY (inducible) defense = trend gate + vol targeting. Cheap when
    quiet, active on signal (Hurst, Ooi & Pedersen; vol targeting is risk
    control, not alpha — Cederburg et al. 2020).
  * SMALL (constitutive) crash sleeve = laddered INDEX put spreads, sized by
    the volatility risk premium, not by the VIX level (Israelov 2019;
    Israelov & Nielsen 2015). Index, not single-name, because index options
    carry the correlation premium that a high-correlation basket is exposed to
    (Driessen, Maenhout & Vilkov 2009) and single-name puts on high-IVOL names
    are the most overpriced (Cao & Han 2013).
  * BTAL (anti-beta ETF) is a live betting-against-beta factor: when it
    trends up, a high-beta sleeve is fighting the tape.

Cadence: one daily job. The screen re-estimates when the stored screen is
from an earlier month; the basket rebalances when it is from an earlier
quarter-start month (config.BETA_REBALANCE_MONTHS); the hedge monitor runs
every trading day. A monitor and screener, not an execution engine — the
put ladder is advisory. Every monthly screen also writes a point-in-time
snapshot to data/beta/exports/ for PARALLAX backtests.

Data confidence: every column carries a [P]/[S]/[E] tag (CONF below).
Design decisions are logged in DECISIONS.md next to this file.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..config import BETA_FILES

DISCLAIMER = ("CLEAN BETA screens the Russell 1000 for high beta that comes from market correlation "
              "rather than idiosyncratic noise, builds a diversified candidate basket, and monitors "
              "hedge conditions (trend gate, vol target, volatility risk premium, index put ladder). "
              "The basket and the put ladder are research output, not positions or recommendations. "
              "Decision-support and a research monitor, not investment advice.")

# Data-confidence tags, scoped to this tab.
CONF = {
    "P": "Primary: computed here from daily prices (yfinance adjusted closes).",
    "S": "Secondary: a vendor-reported field taken as-is (yfinance .info fundamentals, option-chain IV/quotes, "
         "Nasdaq earnings calendar).",
    "E": "Estimated: a proxy, an imputation, or a model forecast (e.g. accruals from margins, HAR-lite realized "
         "vol, a neutral quality score where fundamentals are missing).",
}

FIELD_CONF = {
    "bswa": "P", "ols": "P", "vasicek": "E", "rho": "P", "r2": "P", "ivol": "P", "vol": "P",
    "jumps": "P", "adv": "P", "mom": "P", "mktcap": "S", "earnings": "S", "quality": "S",
    "score": "P", "basket_beta": "P", "basket_vol": "P", "enb": "P",
    "trend": "P", "vol_scale": "E", "vrp": "E", "vix": "S", "btal": "P", "puts": "S",
}


def tag(field: str) -> str:
    return f"[{FIELD_CONF.get(field, 'E')}]"


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(BETA_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    p = BETA_FILES[name]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scrub(obj), indent=indent, ensure_ascii=False), encoding="utf-8")


def scrub(obj):
    """Recursively replace non-finite floats with None (valid JSON only)."""
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def r(x, nd: int = 4):
    """Round a finite number, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None

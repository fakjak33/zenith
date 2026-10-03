"""CLEAN BETA quality / mispricing proxy.

Liu, Stambaugh & Yuan's beta-anomaly result runs through MISPRICING: the
high-beta names that disappoint are the overpriced ones. Their mispricing
score (Stambaugh, Yu & Yuan 2015) needs eleven accounting anomalies; this is a
free-data proxy built from the fundamentals IDEAS already caches
(data/ideas/fundamentals.json, yfinance .info, refreshed nightly by IDEAS and
read-only here), plus price momentum computed here:

  roa          return on assets                               higher = better  [S]
  gross_margin gross margin (Novy-Marx gross profitability)    higher = better  [S]
  fcf_margin   free cash flow / revenue                        higher = better  [E]
  accruals     (net income − operating cash flow) / revenue    lower  = better  [E]
               (Sloan accruals, scaled by revenue because .info carries no
               total assets; net income = profit margin × revenue)
  leverage     debt / equity (negative equity = worst)         lower  = better  [S]
  mom          12-1 month price momentum                       higher = better  [P]
  issuance     12-month log change in shares outstanding       lower  = better  [P]
               — only once data/beta/shares.json holds 12 months of this
               tab's own monthly snapshots (started with the tab).

Each component is a within-universe percentile; the quality score is their
mean. Fewer than BETA_QUALITY_MIN_COMPONENTS usable components -> a neutral 50
flagged [E] (never fabricated, never a fail).
"""

from __future__ import annotations

import math
from datetime import date

from ..config import BETA_ISSUANCE_MONTHS, BETA_QUALITY_MIN_COMPONENTS
from .estimators import pct_rank

COMPONENTS = ("roa", "gross_margin", "fcf_margin", "accruals", "leverage", "mom", "issuance")
HIGHER_BETTER = {"roa": True, "gross_margin": True, "fcf_margin": True, "accruals": False,
                 "leverage": False, "mom": True, "issuance": False}


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def raw_components(info: dict | None, mom: float | None, issuance: float | None) -> dict:
    info = info or {}
    rev = _f(info.get("totalRevenue"))
    pm = _f(info.get("profitMargins"))
    ocf = _f(info.get("operatingCashflow"))
    fcf = _f(info.get("freeCashflow"))
    de = _f(info.get("debtToEquity"))
    out = {"roa": _f(info.get("returnOnAssets")), "gross_margin": _f(info.get("grossMargins")),
           "fcf_margin": fcf / rev if fcf is not None and rev else None,
           "accruals": (pm * rev - ocf) / rev if None not in (pm, ocf) and rev else None,
           # yfinance reports D/E in percent and omits it for negative equity;
           # a reported negative is treated as the worst leverage.
           "leverage": (1e6 if de is not None and de < 0 else de),
           "mom": mom, "issuance": issuance}
    return out


def scores(raw: dict[str, dict]) -> dict[str, dict]:
    """{ticker: raw components} -> {ticker: {quality, n_components, imputed,
    <component>_pct...}}."""
    tickers = list(raw)
    pcts: dict[str, list] = {}
    for c in COMPONENTS:
        vals = [raw[t].get(c) for t in tickers]
        pr = pct_rank(vals)
        pcts[c] = [None if p is None else (p if HIGHER_BETTER[c] else 100.0 - p) for p in pr]
    out = {}
    for i, t in enumerate(tickers):
        got = {c: pcts[c][i] for c in COMPONENTS if pcts[c][i] is not None}
        imputed = len(got) < BETA_QUALITY_MIN_COMPONENTS
        q = 50.0 if imputed else sum(got.values()) / len(got)
        out[t] = {"quality": round(q, 1), "quality_n": len(got), "quality_imputed": imputed,
                  **{f"q_{c}": (round(v, 1) if v is not None else None)
                     for c, v in ((c, pcts[c][i]) for c in COMPONENTS)}}
    return out


def snapshot_shares(shares_doc: dict, fundamentals: dict[str, dict], today: date) -> dict:
    """Append this month's sharesOutstanding per ticker (idempotent within a month)."""
    month = today.strftime("%Y-%m")
    for t, info in fundamentals.items():
        so = _f((info or {}).get("sharesOutstanding"))
        if so and so > 0:
            shares_doc.setdefault(t, {})[month] = so
    return shares_doc


def issuance(shares_doc: dict, ticker: str, today: date) -> float | None:
    """12-month log change in shares outstanding from this tab's own snapshots."""
    hist = shares_doc.get(ticker) or {}
    if not hist:
        return None
    y, m = today.year, today.month
    now_key = f"{y:04d}-{m:02d}"
    back_m = m - BETA_ISSUANCE_MONTHS
    by, bm = y + (back_m - 1) // 12, (back_m - 1) % 12 + 1
    then_key = f"{by:04d}-{bm:02d}"
    a, b = hist.get(now_key), hist.get(then_key)
    if not a or not b:
        return None
    return math.log(a / b)

"""Who gets screened: cash indices, the liquid ETF list, and the most-traded
Russell 1000 stocks by dollar volume.

Stocks come from EPHEMERIS's tagged universe (data/ephemeris/universe.json,
`cls == "Russell 1000"`) ranked by median 63-day dollar volume from the same
release-asset price store THETA RUN reads — no extra download. The option
chain decides the rest: names whose chains turn out thin simply rate 1★.
"""

from __future__ import annotations

import json

from ..config import EPHEMERIS_FILES, VOLSCREEN_N_STOCKS
from ..theta.build import ETFS

INDICES = {"SPX": ("S&P 500 Index", "^GSPC", 1.0), "XSP": ("Mini-SPX Index", "^GSPC", 0.1),
           "NDX": ("Nasdaq-100 Index", "^NDX", 1.0), "RUT": ("Russell 2000 Index", "^RUT", 1.0)}
EXTRA_ETFS = ["XLC", "XLRE", "SOXX", "IBIT", "EWW", "EWY", "INDA", "EWT", "TQQQ", "SQQQ", "BITO",
              "JETS", "ITB", "XHB", "KBE", "COPX", "SIL", "URA", "TAN", "ICLN", "LIT", "BOTZ",
              "HYG", "EMB", "BND", "AGG", "SHY", "TBT", "FXE", "UUP", "DBA", "CORN", "WEAT",
              "VXX", "UVXY", "SVXY", "MCHI", "VGK", "EWG", "EWC", "EWA", "EWH"]
# leveraged / volatility ETPs are screened for liquidity but their VRP is not
# comparable to an unlevered underlying's: flagged so the checklist can say so
SPECIAL = {"TQQQ": "3x leveraged", "SQQQ": "-3x leveraged", "TBT": "-2x leveraged",
           "VXX": "VIX futures ETN", "UVXY": "1.5x VIX futures", "SVXY": "-0.5x VIX futures",
           "BITO": "bitcoin futures"}


def _rows() -> list[dict]:
    try:
        return json.loads(EPHEMERIS_FILES["universe"].read_text(encoding="utf-8")).get("rows", [])
    except Exception:
        return []


def candidates(store=None, n_stocks: int = VOLSCREEN_N_STOCKS) -> list[dict]:
    """[{ticker, kind, name, sector, note, yf, scale}] in screening order."""
    rows = _rows()
    meta = {r["ticker"]: r for r in rows}
    out = [{"ticker": t, "kind": "index", "name": n, "sector": "Index", "note": "", "yf": yf, "scale": sc}
           for t, (n, yf, sc) in INDICES.items()]
    seen = set(INDICES)
    for t in ETFS + EXTRA_ETFS:
        if t in seen:
            continue
        seen.add(t)
        m = meta.get(t, {})
        out.append({"ticker": t, "kind": "etf", "name": m.get("name", t),
                    "sector": m.get("category") or m.get("cls") or "ETF", "note": SPECIAL.get(t, ""),
                    "yf": t, "scale": 1.0})
    stocks = [r for r in rows if r.get("cls") == "Russell 1000" and r["ticker"] not in seen]
    if store is not None:
        dv = {}
        for r in stocks:
            try:
                df = store.bars(r["ticker"])
            except Exception:
                df = None
            if df is not None and len(df) >= 63:
                t = df.tail(63)
                dv[r["ticker"]] = float((t["close"] * t["volume"]).median())
        stocks = sorted((r for r in stocks if r["ticker"] in dv), key=lambda r: -dv[r["ticker"]])
    for r in stocks[:n_stocks]:
        out.append({"ticker": r["ticker"], "kind": "stock", "name": r.get("name", r["ticker"]),
                    "sector": r.get("sector") or "", "note": "", "yf": r["ticker"], "scale": 1.0})
    return out

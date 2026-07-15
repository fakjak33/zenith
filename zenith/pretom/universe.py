"""Russell 1000 membership + cap weights (free, via Vanguard VONE holdings).

Vanguard's VONE (Russell 1000 ETF) holdings API is a free JSON feed carrying
ticker, name and index weight — the weight doubles as the market-cap /
"prevalence across funds" proxy without a thousand per-ticker calls. Sectors
are enriched from the Wikipedia components table where available. Fallback
chain: 168h cache -> Vanguard -> Wikipedia (no weights) -> last committed
universe.json snapshot.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import BROWSER_HEADERS
from ..cas import store_cas
from . import load as pretom_load, save as pretom_save

VONE_URL = ("https://investor.vanguard.com/investment-products/etfs/profile/api/"
            "VONE/portfolio-holding/stock")


def parse_vone_items(items: list[dict]) -> list[dict]:
    """Normalize Vanguard portfolio-holding entities to universe rows."""
    out = []
    for it in items:
        t = str(it.get("ticker", "")).strip().replace(".", "-").upper()
        if not t or t in ("-", "NAN") or not t[0].isalpha():
            continue
        try:
            w = float(str(it.get("percentWeight", "")).replace(",", ""))
        except Exception:
            w = None
        out.append({"ticker": t,
                    "name": str(it.get("longName") or it.get("shortName") or t).strip(),
                    "sector": "", "weight_pct": w})
    return out


def _fetch_vone() -> list[dict]:
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = "https://investor.vanguard.com/"
    items, start = [], 1
    while True:
        r = requests.get(VONE_URL, params={"start": start, "count": 500},
                         headers=headers, timeout=30)
        r.raise_for_status()
        batch = r.json().get("fund", {}).get("entity", [])
        items += batch
        if len(batch) < 500:
            break
        start += 500
    rows = parse_vone_items(items)
    if len(rows) < 800:          # sanity: Russell 1000 should be ~1000 equities
        raise ValueError(f"VONE parse returned only {len(rows)} rows")
    return rows


def _wikipedia_table() -> pd.DataFrame | None:
    r = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index",
                     headers=BROWSER_HEADERS, timeout=20)
    r.raise_for_status()
    for tbl in pd.read_html(io.StringIO(r.text)):
        cols = {str(c).lower(): c for c in tbl.columns}
        if (cols.get("ticker") or cols.get("symbol")) and len(tbl) >= 800:
            return tbl
    return None


def _fetch_wikipedia() -> list[dict]:
    tbl = _wikipedia_table()
    if tbl is None:
        return []
    cols = {str(c).lower(): c for c in tbl.columns}
    tick = cols.get("ticker") or cols.get("symbol")
    name = cols.get("company") or cols.get("security")
    sect = cols.get("gics sector") or cols.get("sector")
    out = []
    for _, row in tbl.iterrows():
        out.append({"ticker": str(row[tick]).replace(".", "-").strip().upper(),
                    "name": str(row[name]).strip() if name else "",
                    "sector": str(row[sect]).strip() if sect else "",
                    "weight_pct": None})
    return out


def _enrich_sectors(rows: list[dict]) -> None:
    """Fill sector from the Wikipedia components table (best-effort)."""
    try:
        wiki = {r["ticker"]: r["sector"] for r in _fetch_wikipedia() if r["sector"]}
    except Exception:
        return
    for r in rows:
        if not r["sector"]:
            r["sector"] = wiki.get(r["ticker"], "")


def russell1000(max_age_hours: float = 168.0) -> tuple[list[dict], dict]:
    """Current Russell 1000 rows + status {source, n, weights_source}.

    Never raises; falls back cache -> Vanguard -> Wikipedia -> committed
    snapshot.
    """
    cached = store_cas.cache_get("russell1000", max_age_hours)
    if cached:
        return cached["rows"], {"source": cached.get("source", "cache"),
                                "n": len(cached["rows"]),
                                "weights_source": cached.get("weights_source", "vone")}
    for fn, source, wsrc in ((_fetch_vone, "vanguard_vone", "vone"),
                             (_fetch_wikipedia, "wikipedia", "fallback_equal")):
        try:
            rows = fn()
        except Exception:
            rows = []
        if rows:
            if source == "vanguard_vone":
                _enrich_sectors(rows)
            store_cas.cache_put("russell1000", {"rows": rows, "source": source,
                                                "weights_source": wsrc})
            pretom_save("universe", {"rows": rows, "source": source,
                                     "weights_source": wsrc,
                                     "as_of": store_cas.today_str()})
            return rows, {"source": source, "n": len(rows), "weights_source": wsrc}
    last = pretom_load("universe", {})
    rows = last.get("rows", [])
    return rows, {"source": f"stale:{last.get('source', 'none')}", "n": len(rows),
                  "weights_source": last.get("weights_source", "fallback_equal")}


def dividend_payers(tickers: list[str], chunk: int = 100) -> dict[str, bool | None]:
    """Ticker -> paid any dividend in the trailing year (None = unknown)."""
    out: dict[str, bool | None] = {t: None for t in tickers}
    try:
        import yfinance as yf
    except Exception:
        return out
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            raw = yf.download(batch, period="1y", interval="1d", actions=True,
                              auto_adjust=True, group_by="ticker", threads=True,
                              progress=False)
        except Exception:
            continue
        for t in batch:
            try:
                div = (raw[t]["Dividends"] if isinstance(raw.columns, pd.MultiIndex)
                       else raw["Dividends"])
                out[t] = bool(div.fillna(0).sum() > 0)
            except Exception:
                continue
    return out

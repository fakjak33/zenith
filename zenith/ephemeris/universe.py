"""The EPHEMERIS universe: Russell 1000 + the ZENITH ETF universe + spot FX,
crypto and a few deep-history indices, each tagged with a game class.

ZENITH's own taxonomy (etfmom.universe.asset_class_of over Morningstar
categories) has no separate International or Precious Metals class, so two
splits are derived here -- mechanically, from the same category vocabulary:
  * Equity -> "International Equity" when the category names a non-US region
    (Foreign / Emerging / Europe / Japan / China / India / Pacific / Global /
    Focused Region), else "US Equity ETFs".
  * Commodity -> "Precious Metals" when the fund is a gold / silver /
    platinum / palladium vehicle (by name), else "Commodities".
Only committed artefacts are read, so building the universe is offline.
"""

from __future__ import annotations

import json
import re

from ..config import DATA_DIR, EPHEMERIS_DIR, EPHEMERIS_FILES

_ETF_SCORES = DATA_DIR / "etfmom" / "scores_latest.json"
_R1000 = DATA_DIR / "pretom" / "universe.json"
_META = DATA_DIR / "mom" / "meta.json"

_INTL_RE = re.compile(r"Foreign|Emerging|Europe|Japan|China|India|Pacific|Global|Focused Region|"
                      r"Latin|World|Asia|Diversified Emerging", re.I)
_PM_RE = re.compile(r"\b(gold|silver|platinum|palladium|precious)\b", re.I)

_CLASS_MAP = {"Fixed Income": "Bonds", "Real Estate": "Real Estate", "Currency": "Currencies",
              "Digital Assets": "Crypto", "Alternative": "Alternatives", "Allocation": "Alternatives"}

# Spot series for depth where the ETF set is thin. No volume for FX.
SPOT = {
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ether", "SOL-USD": "Solana", "XRP-USD": "XRP",
               "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "LTC-USD": "Litecoin",
               "BCH-USD": "Bitcoin Cash", "LINK-USD": "Chainlink"},
    "Currencies": {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
                   "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
                   "NZDUSD=X": "NZD/USD", "EURJPY=X": "EUR/JPY", "EURGBP=X": "EUR/GBP",
                   "USDMXN=X": "USD/MXN"},
    "Indices": {"^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite",
                "^RUT": "Russell 2000", "^N225": "Nikkei 225", "^FTSE": "FTSE 100",
                "^GDAXI": "DAX", "^HSI": "Hang Seng"},
}


def cap_bucket(mktcap) -> str:
    """Same breakpoints CLEAN BETA uses."""
    if not mktcap:
        return ""
    m = float(mktcap)
    return "Mega" if m >= 200e9 else "Large" if m >= 50e9 else "Mid" if m >= 10e9 else "Small"


def game_class(asset_class: str, category: str, name: str) -> str | None:
    if asset_class == "Equity":
        return "International Equity" if _INTL_RE.search(category or "") else "US Equity ETFs"
    if asset_class == "Commodity":
        return "Precious Metals" if _PM_RE.search(name or "") else "Commodities"
    return _CLASS_MAP.get(asset_class)          # Unknown -> None (dropped)


def _load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def build() -> list[dict]:
    """Every tradable row: {ticker, name, cls, sector, cap, category}."""
    meta = _load(_META, {})
    rows: dict[str, dict] = {}
    for r in _load(_R1000, {}).get("rows", []):
        t = r["ticker"]
        m = meta.get(t, {})
        rows[t] = {"ticker": t, "name": m.get("name") or r.get("name") or t, "cls": "Russell 1000",
                   "sector": m.get("sector") or r.get("sector") or "", "cap": cap_bucket(m.get("mktcap")),
                   "category": m.get("industry") or ""}
    for r in _load(_ETF_SCORES, {}).get("rows", []):
        if r.get("excluded") or r["ticker"] in rows:
            continue
        cls = game_class(r.get("asset_class", ""), r.get("category", ""), r.get("name", ""))
        if cls is None:
            continue
        rows[r["ticker"]] = {"ticker": r["ticker"], "name": r.get("name") or r["ticker"], "cls": cls,
                             "sector": "", "cap": "", "category": r.get("category") or ""}
    for cls, names in SPOT.items():
        for t, nm in names.items():
            rows.setdefault(t, {"ticker": t, "name": nm, "cls": cls, "sector": "", "cap": "",
                                "category": "Spot"})
    return sorted(rows.values(), key=lambda x: (x["cls"], x["ticker"]))


def write(rows: list[dict] | None = None) -> list[dict]:
    rows = build() if rows is None else rows
    EPHEMERIS_DIR.mkdir(parents=True, exist_ok=True)
    EPHEMERIS_FILES["universe"].write_text(json.dumps({"rows": rows}, indent=0), encoding="utf-8")
    return rows


def load() -> list[dict]:
    """Committed universe, or a fresh offline build if it is missing."""
    rows = _load(EPHEMERIS_FILES["universe"], {}).get("rows")
    return rows if rows else build()

"""Bounded universe for the CAS monitor.

Deliberately small so the nightly GitHub Action stays fast and free data sources
(yfinance) don't rate-limit. Covers the asset classes the user cares about:
sectors, themes (semis, gold, DRAM/memory, EM/EEM …), factors, and macro proxies.
Extend the lists over time — nothing else hard-codes membership.
"""

from __future__ import annotations

# Sector SPDRs — the canonical sector-rotation set
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLI": "Industrials", "XLY": "Cons. Discretionary",
    "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication",
}

# Theme / industry ETFs — the kind of rallies the user wants to catch early
THEMES = {
    "SMH": "Semiconductors", "SOXX": "Semiconductors", "XSD": "Semis (equal-wt)",
    "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners",
    "XME": "Metals & Mining", "LIT": "Lithium/Battery", "URA": "Uranium",
    "EEM": "EM Equity", "FXI": "China", "EWZ": "Brazil", "INDA": "India",
    "IBB": "Biotech", "XBI": "Biotech (equal-wt)", "ITB": "Homebuilders",
    "KRE": "Regional Banks", "JETS": "Airlines", "TAN": "Solar",
    "ARKK": "Disruptive Growth", "IGV": "Software", "SKYY": "Cloud",
}

# Broad / factor / style proxies
FACTORS = {
    "SPY": "US Large Cap", "QQQ": "Nasdaq 100", "IWM": "US Small Cap",
    "MTUM": "Momentum", "VLUE": "Value", "QUAL": "Quality",
    "USMV": "Min Volatility", "SPHB": "High Beta", "RSP": "Equal-Weight S&P",
    "VEA": "Dev. ex-US", "VWO": "EM (broad)",
}

# Macro / cross-asset proxies (also map to COT futures markets)
MACRO = {
    "TLT": "20y Treasuries", "IEF": "7-10y Treasuries", "HYG": "High Yield",
    "LQD": "IG Credit", "UUP": "US Dollar", "DBC": "Commodities",
    "USO": "Crude Oil", "UNG": "Nat Gas", "BITO": "Bitcoin (futures)",
}

# ETF -> CFTC COT market name (for the flows/positioning proxy)
COT_MAP = {
    "SPY": "E-MINI S&P 500", "QQQ": "NASDAQ-100 E-MINI", "IWM": "RUSSELL 2000 E-MINI",
    "GLD": "GOLD", "SLV": "SILVER", "USO": "CRUDE OIL, LIGHT SWEET",
    "UNG": "NATURAL GAS", "TLT": "ULTRA U.S. TREASURY BOND", "UUP": "U.S. DOLLAR INDEX",
    "DBC": "BLOOMBERG COMMODITY INDEX", "BITO": "BITCOIN",
}


def all_etfs() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in (SECTORS, THEMES, FACTORS, MACRO):
        out.update(d)
    return out


def asset_class_of(ticker: str) -> str:
    if ticker in SECTORS:
        return "sector"
    if ticker in THEMES:
        return "theme"
    if ticker in FACTORS:
        return "factor"
    if ticker in MACRO:
        return "macro"
    return "equity"


def label_of(ticker: str) -> str:
    return all_etfs().get(ticker, ticker)

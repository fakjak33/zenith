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

# --- Style/factor x region grid for the Style-Trend model ------------------
# All US-listed ETFs so the free yfinance pull works unchanged. One primary
# ticker per (style, region) cell; region keys: US, DEV (international developed
# ex-US), EM (emerging markets). Cells with no clean pure-factor ETF are None
# (EM quality/growth) so they don't pollute the cross-section. Tickers are kept
# unique across cells so reverse lookup + cross-sectional ranking are clean.
STYLE_GRID: dict[str, dict[str, str | None]] = {
    "Value":       {"US": "VLUE", "DEV": "IVLU", "EM": "FNDE"},
    "Momentum":    {"US": "MTUM", "DEV": "IMTM", "EM": "PIE"},
    "Quality":     {"US": "QUAL", "DEV": "IQLT", "EM": "XSOE"},   # XSOE = EM ex-SOE, a quality proxy
    "LowVol":      {"US": "USMV", "DEV": "EFAV", "EM": "EEMV"},
    "Size":        {"US": "SIZE", "DEV": "ISCF", "EM": "DGS"},
    "Dividend":    {"US": "VYM",  "DEV": "IDV",  "EM": "DEM"},
    "Growth":      {"US": "IWF",  "DEV": "EFG",  "EM": None},
    "Buyback":     {"US": "PKW",  "DEV": "IPKW", "EM": "EYLD"},
    "Multifactor": {"US": "LRGF", "DEV": "INTF", "EM": "EMGF"},
}

REGION_LABEL = {"US": "US", "DEV": "Intl-Developed", "EM": "Emerging Mkts",
                "EU": "Europe", "CN": "China"}

# ticker -> (style, region); first occurrence wins (tickers are unique anyway)
_STYLE_INFO: dict[str, tuple[str, str]] = {}
for _style, _row in STYLE_GRID.items():
    for _region, _tkr in _row.items():
        if _tkr and _tkr not in _STYLE_INFO:
            _STYLE_INFO[_tkr] = (_style, _region)

# --- Industry / sub-industry ETFs (US-listed) for the Factor-Rotation model ---
# Comprehensive curated set; the price pull skips any that don't return data, so
# the model degrades gracefully if an ETF is thin/delisted. ticker -> label.
INDUSTRY_ETFS: dict[str, str] = {
    # technology / internet
    "SMH": "Semiconductors", "SOXX": "Semiconductors (iShares)", "XSD": "Semis (equal-wt)",
    "PSI": "Semis (Invesco)", "IGV": "Software", "WCLD": "Cloud (WisdomTree)",
    "SKYY": "Cloud Computing", "CLOU": "Cloud (Global X)", "CIBR": "Cybersecurity",
    "HACK": "Cybersecurity (ETFMG)", "BUG": "Cybersecurity (Global X)", "IYW": "US Technology",
    "FDN": "Internet", "IPAY": "Mobile Payments", "FINX": "FinTech",
    "BOTZ": "Robotics & AI", "ROBO": "Robotics", "IRBO": "AI & Robotics", "AIQ": "AI",
    # financials
    "KRE": "Regional Banks", "KBE": "Banks", "KBWB": "Big Banks", "IAT": "Regional Banks (iShares)",
    "IAI": "Broker-Dealers", "KIE": "Insurance", "IAK": "Insurance (iShares)",
    # health care
    "XBI": "Biotech (equal-wt)", "IBB": "Biotech", "GNOM": "Genomics", "ARKG": "Genomic Revolution",
    "IHI": "Medical Devices", "XHE": "Health Care Equipment", "IHF": "Health Care Providers",
    "PPH": "Pharmaceuticals", "IHE": "US Pharma",
    # industrials / infra / transports
    "ITA": "Aerospace & Defense", "XAR": "Aerospace & Defense (equal-wt)", "PPA": "Defense (Invesco)",
    "PAVE": "US Infrastructure", "IFRA": "Infrastructure (iShares)", "GRID": "Smart Grid",
    "IYT": "Transports", "XTN": "Transportation (equal-wt)", "JETS": "Airlines", "BOAT": "Shipping",
    # consumer / leisure
    "XRT": "Retail (equal-wt)", "RTH": "Retail (VanEck)", "PEJ": "Leisure & Entertainment",
    "ESPO": "Video Games & eSports", "BETZ": "Sports Betting & iGaming",
    # housing / real estate
    "XHB": "Homebuilders", "ITB": "Home Construction", "PKB": "Building & Construction",
    "VNQ": "US REITs", "REM": "Mortgage REITs",
    # energy
    "XOP": "Oil & Gas E&P", "IEO": "Oil & Gas E&P (iShares)", "OIH": "Oil Services",
    "XES": "Oil Equipment & Services", "FCG": "Natural Gas", "AMLP": "MLPs", "MLPX": "Midstream",
    # clean energy
    "ICLN": "Clean Energy", "TAN": "Solar", "FAN": "Wind", "PBW": "Clean Energy (Invesco)",
    "QCLN": "Clean Energy (First Trust)", "LIT": "Lithium & Battery", "URA": "Uranium",
    "URNM": "Uranium Miners", "NLR": "Nuclear",
    # materials / miners / ags
    "XME": "Metals & Mining", "PICK": "Global Metals & Mining", "COPX": "Copper Miners",
    "REMX": "Rare Earth/Strategic Metals", "SIL": "Silver Miners", "GDX": "Gold Miners",
    "GDXJ": "Junior Gold Miners", "RING": "Gold Miners (iShares)", "SLX": "Steel",
    "MOO": "Agribusiness", "VEGI": "Agribusiness (iShares)", "WOOD": "Timber & Forestry",
    "PHO": "Water Resources", "FIW": "Water (First Trust)",
    # thematic / misc
    "MSOS": "US Cannabis", "MJ": "Cannabis (ETFMG)", "BLOK": "Blockchain", "WGMI": "Bitcoin Miners",
    "ARKX": "Space & Exploration", "UFO": "Space (Procure)", "SEA": "Shipping (US Global)",
    "IYZ": "Telecom", "PBJ": "Food & Beverage", "XSW": "Software (equal-wt)",
    "ARKW": "Next-Gen Internet", "EVX": "Environmental Services", "IGE": "Natural Resources",
    "KBWP": "Property & Casualty Insurance", "PSCT": "Small-Cap Tech",
    "FTXL": "Semiconductors (Nasdaq)",
}

# --- Region-specific sector ETFs (US-listed) — genuinely sparse; only the liquid
# ones that actually exist are included. ticker -> (label, region). Many hyper-
# specific combos (e.g. "Brazil precious-metals miners", "Japan banks") have NO
# US-listed ETF; not faked. region keys extend REGION_LABEL below.
REGION_SECTOR_ETFS: dict[str, tuple[str, str]] = {
    "EUFN": ("Europe Financials", "EU"),
    "EUAD": ("Europe Aerospace & Defense", "EU"),
    "KWEB": ("China Internet", "CN"),
    "CQQQ": ("China Technology", "CN"),
    "CHIQ": ("China Consumer", "CN"),
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


def master_etfs() -> list[str]:
    """The curated liquid master ETF list (~400) that the 151-strategies CAS model
    scans. Union of the core universe above + etf_master.MASTER_ETFS + the
    style-grid tickers (so the Style-Trend ETFs are always price-pulled)."""
    from .etf_master import master_tickers
    core = list(all_etfs())
    seen = set(core)
    out = core + [t for t in master_tickers() if t not in seen]
    seen.update(out)
    out += [t for t in frm_tickers() if t not in seen]
    return out


# --- style grid helpers ----------------------------------------------------
def style_grid() -> dict[str, dict[str, str | None]]:
    return STYLE_GRID


def style_tickers() -> list[str]:
    """Unique, sorted list of all (non-None) style-grid ETF tickers."""
    return sorted(_STYLE_INFO)


def style_of(ticker: str) -> str | None:
    info = _STYLE_INFO.get(ticker)
    return info[0] if info else None


def region_of(ticker: str) -> str | None:
    info = _STYLE_INFO.get(ticker)
    return info[1] if info else None


# --- Factor-Rotation tagged universe ---------------------------------------
# ticker -> {"group": style|industry|region_sector, "label": str, "region": str}
# Built once from the style grid + industry + region-sector sets.
def _build_frm_universe() -> dict[str, dict[str, str]]:
    uni: dict[str, dict[str, str]] = {}
    for tkr, (style, region) in _STYLE_INFO.items():
        uni[tkr] = {"group": "style", "label": style, "region": region}
    for tkr, label in INDUSTRY_ETFS.items():
        uni.setdefault(tkr, {"group": "industry", "label": label, "region": "US"})
    for tkr, (label, region) in REGION_SECTOR_ETFS.items():
        uni.setdefault(tkr, {"group": "region_sector", "label": label, "region": region})
    return uni


_FRM_UNIVERSE = _build_frm_universe()


def frm_universe() -> dict[str, dict[str, str]]:
    """Full tagged universe for the Factor-Rotation Momentum model."""
    return _FRM_UNIVERSE


def frm_tickers() -> list[str]:
    return sorted(_FRM_UNIVERSE)


def frm_tag(ticker: str) -> dict[str, str] | None:
    """Tag dict {group,label,region} for a ticker, or None if not in the universe."""
    return _FRM_UNIVERSE.get(ticker)


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

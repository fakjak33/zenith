"""Curated liquid master ETF list for the CAS 151-strategies model.

~400 well-known, liquid ETFs spanning broad market, sectors, industries/themes,
factors/styles, country/region, fixed income, commodities and volatility. Chosen
for liquidity/recognisability (a nightly yfinance pull of this set stays within
GitHub Action limits when chunked + cached). Extend toward the full master list
(ghost/data/etf_master_list.csv, ~2,300 names) later if it proves reliable.

Stored grouped by category so signals can be tagged by asset class; ``MASTER_ETFS``
flattens to ticker -> category.
"""

from __future__ import annotations

_GROUPS: dict[str, list[str]] = {
    "broad": [
        "SPY", "VOO", "IVV", "QQQ", "QQQM", "DIA", "IWM", "IWB", "IWV", "RSP",
        "VTI", "ITOT", "SCHB", "MGC", "OEF", "VV", "VXF", "SCHX", "SPLG", "SPTM",
    ],
    "sector": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
        "XLC", "VGT", "VFH", "VDE", "VHT", "VIS", "VCR", "VDC", "VPU", "VAW",
        "VNQ", "VOX", "FTEC", "FNCL", "FENY", "FHLC", "RYT", "RCD", "RHS", "RYE",
    ],
    "industry_theme": [
        "SMH", "SOXX", "XSD", "SOXL", "IGV", "VGT", "SKYY", "WCLD", "CIBR", "HACK",
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "BOTZ", "ROBO", "IRBO", "AIQ", "BUG",
        "IBB", "XBI", "ARKG", "GNOM", "IHI", "IHF", "XHE", "XHS", "PPH", "FBT",
        "ITB", "XHB", "NAIL", "PKB", "KRE", "KBE", "IAT", "KBWB", "KIE", "IAK",
        "JETS", "XTN", "IYT", "SEA", "BOAT", "TAN", "ICLN", "QCLN", "PBW", "FAN",
        "LIT", "BATT", "URA", "URNM", "NLR", "XME", "PICK", "COPX", "REMX", "SIL",
        "GDX", "GDXJ", "RING", "SGDM", "OIH", "XOP", "XES", "IEO", "FCG", "AMLP",
        "MLPX", "AMJ", "FXN", "PSCE", "MOO", "VEGI", "WOOD", "CUT", "PHO", "FIW",
        "HERO", "ESPO", "NERD", "GAMR", "BETZ", "MJ", "MSOS", "YOLO", "BLOK", "DAPP",
        "FINX", "IPAY", "ARKX", "ROKT", "UFO", "PAVE", "IFRA", "GRID", "SNSR", "IDRV",
    ],
    "factor_style": [
        "MTUM", "VLUE", "QUAL", "USMV", "SPHB", "SPLV", "SPHD", "VYM", "VIG", "DGRO",
        "SCHD", "NOBL", "DVY", "HDV", "SDY", "IWF", "IWD", "IWN", "IWO", "IWP",
        "IWS", "IWR", "VUG", "VTV", "VBR", "VBK", "VOT", "VOE", "MDY", "IJH",
        "IJR", "VB", "VO", "SLY", "FNDX", "FNDA", "RPV", "RPG", "PRF", "COWZ",
        "QUAL", "FQAL", "MOAT", "QGRO", "JEPI", "JEPQ", "DIVO", "QYLD", "RYLD", "XYLD",
    ],
    "country_region": [
        "VEA", "IEFA", "EFA", "VWO", "IEMG", "EEM", "SCHF", "SCHE", "ACWI", "ACWX",
        "VXUS", "VEU", "VT", "BBJP", "EWJ", "DXJ", "FXI", "MCHI", "KWEB", "ASHR",
        "CQQQ", "INDA", "INDY", "EPI", "EWZ", "EWW", "EWC", "EWU", "EWG", "EWQ",
        "EWP", "EWI", "EWL", "EWD", "EWN", "EWA", "EWH", "EWS", "EWT", "EWY",
        "EWM", "THD", "EIDO", "EPHE", "TUR", "EZA", "KSA", "UAE", "ARGT", "ILF",
        "GXG", "ECH", "EWZS", "FM", "FRN", "VPL", "VGK", "IEUR", "IPAC", "EMXC",
    ],
    "fixed_income": [
        "AGG", "BND", "BNDX", "TLT", "IEF", "IEI", "SHY", "GOVT", "VGIT", "VGLT",
        "VGSH", "SCHO", "SCHR", "SPTL", "SPTI", "SPTS", "EDV", "ZROZ", "TLH", "VTIP",
        "TIP", "SCHP", "STIP", "LQD", "VCIT", "VCSH", "IGSB", "IGIB", "USIG", "SPSB",
        "HYG", "JNK", "USHY", "SHYG", "ANGL", "FALN", "SJNK", "HYLB", "BKLN", "SRLN",
        "EMB", "PCY", "EMLC", "VWOB", "MUB", "VTEB", "TFI", "HYD", "PFF", "PGX",
        "BIL", "SGOV", "SHV", "ICSH", "MINT", "JPST", "FLOT", "FLRN", "USFR", "TFLO",
    ],
    "commodity": [
        "GLD", "IAU", "GLDM", "SGOL", "BAR", "SLV", "SIVR", "PPLT", "PALL", "DBC",
        "PDBC", "GSG", "DJP", "COMT", "BCI", "USO", "BNO", "USL", "DBO", "UNG",
        "UGA", "CORN", "WEAT", "SOYB", "DBA", "CANE", "JO", "NIB", "CPER", "DBB",
        "GLTR", "WGMI", "FTGC", "COMB", "USCI", "PIT", "UCO", "BOIL", "KOLD", "SCO",
    ],
    "macro_alt": [
        "UUP", "UDN", "FXE", "FXY", "FXB", "FXF", "CYB", "BITO", "IBIT", "FBTC",
        "ETHE", "GBTC", "BITX", "VIXY", "VXX", "UVXY", "SVXY", "SVIX", "UVIX", "BTAL",
        "PSQ", "SH", "DOG", "RWM", "SQQQ", "SPXU", "SDS", "TBT", "TBF", "PST",
    ],
}

# ticker -> category (first occurrence wins; de-dupes tickers listed twice)
MASTER_ETFS: dict[str, str] = {}
for _cat, _tickers in _GROUPS.items():
    for _t in _tickers:
        MASTER_ETFS.setdefault(_t, _cat)


def master_tickers() -> list[str]:
    return list(MASTER_ETFS)


def category_of(ticker: str) -> str:
    return MASTER_ETFS.get(ticker, "equity")

"""Curated ~500 liquid large/mid-cap US stocks (no penny stocks).

A static, research-backed list (S&P 500 core + liquid mid-caps across sectors)
used by both the Strategy R&D and Accumulation tools. Static by design — robust
and fast vs. live fundamental screening, which is slow/flaky on free data.
"""

from __future__ import annotations

STOCKS_500 = [
    # Mega-cap tech / comms
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "ACN", "IBM", "INTC", "AMD", "QCOM", "TXN", "INTU",
    "NOW", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "NXPI",
    "ON", "MCHP", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "NET", "SNOW", "PLTR",
    "TEAM", "WDAY", "ADSK", "ANSS", "ROP", "CTSH", "IT", "GLW", "HPQ", "HPE",
    "DELL", "WDC", "STX", "KEYS", "TER", "ANET", "SMCI", "ARM", "APP", "ZM",
    "DOCU", "OKTA", "TWLO", "MDB", "HUBS", "DASH", "ABNB", "UBER", "LYFT", "SHOP",
    "SPOT", "RBLX", "U", "PINS", "SNAP", "BILL", "GTLB", "PATH",
    # Comm services / media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD", "FOXA", "PARA",
    "OMC", "IPG", "EA", "TTWO", "LYV", "MTCH", "NWSA",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "AXP", "BLK",
    "BX", "KKR", "APO", "ARES", "SPGI", "MCO", "ICE", "CME", "MSCI", "MMC",
    "AON", "AJG", "MET", "PRU", "AFL", "ALL", "TRV", "PGR", "CB", "HIG",
    "USB", "PNC", "TFC", "COF", "BK", "STT", "FITB", "MTB", "HBAN", "RF",
    "KEY", "CFG", "SYF", "DFS", "AIG", "V", "MA", "PYPL", "FIS", "FISV",
    "GPN", "NDAQ", "CBOE", "COIN", "HOOD", "SOFI", "NU",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "GILD", "ISRG", "VRTX", "REGN", "CVS", "CI", "ELV", "HUM",
    "CNC", "ZTS", "BSX", "SYK", "BDX", "EW", "DXCM", "IDXX", "IQV", "A",
    "MRNA", "BIIB", "HCA", "MCK", "COR", "CAH", "GEHC", "RMD", "WST", "MTD",
    "BAX", "ALGN", "PODD", "STE", "HOLX", "ZBH", "BMRN", "MEDP", "VEEV", "DGX",
    # Consumer discretionary
    "HD", "LOW", "MCD", "NKE", "SBUX", "BKNG", "CMG", "TJX", "ROST", "ORLY",
    "AZO", "YUM", "MAR", "HLT", "GM", "F", "APTV", "LULU", "DHI", "LEN",
    "NVR", "PHM", "TSCO", "ULTA", "BBY", "DG", "DLTR", "TGT", "EBAY", "ETSY",
    "GRMN", "POOL", "DPZ", "DRI", "EXPE", "RCL", "CCL", "NCLH", "WYNN", "LVS",
    "MGM", "CZR", "KMX", "GPC", "WSM", "DECK", "RL", "TPR", "CVNA", "CHWY",
    # Consumer staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "SYY", "KHC", "STZ", "KDP", "MNST", "KR", "HSY", "EL", "ADM",
    "MKC", "CHD", "CLX", "K", "HRL", "TSN", "CAG", "CPB", "SJM", "TAP",
    "BG", "DG", "KVUE",
    # Industrials
    "GE", "CAT", "HON", "UNP", "BA", "RTX", "LMT", "DE", "UPS", "ETN",
    "EMR", "ITW", "GD", "NOC", "CSX", "NSC", "FDX", "WM", "PH", "TT",
    "CMI", "GWW", "PCAR", "ROK", "CARR", "OTIS", "JCI", "AME", "FAST", "PWR",
    "URI", "DOV", "IR", "XYL", "FTV", "EFX", "VRSK", "RSG", "WAB", "GNRC",
    "PNR", "ALLE", "NDSN", "AOS", "SWK", "TXT", "HWM", "AXON", "ODFL", "JBHT",
    "CHRW", "EXPD", "LUV", "DAL", "UAL", "AAL",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "FANG", "DVN", "HAL", "BKR", "TRGP", "CTRA", "MRO",
    "APA", "EQT", "LNG", "MPLX", "ET",
    # Materials
    "LIN", "APD", "SHW", "FCX", "ECL", "NUE", "DOW", "DD", "NEM", "CTVA",
    "PPG", "VMC", "MLM", "ALB", "IFF", "CF", "MOS", "STLD", "PKG", "IP",
    "BALL", "AVY", "AMCR", "CE", "EMN", "LYB",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED",
    "WEC", "ES", "AEE", "DTE", "PPL", "FE", "ETR", "EIX", "CMS", "CNP",
    "ATO", "AES", "LNT", "NI", "PCG", "VST", "CEG", "NRG",
    # Real estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "VICI",
    "SBAC", "AVB", "EQR", "EXR", "INVH", "MAA", "ARE", "VTR", "ESS", "KIM",
    "UDR", "HST", "REG", "CPT", "BXP",
    # Other large/mid liquid names
    "BABA", "PDD", "JD", "NIO", "TSM", "ASML", "SE", "MELI", "SHOP", "SQ",
    "AFRM", "ROKU", "DKNG", "RIVN", "LCID", "MSTR", "MARA", "RIOT", "Z", "GME",
]

# de-dup while preserving order
STOCKS_500 = list(dict.fromkeys(STOCKS_500))

"""REGIMES indicator registry — every macro series the classifier uses, with
enough metadata to place it point-in-time and combine it correctly.

Verified live against the free `fredgraph.csv` endpoint during this feature's
exploration phase: 44 of 45 candidate series returned real data with no API
key. Two series that LOOK available are not usable for deep history and are
deliberately excluded here, with the substitute noted:

  * BAMLH0A0HYM2 (ICE BofA HY OAS) — the credit input CAS's existing
    risk-regime scalar uses — is license-truncated to ~3 years on the free
    endpoint. Long-history credit uses BAA10Y/AAA10Y (1983-86->) and
    NFCI/ANFCI (1971->) instead.
  * USSLIND (the St. Louis Fed's old leading index) stopped updating in
    2020-02 (discontinued). CFNAI substitutes — deeper history (1967->) and
    already a constructed composite.
  * VXTYN (CBOE/CBOT Treasury volatility) stopped updating 2020-05
    (discontinued). There is no free MOVE-index equivalent; the volatility
    dimension's rates-vol component is realized volatility on TLT (computed
    from yfinance prices, not FRED) and is labelled as a proxy everywhere
    it's shown, never presented as the real MOVE index.

FIELDS
  fred_id     — the FRED series ID (None for a derived/computed series).
  label       — human label for the UI.
  dimension   — one of regimes.DIMENSIONS.
  direction   — +1 or -1: the sign multiplier so that, after z-scoring, a
                POSITIVE value always means "this series pushes its
                dimension toward the positive pole" (growth up / inflation
                up / accommodative / expanding / easy credit / loose
                conditions / strong dollar / high volatility). Purely a
                sign convention for combining series with opposite economic
                polarity (e.g. UNRATE rising is bad for growth: direction=-1)
                — it is not a claim about what is "good".
  freq        — 'D' | 'W' | 'M' | 'Q' — native release cadence.
  lag_days    — best-effort estimate of real-world publication lag (the
                delay between a reference period ending and the data
                actually being published). Used to point-in-time-shift the
                series before it enters any historical classification (spec
                section 42's look-ahead-bias rule) — daily market series
                (VIX, yields, breakevens) publish same-day so lag=0; monthly
                survey/administrative series lag 5-45 days depending on the
                source's own release calendar. These are ESTIMATES, not a
                per-release calendar (see vintage.py for what a true
                point-in-time audit requires); documented as such wherever
                shown.
  transform   — 'level' | 'yoy' | 'mom_diff' | 'chg3m', applied AFTER the
                series has been resampled onto the point-in-time monthly
                grid (macro.py) — see that module for why transforms are
                applied post-resample rather than pre-resample.
  deep_history— True if the series has meaningful data before 1990 (informs
                UI messaging about historical coverage; does not change how
                the series is used).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    id: str                 # registry key (usually == fred_id, except derived series)
    fred_id: str | None
    label: str
    dimension: str
    direction: int
    freq: str
    lag_days: int
    transform: str
    deep_history: bool = False


REGISTRY: tuple[SeriesSpec, ...] = (
    # ------------------------------------------------------------- growth --
    SeriesSpec("CFNAI", "CFNAI", "Chicago Fed National Activity Index", "growth", +1, "M", 35, "level", True),
    SeriesSpec("INDPRO", "INDPRO", "Industrial Production", "growth", +1, "M", 15, "yoy", True),
    SeriesSpec("PAYEMS", "PAYEMS", "Nonfarm Payrolls (level)", "growth", +1, "M", 5, "mom_diff", True),
    SeriesSpec("UNRATE", "UNRATE", "Unemployment Rate", "growth", -1, "M", 5, "level", True),
    SeriesSpec("ICSA", "ICSA", "Initial Jobless Claims", "growth", -1, "W", 5, "level", True),
    SeriesSpec("RSAFS", "RSAFS", "Retail Sales", "growth", +1, "M", 15, "yoy", False),
    SeriesSpec("HOUST", "HOUST", "Housing Starts", "growth", +1, "M", 18, "yoy", True),
    SeriesSpec("PERMIT", "PERMIT", "Building Permits", "growth", +1, "M", 18, "yoy", True),
    SeriesSpec("NEWORDER", "NEWORDER", "Core Capital Goods New Orders", "growth", +1, "M", 35, "yoy", False),
    SeriesSpec("AWHMAN", "AWHMAN", "Avg Weekly Hours, Manufacturing", "growth", +1, "M", 5, "level", True),
    SeriesSpec("SAHMREALTIME", "SAHMREALTIME", "Sahm Rule Recession Indicator", "growth", -1, "M", 5, "level", True),

    # ----------------------------------------------------------- inflation --
    SeriesSpec("CPIAUCSL", "CPIAUCSL", "CPI (headline)", "inflation", +1, "M", 12, "yoy", True),
    SeriesSpec("CPILFESL", "CPILFESL", "Core CPI", "inflation", +1, "M", 12, "yoy", True),
    SeriesSpec("PCEPI", "PCEPI", "PCE Price Index", "inflation", +1, "M", 30, "yoy", True),
    SeriesSpec("PCEPILFE", "PCEPILFE", "Core PCE", "inflation", +1, "M", 30, "yoy", True),
    SeriesSpec("PPIFIS", "PPIFIS", "PPI Final Demand", "inflation", +1, "M", 12, "yoy", False),
    SeriesSpec("CES0500000003", "CES0500000003", "Avg Hourly Earnings (wage growth)", "inflation", +1, "M", 5, "yoy", False),
    SeriesSpec("T10YIE", "T10YIE", "10y Breakeven Inflation", "inflation", +1, "D", 0, "level", False),
    SeriesSpec("T5YIFR", "T5YIFR", "5y5y Forward Inflation Expectation", "inflation", +1, "D", 0, "level", False),
    SeriesSpec("EXPINF1YR", "EXPINF1YR", "Cleveland Fed 1y Inflation Expectation", "inflation", +1, "M", 5, "level", False),
    SeriesSpec("MICH", "MICH", "U Mich 1y Inflation Expectation", "inflation", +1, "M", 15, "level", True),
    SeriesSpec("CUSR0000SEHA", "CUSR0000SEHA", "CPI Shelter", "inflation", +1, "M", 12, "yoy", True),
    SeriesSpec("CUSR0000SASLE", "CUSR0000SASLE", "CPI Services less Energy", "inflation", +1, "M", 12, "yoy", True),

    # ------------------------------------------------------------ monetary --
    SeriesSpec("DFF", "DFF", "Effective Fed Funds Rate", "monetary", -1, "D", 0, "level", True),
    SeriesSpec("REAL_FFR", None, "Real Fed Funds Rate (DFF - core PCE YoY)", "monetary", -1, "M", 30, "level", False),
    SeriesSpec("T10Y3M", "T10Y3M", "10y-3m Curve", "monetary", +1, "D", 0, "level", False),

    # ----------------------------------------------------------- liquidity --
    SeriesSpec("NET_LIQ", None, "Fed Net Liquidity (WALCL - RRP - TGA)", "liquidity", +1, "W", 0, "chg3m", False),
    SeriesSpec("M2SL", "M2SL", "M2 Money Supply", "liquidity", +1, "M", 25, "yoy", True),

    # -------------------------------------------------------------- credit --
    SeriesSpec("BAA10Y", "BAA10Y", "Baa Corporate - 10y Treasury Spread", "credit", -1, "D", 0, "level", True),
    SeriesSpec("AAA10Y", "AAA10Y", "Aaa Corporate - 10y Treasury Spread", "credit", -1, "D", 0, "level", True),
    SeriesSpec("TOTCI", "TOTCI", "Commercial & Industrial Loans", "credit", +1, "W", 10, "yoy", True),
    SeriesSpec("DRTSCILM", "DRTSCILM", "% Banks Tightening C&I Loan Standards (SLOOS)", "credit", -1, "Q", 30, "level", False),
    SeriesSpec("BUSLOANS", "BUSLOANS", "Commercial & Industrial Loans (H.8)", "credit", +1, "M", 20, "yoy", True),

    # ----------------------------------------------------- financial cond. --
    SeriesSpec("NFCI", "NFCI", "Chicago Fed Financial Conditions Index", "financial_conditions", -1, "W", 5, "level", True),
    SeriesSpec("ANFCI", "ANFCI", "Adjusted Financial Conditions Index", "financial_conditions", -1, "W", 5, "level", True),
    SeriesSpec("STLFSI4", "STLFSI4", "St. Louis Fed Financial Stress Index", "financial_conditions", -1, "W", 5, "level", False),
    SeriesSpec("VIXCLS", "VIXCLS", "VIX", "financial_conditions", -1, "D", 0, "level", True),

    # -------------------------------------------------------------- dollar --
    SeriesSpec("DTWEXBGS", "DTWEXBGS", "Broad Trade-Weighted Dollar", "dollar", +1, "D", 0, "level", False),
    SeriesSpec("DTWEXAFEGS", "DTWEXAFEGS", "Advanced Foreign Economies Dollar", "dollar", +1, "D", 0, "level", False),
    SeriesSpec("DEXUSEU", "DEXUSEU", "USD per EUR", "dollar", -1, "D", 0, "level", False),
    SeriesSpec("DEXJPUS", "DEXJPUS", "JPY per USD", "dollar", +1, "D", 0, "level", True),

    # --------------------------------------------------------- volatility --
    SeriesSpec("OVXCLS", "OVXCLS", "Oil Volatility (OVX)", "volatility", +1, "D", 0, "level", False),
    SeriesSpec("GVZCLS", "GVZCLS", "Gold Volatility (GVZ)", "volatility", +1, "D", 0, "level", False),
    # Realized Treasury-bond vol (yfinance TLT, computed in macro.py, no
    # fred_id) — a MOVE-index PROXY: VXTYN, the real free Treasury-vol
    # index, was discontinued 2020-05 (see this module's docstring). Labelled
    # as a proxy everywhere it's shown, never presented as the real index.
    SeriesSpec("TLT_RVOL", None, "TLT Realized Volatility (MOVE proxy)", "volatility", +1, "D", 0, "level", False),
)

BY_ID = {s.id: s for s in REGISTRY}
BY_DIMENSION: dict[str, tuple[SeriesSpec, ...]] = {}
for _s in REGISTRY:
    BY_DIMENSION.setdefault(_s.dimension, []).append(_s)
BY_DIMENSION = {k: tuple(v) for k, v in BY_DIMENSION.items()}

# Direct FRED fetches only — excludes derived (REAL_FFR, NET_LIQ) series and
# the yfinance-only TLT realized-vol proxy.
FRED_IDS = sorted({s.fred_id for s in REGISTRY if s.fred_id})

# The raw FRED inputs a derived series needs, so macro.py knows what to fetch
# even though DERIVED itself has no fred_id.
DERIVED_INPUTS = {
    "REAL_FFR": ("DFF", "PCEPILFE"),
    "NET_LIQ": ("WALCL", "RRPONTSYD", "WTREGEN"),
}
# WALCL/RRPONTSYD/WTREGEN are fetched solely as NET_LIQ's inputs — they are
# not independently registered SeriesSpecs (no standalone dimension role), so
# their freq/lag live here rather than as a full SeriesSpec. Units, for the
# unit-alignment NET_LIQ needs (see macro.py): WALCL and WTREGEN are $
# millions (H.4.1 release); RRPONTSYD is $ billions (the Fed's own daily
# reverse-repo release) — NOT the same units as each other.
EXTRA_META = {
    "WALCL": {"freq": "W", "lag_days": 2, "units": "millions"},
    "RRPONTSYD": {"freq": "W", "lag_days": 1, "units": "billions"},
    "WTREGEN": {"freq": "W", "lag_days": 2, "units": "millions"},
}
EXTRA_FRED_IDS = sorted({fid for ids in DERIVED_INPUTS.values() for fid in ids} - set(FRED_IDS))
ALL_FRED_IDS = sorted(set(FRED_IDS) | set(EXTRA_FRED_IDS))


def freq_of(fred_id: str) -> str:
    spec = BY_ID.get(fred_id)
    if spec is not None:
        return spec.freq
    return EXTRA_META.get(fred_id, {}).get("freq", "M")


def lag_of(fred_id: str) -> int:
    spec = BY_ID.get(fred_id)
    if spec is not None:
        return spec.lag_days
    return EXTRA_META.get(fred_id, {}).get("lag_days", 20)

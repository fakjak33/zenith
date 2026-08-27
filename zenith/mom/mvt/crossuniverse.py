"""Phase 3 — cross-universe analysis: is an equity's apparent trend broad
(shared with its whole sector) or idiosyncratic (particular to that name)?

Spec section 25's ask, scoped to the practical, well-supported comparison:
equity vs its own GICS sector's SPDR sector ETF (both already scored by the
SAME mvt engine on the SAME -20..+20 scale, so the gap is directly
comparable without any re-normalization). Broader combinations (equity vs
industry ETF, commodity vs related equities) are NOT attempted here --
sector is the one mapping this repo already carries a reliable GICS label
for (mom's own metadata cache); industry-level and commodity-linkage
mappings would need a new, unvalidated taxonomy this module doesn't invent.
"""

from __future__ import annotations

# yfinance's `.info` sector strings (what mom.universe.refresh_metadata
# actually stores -- verified against a live scores_latest.json before
# writing this) -> the SPDR sector ETF ticker. NOT the same vocabulary as
# cas.universe.SECTORS (GICS/Wikipedia-style labels like "Cons.
# Discretionary") -- that dict's keys don't string-match yfinance's, so a
# dedicated mapping lives here rather than reusing it incorrectly.
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# A gap this large (on the shared -20..+20 scale) between an equity's own
# score and its sector ETF's score is called out as "idiosyncratic" in the
# UI -- a fixed, documented threshold (not fit to any outcome), matching
# this repo's general preference for a simple, stated rule over a tuned one.
IDIOSYNCRATIC_GAP_THRESHOLD = 8.0


def build_comparison(equity_rows: list[dict], etf_rows: list[dict], sector_by_ticker: dict[str, str],
                     gap_threshold: float = IDIOSYNCRATIC_GAP_THRESHOLD) -> list[dict]:
    """`equity_rows`/`etf_rows`: mvt score rows (need ticker + normalized_
    score). `sector_by_ticker`: ticker -> yfinance-style sector string
    (mom's own committed meta.json). Returns one row per equity that has
    both a resolvable sector-ETF mapping and a scored sector ETF -- an
    equity in a sector with no mapped ETF, or whose sector ETF wasn't
    scored that day, is simply absent, not padded with a fabricated gap."""
    etf_by_ticker = {r["ticker"]: r for r in etf_rows if r.get("normalized_score") is not None}
    out = []
    for r in equity_rows:
        if r.get("normalized_score") is None:
            continue
        sector = sector_by_ticker.get(r["ticker"])
        etf_ticker = SECTOR_TO_ETF.get(sector) if sector else None
        if not etf_ticker or etf_ticker not in etf_by_ticker:
            continue
        etf_row = etf_by_ticker[etf_ticker]
        gap = r["normalized_score"] - etf_row["normalized_score"]
        out.append({
            "ticker": r["ticker"], "sector": sector, "sector_etf": etf_ticker,
            "equity_score": r["normalized_score"], "sector_etf_score": etf_row["normalized_score"],
            "gap": round(gap, 3),
            "classification": "idiosyncratic" if abs(gap) >= gap_threshold else "broad/systemic",
        })
    return out


def sector_breadth(comparison: list[dict]) -> dict:
    """Per sector: how many of its names are trending broadly WITH the
    sector ETF vs idiosyncratically apart from it, and the sector ETF's own
    score -- answers "is small-cap-style strength broad or isolated" at the
    sector level (section 24's example question, generalized)."""
    by_sector: dict[str, dict] = {}
    for row in comparison:
        s = row["sector"]
        bucket = by_sector.setdefault(s, {"sector_etf": row["sector_etf"],
                                          "sector_etf_score": row["sector_etf_score"],
                                          "n": 0, "n_idiosyncratic": 0, "n_broad": 0})
        bucket["n"] += 1
        if row["classification"] == "idiosyncratic":
            bucket["n_idiosyncratic"] += 1
        else:
            bucket["n_broad"] += 1
    for s, b in by_sector.items():
        b["pct_idiosyncratic"] = round(b["n_idiosyncratic"] / b["n"], 4) if b["n"] else None
    return by_sector

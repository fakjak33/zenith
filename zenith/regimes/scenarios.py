"""REGIMES "What If?" scenario engine (spec section 25) — CONTINGENCY
PLANNING, explicitly not a prediction (spec's own words: "these should not
be presented as predictions... they are contingency-planning scenarios").

Generalizes `cas/contingency.py`'s trigger/proactive/reactive shape (the
existing CAS Playbook tab): instead of a fixed set of triggers evaluated
against live data, these are user-selectable hypotheticals mapped to an
IMPLIED axis shift, with the historical performance-by-regime data
(performance.py) as the grounded "what has worked" answer where the
scenario cleanly maps to the growth/inflation quadrant.

Two scenario kinds, both honestly labelled:
  * QUADRANT scenarios (inflation accelerates/decelerates, growth
    accelerates/decelerates) flip growth_rising or infl_rising and pull
    REAL regime-conditional performance numbers for the resulting quadrant.
  * DIMENSION scenarios (dollar falls 10%, credit spreads widen, Fed
    restarts QE, oil rises 50%) push a SECONDARY dimension toward a pole;
    since performance.py only conditions on the top-level quadrant (building
    the same historical-timeline infrastructure per secondary dimension is
    out of scope here), these are explicitly labelled "qualitative — no
    regime-conditional backtest for this dimension" rather than presenting
    a number that doesn't exist.
"""

from __future__ import annotations

from . import REGIME_LABELS

QUADRANT_SCENARIOS = (
    {"id": "inflation_accelerates", "name": "Inflation Accelerates",
     "description": "Inflation indicators broadly turn higher — a resurgence in prints, "
                    "wage growth, or breakevens.", "axis": "inflation", "flip_to": True},
    {"id": "inflation_decelerates", "name": "Inflation Decelerates",
     "description": "Inflation indicators broadly cool — disinflation across CPI/PCE, "
                    "breakevens, wage growth.", "axis": "inflation", "flip_to": False},
    {"id": "growth_accelerates", "name": "Growth Accelerates",
     "description": "Growth indicators broadly strengthen — payrolls, industrial "
                    "production, retail sales all firm up.", "axis": "growth", "flip_to": True},
    {"id": "growth_decelerates_sharply", "name": "Growth Decelerates Sharply",
     "description": "Unemployment rises sharply, claims spike, growth indicators broadly "
                    "roll over — a recession-risk scenario.", "axis": "growth", "flip_to": False},
)

DIMENSION_SCENARIOS = (
    {"id": "dollar_falls_10pct", "name": "Dollar Falls 10%", "dimension": "dollar", "flip_to": False,
     "relevant_tickers": {"UUP": "US Dollar (inverse exposure)", "GLD": "Gold", "EEM": "Emerging Markets",
                          "EFA": "International Developed"}},
    {"id": "credit_spreads_widen", "name": "Credit Spreads Widen Dramatically", "dimension": "credit",
     "flip_to": False, "relevant_tickers": {"HYG": "High Yield Credit", "LQD": "IG Credit",
                                            "TLT": "Long Treasuries (flight to quality)"}},
    {"id": "fed_restarts_qe", "name": "Fed Restarts QE", "dimension": "liquidity", "flip_to": True,
     "relevant_tickers": {"TLT": "Long Treasuries", "GLD": "Gold", "QQQ": "Nasdaq 100 (duration-sensitive)"}},
    {"id": "oil_rises_50pct", "name": "Oil Rises 50%", "dimension": "inflation", "flip_to": True,
     "relevant_tickers": {"USO": "Crude Oil", "XLE": "Energy Sector", "DBC": "Broad Commodities"},
     "note": "Modeled as an inflation-axis shock (a large oil move is a classic inflation-side "
             "catalyst) rather than a distinct dimension — see the Inflation Accelerates scenario "
             "for the grounded quadrant-level read; this entry adds the energy-specific tickers."},
    {"id": "ten_year_yield_to_6pct", "name": "10-Year Yield Rises to 6%", "dimension": "monetary",
     "flip_to": False, "relevant_tickers": {"TLT": "Long Treasuries", "XLF": "Financials",
                                            "XLRE": "Real Estate (rate-sensitive)"}},
)


def _implied_quadrant(current_growth_rising: bool | None, current_infl_rising: bool | None,
                      axis: str, flip_to: bool) -> str | None:
    g = flip_to if axis == "growth" else current_growth_rising
    i = flip_to if axis == "inflation" else current_infl_rising
    if g is None or i is None:
        return None
    return REGIME_LABELS.get((g, i))


def evaluate_quadrant_scenario(scenario: dict, current_growth_rising: bool | None,
                               current_infl_rising: bool | None, perf_table: dict) -> dict:
    target = _implied_quadrant(current_growth_rising, current_infl_rising, scenario["axis"], scenario["flip_to"])
    beneficiaries, losers = [], []
    if target:
        rows = []
        for ticker, data in perf_table.items():
            stats = data.get("by_regime", {}).get(target)
            if stats:
                rows.append({"ticker": ticker, "label": data["label"], **stats})
        rows.sort(key=lambda r: r["avg_return"], reverse=True)
        beneficiaries = rows[:5]
        losers = list(reversed(rows[-5:])) if len(rows) >= 5 else []
    return {**scenario, "implied_regime": target,
           "historical_beneficiaries": beneficiaries, "historical_losers": losers,
           "grounded": bool(target and beneficiaries)}


def evaluate_dimension_scenario(scenario: dict) -> dict:
    return {**scenario, "grounded": False,
           "caveat": "Qualitative — no regime-conditional historical backtest exists for this "
                     "secondary dimension yet. Relevant tickers are listed for context, not "
                     "backed by a performance table the way quadrant scenarios are."}


def build(current_growth_rising: bool | None, current_infl_rising: bool | None,
         perf_table: dict) -> dict:
    quadrant = [evaluate_quadrant_scenario(s, current_growth_rising, current_infl_rising, perf_table)
               for s in QUADRANT_SCENARIOS]
    dimension = [evaluate_dimension_scenario(s) for s in DIMENSION_SCENARIOS]
    return {"quadrant_scenarios": quadrant, "dimension_scenarios": dimension}

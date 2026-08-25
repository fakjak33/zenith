"""IDEAS macro/regime: wraps CAS's existing risk-on/off classifier.

cas.signals.regime.compute() already builds a risk score from FRED macro
(VIX/HY-OAS percentiles, the 10y-2y curve) plus SPY's trend -- no API key, no
new data source. This module is a thin adapter: it fetches the same two
inputs CAS's own nightly job fetches (cas.sources.fred, cas.sources.prices),
calls the existing classifier, and turns the result into a per-stock macro
GROUP score by tilting the market-wide risk score by each stock's own beta
(spec section 23: "the system should adapt its scoring" by regime, not just
report the regime as text). This is coarse and three-state by design (spec
section 21's anti-overfitting note) -- it is a context layer, not a fitted
regime model.
"""

from __future__ import annotations

from ..cas.sources import fred as cas_fred
from ..cas.sources import prices as cas_prices
from ..cas.signals import regime as cas_regime


def compute_regime() -> dict:
    """One call per run -- the same market-wide read for every security
    today. Returns the cas.signals.regime summary dict (label, risk_score,
    vix_percentile, hy_oas_percentile, curve_10y2y, spy_trend)."""
    fred_data, _fred_status = cas_fred.get_series(list(cas_fred.DEFAULT_SERIES))
    px, _px_status = cas_prices.get_history(["SPY"], period="2y")
    _signals, summary = cas_regime.compute(fred_data, px)
    return summary


def macro_score(regime_summary: dict, beta: float | None) -> tuple[float, dict]:
    """[-1,1] macro tilt for one security: the market-wide risk_score,
    amplified for high-beta names and dampened for low-beta/defensive names.
    beta=None (no fundamentals coverage yet) falls back to a neutral 1.0x --
    never fabricated, just untilted."""
    risk = float(regime_summary.get("risk_score", 0.0) or 0.0)
    mult = 1.0
    if beta is not None:
        try:
            mult = max(0.5, min(2.0, float(beta)))
        except (TypeError, ValueError):
            mult = 1.0
    score = max(-1.0, min(1.0, risk * mult))
    return score, {"regime_label": regime_summary.get("label"), "risk_score": risk,
                   "beta_used": beta, "beta_multiplier": round(mult, 2)}

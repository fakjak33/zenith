"""Regime segment — market-wide risk state from macro + breadth.

Feeds the contingency playbook and the BCT "physical-equivalence" layer. Uses
FRED macro (VIX, curve, HY spreads, USD) plus SPY trend to classify a risk-on /
risk-off state and emit a market-level signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal


def _last_val(series: list[dict]) -> float:
    return float(series[-1]["value"]) if series else float("nan")


def _percentile(series: list[dict], lookback: int = 252) -> float:
    s = pd.Series([p["value"] for p in series]).tail(lookback)
    if len(s) < 20:
        return 0.5
    return float((s <= s.iloc[-1]).mean())


def compute(fred: dict[str, list], data: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    """Returns (signals, regime_summary)."""
    vix_pct = _percentile(fred.get("VIXCLS", []))
    hy_pct = _percentile(fred.get("BAMLH0A0HYM2", []))
    curve = _last_val(fred.get("T10Y2Y", []))

    spy = data.get("SPY", {}).get("close") if "SPY" in data else None
    spy_trend = 0.0
    if spy is not None and len(spy) > 200:
        spy_trend = ind.clip1((spy.iloc[-1] / ind.sma(spy, 200).iloc[-1] - 1.0) * 8)

    # risk score: high VIX & HY percentiles = risk-off (negative)
    stress = np.nanmean([vix_pct, hy_pct])
    risk = ind.clip1(0.6 * spy_trend - 0.8 * (stress - 0.5) * 2)

    if risk > 0.2:
        label = "risk-on"
    elif risk < -0.2:
        label = "risk-off"
    else:
        label = "neutral / transition"

    summary = {
        "label": label, "risk_score": round(risk, 3),
        "vix_percentile": round(vix_pct, 3), "hy_oas_percentile": round(hy_pct, 3),
        "curve_10y2y": curve, "spy_trend": round(spy_trend, 3),
    }
    sig = [make_signal("MARKET", "regime", "risk_regime", risk, asset_class="market",
                       horizon="weeks", source="fred+yfinance",
                       rationale=f"{label}: VIX pct {vix_pct:.0%}, HY pct {hy_pct:.0%}, "
                                 f"curve {curve:+.2f}, SPY trend {spy_trend:+.2f}")]
    return sig, summary

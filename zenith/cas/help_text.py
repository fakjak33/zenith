"""Central glossary of plain-English explanations for CAS / Factor-Rotation terms.

Surfaced as "?" hover tooltips in the viewer (via ui_theme.help_badge and the
native Streamlit ``help=`` arg) so nothing on screen is unexplained jargon.
"""

from __future__ import annotations

HELP: dict[str, str] = {
    # --- core CAS concepts ---
    "cas": ("Complex Adaptive Systems monitor: a multi-model dashboard that turns free market data "
            "into buy / neutral / sell signals across several 'segments', then shows where those "
            "signals agree. Decision-support, not investment advice."),
    "segment": ("A family of related models. Each segment (Strategies, Themes, Factor Rotation, Flows, "
                "Regime, Rebalance) scores assets independently; the Overview shows where they overlap."),
    "signal": ("A model's view on an asset, from -1 (strong sell) to +1 (strong buy). 0 means neutral / "
               "no edge. All models use this same scale so they can be compared and combined."),
    "state": ("The signal bucketed into buy / neutral / sell. Buy when signal ≥ +0.15, sell when "
              "≤ -0.15, otherwise neutral."),
    "horizon": ("The time frame the signal is meant for: days, weeks or months. Trend/momentum signals "
                "are months; mean-reversion signals are days."),
    "confidence": ("How much weight to give the signal (low / medium / high), based on how much history "
                   "and corroboration it has — not a probability."),
    "regime": ("The current market 'weather' (e.g. risk-on / risk-off) inferred from macro data: the "
               "yield curve, credit spreads, the dollar and volatility."),
    # --- overlap / consensus ---
    "overlap": ("Ranks assets by how many models agree, and how strongly. An asset flashing buys across "
                "several segments is higher-conviction than one buy in isolation."),
    "consensus": ("An ensemble score blending all of an asset's signals. 'Behavioral' = crowd/positioning "
                  "& mean-reversion extremes; 'Physical' = trend/regime/volatility; 'Entanglement' = "
                  "agreement across time horizons."),
    "behavioral": "Crowd-driven signals: positioning extremes and mean-reversion (where the herd dominates).",
    "physical": "Price-'physics' signals: trend, regime and volatility state.",
    "entanglement": "How well an asset's signals agree across short, medium and long horizons (1 = all aligned).",
    "signal_variety": ("How many distinct models weighed in (Ashby's Law analogue) — more independent "
                       "models agreeing = more reliable."),
    "entropy": "Disagreement among the signals (0 = all agree, 1 = evenly split). High entropy = low conviction.",
    # --- factor rotation momentum ---
    "frm": ("Factor Rotation Momentum: trades trend + rotation across equity style factors (Value, "
            "Momentum, Quality, Low-Vol, Size, Dividend, Growth, Buyback, Multi), industries, and a few "
            "region-specific sectors — all via US-listed ETFs. Based on Gupta & Kelly (2019) 'Factor "
            "Momentum Everywhere' and the Man Group style-trend strategy."),
    "frm_ts_mom": ("Time-series momentum: each ETF timed on its OWN recent return — a vol-scaled blend of "
                   "1/3/6/12-month trailing returns (the 12m leg skips the latest month). Positive = the "
                   "factor has been trending up, which tends to persist."),
    "frm_cs_region": ("Cross-sectional rank within an ETF's peer group in its region (e.g. Value vs the "
                      "other US styles). +1 = strongest momentum in its peer set, -1 = weakest."),
    "frm_cs_peer": ("Cross-sectional rank of an ETF versus the SAME factor across regions (e.g. US Value "
                    "vs Intl Value vs EM Value)."),
    "frm_composite": "Blended factor-rotation signal: 0.6×time-series + 0.3×region rank + 0.1×peer rank.",
    "frm_group": "Which bucket the ETF belongs to: a style factor, an industry, or a region-specific sector.",
    # --- backtest ---
    "backtest": ("Validation of the factor-momentum effect on decades of academic factor returns (Ken "
                 "French regional factors + AQR QMJ/BAB) — far more history than the ETFs themselves have."),
    "ar1": ("AR(1) autocorrelation: does a factor's return last period predict its return this period? "
            "Positive = momentum/persistence. Gupta & Kelly found 59 of 65 factors positive, 49 significant."),
    "t_stat": "Statistical significance of the AR(1) estimate. |t| > ~2 means it's unlikely to be chance.",
    "ts_sharpe": "Return-per-unit-risk (annualised) of timing that factor with time-series momentum.",
    # --- history / hit rate ---
    "history": "How each signal has evolved week over week — so you can see trends and turning points.",
    "hitrate": ("How often a signal's direction matched the asset's ACTUAL forward return, measured at "
                "1, 3, 6 and 12 months. 50% = coin flip; above 50% = the signal has had predictive value. "
                "Past hit-rate is not a guarantee."),
    # --- flows / rebalance ---
    "flows": ("Positioning & flow pressure: futures positioning (CFTC Commitments of Traders) and a "
              "volume-based proxy. Extremes often precede reversals."),
    "rebalance": ("Recurring calendar dates that drive flows: options expiration, index rebalances "
                  "(S&P, Russell), and quarter-end."),
    "themes": "Which sectors / industries / themes are gaining or losing relative strength versus the S&P 500.",
}


def get(key: str) -> str:
    return HELP.get(key, "")

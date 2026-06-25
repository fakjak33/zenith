"""CAS — Complex Adaptive Systems market-dynamics monitor for Zenith.

A multifactor monitor that emits buy/neutral/sell signals across segments
(strategies, flows & positioning, themes, rebalance, regime), aggregates them
into a Behavioral-Consensus-Theory-inspired ensemble, and surfaces where signals
*overlap* across factors. Free data only; flow/dealer/options dynamics that have
no free feed are implemented as labelled proxies or manual inputs.

Decision-support heuristics, NOT investment advice.
"""

DISCLAIMER = (
    "CAS signals are heuristic, decision-support analytics built from free / "
    "proxy data — not investment advice. Flow, dealer and options dynamics "
    "without a free feed are shown as labelled proxies or manual inputs."
)

"""Behavioral Consensus Theory (BCT) — inspired ensemble layer.

This is a pragmatic, honest interpretation of the BCT / High-Entropy-Intelligence
framing the user provided: it is an *ensemble consensus analytic*, not a literal
quantum-mechanical model. The mapping we implement:

  * Psychological Causality (behavioral)  -> positioning/flow & mean-reversion
    extremes (where crowd behavior dominates).
  * Physical Equivalence (technical)      -> trend / regime / volatility state
    signals (price 'physics').
  * Spatiotemporal Entanglement           -> agreement of an asset's signals
    across horizons (days/weeks/months).

Ashby's Law of Requisite Variety is honored by an explicit *signal variety /
entropy* metric: the more diverse, independent signals that agree, the more the
controller's internal variety matches the market's — and the higher our
confidence. We surface that variety alongside the composite score rather than
claiming the physics literally.
"""

from __future__ import annotations

import math
from collections import defaultdict

BEHAVIORAL = {"mean_reversion", "rsi_reversion", "cot_leveraged_positioning",
              "cot_managed_money_positioning", "etf_volume_flow_proxy"}
PHYSICAL = {"momentum", "ma_single_200", "ma_cross_50_200", "multi_ma",
            "donchian_breakout", "vol_regime", "trend_following", "risk_regime",
            "relative_strength"}

_HORIZON_AXIS = {"intraday": 0, "days": 0, "weeks": 1, "months": 2}


def _entropy(states: list[str]) -> float:
    """Shannon entropy (0..1) over buy/neutral/sell — high = disagreement."""
    if not states:
        return 0.0
    n = len(states)
    h = 0.0
    for st in ("buy", "neutral", "sell"):
        p = states.count(st) / n
        if p > 0:
            h -= p * math.log(p, 3)        # base-3 so max == 1
    return h


def build(signals: list[dict]) -> list[dict]:
    """Aggregate per-asset signals into a BCT-inspired consensus record."""
    by_asset: dict[str, list[dict]] = defaultdict(list)
    for s in signals:
        by_asset[s["asset"]].append(s)

    out: list[dict] = []
    for asset, recs in by_asset.items():
        beh = [r["signal"] for r in recs if r["family"] in BEHAVIORAL]
        phy = [r["signal"] for r in recs if r["family"] in PHYSICAL]

        # spatiotemporal: mean signal per horizon axis, then their agreement
        axis_vals: dict[int, list[float]] = defaultdict(list)
        for r in recs:
            axis_vals[_HORIZON_AXIS.get(r["horizon"], 0)].append(r["signal"])
        axis_means = [sum(v) / len(v) for v in axis_vals.values() if v]
        entangle = (sum(axis_means) / len(axis_means)) if axis_means else 0.0
        # agreement across horizons (1 = all same direction)
        if len(axis_means) > 1:
            signs = [1 if a > 0.05 else (-1 if a < -0.05 else 0) for a in axis_means]
            agree = abs(sum(signs)) / len(signs)
        else:
            agree = 0.5

        beh_c = sum(beh) / len(beh) if beh else 0.0
        phy_c = sum(phy) / len(phy) if phy else 0.0
        overall = sum(r["signal"] for r in recs) / len(recs)
        # base on the overall mean so every signal counts, then tilt by the three
        # BCT effects (behavioral / physical / spatiotemporal entanglement)
        composite = round(0.25 * overall + 0.25 * beh_c + 0.3 * phy_c
                          + 0.2 * entangle, 4)

        states = [r["state"] for r in recs]
        variety = len({r["family"] for r in recs})       # distinct models = Ashby variety
        entropy = round(_entropy(states), 3)
        # confidence: more variety + lower entropy + cross-horizon agreement
        conf = round(min(1.0, (variety / 10.0) * (1 - entropy) * (0.5 + agree / 2)), 3)

        out.append({
            "asset": asset,
            "asset_class": recs[0]["asset_class"],
            "composite": composite,
            "state": "buy" if composite >= 0.15 else ("sell" if composite <= -0.15 else "neutral"),
            "behavioral": round(beh_c, 4),
            "physical": round(phy_c, 4),
            "entanglement": round(entangle, 4),
            "horizon_agreement": round(agree, 3),
            "signal_variety": variety,
            "entropy": entropy,
            "confidence": conf,
            "n_signals": len(recs),
        })
    out.sort(key=lambda r: r["composite"], reverse=True)
    return out

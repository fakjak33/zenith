"""Overlap view — where do buy/sell signals align across factors?

Builds the assets x signal-family matrix and ranks assets by the *count and
strength* of aligned signals across segments. This is the payoff of the shared
schema: an asset flashing buys across strategies + flows + themes is a higher-
conviction idea than one buy in isolation.
"""

from __future__ import annotations

from collections import defaultdict


def build(signals: list[dict], min_abs: float = 0.15) -> dict:
    """Return {matrix, ranked}.

    matrix: asset -> {family -> signal}
    ranked: list of {asset, asset_class, buy_count, sell_count, net, families,
                     segments, mean_strength} sorted by conviction.
    """
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    segs: dict[str, set] = defaultdict(set)
    for s in signals:
        matrix[s["asset"]][s["family"]] = s["signal"]
        segs[s["asset"]].add(s["segment"])

    ranked = []
    for asset, fam in matrix.items():
        buys = [v for v in fam.values() if v >= min_abs]
        sells = [v for v in fam.values() if v <= -min_abs]
        strengths = [abs(v) for v in fam.values() if abs(v) >= min_abs]
        net = round(sum(fam.values()) / len(fam), 4) if fam else 0.0
        any_rec = next(s for s in signals if s["asset"] == asset)
        ranked.append({
            "asset": asset,
            "asset_class": any_rec["asset_class"],
            "buy_count": len(buys),
            "sell_count": len(sells),
            "net": net,
            "families": len(fam),
            "segments": sorted(segs[asset]),
            "n_segments": len(segs[asset]),
            "mean_strength": round(sum(strengths) / len(strengths), 4) if strengths else 0.0,
        })

    # conviction = net direction weighted by how many segments + families agree
    ranked.sort(key=lambda r: (abs(r["net"]) * r["n_segments"] * max(r["buy_count"],
                               r["sell_count"])), reverse=True)
    return {"matrix": matrix, "ranked": ranked}

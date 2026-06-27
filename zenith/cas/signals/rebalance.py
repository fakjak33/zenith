"""Rebalance segment — proximity flags to upcoming market-structure dates.

Converts the computed calendar into signals: an imminent quarter-end / OPEX /
index rebalance raises a near-term "flow event" flag. These are *attention*
signals (horizon-aware) rather than directional buy/sell calls, so they enter the
overlap view as low-magnitude context.
"""

from __future__ import annotations

from ..schema import make_signal
from ..sources import calendar as cal


def compute(events: list[dict] | None = None) -> list[dict]:
    events = events if events is not None else cal.upcoming()
    out: list[dict] = []
    for e in events:
        du = e.get("days_until", 999)
        if du > 21:
            continue
        # closer event = stronger attention flag (magnitude only, neutral sign)
        mag = max(0.0, 1.0 - du / 21.0) * 0.5
        out.append(make_signal(
            e["event"], "rebalance", e.get("kind", "event"), mag,
            asset_class="market", horizon="weeks", source="calendar", confidence="medium",
            rationale=f"{e['event']} in {du}d — {e.get('note', '')}"))
    return out

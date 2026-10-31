"""STRATEGY INDICES — which option strategies the market is paying right now.

Cboe publishes free daily levels for its options-strategy benchmarks (put-
writing, buy-writing, iron condors and butterflies, collars, put protection,
risk reversals, VRP harvesting). Tracking them side by side, with the same
trend overlays a CTA would use on any return stream (time-series momentum,
cross-sectional momentum, Donchian breakouts, EMA-crossover trend), reads the
ENVIRONMENT: is short volatility being paid, are tails paying, is the tape
range-bound (condors over puts) or trending — plus the volatility context
(VIX term structure, VVIX, SKEW, implied correlation, dispersion) and S&P 500
breadth (% of members above their 20/50/200-day averages, the S5TW / S5FI /
S5TH family).

Every index links to its official Cboe methodology document (verified on each
build). Not investment advice: indices are hypothetical strategies without
costs, and past regimes do not repeat on schedule.
"""

from __future__ import annotations

import json
import math

from ..config import STRATIDX_FILES

DISCLAIMER = ("Cboe strategy benchmarks are hypothetical, rules-based strategies without trading costs or "
              "taxes; levels are Cboe's published closes. Trend overlays describe what HAS worked — they "
              "are context for choosing a strategy, not a forecast. Breadth uses today's S&P 500 members "
              "(survivorship). Not investment advice.")


def _scrub(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def load(name: str, default=None):
    path = STRATIDX_FILES[name]
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(name: str, obj) -> None:
    path = STRATIDX_FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(_scrub(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)

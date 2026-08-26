"""REGIMES transition probabilities — EMPIRICAL BASE RATES ONLY, read straight
off the reconstructed timeline (per the user's explicit choice over a fitted
Markov/HMM model: "nothing is fitted, every number is a countable frequency,
and the explanation IS the method").

For every month the DECLARED regime was R, look forward H months and check
what the declared regime was then. The fraction that ended up in each
destination regime, with `n` (the count backing it) ALWAYS carried alongside
the percentage — spec section 8: "not a precise forecast... show which
signals are driving the probability" (here, the honest answer to "what's
driving it" is "n historical months that started in the same regime").

Momentum conditioning (spec section 8's worked example: "with growth
momentum in the bottom tercile...") is a SINGLE binary split (momentum
score >= 0 vs < 0 at the start month, from momentum.momentum_series), not
full terciles — a genuine simplification, documented here: terciles would
cut the already-limited 36 years of monthly history into thirds-of-thirds
per regime, and small-n cells are exactly what config.MIN_N_FOR_PROBABILITY
exists to refuse to report rather than present a spuriously precise number.
"""

from __future__ import annotations

import pandas as pd

from . import REGIME_LABELS

HORIZONS_MONTHS = {30: 1, 90: 3, 180: 6, 365: 12}   # "30/90/180/365 days" -> months on this monthly grid
MIN_N_FOR_PROBABILITY = 8


def _empirical_table(timeline: pd.DataFrame, momentum_filter=None) -> dict:
    """{start_regime: {horizon_days: {dest_regime: {"p": float, "n": int}, "n_start": int}}}."""
    regimes = list(dict.fromkeys(REGIME_LABELS.values()))
    declared = timeline["declared_regime"]
    if momentum_filter is not None:
        mask = momentum_filter
    else:
        mask = pd.Series(True, index=timeline.index)

    out: dict = {}
    for start_regime in regimes:
        start_idx = [i for i, (m, ok) in enumerate(zip(declared, mask))
                    if m == start_regime and bool(ok)]
        out[start_regime] = {}
        for horizon_days, h_months in HORIZONS_MONTHS.items():
            counts = {r: 0 for r in regimes}
            n_valid = 0
            for i in start_idx:
                j = i + h_months
                if j >= len(declared):
                    continue
                dest = declared.iloc[j]
                if dest is None:
                    continue
                counts[dest] += 1
                n_valid += 1
            cell = {"n_start": n_valid, "destinations": {}}
            for r in regimes:
                if n_valid >= MIN_N_FOR_PROBABILITY:
                    cell["destinations"][r] = {"p": round(counts[r] / n_valid, 3), "n": counts[r]}
                else:
                    cell["destinations"][r] = {"p": None, "n": counts[r]}
            out[start_regime][str(horizon_days)] = cell
    return out


def build_tables(timeline: pd.DataFrame, momentum: pd.Series | None = None) -> dict:
    """Unconditional table + (if momentum is provided) two momentum-conditioned
    tables (momentum >= 0 / momentum < 0 AT THE START MONTH)."""
    result = {"unconditional": _empirical_table(timeline)}
    if momentum is not None and not momentum.dropna().empty:
        aligned = momentum.reindex(timeline.index)
        result["momentum_improving"] = _empirical_table(timeline, aligned >= 0)
        result["momentum_deteriorating"] = _empirical_table(timeline, aligned < 0)
    return result


def for_current(tables: dict, current_regime: str | None, current_momentum: float | None) -> dict:
    """The slice of `tables` relevant to today: the current regime's row from
    the unconditional table, plus the momentum-conditioned row matching
    today's OWN momentum sign, if available (falls back to unconditional if
    momentum is unknown or the conditioned cell is too thin)."""
    if current_regime is None or current_regime not in tables.get("unconditional", {}):
        return {"available": False}
    uncond = tables["unconditional"][current_regime]
    cond_key = None
    if current_momentum is not None:
        cond_key = "momentum_improving" if current_momentum >= 0 else "momentum_deteriorating"
    cond = tables.get(cond_key, {}).get(current_regime) if cond_key else None
    return {"available": True, "regime": current_regime, "unconditional": uncond,
           "momentum_conditioned": cond, "momentum_bucket": cond_key}

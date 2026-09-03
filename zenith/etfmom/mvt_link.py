"""Reads the committed Multivariate Trend ETF scores — MOMENTUM's 6th factor.

mom.yml already runs the full NxN pairwise/residual-momentum engine over this
same ETF universe every night and commits data/mom/mvt/etfs_latest.json. There
is no reason to recompute it here, and every reason not to: it is the single
most expensive stage in the MOMENTUM run.

The one wrinkle is that mvt's universe is NOT this one. mvt collapses
near-duplicate funds (SPY/VOO/IVV/SPLG → one keeper) because a pairwise spread
between two instruments correlated ≥ 0.99 divides by a near-zero spread vol and
explodes. ETF MOMENTUM deliberately keeps those duplicates, so ~102 of its
names have no mvt row of their own.

INHERITANCE is what closes that gap. mvt commits its `empirical_exclusions`
dict, whose values name the surviving keeper verbatim ("near_duplicate_of:SPY").
A fund excluded for that reason was excluded *precisely because* its returns
track the keeper's at ≥ 0.99, so its residual relative strength against the
peer set is the keeper's by construction — inheriting the score is a statement
about the data, not a convenience. Measured live, this takes the number of
five-factor rows from 106 down to 4.

Anything still missing gets NO mvt score, and engine.composite() renormalizes
the other five factors' weights for that row. That is the designed fail-soft
path (see mom/engine.py::composite's docstring), not a workaround — a missing
6th factor must never be imputed as a neutral 0, which would drag the row's
composite toward the middle and make it look more confident than it is.
"""

from __future__ import annotations

from datetime import date

from ..config import ETFMOM_MVT_MAX_STALE_DAYS
from ..mom.mvt import load as mvt_load

_DUP_PREFIX = "near_duplicate_of:"


def _stale_days(as_of: str | None, today: date) -> int | None:
    try:
        return (today - date.fromisoformat(as_of)).days
    except Exception:
        return None


def scores(today: date | None = None,
           max_stale_days: int = ETFMOM_MVT_MAX_STALE_DAYS) -> tuple[dict, dict]:
    """Returns ({ticker: {"score": float, "source": ticker}}, status).

    `source` is the ticker the score was actually computed for — equal to the
    key for a direct hit, or the near-duplicate keeper for an inherited one, so
    the UI can label `mvt via SPY` rather than implying it was measured.

    Past `max_stale_days` the whole factor is dropped rather than blended: a
    stale 6th factor mixed into five fresh ones is worse than five fresh ones,
    because nothing downstream would show that the composite is part history.
    """
    today = today or date.today()
    doc = mvt_load("etfs", {})
    rows = doc.get("rows") or []
    as_of = doc.get("as_of")
    stale = _stale_days(as_of, today)

    status = {"mvt_as_of": as_of, "mvt_stale_days": stale,
              "n_rows": len(rows), "direct": 0, "inherited": 0}

    if not rows:
        status["error"] = "mvt_etf_artefact_missing_or_empty"
        return {}, status
    if stale is None:
        status["error"] = "mvt_as_of_unparseable"
        return {}, status
    if stale > max_stale_days:
        status["error"] = f"mvt_too_stale({stale}d>{max_stale_days}d)"
        return {}, status

    direct = {r["ticker"]: r["normalized_score"] for r in rows
              if r.get("ticker") and r.get("normalized_score") is not None}
    out = {t: {"score": v, "source": t} for t, v in direct.items()}
    status["direct"] = len(out)

    for ticker, reason in (doc.get("empirical_exclusions") or {}).items():
        if ticker in out or not isinstance(reason, str) or not reason.startswith(_DUP_PREFIX):
            continue
        keeper = reason[len(_DUP_PREFIX):].strip()
        # The keeper can itself be absent (too little history to score) -- in
        # that case there is nothing to inherit and the row honestly falls
        # through to a five-factor composite.
        if keeper in direct:
            out[ticker] = {"score": direct[keeper], "source": keeper}
            status["inherited"] += 1

    return out, status

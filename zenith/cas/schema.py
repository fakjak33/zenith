"""The standard CAS signal record — the common shape every model emits.

Keeping one schema is what makes the *overlap* view possible: strategies, flows,
themes and rebalance signals are all comparable because they share these fields.

A Signal is intentionally a plain dict (JSON-native, no pandas needed to read it
in the Streamlit viewer). ``make_signal`` validates/normalises one.
"""

from __future__ import annotations

from datetime import date

SEGMENTS = ("strategies", "flows", "themes", "rebalance", "regime", "factor_rotation")
STATES = ("buy", "neutral", "sell")
HORIZONS = ("intraday", "days", "weeks", "months")
CONFIDENCE = ("low", "medium", "high")


def state_from_signal(x: float, band: float = 0.15) -> str:
    """Map a continuous signal in [-1, 1] to buy / neutral / sell."""
    if x >= band:
        return "buy"
    if x <= -band:
        return "sell"
    return "neutral"


def make_signal(asset: str, segment: str, family: str, signal: float, *,
                asset_class: str = "equity", horizon: str = "days",
                rationale: str = "", source: str = "", confidence: str = "medium",
                percentile: float | None = None, asof: str | None = None) -> dict:
    """Build one validated signal record. ``signal`` is clamped to [-1, 1]."""
    s = max(-1.0, min(1.0, float(signal)))
    if segment not in SEGMENTS:
        raise ValueError(f"bad segment {segment!r}; expected one of {SEGMENTS}")
    if horizon not in HORIZONS:
        raise ValueError(f"bad horizon {horizon!r}")
    if confidence not in CONFIDENCE:
        raise ValueError(f"bad confidence {confidence!r}")
    return {
        "asof": asof or date.today().isoformat(),
        "asset": asset,
        "asset_class": asset_class,
        "segment": segment,
        "family": family,
        "horizon": horizon,
        "signal": round(s, 4),
        "state": state_from_signal(s),
        "score": round(s, 4),
        "percentile": None if percentile is None else round(float(percentile), 4),
        "rationale": rationale,
        "source": source,
        "confidence": confidence,
    }


def validate(rec: dict) -> bool:
    """True if a record has the required fields with sane values."""
    try:
        return (
            isinstance(rec["asset"], str)
            and rec["segment"] in SEGMENTS
            and rec["state"] in STATES
            and -1.0 <= float(rec["signal"]) <= 1.0
            and rec["horizon"] in HORIZONS
        )
    except Exception:
        return False

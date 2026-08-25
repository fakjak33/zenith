"""IDEAS risk/reward: entry zone, stop, target, scale-in plan, invalidation.

Two distinct entry points, deliberately kept separate for cost reasons:

  * `proxy_score(row)` -- a no-network risk/reward READ for the group-score
    conviction blend (groups.py), computed only from what panel.py already
    carries: MOMENTUM's committed breakout_grid trailing high/low (as
    resistance/support) and EDGE's lottery MAXβ percentile (a quality
    penalty). Runs on the WHOLE universe every day at zero extra IO cost.

  * `build(ticker, side, df, ...)` -- the full spec-section-8 construction
    (entry zone / stop / target / R:R / scale-in / invalidation), which needs
    ATR and requires one fresh price fetch per name. This only runs on the
    narrowed candidate pool (~150-300 names, after conviction/unusualness
    selection) -- never the whole universe -- matching panel.py's own stated
    "one on-demand price pull for technical detail" design note.

Stops are ATR-based (a conventional swing-trade multiple, wide enough to
survive daily noise across a multi-week/month holding period -- spec
section 3 explicitly rules out day-trading horizons), WIDENED to the nearest
real structural support/resistance level when that gives MORE room than the
pure ATR distance -- never less. A stock sitting right at its own 63-day low
(support roughly equal to entry) must not collapse the stop toward zero risk,
which would blow out the R:R ratio into a value that only looks attractive
because the denominator is nearly zero.
"""

from __future__ import annotations

import math

import pandas as pd

from ..cas.signals import indicators as ind
from ..mom.normalize import tanh_clip

ATR_STOP_MULT = 2.5      # stop distance in ATRs
ATR_TARGET_FALLBACK_MULT = 4.0   # used only when no structural level exists


# ------------------------------------------------------------- no-network proxy
def proxy_score(row: dict) -> tuple[float, dict]:
    """[-1,1] structural risk/reward read + explain, using only committed
    MOMENTUM/EDGE data already in the panel (see module docstring)."""
    tech = row.get("technicals") or {}
    detail = tech.get("detail") or {}
    last_close = detail.get("last_close")
    bg = tech.get("breakout_grid") or {}
    horizon = bg.get("12m") or bg.get("6m") or bg.get("9m")

    parts: dict[str, float] = {}
    if last_close and horizon and horizon.get("high") and horizon.get("low") and horizon["high"] > horizon["low"]:
        upside = (horizon["high"] - last_close) / last_close
        downside = (last_close - horizon["low"]) / last_close
        if downside > 1e-6 and upside > -0.999:
            rr = max(upside, 1e-3) / downside
            # log so rr=1x -> 0, rr~2.7x -> ~0.76, rr<1x -> negative
            parts["structural_rr"] = tanh_clip(math.log(max(rr, 1e-3)), scale=1.0)

    lot = row.get("lottery")
    if lot and lot.get("pctile") is not None:
        # high lottery/MAXbeta percentile predicts underperformance (EDGE B+,
        # Bali-Ince-Ozsoylev 2026) -- a dampened penalty, not a full-weight signal
        parts["lottery_penalty"] = -((lot["pctile"] - 50.0) / 50.0) * 0.5

    if not parts:
        return 0.0, {"coverage": False}
    score = sum(parts.values()) / len(parts)
    return max(-1.0, min(1.0, score)), {"coverage": True, "parts": parts}


# --------------------------------------------------------------- full construction
def _scale_in_plan(side: str, breakout_confirmed: bool | None, state: str | None) -> dict:
    """Rule-based entry approach (spec section 8): immediate vs scaled vs
    wait-for-confirmation, from the same technical state already on the card
    -- never a separate discretionary call."""
    if breakout_confirmed:
        return {"style": "immediate", "note": "Breakout already confirmed on the daily close -- "
                "full position size is reasonable at the current entry zone."}
    if state in ("NEUTRAL",):
        return {"style": "scaled", "note": "Signal is not yet decisive -- scale in.",
                "tranches": [{"pct": 25, "when": "initial"}, {"pct": 25, "when": "on confirmation"},
                             {"pct": 25, "when": "on a favorable pullback"},
                             {"pct": 25, "when": "after the catalyst/technical confirms"}]}
    return {"style": "scaled", "note": "Standard scale-in given moderate signal strength.",
            "tranches": [{"pct": 50, "when": "initial"}, {"pct": 50, "when": "on confirmation or a pullback"}]}


def _invalidation_text(side: str, stop: float, state: str | None) -> str:
    verb = "closes below" if side == "long" else "closes above"
    return f"Thesis invalidated if price {verb} {stop:.2f} on a daily-close basis, or if the technical state reverses to the opposite extreme."


def build(ticker: str, side: str, df: pd.DataFrame,
         breakout_confirmed: bool | None = None, state: str | None = None) -> dict:
    """side: 'long' or 'short'. df: OHLCV DataFrame (>=60 bars, from
    cas.sources.prices.get_history) for `ticker`. Returns {available: False}
    rather than a fabricated level when there is not enough history for a
    reliable ATR (spec section 29)."""
    if df is None or df.empty or len(df) < 60:
        return {"available": False, "reason": "insufficient_history"}
    close = df["close"].dropna()
    last = float(close.iloc[-1])
    atr_series = ind.atr(df, 14).dropna()
    if atr_series.empty or last <= 0:
        return {"available": False, "reason": "insufficient_history"}
    atr14 = float(atr_series.iloc[-1])
    if not math.isfinite(atr14) or atr14 <= 0:
        return {"available": False, "reason": "zero_volatility"}

    hi252 = float(close.shift(1).rolling(252).max().dropna().iloc[-1]) if len(close) > 253 else None
    lo252 = float(close.shift(1).rolling(252).min().dropna().iloc[-1]) if len(close) > 253 else None
    hi63 = float(close.shift(1).rolling(63).max().dropna().iloc[-1]) if len(close) > 64 else None
    lo63 = float(close.shift(1).rolling(63).min().dropna().iloc[-1]) if len(close) > 64 else None

    if side == "long":
        atr_stop = last - ATR_STOP_MULT * atr14
        stop = atr_stop
        if lo63 is not None and lo63 < last:
            # widen to the structural level when it gives MORE room than pure
            # ATR, never less -- a stock sitting right at its own 63d low
            # (structural support ~= entry) must not collapse the stop to
            # near-zero risk, which would blow out the R:R ratio (see
            # riskreward.build's own history of catching exactly this in
            # scripts/screen.py's sane-band check)
            stop = min(atr_stop, lo63 * 0.99)
        stop = min(stop, last - 0.01)     # defensive: a long stop must be below entry, always
        target = hi252 if (hi252 and hi252 > last) else last + ATR_TARGET_FALLBACK_MULT * atr14
        entry_zone = (round(last * 0.985, 2), round(last * 1.01, 2))
    else:
        atr_stop = last + ATR_STOP_MULT * atr14
        stop = atr_stop
        if hi63 is not None and hi63 > last:
            stop = max(atr_stop, hi63 * 1.01)
        stop = max(stop, last + 0.01)     # defensive: a short stop must be above entry, always
        target = lo252 if (lo252 and lo252 < last) else last - ATR_TARGET_FALLBACK_MULT * atr14
        entry_zone = (round(last * 0.99, 2), round(last * 1.015, 2))

    risk = abs(last - stop)
    reward = abs(target - last)
    if risk <= 1e-9:
        return {"available": False, "reason": "degenerate_stop"}
    rr = round(reward / risk, 2)

    return {
        "available": True, "side": side, "last_close": round(last, 2), "atr14": round(atr14, 2),
        "entry_zone": entry_zone, "stop": round(stop, 2), "target": round(target, 2),
        "risk_per_share": round(risk, 2), "reward_per_share": round(reward, 2),
        "rr_ratio": rr, "max_drawdown_pct": round(risk / last, 4),
        "scale_in": _scale_in_plan(side, breakout_confirmed, state),
        "invalidation": _invalidation_text(side, stop, state),
    }

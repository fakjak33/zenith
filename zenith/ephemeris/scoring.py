"""Trade scoring — the exact rules, in one place.

Entry is the CLOSE of the last visible candle, close[t].

No stops: the trade is held to close[t+N].
    market_ret = close[t+N] / close[t] - 1        (outcome = its sign)
    trade_ret  = direction * market_ret
    pnl        = trade_ret * stake

With SL and/or TP: walk forward candle by candle on OHLC.
  * A gap through a level fills at the OPEN, not at the level (either side:
    a gap through the stop is a worse fill, a gap through the target better).
  * Otherwise the first level touched exits, at the level.
  * Both touched inside one candle -> assume the STOP hit first
    (conservative) and flag the trade `ambiguous`.
  * Neither touched -> exit at close[t+N].

Also recorded: R-multiple (when a stop exists), ATR-normalised return, MFE /
MAE over the bars actually held and over the full horizon, candles held and
exit reason. MFE is >= 0 and MAE <= 0, both as fractions of entry, signed in
the trade's favour.
"""

from __future__ import annotations

import math

import numpy as np


class InvalidLevels(ValueError):
    pass


def validate_levels(direction: int, entry: float, sl: float | None, tp: float | None) -> None:
    if direction not in (1, -1):
        raise InvalidLevels("direction must be +1 (long) or -1 (short)")
    for name, lvl in (("stop", sl), ("target", tp)):
        if lvl is not None and (not math.isfinite(lvl) or lvl <= 0):
            raise InvalidLevels(f"{name} must be a positive price")
    if direction == 1:
        if sl is not None and sl >= entry:
            raise InvalidLevels("A long's stop must be BELOW entry.")
        if tp is not None and tp <= entry:
            raise InvalidLevels("A long's target must be ABOVE entry.")
    else:
        if sl is not None and sl <= entry:
            raise InvalidLevels("A short's stop must be ABOVE entry.")
        if tp is not None and tp >= entry:
            raise InvalidLevels("A short's target must be BELOW entry.")


def level_from_spec(direction: int, entry: float, kind: str, mode: str, value: float,
                    atr: float | None) -> float:
    """SL/TP price from a % distance or an ATR multiple. kind: 'sl' | 'tp'."""
    if mode == "%":
        dist = entry * value / 100.0
    elif mode == "ATR":
        if not atr or not math.isfinite(atr):
            raise InvalidLevels("ATR is not available for this chart.")
        dist = atr * value
    else:
        raise InvalidLevels(f"unknown level mode {mode!r}")
    sign = -direction if kind == "sl" else direction
    return entry + sign * dist


def _excursions(direction: int, entry: float, h: np.ndarray, l: np.ndarray) -> tuple[float, float]:
    if h.size == 0:
        return 0.0, 0.0
    if direction == 1:
        return max(0.0, h.max() / entry - 1), min(0.0, l.min() / entry - 1)
    return max(0.0, 1 - l.min() / entry), min(0.0, -(h.max() / entry - 1))


def score_trade(*, direction: int, entry: float, o, h, l, c, stake: float,
                sl: float | None = None, tp: float | None = None,
                atr: float | None = None) -> dict:
    """Score one call over the future bars (arrays of length N >= 1)."""
    o, h, l, c = (np.asarray(x, dtype=np.float64) for x in (o, h, l, c))
    n = len(c)
    if n == 0:
        raise ValueError("no future bars")
    validate_levels(direction, entry, sl, tp)

    exit_px, exit_i, reason = float(c[-1]), n - 1, "horizon"
    ambiguous = gap = False
    if sl is not None or tp is not None:
        for i in range(n):
            if direction == 1:
                sl_gap = sl is not None and o[i] <= sl
                tp_gap = tp is not None and o[i] >= tp
                sl_hit = sl is not None and l[i] <= sl
                tp_hit = tp is not None and h[i] >= tp
            else:
                sl_gap = sl is not None and o[i] >= sl
                tp_gap = tp is not None and o[i] <= tp
                sl_hit = sl is not None and h[i] >= sl
                tp_hit = tp is not None and l[i] <= tp
            if sl_gap:
                exit_px, exit_i, reason, gap = float(o[i]), i, "stop", True
                break
            if tp_gap:
                exit_px, exit_i, reason, gap = float(o[i]), i, "target", True
                break
            if sl_hit and tp_hit:
                exit_px, exit_i, reason, ambiguous = float(sl), i, "stop", True
                break
            if sl_hit:
                exit_px, exit_i, reason = float(sl), i, "stop"
                break
            if tp_hit:
                exit_px, exit_i, reason = float(tp), i, "target"
                break

    market_ret = float(c[-1] / entry - 1)
    trade_ret = float(direction * (exit_px / entry - 1))
    mfe, mae = _excursions(direction, entry, h[:exit_i + 1], l[:exit_i + 1])
    mfe_full, mae_full = _excursions(direction, entry, h, l)
    risk = abs(entry - sl) / entry if sl is not None else None
    atr_pct = (atr / entry) if atr and math.isfinite(atr) and atr > 0 else None
    return {
        "outcome": "UP" if market_ret > 0 else "DOWN" if market_ret < 0 else "FLAT",
        "market_ret": market_ret,
        "trade_ret": trade_ret,
        "pnl": trade_ret * stake,
        "win": trade_ret > 0,
        "exit_price": exit_px,
        "exit_index": exit_i,             # 0-based into the future bars
        "candles_held": exit_i + 1,
        "exit_reason": reason,
        "ambiguous": ambiguous,
        "gap_fill": gap,
        "r_mult": (trade_ret / risk) if risk else None,
        "atr_ret": (trade_ret / atr_pct) if atr_pct else None,
        "mfe": mfe, "mae": mae, "mfe_full": mfe_full, "mae_full": mae_full,
    }

"""Apply stop-loss / take-profit on top of a continuous position.

Supports ATR-based and fixed-percent stops & take-profits, with optional
trailing for the stop. The Carver position is continuous, but stops/TPs are
discrete events: when a position of a given sign opens we record the entry
price and stop/target levels; on a breach we flatten until the underlying
signal flips or goes flat, then re-arm. Returns the gated position plus an
event log for the blotter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RiskConfig
from .atr import atr


def _stop_distance(cfg: RiskConfig, atr_val: float, price: float) -> float | None:
    """Absolute price distance for the stop, or None if no stop active."""
    if cfg.use_atr_stop:
        return cfg.atr_stop_mult * atr_val
    if cfg.use_pct_stop:
        return cfg.pct_stop * price
    return None


def _tp_distance(cfg: RiskConfig, atr_val: float, price: float) -> float | None:
    """Absolute price distance for the take-profit, or None if inactive."""
    if cfg.use_atr_tp:
        return cfg.atr_tp_mult * atr_val
    if cfg.use_pct_tp:
        return cfg.pct_tp * price
    return None


def apply_risk_overlay(
    position: pd.Series,
    ohlcv: pd.DataFrame,
    cfg: RiskConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Gate ``position`` by stop-loss / take-profit (ATR or fixed-percent).

    Returns (gated_position, events) where events has columns
    [date, type, price] with type in {entry, stop, take_profit}.
    """
    stop_on = cfg.use_atr_stop or cfg.use_pct_stop
    tp_on = cfg.use_atr_tp or cfg.use_pct_tp
    if not (stop_on or tp_on):
        return position, pd.DataFrame(columns=["date", "type", "price"])

    close = ohlcv["close"].reindex(position.index)
    atr_series = atr(ohlcv, cfg.atr_period).reindex(position.index)  # causal: no back-fill

    pos = position.to_numpy(dtype=float)
    px = close.to_numpy(dtype=float)
    av = atr_series.to_numpy(dtype=float)
    out = pos.copy()

    events: list[dict] = []
    in_trade = False
    sign = 0
    entry_px = stop_px = tp_px = np.nan
    suppressed = False

    for i in range(len(pos)):
        cur_sign = int(np.sign(pos[i]))

        if cur_sign == 0:
            in_trade = False
            suppressed = False
            sign = 0
            continue

        if not in_trade or cur_sign != sign:
            in_trade = True
            suppressed = False
            sign = cur_sign
            entry_px = px[i]
            d_stop = _stop_distance(cfg, av[i], entry_px)
            d_tp = _tp_distance(cfg, av[i], entry_px)
            stop_px = entry_px - sign * d_stop if d_stop is not None else np.nan
            tp_px = entry_px + sign * d_tp if d_tp is not None else np.nan
            events.append({"date": position.index[i], "type": "entry", "price": entry_px})

        if suppressed:
            out[i] = 0.0
            continue

        # trailing stop ratchets in the trade's favor (never loosens)
        if stop_on and cfg.trailing_stop:
            d_stop = _stop_distance(cfg, av[i], px[i])
            if d_stop is not None:
                trail = px[i] - sign * d_stop
                if np.isnan(stop_px):
                    stop_px = trail
                elif not np.isnan(trail):
                    stop_px = max(stop_px, trail) if sign > 0 else min(stop_px, trail)
        # arm stops/targets once the (causal) ATR warm-up has data
        if stop_on and np.isnan(stop_px):
            d_stop = _stop_distance(cfg, av[i], entry_px)
            if d_stop is not None and np.isfinite(d_stop):
                stop_px = entry_px - sign * d_stop
        if tp_on and np.isnan(tp_px):
            d_tp = _tp_distance(cfg, av[i], entry_px)
            if d_tp is not None and np.isfinite(d_tp):
                tp_px = entry_px + sign * d_tp

        hit_stop = stop_on and not np.isnan(stop_px) and (
            (sign > 0 and px[i] <= stop_px) or (sign < 0 and px[i] >= stop_px))
        hit_tp = tp_on and not np.isnan(tp_px) and (
            (sign > 0 and px[i] >= tp_px) or (sign < 0 and px[i] <= tp_px))

        if hit_stop:
            out[i] = 0.0
            suppressed = True
            events.append({"date": position.index[i], "type": "stop", "price": px[i]})
        elif hit_tp:
            out[i] = 0.0
            suppressed = True
            events.append({"date": position.index[i], "type": "take_profit", "price": px[i]})

    gated = pd.Series(out, index=position.index)
    ev_df = pd.DataFrame(events) if events else pd.DataFrame(columns=["date", "type", "price"])
    return gated, ev_df


# Backwards-compatible alias (older callers/tests use this name).
apply_atr_overlay = apply_risk_overlay

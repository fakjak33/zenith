"""Extract discrete entry/exit trades from a continuous position series.

A 'trade' is a contiguous run where the position holds one sign. We record the
entry and exit date/price, direction, bars held, and the price move — so the
table lines up exactly with the long/short markers drawn on the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trades(position: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Return a tidy trade ledger from a position series and close prices.

    Columns: side, entry_date, entry_price, exit_date, exit_price,
    bars_held, price_move_%. The price move is signed by direction (a profitable
    short shows positive). Prices are taken from ``close`` aligned to position.
    """
    close = close.reindex(position.index).ffill()
    sign = np.sign(position).fillna(0.0).to_numpy()
    dates = position.index
    px = close.to_numpy(dtype=float)

    rows: list[dict] = []
    cur = 0
    entry_i = None
    for i in range(len(sign)):
        s = sign[i]
        if s != cur:
            # close any open trade
            if cur != 0 and entry_i is not None:
                rows.append(_make_row(cur, entry_i, i, dates, px))
            # open a new trade if entering a nonzero position
            entry_i = i if s != 0 else None
            cur = s
    # close trade still open at the end
    if cur != 0 and entry_i is not None:
        rows.append(_make_row(cur, entry_i, len(sign) - 1, dates, px))

    if not rows:
        return pd.DataFrame(columns=[
            "side", "entry_date", "entry_price", "exit_date",
            "exit_price", "bars_held", "price_move_%",
        ])
    return pd.DataFrame(rows)


def _make_row(sign_val, entry_i, exit_i, dates, px) -> dict:
    side = "LONG" if sign_val > 0 else "SHORT"
    entry_px, exit_px = px[entry_i], px[exit_i]
    raw_move = (exit_px / entry_px - 1.0) if entry_px else 0.0
    move = raw_move * (1 if sign_val > 0 else -1)
    return {
        "side": side,
        "entry_date": pd.Timestamp(dates[entry_i]).date(),
        "entry_price": round(float(entry_px), 4),
        "exit_date": pd.Timestamp(dates[exit_i]).date(),
        "exit_price": round(float(exit_px), 4),
        "bars_held": int(exit_i - entry_i),
        "price_move_%": round(float(move) * 100, 2),
    }

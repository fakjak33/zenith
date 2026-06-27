"""Seasonality — computed from price history (no external data).

Given a price DataFrame, scores the current calendar month by its historical
average forward 1-month return and hit-rate. A simple, transparent free signal.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def month_bias(df: pd.DataFrame, today: date | None = None) -> dict:
    """Average historical return and hit-rate for the current month."""
    today = today or date.today()
    m = today.month
    close = df["close"].dropna()
    monthly = close.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return {"month": m, "avg_return": 0.0, "hit_rate": 0.0, "n": 0}
    same = monthly[monthly.index.month == m]
    if same.empty:
        return {"month": m, "avg_return": 0.0, "hit_rate": 0.0, "n": 0}
    return {
        "month": m,
        "avg_return": round(float(same.mean()), 4),
        "hit_rate": round(float((same > 0).mean()), 3),
        "n": int(same.shape[0]),
    }

"""Overnight vs intraday decomposition math (pure functions)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BARS = 60
MOM_WINDOW = 21         # ~1 month for the overnight-momentum signal
VOL_WINDOW = 63


def streams(df: pd.DataFrame) -> pd.DataFrame:
    """overnight (open/prevclose-1) + intraday (close/open-1) daily returns."""
    o, c = df["open"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    overnight = (o / prev_c - 1.0)
    intraday = (c / o - 1.0)
    out = pd.DataFrame({"overnight": overnight, "intraday": intraday},
                       index=df.index).dropna()
    # guard against split/data glitches: cap daily legs at +/-50%
    return out[(out["overnight"].abs() < 0.5) & (out["intraday"].abs() < 0.5)]


def _t_stat(r: pd.Series) -> float | None:
    r = r.dropna()
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    return round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else None


def stock_stats(df: pd.DataFrame) -> dict | None:
    """Trailing decomposition stats + the overnight-momentum signal for one
    name. share_overnight = overnight's fraction of the summed absolute drift."""
    s = streams(df)
    if len(s) < MIN_BARS:
        return None
    cum_on = float((1 + s["overnight"]).prod() - 1)
    cum_id = float((1 + s["intraday"]).prod() - 1)
    tot = float((1 + s["overnight"]).prod() * (1 + s["intraday"]).prod() - 1)
    denom = abs(cum_on) + abs(cum_id)
    share_on = round(cum_on / denom, 4) if denom > 0 else None
    # overnight-momentum signal (LPS continuation): last-month overnight sum,
    # vol-scaled by the trailing daily overnight vol.
    recent = s["overnight"].tail(MOM_WINDOW)
    vol = float(s["overnight"].tail(VOL_WINDOW).std()) or None
    on_mom = (round(float(recent.sum()) / (vol * np.sqrt(MOM_WINDOW)), 4)
              if vol else None)
    tug = round(cum_on - cum_id, 4)     # overnight-minus-intraday total
    return {
        "cum_overnight": round(cum_on, 4), "cum_intraday": round(cum_id, 4),
        "cum_total": round(tot, 4), "share_overnight": share_on,
        "overnight_avg_bp": round(float(s["overnight"].mean()) * 1e4, 2),
        "intraday_avg_bp": round(float(s["intraday"].mean()) * 1e4, 2),
        "overnight_t": _t_stat(s["overnight"]), "intraday_t": _t_stat(s["intraday"]),
        "on_mom": on_mom, "tug": tug, "n_days": int(len(s)),
    }


def cumulative_panel(df: pd.DataFrame, max_points: int = 260) -> list[dict]:
    """Down-sampled cumulative overnight/intraday/total series for charting."""
    s = streams(df)
    if len(s) < MIN_BARS:
        return []
    cum = pd.DataFrame({
        "overnight": (1 + s["overnight"]).cumprod() - 1,
        "intraday": (1 + s["intraday"]).cumprod() - 1,
        "total": (1 + s["overnight"]).cumprod() * (1 + s["intraday"]).cumprod() - 1,
    }, index=s.index)
    step = max(1, len(cum) // max_points)
    cum = cum.iloc[::step]
    return [{"d": d.date().isoformat(),
             "overnight": round(float(r["overnight"]), 5),
             "intraday": round(float(r["intraday"]), 5),
             "total": round(float(r["total"]), 5)}
            for d, r in cum.iterrows()]

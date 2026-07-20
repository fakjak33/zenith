"""Lottery / MAX short screen — pure math (Bali-Cakici-Whitelaw; Bali-Ince-Ozsoylev).

Classic MAX (BCW 2011): the average of a stock's N highest daily returns last
month predicts LOW future returns (lottery demand). Raw MAX is largely subsumed
by modern mispricing factors (Bali-Ince-Ozsoylev 2026); their beta-purged MAXβ
survives (-0.81%/mo, t=-3.62, incl. large caps) — so MAXβ is the headline.

MAXβ construction: for each stock estimate market beta over a trailing window,
strip the systematic part (resid = r - beta*mkt), then take the mean of the
N largest DAILY RESIDUAL returns over the last ~21 trading days. High MAXβ =
strongest idiosyncratic-lottery demand = the SHORT screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import assemble

HORIZON_TD = 20
BETA_WINDOW = 60        # trading days for the rolling market beta
BETA_MIN = 40
MAX_WINDOW = 21         # ~one month
MAX_N = 5               # BCW use the 5 highest daily returns


def _stock_max(close: pd.Series, mkt_ret: pd.Series) -> dict | None:
    """MAX and beta-purged MAXβ for one stock from aligned daily closes."""
    r = close.pct_change().dropna()
    m = mkt_ret.reindex(r.index).dropna()
    r = r.reindex(m.index).dropna()
    m = m.reindex(r.index)
    if len(r) < BETA_MIN + MAX_WINDOW:
        return None
    # beta over the window ending before the MAX window (no look-ahead into MAX)
    beta_slice_r = r.iloc[-(BETA_WINDOW + MAX_WINDOW):-MAX_WINDOW]
    beta_slice_m = m.iloc[-(BETA_WINDOW + MAX_WINDOW):-MAX_WINDOW]
    if len(beta_slice_r) < BETA_MIN:
        return None
    var = float(np.var(beta_slice_m, ddof=1))
    beta = (float(np.cov(beta_slice_r, beta_slice_m, ddof=1)[0, 1] / var)
            if var > 0 else 0.0)
    recent_r = r.iloc[-MAX_WINDOW:]
    recent_m = m.iloc[-MAX_WINDOW:]
    resid = recent_r - beta * recent_m
    max_raw = float(recent_r.nlargest(MAX_N).mean())
    max_beta = float(resid.nlargest(MAX_N).mean())
    return {"max_raw": round(max_raw, 6), "max_beta": round(max_beta, 6),
            "beta": round(beta, 3)}


def build(px: dict[str, pd.DataFrame], spy_close: pd.Series,
          meta: dict[str, dict]) -> dict:
    """px: {ticker: DataFrame[close]}; meta: {ticker: {name, sector}}.
    Ranks by MAXβ; SHORT screen = top decile (highest idiosyncratic lottery
    demand). Raw MAX is carried alongside for comparison."""
    mkt = spy_close.pct_change().dropna()
    rows = []
    for t, df in px.items():
        if df is None or df.empty or "close" not in df:
            continue
        s = _stock_max(df["close"].dropna(), mkt)
        if s is None:
            continue
        info = meta.get(t, {})
        rows.append({"ticker": t, "name": info.get("name", ""),
                     "sector": info.get("sector", ""), **s,
                     "max_raw_pct": round(s["max_raw"] * 100, 2),
                     "max_beta_pct": round(s["max_beta"] * 100, 2)})
    a = assemble(rows, "max_beta", higher_is_long=False)
    return {"screen": "lottery", "horizon_td": HORIZON_TD, **a}

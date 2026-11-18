"""Inputs for the synthetic chains — all free: Cboe index CSVs, Yahoo prices, FRED T-bills."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...stratidx.fetch import cboe_csv
from ..attrib.benchmarks import cash_index
from ..data.providers import get_prices
from .engine import TERM_COLS, dividend_yield

# underlying → (option-settlement price ticker, total-return ticker, own Cboe vol index or None)
UNDERLYINGS = {
    "SPX": ("^GSPC", "^SP500TR", None),
    "SPY": ("SPY", "SPY", None),
    "QQQ": ("QQQ", "QQQ", "VXN"),
    "IWM": ("IWM", "IWM", "RVX"),
    "DIA": ("DIA", "DIA", "VXD"),
    "EEM": ("EEM", "EEM", "VXEEM"),
    "GLD": ("GLD", "GLD", "GVZ"),
    "USO": ("USO", "USO", "OVX"),
}


def _px(t: str) -> pd.Series:
    return get_prices(t, start=None)["close"]


def load(underlying: str, start_year: int = 2007) -> dict:
    """Everything ``engine.simulate`` needs for one underlying, trimmed to ``start_year``."""
    s_t, tr_t, own = UNDERLYINGS.get(underlying, (underlying, underlying, None))
    S, tr = _px(s_t), _px(tr_t)
    term = pd.DataFrame({c: cboe_csv(c) for c in TERM_COLS})
    if term["VIX"].empty:
        raise ValueError("VIX history unavailable (Cboe download failed and no cache)")
    scale = None
    if own:
        ov = cboe_csv(own)
        if not ov.empty:
            scale = (ov / term["VIX"]).dropna()                 # shape own vol index by the VIX term structure
    elif s_t not in ("^GSPC", "SPY"):
        spx = _px("^GSPC")
        rv = lambda x: np.log(x).diff().rolling(63).std()        # noqa: E731
        scale = (rv(S) / rv(spx).reindex(S.index).ffill()).shift(1).clip(0.3, 5.0)
    skew = cboe_csv("SKEW")
    cash = cash_index()
    rf = (cash.pct_change() * 252).clip(lower=0) if not cash.empty else pd.Series(0.0, index=S.index)
    start = pd.Timestamp(f"{start_year}-01-01")
    S = S[S.index >= start]
    # ETFs come dividend-adjusted from Yahoo, so their own yield is already in the price path
    q = dividend_yield(_px("^GSPC"), _px("^SP500TR")) if s_t == "^GSPC" else None
    return {"S": S, "stock_tr": tr, "term": term, "skew": skew, "rf": rf, "vol_scale": scale, "q": q,
            "settle": s_t, "vol_source": own or ("VIX" if s_t in ("^GSPC", "SPY") else "VIX × realised-vol ratio")}


def published(name: str) -> pd.Series:
    """Cboe benchmark index level (PUT, BXM, …) for calibration."""
    return cboe_csv(name)

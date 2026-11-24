"""Inputs for the synthetic chains — all free: Cboe index CSVs, Yahoo prices, FRED T-bills."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...stratidx.fetch import cboe_csv
from ..attrib.benchmarks import cash_index
from ..data.providers import get_prices, get_raw_prices
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


def load(underlying: str, start: int | str = 2007, end: str | None = None) -> dict:
    """Everything ``engine.simulate`` needs for one underlying, trimmed to [start, end].
    ``start`` is a year (int) or an ISO date; ``end`` None/'' = latest bar."""
    s_t, tr_t, own = UNDERLYINGS.get(underlying, (underlying, underlying, None))
    S, tr = _px(s_t), _px(tr_t)
    # Options are struck and settle on the price AS TRADED. Yahoo's default ETF closes are
    # dividend-adjusted (old prices scaled down by every later dividend), so ETFs use the raw close,
    # and the dividends come back in as a yield q (raw vs total-return), exactly as for SPX.
    raw = None if s_t.startswith("^") else get_raw_prices(s_t)
    if raw is not None and not raw.empty and raw["close"].notna().sum() > 260:
        S = raw["close"].dropna()
        basis = "as traded (unadjusted)"
    else:
        raw = None
        basis = "price index" if s_t.startswith("^") else "dividend-adjusted (raw closes unavailable)"
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
    start = pd.Timestamp(f"{start}-01-01") if isinstance(start, int) or str(start).isdigit() else pd.Timestamp(start)
    stop = pd.Timestamp(end) if end else None
    S = S[S.index >= start]
    if stop is not None:
        S = S[S.index <= stop]
    if s_t == "^GSPC":
        q = dividend_yield(_px("^GSPC"), _px("^SP500TR"))
    elif raw is not None:
        q = dividend_yield(raw["close"].dropna(), tr)          # trailing-year yield from raw vs total return
    else:
        q = None                                                # adjusted path already carries the dividends
    ohlc = (raw if raw is not None else get_prices(s_t, start=None))[["open", "high", "low", "close"]]
    ohlc = ohlc[ohlc.index >= start]
    if stop is not None:
        ohlc = ohlc[ohlc.index <= stop]
    iv1d, iv1d_src = one_day_vol(term)
    return {"S": S, "stock_tr": tr, "term": term, "skew": skew, "rf": rf, "vol_scale": scale, "q": q,
            "settle": s_t, "vol_source": own or ("VIX" if s_t in ("^GSPC", "SPY") else "VIX × realised-vol ratio"),
            "ohlc": ohlc, "iv1d": iv1d, "iv1d_src": iv1d_src, "price_basis": basis}


def one_day_vol(term: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cboe VIX1D (from May 2022), extended back with VIX9D × the median VIX1D/VIX9D ratio on
    their overlap. Returns (series, source label per date)."""
    v1 = cboe_csv("VIX1D")
    v9 = term["VIX9D"].dropna() if "VIX9D" in term else pd.Series(dtype=float)
    if v1.empty and v9.empty:
        return pd.Series(dtype=float), pd.Series(dtype=object)
    ov = pd.concat([v1, v9], axis=1, keys=["a", "b"]).dropna()
    ratio = float((ov["a"] / ov["b"]).median()) if len(ov) > 20 else 1.0
    fill = v9 * ratio
    out = fill.combine_first(v1) if v1.empty else v1.combine_first(fill)
    src = pd.Series(np.where(out.index.isin(v1.index), "VIX1D", f"VIX9D x {ratio:.2f}"), index=out.index)
    return out.sort_index(), src


def published(name: str) -> pd.Series:
    """Cboe benchmark index level (PUT, BXM, …) for calibration."""
    return cboe_csv(name)

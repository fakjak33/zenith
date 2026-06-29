"""Data for the Weekly Brief. Free / best-effort only; every function degrades
gracefully (returns empty on failure) so the weekly Action never breaks.

Reuses the CAS data sources (prices via yfinance, FRED CSV, finviz groups) and
adds the brief-specific bits: S&P 500 constituents, earnings dates, a rough
dealer-gamma (GEX) proxy, and the fed-funds path proxy.
"""

from __future__ import annotations

import io
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

from ..config import BROWSER_HEADERS
from ..cas.sources import prices as cas_prices
from ..cas.sources import fred as cas_fred
from ..cas import store_cas

# --- market overview universe (ordered) ------------------------------------
OVERVIEW: dict[str, str] = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Small Caps",
    "EFA": "Dev. ex-US", "EEM": "Emerging Mkts", "GLD": "Gold", "SLV": "Silver",
    "TLT": "20y Treasuries", "HYG": "High Yield", "LQD": "IG Credit", "UUP": "US Dollar",
    "USO": "Crude Oil", "COPX": "Copper Miners", "BITO": "Bitcoin", "^VIX": "VIX",
}

# mega-cap watchlist for the earnings section (kept small — yfinance is per-ticker)
EARNINGS_WATCH = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ABBV",
    "WMT", "NFLX", "CRM", "BAC", "KO", "PEP", "AMD", "ORCL", "MCD", "CSCO",
]

# GEX is heavy (per-ticker option chains) — keep to the most liquid index ETFs.
GEX_TICKERS = ["SPY", "QQQ", "IWM"]

# FRED series for the rates + fixed-income sections
_FRED = {
    "DGS3MO": "3M UST", "DGS6MO": "6M UST", "DGS1": "1Y UST", "DGS2": "2Y UST",
    "DGS5": "5Y UST", "DGS10": "10Y UST", "DGS30": "30Y UST",
    "T10Y2Y": "10y-2y", "T10Y3M": "10y-3m", "T10YIE": "10y breakeven",
    "BAMLH0A0HYM2": "HY OAS", "BAMLC0A0CM": "IG OAS", "MORTGAGE30US": "30y mortgage",
    "DFF": "Fed funds (eff)", "DFEDTARU": "Target upper", "DFEDTARL": "Target lower",
    "IRLTLT01DEM156N": "Germany 10y", "IRLTLT01JPM156N": "Japan 10y",
    "IRLTLT01GBM156N": "UK 10y", "IRLTLT01CAM156N": "Canada 10y",
}

# FOMC decision dates (2nd day of each meeting). Scheduled dates, not forecasts.
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]


# --- helpers ---------------------------------------------------------------
def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    a, b = close.iloc[-1], close.iloc[-1 - n]
    return round(float(a / b - 1.0), 4) if b else None


def _ytd(close: pd.Series) -> float | None:
    if close.empty:
        return None
    yr = close[close.index >= pd.Timestamp(date.today().year, 1, 1)]
    if len(yr) < 2 or yr.iloc[0] == 0:
        return None
    return round(float(close.iloc[-1] / yr.iloc[0] - 1.0), 4)


def _spark(close: pd.Series, points: int = 40) -> list[float]:
    s = close.dropna().tail(126)
    if len(s) > points:
        s = s.iloc[:: max(1, len(s) // points)]
    return [round(float(x), 4) for x in s.tolist()]


def _perf(close: pd.Series) -> dict:
    return {"last": round(float(close.iloc[-1]), 2),
            "w1": _ret(close, 5), "m1": _ret(close, 21), "m3": _ret(close, 63),
            "ytd": _ytd(close), "spark": _spark(close)}


# --- sections --------------------------------------------------------------
def market_overview(px: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for t, label in OVERVIEW.items():
        df = px.get(t)
        if df is None or df.empty:
            continue
        out.append({"ticker": t, "label": label, **_perf(df["close"])})
    return out


def sector_perf(px: dict[str, pd.DataFrame]) -> list[dict]:
    from ..cas.universe import SECTORS
    out = []
    for t, label in SECTORS.items():
        df = px.get(t)
        if df is None or df.empty:
            continue
        out.append({"ticker": t, "label": label,
                    "w1": _ret(df["close"], 5), "m1": _ret(df["close"], 21)})
    return sorted(out, key=lambda r: (r["w1"] is None, -(r["w1"] or 0)))


def industry_perf(px: dict[str, pd.DataFrame]) -> list[dict]:
    from ..cas.universe import INDUSTRY_ETFS
    out = []
    for t, label in INDUSTRY_ETFS.items():
        df = px.get(t)
        if df is None or df.empty:
            continue
        out.append({"ticker": t, "label": label,
                    "w1": _ret(df["close"], 5), "m1": _ret(df["close"], 21)})
    return sorted(out, key=lambda r: (r["w1"] is None, -(r["w1"] or 0)))


def sp500_constituents(max_age_hours: float = 168.0) -> list[str]:
    cached = store_cas.cache_get("sp500_list", max_age_hours)
    if cached:
        return cached
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=BROWSER_HEADERS, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        syms = [str(s).replace(".", "-") for s in tables[0]["Symbol"].tolist()]
        if syms:
            store_cas.cache_put("sp500_list", syms)
        return syms
    except Exception:
        return []


def stock_heatmap(top: int = 20) -> dict:
    """Best/worst S&P 500 movers (1w & 1m) + a simple breadth read."""
    syms = sp500_constituents()
    if not syms:
        return {}
    px, _ = cas_prices.get_history(syms, period="1y")
    rows, above50, above200, n = [], 0, 0, 0
    for t, df in px.items():
        c = df["close"]
        if len(c) < 60:
            continue
        rows.append({"ticker": t, "w1": _ret(c, 5), "m1": _ret(c, 21)})
        n += 1
        if len(c) >= 50 and c.iloc[-1] > c.tail(50).mean():
            above50 += 1
        if len(c) >= 200 and c.iloc[-1] > c.tail(200).mean():
            above200 += 1
    rows = [r for r in rows if r["w1"] is not None]
    by_w = sorted(rows, key=lambda r: r["w1"], reverse=True)
    return {
        "best_1w": by_w[:top], "worst_1w": by_w[-top:][::-1],
        "breadth": {"n": n,
                    "pct_above_50dma": round(above50 / n, 3) if n else None,
                    "pct_above_200dma": round(above200 / n, 3) if n else None},
    }


def rates() -> dict:
    """Treasury curve, spreads, credit, global long rates + a fed-funds proxy."""
    # force a fresh full pull: the shared "fred" cache key may only hold the small
    # CAS regime set, which would leave most of our series empty.
    data, _ = cas_fred.get_series(list(_FRED), max_age_hours=0.0)
    last = {}
    for sid, label in _FRED.items():
        pts = data.get(sid, [])
        if pts:
            last[sid] = {"label": label, "value": pts[-1]["value"], "date": pts[-1]["date"]}

    def _v(sid):
        return last.get(sid, {}).get("value")

    # fed-funds proxy: market-implied 12m change ~ (1y UST - effective fed funds)
    eff, y1 = _v("DFF"), _v("DGS1")
    implied_12m_bps = round((y1 - eff) * 100, 1) if (eff is not None and y1 is not None) else None
    upcoming = [d for d in FOMC_2026 if d >= date.today().isoformat()][:4]

    return {
        "curve": last,
        "fed_funds": {
            "effective": eff, "target_lower": _v("DFEDTARL"), "target_upper": _v("DFEDTARU"),
            "implied_12m_change_bps": implied_12m_bps,
            "next_fomc": upcoming,
            "note": "Implied path is a rough proxy (1Y UST minus effective fed funds). Full "
                    "FedWatch probabilities require CME data (paid).",
        },
    }


def earnings(watch: list[str] | None = None, window_days: int = 10) -> list[dict]:
    """Upcoming earnings dates for the mega-cap watchlist (best-effort, yfinance)."""
    watch = watch or EARNINGS_WATCH
    try:
        import yfinance as yf
    except Exception:
        return []
    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=window_days)
    out = []
    for t in watch:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                continue
            idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
            for dt in idx:
                if today <= dt <= end:
                    out.append({"ticker": t, "date": dt.date().isoformat()})
                    break
        except Exception:
            continue
    return sorted(out, key=lambda r: r["date"])


# --- options / dealer gamma (GEX) proxy ------------------------------------
def _bs_gamma(S, K, T, sigma, r=0.04):
    # the compound guard also rejects NaN inputs (NaN > 0 is False)
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        g = math.exp(-0.5 * d1 * d1) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
        return g if math.isfinite(g) else 0.0
    except Exception:
        return 0.0


def gex(tickers: list[str] | None = None, max_expiries: int = 4) -> list[dict]:
    """Rough net dealer gamma exposure per ticker, from yfinance option chains
    (Black-Scholes gamma from implied vol; dealers assumed short calls / long puts).
    A proxy — expiry set and the dealer-sign assumption materially affect results."""
    tickers = tickers or GEX_TICKERS
    try:
        import yfinance as yf
    except Exception:
        return []
    out = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            spot = float(tk.fast_info.get("last_price") or tk.history(period="1d")["Close"].iloc[-1])
            net = 0.0
            for exp in (tk.options or [])[:max_expiries]:
                T = max((pd.Timestamp(exp) - pd.Timestamp.today()).days, 0) / 365.0
                if T <= 0:
                    continue
                ch = tk.option_chain(exp)
                for df, sign in ((ch.calls, +1.0), (ch.puts, -1.0)):
                    for _, row in df.iterrows():
                        iv = row.get("impliedVolatility")
                        oi = row.get("openInterest")
                        iv = 0.0 if (iv is None or pd.isna(iv)) else float(iv)
                        oi = 0.0 if (oi is None or pd.isna(oi)) else float(oi)
                        g = _bs_gamma(spot, float(row["strike"]), T, iv)
                        net += sign * g * oi * 100 * spot * spot * 0.01
            if not math.isfinite(net):
                continue
            out.append({"ticker": t, "spot": round(spot, 2),
                        "net_gex_bn": round(net / 1e9, 3),
                        "regime": "positive (vol-dampening)" if net >= 0
                                  else "negative (vol-amplifying)"})
        except Exception:
            continue
    return out


def market_moving_news(movers: set[str], limit: int = 8) -> list[dict]:
    """Zenith's scraped news, prioritising items mentioning a big mover."""
    try:
        from ..store import load_latest
        items = [i for i in load_latest() if i.get("category") == "news"]
    except Exception:
        items = []
    def _hot(it):
        title = (it.get("title", "") + " " + it.get("source", "")).upper()
        return any(m in title for m in movers)
    items.sort(key=lambda it: (not _hot(it), it.get("published", "")), reverse=False)
    return [{"title": i["title"], "source": i["source"], "link": i.get("link", ""),
             "hot": _hot(i)} for i in items[:limit]]

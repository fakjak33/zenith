"""Data for the Weekly Brief. Free / best-effort only; every function degrades
gracefully (returns empty on failure) so the weekly Action never breaks.

Reuses the CAS data sources (prices via yfinance, FRED CSV, finviz groups) and
adds the brief-specific bits: S&P 500 constituents, earnings dates, a rough
dealer-gamma (GEX) proxy, and the fed-funds path proxy.
"""

from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from ..config import BROWSER_HEADERS
from ..cas.sources import prices as cas_prices
from ..cas.sources import fred as cas_fred
from ..cas import store_cas

# --- market overview, grouped by asset class -------------------------------
EQUITY: dict[str, str] = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Small Caps",
    "RSP": "S&P Equal-Wt", "EFA": "Dev. ex-US", "VGK": "Europe", "EWJ": "Japan",
    "EEM": "Emerging Mkts", "^VIX": "VIX",
}
COMMODITY: dict[str, str] = {
    "DBC": "Broad Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "Crude Oil",
    "UNG": "Nat Gas", "COPX": "Copper Miners", "GDX": "Gold Miners", "DBA": "Agriculture",
    "URA": "Uranium",
}
BOND: dict[str, str] = {
    "SHY": "1-3y UST", "IEF": "7-10y UST", "TLT": "20y+ UST", "LQD": "IG Credit",
    "HYG": "High Yield", "EMB": "EM Bonds", "MUB": "Munis", "TIP": "TIPS",
    "BND": "US Agg", "BNDX": "Intl Bonds",
}
FX: dict[str, str] = {
    "UUP": "US Dollar", "FXE": "Euro", "FXY": "Yen", "BITO": "Bitcoin",
}
# group key -> (label, dict). Ordered for display.
GROUPS: dict[str, tuple[str, dict]] = {
    "equity": ("Equities", EQUITY), "commodity": ("Commodities", COMMODITY),
    "bond": ("Bonds & rates", BOND), "fx": ("FX & crypto", FX),
}
# union, kept for compatibility / one-shot price pulls
OVERVIEW: dict[str, str] = {**EQUITY, **COMMODITY, **BOND, **FX}

# mega/large-cap watchlist — the yfinance fallback when the Nasdaq calendar is
# unreachable (the primary source is the full Nasdaq earnings calendar).
EARNINGS_WATCH = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ABBV",
    "WMT", "NFLX", "CRM", "BAC", "KO", "PEP", "AMD", "ORCL", "MCD", "CSCO",
    "MU", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "ADBE", "NOW", "PANW", "SNOW",
    "NKE", "SBUX", "FDX", "GS", "MS", "C", "WFC", "DIS", "BA", "CAT",
    "GE", "DE", "PFE", "MRK", "TMO", "DHR", "ABT", "UPS", "T", "VZ",
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


def _series(close: pd.Series, points: int = 252) -> list[dict]:
    """~1y of DAILY closes [{d,c}] for rebased comparative charts (daily so the
    1-week / 1-month windows still have resolution)."""
    s = close.dropna().tail(points)
    return [{"d": i.date().isoformat(), "c": round(float(v), 4)} for i, v in s.items()]


def _perf(close: pd.Series) -> dict:
    return {"last": round(float(close.iloc[-1]), 2),
            "w1": _ret(close, 5), "m1": _ret(close, 21), "m3": _ret(close, 63),
            "ytd": _ytd(close), "spark": _spark(close), "series": _series(close)}


# --- sections --------------------------------------------------------------
def group_overview(px: dict[str, pd.DataFrame], group: dict[str, str]) -> list[dict]:
    out = []
    for t, label in group.items():
        df = px.get(t)
        if df is None or df.empty:
            continue
        out.append({"ticker": t, "label": label, **_perf(df["close"])})
    return out


def market_overview(px: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """All asset-class groups -> {group_key: [rows]}."""
    return {key: group_overview(px, gd) for key, (_, gd) in GROUPS.items()}


def sector_perf(px: dict[str, pd.DataFrame]) -> list[dict]:
    from ..cas.universe import SECTORS
    out = []
    for t, label in SECTORS.items():
        df = px.get(t)
        if df is None or df.empty:
            continue
        out.append({"ticker": t, "label": label,
                    "w1": _ret(df["close"], 5), "m1": _ret(df["close"], 21),
                    "m3": _ret(df["close"], 63), "series": _series(df["close"])})
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


def sp500_map(max_age_hours: float = 168.0) -> dict[str, str]:
    """Ticker -> company name for the current S&P 500 (free, via Wikipedia)."""
    cached = store_cas.cache_get("sp500_map", max_age_hours)
    if cached:
        return cached
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=BROWSER_HEADERS, timeout=20)
        r.raise_for_status()
        tbl = pd.read_html(io.StringIO(r.text))[0]
        mp = {str(s).replace(".", "-").strip(): str(n).strip()
              for s, n in zip(tbl["Symbol"], tbl["Security"])}
        if mp:
            store_cas.cache_put("sp500_map", mp)
        return mp
    except Exception:
        return {}


def sp500_constituents(max_age_hours: float = 168.0) -> list[str]:
    return list(sp500_map(max_age_hours))


def stock_heatmap(top: int = 20) -> dict:
    """Leaders/laggards among S&P 500 names (1w & 1m, with company names) + breadth."""
    names = sp500_map()
    syms = list(names)
    if not syms:
        return {}
    px, _ = cas_prices.get_history(syms, period="1y")
    rows, above50, above200, n = [], 0, 0, 0
    for t, df in px.items():
        c = df["close"]
        if len(c) < 60:
            continue
        rows.append({"ticker": t, "name": names.get(t, t),
                     "w1": _ret(c, 5), "m1": _ret(c, 21)})
        n += 1
        if len(c) >= 50 and c.iloc[-1] > c.tail(50).mean():
            above50 += 1
        if len(c) >= 200 and c.iloc[-1] > c.tail(200).mean():
            above200 += 1
    rows = [r for r in rows if r["w1"] is not None]
    by_w = sorted(rows, key=lambda r: r["w1"], reverse=True)
    return {
        "leaders_1w": by_w[:top], "laggards_1w": by_w[-top:][::-1],
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


def _nasdaq_earnings(day: str) -> list[dict]:
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={day}"
    r = requests.get(url, headers={**BROWSER_HEADERS, "Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return (r.json().get("data") or {}).get("rows") or []


def _yf_earnings_fallback(window_days: int = 8) -> list[dict]:
    try:
        import yfinance as yf
    except Exception:
        return []
    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=window_days)
    out = []
    for t in EARNINGS_WATCH:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                continue
            idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
            for dt in idx:
                if today <= dt <= end:
                    out.append({"ticker": t, "name": t, "date": dt.date().isoformat(), "time": ""})
                    break
        except Exception:
            continue
    return out


def earnings(window_back: int = 5, window_fwd: int = 8, max_recent_px: int = 25,
             max_age_hours: float = 18.0) -> dict:
    """Recent (with 1-week price reaction) + upcoming major earnings. Primary source
    is the free Nasdaq earnings calendar; falls back to a yfinance watchlist."""
    cached = store_cas.cache_get("earnings_cal", max_age_hours)
    if cached:
        return cached
    today = date.today()
    recent, upcoming = [], []
    try:
        for delta in range(-window_back, window_fwd + 1):
            d = today + timedelta(days=delta)
            if d.weekday() >= 5:
                continue
            for row in _nasdaq_earnings(d.isoformat()):
                sym = str(row.get("symbol", "")).strip()
                if not sym:
                    continue
                rec = {"ticker": sym, "name": str(row.get("name", "")).strip(),
                       "date": d.isoformat(), "time": str(row.get("time", "")),
                       "eps_forecast": row.get("epsForecast", ""), "surprise": row.get("surprise", "")}
                (recent if d < today else upcoming).append(rec)
    except Exception:
        pass
    if not recent and not upcoming:                       # Nasdaq blocked -> fallback
        upcoming = _yf_earnings_fallback()
    if recent:                                            # 1-week price reaction (best-effort)
        syms = list({r["ticker"] for r in recent})[:max_recent_px]
        px, _ = cas_prices.get_history(syms, period="6mo")
        for r in recent:
            df = px.get(r["ticker"])
            r["move_1w"] = _ret(df["close"], 5) if (df is not None and not df.empty) else None
    out = {"recent": sorted(recent, key=lambda r: r["date"], reverse=True),
           "upcoming": sorted(upcoming, key=lambda r: r["date"])}
    store_cas.cache_put("earnings_cal", out)
    return out


def ticker_news(movers: list[dict], per: int = 2, max_tickers: int = 10) -> list[dict]:
    """Ticker-specific headlines for the week's biggest movers via Google News RSS
    (yfinance's per-ticker news is broken). ``movers`` = [{ticker,name,w1}]."""
    try:
        import feedparser
    except Exception:
        return []
    out = []
    for m in movers[:max_tickers]:
        query = (m.get("name") or m["ticker"]) + " stock"
        url = ("https://news.google.com/rss/search?q=" + requests.utils.quote(query)
               + "&hl=en-US&gl=US&ceid=US:en")
        try:
            feed = feedparser.parse(url)
            items = [{"title": e.title, "link": e.link,
                      "published": getattr(e, "published", "")[:16]}
                     for e in feed.entries[:per]]
        except Exception:
            items = []
        if items:
            out.append({"ticker": m["ticker"], "name": m.get("name", ""),
                        "move": m.get("w1"), "items": items})
    return out


# --- factor rotation by timeframe (for the toggle in Additional charts) -----
_TF_DAYS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}


def rotation_by_timeframe() -> dict[str, list[dict]]:
    """Cross-sectional factor/industry rotation ranked within peer groups, for each
    look-back window. {timeframe: [{ticker,label,group,region,signal}]}."""
    from ..cas.universe import frm_universe
    from ..cas.signals.factor_rotation import _trailing_return, _cross_section
    uni = frm_universe()
    px, _ = cas_prices.get_history(list(uni), period="2y")
    closes = {t: px[t]["close"] for t in uni if t in px and len(px[t]) >= 60}
    out: dict[str, list[dict]] = {}
    for tf, lb in _TF_DAYS.items():
        tr = {t: _trailing_return(c, lb=lb, skip=0) for t, c in closes.items()}
        cs = _cross_section(tr, lambda t: (uni[t]["group"], uni[t]["region"]))
        out[tf] = [{"ticker": t, "label": uni[t]["label"], "group": uni[t]["group"],
                    "region": uni[t]["region"], "signal": round(sig, 4)}
                   for t, sig in cs.items()]
    return out


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

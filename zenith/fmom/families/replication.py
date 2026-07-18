"""Russell 1000 replication layer: what's inside each AQR factor right now.

Rebuilds the paper's portfolio construction on the CURRENT Russell 1000
(winsorize 1% → median-market-cap size split → 30/40/30 characteristic
buckets → value-weighted bins; factor = 0.5(LH+SH) − 0.5(LL+SL)) for every
AQR65 row with a computable characteristic (yfinance info fields plus
price/volume history). Output is holdings visibility only — per the paper's
survivorship constraint this layer produces NO backtest; the AQR family's
returns and signals come from published data.

The fundamentals pull (~1000 yf.Ticker().info calls) is slow, so it runs as
its own compute action with a long-lived cache and never blocks signals.
"""

from __future__ import annotations

import math
import time
from datetime import date

import numpy as np
import pandas as pd

from .. import core
from ..catalog import AQR65
from ...cas import store_cas

_FUND_CACHE_KEY = "fmom_fundamentals"
_FUND_TTL_HOURS = 600.0          # ~25 days; refreshed by the monthly action

# yfinance info fields needed per characteristic
_INFO_FIELDS = ("marketCap", "priceToBook", "trailingPE",
                "priceToSalesTrailing12Months", "operatingCashflow",
                "dividendYield", "returnOnEquity", "returnOnAssets",
                "profitMargins", "revenueGrowth", "debtToEquity",
                "enterpriseToEbitda", "totalCash", "freeCashflow")


def replicable_rows() -> list[dict]:
    return [r for r in AQR65 if r["yf_char"]]


def get_fundamentals(tickers: list[str], sleep: float = 0.15,
                     max_age_hours: float = _FUND_TTL_HOURS) -> dict[str, dict]:
    """ticker -> {info field: value}. Cached; best-effort, never raises."""
    cached = store_cas.cache_get(_FUND_CACHE_KEY, max_age_hours) or {}
    todo = [t for t in tickers if t not in cached]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        for i, t in enumerate(todo):
            try:
                info = yf.Ticker(t).info or {}
                cached[t] = {f: info.get(f) for f in _INFO_FIELDS}
            except Exception:
                cached[t] = {}
            if sleep:
                time.sleep(sleep)
            if i and i % 200 == 0:
                store_cas.cache_put(_FUND_CACHE_KEY, cached)   # checkpoint
                print(f"[fmom] fundamentals {i}/{len(todo)}")
        store_cas.cache_put(_FUND_CACHE_KEY, cached)
    return cached


def _safe_div(a, b):
    try:
        a, b = float(a), float(b)
        return a / b if b not in (0.0, None) and not math.isnan(b) else None
    except (TypeError, ValueError):
        return None


def _inv(x):
    return _safe_div(1.0, x)


def _info_chars(fund: dict[str, dict]) -> dict[str, pd.Series]:
    """Characteristic series computable from info fields alone."""
    rows: dict[str, dict[str, float]] = {}
    for t, f in fund.items():
        if not f or not f.get("marketCap"):
            continue
        mc = f["marketCap"]
        rows[t] = {
            "mktcap": mc,
            "book_to_market": _inv(f.get("priceToBook")),
            "ep": _inv(f.get("trailingPE")),
            "sp": _inv(f.get("priceToSalesTrailing12Months")),
            "cfp": _safe_div(f.get("operatingCashflow"), mc),
            "div_yield": f.get("dividendYield") or 0.0,
            "roe": f.get("returnOnEquity"),
            "roa": f.get("returnOnAssets"),
            "profit_margin": f.get("profitMargins"),
            "rev_growth": f.get("revenueGrowth"),
            "debt_to_equity": f.get("debtToEquity"),
            "ev_ebitda": f.get("enterpriseToEbitda"),
            "cash_to_mv": _safe_div(f.get("totalCash"), mc),
            "fcf_to_mv": _safe_div(f.get("freeCashflow"), mc),
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return {c: frame[c].dropna() for c in frame.columns}


def _price_chars(px: dict[str, pd.DataFrame],
                 spy: pd.DataFrame | None) -> dict[str, pd.Series]:
    """Characteristics from ~2y of daily prices/volume."""
    spy_ret = (spy["close"].pct_change().dropna()
               if spy is not None and not spy.empty else None)
    out: dict[str, dict[str, float]] = {}
    for t, df in px.items():
        if df is None or len(df) < 260 or t == "SPY":
            continue
        close, vol = df["close"].dropna(), df["volume"]
        ret = close.pct_change().dropna()
        if len(close) < 260 or len(ret) < 200:
            continue
        rec: dict[str, float] = {}
        # momentum 12-2: t-252 .. t-21
        try:
            rec["mom_12_2"] = float(close.iloc[-21] / close.iloc[-252] - 1.0)
        except Exception:
            pass
        try:
            rec["mom_6m"] = float(close.iloc[-21] / close.iloc[-126] - 1.0)
        except Exception:
            pass
        rec["str_1m"] = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        rec["wh52"] = float(close.iloc[-1] / close.iloc[-252:].max())
        rec["maxret"] = float(ret.iloc[-21:].max())
        rec["price"] = float(close.iloc[-1])
        dollar = (close * vol).iloc[-63:]
        adv = float(dollar.mean())
        if adv > 0:
            rec["amihud"] = float((ret.iloc[-63:].abs() /
                                   dollar.reindex(ret.index).iloc[-63:]).mean() * 1e9)
        r1y = ret.iloc[-252:]
        rec["iskew"] = float(r1y.skew())
        rec["realized_vol"] = float(r1y.std() * math.sqrt(252))
        if spy_ret is not None:
            j = pd.concat([r1y, spy_ret], axis=1, join="inner").dropna()
            if len(j) > 200:
                var = float(j.iloc[:, 1].var())
                beta = float(j.cov().iloc[0, 1] / var) if var > 0 else None
                if beta is not None:
                    rec["beta"] = beta
                    resid = j.iloc[:, 0] - beta * j.iloc[:, 1]
                    rec["ivol"] = float(resid.std() * math.sqrt(252))
        out[t] = rec
    frame = pd.DataFrame.from_dict(out, orient="index")
    return {c: frame[c].dropna() for c in frame.columns}


_STMT_CACHE_KEY = "fmom_statements"


def get_statement_chars(tickers: list[str], sleep: float = 0.1,
                        max_age_hours: float = _FUND_TTL_HOURS) -> dict[str, dict]:
    """Best-effort annual-statement characteristics per ticker:
    asset_growth (YoY total assets) and gross_prof_assets (gross profit /
    total assets). Cached; one yfinance call pair per ticker, never raises."""
    cached = store_cas.cache_get(_STMT_CACHE_KEY, max_age_hours) or {}
    todo = [t for t in tickers if t not in cached]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        for i, t in enumerate(todo):
            rec: dict[str, float] = {}
            try:
                tk = yf.Ticker(t)
                bs = tk.balance_sheet
                if bs is not None and "Total Assets" in bs.index and bs.shape[1] >= 2:
                    a0, a1 = float(bs.loc["Total Assets"].iloc[0]), \
                             float(bs.loc["Total Assets"].iloc[1])
                    if a1 > 0:
                        rec["asset_growth"] = a0 / a1 - 1.0
                    inc = tk.income_stmt
                    if (inc is not None and "Gross Profit" in inc.index
                            and a0 > 0):
                        rec["gross_prof_assets"] = \
                            float(inc.loc["Gross Profit"].iloc[0]) / a0
            except Exception:
                pass
            cached[t] = rec
            if sleep:
                time.sleep(sleep)
            if i and i % 200 == 0:
                store_cas.cache_put(_STMT_CACHE_KEY, cached)
                print(f"[fmom] statements {i}/{len(todo)}")
        store_cas.cache_put(_STMT_CACHE_KEY, cached)
    return cached


def compute_chars(universe: list[dict], px: dict[str, pd.DataFrame],
                  fund: dict[str, dict],
                  stmt: dict[str, dict] | None = None) -> dict[str, pd.Series]:
    """All computable characteristics, keyed by CHAR_META names."""
    chars = _info_chars(fund)
    chars.update(_price_chars(px, px.get("SPY")))
    if stmt:
        frame = pd.DataFrame.from_dict(stmt, orient="index")
        for c in frame.columns:
            chars[c] = frame[c].dropna()
    # turnover needs both prices and shares outstanding via mktcap
    if "mktcap" in chars:
        dollar = {}
        for t, df in px.items():
            if df is not None and len(df) > 63 and t in chars["mktcap"].index:
                dollar[t] = float((df["close"] * df["volume"]).iloc[-63:].mean())
        adv = pd.Series(dollar)
        chars["turnover"] = (adv / chars["mktcap"]).dropna()
    tickers = {u["ticker"] for u in universe}
    return {k: v[v.index.isin(tickers)] for k, v in chars.items()}


def build_screens(universe: list[dict], chars: dict[str, pd.Series],
                  n: int = 30) -> dict:
    """screens_latest.json payload: per characteristic, the top-N Russell 1000
    names on the LONG side of the classic premium and the top-N on the SHORT
    side (see catalog.CHAR_META for orientation)."""
    from ..catalog import CHAR_META
    names = {u["ticker"]: u.get("name", "") for u in universe}
    mktcap = chars.get("mktcap", pd.Series(dtype=float))
    screens: dict[str, dict] = {}
    for key, meta in CHAR_META.items():
        c = chars.get(key)
        if c is None or len(c) < 40:
            continue
        asc = meta["high_is"] == "short"      # long side = lowest values
        ordered = c.sort_values(ascending=asc)

        def side(vals):
            return [{"ticker": t, "name": names.get(t, "")[:40],
                     "value": round(float(v), 6),
                     "mktcap_b": round(float(mktcap.get(t, float("nan"))) / 1e9, 2)
                     if t in mktcap.index else None}
                    for t, v in vals.items()]
        screens[key] = {"label": meta["label"], "desc": meta["desc"],
                        "high_is": meta["high_is"], "n_universe": int(len(c)),
                        "long": side(ordered.head(n)),
                        "short": side(ordered.tail(n).iloc[::-1])}
    return {"as_of": date.today().isoformat(),
            "universe": {"n": len(universe), "source": "VONE (Russell 1000)"},
            "note": ("Top names on each side of every computable characteristic "
                     "— the LONG list fits the factor's classic premium side. "
                     "Current constituents only; a screen, not a portfolio."),
            "n_screens": len(screens), "screens": screens}


def get_etf_top_holdings(tickers: list[str],
                         max_age_hours: float = _FUND_TTL_HOURS) -> dict:
    """Best-effort top-10 holdings per proxy ETF via yfinance funds_data."""
    cached = store_cas.cache_get("fmom_etf_holdings", max_age_hours) or {}
    todo = [t for t in tickers if t not in cached]
    if todo:
        try:
            import yfinance as yf
        except Exception:
            return cached
        for t in todo:
            rows = []
            try:
                th = yf.Ticker(t).funds_data.top_holdings
                if th is not None:
                    for sym, r in th.iterrows():
                        rows.append({"ticker": str(sym),
                                     "name": str(r.get("Name", ""))[:40],
                                     "weight": round(float(r.get("Holding Percent", 0.0)), 4)})
            except Exception:
                pass
            cached[t] = rows
        store_cas.cache_put("fmom_etf_holdings", cached)
    return cached


def build_holdings(universe: list[dict], px: dict[str, pd.DataFrame],
                   fund: dict[str, dict],
                   chars: dict[str, pd.Series] | None = None) -> dict:
    """holdings_latest.json payload: per-factor six-bin membership."""
    if chars is None:
        chars = compute_chars(universe, px, fund)

    mktcap = chars.get("mktcap", pd.Series(dtype=float))
    n_universe = len(universe)
    names = {u["ticker"]: u.get("name", "") for u in universe}
    factors: dict[str, dict] = {}
    for row in replicable_rows():
        c = chars.get(row["yf_char"])
        if c is None or c.empty:
            continue
        c = c[c.index.isin(names)]
        out = core.build_factor_bins(c, mktcap[mktcap.index.isin(c.index)])
        if not out["bins"]:
            continue
        long_high = row["long_side"] != "low"
        long_keys = ["LH", "SH"] if long_high else ["LL", "SL"]
        short_keys = ["LL", "SL"] if long_high else ["LH", "SH"]
        trim = {k: v if k[1] != "M" else [] for k, v in out["bins"].items()}
        factors[row["abbr"]] = {
            "name": row["name"], "family": row["family"],
            "char": row["yf_char"], "long_side": row["long_side"],
            "note": row.get("note"),
            "coverage": round(len(c) / n_universe, 4),
            "n_names": len(c),
            "size_split": out["size_split"],
            "long_bins": long_keys, "short_bins": short_keys,
            "bins": {k: [{**m, "name": names.get(m["ticker"], "")[:40]}
                         for m in v]
                     for k, v in trim.items()},
            "mid_counts": {k: len(v) for k, v in out["bins"].items()
                           if k[1] == "M"},
        }
    return {"as_of": date.today().isoformat(),
            "universe": {"n": n_universe, "source": "VONE (Russell 1000)"},
            "n_factors": len(factors),
            "construction": ("Gupta & Kelly: winsorize 1% -> median-mktcap "
                             "size split -> 30/40/30 characteristic buckets -> "
                             "value-weighted bins; factor = 0.5(LH+SH) - "
                             "0.5(LL+SL). Current holdings only — no "
                             "survivorship-biased backtest."),
            "factors": factors}

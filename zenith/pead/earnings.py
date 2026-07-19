"""PEAD earnings-data fetchers — ALL network access for the package lives here.

Sources (free, best-effort):
  * Nasdaq earnings calendar (api.nasdaq.com/api/calendar/earnings?date=...)
    — the daily driver: who reported, actual EPS, consensus forecast, number
    of estimates, last-year EPS, market cap. Same endpoint the Weekly Brief
    uses (zenith/brief/sources.py), extended to capture the actual EPS.
  * yfinance get_earnings_dates — per-ticker history (EPS estimate/reported)
    for the backfill and as the consensus fallback; also carries the NEXT
    scheduled report date used by the tracker's next-earnings exit.
  * yfinance quarterly income statement — quarterly revenue for the RUE
    component. Yahoo only serves ~5 quarters, so an accumulating cache merges
    each pull; RUE coverage grows over time and is flagged when missing.

Everything is cached through zenith.cas.store_cas so reruns within a day (and
resumed backfills) don't re-download.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from ..config import BROWSER_HEADERS, REQUEST_TIMEOUT
from ..cas import store_cas
from .signals import parse_eps

NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings"


def _mktcap_num(x) -> float | None:
    try:
        v = float(str(x).replace("$", "").replace(",", "").strip())
        return v if v > 0 else None
    except Exception:
        return None


def nasdaq_day(day: date, max_age_hours: float | None = None) -> list[dict]:
    """All reporters on one calendar date, normalized. Cached: past days for a
    week (the row set is final), today/future for 6h (rows fill in all day)."""
    key = f"pead_nasdaq_{day.isoformat()}"
    if max_age_hours is None:
        max_age_hours = 168.0 if day < date.today() else 6.0
    cached = store_cas.cache_get(key, max_age_hours=max_age_hours)
    if cached is not None:
        return cached
    rows = []
    try:
        r = requests.get(NASDAQ_URL, params={"date": day.isoformat()},
                         headers={**BROWSER_HEADERS, "Accept": "application/json"},
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        for row in (r.json().get("data") or {}).get("rows") or []:
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                continue
            rows.append({
                "ticker": sym.replace(".", "-"),
                "name": str(row.get("name", "")).strip(),
                "report_date": day.isoformat(),
                "time": str(row.get("time", "")),
                "eps_actual": parse_eps(row.get("eps")),
                "eps_consensus": parse_eps(row.get("epsForecast")),
                "last_year_eps": parse_eps(row.get("lastYearEPS")),
                "n_estimates": _int_or_none(row.get("noOfEsts")),
                "mktcap": _mktcap_num(row.get("marketCap")),
            })
        store_cas.cache_put(key, rows)
    except Exception:
        return cached or []
    return rows


def _int_or_none(x):
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def upcoming_calendar(days_fwd: int = 7,
                      universe_by_ticker: dict | None = None) -> list[dict]:
    """Scheduled reporters over the next days_fwd calendar days (today
    included), optionally filtered to a universe. Reuses nasdaq_day and its
    6h future-day cache; weekend dates are skipped (no announcements).
    Powers the EAP upcoming-reporters view."""
    out, seen = [], set()
    d = date.today()
    for _ in range(days_fwd + 1):
        if d.weekday() < 5:
            for rep in nasdaq_day(d):
                t = rep["ticker"]
                if universe_by_ticker is not None and t not in universe_by_ticker:
                    continue
                if (t, rep["report_date"]) in seen:
                    continue
                seen.add((t, rep["report_date"]))
                if rep.get("time", "").endswith("not-supplied"):
                    slot = infer_time(t, rep["report_date"])
                    if slot:
                        rep = {**rep, "time": slot}
                out.append(rep)
        d += timedelta(days=1)
    return out


# --- per-ticker earnings history (yfinance) ---------------------------------

def earnings_history(ticker: str, max_age_hours: float = 168.0) -> list[dict]:
    """[{date, eps_estimate, eps_actual}] newest-first, including FUTURE
    scheduled dates (estimate only) — powers backfill + next-earnings lookup."""
    key = f"pead_earnhist_{ticker}"
    cached = store_cas.cache_get(key, max_age_hours=max_age_hours)
    if cached is not None:
        return cached
    out = []
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).get_earnings_dates(limit=16)
        if df is not None and not df.empty:
            for ts, row in df.iterrows():
                est = row.get("EPS Estimate")
                act = row.get("Reported EPS")
                out.append({
                    "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10],
                    "hour": int(getattr(ts, "hour", 17)),   # <12 = pre-market report
                    "eps_estimate": None if pd.isna(est) else float(est),
                    "eps_actual": None if pd.isna(act) else float(act),
                })
        store_cas.cache_put(key, out)
    except Exception:
        return cached or []
    return out


def infer_time(ticker: str, report_date: str) -> str | None:
    """Classify a report's session slot from the yfinance timestamp hour when
    Nasdaq says 'time-not-supplied' (frequent, and wrong to treat pre-market
    reporters as after-hours: it would shift the reaction window a day late
    and miss the announcement move entirely)."""
    rd = date.fromisoformat(report_date)
    for h in earnings_history(ticker):
        try:
            if abs((date.fromisoformat(h["date"]) - rd).days) <= 3:
                return ("time-pre-market" if (h.get("hour") or 17) < 12
                        else "time-after-hours")
        except ValueError:
            continue
    return None


def next_earnings_date(ticker: str, after: date) -> str | None:
    """Next scheduled report date strictly after `after` (from the same cached
    yfinance earnings-dates frame)."""
    rows = earnings_history(ticker)
    future = sorted(r["date"] for r in rows if r["date"] > after.isoformat())
    return future[0] if future else None


def resolve_eps(rep: dict, ticker_history: list[dict] | None = None) -> dict:
    """Fill eps_actual / eps_consensus via the source ladder, stamping
    sue_source: 'nasdaq' -> 'yf' -> 'tsq' (last-year EPS seasonal proxy)."""
    out = dict(rep)
    if out.get("eps_actual") is not None and out.get("eps_consensus") is not None:
        out["sue_source"] = "nasdaq"
        return out
    hist = ticker_history if ticker_history is not None else earnings_history(out["ticker"])
    match = next((h for h in hist
                  if abs((date.fromisoformat(h["date"])
                          - date.fromisoformat(out["report_date"])).days) <= 3), None)
    if match:
        if out.get("eps_actual") is None and match["eps_actual"] is not None:
            out["eps_actual"] = match["eps_actual"]
        if out.get("eps_consensus") is None and match["eps_estimate"] is not None:
            out["eps_consensus"] = match["eps_estimate"]
            out["sue_source"] = "yf"
    if out.get("eps_consensus") is None and out.get("last_year_eps") is not None:
        out["eps_consensus"] = out["last_year_eps"]
        out["sue_source"] = "tsq"
    if "sue_source" not in out:
        out["sue_source"] = "nasdaq" if out.get("eps_actual") is not None else ""
    return out


# --- quarterly revenue (RUE component) --------------------------------------

def quarterly_revenue(ticker: str, max_age_hours: float = 168.0) -> pd.Series | None:
    """Quarterly Total Revenue, oldest-first. Yahoo serves only ~5 quarters, so
    the cache MERGES each pull with what it already holds — coverage of the
    8-quarter RUE requirement grows across quarters."""
    key = f"pead_rev_{ticker}"
    cached = store_cas.cache_get(key, max_age_hours=1e9) or {}   # merge base
    fresh = store_cas.cache_get(key, max_age_hours=max_age_hours)
    if fresh is None:
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).quarterly_income_stmt
            if df is not None and not df.empty and "Total Revenue" in df.index:
                for ts, v in df.loc["Total Revenue"].items():
                    if pd.isna(v):
                        continue
                    cached[str(ts)[:10]] = float(v)
            store_cas.cache_put(key, cached)
        except Exception:
            pass
    else:
        cached = fresh
    if not cached:
        return None
    ser = pd.Series(cached).sort_index()
    ser.index = pd.to_datetime(ser.index)
    return ser


def polite_sleep(seconds: float = 0.3) -> None:
    time.sleep(seconds)


def calendar_dates_for_reaction_day(r_day: date) -> list[date]:
    """Calendar dates whose reports map onto reaction day r_day: the prior
    trading day (after-hours) plus any weekend/holiday gap, plus r_day itself
    (pre-market)."""
    from .signals import prev_trading_day
    prev_td = prev_trading_day(r_day)
    out, d = [], prev_td
    while d <= r_day:
        out.append(d)
        d += timedelta(days=1)
    return out

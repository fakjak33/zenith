"""Cboe delayed-quote chains: fetch politely, parse into one tidy frame.

The endpoint is Cboe's public CDN JSON (the same one cboe.com's own quote
pages read): {timestamp, symbol, data: {<underlying fields>, options: [...]}}.
Each option row is keyed by an OCC symbol `ROOTyymmddCssssssss` (strike x1000)
and carries bid/ask/sizes, iv, open_interest, volume and greeks. Index
underlyings take a leading underscore (_SPX) and list several roots (SPX =
AM-settled monthlies, SPXW = PM-settled weeklies/dailies).

Politeness: one sequential session, a fixed gap between calls, exponential
backoff on 403/429/5xx and a circuit breaker so a blocked run degrades to the
yfinance fallback instead of hammering the CDN. Raw responses are cached for
the day in VOLSCREEN_CACHE_DIR (gitignored) — they are never committed.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BROWSER_HEADERS, CBOE_QUOTES_URL, VOLSCREEN_CACHE_DIR, VOLSCREEN_THROTTLE_S

# cash indices: Cboe symbol and the roots that make up the listed chain
INDEX_ROOTS = {"SPX": ("SPX", "SPXW"), "NDX": ("NDX", "NDXP"), "RUT": ("RUT", "RUTW"),
               "XSP": ("XSP",), "VIX": ("VIX", "VIXW")}
AM_SETTLED = {"SPX", "NDX", "RUT", "VIX"}          # monthly roots that settle at the open
COLS = ["root", "expiry", "cp", "strike", "bid", "ask", "bid_size", "ask_size", "iv", "oi",
        "volume", "delta", "gamma", "vega", "theta", "theo", "last"]


def cboe_symbol(ticker: str) -> str:
    return f"_{ticker}" if ticker in INDEX_ROOTS else ticker.replace("-", ".")


def roots_for(ticker: str) -> tuple[str, ...]:
    return INDEX_ROOTS.get(ticker, (ticker.replace("-", "").replace(".", ""),))


def parse_occ(sym: str) -> tuple[str, date, str, float]:
    """'SPXW261116C06000000' -> ('SPXW', 2026-11-16, 'C', 6000.0)."""
    root, tail = sym[:-15], sym[-15:]
    exp = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
    return root, exp, tail[6], int(tail[7:]) / 1000.0


def chain_frame(js: dict, ticker: str | None = None) -> pd.DataFrame:
    """Every option row as a DataFrame (COLS). Adjusted roots (e.g. 'AAPL1'
    after a corporate action) are dropped: they deliver non-standard deliverables."""
    opts = ((js or {}).get("data") or {}).get("options") or []
    if not opts:
        return pd.DataFrame(columns=COLS)
    keep = set(roots_for(ticker)) if ticker else None
    rows = []
    for o in opts:
        try:
            root, exp, cp, k = parse_occ(o["option"])
        except Exception:
            continue
        if keep is not None and root not in keep:
            continue
        rows.append((root, exp, cp, k, o.get("bid"), o.get("ask"), o.get("bid_size"),
                     o.get("ask_size"), o.get("iv"), o.get("open_interest"), o.get("volume"),
                     o.get("delta"), o.get("gamma"), o.get("vega"), o.get("theta"), o.get("theo"),
                     o.get("last_trade_price")))
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def underlying(js: dict) -> dict:
    """Spot and the day's bar from the chain payload (Cboe ships both)."""
    d = (js or {}).get("data") or {}
    def f(k):
        try:
            v = float(d.get(k))
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    spot = f("close") or f("current_price")
    ts = js.get("timestamp") if js else None
    return {"spot": spot, "open": f("open"), "high": f("high"), "low": f("low"),
            "close": f("close") or spot, "prev": f("prev_day_close"), "volume": f("volume"),
            "iv30_cboe": (f("iv30") / 100.0) if f("iv30") else None,
            "last_trade": d.get("last_trade_time"), "timestamp": ts,
            "security_type": d.get("security_type")}


def quote_time(js: dict) -> datetime:
    """Quote timestamp as naive US/Eastern wall clock. The payload's `timestamp`
    is UTC (its per-option last_trade_time fields are Eastern); the yfinance
    fallback stamps local Eastern time and says so with source='yfinance'."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    ts = (js or {}).get("timestamp") or ""
    try:
        t = datetime.fromisoformat(str(ts).replace(" ", "T"))
    except ValueError:
        return datetime.now(et).replace(tzinfo=None)
    if (js or {}).get("source") == "yfinance":
        return t
    return t.replace(tzinfo=ZoneInfo("UTC")).astimezone(et).replace(tzinfo=None)


def schema_ok(js: dict) -> bool:
    d = (js or {}).get("data") or {}
    opts = d.get("options") or []
    if not opts:
        return False
    need = {"option", "bid", "ask", "iv", "open_interest", "volume", "delta"}
    return need <= set(opts[0])


class Fetcher:
    """Sequential, throttled, cached. `get(ticker)` -> payload dict or None."""

    def __init__(self, cache_dir: Path = VOLSCREEN_CACHE_DIR, throttle: float = VOLSCREEN_THROTTLE_S,
                 offline_dir: Path | None = None, breaker: int = 10, day: str | None = None):
        self.cache_dir = cache_dir
        self.throttle = throttle
        self.offline_dir = offline_dir
        self.breaker = breaker
        self.day = day or date.today().isoformat()
        self.fails_in_row = 0
        self.tripped = False
        self.errors: dict[str, str] = {}
        self._session = None
        self._last = 0.0

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / self.day / f"{ticker}.json.gz"

    def _http(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({"User-Agent": BROWSER_HEADERS["User-Agent"],
                              "Accept": "application/json", "Accept-Encoding": "gzip, deflate"})
            self._session = s
        return self._session

    def get(self, ticker: str) -> dict | None:
        if self.offline_dir is not None:
            p = Path(self.offline_dir) / f"{ticker}.json"
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        p = self._path(ticker)
        if p.exists():
            try:
                return json.loads(gzip.decompress(p.read_bytes()))
            except Exception:
                pass
        if self.tripped:
            return None
        js = self._download(ticker)
        if js is not None:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(gzip.compress(json.dumps(js, separators=(",", ":")).encode()))
            except Exception:
                pass
        return js

    def _download(self, ticker: str) -> dict | None:
        url = CBOE_QUOTES_URL.format(sym=cboe_symbol(ticker))
        for attempt in range(4):
            wait = self.throttle - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self._http().get(url, timeout=45)
            except Exception as e:
                self.errors[ticker] = f"net: {str(e)[:80]}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    js = r.json()
                except ValueError:
                    self.errors[ticker] = "bad json"
                    break
                if not schema_ok(js):
                    self.errors[ticker] = "schema"
                    break
                self.fails_in_row = 0
                return js
            if r.status_code == 404:
                self.errors[ticker] = "404 (no listed options)"
                return None                     # a missing chain is not a CDN failure
            self.errors[ticker] = f"http {r.status_code}"
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        self.fails_in_row += 1
        if self.fails_in_row >= self.breaker:
            self.tripped = True
        return None

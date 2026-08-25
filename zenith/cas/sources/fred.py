"""FRED macro series — free, via the public CSV download endpoint (no API key,
no extra dependency). Used by the regime layer.

Returns series_id -> list of {date, value}. Degrades gracefully.

Three additions beyond the original 5-series CAS regime call (backward
compatible — every new parameter has a default that reproduces the old
behavior exactly):

  * `sleep` — a throttle between requests. Verified live during the REGIMES
    feature's exploration phase: fetching ~90 series back-to-back with no
    delay got this IP rate-limited (a control request that had worked
    moments earlier started timing out for over a minute). CAS's own 5-series
    pull never hit this; REGIMES' ~45-series pull would, every run.
  * `limit` — the old code hardcoded `pts[-520:]` (~2y of daily) with no way
    to opt out. REGIMES needs full history to reconstruct historical regimes,
    so `limit=None` returns everything; the default (520) keeps every
    existing caller's behavior unchanged.
  * `cache_key` — the old code used one shared cache key ("fred") for every
    caller, so CAS's 5-series set and brief's 20-series set already silently
    collide (brief works around it by forcing `max_age_hours=0.0`, which
    itself adds to rate-limit exposure). REGIMES uses its own key so its much
    larger, much-less-frequently-changing pull doesn't fight with either.
"""

from __future__ import annotations

import csv
import io
import time

import requests

from .. import store_cas

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIMEOUT = 20

# A small, durable macro set for regime detection
DEFAULT_SERIES = {
    "VIXCLS": "VIX",
    "T10Y2Y": "10y-2y curve",
    "BAMLH0A0HYM2": "HY OAS",
    "DTWEXBGS": "Broad USD",
    "DGS10": "10y yield",
}


def get_series(series_ids: list[str], max_age_hours: float = 18.0,
               sleep: float = 0.0, limit: int | None = 520,
               cache_key: str = "fred") -> tuple[dict[str, list], dict]:
    cached = store_cas.cache_get(cache_key, max_age_hours)
    if cached is not None:
        out = {s: cached.get(s, []) for s in series_ids}
        if limit is not None:
            out = {s: v[-limit:] for s, v in out.items()}
        return out, {"ok": True, "n": sum(len(v) for v in out.values()),
                     "source": "fred(cache)"}

    out: dict[str, list] = {}
    err = ""
    for i, sid in enumerate(series_ids):
        try:
            r = requests.get(CSV_URL, params={"id": sid}, timeout=TIMEOUT)
            r.raise_for_status()
            rdr = csv.reader(io.StringIO(r.text))
            rows = list(rdr)[1:]
            pts = []
            for row in rows:
                if len(row) >= 2 and row[1] not in (".", ""):
                    try:
                        pts.append({"date": row[0], "value": float(row[1])})
                    except ValueError:
                        pass
            if pts:
                out[sid] = pts            # full history cached; truncated on the way out below
        except Exception as e:
            err = str(e)[:160]
            continue
        if sleep and i < len(series_ids) - 1:
            time.sleep(sleep)

    if out:
        store_cas.cache_put(cache_key, out)
    result = out
    if limit is not None:
        result = {s: v[-limit:] for s, v in out.items()}
    return result, {"ok": bool(out), "n": sum(len(v) for v in result.values()),
                    "source": "fred", "error": "" if out else (err or "no series")}

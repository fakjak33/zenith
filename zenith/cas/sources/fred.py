"""FRED macro series — free, via the public CSV download endpoint (no API key,
no extra dependency). Used by the regime layer.

Returns series_id -> list of {date, value}. Degrades gracefully.
"""

from __future__ import annotations

import csv
import io

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


def get_series(series_ids: list[str], max_age_hours: float = 18.0) -> tuple[dict[str, list], dict]:
    key = "fred"
    cached = store_cas.cache_get(key, max_age_hours)
    if cached is not None:
        return {s: cached.get(s, []) for s in series_ids}, {
            "ok": True, "n": sum(len(cached.get(s, [])) for s in series_ids),
            "source": "fred(cache)"}

    out: dict[str, list] = {}
    err = ""
    for sid in series_ids:
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
                out[sid] = pts[-520:]            # ~2y of daily
        except Exception as e:
            err = str(e)[:160]
            continue

    if out:
        store_cas.cache_put(key, out)
    return out, {"ok": bool(out), "n": sum(len(v) for v in out.values()),
                 "source": "fred", "error": "" if out else (err or "no series")}

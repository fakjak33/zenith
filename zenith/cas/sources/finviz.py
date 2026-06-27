"""Finviz group performance — free sector / industry momentum & breadth.

Scrapes the Finviz "groups" performance table (sectors or industries) for
multi-timeframe performance (1w/1m/3m/6m/1y). Used by the themes segment as a
breadth/relative-strength input. Polite single request, cached, degrades
gracefully (Finviz can rate-limit or change markup).
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from .. import store_cas
from ...config import BROWSER_HEADERS

GROUPS_URL = "https://finviz.com/grp_export.ashx"   # CSV export of the groups table
TIMEOUT = 20


def get_groups(kind: str = "sector", max_age_hours: float = 18.0) -> tuple[list[dict], dict]:
    """kind: 'sector' or 'industry'. Returns list of {name, perf_week, perf_month,
    perf_quart, perf_half, perf_year, ...}."""
    key = f"finviz_{kind}"
    cached = store_cas.cache_get(key, max_age_hours)
    if cached is not None:
        return cached, {"ok": True, "n": len(cached), "source": "finviz(cache)"}

    group_code = {"sector": "sec", "industry": "ind"}.get(kind, "sec")
    try:
        r = requests.get(GROUPS_URL, params={"g": group_code, "v": "140"},
                         headers=BROWSER_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        return [], {"ok": False, "n": 0, "source": "finviz", "error": str(e)[:160]}

    df.columns = [c.strip().lower().replace(" ", "_").replace("%", "pct") for c in df.columns]
    recs = df.to_dict("records")
    if recs:
        store_cas.cache_put(key, recs)
    return recs, {"ok": bool(recs), "n": len(recs), "source": "finviz",
                  "error": "" if recs else "empty table"}

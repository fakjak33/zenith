"""REGIMES vintage audit (spec section 4 / 42) — measures how much data
REVISION (not publication timing, which macro.py's lag-shift already
handles) would have changed historical regime labels, on a handful of
headline drivers.

NOT part of the nightly Action and NEVER required by it — this needs a free,
user-registered FRED_API_KEY (https://fred.stlouisfed.org/docs/api/api_key.html)
read from the environment, and is meant to be run ONCE, locally, by the user
or maintainer:

    FRED_API_KEY=... python -m zenith.regimes.vintage

Without a key it degrades to an explicit "not run" state (`ran: False`) —
never a fabricated caveat number.

METHODOLOGY: for a given reference month M, "first published" queries the
FRED API's `observations` endpoint with `realtime_start`/`realtime_end`
pinned to a narrow window just after M's real publication lag (from
series.py's own `lag_of()`, plus a small buffer) — the value the API
returns is what was actually known on that date, which for a monthly series
is normally its first release for that reference period. "Current" queries
the same reference month with realtime pinned to today. The audit's
headline number is simply: of all reconstructed months, what fraction would
have been assigned a DIFFERENT quadrant if the classifier had used the
first-published values instead of today's revised ones.

LIVE-VERIFICATION STATUS (documented honestly, not silently): the exact
ALFRED/FRED vintage CSV endpoint shape was probed during this feature's
exploration and the probe was rate-limited before a real response could be
confirmed. This module targets FRED's documented JSON API
(api.stlouisfed.org/fred/series/observations) instead, which is the
supported, stable way to query vintages — but it has only been exercised
against MOCKED responses in this repo's test suite (tests/test_regimes.py),
not a live network call, because no FRED_API_KEY was available in the
development session. A user running this locally with their own key is the
first live test of the network path.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import requests

from .series import lag_of

API_URL = "https://api.stlouisfed.org/fred/series/observations"
TIMEOUT = 20

HEADLINE_DRIVERS = ("CPIAUCSL", "PAYEMS", "INDPRO", "GDPC1", "UNRATE", "PCEPILFE")


def _api_key() -> str:
    return os.environ.get("FRED_API_KEY", "").strip()


def _query(series_id: str, obs_start: str, obs_end: str,
          realtime_start: str, realtime_end: str, api_key: str) -> list[dict]:
    r = requests.get(API_URL, params={
        "series_id": series_id, "observation_start": obs_start, "observation_end": obs_end,
        "realtime_start": realtime_start, "realtime_end": realtime_end,
        "file_type": "json", "api_key": api_key,
    }, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("observations", [])


def first_published(series_id: str, ref_month: date, api_key: str) -> float | None:
    """The value first available for `ref_month`'s reference period —
    queries a realtime window starting at the series' own publication lag
    past `ref_month`'s end, with a 21-day buffer for release-calendar
    slippage."""
    lag = lag_of(series_id)
    window_start = ref_month + timedelta(days=lag)
    window_end = window_start + timedelta(days=21)
    obs = _query(series_id, ref_month.isoformat(), ref_month.isoformat(),
                window_start.isoformat(), window_end.isoformat(), api_key)
    vals = [float(o["value"]) for o in obs if o.get("value") not in (None, ".", "")]
    return vals[0] if vals else None


def current_value(series_id: str, ref_month: date, api_key: str, today: date | None = None) -> float | None:
    today = today or date.today()
    obs = _query(series_id, ref_month.isoformat(), ref_month.isoformat(),
                ref_month.isoformat(), today.isoformat(), api_key)
    vals = [float(o["value"]) for o in obs if o.get("value") not in (None, ".", "")]
    return vals[-1] if vals else None


def run_audit(series_ids: tuple[str, ...] = HEADLINE_DRIVERS,
             start_year: int = 2000, end_year: int | None = None) -> dict:
    """The full audit: for each series, for each month in [start_year, end_year],
    first-published vs current value. Returns `ran: False` with a reason if no
    API key is set — the nightly Action never sets one, by design."""
    api_key = _api_key()
    if not api_key:
        return {"ran": False, "reason": "FRED_API_KEY not set — this audit is local-only, "
                                        "opt-in, and never required by the nightly Action."}
    end_year = end_year or date.today().year
    months = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")

    results = {}
    for sid in series_ids:
        rows = []
        for m in months:
            ref = m.date()
            try:
                fp = first_published(sid, ref, api_key)
                cv = current_value(sid, ref, api_key)
            except Exception as e:
                rows.append({"month": ref.isoformat(), "error": str(e)[:160]})
                continue
            if fp is None or cv is None:
                continue
            pct_diff = None if fp == 0 else round((cv - fp) / abs(fp), 4)
            rows.append({"month": ref.isoformat(), "first_published": fp, "current": cv, "pct_diff": pct_diff})
        valid = [r for r in rows if "pct_diff" in r and r["pct_diff"] is not None]
        results[sid] = {
            "n_months": len(valid),
            "avg_abs_pct_diff": (round(sum(abs(r["pct_diff"]) for r in valid) / len(valid), 4)
                                 if valid else None),
            "rows": rows,
        }
    return {"ran": True, "as_of": date.today().isoformat(), "series": results}

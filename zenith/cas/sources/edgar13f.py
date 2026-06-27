"""SEC EDGAR 13F — quarterly institutional holdings, free (proxy for hedge-fund /
institutional positioning).

13F filings are quarterly and lagged ~45 days, so this is a slow-moving
positioning input. We query EDGAR full-text search for recent 13F-HR filings by a
small watch-list of well-known managers and record filing dates; per-holding
parsing is intentionally out of scope for the free MVP (the XML is large). Used
mainly to flag *when* fresh 13F data is available and which managers filed.

Degrades gracefully; EDGAR requires a descriptive User-Agent.
"""

from __future__ import annotations

import requests

from .. import store_cas

EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index?q=%22&forms=13F-HR"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
HEADERS = {"User-Agent": "ZenithResearch/0.1 (personal research; contact fakjak33@gmail.com)"}
TIMEOUT = 20

# A few well-known managers (CIK) as a starting watch-list; extend over time.
MANAGERS = {
    "Bridgewater": 1350694,
    "Renaissance Technologies": 1037389,
    "AQR Capital": 1167557,
    "Two Sigma": 1179392,
    "Millennium Mgmt": 1273087,
}


def recent_13f(max_age_hours: float = 168.0) -> tuple[list[dict], dict]:
    """Most recent 13F-HR filing date per watched manager."""
    key = "edgar13f"
    cached = store_cas.cache_get(key, max_age_hours)
    if cached is not None:
        return cached, {"ok": True, "n": len(cached), "source": "edgar(cache)"}

    out, err = [], ""
    for name, cik in MANAGERS.items():
        try:
            r = requests.get(EDGAR_SUBMISSIONS.format(cik=cik), headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            recent = r.json().get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            for f, d in zip(forms, dates):
                if f.startswith("13F"):
                    out.append({"manager": name, "cik": cik, "form": f, "filed": d})
                    break
        except Exception as e:
            err = str(e)[:160]
            continue

    if out:
        store_cas.cache_put(key, out)
    return out, {"ok": bool(out), "n": len(out), "source": "edgar",
                 "error": "" if out else (err or "no filings")}

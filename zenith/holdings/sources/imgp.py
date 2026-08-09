"""iM Global Partner fund-page adapter (DBMF and any sibling iMGP ETF).

The fund page renders its complete daily holdings table server-side, so there
is no JSON endpoint to call — the markup IS the API:

    <tr class="holding row">
      <td class="value_date">08/07/2026</td>
      <td class="security_name">US 2YR NOTE (CBT) SEP26</td>
      <td class="isin">ADI38LHK0</td>          <!-- labelled isin, holds a CUSIP -->
      <td class="ticker">TUU6</td>
      <td class="shares_qty">-3,879,200,000</td>
      <td class="market_value">$   -3,991,787,728.45</td>
      <td class="weight">-0.96</td>
    </tr>

Two things to know before touching this:

* The page 403s a bot user-agent but returns 200 for a browser one, so it goes
  out with `config.BROWSER_HEADERS` exactly like `pretom/universe.py`.
* The response is ~6 MB of WordPress for ~18 rows of data. We cache the PARSED
  rows, never the HTML, and use the date-aware TTL from `pead/earnings.py`:
  a past day's holdings file is immutable, today's may still be updating.

robots.txt at www.imgp.com is `Disallow:` — i.e. permits everything.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import requests

from ...config import BROWSER_HEADERS
from ...cas import store_cas

KIND = "imgp_web"
TIMEOUT = 60
_CACHE_PREFIX = "holdings_imgp"

_CELLS = {
    "value_date": "value_date",
    "security_name": "security_name",
    "isin": "cusip",
    "ticker": "ticker",
    "shares_qty": "qty",
    "market_value": "notional",
    "weight": "weight",
}
_NUMERIC = ("qty", "notional", "weight")

# The header row carries the same `holding row` class as the data rows, so the
# selector alone is not enough — a real row must lead with a real date.
_DATE_CELL = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\s*$")


def to_number(raw: str | None):
    """Coerce a custodian cell to a float.

    Handles ``"$                    -3,991,787,728.45"`` (the markup pads the
    dollar sign away from the digits), parenthesised negatives, and the ``"-"``
    placeholder used for "not applicable" — which must become None, not 0.
    """
    if raw is None:
        return None
    s = str(raw).replace("\xa0", " ").strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s in ("", "-", "--", "N/A", "NA", "n/a"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse(html: str) -> tuple[list[dict], dict]:
    """Extract holdings rows from the fund page. Pure — no network."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    rows: list[dict] = []
    skipped_header = 0
    for tr in soup.select("tr.holding.row"):
        row: dict = {}
        for cls, key in _CELLS.items():
            td = tr.find("td", class_=cls)
            txt = td.get_text(" ", strip=True) if td else ""
            row[key] = to_number(txt) if key in _NUMERIC else txt
        if not _DATE_CELL.match(row.get("value_date") or ""):
            skipped_header += 1
            continue
        if any(row.get(k) not in (None, "") for k in ("security_name", "ticker")):
            rows.append(row)

    dates = [r["value_date"] for r in rows if r.get("value_date")]
    meta = {"kind": KIND, "value_date": dates[0] if dates else "",
            "n_rows": len(rows), "html_bytes": len(html or ""),
            "skipped_non_data_rows": skipped_header}
    if len(set(dates)) > 1:
        # Every row in one publication carries the same date; more than one
        # means the page is mid-update or the markup changed.
        meta["mixed_value_dates"] = sorted(set(dates))
    return rows, meta


def _download(url: str) -> tuple[str | None, str]:
    """Direct browser-UA GET, falling back to Zenith's tiered fetcher."""
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.encoding or r.apparent_encoding
        return r.text, "direct"
    except Exception:
        # Firecrawl -> Apify tiers; cheap insurance if the site adds a bot wall.
        from ... import fetch as zfetch
        html, via = zfetch.get_html(url)
        return html, via or "blocked"


def fetch(fund, max_age_hours: float | None = None) -> tuple[list[dict], dict]:
    """Fetch + parse today's holdings for `fund`, via a parsed-row cache."""
    key = f"{_CACHE_PREFIX}_{fund.key}_{date.today().isoformat()}"
    if max_age_hours is None:
        max_age_hours = 6.0            # today's file may still be republished
    cached = store_cas.cache_get(key, max_age_hours)
    if cached and cached.get("rows"):
        meta = dict(cached["meta"])
        meta["via"] = f"cache:{meta.get('via', 'direct')}"
        return cached["rows"], meta

    html, via = _download(fund.source_url)
    if not html:
        return [], {"kind": KIND, "url": fund.source_url, "via": via,
                    "error": "no html returned", "value_date": "",
                    "html_bytes": 0}

    rows, meta = parse(html)
    meta.update({"url": fund.source_url, "via": via,
                 "retrieved_at": datetime.now(timezone.utc)
                 .strftime("%Y-%m-%dT%H:%M:%SZ")})
    if rows:
        store_cas.cache_put(key, {"rows": rows, "meta": meta})
    return rows, meta

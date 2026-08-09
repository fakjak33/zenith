"""Wayback Machine replay — backfill history from archived copies of the page.

Daily snapshots only exist from the day we start scraping, which would leave
the historical matrix empty for months. The Internet Archive has crawled the
iMGP fund page on and off (dense in places, sparse in others), and because it
stores the raw HTML we can run those captures through the *same* parser as a
live fetch and recover real, dated snapshots.

Coverage is genuinely patchy — that is a property of the archive, not a bug.
Missing days stay missing and are rendered as gaps rather than interpolated.

A second, subtler kind of gap: the Archive truncates large captures at 5 MiB,
and the iMGP page is ~6 MB with the holdings table near the bottom. Those
captures come back looking healthy (5,242,8xx bytes, HTTP 200) but parse to
zero rows. They are counted as failures and skipped — there is nothing to
recover from them, so do not go looking for a parser bug when it happens.

Three operational notes learned the hard way:
  * an unbounded CDX query times out — always pass `limit` and `from`;
  * the Archive answers 498 to a generic Chrome user-agent. It wants a UA that
    identifies the caller, so this module sends Zenith's polite UA rather than
    the browser one every other fetcher here uses;
  * `.../web/<ts>id_/<url>` returns the ORIGINAL bytes; without the `id_`
    suffix the Archive injects its own toolbar into the markup.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import requests

from ...config import USER_AGENT
from . import imgp

KIND = "wayback"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
SNAP_URL = "https://web.archive.org/web/{ts}id_/{url}"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
TIMEOUT = 90
POLITE_SLEEP = 1.0          # the Archive is a donation-funded service
CDX_LIMIT = 400


def list_snapshots(url: str, since: date | None = None,
                   limit: int = CDX_LIMIT, tries: int = 3) -> list[str]:
    """Distinct-day capture timestamps for `url`, oldest first.

    `collapse=timestamp:8` keeps one capture per calendar day. The CDX service
    504s under load often enough that a single failure means nothing — retry
    with a widening pause before concluding a URL has no captures.
    """
    params = {
        "url": url,
        "output": "json",
        "collapse": "timestamp:8",
        "filter": "statuscode:200",
        "fl": "timestamp",
        "limit": str(limit),
    }
    if since:
        params["from"] = since.strftime("%Y%m%d")
    for attempt in range(tries):
        try:
            r = requests.get(CDX_URL, params=params, headers=HEADERS,
                             timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data and len(data) > 1:
                return [row[0] for row in data[1:] if row and row[0]]
            return []
        except Exception as e:
            if attempt == tries - 1:
                print(f"[holdings] wayback CDX gave up on {url}: {str(e)[:120]}")
                return []
            time.sleep(3 * (attempt + 1))
    return []


def snapshot_day(ts: str) -> str:
    """`20260507094919` -> `2026-05-07` (the CAPTURE day, not the value date)."""
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"


def fetch_snapshot(ts: str, url: str) -> tuple[list[dict], dict]:
    """Fetch one archived capture and parse it with the live parser."""
    src = SNAP_URL.format(ts=ts, url=url)
    try:
        r = requests.get(src, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.encoding or r.apparent_encoding
        html = r.text
    except Exception as e:
        return [], {"kind": KIND, "url": src, "via": f"wayback:{ts}",
                    "error": str(e)[:200], "value_date": "", "html_bytes": 0}

    rows, meta = imgp.parse(html)
    meta.update({
        "kind": KIND,
        "url": src,
        "via": f"wayback:{ts}",
        "capture_day": snapshot_day(ts),
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    return rows, meta


def candidate_urls(fund) -> list[str]:
    """Live page first, then any legacy URLs the fund used to publish at."""
    return [fund.source_url, *getattr(fund, "legacy_urls", ())]


def replay(fund, days: int = 365, skip_days: set[str] | None = None,
           limit: int = CDX_LIMIT):
    """Yield ``(rows, meta)`` for each archived capture worth parsing.

    `skip_days` are CAPTURE days already archived locally; skipping them keeps
    a re-run cheap and makes the backfill resumable after a CI timeout.
    """
    skip = skip_days or set()
    since = date.today() - timedelta(days=days)
    seen_ts: set[str] = set()
    for url in candidate_urls(fund):
        stamps = list_snapshots(url, since=since, limit=limit)
        for ts in stamps:
            if ts in seen_ts or snapshot_day(ts) in skip:
                continue
            seen_ts.add(ts)
            rows, meta = fetch_snapshot(ts, url)
            time.sleep(POLITE_SLEEP)
            yield rows, meta

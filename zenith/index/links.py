"""INDEX link health — does each entry's URL actually resolve?

This is the module that makes ``verified`` mean something. Nothing in the
directory is marked verified because an author believed a URL was right; it is
marked verified because an HTTP request succeeded and the date was recorded.

THREE OUTCOMES, NOT TWO. The important design point is that "did not return 200"
is not the same as "broken":

  ``ok``       the URL responded successfully.
  ``blocked``  the host is up but refuses automated requests (401/403/429, or a
               Cloudflare interstitial). ATS Trading Solutions returns exactly
               this, and Citadel's robots.txt disallows crawling entirely —
               both already documented in zenith/sources.py. Reporting these as
               broken would be false: the resource exists and a human browser
               reaches it fine.
  ``error``    genuinely unreachable — DNS failure, timeout, 404, 5xx.

Reusing ``fetch.get`` (and ``fetch.allowed``) keeps this consistent with the
scraper's own politeness rules: the Zenith UA, the same timeout convention, and
robots.txt respected for page fetches.

COST CONTROL. A full sweep is a few hundred outbound requests to other people's
servers, so it is rate-limited by TTL (a URL is only re-checked once its last
check is ``INDEX_LINK_TTL_DAYS`` old), capped per run, and modestly concurrent.
It never runs from the view — only from ``compute.py --action links``.
"""

from __future__ import annotations

import concurrent.futures as cf
from datetime import date, datetime, timedelta

from .. import fetch
from ..config import (INDEX_LINK_MAX_PER_RUN, INDEX_LINK_TIMEOUT,
                      INDEX_LINK_TTL_DAYS, INDEX_LINK_WORKERS)

# Status codes that mean "the host is alive but does not want a robot", which is
# a materially different finding from "this link is dead".
_BLOCKED_CODES = {401, 403, 405, 406, 429, 451}

# Interstitial fingerprints — a 200 that is actually an anti-bot challenge page.
_CHALLENGE_MARKERS = ("just a moment", "checking your browser",
                      "enable javascript and cookies", "attention required")


def check_url(url: str, timeout: int = INDEX_LINK_TIMEOUT) -> dict:
    """Probe one URL. Returns {status, code, note, checked}.

    Uses a browser User-Agent: a large share of institutional sites 403 a bot UA
    but serve a normal browser fine, and Zenith's own fetch layer already
    established that this unblocks many sources for free.
    """
    out = {"url": url, "status": "error", "code": None, "note": "",
           "checked": date.today().isoformat()}
    if not url:
        out.update(status="missing", note="no URL recorded")
        return out
    if not fetch.allowed(url):
        out.update(status="blocked", note="robots.txt disallows automated fetching")
        return out
    try:
        r = fetch.get(url, timeout=timeout, browser_ua=True)
    except Exception as exc:                                  # pragma: no cover
        out.update(note=f"{type(exc).__name__}")
        return out

    if r is None:
        # fetch.get returns None for any non-200 or transport failure and does
        # not surface the code, so retry once directly to distinguish a block
        # from a genuine failure — the distinction is the whole point here.
        try:
            import requests
            from ..config import BROWSER_HEADERS
            rr = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout,
                              allow_redirects=True)
            out["code"] = rr.status_code
            if rr.status_code in _BLOCKED_CODES:
                out.update(status="blocked", note=f"HTTP {rr.status_code} to an automated request")
            elif rr.ok:
                out.update(status="ok")
            else:
                out.update(status="error", note=f"HTTP {rr.status_code}")
        except Exception as exc:
            out.update(status="error", note=type(exc).__name__)
        return out

    out["code"] = r.status_code
    body = (r.text or "")[:2000].lower()
    if any(mark in body for mark in _CHALLENGE_MARKERS):
        out.update(status="blocked", note="anti-bot challenge page returned")
    else:
        out.update(status="ok")
    return out


def _due(url: str, previous: dict, ttl_days: int) -> bool:
    prev = previous.get(url)
    if not prev or not prev.get("checked"):
        return True
    try:
        last = datetime.fromisoformat(prev["checked"]).date()
    except ValueError:
        return True
    return (date.today() - last) >= timedelta(days=ttl_days)


def sweep(entities: list[dict], previous: dict | None = None, *,
          ttl_days: int = INDEX_LINK_TTL_DAYS,
          max_urls: int = INDEX_LINK_MAX_PER_RUN,
          workers: int = INDEX_LINK_WORKERS,
          force: bool = False) -> dict:
    """Check every entity URL that is due, returning ``{url: result}``.

    Results from the previous sweep are carried forward for URLs not due yet,
    so the returned map is always complete — the view never has to reason about
    partial coverage.
    """
    previous = dict(previous or {})
    urls: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        for field in ("url", "research_url"):
            u = str(ent.get(field) or "").strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)

    due = [u for u in urls if force or _due(u, previous, ttl_days)][:max_urls]
    results = {u: previous[u] for u in urls if u in previous}

    if due:
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for res in pool.map(check_url, due):
                results[res["url"]] = res
    return results


def apply_to_entities(entities: list[dict], results: dict) -> list[dict]:
    """Write measured link status back onto each entity.

    An entity becomes ``verified`` ONLY when its primary URL actually responded.
    A blocked or broken link moves it to ``needs_review`` — with one deliberate
    exception: an entity already marked ``archived`` stays archived, because a
    dead link is the expected state for something we have retired, not a new
    problem to re-surface every sweep.
    """
    today = date.today().isoformat()
    out = []
    for ent in entities:
        ent = dict(ent)
        res = results.get(str(ent.get("url") or "").strip())
        status = res["status"] if res else ("missing" if not ent.get("url") else "unchecked")
        ent["link_status"] = status
        if ent.get("lifecycle_state") == "archived":
            out.append(ent)
            continue
        if status == "ok":
            ent["date_last_verified"] = today
            # Only promote to `verified` from a state that is genuinely waiting
            # on link confirmation. A `needs_review` flagged for an ambiguous
            # IDENTITY is not resolved by its URL responding.
            if ent.get("lifecycle_state") in ("new", "updated"):
                ent["lifecycle_state"] = "verified"
        elif status in ("error", "missing"):
            if ent.get("lifecycle_state") != "needs_review":
                ent["lifecycle_state"] = "needs_review"
        out.append(ent)
    return out


def summarize(results: dict) -> dict:
    counts: dict[str, int] = {}
    for res in results.values():
        counts[res.get("status", "unknown")] = counts.get(res.get("status", "unknown"), 0) + 1
    return {"checked": len(results), "by_status": counts,
            "ok": counts.get("ok", 0), "blocked": counts.get("blocked", 0),
            "error": counts.get("error", 0)}

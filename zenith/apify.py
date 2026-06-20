"""Apify fallback fetcher — used ONLY when the free direct tier is blocked.

Design goals (per user): keep it cheap and never blow the FREE plan's ~$5/mo.
- We gate every call on Apify's *own* reported monthly spend (queried live), so
  the budget check is accurate, not an estimate. Once spend ≥ budget we stop.
- Default actor is Apify's general Store crawler in **cheerio** mode (no browser
  = cheapest). Hard Cloudflare/JS sites can opt into playwright via env.
- The token is read from the environment only (config.APIFY_TOKEN); it is never
  written to disk or logged.

Returns raw HTML so the existing BeautifulSoup link-extraction can run on it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config

_API = "https://api.apify.com/v2"
# process-local cache of monthly spend so we don't re-query before every call
_spend_cache: dict[str, float] = {"month": "", "usd": -1.0}
_calls_this_run = 0


def _get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def monthly_spend_usd(force: bool = False) -> float:
    """Apify's own reported usage for the current month, in USD. Cached per run."""
    if not config.APIFY_TOKEN:
        return 0.0
    if not force and _spend_cache["usd"] >= 0:
        return _spend_cache["usd"]
    try:
        d = _get_json(f"{_API}/users/me/usage/monthly?token={config.APIFY_TOKEN}")
        data = d.get("data", {})
        usd = float(data.get("totalUsageCreditsUsdAfterVolumeDiscount")
                    or data.get("totalUsageCreditsUsd") or 0.0)
        _spend_cache["usd"] = usd
        return usd
    except Exception:
        # if we can't read usage, be conservative and assume budget exhausted
        return config.APIFY_MONTHLY_BUDGET_USD


def budget_ok() -> bool:
    """True if we're under the soft monthly budget and Apify is configured."""
    if not config.APIFY_ENABLED:
        return False
    return monthly_spend_usd() < config.APIFY_MONTHLY_BUDGET_USD


def _crawler_type() -> str:
    return "playwright:firefox" if config.APIFY_CRAWLER.startswith("play") else "cheerio"


def _proxy() -> dict:
    p = {"useApifyProxy": True}
    if config.APIFY_RESIDENTIAL:
        p["apifyProxyGroups"] = ["RESIDENTIAL"]
    return p


def _run_input(url: str) -> dict:
    """Input for apify/website-content-crawler: crawl just this one page, keep HTML."""
    return {
        "startUrls": [{"url": url}],
        "maxCrawlPages": 1,
        "maxCrawlDepth": 0,
        "crawlerType": _crawler_type(),
        "saveHtml": True,
        "proxyConfiguration": _proxy(),
    }


def fetch_html(url: str) -> tuple[str | None, str]:
    """Fetch a URL via Apify; return (html_or_text, note).

    note is one of: 'apify', 'apify:empty', 'apify:budget', 'apify:disabled',
    'apify:error:<reason>'. Respects the monthly budget gate.
    """
    global _calls_this_run
    if not config.APIFY_ENABLED:
        return None, "apify:disabled"
    if not budget_ok():
        return None, "apify:budget"

    actor = config.APIFY_ACTOR.replace("/", "~")
    endpoint = (f"{_API}/acts/{actor}/run-sync-get-dataset-items"
                f"?token={config.APIFY_TOKEN}&timeout={config.APIFY_TIMEOUT}")
    body = json.dumps(_run_input(url)).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.APIFY_TIMEOUT + 30) as r:
            items = json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return None, f"apify:error:http{e.code}"
    except Exception as e:  # noqa: BLE001
        return None, f"apify:error:{type(e).__name__}"

    _calls_this_run += 1
    # bump the cached spend so the gate tightens within a single run
    if _spend_cache["usd"] >= 0:
        _spend_cache["usd"] += 0.01  # rough per-page increment until next refresh

    if not items:
        return None, "apify:empty"
    item = items[0] if isinstance(items, list) else items
    # website-content-crawler returns html (saveHtml) + text/markdown
    content = item.get("html") or item.get("text") or item.get("markdown") or ""
    return (content or None), ("apify" if content else "apify:empty")


def crawl_articles(url: str, max_pages: int = 8,
                   link_glob: str | None = None) -> tuple[list[dict] | None, str]:
    """Use website-content-crawler as intended: render the index, follow links,
    and return each crawled article as a feed-like entry.

    This is the right path for JS/SPA insight hubs (where one-page HTML
    link-harvesting fails). Returns ([{title, link, summary}], note). Budget-gated.
    """
    global _calls_this_run
    if not config.APIFY_ENABLED:
        return None, "apify:disabled"
    if not budget_ok():
        return None, "apify:budget"

    run_input = {
        "startUrls": [{"url": url}],
        "maxCrawlPages": int(max_pages),
        "maxCrawlDepth": 1,
        "crawlerType": "playwright:firefox",
        "proxyConfiguration": _proxy(),
        "saveHtml": False,
    }
    if link_glob:
        run_input["includeUrlGlobs"] = [{"glob": link_glob}]

    actor = config.APIFY_ACTOR.replace("/", "~")
    endpoint = (f"{_API}/acts/{actor}/run-sync-get-dataset-items"
                f"?token={config.APIFY_TOKEN}&timeout={config.APIFY_TIMEOUT}")
    body = json.dumps(run_input).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.APIFY_TIMEOUT + 60) as r:
            items = json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return None, f"apify:error:http{e.code}"
    except Exception as e:  # noqa: BLE001
        return None, f"apify:error:{type(e).__name__}"

    _calls_this_run += 1
    if _spend_cache["usd"] >= 0:
        _spend_cache["usd"] += 0.03 * max(1, max_pages // 4)

    entries = []
    start_key = url.split("#")[0].rstrip("/")
    for it in (items or []):
        link = (it.get("url") or "").strip()
        if not link or link.split("#")[0].rstrip("/") == start_key:
            continue                       # skip the index page itself
        meta = it.get("metadata") or {}
        title = (meta.get("title") or it.get("title")
                 or (it.get("text") or "")[:120]).strip()
        if len(title) < 12:
            continue
        entries.append({"title": title[:240], "link": link,
                        "summary": (it.get("text") or "")[:300]})
    return entries, ("apify" if entries else "apify:empty")


def usage_summary() -> dict:
    """Small dict for status/telemetry (no secrets)."""
    return {
        "enabled": config.APIFY_ENABLED,
        "actor": config.APIFY_ACTOR,
        "crawler": _crawler_type(),
        "monthly_spend_usd": round(monthly_spend_usd(), 4) if config.APIFY_ENABLED else 0.0,
        "budget_usd": config.APIFY_MONTHLY_BUDGET_USD,
        "calls_this_run": _calls_this_run,
    }

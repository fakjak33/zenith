"""Firecrawl fallback fetcher — a free-tier alternative for JS / light anti-bot
sites that the direct tier can't reach, tried BEFORE the (budget-gated) Apify
tier because Firecrawl's free credits are the cheaper option.

Firecrawl renders JS and returns clean HTML + a link list, which suits both
article pages and insight-hub link harvesting. Enable by setting
FIRECRAWL_API_KEY (env / Streamlit secret / GitHub secret). No key ⇒ skipped.

Docs: https://docs.firecrawl.dev — POST /v1/scrape.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config

_API = "https://api.firecrawl.dev/v1/scrape"
_calls_this_run = 0


def enabled() -> bool:
    return bool(config.FIRECRAWL_API_KEY)


def fetch_html(url: str) -> tuple[str | None, str]:
    """Scrape ``url`` via Firecrawl; return (html_or_none, note).

    note ∈ {'firecrawl', 'firecrawl:empty', 'firecrawl:disabled',
    'firecrawl:error:<reason>'}.
    """
    global _calls_this_run
    if not enabled():
        return None, "firecrawl:disabled"

    body = json.dumps({
        "url": url,
        "formats": ["html"],
        "onlyMainContent": False,
        "timeout": config.FIRECRAWL_TIMEOUT * 1000,
    }).encode("utf-8")
    req = urllib.request.Request(_API, data=body, headers={
        "Authorization": f"Bearer {config.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=config.FIRECRAWL_TIMEOUT + 15) as r:
            payload = json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return None, f"firecrawl:error:http{e.code}"
    except Exception as e:  # noqa: BLE001
        return None, f"firecrawl:error:{type(e).__name__}"

    _calls_this_run += 1
    data = payload.get("data") or {}
    html = data.get("html") or data.get("rawHtml") or ""
    return (html or None), ("firecrawl" if html else "firecrawl:empty")


def usage_summary() -> dict:
    return {"enabled": enabled(), "calls_this_run": _calls_this_run}

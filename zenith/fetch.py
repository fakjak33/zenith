"""Polite fetching: robots.txt-aware HTTP + feed parsing."""

from __future__ import annotations

import urllib.robotparser as robotparser
from functools import lru_cache
from urllib.parse import urlsplit

import requests

from .config import USER_AGENT, REQUEST_TIMEOUT, BROWSER_HEADERS
from . import apify, firecrawl


@lru_cache(maxsize=256)
def _robots(host_scheme: str):
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{host_scheme}/robots.txt")
    try:
        rp.read()
    except Exception:
        return None
    return rp


def allowed(url: str) -> bool:
    """True if robots.txt permits our UA (fail-open if robots unreadable)."""
    try:
        s = urlsplit(url)
        rp = _robots(f"{s.scheme}://{s.netloc}")
        if rp is None:
            return True
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def get(url: str, timeout: int = REQUEST_TIMEOUT,
        browser_ua: bool = False) -> requests.Response | None:
    """GET with timeout; None on failure.

    ``browser_ua=True`` sends a realistic browser User-Agent (helps with sites
    that 403 a bot UA). Defaults to the polite Zenith UA for feed endpoints.
    """
    headers = dict(BROWSER_HEADERS) if browser_ua else {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None


def get_html(url: str, timeout: int = REQUEST_TIMEOUT) -> tuple[str | None, str]:
    """Hybrid page fetch for *blocked* sources.

    Tier 1 (free): direct request with a browser UA.
    Tier 2 (free-tier): Firecrawl, if FIRECRAWL_API_KEY is set.
    Tier 3 (paid, budget-gated): Apify, last resort.

    Returns (html_or_none, via) where via ∈ {'direct', 'firecrawl', 'apify',
    'apify:…' / 'firecrawl:…', 'blocked', 'robots'}.
    """
    if not allowed(url):
        return None, "robots"
    r = get(url, timeout=timeout, browser_ua=True)
    if r is not None and r.content:
        # honor the page's real charset (avoids mojibake on smart quotes etc.)
        r.encoding = r.apparent_encoding or r.encoding
        return r.text, "direct"
    # direct blocked/empty -> Firecrawl (free-tier) -> Apify (budget-gated)
    last = "blocked"
    if firecrawl.enabled():
        html, note = firecrawl.fetch_html(url)
        if html:
            return html, "firecrawl"
        last = note
    html, note = apify.fetch_html(url)
    if html:
        return html, "apify"
    if note != "apify:disabled":
        last = note
    return None, last


def parse_feed(url: str):
    """Return feedparser result (entries) for a feed URL, or None on failure."""
    import feedparser

    # NOTE: we do NOT apply robots.txt to feed endpoints — RSS/Atom feeds are
    # published expressly for syndication and feed readers fetch them directly.
    # robots.txt IS respected for article-page fetches during classification.
    try:
        # fetch ourselves so we control UA/timeout, then hand bytes to feedparser
        r = get(url)
        if r is None:
            return None
        parsed = feedparser.parse(r.content)
        if parsed.bozo and not parsed.entries:
            return None
        return parsed
    except Exception:
        return None

"""Polite fetching: robots.txt-aware HTTP + feed parsing."""

from __future__ import annotations

import urllib.robotparser as robotparser
from functools import lru_cache
from urllib.parse import urlsplit

import requests

from .config import USER_AGENT, REQUEST_TIMEOUT


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


def get(url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    """GET with our UA + timeout; None on failure."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None


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

"""Extract article links from a hub/insights page (for feed-less sources).

Many firms publish no RSS. For those we fetch their insights/research index
page (via the hybrid fetcher) and pull candidate article links out of the HTML,
producing feed-like dicts ({title, link, summary}) the scraper can treat just
like RSS entries. Heuristic and conservative: prefers links inside the main
content region, with real anchor text, optionally matching a URL pattern.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

# anchor text we never want (nav chrome, sharing, legal, etc.)
_JUNK = re.compile(r"^(home|about|contact|subscribe|sign in|log ?in|menu|search|"
                   r"privacy|terms|cookie|careers|read more|learn more|next|prev|"
                   r"previous|all|more|share|tweet|back)\b", re.I)
_SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|css|js|ico|woff2?)(\?|$)", re.I)


def extract_links(html: str, base_url: str, link_pattern: str | None = None,
                  max_items: int = 25) -> list[dict]:
    """Return [{title, link, summary}] of plausible article links from ``html``.

    ``link_pattern`` (regex) restricts hrefs to e.g. r"/insights/|/research/".
    Same-host links are preferred; cross-host links are dropped.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    region = (soup.find("main") or soup.find("article")
              or soup.find(attrs={"role": "main"}) or soup.body or soup)
    host = urlsplit(base_url).netloc.lower()
    pat = re.compile(link_pattern, re.I) if link_pattern else None

    seen: set[str] = set()
    out: list[dict] = []
    for a in region.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        url = urljoin(base_url, href)
        s = urlsplit(url)
        if s.scheme not in ("http", "https"):
            continue
        if s.netloc.lower() != host:           # stay on the firm's own domain
            continue
        if _SKIP_EXT.search(s.path):
            continue
        if pat and not pat.search(s.path + ("?" + s.query if s.query else "")):
            continue
        title = " ".join(a.get_text(" ", strip=True).split())
        if len(title) < 25 or _JUNK.match(title):   # require a headline-ish anchor
            continue
        key = url.split("#")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title[:240], "link": url, "summary": ""})
        if len(out) >= max_items:
            break
    return out

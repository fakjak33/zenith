"""Classify an insight as 'text' (text-only) or 'visual' (charts/tables/graphs).

Two-stage heuristic:
1. Cheap: inspect the feed entry itself for media (media_content/thumbnail/
   enclosures, or <img>/<figure>/<table> in the summary HTML).
2. Optional: a light GET of the article page, looking for figures/tables/charts
   or chart-ish keywords. Bounded by a per-run fetch budget for politeness.
Imperfect by nature — favors recall of 'visual'.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .fetch import get, allowed

_VISUAL_TAGS = ("img", "figure", "table", "svg", "picture", "canvas")
_CHART_WORDS = re.compile(r"\b(chart|figure|exhibit|graph|table|heatmap|infographic)\b", re.I)


def _html_has_visual(html: str) -> bool:
    if not html:
        return False
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(_VISUAL_TAGS):
        return True
    return False


def from_entry(entry) -> bool | None:
    """Return True/False if the feed entry alone is conclusive, else None."""
    if entry.get("media_content") or entry.get("media_thumbnail"):
        return True
    for enc in entry.get("enclosures", []) or []:
        if "image" in (enc.get("type") or ""):
            return True
    summary = entry.get("summary", "") or ""
    if _html_has_visual(summary):
        return True
    return None


def classify(entry, link: str, fetch_page: bool, budget: list[int]) -> str:
    """Return 'visual' or 'text'. ``budget`` is a 1-element list used as a
    mutable counter to cap page fetches across a run."""
    hint = from_entry(entry)
    if hint:
        return "visual"
    if fetch_page and budget[0] > 0 and link and allowed(link):
        budget[0] -= 1
        r = get(link, timeout=10)
        if r is not None and r.text:
            soup = BeautifulSoup(r.text, "html.parser")
            # focus on the article body when identifiable
            body = soup.find("article") or soup.find("main") or soup
            imgs = [i for i in body.find_all("img")
                    if not re.search(r"logo|icon|avatar|sprite", " ".join(
                        filter(None, [i.get("src", ""), i.get("alt", "")])), re.I)]
            if imgs or body.find(("figure", "table", "svg")):
                return "visual"
            if _CHART_WORDS.search(body.get_text(" ", strip=True)[:4000]):
                return "visual"
    return "text"

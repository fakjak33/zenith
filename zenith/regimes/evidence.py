"""REGIMES evidence board — mines Zenith's OWN existing scraped research
archive (data/archive/*.json, ~78 sources, already deduped and committed
nightly by the base scraper) for items relevant to a theme, and tags each
one Fact / Interpretation / Forecast / Speculation (spec section 20).

The tagging is a SIMPLE, TRANSPARENT, DOCUMENTED heuristic — keyword and
source-category based — not verified fact-checking, and every place this
module's output is shown says so. This is the deliberate, approved design
(over inventing a probability from mention-frequency, which would measure
narrative volume, not the underlying reality): no viral narrative becomes a
"regime signal" just by being mentioned often, and nothing here is scored
0-100 the way the quant themes are.
"""

from __future__ import annotations

from .. import store

FORECAST_WORDS = ("could ", "may ", "expected to", "likely to", "forecast", "outlook",
                  "projected", "is set to", "poised to")
SPECULATION_WORDS = ("speculat", "rumor", "unconfirmed", "some believe", "some say",
                     "reportedly", "sources say", "alleged")
CATEGORIES = ("Fact", "Interpretation", "Forecast", "Speculation")


def _classify(item: dict) -> str:
    text = ((item.get("title") or "") + " " + (item.get("summary") or "")).lower()
    if any(w in text for w in SPECULATION_WORDS):
        return "Speculation"
    if any(w in text for w in FORECAST_WORDS):
        return "Forecast"
    if item.get("category") == "research":
        # a working paper is itself an INTERPRETATION of underlying data, not the raw fact
        return "Interpretation"
    if item.get("category") == "news" and not any(w in text for w in FORECAST_WORDS + SPECULATION_WORDS):
        return "Fact"
    return "Interpretation"


def mine(keywords: list[str], lookback_days: int = 30, limit: int = 12) -> list[dict]:
    """Every recent archived item whose title or summary mentions ANY of
    `keywords` (case-insensitive substring match — simple and auditable),
    newest first, tagged with its heuristic category."""
    dates = store.archive_dates()[:lookback_days]
    kws = [k.lower() for k in keywords]
    hits = []
    for d in dates:
        for item in store.load_archive(d):
            text = ((item.get("title") or "") + " " + (item.get("summary") or "")).lower()
            if any(kw in text for kw in kws):
                hits.append({
                    "title": item.get("title"), "source": item.get("source"),
                    "link": item.get("link"), "published": item.get("published"),
                    "archive_date": d, "summary": item.get("summary"),
                    "category": _classify(item),
                })
    hits.sort(key=lambda h: h.get("published") or h.get("archive_date") or "", reverse=True)
    return hits[:limit]

"""INDEX Phase 2 — podcast feed registry and episode harvesting.

WHY A REGISTRY RATHER THAN GUESSED URLS. Podcast websites and their RSS feeds
are usually different hosts, and the feed URL is rarely derivable from the show
name: of eleven feed URLs guessed by pattern during exploration, most returned
404 or, worse, a DIFFERENT show's feed (two guessed Buzzsprout ids resolved to
the same unrelated podcast). So every feed below was resolved through Apple's
free, keyless **iTunes Search API** and then PROBED — the same "every enabled
URL was confirmed to return entries" discipline zenith/sources.py already
follows. ``feed_url`` is the pinned, verified URL; ``itunes_term`` is kept so a
feed that moves can be re-resolved without hand-editing.

ARCHIVE DEPTH IS REAL. These are not "latest 10 episodes" feeds. Verified
counts at pinning time (2026-09-01): Odd Lots 1,265 episodes back to 2015,
Top Traders Unplugged 961 back to 2014, Capital Allocators 820, Meb Faber 715,
Excess Returns 554, Rational Reminder 447, The Long View 389, Monetary Matters
297, Alpha Exchange 268, Money Maze 241, The Derivative 237, Flirting with
Models 126, COMPLEXITY 119, Other People's Money 76 — roughly 5,700 episodes.
That is what makes a guest database real harvested data rather than a
hand-compiled list.

ONE TRAP WORTH KNOWING: ``toptradersunplugged.com/feed/`` is the WordPress site
feed and returns only the latest 10 posts. The podcast host feed
(feeds.captivate.fm) is the one carrying the full archive. A show's own domain
is usually the wrong place to look.

POLITENESS. Harvesting reuses ``fetch.parse_feed`` (the Zenith UA with a browser
retry, robots deliberately not applied to feed endpoints since RSS is published
for syndication) and runs one feed at a time.
"""

from __future__ import annotations

import hashlib
import html as _html
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .. import fetch
from ..config import INDEX_EPISODE_SUMMARY_CHARS

ITUNES_SEARCH = "https://itunes.apple.com/search"


@dataclass(frozen=True)
class Podcast:
    name: str                       # MUST match the entity name in seed.py
    feed_url: str                   # pinned + probed
    itunes_term: str                # re-resolution query if the feed moves
    hosts: tuple[str, ...] = ()     # host names, excluded from guest extraction
    note: str = ""
    # Episode-title shapes this show actually uses, most specific first. Written
    # empirically from its real archive, never guessed — see guests.py.
    patterns: tuple[str, ...] = field(default_factory=tuple)


# The 14 shows the user asked to monitor. `patterns` names the extraction
# strategies in guests.py that apply to each; guests.py owns the regexes.
PODCASTS: tuple[Podcast, ...] = (
    Podcast("Top Traders Unplugged",
            "https://feeds.captivate.fm/top-traders-unplugged/",
            "Top Traders Unplugged",
            hosts=("Niels Kaastrup-Larsen", "Alan Dunne", "Rich Brennan",
                   "Richard Brennan", "Mark Rzepczynski", "Katy Kaminski",
                   "Cem Karsan", "Moritz Seibert"),
            note="Site feed toptradersunplugged.com/feed/ returns only 10 posts; "
                 "the Captivate host feed carries all 961.",
            patterns=("ft_suffix", "with_suffix")),
    Podcast("Flirting with Models",
            "https://feeds.captivate.fm/flirting-with-models/",
            "Flirting with Models",
            hosts=("Corey Hoffstein",),
            patterns=("dash_prefix", "with_suffix")),
    Podcast("The Derivative",
            "https://anchor.fm/s/11908b3c/podcast/rss",
            "The Derivative Jeff Malec",
            hosts=("Jeff Malec",),
            patterns=("with_suffix", "ft_suffix")),
    Podcast("Alpha Exchange",
            "https://feeds.simplecast.com/8g9ryFGf",
            "Alpha Exchange Dean Curnutt",
            hosts=("Dean Curnutt",),
            note="Richest guest metadata of any monitored show: titles are "
                 "literally 'Name, Role, Firm'.",
            patterns=("name_role_firm",)),
    Podcast("Excess Returns",
            "https://anchor.fm/s/9a1dfac/podcast/rss",
            "Excess Returns",
            hosts=("Jack Forehand", "Justin Carbonneau", "Matt Zeigler"),
            patterns=("with_suffix", "on_infix", "pipe_segments")),
    Podcast("Capital Allocators",
            "https://rss.libsyn.com/shows/94820/destinations/482814.xml",
            "Capital Allocators Ted Seides",
            hosts=("Ted Seides",),
            patterns=("dash_prefix", "colon_prefix")),
    Podcast("Other People's Money",
            "https://feeds.megaphone.fm/opm",
            "Other Peoples Money Max Wiethe",
            hosts=("Max Wiethe",),
            patterns=("pipe_segments", "with_suffix", "on_infix")),
    Podcast("Rational Reminder",
            "https://rationalreminder.libsyn.com/rss",
            "Rational Reminder Benjamin Felix Cameron Passmore",
            hosts=("Benjamin Felix", "Cameron Passmore", "Mark McGrath",
                   "Dan Bortolotti"),
            patterns=("colon_prefix", "paren_with", "with_suffix")),
    Podcast("The Meb Faber Show",
            "https://mebfaber.libsyn.com/rss",
            "The Meb Faber Show",
            hosts=("Meb Faber",),
            patterns=("pipe_segments", "on_infix", "dash_prefix", "colon_prefix")),
    Podcast("Money Maze Podcast",
            "https://audioboom.com/channels/5118743.rss",
            "Money Maze Podcast",
            hosts=("Simon Brewer", "Will Campion"),
            patterns=("with_suffix", "colon_prefix", "dash_prefix")),
    Podcast("Monetary Matters",
            "https://feeds.megaphone.fm/EWWMN1909747317",
            "Monetary Matters Jack Farley",
            hosts=("Jack Farley",),
            patterns=("pipe_segments", "with_suffix", "on_infix")),
    Podcast("Odd Lots",
            "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-"
            "ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/"
            "982f5071-765c-403d-969d-ae27003a8d83/podcast.rss",
            "Odd Lots",
            hosts=("Joe Weisenthal", "Tracy Alloway"),
            note="Deepest archive of any monitored show (1,265 episodes). Titles "
                 "are editorial rather than structured, so guest yield is lower "
                 "here than elsewhere and leans on the description.",
            patterns=("on_infix", "with_suffix")),
    Podcast("The Long View",
            "https://feeds.simplecast.com/5SEwkJYi",
            "The Long View Morningstar",
            hosts=("Christine Benz", "Dan Lefkovitz", "Jeff Ptak", "Amy Arnott"),
            patterns=("colon_prefix",)),
    Podcast("COMPLEXITY",
            "https://feeds.simplecast.com/OzDH_At2",
            "Complexity Santa Fe Institute",
            hosts=("Michael Garfield", "Abha Eli Phoboo"),
            patterns=("on_infix", "with_suffix", "colon_prefix")),
)

BY_NAME: dict[str, Podcast] = {p.name: p for p in PODCASTS}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Strip HTML, unescape entities, collapse whitespace.

    Feed descriptions are HTML fragments of wildly varying quality; everything
    downstream (guest extraction, the UI) wants flat text.
    """
    txt = _TAG_RE.sub(" ", _html.unescape(str(raw or "")))
    txt = txt.replace(" ", " ").replace("​", "")
    return _WS_RE.sub(" ", txt).strip()


def discover_feed(term: str, timeout: int = 20) -> tuple[str, str] | None:
    """Resolve a show name to its authoritative feed URL via the iTunes Search
    API. Free, keyless. Returns (collection_name, feed_url) or None.

    Used to re-pin a feed that has moved — not called during a normal harvest,
    because the pinned URLs above are already verified and hitting Apple on
    every run would be gratuitous.
    """
    r = fetch.get(f"{ITUNES_SEARCH}?term={_quote(term)}&entity=podcast&limit=1",
                  timeout=timeout, browser_ua=True)
    if r is None:
        return None
    try:
        results = r.json().get("results") or []
    except Exception:
        return None
    if not results:
        return None
    top = results[0]
    feed = top.get("feedUrl")
    return (top.get("collectionName", ""), feed) if feed else None


def _quote(s: str) -> str:
    from urllib.parse import quote_plus
    return quote_plus(str(s or ""))


def episode_id(podcast: str, guid: str, link: str, title: str) -> str:
    """Stable episode id. Prefers the feed's own GUID, which is the only field
    a publisher promises to keep stable; falls back to link, then title."""
    key = f"{podcast}|{guid or link or title}".lower().strip()
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]


def _published(entry) -> str:
    """ISO date from whichever date field the feed provides."""
    for attr in ("published_parsed", "updated_parsed"):
        tm = getattr(entry, attr, None)
        if tm:
            try:
                return date(tm.tm_year, tm.tm_mon, tm.tm_mday).isoformat()
            except (ValueError, TypeError):
                continue
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, "") or ""
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(raw).strip(), fmt).date().isoformat()
            except ValueError:
                continue
    return ""


def harvest_one(pod: Podcast) -> dict:
    """Fetch and normalise one podcast's full archive.

    Returns {podcast, feed_url, ok, n, episodes, error}. Never raises — a feed
    that has moved or gone down must degrade to an honest zero, exactly as the
    scraper's per-source status does, rather than breaking the whole run.
    """
    out = {"podcast": pod.name, "feed_url": pod.feed_url, "ok": False,
           "n": 0, "episodes": [], "error": "", "fetched": date.today().isoformat()}
    parsed = fetch.parse_feed(pod.feed_url)
    if parsed is None or not getattr(parsed, "entries", None):
        out["error"] = "feed returned no entries"
        return out

    episodes = []
    for e in parsed.entries:
        title = clean_text(getattr(e, "title", ""))
        if not title:
            continue
        summary = clean_text(getattr(e, "summary", "") or getattr(e, "description", ""))
        link = str(getattr(e, "link", "") or "")
        guid = str(getattr(e, "id", "") or getattr(e, "guid", "") or "")
        episodes.append({
            "id": episode_id(pod.name, guid, link, title),
            "podcast": pod.name,
            "title": title,
            # Descriptions run to many KB of show notes, sponsor copy and
            # timestamps. The guest's affiliation is stated in the opening
            # sentences, so we keep a bounded prefix: enough to mine, small
            # enough that ~5,700 episodes stay a reasonable committed artifact.
            "summary": summary[:INDEX_EPISODE_SUMMARY_CHARS],
            "published": _published(e),
            "url": link,
        })

    # Newest first, and de-duplicated by id: several of these feeds legitimately
    # republish an episode (a "best of" rerun keeps the original guid).
    seen: set[str] = set()
    unique = []
    for ep in sorted(episodes, key=lambda x: x["published"], reverse=True):
        if ep["id"] in seen:
            continue
        seen.add(ep["id"])
        unique.append(ep)

    out.update(ok=True, n=len(unique), episodes=unique)
    return out


def harvest(podcasts: tuple[Podcast, ...] = PODCASTS,
            only: str | None = None) -> tuple[list[dict], list[dict]]:
    """Harvest every registered podcast.

    Returns ``(episodes, registry_rows)`` where registry_rows records per-show
    health — count, date range and any error — so the UI can show which feeds
    are actually working rather than silently rendering a short list.
    """
    all_eps: list[dict] = []
    registry: list[dict] = []
    for pod in podcasts:
        if only and pod.name != only:
            continue
        res = harvest_one(pod)
        all_eps.extend(res["episodes"])
        dates = [e["published"] for e in res["episodes"] if e["published"]]
        registry.append({
            "podcast": pod.name,
            "feed_url": pod.feed_url,
            "itunes_term": pod.itunes_term,
            "hosts": list(pod.hosts),
            "note": pod.note,
            "ok": res["ok"],
            "error": res["error"],
            "episodes": res["n"],
            "earliest": min(dates) if dates else "",
            "latest": max(dates) if dates else "",
            "fetched": res["fetched"],
        })
    return all_eps, registry


def merge_episodes(stored: list[dict], fresh: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge a fresh harvest into the stored archive, preserving history.

    Returns ``(merged, new_episodes)``. Episodes already stored keep their
    original record — a publisher editing a title later must not silently
    rewrite what a guest was linked from — while genuinely new episodes are
    added and reported, which is what drives the "new episode detected" flow.
    """
    by_id = {e["id"]: e for e in stored if e.get("id")}
    new: list[dict] = []
    for ep in fresh:
        if ep["id"] not in by_id:
            by_id[ep["id"]] = ep
            new.append(ep)
    merged = sorted(by_id.values(),
                    key=lambda e: (e.get("published", ""), e.get("podcast", "")),
                    reverse=True)
    return merged, new

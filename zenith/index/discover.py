"""INDEX Phase 3 — the discovery surface.

The spec's ambition for this feature was that it become "an investment research
discovery engine, not simply an archive". That framing sets the bar: a panel
that lists the biggest names in the catalog is an archive with a leaderboard.
Discovery means surfacing things the user does NOT already know about.

So every finder below is defined against what the user already has. The
directory knows three things about its own provenance — which entries came from
the user's curated seed list, which Zenith's scraper already ingests, and which
were discovered by harvesting podcast archives — and that is exactly the
information needed to answer "who is here that I did not put here?"

  new_to_you()          frequent, well-connected people who are NOT in the seed
                        list and NOT already an ingested source. The core query.
  cross_pollinators()   people appearing across several different shows — the
                        names the wider ecosystem keeps returning to.
  bridges()             people who connect otherwise-separate parts of the
                        graph, found structurally rather than by popularity.
  discovered_firms()    organisations that surfaced only through the harvest.
  by_topic()            "who works on volatility?" answered from the tags.
  emerging()            recent first-appearances, so a new voice shows up.
  coverage_gaps()       podcasts and topics the catalog under-covers — an
                        honest look at what this directory is bad at.

NOTHING HERE INVENTS A FACT. Every result is a re-query of data already in the
catalog, with the provenance carried through so a reader can see why a row was
surfaced and how much to trust it.
"""

from __future__ import annotations

from datetime import date, datetime

from . import network as idx_network
from . import ranking as idx_ranking
from . import taxonomy as tx

SEED_PROVENANCE_MARK = "seed list"
HARVEST_MARK = "harvest"


def is_curated(ent: dict) -> bool:
    """Came from the user's own supplied resource list."""
    return SEED_PROVENANCE_MARK in str(ent.get("provenance", ""))


def is_discovered(ent: dict) -> bool:
    """Found by the harvest or the prior compilation, not supplied by the user."""
    prov = str(ent.get("provenance", ""))
    return not is_curated(ent) and (HARVEST_MARK in prov or "prior manual" in prov)


def new_to_you(entities: list[dict], scored: dict, limit: int = 40,
               min_appearances: int = 2) -> list[dict]:
    """People the user did not supply and Zenith does not already ingest.

    This is the discovery engine's central query. Someone who recurs across
    these archives, is not on the user's own list, and is not already a Zenith
    source is — almost by definition — a name worth a look.
    """
    out = []
    for ent in entities:
        if ent.get("entity_type") != "person" or not is_discovered(ent):
            continue
        if ent.get("zenith_source"):
            continue
        raw = (scored.get(ent["id"]) or {}).get("raw", {})
        if raw.get("appearances", 0) < min_appearances:
            continue
        out.append(ent)
    out.sort(key=lambda e: -(scored.get(e["id"], {}).get("score", 0.0)))
    return out[:limit]


def cross_pollinators(entities: list[dict], scored: dict, limit: int = 40,
                      min_shows: int = 3) -> list[dict]:
    """People the ecosystem keeps coming back to, across DIFFERENT shows.

    Breadth across shows is a different signal from repetition on one: a
    recurring co-host of a single podcast is not the same as somebody four
    separate programmes independently wanted to interview.
    """
    out = [e for e in entities
           if (scored.get(e["id"]) or {}).get("raw", {}).get("reach", 0) >= min_shows]
    out.sort(key=lambda e: (-(scored.get(e["id"], {}).get("raw", {}).get("reach", 0)),
                            -(scored.get(e["id"], {}).get("score", 0.0))))
    return out[:limit]


def bridges(entities: list[dict], relationships: list[dict], limit: int = 25
            ) -> list[dict]:
    """People who link parts of the graph that are otherwise unconnected.

    Found structurally: for each person, how many DISTINCT podcasts and firms do
    their edges touch? Somebody attached to five different organisations is a
    connector regardless of how often they appear, which is a different and
    more interesting property than raw popularity.
    """
    by_id = {e["id"]: e for e in entities}
    touch: dict[str, set[str]] = {}
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        if s not in by_id or t not in by_id:
            continue
        for a, b in ((s, t), (t, s)):
            if by_id[a].get("entity_type") == "person":
                other = by_id[b]
                if other.get("entity_type") in ("organisation", "podcast"):
                    touch.setdefault(a, set()).add(other["id"])

    rows = []
    for eid, others in touch.items():
        kinds = {by_id[o].get("entity_type") for o in others}
        # A bridge must span BOTH kinds; touching five podcasts and no firm is
        # a frequent guest, not a connector.
        if len(others) >= 3 and len(kinds) > 1:
            rows.append({"entity": by_id[eid], "spans": len(others),
                         "targets": sorted(by_id[o]["name"] for o in others)})
    rows.sort(key=lambda r: (-r["spans"], r["entity"]["name"]))
    return rows[:limit]


def discovered_firms(entities: list[dict], limit: int = 40) -> list[dict]:
    """Organisations that entered the catalog only through the harvest."""
    out = [e for e in entities
           if e.get("entity_type") == "organisation" and is_discovered(e)]
    out.sort(key=lambda e: e.get("name", "").lower())
    return out[:limit]


def by_topic(entities: list[dict], scored: dict, vocab: str, term: str,
             limit: int = 30) -> list[dict]:
    """Everything tagged with one taxonomy term, most central first.

    This is the query the spec framed as "who are the best systematic trading
    researchers?" — with the caveat that this answers "who in this catalog is
    most connected AND tagged systematic", which is a different and more
    honest question than "who is best".
    """
    field = {"investment_approach": "investment_approach",
             "asset_class": "asset_classes",
             "insight_type": "insight_types"}[vocab]
    slug = tx.resolve(vocab, term)
    out = [e for e in entities if slug in (e.get(field) or [])]
    out.sort(key=lambda e: -(scored.get(e["id"], {}).get("score", 0.0)))
    return out[:limit]


def emerging(entities: list[dict], guest_index: dict, within_days: int = 180,
             limit: int = 30, today: date | None = None) -> list[dict]:
    """People whose FIRST appearance in the archives is recent.

    Deliberately first-appearance rather than most-recent: a long-standing guest
    who was on last week is not a new voice, but somebody whose earliest
    appearance anywhere is three months old is.
    """
    today = today or date.today()
    rows = []
    for ent in entities:
        if ent.get("entity_type") != "person":
            continue
        prof = _profile_for(ent, guest_index)
        apps = (prof or {}).get("appearances") or []
        dates = sorted(a.get("published", "") for a in apps if a.get("published"))
        if not dates:
            continue
        try:
            first = datetime.fromisoformat(dates[0]).date()
        except ValueError:
            continue
        age = (today - first).days
        if 0 <= age <= within_days:
            rows.append({"entity": ent, "first_seen": dates[0], "days": age,
                         "appearances": len(apps),
                         "podcasts": (prof or {}).get("podcasts", [])})
    rows.sort(key=lambda r: (r["days"], r["entity"]["name"]))
    return rows[:limit]


def _profile_for(ent: dict, guest_index: dict) -> dict:
    for label in [ent.get("name", "")] + list(ent.get("aliases", [])):
        prof = guest_index.get(str(label).lower())
        if prof:
            return prof
    return {}


def coverage_gaps(entities: list[dict], podcast_doc: dict) -> list[dict]:
    """Where this directory is WEAK — stated plainly rather than hidden.

    A discovery surface that only ever reports success teaches the reader to
    trust it uniformly, which is wrong: guest extraction works far better on
    some shows than others, and several taxonomy terms have almost nothing
    filed under them. Both are worth knowing before relying on a filter.
    """
    gaps: list[dict] = []
    for show in sorted(podcast_doc.get("shows") or [],
                       key=lambda s: s.get("coverage", 0)):
        if show.get("coverage", 0) < 0.5:
            gaps.append({
                "kind": "podcast coverage",
                "subject": show["podcast"],
                # Deliberately does NOT assert a cause. Odd Lots' low yield is
                # demonstrably its editorial titling, but that explanation has
                # not been verified for every show, and inventing one per show
                # would be exactly the kind of confident guess this directory
                # is built to avoid.
                "detail": f"only {show['coverage']:.0%} of {show['episodes']:,} "
                          f"episodes yielded a parseable guest, so this show's "
                          f"guests are under-represented in the directory. Its "
                          f"episodes are all still in the archive.",
            })
    counts: dict[str, int] = {}
    for ent in entities:
        for slug in ent.get("investment_approach", []):
            counts[slug] = counts.get(slug, 0) + 1
    thin = [(s, counts.get(s, 0)) for s in tx.known("investment_approach")
            if counts.get(s, 0) <= 2]
    if thin:
        gaps.append({
            "kind": "thin taxonomy",
            "subject": f"{len(thin)} strategy tags",
            "detail": "Declared but barely used: "
                      + ", ".join(tx.label_of("investment_approach", s)
                                  for s, _n in thin[:12])
                      + ". Filtering on these will return almost nothing — the "
                        "vocabulary is broader than the catalog is deep.",
        })
    return gaps


def build(entities: list[dict], relationships: list[dict],
          guest_index: dict | None = None, podcast_doc: dict | None = None
          ) -> dict:
    """Everything the Discover view needs, computed once."""
    guest_index = guest_index or {}
    podcast_doc = podcast_doc or {}
    scored = idx_ranking.score_all(entities, relationships, guest_index)
    return {
        "scored": scored,
        "new_to_you": new_to_you(entities, scored),
        "cross_pollinators": cross_pollinators(entities, scored),
        "bridges": bridges(entities, relationships),
        "discovered_firms": discovered_firms(entities),
        "emerging": emerging(entities, guest_index),
        "gaps": coverage_gaps(entities, podcast_doc),
        "degrees": idx_network.degrees(entities, relationships),
    }

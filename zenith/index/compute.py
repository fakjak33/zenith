"""INDEX compute orchestrator.

    python -m zenith.index.compute --action auto      # rebuild catalog, no network
    python -m zenith.index.compute --action seed      # same, ignoring stored state
    python -m zenith.index.compute --action links     # live URL sweep (network)
    python -m zenith.index.compute --action podcasts  # harvest feeds + guests (network)
    python -m zenith.index.compute --action quality   # recompute status only

WHY `auto` DOES NOT TOUCH THE NETWORK. Every other Zenith package's nightly
action fetches data, but a directory has nothing to fetch: the catalog is
curated, not scraped. The only network step is the link sweep, which is
rate-limited, TTL-gated and run deliberately — so `auto` is a fast, offline
rebuild-and-merge that is safe to run any number of times.

PHASE 2 adds `--action podcasts`: harvest all 14 monitored feeds, extract the
guests and merge them into the catalog. It is kept out of `auto` for the same
reason `links` is — it is the network step — but `auto` DOES re-apply whatever
the last harvest stored, so rebuilding the catalog never drops guest data.

THE MERGE IS THE POINT. `auto` does not overwrite `entities.json` with the seed;
it merges the seed INTO whatever is already stored, through `model.merge()`.
That is what makes the file a living record: link-verification dates, lifecycle
transitions and any Phase 2 podcast data accumulated on an entity all survive a
re-run, while corrections made in `seed.py` still flow through. Re-running is
idempotent — the second run of the same seed against the same store changes
nothing but the status timestamp.
"""

from __future__ import annotations

import argparse
from datetime import date

from . import DISCLAIMER, load, save
from . import dedupe as idx_dedupe
from . import guests as idx_guests
from . import links as idx_links
from . import model as m
from . import podcasts as idx_podcasts
from . import prior_db as idx_prior
from . import promote as idx_promote
from . import quality as idx_quality
from . import seed as idx_seed
from ..sources import SOURCES


def _match_zenith_sources(entities: list[dict]) -> list[dict]:
    """Link each entity to its matching entry in Zenith's scrape registry.

    This is the integration that stops INDEX and sources.py drifting apart: it
    lets the directory answer "is Zenith already ingesting this?" as a filter,
    and it means a source's real feed status is one hop away from its directory
    row. Matching is on normalised name and alias only — deliberately strict,
    because a wrong match here would attribute one firm's feed health to another.
    """
    by_slug: dict[str, str] = {}
    for s in SOURCES:
        # sources.py names carry qualifiers like "Newfound (Flirting with
        # Models)" and "Meb Faber / Cambria"; index the whole name plus each
        # parenthetical-stripped and slash-separated part.
        raw = s.name
        variants = {raw}
        if "(" in raw:
            variants.add(raw.split("(")[0])
        for part in raw.replace("(", "/").replace(")", "/").split("/"):
            variants.add(part)
        for v in variants:
            slug = m.slug_for(v)
            if slug and len(slug) > 3:
                by_slug.setdefault(slug, s.name)

    out = []
    for ent in entities:
        ent = dict(ent)
        if not ent.get("zenith_source"):
            candidates = [ent.get("slug", "")] + [m.slug_for(a) for a in ent.get("aliases", [])]
            for slug in candidates:
                if slug in by_slug:
                    ent["zenith_source"] = by_slug[slug]
                    break
        out.append(ent)
    return out


def build_catalog(*, fresh: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Build the catalog by merging the seed into whatever is already stored."""
    seed_entities, seed_rels = idx_seed.build()
    stored = [] if fresh else load("entities", [])
    stored_rels = [] if fresh else load("relationships", [])

    by_id = {e["id"]: dict(e) for e in stored if e.get("id")}
    for ent in seed_entities:
        eid = ent["id"]
        by_id[eid] = m.merge(by_id[eid], ent) if eid in by_id else ent

    entities = list(by_id.values())

    seen = {(r.get("source"), r.get("target"), r.get("type")) for r in stored_rels}
    rels = list(stored_rels)
    for r in seed_rels:
        key = (r.get("source"), r.get("target"), r.get("type"))
        if key not in seen:
            seen.add(key)
            rels.append(r)

    entities, rels, report = idx_dedupe.deduplicate(entities, rels)
    entities = _match_zenith_sources(entities)
    entities.sort(key=lambda e: (e.get("primary_category", ""), e.get("name", "").lower()))
    return entities, rels, report


def run_podcasts(only: str | None = None):
    """Harvest the monitored feeds and extract guests.

    Returns (episodes, registry, profiles, new_episodes, records). The stored
    episode archive is MERGED rather than replaced, so an episode a publisher
    later deletes or retitles does not vanish from the record.
    """
    stored = load("episodes", [])
    fresh, registry = idx_podcasts.harvest(only=only)
    episodes, new = idx_podcasts.merge_episodes(stored, fresh)
    records = idx_guests.extract_all(episodes, idx_podcasts.BY_NAME)
    profiles = idx_guests.aggregate(records)
    return episodes, registry, profiles, new, records


def _apply_guest_layers(entities, rels, profiles, report):
    """Apply the two guest sources IN ORDER, then re-deduplicate.

    Order is the mechanism, not a special case: the prior hand-compilation runs
    FIRST and only fills gaps, then the feed harvest runs on top. That makes the
    feeds the authority wherever the two disagree — the user's own decision —
    without either module needing to know about the other.
    """
    entities, rels, prior_report = idx_prior.build(entities, rels)
    entities, rels, promo = idx_promote.build(entities, rels, profiles)
    entities, rels, extra = idx_dedupe.deduplicate(entities, rels)
    return entities, rels, list(report) + extra, promo, prior_report


def _apply_stored_harvest(entities, rels, report):
    """Re-apply the stored harvest so an offline catalog rebuild never drops
    the guest graph."""
    stored_eps = load("episodes", [])
    if not stored_eps:
        entities, rels, prior_report = idx_prior.build(entities, rels)
        entities, rels, extra = idx_dedupe.deduplicate(entities, rels)
        return entities, rels, list(report) + extra
    records = idx_guests.extract_all(stored_eps, idx_podcasts.BY_NAME)
    profiles = idx_guests.aggregate(records)
    entities, rels, report, _promo, _prior = _apply_guest_layers(
        entities, rels, profiles, report)
    return entities, rels, report


def run(action: str = "auto") -> dict:
    today = date.today().isoformat()

    if action in ("auto", "seed"):
        entities, rels, report = build_catalog(fresh=(action == "seed"))
        link_results = load("links", {})
        if link_results:
            entities = idx_links.apply_to_entities(entities, link_results)
    else:
        entities = load("entities", [])
        rels = load("relationships", [])
        report = []
        link_results = load("links", {})

    if action == "links":
        if not entities:
            raise SystemExit("no catalog yet — run `--action auto` first")
        link_results = idx_links.sweep(entities, link_results)
        entities = idx_links.apply_to_entities(entities, link_results)
        save("links", link_results)

    # --- Phase 2: podcast intelligence -------------------------------------
    podcast_doc = load("podcasts", {})
    if action == "podcasts":
        if not entities:
            raise SystemExit("no catalog yet -- run `--action auto` first")
        episodes, registry, profiles, new_eps, records = run_podcasts()
        entities, rels, report, promo, prior_report = _apply_guest_layers(
            entities, rels, profiles, report)
        podcast_doc = {
            "date": today,
            "shows": idx_promote.podcast_stats(registry, records, profiles),
            "promotion": promo,
            "prior_compilation": prior_report,
            "episodes": len(episodes),
            "records": len(records),
            "guests_total": len(profiles),
            "guests_promoted": sum(1 for p in profiles.values()
                                   if idx_guests.meets_threshold(p["confidence"])),
            "new_episodes": [{"podcast": e["podcast"], "title": e["title"],
                              "published": e["published"], "url": e["url"]}
                             for e in new_eps[:60]],
            "guests": _guest_index(profiles),
        }
        # Thousands of rows: compact, for the same reason mom/__init__.save() is.
        save("episodes", episodes, indent=None)
        save("podcasts", podcast_doc, indent=None)
    elif action in ("auto", "seed"):
        entities, rels, report = _apply_stored_harvest(entities, rels, report)
        # Keep podcasts.json in step with the catalog. Extraction rules change
        # (they were tightened repeatedly while this was built), and an offline
        # rebuild re-derives guests from the stored episodes — so leaving the
        # guest index untouched let the two artefacts drift apart, which is
        # exactly what screen.py caught: names the current rules reject were
        # still sitting in podcasts.json claiming to be promoted.
        stored_eps = load("episodes", [])
        if stored_eps and podcast_doc:
            records = idx_guests.extract_all(stored_eps, idx_podcasts.BY_NAME)
            profiles = idx_guests.aggregate(records)
            podcast_doc = {
                **podcast_doc,
                "records": len(records),
                "guests_total": len(profiles),
                "guests_promoted": sum(1 for p in profiles.values()
                                       if idx_guests.meets_threshold(p["confidence"])),
                "guests": _guest_index(profiles),
                # Feed health needs the network, so the last harvest's per-show
                # rows are carried forward with their episode counts refreshed.
                "shows": idx_promote.podcast_stats(
                    [{**s, "episodes": s.get("episodes", 0)}
                     for s in podcast_doc.get("shows", [])], records, profiles),
            }
            save("podcasts", podcast_doc, indent=None)

    status = idx_quality.report(entities, rels)
    status.update({
        "date": today,
        "action": action,
        "disclaimer": DISCLAIMER,
        "dedupe_report": report,
        "duplicate_candidates": idx_dedupe.duplicate_candidates(entities, rels),
        "review_queue": idx_quality.review_queue(entities),
        "links": idx_links.summarize(link_results) if link_results else
                 {"checked": 0, "by_status": {}, "ok": 0, "blocked": 0, "error": 0},
        "zenith_sources_linked": sum(1 for e in entities if e.get("zenith_source")),
        # The status file carries the podcast SUMMARY only; the guest index and
        # episode archive live in their own artefacts so status.json stays small
        # enough to read by hand in a diff.
        "podcasts": {k: v for k, v in podcast_doc.items()
                     if k not in ("guests", "new_episodes")},
        "new_episodes": podcast_doc.get("new_episodes", [])[:20],
        "seed_files": ["data/index/seed/seed_list_2026-09-01.csv",
                       "data/index/seed/ttu_fwm_prior_compilation_2026-08-08.txt"],
    })

    save("entities", entities)
    save("relationships", rels)
    save("status", status)
    return status


def _guest_index(profiles: dict) -> dict:
    """The per-guest record stored in podcasts.json.

    Appearances are capped here because a handful of guests have 30+ of them and
    the full list is always reconstructable from episodes.json.
    """
    return {
        key: {**{k: v for k, v in prof.items() if k != "appearances"},
              "appearances": prof["appearances"][:8]}
        for key, prof in profiles.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zenith INDEX (Master List) compute")
    ap.add_argument("--action", default="auto",
                    choices=["auto", "seed", "links", "podcasts", "quality"])
    args = ap.parse_args()
    status = run(args.action)
    print(f"INDEX {args.action}: {status['total']} entities, "
          f"{status['relationships']} relationships, "
          f"{status['verified']} verified, {status['needs_review']} need review, "
          f"{status['zenith_sources_linked']} linked to Zenith sources")
    ls = status["links"]
    if ls["checked"]:
        print(f"  links: {ls['ok']} ok, {ls['blocked']} blocked, {ls['error']} error "
              f"(of {ls['checked']} checked)")
    pod = status.get("podcasts") or {}
    if pod.get("shows"):
        promo = pod.get("promotion", {})
        print(f"  podcasts: {len(pod['shows'])} shows · {pod.get('episodes', 0)} episodes · "
              f"{pod.get('guests_promoted', 0)}/{pod.get('guests_total', 0)} guests promoted "
              f"({promo.get('created_people', 0)} new, "
              f"{promo.get('matched_existing', 0)} matched, "
              f"{promo.get('created_firms', 0)} firms, "
              f"{promo.get('appeared_on_edges', 0)} appearance edges)")
    if status["orphan_edges"]:
        print(f"  WARNING: {status['orphan_edges']} orphan edges")


if __name__ == "__main__":
    main()

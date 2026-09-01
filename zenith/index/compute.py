"""INDEX compute orchestrator.

    python -m zenith.index.compute --action auto     # rebuild catalog, no network
    python -m zenith.index.compute --action seed     # same, ignoring stored state
    python -m zenith.index.compute --action links    # live URL sweep (network)
    python -m zenith.index.compute --action quality  # recompute status only

WHY `auto` DOES NOT TOUCH THE NETWORK. Every other Zenith package's nightly
action fetches data, but a directory has nothing to fetch: the catalog is
curated, not scraped. The only network step is the link sweep, which is
rate-limited, TTL-gated and run deliberately — so `auto` is a fast, offline
rebuild-and-merge that is safe to run any number of times.

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
from . import links as idx_links
from . import model as m
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
        "seed_files": ["data/index/seed/seed_list_2026-09-01.csv",
                       "data/index/seed/ttu_fwm_prior_compilation_2026-08-08.txt"],
    })

    save("entities", entities)
    save("relationships", rels)
    save("status", status)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Zenith INDEX (Master List) compute")
    ap.add_argument("--action", default="auto",
                    choices=["auto", "seed", "links", "quality"])
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
    if status["orphan_edges"]:
        print(f"  WARNING: {status['orphan_edges']} orphan edges")


if __name__ == "__main__":
    main()

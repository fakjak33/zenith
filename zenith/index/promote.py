"""INDEX Phase 2 — turning harvested guests into directory entities and edges.

This is the join between the podcast archive and the catalog. It is kept in its
own module rather than inside compute.py because the rules here are the ones
most likely to need adjusting as the archive grows, and they deserve to be
readable and testable on their own.

FOUR RULES, IN ORDER OF HOW MUCH THEY MATTER:

  1. ONLY CONFIDENT PARSES BECOME PEOPLE. A guest is promoted to a Person entity
     only at or above ``INDEX_GUEST_MIN_CONFIDENCE``. Everything else stays
     recorded against its episode, so nothing is lost and the yield stays
     auditable, but the directory is not padded with structural guesses.

  2. CURATED DATA WINS OVER HARVESTED DATA. Phase 1 entries were written by
     hand with verified URLs; the harvest must enrich them, never overwrite
     them. Since ``model.merge()`` already refuses to replace a non-empty value
     with an empty one, promotion passes only the fields the harvest genuinely
     knows and leaves the rest absent.

  3. NO DANGLING EDGES. A ``works_at`` edge is created only when the firm named
     in the metadata resolves to an entity that actually exists. A firm seen
     once is still recorded as the person's affiliation TEXT — losing nothing —
     but does not manufacture an organisation stub with no URL. A firm that
     recurs (``INDEX_FIRM_ENTITY_MIN_MENTIONS``) is worth a real entry and gets
     one, flagged for review because nothing about it has been verified.

  4. AN APPEARANCE IS EVIDENCE, SO IT CARRIES ITS SOURCE. Every appearance
     records the podcast, episode title, date and URL it came from, so any
     claim in a guest's profile can be traced back to the episode that produced
     it.
"""

from __future__ import annotations

from datetime import date

from . import guests as idx_guests
from . import model as m
from . import podcasts as idx_podcasts
from ..config import (INDEX_FIRM_ENTITY_MIN_MENTIONS, INDEX_GUEST_MIN_CONFIDENCE,
                      INDEX_INLINE_APPEARANCES)

PROVENANCE = "podcast archive harvest"

# Tags every harvested guest legitimately carries: they were, definitionally,
# interviewed on a finance podcast. Nothing else is assumed about them.
_GUEST_INSIGHTS = ["interviews", "podcasts"]


def _resolver(entities: list[dict]) -> dict[str, str]:
    """Map every name/alias/slug an existing entity answers to -> its id."""
    index: dict[str, str] = {}
    for ent in entities:
        for label in [ent.get("name", "")] + list(ent.get("aliases", [])):
            slug = m.slug_for(label)
            if slug:
                index.setdefault(slug, ent["id"])
        if ent.get("slug"):
            index.setdefault(ent["slug"], ent["id"])
    return index


def _firm_mention_counts(profiles: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prof in profiles.values():
        for firm in prof.get("firms", []):
            key = m.slug_for(firm)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def build(entities: list[dict], relationships: list[dict],
          profiles: dict[str, dict], *,
          min_confidence: str = INDEX_GUEST_MIN_CONFIDENCE,
          today: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """Merge guest profiles into the catalog.

    Returns ``(entities, relationships, report)``. Both inputs are treated as
    immutable; new lists are returned.
    """
    today = today or date.today().isoformat()
    entities = [dict(e) for e in entities]
    relationships = list(relationships)
    by_id = {e["id"]: e for e in entities}
    resolve = _resolver(entities)
    firm_counts = _firm_mention_counts(profiles)

    existing_edges = {(r.get("source"), r.get("target"), r.get("type"))
                      for r in relationships}
    report = {"promoted": 0, "matched_existing": 0, "created_people": 0,
              "created_firms": 0, "works_at_edges": 0, "appeared_on_edges": 0,
              "skipped_low_confidence": 0}

    def _add_edge(src: str, dst: str, rel: str, note: str = "") -> bool:
        key = (src, dst, rel)
        if src == dst or key in existing_edges or src not in by_id or dst not in by_id:
            return False
        existing_edges.add(key)
        relationships.append(m.edge(src, dst, rel, note))
        return True

    # --- firms that recur often enough to deserve their own entry ----------
    for prof in profiles.values():
        if not idx_guests.meets_threshold(prof["confidence"], min_confidence):
            continue
        for firm in prof.get("firms", []):
            slug = m.slug_for(firm)
            if not slug or slug in resolve:
                continue
            if firm_counts.get(slug, 0) < INDEX_FIRM_ENTITY_MIN_MENTIONS:
                continue
            ent = m.make(
                firm,
                entity_type="organisation",
                primary_category="institutional",
                description=f"Organisation named in podcast episode metadata "
                            f"({firm_counts[slug]} mentions across the monitored "
                            f"archive). Not yet independently verified.",
                insight_types=["interviews"],
                provenance=PROVENANCE,
                confidence="low",
                lifecycle_state="needs_review",
                date_added=today,
                notes="Discovered through podcast harvesting, not from the curated "
                      "seed list. No official URL has been confirmed for it yet.",
            )
            if ent["id"] in by_id:
                continue
            entities.append(ent)
            by_id[ent["id"]] = ent
            resolve[ent["slug"]] = ent["id"]
            report["created_firms"] += 1

    # --- guests -----------------------------------------------------------
    for prof in sorted(profiles.values(), key=lambda p: -p["n_appearances"]):
        if not idx_guests.meets_threshold(prof["confidence"], min_confidence):
            report["skipped_low_confidence"] += 1
            continue
        report["promoted"] += 1

        name = prof["name"]
        appearances = [{
            "podcast": a["podcast"], "episode": a["title"],
            "published": a["published"], "url": a["url"],
        } for a in prof["appearances"][:INDEX_INLINE_APPEARANCES]]

        incoming = {
            "entity_type": "person",
            "primary_category": "institutional",
            "insight_types": _GUEST_INSIGHTS,
            "podcast_appearances": appearances,
            "provenance": PROVENANCE,
            "date_added": today,
        }
        # Only assert what the metadata actually said.
        if prof.get("current_firm"):
            incoming["current_affiliation"] = prof["current_firm"]
        if prof.get("past_firms"):
            incoming["historical_affiliations"] = prof["past_firms"]
        if prof.get("role"):
            incoming["role"] = prof["role"]

        eid = resolve.get(m.slug_for(name))
        if eid and eid in by_id and by_id[eid].get("entity_type") not in ("person", None, ""):
            # The name collides with something that is NOT a person -- almost
            # always a parse error that captured an organisation ("Goldman
            # Sachs") out of a title. Merging would rewrite that organisation's
            # entity_type to "person" and quietly corrupt a curated entry, so
            # the guest is dropped rather than allowed to overwrite it.
            report["skipped_type_conflict"] = report.get("skipped_type_conflict", 0) + 1
            report["promoted"] -= 1
            continue

        if eid and eid in by_id:
            # An existing entry -- curated or previously harvested. merge()
            # protects its verified fields; we only add what is new.
            existing = by_id[eid]
            # Provenance ACCUMULATES rather than being replaced. Overwriting it
            # would erase the fact that a hand-curated entry was hand-curated,
            # which is exactly the audit trail this directory exists to keep.
            prior = str(existing.get("provenance") or "").strip()
            if prior and PROVENANCE not in prior:
                incoming["provenance"] = f"{prior}; {PROVENANCE}"
            elif prior:
                incoming["provenance"] = prior
            merged = m.merge(existing, incoming)
            # A hand-curated description must survive a harvest that has none.
            by_id[eid] = merged
            for i, e in enumerate(entities):
                if e["id"] == eid:
                    entities[i] = merged
                    break
            report["matched_existing"] += 1
        else:
            ent = m.make(
                name,
                description=_describe(prof),
                confidence=prof["confidence"],
                lifecycle_state="needs_review",
                notes=_provenance_note(prof),
                **incoming,
            )
            if ent["id"] in by_id:
                continue
            entities.append(ent)
            by_id[ent["id"]] = ent
            resolve[ent["slug"]] = ent["id"]
            eid = ent["id"]
            report["created_people"] += 1

        # --- edges ---------------------------------------------------------
        for show in prof.get("podcasts", []):
            sid = resolve.get(m.slug_for(show))
            if sid and _add_edge(eid, sid, "appeared_on",
                                 f"{prof['n_appearances']} appearance(s)"):
                report["appeared_on_edges"] += 1
        if prof.get("current_firm"):
            fid = resolve.get(m.slug_for(prof["current_firm"]))
            if fid and _add_edge(eid, fid, "works_at", "stated in episode metadata"):
                report["works_at_edges"] += 1
        for past in prof.get("past_firms", []):
            fid = resolve.get(m.slug_for(past))
            if fid:
                _add_edge(eid, fid, "worked_at", "stated in earlier episode metadata")

    return entities, relationships, report


def _describe(prof: dict) -> str:
    """A description built only from what the archive actually shows."""
    shows = prof.get("podcasts", [])
    n = prof["n_appearances"]
    bits = [f"Podcast guest: {n} appearance{'s' if n != 1 else ''} across "
            f"{len(shows)} monitored show{'s' if len(shows) != 1 else ''} "
            f"({', '.join(shows[:4])}{'…' if len(shows) > 4 else ''})."]
    if prof.get("role") and prof.get("current_firm"):
        bits.append(f"Described in episode metadata as {prof['role']} at "
                    f"{prof['current_firm']}.")
    elif prof.get("current_firm"):
        bits.append(f"Associated with {prof['current_firm']} in episode metadata.")
    latest = prof["appearances"][0] if prof.get("appearances") else None
    if latest and latest.get("published"):
        bits.append(f"Most recent appearance {latest['published']} on "
                    f"{latest['podcast']}.")
    return " ".join(bits)


def _provenance_note(prof: dict) -> str:
    strategies = ", ".join(prof.get("strategies", [])) or "unknown"
    return ("Discovered by parsing podcast episode metadata (pattern: "
            f"{strategies}; confidence: {prof['confidence']}). Name, role and "
            "affiliation are as stated by the publisher in the episode title or "
            "show notes -- no personal website or biography has been verified, "
            "which is why this entry is flagged for review.")


def episode_index(episodes: list[dict], records: list[dict]) -> dict[str, list[str]]:
    """episode id -> guest names, for the episode-level views."""
    out: dict[str, list[str]] = {}
    for rec in records:
        out.setdefault(rec["episode_id"], []).append(rec["name"])
    return out


def podcast_stats(registry: list[dict], records: list[dict],
                  profiles: dict[str, dict]) -> list[dict]:
    """Per-show rollup for the Podcasts view."""
    by_show: dict[str, dict] = {}
    for row in registry:
        by_show[row["podcast"]] = {
            **row, "guests": set(), "episodes_with_guest": set(),
        }
    for rec in records:
        slot = by_show.get(rec["podcast"])
        if slot is None:
            continue
        slot["guests"].add(rec["name"])
        slot["episodes_with_guest"].add(rec["episode_id"])

    # Guests who also appear on at least one OTHER monitored show -- the
    # cross-pollination that makes the graph more than a set of lists.
    shared = {name: prof["podcasts"] for name, prof in
              ((k, v) for k, v in profiles.items()) if prof["n_podcasts"] > 1}

    out = []
    for name, slot in by_show.items():
        n_eps = slot.get("episodes", 0) or 0
        guests = sorted(slot["guests"])
        cross = sum(1 for g in guests if g.lower() in shared)
        out.append({
            "podcast": name, "feed_url": slot.get("feed_url", ""),
            "ok": slot.get("ok", False), "error": slot.get("error", ""),
            "hosts": slot.get("hosts", []), "note": slot.get("note", ""),
            "episodes": n_eps,
            "earliest": slot.get("earliest", ""), "latest": slot.get("latest", ""),
            "episodes_with_guest": len(slot["episodes_with_guest"]),
            "coverage": round(len(slot["episodes_with_guest"]) / n_eps, 3) if n_eps else 0.0,
            "unique_guests": len(guests),
            "cross_show_guests": cross,
        })
    return sorted(out, key=lambda r: -r["episodes"])

"""INDEX Phase 3 — transparent prominence ranking.

WHAT THIS SCORE IS, AND WHAT IT IS EMPHATICALLY NOT.

It is a measure of how CENTRAL an entry is *within the data this directory has
collected*. It is not a judgement of quality, insight, rigour or importance, and
it must never be presented as one. AQR scoring higher than a brilliant one-person
research shop means AQR appears more often in these particular podcast archives
and has more stated connections in this particular catalog — nothing more.

The spec that asked for ranking also said: "Do not invent fake precision or
pretend that subjective judgments are objective facts. If ranking is used, make
the methodology transparent." Three rules follow from that, and they are the
whole design:

  1. EVERY INPUT IS A COUNT WE ACTUALLY MEASURED. Podcast appearances, distinct
     shows, graph degree, profile completeness, whether a link resolved, whether
     Zenith already ingests the source. No editorial weighting of "importance",
     no sentiment, no proxy for quality.

  2. THE FORMULA IS PUBLISHED AND THE COMPONENTS ARE RETURNED. ``score()``
     returns the contribution of each component alongside the total, so any
     ranking in the UI can be expanded into "why is this here" without the
     reader taking anything on trust. ``METHODOLOGY`` is rendered verbatim in
     the app.

  3. COUNTS ARE COMPRESSED, NOT LEFT RAW. A guest with 39 appearances is not
     thirty-nine times more central than one with a single appearance, so counts
     go through log1p and are then normalised against the observed maximum. This
     keeps one prolific recurring guest from flattening the entire scale — and
     it is stated here rather than hidden in a magic constant.

The weights below are a stated editorial choice about what "central" should
mean, not a fitted or optimised quantity. They are round numbers on purpose:
pretending to three decimal places would be exactly the false precision this
module exists to avoid.
"""

from __future__ import annotations

import math

# component -> (weight, human label, what it measures)
COMPONENTS: dict[str, tuple[float, str, str]] = {
    "connections": (0.35, "Graph connections",
                    "Distinct stated relationships to other entries — employment, "
                    "founding, parent/brand, hosting, podcast appearances."),
    "appearances": (0.25, "Podcast appearances",
                    "Episodes across the monitored archives where this person was "
                    "identified as a guest."),
    "reach": (0.20, "Cross-show reach",
              "How many DIFFERENT monitored podcasts an entry appears on. Weighted "
              "separately from raw appearances because breadth across shows is a "
              "different signal from repetition on one."),
    "completeness": (0.10, "Profile completeness",
                     "Share of the applicable catalog fields that are filled in."),
    "verification": (0.10, "Verification",
                     "Whether the official link was confirmed live and the entry "
                     "reviewed. Rewards entries we can actually stand behind."),
}

METHODOLOGY = """
**What this ranks.** How CENTRAL an entry is inside the data this directory has
collected — not how good, insightful or important it is. A firm that scores
highly appears more often in these fourteen podcast archives and carries more
stated connections in this catalog. That is all it means.

**Every input is a count that was measured**, never an editorial judgement:
graph connections, podcast appearances, how many different shows someone
appeared on, profile completeness, and whether the link was confirmed live.

**Counts are compressed with log1p and normalised against the observed maximum,**
so one guest with 39 appearances does not flatten the scale for everyone else.

**The weights are a stated choice, not a fitted parameter.** They are round
numbers deliberately — tuning them to three decimals would imply a precision
that does not exist. Every ranked row can be expanded to show exactly what each
component contributed.

**Known biases, stated rather than hidden.** Shows with structured episode titles
(Alpha Exchange, The Long View) yield far more guests than shows with editorial
ones (Odd Lots), so their guests are over-represented. People are better covered
than institutions, because podcasts name people. An entry added from the curated
seed list with no podcast presence will always score lower than a frequent guest,
which says nothing whatsoever about its worth as a research source.
"""


def _norm(values: dict[str, float]) -> dict[str, float]:
    """log1p-compress, then scale to 0..1 against the observed maximum."""
    if not values:
        return {}
    logged = {k: math.log1p(max(0.0, float(v))) for k, v in values.items()}
    top = max(logged.values(), default=0.0)
    if top <= 0:
        return {k: 0.0 for k in logged}
    return {k: v / top for k, v in logged.items()}


def score_all(entities: list[dict], relationships: list[dict],
              guest_index: dict | None = None) -> dict[str, dict]:
    """Score every entity. Returns {entity_id: {score, components, rank}}.

    ``components`` holds each part's CONTRIBUTION (weight x normalised value),
    so the numbers in a breakdown add up to the total exactly.
    """
    guest_index = guest_index or {}
    ids = [e["id"] for e in entities]

    deg: dict[str, float] = {i: 0.0 for i in ids}
    known = set(ids)
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        if s in known and t in known and s != t:
            deg[s] += 1
            deg[t] += 1

    appearances: dict[str, float] = {}
    reach: dict[str, float] = {}
    completeness: dict[str, float] = {}
    verification: dict[str, float] = {}
    for e in entities:
        eid = e["id"]
        # Look the guest profile up by NAME OR ALIAS. The catalog and the
        # harvest often disagree on the form of a name -- the curated entry is
        # "Robert Carver", the podcast titles say "Rob Carver" -- and matching
        # on the primary name alone quietly under-counted exactly the
        # best-connected people, who are the ones most likely to have both.
        prof = {}
        for label in [e.get("name", "")] + list(e.get("aliases", [])):
            prof = guest_index.get(str(label).lower()) or {}
            if prof:
                break
        appearances[eid] = float(max(int(prof.get("n_appearances", 0) or 0),
                                     len(e.get("podcast_appearances") or [])))
        reach[eid] = float(prof.get("n_podcasts", 0))
        completeness[eid] = _completeness(e)
        verification[eid] = _verification(e)

    normed = {
        "connections": _norm(deg),
        "appearances": _norm(appearances),
        "reach": _norm(reach),
        # These two are already 0..1 ratios; compressing them would distort a
        # scale that is meaningful as-is.
        "completeness": completeness,
        "verification": verification,
    }

    out: dict[str, dict] = {}
    for eid in ids:
        parts = {name: round(weight * normed[name].get(eid, 0.0), 4)
                 for name, (weight, _label, _desc) in COMPONENTS.items()}
        out[eid] = {"score": round(sum(parts.values()), 4), "components": parts,
                    "raw": {"connections": int(deg.get(eid, 0)),
                            "appearances": int(appearances.get(eid, 0)),
                            "reach": int(reach.get(eid, 0)),
                            "completeness": f"{completeness.get(eid, 0.0):.0%}",
                            "verification": f"{verification.get(eid, 0.0):.0%}"}}

    for rank, eid in enumerate(sorted(out, key=lambda k: -out[k]["score"]), start=1):
        out[eid]["rank"] = rank
    return out


def _completeness(ent: dict) -> float:
    """Reuse the quality module's definition so one notion of "complete" exists."""
    from . import quality as idx_quality
    return idx_quality.completeness(ent)


def _verification(ent: dict) -> float:
    """1.0 verified with a live link, 0.5 alive but robot-blocked, else 0.

    A blocked host scores half rather than zero: the resource demonstrably
    exists and opens in a browser, which is materially different from a link
    that does not resolve. Treating the two the same would penalise Citadel and
    SSRN for having anti-bot rules.
    """
    state = ent.get("lifecycle_state")
    link = ent.get("link_status")
    if link == "ok" and state in ("verified", "updated"):
        return 1.0
    if link == "ok":
        return 0.8
    if link == "blocked":
        return 0.5
    return 0.0


def explain(entity: dict, scored: dict) -> list[dict]:
    """Per-component breakdown rows for one entity, largest contribution first."""
    row = scored.get(entity["id"]) or {}
    parts = row.get("components", {})
    raw = row.get("raw", {})
    out = []
    for name, (weight, label, desc) in COMPONENTS.items():
        out.append({
            "component": label,
            "weight": weight,
            # Always a string: this column mixes counts (3) with ratios ("100%"),
            # and a mixed-type column forces Arrow to coerce on every render.
            "measured": str(raw.get(name, "")),
            "contribution": parts.get(name, 0.0),
            "what it measures": desc,
        })
    out.sort(key=lambda r: -r["contribution"])
    return out


def top(entities: list[dict], scored: dict, limit: int = 50,
        entity_types: tuple[str, ...] | None = None) -> list[dict]:
    pool = [e for e in entities
            if not entity_types or e.get("entity_type") in entity_types]
    pool.sort(key=lambda e: -(scored.get(e["id"], {}).get("score", 0.0)))
    return pool[:limit]

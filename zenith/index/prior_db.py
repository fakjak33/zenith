"""INDEX Phase 2 — the prior hand-compiled TTU / Flirting with Models database.

The user supplied an earlier best-effort guest compilation
(``data/index/seed/ttu_fwm_prior_compilation_2026-08-08.txt``) alongside the
resource list. It is imported here with EXPLICIT PROVENANCE and, per the user's
own decision, is subordinate to the feed harvest wherever the two disagree.

WHY IT IS STILL WORTH IMPORTING. The RSS harvest is far broader — thousands of
guests versus a few hundred — but it is thin on AFFILIATION: only a minority of
harvested guests have a firm stated in the episode metadata at all. This file
carries a firm and often a role for essentially every person in it, in a
consistent shape:

    Jerry Parker - Chesapeake Capital (founder). Round Table #07-08; TTU145.
    Harold de Boer - Transtrend (Head of R&D & Managing Director). TTU103/104.
    Eric Crittenden - Standpoint Asset Management (CIO). TTU143; SI404. [BOTH PODCASTS]

So it is used to FILL GAPS, not to overwrite. compute.py applies this import
BEFORE the harvest promotion, which means any affiliation the feeds state wins
on top of it automatically — the ordering is the mechanism, rather than a
special case in the merge.

HONESTY. The file's own header is unusually candid about its limits: it reports
~232 people, admits the guest index was truncated by a fetch limit, and marks
several entries "STATUS: current firm unverified". Those caveats are carried
through into the entity notes and confidence rather than being dropped, and the
whole import is tagged `prior manual compilation` so it can always be told
apart from both the curated seed list and the feed harvest.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from . import guests as idx_guests
from . import model as m
from ..config import INDEX_SEED_DIR

PROVENANCE = "prior manual compilation (ttu_fwm, 2026-08-08)"
DEFAULT_FILE = INDEX_SEED_DIR / "ttu_fwm_prior_compilation_2026-08-08.txt"

# Section headings map to the strategy tags the compiler filed people under.
# Taken from the file's own PART A structure, not invented.
_SECTION_TAGS: dict[str, tuple[str, ...]] = {
    "A1": ("trend_following", "managed_futures", "systematic"),
    "A2": ("volatility", "options", "derivatives"),
    "A3": ("global_macro", "macro"),
    "A4": ("asset_allocation", "institutional_investing", "manager_selection"),
    "A5": ("quantitative", "factor_investing", "multi_strategy", "systematic"),
    "A6": (),
    "A7": ("crypto", "market_microstructure"),
}
_SECTION_INSIGHTS: dict[str, tuple[str, ...]] = {
    "A6": ("news", "books", "financial_education"),
}

_SECTION_RE = re.compile(r"^(A[1-7])\.\s")
_PART_RE = re.compile(r"^PART\s+([A-D])\b")
# "Name - Firm (role). Episode refs."  — the en-dash and hyphen both occur.
_ENTRY_RE = re.compile(r"^(?P<name>[^-–—]{3,45}?)\s+[-–—]\s+(?P<rest>.+)$")
_STATUS_RE = re.compile(r"STATUS:\s*(?P<status>[^.\[]+)", re.I)
_BOTH_RE = re.compile(r"\[BOTH PODCASTS[^\]]*\]", re.I)
_PAREN_RE = re.compile(r"\(([^)]*)\)")


def _split_firm_role(rest: str) -> tuple[str, str]:
    """'Chesapeake Capital (founder). Round Table #07-08.' -> firm, role."""
    head = rest.split(".")[0].strip()
    role = ""
    paren = _PAREN_RE.search(head)
    if paren:
        role = paren.group(1).strip()
        head = _PAREN_RE.sub("", head).strip()
    # The compilation's own format puts the firm in this exact slot, so the
    # generic shape heuristics would only second-guess a fact the format
    # already establishes -- which is how "Transtrend", "Winton" and "Amigo"
    # were being dropped on a first pass for the crime of being one word long.
    firm = idx_guests.clean_firm(head, positional=True)
    return firm, role[:80]


def parse(path: Path | None = None) -> list[dict]:
    """Parse the compilation into person records.

    Only PART A (people, by strategy) is imported. PART B is a cross-reference
    of people already in PART A, PART C is an organisation index whose entries
    carry no URLs, and PART D is the compiler's own list of gaps — none of them
    add facts that PART A does not already state.
    """
    path = path or DEFAULT_FILE
    if not path.exists():
        return []

    out: list[dict] = []
    part = ""
    section = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        pm = _PART_RE.match(line)
        if pm:
            part = pm.group(1)
            section = ""
            continue
        sm = _SECTION_RE.match(line)
        if sm:
            section = sm.group(1)
            continue
        if part != "A" or not section or line.startswith(("-", "=", "•")):
            continue

        both = bool(_BOTH_RE.search(line))
        status_m = _STATUS_RE.search(line)
        status = status_m.group("status").strip() if status_m else ""
        body = _BOTH_RE.sub("", line)
        body = _STATUS_RE.sub("", body).strip()

        em = _ENTRY_RE.match(body)
        if not em:
            continue
        name = em.group("name").strip()
        # "Martin (Marty) Lueck" / "Robert (Roberto) Osorio" — the parenthetical
        # is a nickname, not part of the canonical name.
        name = _PAREN_RE.sub(" ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if not idx_guests.looks_like_person(name):
            continue
        firm, role = _split_firm_role(em.group("rest"))
        out.append({
            "name": idx_guests.canonical_name(name),
            "firm": firm,
            "role": role,
            "section": section,
            "both_podcasts": both,
            "status": status,
            "source_line": line[:300],
        })
    return out


def build(entities: list[dict], relationships: list[dict],
          records: list[dict] | None = None,
          today: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """Merge the prior compilation into the catalog, filling gaps only."""
    today = today or date.today().isoformat()
    records = parse() if records is None else records
    entities = [dict(e) for e in entities]
    relationships = list(relationships)
    by_id = {e["id"]: e for e in entities}
    index: dict[str, str] = {}
    for ent in entities:
        for label in [ent.get("name", "")] + list(ent.get("aliases", [])):
            slug = m.slug_for(label)
            if slug:
                index.setdefault(slug, ent["id"])

    edges = {(r.get("source"), r.get("target"), r.get("type")) for r in relationships}
    report = {"parsed": len(records), "created": 0, "enriched": 0, "works_at_edges": 0}

    for rec in records:
        tags = list(_SECTION_TAGS.get(rec["section"], ()))
        insights = list(_SECTION_INSIGHTS.get(rec["section"], ("interviews", "podcasts")))
        note = (f"From a prior hand-compiled TTU / Flirting with Models guest database "
                f"(2026-08-08). Source line: \"{rec['source_line']}\"")
        if rec["status"]:
            note += f" The compiler flagged: {rec['status']}."
        if rec["both_podcasts"]:
            note += " Confirmed by that compilation on BOTH podcasts."

        incoming = {
            "entity_type": "person",
            "primary_category": "institutional",
            "investment_approach": tags,
            "insight_types": insights,
            "date_added": today,
        }
        if rec["firm"]:
            incoming["current_affiliation"] = rec["firm"]
        if rec["role"]:
            incoming["role"] = rec["role"]

        eid = index.get(m.slug_for(rec["name"]))
        if eid and eid in by_id and by_id[eid].get("entity_type") not in ("person", None, ""):
            # Same guard as promote.build(): never let a person record rewrite
            # the entity_type of an organisation it merely shares a name with.
            report["skipped_type_conflict"] = report.get("skipped_type_conflict", 0) + 1
            continue

        if eid and eid in by_id:
            existing = by_id[eid]
            prior = str(existing.get("provenance") or "").strip()
            incoming["provenance"] = (f"{prior}; {PROVENANCE}" if prior and PROVENANCE not in prior
                                      else prior or PROVENANCE)
            # Gap-filling only: an affiliation already recorded is left alone,
            # so a later feed harvest remains the authority.
            if existing.get("current_affiliation"):
                incoming.pop("current_affiliation", None)
            if existing.get("role"):
                incoming.pop("role", None)
            merged = m.merge(existing, incoming)
            if not str(merged.get("notes") or "").strip():
                merged["notes"] = note
            by_id[eid] = merged
            for i, e in enumerate(entities):
                if e["id"] == eid:
                    entities[i] = merged
                    break
            report["enriched"] += 1
        else:
            ent = m.make(
                rec["name"],
                description=_describe(rec),
                provenance=PROVENANCE,
                confidence="low",
                lifecycle_state="needs_review",
                notes=note,
                **incoming,
            )
            if ent["id"] in by_id:
                continue
            entities.append(ent)
            by_id[ent["id"]] = ent
            index[ent["slug"]] = ent["id"]
            eid = ent["id"]
            report["created"] += 1

        if rec["firm"]:
            fid = index.get(m.slug_for(rec["firm"]))
            key = (eid, fid, "works_at")
            if fid and fid in by_id and fid != eid and key not in edges:
                edges.add(key)
                relationships.append(
                    m.edge(eid, fid, "works_at", "per the prior compilation"))
                report["works_at_edges"] += 1
    return entities, relationships, report


def _describe(rec: dict) -> str:
    bits = []
    if rec["role"] and rec["firm"]:
        bits.append(f"{rec['role'].capitalize()} at {rec['firm']}.")
    elif rec["firm"]:
        bits.append(f"Associated with {rec['firm']}.")
    bits.append("Recorded as a Top Traders Unplugged / Flirting with Models guest "
                "in a prior hand-compiled database.")
    if rec["status"]:
        bits.append(f"Caveat carried over from that compilation: {rec['status']}.")
    return " ".join(bits)

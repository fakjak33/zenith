"""INDEX entity model — schema, normalisation, and the history-preserving merge.

FIELD DISCIPLINE. The spec explicitly warned against adding fields "merely for
the sake of complexity", so every field below earns its place by answering a
question a researcher actually asks: what is this, who runs it, what do they do,
where do I read it, how sure are we, and when did we last check. Fields that
would only ever hold a guess (an "influence score", a fabricated founding date)
are absent by design.

THE MERGE RULE is the important part of this module. When an entity is seen
again with new information, ``merge()`` never silently discards what was there:

  * a person's changed employer moves the old one into
    ``historical_affiliations`` and becomes ``current_affiliation``;
  * list fields union rather than replace;
  * a non-empty stored value is never overwritten by an empty incoming one;
  * ``date_last_verified`` only ever moves forward.

This is what makes the directory a record rather than a snapshot — the spec's
"avoid silently overwriting useful historical information" requirement, and the
reason a person who moves from one firm to another remains traceable to both.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from . import taxonomy as tx

# Every entity field, with its default. Ordering here is the ORDER USED IN THE
# EXPORT, so it is arranged the way someone reading a spreadsheet would want it:
# identity first, then classification, then links, then people, then tags, then
# provenance/housekeeping last.
FIELDS: dict[str, object] = {
    # --- identity ---
    "id": "",
    "name": "",
    "slug": "",
    "aliases": [],
    # --- classification ---
    "entity_type": "organisation",
    "org_subtype": "",
    "primary_category": "institutional",
    "secondary_categories": [],
    # --- description & links ---
    "description": "",
    "url": "",
    "research_url": "",
    "social_url": "",
    # --- organisation facts ---
    "location": "",
    "founded": "",
    "status": "active",
    "key_people": [],
    # --- person facts (empty for organisations) ---
    "current_affiliation": "",
    "historical_affiliations": [],
    "role": "",
    # --- descriptive tags ---
    "investment_approach": [],
    "asset_classes": [],
    "insight_types": [],
    # --- Zenith integration ---
    "zenith_source": "",          # name of the matching zenith/sources.py Source
    # --- Phase 2/3 reserved: declared so the export shape never changes ---
    "podcast_appearances": [],
    "books": [],
    "papers": [],
    # --- provenance & housekeeping ---
    "provenance": "",
    "confidence": "medium",
    "lifecycle_state": "new",
    "link_status": "unchecked",
    "date_added": "",
    "date_last_verified": "",
    "notes": "",
}

LIST_FIELDS = tuple(k for k, v in FIELDS.items() if isinstance(v, list))

# Fields whose presence counts toward the completeness score (quality.py).
# Deliberately excludes fields that legitimately do not apply to many entries
# (founded, location, role) — see INDEX_COMPLETENESS_TARGET's rationale.
COMPLETENESS_FIELDS = (
    "name", "description", "url", "entity_type", "primary_category",
    "insight_types", "provenance",
)

# Vocabulary each tag field is validated against.
_TAG_VOCAB = {
    "investment_approach": "investment_approach",
    "asset_classes": "asset_class",
    "insight_types": "insight_type",
}

# Suffixes stripped when generating a matching slug, so "Gresham Investment
# Management LLC" and "Gresham Investment Management" collide (they should) —
# but kept in `name`, because the full legal name is the correct display form.
#
# Split into two classes because they behave differently. A LEGAL form carries
# no naming information at all ("BlackRock, Inc." IS "BlackRock"), so it is
# always safe to drop. A DESCRIPTIVE word is often load-bearing — dropping
# "Group" from "Man Group" leaves "man", and "Capital Group" becomes "capital",
# words generic enough to collide with unrelated entries — so those are only
# dropped while at least two words remain.
_LEGAL_SUFFIXES = (
    "llc", "l l c", "lp", "l p", "llp", "ltd", "limited", "inc", "incorporated",
    "plc", "gmbh", "ag", "sa", "nv", "bv", "corp", "corporation",
)
_DESCRIPTIVE_SUFFIXES = (
    "co", "company", "partners", "group", "holdings", "management",
    "capital management", "asset management", "investment management",
)
_SUFFIXES = _LEGAL_SUFFIXES + _DESCRIPTIVE_SUFFIXES

_AMP = re.compile(r"\s*&\s*")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Display-normalised name: collapse whitespace, tidy '&' spacing, and undo
    the SHOUTING the seed list arrives in when a better form is not supplied.

    Deliberately conservative — it does NOT title-case a name that already has
    mixed case, because doing so would mangle real forms like "iShares",
    "AQR", "testfol.io" and "eVestment".
    """
    s = _WS.sub(" ", str(name or "").strip())
    s = _AMP.sub(" & ", s)
    if s and s == s.upper() and len(s) > 4 and not s.replace(" ", "").isalpha():
        return s
    return s


def slug_for(name: str) -> str:
    """Matching slug: lowercase, punctuation dropped, trailing legal-form
    suffixes removed. This is the primary duplicate key.

    Legal forms are always dropped (down to a 3-character floor, so a name that
    IS its legal form survives); descriptive words are only dropped while at
    least two words remain. See the _SUFFIXES comment for why the two classes
    cannot share one rule.
    """
    s = _PUNCT.sub(" ", str(name or "").lower())
    s = _WS.sub(" ", s).strip()
    changed = True
    while changed and s:
        changed = False
        for suf in sorted(_SUFFIXES, key=len, reverse=True):
            if not s.endswith(" " + suf):
                continue
            candidate = s[: -(len(suf) + 1)].strip()
            if suf in _LEGAL_SUFFIXES:
                if len(candidate) < 3:
                    continue
            elif len(candidate.split()) < 2:
                continue
            s = candidate
            changed = True
    return tx.slugify(s)


def entity_id(slug: str) -> str:
    """Stable short id derived from the slug — mirrors store.item_id()'s
    approach so ids stay comparable in spirit across the codebase."""
    return hashlib.sha1(str(slug or "").encode("utf-8", "ignore")).hexdigest()[:12]


def make(name: str, **kw) -> dict:
    """Build a fully-populated entity record from a partial specification.

    Tags are resolved through the taxonomy on the way in, so the stored record
    is always in canonical slugs and the views never have to normalise.
    """
    ent = {k: (list(v) if isinstance(v, list) else v) for k, v in FIELDS.items()}
    ent["name"] = normalize_name(name)
    ent["slug"] = kw.pop("slug", None) or slug_for(ent["name"])
    ent["id"] = kw.pop("id", None) or entity_id(ent["slug"])
    ent["date_added"] = kw.pop("date_added", None) or date.today().isoformat()

    for k, v in kw.items():
        if k not in FIELDS:
            raise KeyError(f"unknown entity field {k!r} (add it to model.FIELDS first)")
        ent[k] = list(v) if isinstance(FIELDS[k], list) and v is not None else v

    # canonicalise controlled fields
    ent["entity_type"] = tx.resolve("entity_type", ent["entity_type"]) or "organisation"
    ent["primary_category"] = tx.resolve("primary_category", ent["primary_category"]) or "institutional"
    if ent["org_subtype"]:
        ent["org_subtype"] = tx.resolve("org_subtype", ent["org_subtype"]) or ""
    for field, vocab in _TAG_VOCAB.items():
        ent[field] = tx.resolve_many(vocab, ent[field])
    ent["aliases"] = _dedupe_strings(ent["aliases"])
    return ent


def _dedupe_strings(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values or []:
        s = _WS.sub(" ", str(v or "").strip())
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _union(a, b) -> list:
    """Union of two lists preserving order, comparing dicts by their JSON-ish
    identity so repeated podcast appearances do not accumulate duplicates."""
    out: list = []
    seen: set = set()
    for v in list(a or []) + list(b or []):
        key = repr(sorted(v.items())) if isinstance(v, dict) else str(v).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def merge(existing: dict, incoming: dict, *, today: str | None = None) -> dict:
    """Merge ``incoming`` into ``existing`` WITHOUT losing history.

    Rules, in the order they matter:
      1. A person's affiliation change is recorded, not overwritten — the old
         ``current_affiliation`` is appended to ``historical_affiliations``.
      2. List fields union.
      3. A non-empty stored scalar is never replaced by an empty incoming one.
      4. ``date_last_verified`` only moves forward; ``date_added`` never moves.
      5. If any substantive field actually changed, the state becomes
         ``updated`` — unless the entry is archived, which is sticky.
    """
    today = today or date.today().isoformat()
    out = dict(existing)
    changed = False

    # --- 1. affiliation history -------------------------------------------
    new_aff = str(incoming.get("current_affiliation") or "").strip()
    old_aff = str(existing.get("current_affiliation") or "").strip()
    if new_aff and old_aff and new_aff.lower() != old_aff.lower():
        hist = list(existing.get("historical_affiliations") or [])
        if not any(str(h).strip().lower() == old_aff.lower() for h in hist):
            hist.append(old_aff)
        out["historical_affiliations"] = hist
        out["current_affiliation"] = new_aff
        changed = True
    elif new_aff and not old_aff:
        # FIRST affiliation. This branch is not an afterthought: because
        # `current_affiliation` is excluded from the general field loop below
        # (its history handling has to happen here), omitting it meant a person
        # recorded with no employer could never gain one — every later source
        # naming their firm was silently discarded.
        out["current_affiliation"] = new_aff
        changed = True

    for key in FIELDS:
        if key in ("current_affiliation", "historical_affiliations",
                   "date_added", "date_last_verified", "lifecycle_state"):
            continue
        if key not in incoming:
            continue
        new = incoming[key]
        old = out.get(key)
        # --- 2. lists union ------------------------------------------------
        if key in LIST_FIELDS:
            merged = _union(old, new)
            if merged != list(old or []):
                out[key] = merged
                changed = True
            continue
        # --- 3. never overwrite a real value with an empty one --------------
        if new in (None, "") and old not in (None, ""):
            continue
        if new != old:
            out[key] = new
            changed = True

    # historical affiliations may also arrive directly
    if incoming.get("historical_affiliations"):
        merged = _union(out.get("historical_affiliations"),
                        incoming["historical_affiliations"])
        cur = str(out.get("current_affiliation") or "").strip().lower()
        merged = [h for h in merged if str(h).strip().lower() != cur]
        if merged != list(existing.get("historical_affiliations") or []):
            out["historical_affiliations"] = merged
            changed = True

    # --- 4. dates ----------------------------------------------------------
    out["date_added"] = existing.get("date_added") or incoming.get("date_added") or today
    old_ver = str(existing.get("date_last_verified") or "")
    new_ver = str(incoming.get("date_last_verified") or "")
    out["date_last_verified"] = max(old_ver, new_ver) if (old_ver or new_ver) else ""

    # --- 5. lifecycle ------------------------------------------------------
    state = incoming.get("lifecycle_state") or existing.get("lifecycle_state") or "new"
    if existing.get("lifecycle_state") == "archived":
        state = "archived"
    elif changed and state == "verified":
        state = "updated"
    out["lifecycle_state"] = state
    return out


def search_blob(ent: dict) -> str:
    """Everything a free-text search should match, lowercased.

    Includes the human LABELS of tags as well as their slugs, so searching
    "trend following" finds an entity tagged ``trend_following`` — a directory
    where the search box only matches internal slugs is a broken directory.
    """
    parts: list[str] = [
        str(ent.get("name", "")), str(ent.get("description", "")),
        str(ent.get("notes", "")), str(ent.get("location", "")),
        str(ent.get("role", "")), str(ent.get("current_affiliation", "")),
        str(ent.get("url", "")), str(ent.get("zenith_source", "")),
    ]
    parts += [str(a) for a in ent.get("aliases", [])]
    parts += [str(p) for p in ent.get("key_people", [])]
    parts += [str(h) for h in ent.get("historical_affiliations", [])]
    parts += [str(b) for b in ent.get("books", [])]
    for field, vocab in _TAG_VOCAB.items():
        for slug in ent.get(field, []):
            parts.append(str(slug))
            parts.append(tx.label_of(vocab, slug))
    for vocab, key in (("entity_type", "entity_type"), ("org_subtype", "org_subtype"),
                       ("primary_category", "primary_category")):
        v = ent.get(key)
        if v:
            parts.append(str(v))
            parts.append(tx.label_of(vocab, v))
    return " ".join(p for p in parts if p).lower()


def edge(source_id: str, target_id: str, rel_type: str, note: str = "",
         since: str = "") -> dict:
    """One relationship in the knowledge graph."""
    return {"source": source_id, "target": target_id, "type": rel_type,
            "note": note, "since": since}

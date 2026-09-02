"""INDEX data quality — completeness, staleness and the review queue.

A directory of a few hundred entries degrades quietly: links rot, people change
firms, half-finished entries sit unnoticed. This module measures that decay so
the Data Management view can show it rather than everyone assuming the catalog
is fine.

WHAT IS AND IS NOT MEASURED. Completeness counts only fields that apply to
essentially every entry (``model.COMPLETENESS_FIELDS``). Fields like `founded`,
`location` or `role` are excluded on purpose: an academic journal has no
founder and a screener has no investment approach, so scoring them against
those fields would permanently flag legitimate entries as deficient and train
the reader to ignore the flag. This is why ``INDEX_COMPLETENESS_TARGET`` is
0.6 rather than 1.0.

STALENESS is measured against the date a link was last actually verified, not
the date a row was last touched — an entry can be edited without anyone
re-checking whether its URL still resolves.
"""

from __future__ import annotations

from datetime import date, datetime

from . import model as m
from . import taxonomy as tx
from ..config import INDEX_COMPLETENESS_TARGET


def completeness(ent: dict) -> float:
    """Fraction of the applicable fields that are populated (0..1)."""
    fields = m.COMPLETENESS_FIELDS
    if not fields:
        return 1.0
    filled = 0
    for f in fields:
        v = ent.get(f)
        if isinstance(v, list):
            filled += 1 if v else 0
        elif str(v or "").strip():
            filled += 1
    return filled / len(fields)


def days_since_verified(ent: dict, today: date | None = None) -> int | None:
    raw = str(ent.get("date_last_verified") or "").strip()
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(raw).date()
    except ValueError:
        return None
    return ((today or date.today()) - d).days


def issues(ent: dict) -> list[str]:
    """Every quality problem this entry has, as short human-readable strings."""
    out: list[str] = []
    if not str(ent.get("url") or "").strip():
        out.append("no URL recorded")
    elif ent.get("link_status") == "error":
        out.append("link did not resolve")
    elif ent.get("link_status") == "blocked":
        out.append("link blocked to automated checks (may be fine in a browser)")
    elif ent.get("link_status") in ("unchecked", None, ""):
        out.append("link never checked")
    if not str(ent.get("description") or "").strip():
        out.append("no description")
    if not ent.get("insight_types"):
        out.append("no insight-type tags")
    if ent.get("confidence") == "low":
        out.append("low confidence — identity or official source unconfirmed")
    if completeness(ent) < INDEX_COMPLETENESS_TARGET:
        out.append("incomplete profile")
    if ent.get("entity_type") == "person" and not str(ent.get("current_affiliation") or "").strip():
        out.append("person with no current affiliation")
    return out


def annotate(entities: list[dict]) -> list[dict]:
    """Attach derived quality fields. These are NOT stored on the entity record
    itself in ``entities.json`` — they are computed, so a stale cached score can
    never disagree with the data it describes."""
    out = []
    for ent in entities:
        e = dict(ent)
        e["_completeness"] = round(completeness(ent), 3)
        e["_issues"] = issues(ent)
        e["_days_since_verified"] = days_since_verified(ent)
        out.append(e)
    return out


def orphan_edges(entities: list[dict], relationships: list[dict]) -> list[dict]:
    """Edges pointing at an entity that does not exist. Should always be empty;
    a non-empty result means an import dropped a row without its edges."""
    ids = {e["id"] for e in entities}
    return [r for r in relationships
            if r.get("source") not in ids or r.get("target") not in ids]


def undeclared_tags(entities: list[dict]) -> dict[str, list[str]]:
    """Tags in use that are not in the taxonomy — the signal for growing the
    vocabulary deliberately instead of letting it drift."""
    fields = {"investment_approach": "investment_approach",
              "asset_classes": "asset_class",
              "insight_types": "insight_type"}
    out: dict[str, set[str]] = {v: set() for v in fields.values()}
    for ent in entities:
        for field, vocab in fields.items():
            for t in tx.unknown_terms(vocab, ent.get(field, [])):
                out[vocab].add(t)
    return {k: sorted(v) for k, v in out.items() if v}


def report(entities: list[dict], relationships: list[dict],
           today: date | None = None) -> dict:
    """The full quality picture, as written to status.json and shown in the UI."""
    today = today or date.today()
    ann = annotate(entities)
    n = len(ann)
    verified = [e for e in ann if e.get("lifecycle_state") == "verified"]
    needs = [e for e in ann if e.get("lifecycle_state") == "needs_review"]
    link_ok = [e for e in ann if e.get("link_status") == "ok"]
    blocked = [e for e in ann if e.get("link_status") == "blocked"]
    broken = [e for e in ann if e.get("link_status") == "error"]
    no_url = [e for e in ann if not str(e.get("url") or "").strip()]
    stale_days = [e["_days_since_verified"] for e in ann
                  if e.get("_days_since_verified") is not None]

    by = lambda key: _counter(ann, key)                        # noqa: E731
    return {
        "total": n,
        "by_entity_type": by("entity_type"),
        "by_primary_category": by("primary_category"),
        "by_lifecycle_state": by("lifecycle_state"),
        "by_confidence": by("confidence"),
        "by_link_status": by("link_status"),
        "verified": len(verified),
        "needs_review": len(needs),
        "link_ok": len(link_ok),
        "link_blocked": len(blocked),
        "link_broken": len(broken),
        "missing_url": len(no_url),
        "relationships": len(relationships),
        "orphan_edges": len(orphan_edges(entities, relationships)),
        "mean_completeness": round(sum(e["_completeness"] for e in ann) / n, 3) if n else 0.0,
        "incomplete": sum(1 for e in ann if e["_completeness"] < INDEX_COMPLETENESS_TARGET),
        "oldest_verification_days": max(stale_days) if stale_days else None,
        "never_verified": sum(1 for e in ann if e.get("_days_since_verified") is None),
        "undeclared_tags": undeclared_tags(entities),
        "as_of": today.isoformat(),
    }


def _counter(entities: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entities:
        v = str(e.get(key) or "unknown")
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def review_queue(entities: list[dict], limit: int = 200) -> list[dict]:
    """Entries most in need of attention, worst first.

    Ordered by issue count then by lowest completeness, so the rows at the top
    are the ones where a few minutes of work removes the most uncertainty.
    """
    ann = [e for e in annotate(entities) if e["_issues"]]
    ann.sort(key=lambda e: (-len(e["_issues"]), e["_completeness"], e["name"]))
    return [{"name": e["name"], "id": e["id"], "entity_type": e["entity_type"],
             "confidence": e.get("confidence"), "lifecycle_state": e.get("lifecycle_state"),
             "link_status": e.get("link_status"), "url": e.get("url", ""),
             "completeness": e["_completeness"], "issues": e["_issues"]}
            for e in ann[:limit]]

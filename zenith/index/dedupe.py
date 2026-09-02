"""INDEX deduplication — collapsing the same entity arriving under several names.

The seed list genuinely contains duplicates, and they are of three different
kinds, which is why this module distinguishes between them rather than merging
everything that looks similar:

  1. THE SAME ENTITY, LISTED TWICE. "OFFICE OF FINANCIAL RESEARCH (OFR)" appears
     under both institutions and tools; "CME" appears under institutions and
     again under tools as "CME (E.G., FED WATCH)"; "S&P GLOBAL" and "SPGI" are
     a name and its ticker. These MERGE.

  2. A BRAND OR DIVISION OF A PARENT. "iShares" is BlackRock's ETF brand;
     "Man AHL" and "Man Institute" are divisions of Man Group; "SPDR" is State
     Street's ETF brand. These must NOT merge — a researcher looks for either
     name and both deserve their own row — so they are joined by a
     ``subsidiary_of`` EDGE instead (declared in seed.py).

  3. COINCIDENTAL SIMILARITY. Different firms with overlapping words. These
     must not merge at all.

Matching therefore uses only high-precision keys — slug, alias, and registered
domain — never fuzzy string similarity, which would happily merge "Capital
Group" with "Capital Fund Management". Anything less certain than that is
reported as a *candidate* for a human to look at, not merged automatically.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from . import model as m

# Platforms that many unrelated entities legitimately share, so a domain match
# on them means nothing. Without this list every journal hosted on Elsevier,
# Wiley or OUP would collapse into a single entity, as would every podcast on a
# shared hosting service.
#
# NOTE these must be REGISTERED domains (oup.com, not academic.oup.com), because
# that is what they are compared against — an earlier version listed the full
# hostname and consequently merged the Review of Financial Studies into the
# Review of Asset Pricing Studies.
_GENERIC_HOSTS = {
    "sciencedirect.com", "tandfonline.com", "oup.com", "wiley.com",
    "pm-research.com", "risk.net", "simplecast.com", "libsyn.com", "megaphone.fm",
    "captivate.fm", "buzzsprout.com", "anchor.fm", "audioboom.com", "omnycontent.com",
    "substack.com", "blogspot.com", "wordpress.com", "medium.com", "google.com",
    "bloomberg.com", "youtube.com", "apple.com", "spotify.com",
}

# Entity types whose website IS their identity, so a shared domain is real
# evidence of duplication. `person` is deliberately excluded: a person's URL
# normally points at their EMPLOYER's site, so matching people on domain merges
# every researcher into their own firm — which an earlier version of this module
# did, silently destroying 28 person records.
_DOMAIN_MATCHABLE = {"organisation", "tool", "academic_source", "podcast", "publication"}


def registered_domain(url: str) -> str:
    """Best-effort registered domain ('www.aqr.com/Insights' -> 'aqr.com').

    Deliberately simple — a two-label heuristic with a short list of known
    multi-part public suffixes. A full PSL dependency is not justified for a
    directory of a few hundred hand-curated URLs.
    """
    try:
        host = urlsplit(str(url or "").strip()).netloc.lower()
    except Exception:
        return ""
    if not host:
        return ""
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    multi = {"co.uk", "org.uk", "ac.uk", "com.au", "co.jp", "co.za", "com.br", "co.in"}
    if ".".join(parts[-2:]) in multi and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _keys(ent: dict) -> set[str]:
    """The high-precision identity keys an entity can be matched on."""
    keys = {"slug:" + str(ent.get("slug") or "")}
    for alias in ent.get("aliases", []):
        s = m.slug_for(alias)
        if s:
            keys.add("slug:" + s)
    if ent.get("entity_type") in _DOMAIN_MATCHABLE:
        dom = registered_domain(ent.get("url"))
        if dom and dom not in _GENERIC_HOSTS:
            keys.add("dom:" + dom)
    return {k for k in keys if not k.endswith(":")}


def find_duplicates(entities: list[dict]) -> list[list[int]]:
    """Group indices of entities that share a high-precision key.

    Entities already joined by a ``subsidiary_of`` edge are NOT considered
    duplicates — that check happens in ``deduplicate()``, which has the edges.
    """
    key_to_idx: dict[str, list[int]] = {}
    for i, ent in enumerate(entities):
        for k in _keys(ent):
            key_to_idx.setdefault(k, []).append(i)

    parent: dict[int, int] = {i: i for i in range(len(entities))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for idxs in key_to_idx.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    groups: dict[int, list[int]] = {}
    for i in range(len(entities)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def deduplicate(entities: list[dict], relationships: list[dict] | None = None
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """Merge true duplicates; leave parent/brand pairs alone.

    Returns ``(entities, relationships, report)``. The report records every
    merge performed AND every pair that was deliberately left separate because
    a ``subsidiary_of`` edge already explains the overlap — so the Data
    Management view can show what the importer actually did rather than
    silently changing the catalog.
    """
    relationships = list(relationships or [])
    # ANY declared relationship between two entities is evidence they are
    # distinct-but-related, not the same thing said twice. Restricting this to
    # subsidiary_of alone was too narrow: Morningstar and The Long View share a
    # registered domain and are joined by a `publishes` edge, and would
    # otherwise have been merged into one row.
    declared_pairs = {
        frozenset((r["source"], r["target"])) for r in relationships
        if r.get("source") != r.get("target")
    }
    report: list[dict] = []
    drop: set[int] = set()
    remap: dict[str, str] = {}

    for group in find_duplicates(entities):
        group_ids = [entities[i]["id"] for i in group]
        keep_i, *rest = group
        for j in rest:
            a, b = entities[keep_i], entities[j]
            # An entity is kept separate if it has a declared relationship with
            # ANY member of the group, not merely with the group's chosen
            # representative. Siblings are why: Man Institute and Man AHL are
            # both divisions of Man Group and share its domain, but have no edge
            # to EACH OTHER — so comparing only against the representative
            # merged the two whenever Man AHL happened to sort first, silently
            # losing a row depending on input order.
            if any(frozenset((b["id"], other)) in declared_pairs
                   for other in group_ids if other != b["id"]):
                report.append({
                    "action": "kept_separate", "primary": a["name"], "other": b["name"],
                    "reason": "related, not duplicate — an explicit relationship ties it to "
                              "another entity in this group (parent/brand or publisher)",
                })
                continue
            entities[keep_i] = m.merge(a, {
                **{k: v for k, v in b.items() if k not in ("id", "slug", "name")},
                "aliases": list(b.get("aliases", [])) + [b["name"]],
            })
            remap[b["id"]] = a["id"]
            drop.add(j)
            report.append({
                "action": "merged", "primary": a["name"], "other": b["name"],
                "reason": "same entity under a second name or domain",
            })

    kept = [e for i, e in enumerate(entities) if i not in drop]

    # Re-point edges at the surviving id, then drop self-loops and duplicates.
    seen: set[tuple] = set()
    out_rels: list[dict] = []
    kept_ids = {e["id"] for e in kept}
    for r in relationships:
        s = remap.get(r["source"], r["source"])
        t = remap.get(r["target"], r["target"])
        if s == t or s not in kept_ids or t not in kept_ids:
            continue
        key = (s, t, r.get("type"))
        if key in seen:
            continue
        seen.add(key)
        out_rels.append({**r, "source": s, "target": t})
    return kept, out_rels, report


def duplicate_candidates(entities: list[dict],
                         relationships: list[dict] | None = None) -> list[dict]:
    """Lower-confidence overlaps worth a human look, NOT merged automatically.

    Entities sharing a registered domain, EXCLUDING two cases that are normal
    rather than suspicious:

      * a person and their employer — a person's URL points at the firm's site,
        so this pair is expected and reporting it buries the real signal;
      * pairs with an explicit relationship already declared between them.

    What survives is the useful case: two same-type entities on one domain with
    nothing declared to explain it, which usually means a brand/parent pair
    whose edge has not been written yet.
    """
    declared = {
        frozenset((r["source"], r["target"])) for r in (relationships or [])
    }
    by_dom: dict[str, list[dict]] = {}
    for ent in entities:
        if ent.get("entity_type") not in _DOMAIN_MATCHABLE:
            continue
        dom = registered_domain(ent.get("url"))
        if dom and dom not in _GENERIC_HOSTS:
            by_dom.setdefault(dom, []).append(ent)
    out = []
    for dom, group in sorted(by_dom.items()):
        if len(group) < 2:
            continue
        unexplained = [
            g for g in group
            if not any(frozenset((g["id"], o["id"])) in declared
                       for o in group if o["id"] != g["id"])
        ]
        if len(unexplained) > 1:
            out.append({"domain": dom, "names": [g["name"] for g in unexplained],
                        "ids": [g["id"] for g in unexplained]})
    return out

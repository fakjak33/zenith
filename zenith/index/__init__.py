"""INDEX — Zenith's Master List: a financial intelligence directory & knowledge graph.

Every other Zenith package answers an ANALYTICAL question about securities:
what is trending (MOMENTUM), what is mispriced (IDEAS), what macro regime are
we in (REGIMES). None of them answers the question this package exists for:

    "Who is doing interesting work in finance, where do they work, what do they
    specialise in, and where do I go to learn more?"

INDEX maps the INFORMATION ECOSYSTEM itself -- the firms, people, academic
sources, podcasts and tools an investor actually draws on -- as a searchable,
filterable, exportable directory whose entries are connected to each other.

STRUCTURE. The catalog (`entities.json`) and the relationships between entries
(`relationships.json`, a plain edge list) are stored SEPARATELY on purpose. A
directory row and a graph edge have different lifetimes: a person's firm can
change without the person changing, and Phase 2 will attach thousands of
podcast-appearance edges to entities that already exist. Keeping edges out of
the entity records means that growth never reshapes the catalog or the export.

HONESTY POSTURE -- stated once here, the way the other packages state their
anti-overfitting rules, because for a directory the failure mode is different
and worth naming explicitly:

  1. NO EVIDENCE TIER. Every other Zenith tab carries an A/B/C
     `evidence_rating` badge, but that badge means "how much out-of-sample
     predictive strength does this SIGNAL have". A directory has no such
     property. Stamping a letter grade on it would be inventing exactly the
     fake precision this package is supposed to avoid. INDEX shows a
     VERIFICATION STRIP instead: how many entries there are, how many have a
     URL confirmed live by an actual HTTP request, how many need review, and
     how stale the oldest verification is. Those are facts we measure.

  2. AN EMPTY FIELD BEATS A PLAUSIBLE GUESS. Where a URL, founding date or
     affiliation could not be verified, it is left empty and the entity is
     flagged `needs_review` -- never filled in with something that merely looks
     right. `confidence` records how sure we are, and `provenance` records
     where each entry came from.

  3. LINK STATUS IS MEASURED, NOT ASSUMED. `verified` means an HTTP request
     actually succeeded (links.py), with the date recorded. A site that
     anti-bot-walls us (ATS Trading Solutions returns a Cloudflare challenge;
     Citadel's robots.txt disallows crawling -- both already documented in
     zenith/sources.py) is reported as BLOCKED, which is honest, rather than
     as broken, which is not.

  4. HISTORY IS PRESERVED, NOT OVERWRITTEN. When a person changes firm,
     `model.merge()` moves the old affiliation into `historical_affiliations`
     rather than discarding it. The current affiliation is always identifiable,
     but the record of where someone came from is never silently lost.

  5. RANKING IS SUBJECTIVE AND SAYS SO. No entry carries a fabricated quality
     score. (Phase 3's relevance ranking will publish its methodology in full
     and label itself an opinion.)

SEED PROVENANCE. `data/index/seed/` holds the user's raw supplied lists
verbatim -- the resource list this package was built from, and a prior
best-effort podcast-guest compilation whose own header admits it is partial.
Nothing is enriched in place: the raw files stay as the audit trail, and every
derived entity points back at them through its `provenance` field.

PHASE 1 (this module) delivers the catalog, taxonomy, dedupe, link health,
quality tracking, search/filter, the directory views and export.
PHASE 2 adds podcast harvesting and the guest graph (all 14 podcasts confirmed
to expose full-archive RSS -- ~5,700 episodes -- via the keyless iTunes Search
API). PHASE 3 adds the network visualisation, the discovery surface and
transparent relevance ranking. The schema already reserves room for both.
"""

from __future__ import annotations

import json

from ..config import INDEX_FILES

DISCLAIMER = (
    "INDEX is a curated DIRECTORY of financial research resources -- institutions, people, "
    "academic sources, podcasts and tools -- not a signal engine and not investment advice. "
    "Inclusion is not endorsement, and the descriptive tags (strategy, asset class, insight "
    "type) are editorial classifications, not claims about performance. Every entry carries "
    "its provenance, a confidence level and the date its link was last verified; entries we "
    "could not confirm are flagged 'needs review' with the field left empty rather than "
    "filled with a guess."
)


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(INDEX_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write an INDEX artefact.

    Pretty-printed by default: unlike MOM's ~1000-row daily score files, these
    artefacts are small, change rarely, and are genuinely read by hand in diffs
    (a directory's value is in its human-checkable content), so the extra bytes
    buy real reviewability. `episodes` (Phase 2, thousands of rows) should pass
    indent=None for the same reason mom/__init__.py's save() does.
    """
    INDEX_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILES[name].write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

"""OPTIONS VAULT — an evidence-graded reference of options strategies, edges and
portfolio mechanics, distilled from tastylive research content and cross-checked
against the academic and practitioner literature.

SOURCES. The corpus is tastylive's research output: Market Measures (2012-), From
Theory to Practice, Options Jive, The Skinny on Options (Math / Abstract
Applications / Data Science) and the smaller research series, plus the written
Learn Center (concepts & strategies, definitions), research-style news articles
and tastytrade help-center pages on margin, assignment, rolling and account types.
Phase 01 (harvest.py) builds the inventory and transcript cache; corpus.py holds
the pure helpers (VTT parsing, show classification, episode matching).

WHERE THE TEXT COMES FROM (verified 2026-10-23, see prompts/vault/HANDOFF.md).
tastylive.com's episode sitemap lists the whole archive (title, description, tags,
date) but episode pages older than ~3 months redirect as "expired" and their media
404, so tastylive.com is a METADATA source only. Transcripts come from YouTube
captions on the @tastyliveshow channel, which carries only a fraction of the
archive. Every manifest row records where its transcript came from (or "none").

CITATIONS. One format everywhere: ``ep:<episode_id>@mm:ss``, ``lit:<paper_id>``,
``repro:<name>``, ``page:<slug>``. Every number shown in the app traces to one.

EVIDENCE TIERS (ui_theme.evidence_rating). A = replicated across periods and
markets in peer-reviewed or major practitioner research and consistent with
tastylive data; B = documented but period/cost/implementation sensitive, or
tastylive plus one independent source; C = tastylive-only, small sample, period
specific or contradicted. Management CONVENTIONS (21 DTE, 50% profit) are labelled
"rule of thumb" with their effect on the P&L distribution instead of a tier.

COPYRIGHT. Transcripts and page text stay in data/vault/cache/ (gitignored): never
committed, never displayed. Committed artefacts hold paraphrased takeaways and
numbers only, each linked to its source.
"""

from __future__ import annotations

import json

from ..config import VAULT_FILES

DISCLAIMER = ("OPTIONS VAULT condenses tastylive educational content (mostly SPY/SPX studies "
              "with mid-price fills and unpublished methodology, often over short windows) and "
              "published research. Short premium has fat left tails (2008, Feb 2018, Mar 2020). "
              "Every number links to its source; check it before acting. Decision-support, not "
              "investment advice.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(name: str, default=None):
    """Load a named VAULT artefact (see config.VAULT_FILES)."""
    return _read(VAULT_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(VAULT_FILES[name], obj)

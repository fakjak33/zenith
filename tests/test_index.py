"""INDEX (Master List) tests — offline, synthetic, no network.

Follows the repo convention: nothing here touches the network, and the checks
that matter most are the ones guarding the guarantees the feature makes about
itself — history is preserved, nothing is marked verified without evidence,
the taxonomy grows without code changes, and the export survives leaving Zenith.

Several tests are REGRESSION tests for bugs found while building this, and are
labelled as such so nobody quietly reverts the fix:
  * ``test_slug_keeps_short_distinctive_names``
  * ``test_person_not_merged_into_their_employer``
  * ``test_sibling_brands_survive_regardless_of_order``
  * ``test_generic_publisher_hosts_do_not_merge_journals``
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from zenith.index import dedupe, export, links, model as m, quality, seed, taxonomy as tx


# ------------------------------------------------------------- name handling --
def test_slug_strips_legal_but_not_descriptive_suffixes():
    assert m.slug_for("Gresham Investment Management LLC") == "gresham_investment"
    assert m.slug_for("GRESHAM INVESTMENT MANAGEMENT") == "gresham_investment"
    assert m.slug_for("BlackRock, Inc.") == m.slug_for("BlackRock")
    assert m.slug_for("Baker Bros. Advisors LP") == "baker_bros_advisors"


def test_slug_keeps_short_distinctive_names():
    """REGRESSION: stripping descriptive suffixes unconditionally collapsed
    'Man Group' to 'man' and 'Capital Group' to 'capital' — generic enough to
    collide with unrelated entries."""
    assert m.slug_for("Man Group") == "man_group"
    assert m.slug_for("Capital Group") == "capital_group"
    assert m.slug_for("Man Group") != m.slug_for("Man AHL")


def test_normalize_name_preserves_real_mixed_case():
    assert m.normalize_name("  iShares   ") == "iShares"
    assert m.normalize_name("S&P   Global") == "S & P Global"


# ------------------------------------------------------------------ taxonomy --
def test_resolve_maps_aliases_and_spellings():
    assert tx.resolve("investment_approach", "quant") == "quantitative"
    assert tx.resolve("investment_approach", "Trend Following") == "trend_following"
    assert tx.resolve("asset_class", "forex") == "fx"
    assert tx.resolve("entity_type", "organization") == "organisation"


def test_unknown_terms_are_preserved_not_dropped():
    resolved = tx.resolve_many("asset_class", ["equities", "Martian Bonds"])
    assert "equities" in resolved and "martian_bonds" in resolved
    assert tx.unknown_terms("asset_class", ["equities", "Martian Bonds"]) == ["martian_bonds"]


def test_taxonomy_extends_without_code(monkeypatch):
    """The extensibility promise: a new term is a dict entry, nothing else."""
    table = dict(tx.INVESTMENT_APPROACH)
    table["insurance_linked"] = ("Insurance-linked securities", ("ils", "cat bonds"))
    monkeypatch.setitem(tx.VOCABULARIES, "investment_approach", table)
    assert tx.resolve("investment_approach", "cat bonds") == "insurance_linked"
    assert tx.label_of("investment_approach", "insurance_linked") == "Insurance-linked securities"


def test_label_of_falls_back_readably():
    assert tx.label_of("investment_approach", "some_new_thing") == "Some New Thing"


# --------------------------------------------------------------------- merge --
def test_merge_preserves_historical_affiliation():
    """The central promise: a person changing firms keeps their history."""
    a = m.make("Jane Doe", entity_type="person", current_affiliation="Alpha Capital",
               lifecycle_state="verified")
    b = m.merge(a, {"current_affiliation": "Beta Partners"})
    assert b["current_affiliation"] == "Beta Partners"
    assert b["historical_affiliations"] == ["Alpha Capital"]
    assert b["lifecycle_state"] == "updated"

    c = m.merge(b, {"current_affiliation": "Gamma Advisors"})
    assert c["current_affiliation"] == "Gamma Advisors"
    assert c["historical_affiliations"] == ["Alpha Capital", "Beta Partners"]


def test_merge_never_overwrites_a_value_with_an_empty_one():
    a = m.make("Acme", description="A real description", url="https://acme.test")
    b = m.merge(a, {"description": "", "url": None})
    assert b["description"] == "A real description"
    assert b["url"] == "https://acme.test"


def test_merge_unions_lists_and_does_not_duplicate():
    a = m.make("Acme", insight_types=["research"], aliases=["ACME"])
    b = m.merge(a, {"insight_types": ["research", "market_data"], "aliases": ["ACME", "Acme Inc"]})
    assert b["insight_types"] == ["research", "market_data"]
    assert b["aliases"] == ["ACME", "Acme Inc"]


def test_merge_keeps_archived_sticky():
    a = m.make("Defunct Co", lifecycle_state="archived")
    b = m.merge(a, {"lifecycle_state": "verified", "description": "changed"})
    assert b["lifecycle_state"] == "archived"


def test_merge_moves_verification_date_forward_only():
    a = m.make("Acme", date_last_verified="2026-08-01")
    assert m.merge(a, {"date_last_verified": "2026-07-01"})["date_last_verified"] == "2026-08-01"
    assert m.merge(a, {"date_last_verified": "2026-09-01"})["date_last_verified"] == "2026-09-01"


# ---------------------------------------------------------------------- search --
def test_search_blob_matches_tag_labels_not_just_slugs():
    e = m.make("Acme", investment_approach=["trend"], insight_types=["academic_research"])
    blob = m.search_blob(e)
    assert "trend following" in blob and "trend_following" in blob
    assert "academic research" in blob


# --------------------------------------------------------------------- dedupe --
def _org(name, url="", etype="organisation", **kw):
    return m.make(name, entity_type=etype, url=url, **kw)


def test_same_entity_under_two_names_merges():
    """The real seed cases: OFR listed twice, CME listed twice, S&P/SPGI."""
    ents = [_org("Office of Financial Research", "https://www.financialresearch.gov",
                 aliases=["OFR"]),
            _org("OFR", "https://www.financialresearch.gov")]
    kept, _rels, report = dedupe.deduplicate(ents, [])
    assert len(kept) == 1
    assert any(r["action"] == "merged" for r in report)


def test_ticker_alias_merges_with_company_name():
    ents = [_org("S&P Global", "https://www.spglobal.com", aliases=["SPGI"]),
            _org("SPGI", "https://www.spglobal.com")]
    kept, _r, _rep = dedupe.deduplicate(ents, [])
    assert len(kept) == 1


def test_person_not_merged_into_their_employer():
    """REGRESSION: matching people on domain merged every researcher into their
    own firm, because a person's URL points at their employer's site. This
    silently destroyed 28 person records."""
    firm = _org("Tier1 Alpha", "https://tier1alpha.com")
    people = [_org(n, "https://tier1alpha.com", etype="person")
              for n in ("Mike Green", "Craig Peterson", "David Pegler")]
    kept, _r, _rep = dedupe.deduplicate([firm] + people, [])
    assert len(kept) == 4
    assert {e["name"] for e in kept} == {"Tier1 Alpha", "Mike Green", "Craig Peterson",
                                         "David Pegler"}


def test_parent_and_brand_kept_separate_when_related():
    parent = _org("BlackRock", "https://www.blackrock.com")
    brand = _org("iShares", "https://www.blackrock.com")     # same domain on purpose
    rels = [m.edge(brand["id"], parent["id"], "subsidiary_of")]
    kept, _r, report = dedupe.deduplicate([parent, brand], rels)
    assert len(kept) == 2
    assert any(r["action"] == "kept_separate" for r in report)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_sibling_brands_survive_regardless_of_order(order):
    """REGRESSION: Man Institute and Man AHL are both divisions of Man Group and
    share its domain, but have no edge to EACH OTHER. Comparing only against the
    group's chosen representative merged two of them whenever the sort order put
    a sibling first — losing a row depending on input order alone."""
    parent = _org("Man Group", "https://www.man.com")
    a = _org("Man Institute", "https://www.man.com/insights")
    b = _org("Man AHL", "https://www.man.com/ahl")
    rels = [m.edge(a["id"], parent["id"], "subsidiary_of"),
            m.edge(b["id"], parent["id"], "subsidiary_of")]
    ents = [[parent, a, b][i] for i in order]
    kept, _r, _rep = dedupe.deduplicate(ents, rels)
    assert len(kept) == 3, [e["name"] for e in kept]


def test_generic_publisher_hosts_do_not_merge_journals():
    """REGRESSION: the generic-host list held full hostnames while the code
    compared registered domains, so academic.oup.com never matched and the
    Review of Financial Studies was merged into the Review of Asset Pricing
    Studies."""
    a = _org("Review of Financial Studies", "https://academic.oup.com/rfs",
             etype="academic_source")
    b = _org("Review of Asset Pricing Studies", "https://academic.oup.com/raps",
             etype="academic_source")
    kept, _r, _rep = dedupe.deduplicate([a, b], [])
    assert len(kept) == 2


def test_registered_domain_extraction():
    assert dedupe.registered_domain("https://www.aqr.com/Insights") == "aqr.com"
    assert dedupe.registered_domain("https://academic.oup.com/rfs") == "oup.com"
    assert dedupe.registered_domain("https://foo.co.uk/bar") == "foo.co.uk"
    assert dedupe.registered_domain("") == ""


def test_dedupe_repoints_edges_and_leaves_no_orphans():
    a = _org("Acme Research", "https://acme.test", aliases=["Acme"])
    b = _org("Acme", "https://acme.test")
    other = _org("Beta Corp", "https://beta.test")
    rels = [m.edge(b["id"], other["id"], "related_to")]
    kept, krels, _rep = dedupe.deduplicate([a, b, other], rels)
    assert len(kept) == 2
    assert quality.orphan_edges(kept, krels) == []


# --------------------------------------------------------------- link status --
def test_apply_link_results_only_verifies_on_a_live_link():
    ents = [m.make("Live", url="https://live.test"),
            m.make("Blocked", url="https://blocked.test"),
            m.make("Dead", url="https://dead.test")]
    results = {
        "https://live.test": {"url": "https://live.test", "status": "ok"},
        "https://blocked.test": {"url": "https://blocked.test", "status": "blocked"},
        "https://dead.test": {"url": "https://dead.test", "status": "error"},
    }
    out = {e["name"]: e for e in links.apply_to_entities(ents, results)}
    assert out["Live"]["lifecycle_state"] == "verified"
    assert out["Live"]["date_last_verified"]
    # Blocked is NOT an error and NOT a verification — it stays as it was.
    assert out["Blocked"]["lifecycle_state"] == "new"
    assert out["Dead"]["lifecycle_state"] == "needs_review"


def test_link_status_does_not_resurrect_archived_entries():
    ents = [m.make("Gone", url="https://gone.test", lifecycle_state="archived")]
    results = {"https://gone.test": {"url": "https://gone.test", "status": "error"}}
    assert links.apply_to_entities(ents, results)[0]["lifecycle_state"] == "archived"


def test_link_ok_does_not_clear_an_identity_review_flag():
    """A URL responding does not resolve an AMBIGUOUS IDENTITY — those are
    different questions, and conflating them would silently mark unconfirmed
    entities as verified."""
    ents = [m.make("Ambiguous", url="https://x.test", lifecycle_state="needs_review",
                   confidence="low")]
    out = links.apply_to_entities(ents, {"https://x.test": {"url": "https://x.test",
                                                            "status": "ok"}})
    assert out[0]["lifecycle_state"] == "needs_review"


def test_sweep_respects_ttl_and_makes_no_calls_when_nothing_is_due(monkeypatch):
    called = []
    monkeypatch.setattr(links, "check_url", lambda u, **kw: called.append(u) or {})
    ents = [m.make("A", url="https://a.test")]
    previous = {"https://a.test": {"url": "https://a.test", "status": "ok",
                                   "checked": "2999-01-01"}}
    out = links.sweep(ents, previous, ttl_days=14)
    assert called == []
    assert out["https://a.test"]["status"] == "ok"


# ------------------------------------------------------------------- quality --
def test_issues_flags_the_things_that_matter():
    bare = m.make("Nothing Known")
    found = quality.issues(bare)
    assert "no URL recorded" in found
    assert "no description" in found
    assert "no insight-type tags" in found


def test_report_counts_and_orphan_detection():
    a = m.make("A", url="https://a.test", description="d", insight_types=["research"],
               provenance="test", lifecycle_state="verified", link_status="ok")
    b = m.make("B", url="https://b.test", description="d", insight_types=["research"],
               provenance="test", lifecycle_state="needs_review", link_status="error")
    rels = [m.edge(a["id"], b["id"], "related_to"),
            m.edge(a["id"], "missing-id", "related_to")]
    rep = quality.report([a, b], rels)
    assert rep["total"] == 2
    assert rep["verified"] == 1 and rep["needs_review"] == 1
    assert rep["orphan_edges"] == 1
    assert rep["by_link_status"]["ok"] == 1


def test_review_queue_is_worst_first():
    good = m.make("Good", url="https://g.test", description="d",
                  insight_types=["research"], provenance="p", link_status="ok")
    bad = m.make("Bad")
    queue = quality.review_queue([good, bad])
    assert queue[0]["name"] == "Bad"


# -------------------------------------------------------------------- export --
def _sample():
    a = m.make("Acme Capital", url="https://acme.test", description="A firm",
               investment_approach=["trend"], asset_classes=["commodities"],
               insight_types=["research"], provenance="test")
    b = m.make("Jane Doe", entity_type="person", current_affiliation="Acme Capital",
               url="https://jane.test", description="A person", provenance="test")
    return [a, b], [m.edge(b["id"], a["id"], "works_at")]


def test_csv_round_trip_preserves_rows_and_fields():
    ents, rels = _sample()
    raw = export.to_csv(ents, rels)
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert len(df) == len(ents)
    assert set(df["name"]) == {"Acme Capital", "Jane Doe"}
    assert "https://acme.test" in set(df["url"])
    row = df[df["name"] == "Acme Capital"].iloc[0]
    assert "Trend following" in row["investment_approach_labels"]
    assert "employs: Jane Doe" in row["relationships"]


def test_csv_uses_bom_so_excel_reads_accents():
    """Without the BOM, 'Torsten Sløk' arrives mojibaked in Excel on Windows."""
    ents = [m.make("Torsten Sløk", entity_type="person", provenance="t")]
    raw = export.to_csv(ents, [])
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "Sløk" in pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")["name"].iloc[0]


def test_json_export_is_full_fidelity_and_self_describing():
    ents, rels = _sample()
    payload = json.loads(export.to_json(ents, rels).decode("utf-8"))
    assert payload["counts"]["entities"] == 2
    assert len(payload["entities"]) == 2 and len(payload["relationships"]) == 1
    assert payload["entities"][0]["investment_approach"] == ["trend_following"]
    assert "investment_approach" in payload["taxonomy"]


def test_xlsx_export_has_the_documented_sheets():
    ents, rels = _sample()
    raw = export.to_xlsx(ents, rels, {"total": 2})
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    assert {"Master List", "Relationships", "Taxonomy", "Status"} <= set(sheets)
    assert len(sheets["Master List"]) == 2
    assert set(sheets["Relationships"]["source"]) == {"Jane Doe"}
    assert set(sheets["Relationships"]["target"]) == {"Acme Capital"}


def test_export_filename_shape():
    assert export.filename("csv", "2026-09-01") == "zenith_master_list_2026-09-01.csv"


# ---------------------------------------------------------------------- seed --
def test_seed_builds_a_coherent_catalog():
    ents, rels = seed.build()
    assert len(ents) >= 150
    assert quality.orphan_edges(ents, rels) == []
    slugs = [e["slug"] for e in ents]
    assert len(slugs) == len(set(slugs)), "duplicate slugs in the seed"
    for e in ents:
        assert e["primary_category"] in tx.PRIMARY_CATEGORIES
        assert e["entity_type"] in tx.ENTITY_TYPES
        assert e["confidence"] in tx.CONFIDENCE_LEVELS
        assert e["lifecycle_state"] in tx.LIFECYCLE_STATES


def test_seed_declares_all_fourteen_monitored_podcasts():
    ents, _r = seed.build()
    podcasts = {e["name"] for e in ents if e["entity_type"] == "podcast"}
    assert len(podcasts) == 14, sorted(podcasts)
    for expected in ("Top Traders Unplugged", "Flirting with Models", "Capital Allocators",
                     "Rational Reminder", "Odd Lots", "COMPLEXITY", "Alpha Exchange"):
        assert expected in podcasts


def test_seed_never_asserts_a_url_it_could_not_confirm():
    """The honesty rule: an unresolved entity ships with an EMPTY url and a
    needs_review flag, not a plausible-looking guess."""
    ents, _r = seed.build()
    for e in ents:
        if not e["url"]:
            assert e["confidence"] == "low", e["name"]
            assert e["lifecycle_state"] == "needs_review", e["name"]


def test_seed_relationships_are_typed_and_declared():
    _e, rels = seed.build()
    assert rels
    for r in rels:
        assert r["type"] in tx.RELATIONSHIP_TYPES
        assert r["source"] != r["target"]


def test_robert_carver_is_the_worked_reference_example():
    """The spec's named example — person linked to blog, books, strategy tags
    and preserved employment history."""
    ents, _r = seed.build()
    carver = next(e for e in ents if e["name"] == "Robert Carver")
    assert carver["entity_type"] == "person"
    assert "qoppac.blogspot.com" in carver["url"]
    assert "trend_following" in carver["investment_approach"]
    assert "systematic" in carver["investment_approach"]
    assert any("Systematic Trading" in b for b in carver["books"])
    assert carver["historical_affiliations"], "his AHL/Barclays history should be recorded"


def test_person_firm_edges_exist_from_day_one():
    """The parentheticals in the source list are person data, so the graph has
    real edges before any podcast harvesting."""
    ents, rels = seed.build()
    by_id = {e["id"]: e for e in ents}
    works_at = [r for r in rels if r["type"] == "works_at"]
    assert len(works_at) >= 20
    kaminski = next(e for e in ents if e["name"] == "Katy Kaminski")
    targets = {by_id[r["target"]]["name"] for r in works_at if r["source"] == kaminski["id"]}
    assert "AlphaSimplex Group" in targets

"""INDEX Phase 2 tests — podcast harvesting, guest extraction and promotion.

Offline and synthetic, per the repo convention: no test here touches the
network. The live-artefact checks at the bottom read only committed JSON.

Several tests are REGRESSION tests for bugs found by auditing the real
6,515-episode corpus while building this, and say so, because each of them
represents a rule that looks arbitrary until you know what it prevents:
  * ``test_every_declared_pattern_exists``
  * ``test_surnames_ending_in_ing_are_not_gerunds``
  * ``test_positional_patterns_are_not_used_as_generic_fallback``
  * ``test_positional_mode_allows_single_word_firms``
  * ``test_merge_gives_a_person_their_first_affiliation``
"""

from __future__ import annotations

import pytest

from zenith import config
from zenith.index import (guests, model as m, podcasts as idx_podcasts,
                          prior_db, promote, quality, seed)


# ------------------------------------------------------------- feed registry --
def test_every_declared_pattern_exists():
    """REGRESSION: several shows declared pattern names ("colon_suffix",
    "dash_suffix") that were never implemented, so those feeds silently fell
    through to the loose generic fallback and produced junk guests."""
    unknown = [(p.name, s) for p in idx_podcasts.PODCASTS
               for s in p.patterns if s not in guests.STRATEGIES]
    assert not unknown, unknown


def test_registry_covers_the_fourteen_shows():
    assert len(idx_podcasts.PODCASTS) == 14
    assert len(idx_podcasts.BY_NAME) == 14
    for pod in idx_podcasts.PODCASTS:
        assert pod.feed_url.startswith("http")
        assert pod.itunes_term
        assert pod.patterns, f"{pod.name} declares no extraction pattern"


def test_registry_names_match_catalog_entities():
    """A podcast's registry name must equal its entity name, or promotion
    cannot attach appearance edges to it."""
    ents, _rels = seed.build()
    podcast_names = {e["name"] for e in ents if e["entity_type"] == "podcast"}
    for pod in idx_podcasts.PODCASTS:
        assert pod.name in podcast_names, pod.name


def test_episode_id_prefers_guid_and_is_stable():
    a = idx_podcasts.episode_id("Show", "guid-1", "http://x", "Title")
    b = idx_podcasts.episode_id("Show", "guid-1", "http://different", "Other title")
    assert a == b, "a stable GUID must win over a changed link or title"
    assert idx_podcasts.episode_id("Show", "", "http://x", "Title") != a


def test_merge_episodes_preserves_history_and_reports_new():
    stored = [{"id": "1", "podcast": "S", "title": "Original", "published": "2026-01-01"}]
    fresh = [{"id": "1", "podcast": "S", "title": "RETITLED", "published": "2026-01-01"},
             {"id": "2", "podcast": "S", "title": "Brand new", "published": "2026-02-01"}]
    merged, new = idx_podcasts.merge_episodes(stored, fresh)
    assert len(merged) == 2
    assert [e["id"] for e in new] == ["2"]
    kept = next(e for e in merged if e["id"] == "1")
    assert kept["title"] == "Original", "a later retitle must not rewrite the record"


def test_clean_text_strips_html():
    assert idx_podcasts.clean_text("<p>Hello &amp; welcome</p>") == "Hello & welcome"


# ------------------------------------------------------------ name validator --
@pytest.mark.parametrize("name", [
    "Rob Carver", "David Harding", "Ulrike Hoffmann-Burchardi", "Louis-Vincent Gave",
    "W. Brian Arthur", "Conor O'Brien", "Jules van Binsbergen", "Alex Fleming",
])
def test_validator_accepts_real_names(name):
    assert guests.looks_like_person(name), name


@pytest.mark.parametrize("junk", [
    "Michael Mauboussin | AI",          # pipe leaked from the title
    "Energizing Lives",                 # gerund-headed phrase
    "JPMorgan's Jay Barry",             # possessive org prefix
    "Understanding the 401(k) Market",  # digits + stopword
    "ALO32: AI Booms",                  # episode code
    "KEEP THE DEFERRED SALES",          # shouted title
    "Aristides Capital",                # organisation
    "Top IPO Scholar",                  # all-caps acronym token
    "Managing Director",                # job title
])
def test_validator_rejects_fragments(junk):
    assert not guests.looks_like_person(junk), junk


def test_surnames_ending_in_ing_are_not_gerunds():
    """REGRESSION: rejecting every -ing token killed ordinary surnames. Worse,
    with "David Harding" failing the person test he was then captured as a
    co-guest's EMPLOYER instead."""
    for name in ("David Harding", "Peyton Manning", "Alex Fleming"):
        assert guests.looks_like_person(name), name
    # All three are guests on that episode: the comma separates co-guests here,
    # so Harding is captured as a PERSON rather than as Michael Adam's employer.
    assert guests._people_from_chunk("Michael Adam, David Harding & Marty Lueck") == [
        ("Michael Adam", "", ""), ("David Harding", "", ""), ("Marty Lueck", "", "")]


def test_canonical_name_strips_honorifics():
    assert guests.canonical_name("Prof. William Goetzmann") == "William Goetzmann"
    assert guests.canonical_name("Sir Tom Beckett") == "Tom Beckett"
    assert guests.canonical_name("  Rob   Carver ") == "Rob Carver"


def test_split_people_handles_multi_guest_but_not_role_commas():
    assert guests.split_people("Andrew Beer & Tom Wrobel") == ["Andrew Beer", "Tom Wrobel"]
    assert guests.split_people("Gene Munster and Doug Clinton") == ["Gene Munster",
                                                                   "Doug Clinton"]
    # A comma introduces a role or firm, never another person.
    assert guests.split_people("Anastasia Titarchuk, CIO") == ["Anastasia Titarchuk, CIO"]


# -------------------------------------------------------------------- firms --
def test_clean_firm_rejects_prose_roles_and_people():
    assert guests.clean_firm("his episode, we speak with Dyn") == ""
    assert guests.clean_firm("Managing Director") == ""
    assert guests.clean_firm("Head of Equity Derivatives Strategy") == ""
    assert guests.clean_firm("Ben Hunt, Brent Kochuba") == ""


def test_clean_firm_normalises_real_firms():
    assert guests.clean_firm("the Acquirers Fund") == "Acquirers Fund"
    assert guests.clean_firm("Verdad Advisers - Emerging Markets") == "Verdad Advisers"
    assert guests.clean_firm("Partner at Ruffer Investment Management") == \
        "Ruffer Investment Management"
    assert guests.clean_firm("Research at Dimensional Fund Advisors") == \
        "Dimensional Fund Advisors"
    assert guests.clean_firm("Research Affiliates") == "Research Affiliates"


def test_positional_mode_allows_single_word_firms():
    """REGRESSION: Transtrend, Winton and Principalium are all real firms in this
    catalog and were being dropped for being one word long."""
    for firm in ("Transtrend", "Winton", "Principalium"):
        assert guests.clean_firm(firm) == "", "not accepted without positional context"
        assert guests.clean_firm(firm, positional=True) == firm
    # Position never excuses prose or a bare job title.
    assert guests.clean_firm("we speak with Dyn", positional=True) == ""
    assert guests.clean_firm("Managing Director", positional=True) == ""


# --------------------------------------------------------- extraction shapes --
def _ep(title, podcast, summary=""):
    return {"id": "e1", "podcast": podcast, "title": title, "summary": summary,
            "published": "2026-01-01", "url": "http://x"}


@pytest.mark.parametrize("podcast,title,expected", [
    ("Alpha Exchange", "Benn Eifert, Founder and CIO, QVR Advisors", "Benn Eifert"),
    ("Top Traders Unplugged", "SI196: Where Next for Trend Following? ft. Rob Carver",
     "Rob Carver"),
    ("Flirting with Models", "Adam Butler - Questioning the Quant Orthodoxy (S5E13)",
     "Adam Butler"),
    ("Capital Allocators", "David Lyon - Hybrid Capital Solutions (EP.471)", "David Lyon"),
    ("The Long View", "Michael Mauboussin: Finding Easy Games", "Michael Mauboussin"),
    ("The Meb Faber Show", "Victor Haghani on Predicting the Market | #588",
     "Victor Haghani"),
    ("The Derivative", "The principles of VIX trading with Alex Orus of Principalium",
     "Alex Orus"),
    ("Monetary Matters",
     "Inside the Alternative Platform Megatrend | Alan Strauss of Crystal Capital",
     "Alan Strauss"),
])
def test_each_show_shape_extracts_its_guest(podcast, title, expected):
    pod = idx_podcasts.BY_NAME[podcast]
    names = [r["name"] for r in guests.extract_from_episode(_ep(title, podcast), pod)]
    assert expected in names, (title, names)


def test_extraction_captures_role_and_firm_where_stated():
    pod = idx_podcasts.BY_NAME["Alpha Exchange"]
    rec = guests.extract_from_episode(
        _ep("Benn Eifert, Founder and CIO, QVR Advisors", "Alpha Exchange"), pod)[0]
    assert rec["name"] == "Benn Eifert"
    assert "CIO" in rec["role"]
    assert rec["firm"] == "QVR Advisors"


def test_of_firm_is_captured_from_the_title():
    pod = idx_podcasts.BY_NAME["The Derivative"]
    rec = guests.extract_from_episode(
        _ep("The principles of VIX trading with Alex Orus of Principalium",
            "The Derivative"), pod)[0]
    assert rec["firm"] == "Principalium"


def test_hosts_are_never_extracted_as_guests():
    pod = idx_podcasts.BY_NAME["Top Traders Unplugged"]
    recs = guests.extract_from_episode(
        _ep("SI400: A chat ft. Niels Kaastrup-Larsen & Rob Carver",
            "Top Traders Unplugged"), pod)
    names = {r["name"] for r in recs}
    assert "Rob Carver" in names
    assert "Niels Kaastrup-Larsen" not in names


def test_positional_patterns_are_not_used_as_generic_fallback():
    """REGRESSION: running colon_prefix generically turned Odd Lots' "Listen
    Now: The Big Take" into a guest called "Listen Now", and The Derivative's
    "Family Offices: an inside look..." into "Family Offices"."""
    for loose in ("colon_prefix", "dash_prefix", "on_infix"):
        assert loose not in guests._GENERIC_ORDER
    # "Listen Now" is name-SHAPED, so the validator alone cannot reject it --
    # which is precisely why the loose positional patterns must not run on
    # shows that do not declare them.
    assert guests.looks_like_person("Listen Now")
    pod = idx_podcasts.BY_NAME["Other People's Money"]   # does not declare colon_prefix
    recs = guests.extract_from_episode(_ep("Listen Now: The Big Take",
                                           "Other People's Money"), pod)
    assert [r["name"] for r in recs] == []


# ---------------------------------------------------------------- vocabulary --
def test_common_vocabulary_filters_phrase_shaped_names():
    episodes = [{"podcast": "S", "id": str(i), "title": "t", "published": "",
                 "url": "", "summary": "the team dynamics of the group were discussed "
                                       "and the dynamics of the team mattered"}
                for i in range(12)]
    vocab = guests.common_vocabulary(episodes)
    assert "dynamics" in vocab and "team" in vocab
    assert guests.is_common_phrase("Team Dynamics", vocab)
    assert not guests.is_common_phrase("Rob Carver", vocab)


# ----------------------------------------------------------------- aggregate --
def _rec(name, podcast, published, **kw):
    base = {"name": name, "role": "", "firm": "", "confidence": "high",
            "strategy": "ft_suffix", "episode_id": f"{podcast}-{published}",
            "podcast": podcast, "episode_title": "t", "published": published,
            "url": "", "corroborated": True}
    base.update(kw)
    return base


def test_aggregate_collects_appearances_and_treats_latest_firm_as_current():
    recs = [_rec("Jane Doe", "Show A", "2024-01-01", firm="Alpha Capital"),
            _rec("Jane Doe", "Show B", "2026-01-01", firm="Beta Partners")]
    prof = guests.aggregate(recs)["jane doe"]
    assert prof["n_appearances"] == 2 and prof["n_podcasts"] == 2
    assert prof["current_firm"] == "Beta Partners"
    assert prof["past_firms"] == ["Alpha Capital"]
    assert prof["appearances"][0]["published"] == "2026-01-01", "newest first"


def test_single_uncorroborated_appearance_is_demoted():
    """A name seen once, in one title, that the show's own notes never mention
    rests on a single structural guess — recorded, but not promoted."""
    prof = guests.aggregate([_rec("Solo Person", "Show A", "2026-01-01",
                                  corroborated=False, confidence="high")])["solo person"]
    assert prof["confidence"] == "low"
    assert not guests.meets_threshold(prof["confidence"])


def test_recurrence_across_shows_promotes_confidence():
    recs = [_rec("Jane Doe", "Show A", "2026-01-01", corroborated=False, confidence="low"),
            _rec("Jane Doe", "Show B", "2026-02-01", corroborated=False, confidence="low")]
    assert guests.aggregate(recs)["jane doe"]["confidence"] == "high"


# ----------------------------------------------------------------- promotion --
def _catalog():
    return [m.make("Show A", entity_type="podcast", provenance="seed"),
            m.make("Alpha Capital", entity_type="organisation", provenance="seed")], []


def _thrice(name, podcast="Show A", **kw):
    return guests.aggregate([_rec(name, podcast, d, **kw)
                             for d in ("2026-01-01", "2026-02-01", "2026-03-01")])


def test_promotion_creates_people_and_appearance_edges():
    ents, rels = _catalog()
    out, out_rels, report = promote.build(ents, rels,
                                          _thrice("Jane Doe", firm="Alpha Capital"))
    jane = next(e for e in out if e["name"] == "Jane Doe")
    assert jane["entity_type"] == "person"
    assert jane["current_affiliation"] == "Alpha Capital"
    assert jane["podcast_appearances"]
    assert jane["lifecycle_state"] == "needs_review", "harvested entries are unverified"
    types = {r["type"] for r in out_rels}
    assert "appeared_on" in types and "works_at" in types
    assert report["created_people"] == 1
    assert quality.orphan_edges(out, out_rels) == []


def test_promotion_skips_low_confidence_guests():
    ents, rels = _catalog()
    profiles = guests.aggregate([_rec("Solo Person", "Show A", "2026-01-01",
                                      corroborated=False)])
    out, _r, report = promote.build(ents, rels, profiles)
    assert report["skipped_low_confidence"] == 1
    assert not any(e["name"] == "Solo Person" for e in out)


def test_promotion_never_overwrites_curated_data():
    """Curated entries win: a hand-written URL, description and provenance must
    survive a harvest that knows none of them."""
    curated = m.make("Jane Doe", entity_type="person", url="https://jane.example",
                     description="Hand written.", provenance="user seed list",
                     lifecycle_state="verified", link_status="ok")
    ents = [curated, m.make("Show A", entity_type="podcast", provenance="seed")]
    out, _rels, report = promote.build(ents, [], _thrice("Jane Doe"))
    jane = next(e for e in out if e["name"] == "Jane Doe")
    assert jane["url"] == "https://jane.example"
    assert jane["description"] == "Hand written."
    assert "user seed list" in jane["provenance"], "original provenance must survive"
    assert promote.PROVENANCE in jane["provenance"], "and the harvest is recorded too"
    assert jane["podcast_appearances"]
    assert report["matched_existing"] == 1


def test_promotion_creates_no_dangling_firm_entities():
    """A firm mentioned once is recorded as affiliation TEXT but does not become
    an unverifiable organisation stub with no URL."""
    ents, rels = _catalog()
    out, out_rels, report = promote.build(ents, rels,
                                          _thrice("Jane Doe", firm="Obscure Shop"))
    jane = next(e for e in out if e["name"] == "Jane Doe")
    assert jane["current_affiliation"] == "Obscure Shop"
    assert report["created_firms"] == 0, "one mention is not enough for its own entry"
    assert quality.orphan_edges(out, out_rels) == []


def test_a_guest_name_never_rewrites_an_organisation():
    """REGRESSION: a title parse that captured an ORGANISATION as a guest name
    ("Goldman Sachs") merged into the existing organisation entity and rewrote
    its entity_type to "person", silently corrupting a curated entry."""
    org = m.make("Goldman Sachs", entity_type="organisation",
                 url="https://www.goldmansachs.com", provenance="user seed list",
                 lifecycle_state="verified", link_status="ok")
    ents = [org, m.make("Show A", entity_type="podcast", provenance="seed")]
    out, _rels, report = promote.build(ents, [], _thrice("Goldman Sachs"))
    gs = next(e for e in out if e["name"] == "Goldman Sachs")
    assert gs["entity_type"] == "organisation"
    assert gs["url"] == "https://www.goldmansachs.com"
    assert gs["provenance"] == "user seed list"
    assert report.get("skipped_type_conflict") == 1
    assert sum(1 for e in out if e["name"] == "Goldman Sachs") == 1


def test_prior_db_also_refuses_to_retype_an_organisation():
    org = m.make("Winton", entity_type="organisation", provenance="user seed list")
    recs = [{"name": "Winton Capital", "firm": "", "role": "", "section": "A1",
             "both_podcasts": False, "status": "",
             "source_line": "Winton - x."}]
    recs[0]["name"] = "Winton"
    out, _rels, report = prior_db.build([org], [], recs)
    assert next(e for e in out if e["name"] == "Winton")["entity_type"] == "organisation"
    assert report.get("skipped_type_conflict") == 1


def test_merge_gives_a_person_their_first_affiliation():
    """REGRESSION: `current_affiliation` is excluded from merge()'s general
    field loop so its history handling can run first. Handling only the CHANGE
    case meant a person recorded with no employer could never gain one, and
    every later source naming their firm was silently discarded."""
    person = m.make("Jane Doe", entity_type="person")
    assert person["current_affiliation"] == ""
    merged = m.merge(person, {"current_affiliation": "Alpha Capital"})
    assert merged["current_affiliation"] == "Alpha Capital"
    assert merged["historical_affiliations"] == []


# ---------------------------------------------------------- prior compilation --
def test_prior_db_parses_the_supplied_compilation():
    recs = prior_db.parse()
    if not recs:
        pytest.skip("prior compilation file not present")
    assert len(recs) > 150
    by_name = {r["name"]: r for r in recs}
    assert by_name["Jerry Parker"]["firm"] == "Chesapeake Capital"
    assert by_name["Harold de Boer"]["firm"] == "Transtrend"
    assert "Head of R&D" in by_name["Harold de Boer"]["role"]
    assert any(r["both_podcasts"] for r in recs)
    for r in recs:
        assert guests.looks_like_person(r["name"]), r["name"]


def test_prior_db_only_fills_gaps():
    """The feeds are the authority: an affiliation already recorded is left
    alone, so applying the compilation first and the harvest second makes the
    harvest win without either module knowing about the other."""
    existing = m.make("Jerry Parker", entity_type="person",
                      current_affiliation="Somewhere Else", provenance="harvest")
    recs = [{"name": "Jerry Parker", "firm": "Chesapeake Capital", "role": "founder",
             "section": "A1", "both_podcasts": False, "status": "",
             "source_line": "Jerry Parker - Chesapeake Capital (founder)."}]
    out, _rels, report = prior_db.build([existing], [], recs)
    jerry = next(e for e in out if e["name"] == "Jerry Parker")
    assert jerry["current_affiliation"] == "Somewhere Else"
    assert report["enriched"] == 1


def test_prior_db_records_its_own_caveats():
    recs = [{"name": "Alex Greyserman", "firm": "", "role": "", "section": "A1",
             "both_podcasts": True, "status": "current firm unverified",
             "source_line": "Alex Greyserman - research."}]
    out, _rels, _r = prior_db.build([], [], recs)
    ent = out[0]
    assert ent["confidence"] == "low"
    assert ent["lifecycle_state"] == "needs_review"
    assert "current firm unverified" in ent["notes"]
    assert "BOTH podcasts" in ent["notes"]
    assert prior_db.PROVENANCE in ent["provenance"]


# ------------------------------------------------------------ live artefacts --
@pytest.mark.skipif(not config.INDEX_FILES["podcasts"].exists(),
                    reason="no committed podcast harvest")
def test_committed_harvest_is_coherent():
    from zenith.index import load as index_load
    doc = index_load("podcasts", {})
    eps = index_load("episodes", [])
    assert len(doc.get("shows", [])) == 14
    assert len(eps) > 5000
    assert len({e["id"] for e in eps}) == len(eps), "episode ids must be unique"
    for g in doc.get("guests", {}).values():
        if guests.meets_threshold(g.get("confidence", "low")):
            assert guests.looks_like_person(g["name"]), g["name"]


@pytest.mark.skipif(not config.INDEX_FILES["podcasts"].exists(),
                    reason="no committed podcast harvest")
def test_committed_catalog_links_guests_to_shows():
    from zenith.index import load as index_load
    ents = index_load("entities", [])
    rels = index_load("relationships", [])
    by_id = {e["id"]: e for e in ents}
    appeared = [r for r in rels if r["type"] == "appeared_on"]
    assert len(appeared) > 500
    for r in appeared[:200]:
        assert by_id[r["source"]]["entity_type"] == "person"
        assert by_id[r["target"]]["entity_type"] == "podcast"
    assert quality.orphan_edges(ents, rels) == []

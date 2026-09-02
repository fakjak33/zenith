"""INDEX Phase 3 tests — knowledge graph, ranking and the discovery surface.

Offline and synthetic. The live-artefact checks at the bottom read only
committed JSON.

The ranking tests are deliberately strict about the *honesty* properties rather
than about particular numbers: that contributions sum exactly to the score, that
the published component set matches what is actually computed, and that a raw
count can never dominate the scale. Those are the promises the module makes to
the reader, so they are the ones worth pinning.
"""

from __future__ import annotations

import pytest

from zenith import config
from zenith.index import (discover, model as m, network as nw, quality,
                          ranking as rk, taxonomy as tx)


# --------------------------------------------------------------- fixtures --
def _graph():
    """A small catalog with a known shape:

        Alice --works_at--> AcmeCap
        Alice --appeared_on--> ShowA
        Bob   --appeared_on--> ShowA
        Bob   --appeared_on--> ShowB
        Carol --works_at--> AcmeCap        (isolated from the shows)
        Dave  (no edges at all)
    """
    ents = [
        m.make("Alice Smith", entity_type="person", provenance="seed"),
        m.make("Bob Jones", entity_type="person", provenance="seed"),
        m.make("Carol White", entity_type="person", provenance="seed"),
        m.make("Dave Brown", entity_type="person", provenance="seed"),
        m.make("AcmeCap", entity_type="organisation", provenance="seed"),
        m.make("Show A", entity_type="podcast", provenance="seed"),
        m.make("Show B", entity_type="podcast", provenance="seed"),
    ]
    by = {e["name"]: e["id"] for e in ents}
    rels = [
        m.edge(by["Alice Smith"], by["AcmeCap"], "works_at"),
        m.edge(by["Alice Smith"], by["Show A"], "appeared_on"),
        m.edge(by["Bob Jones"], by["Show A"], "appeared_on"),
        m.edge(by["Bob Jones"], by["Show B"], "appeared_on"),
        m.edge(by["Carol White"], by["AcmeCap"], "works_at"),
    ]
    return ents, rels, by


# ----------------------------------------------------------------- network --
def test_degrees_counts_both_edge_directions():
    ents, rels, by = _graph()
    deg = nw.degrees(ents, rels)
    assert deg[by["Alice Smith"]] == 2
    assert deg[by["AcmeCap"]] == 2
    assert deg[by["Show A"]] == 2
    assert deg[by["Dave Brown"]] == 0


def test_ego_ids_respects_hops():
    ents, rels, by = _graph()
    one = set(nw.ego_ids(ents, rels, by["Alice Smith"], hops=1))
    assert one == {by["Alice Smith"], by["AcmeCap"], by["Show A"]}
    two = set(nw.ego_ids(ents, rels, by["Alice Smith"], hops=2))
    assert by["Bob Jones"] in two and by["Carol White"] in two
    assert by["Dave Brown"] not in two, "an unconnected node is never reachable"


def test_ego_ids_is_bounded_and_deterministic():
    ents, rels, by = _graph()
    a = nw.ego_ids(ents, rels, by["Alice Smith"], hops=3, limit=3)
    b = nw.ego_ids(ents, rels, by["Alice Smith"], hops=3, limit=3)
    assert len(a) <= 3
    assert a == b, "the same focus must yield the same subgraph across runs"


def test_build_lays_out_nodes_and_edges():
    ents, rels, by = _graph()
    g = nw.build(ents, rels, [e["id"] for e in ents])
    assert g is not None
    # Six, not seven: Dave Brown has no edges and is dropped (see below).
    assert g["n"] == 6
    assert len(g["edges"]) == 5
    for node in g["nodes"]:
        assert isinstance(node["x"], float) and isinstance(node["y"], float)
    # Two rows per edge, sharing an id, so a line mark can join them.
    segs = nw.edge_segments(g)
    assert len(segs) == 2 * len(g["edges"])
    assert len({s["edge_id"] for s in segs}) == len(g["edges"])


def test_build_drops_nodes_isolated_within_the_subgraph():
    """REGRESSION: a node with no edge INSIDE the selected subgraph is pushed
    away by every other node and pulled back by none, so it flies to the edge of
    the canvas and squeezes the connected core into a corner. The "most
    connected" view collapsed into an unreadable clump because of exactly this."""
    ents, rels, by = _graph()
    g = nw.build(ents, rels, [e["id"] for e in ents])
    assert by["Dave Brown"] not in {n["id"] for n in g["nodes"]}
    # Alice and Carol share AcmeCap, so a subgraph of just those three holds up.
    trio = nw.build(ents, rels, [by["Alice Smith"], by["Carol White"], by["AcmeCap"]])
    assert trio["n"] == 3


def test_build_rescales_coordinates_to_a_fixed_box():
    """Without rescaling, a 12-node graph and a 90-node graph render at wildly
    different zooms and the chart's axis range jumps between views."""
    ents, rels, _by = _graph()
    g = nw.build(ents, rels, [e["id"] for e in ents])
    xs = [n["x"] for n in g["nodes"]]
    ys = [n["y"] for n in g["nodes"]]
    assert min(xs) == pytest.approx(-1.0) and max(xs) == pytest.approx(1.0)
    assert min(ys) == pytest.approx(-1.0) and max(ys) == pytest.approx(1.0)


def test_build_is_deterministic():
    ents, rels, _by = _graph()
    ids = [e["id"] for e in ents]
    a = nw.build(ents, rels, ids)
    b = nw.build(ents, rels, ids)
    assert [n["x"] for n in a["nodes"]] == [n["x"] for n in b["nodes"]]


def test_build_returns_none_when_there_is_nothing_to_draw():
    ents, rels, by = _graph()
    assert nw.build(ents, rels, []) is None
    assert nw.build(ents, rels, [by["Dave Brown"]]) is None
    # Two nodes with no edge between them is not a graph.
    assert nw.build(ents, rels, [by["Dave Brown"], by["Carol White"]]) is None


def test_build_never_exceeds_the_node_limit():
    ents = [m.make(f"Person {i:03d}", entity_type="person", provenance="s")
            for i in range(400)]
    rels = [m.edge(ents[i]["id"], ents[i + 1]["id"], "related_to")
            for i in range(len(ents) - 1)]
    g = nw.build(ents, rels, [e["id"] for e in ents])
    assert g["n"] <= nw.NODE_LIMIT
    assert g["truncated"]


def test_path_between_finds_the_shortest_chain():
    ents, rels, by = _graph()
    path = nw.path_between(ents, rels, by["Alice Smith"], by["Bob Jones"])
    assert path == [by["Alice Smith"], by["Show A"], by["Bob Jones"]]
    hops = nw.describe_path(ents, rels, path)
    assert [h["type"] for h in hops] == ["appeared on", "appeared on"]
    assert hops[0]["from"] == "Alice Smith" and hops[-1]["to"] == "Bob Jones"


def test_path_between_returns_empty_when_disconnected():
    ents, rels, by = _graph()
    assert nw.path_between(ents, rels, by["Alice Smith"], by["Dave Brown"]) == []
    assert nw.path_between(ents, rels, by["Alice Smith"], by["Alice Smith"]) == \
        [by["Alice Smith"]]


def test_top_connected_excludes_isolated_nodes():
    ents, rels, by = _graph()
    ids = nw.top_connected_ids(ents, rels, limit=10)
    assert by["Dave Brown"] not in ids
    assert nw.top_connected_ids(ents, rels, limit=10, entity_types=("podcast",)) == \
        [by["Show A"], by["Show B"]]


# ----------------------------------------------------------------- ranking --
def test_contributions_sum_exactly_to_the_score():
    """The transparency promise: nothing hides in a residual."""
    ents, rels, _by = _graph()
    scored = rk.score_all(ents, rels)
    for row in scored.values():
        assert abs(sum(row["components"].values()) - row["score"]) < 1e-9


def test_published_components_match_what_is_computed():
    """METHODOLOGY and the UI both describe COMPONENTS; if the computation grew
    a term that was not declared there, the documentation would be a lie."""
    ents, rels, _by = _graph()
    scored = rk.score_all(ents, rels)
    any_row = next(iter(scored.values()))
    assert set(any_row["components"]) == set(rk.COMPONENTS)
    assert abs(sum(w for w, _l, _d in rk.COMPONENTS.values()) - 1.0) < 1e-9


def test_weights_are_round_numbers():
    """Stated as an editorial choice, not a fitted parameter — so they must not
    drift into implying precision that does not exist."""
    for weight, _label, _desc in rk.COMPONENTS.values():
        assert round(weight, 2) == weight, weight


def test_more_connections_scores_higher_all_else_equal():
    ents, rels, by = _graph()
    scored = rk.score_all(ents, rels)
    assert scored[by["Alice Smith"]]["score"] > scored[by["Dave Brown"]]["score"]


def test_log_compression_stops_one_count_dominating():
    """A guest with 1,000 appearances must not be 1,000x a guest with one."""
    ents = [m.make("Prolific", entity_type="person", provenance="s"),
            m.make("Occasional", entity_type="person", provenance="s")]
    guest_index = {"prolific": {"n_appearances": 1000, "n_podcasts": 1},
                   "occasional": {"n_appearances": 1, "n_podcasts": 1}}
    scored = rk.score_all(ents, [], guest_index)
    hi = scored[ents[0]["id"]]["components"]["appearances"]
    lo = scored[ents[1]["id"]]["components"]["appearances"]
    assert hi > lo
    assert hi < lo * 15, "log1p compression should keep the ratio far below 1000x"


def test_blocked_link_scores_between_broken_and_verified():
    """A robot-blocked host demonstrably exists; scoring it as zero would
    penalise Citadel and SSRN for having anti-bot rules."""
    good = m.make("Good", url="https://a.test", lifecycle_state="verified",
                  link_status="ok", provenance="s")
    blocked = m.make("Blocked", url="https://b.test", lifecycle_state="new",
                     link_status="blocked", provenance="s")
    broken = m.make("Broken", url="https://c.test", lifecycle_state="needs_review",
                    link_status="error", provenance="s")
    assert rk._verification(good) > rk._verification(blocked) > rk._verification(broken)
    assert rk._verification(broken) == 0.0


def test_guest_profile_is_matched_by_alias():
    """REGRESSION: the catalog says "Robert Carver" and the podcast titles say
    "Rob Carver". Matching on the primary name alone under-counted exactly the
    best-connected people."""
    ent = m.make("Robert Carver", entity_type="person", aliases=["Rob Carver"],
                 provenance="s")
    scored = rk.score_all([ent], [], {"rob carver": {"n_appearances": 39,
                                                     "n_podcasts": 1}})
    assert scored[ent["id"]]["raw"]["appearances"] == 39


def test_explain_returns_every_component_as_strings():
    ents, rels, by = _graph()
    scored = rk.score_all(ents, rels)
    rows = rk.explain(ents[0], scored)
    assert len(rows) == len(rk.COMPONENTS)
    assert rows == sorted(rows, key=lambda r: -r["contribution"])
    # Mixed int/str in one column forces an Arrow coercion on every render.
    assert all(isinstance(r["measured"], str) for r in rows)


def test_methodology_states_what_the_score_is_not():
    text = rk.METHODOLOGY.lower()
    assert "not how good" in text or "not a quality" in text
    assert "bias" in text, "known biases must be stated, not hidden"


# ---------------------------------------------------------------- discover --
def _provenanced():
    ents, rels, by = _graph()
    # Alice is on the user's own list; Bob was discovered by the harvest.
    for e in ents:
        if e["name"] == "Alice Smith":
            e["provenance"] = "user seed list 2026-09-01"
        elif e["name"] == "Bob Jones":
            e["provenance"] = "podcast archive harvest"
        elif e["name"] == "Carol White":
            e["provenance"] = "podcast archive harvest"
            e["zenith_source"] = "Carol (already scraped)"
    return ents, rels, by


def test_provenance_predicates():
    ents, _rels, _by = _provenanced()
    alice = next(e for e in ents if e["name"] == "Alice Smith")
    bob = next(e for e in ents if e["name"] == "Bob Jones")
    assert discover.is_curated(alice) and not discover.is_discovered(alice)
    assert discover.is_discovered(bob) and not discover.is_curated(bob)


def test_new_to_you_excludes_curated_and_already_ingested():
    ents, rels, _by = _provenanced()
    guest_index = {"alice smith": {"n_appearances": 9, "n_podcasts": 3},
                   "bob jones": {"n_appearances": 9, "n_podcasts": 3},
                   "carol white": {"n_appearances": 9, "n_podcasts": 3}}
    scored = rk.score_all(ents, rels, guest_index)
    names = [e["name"] for e in discover.new_to_you(ents, scored)]
    assert names == ["Bob Jones"], names


def test_cross_pollinators_needs_several_distinct_shows():
    ents, rels, _by = _provenanced()
    scored = rk.score_all(ents, rels, {
        "bob jones": {"n_appearances": 4, "n_podcasts": 4},
        "alice smith": {"n_appearances": 40, "n_podcasts": 1},
    })
    names = [e["name"] for e in discover.cross_pollinators(ents, scored, min_shows=3)]
    assert "Bob Jones" in names
    assert "Alice Smith" not in names, "40 appearances on ONE show is not reach"


def test_bridges_require_spanning_both_firms_and_shows():
    ents, rels, by = _graph()
    rows = {r["entity"]["name"]: r for r in discover.bridges(ents, relationships=rels)}
    # Alice touches AcmeCap AND Show A but only 2 targets — below the threshold.
    assert not rows
    rels = rels + [m.edge(by["Alice Smith"], by["Show B"], "appeared_on")]
    rows = {r["entity"]["name"]: r for r in discover.bridges(ents, rels)}
    assert "Alice Smith" in rows and rows["Alice Smith"]["spans"] == 3
    # Bob touches two podcasts and no firm — frequent guest, not a connector.
    assert "Bob Jones" not in rows


def test_emerging_uses_first_appearance_not_latest():
    from datetime import date
    ents = [m.make("New Voice", entity_type="person", provenance="harvest"),
            m.make("Old Hand", entity_type="person", provenance="harvest")]
    gi = {"new voice": {"appearances": [{"published": "2026-08-01"}]},
          "old hand": {"appearances": [{"published": "2019-01-01"},
                                       {"published": "2026-08-30"}]}}
    rows = discover.emerging(ents, gi, within_days=180, today=date(2026, 9, 2))
    assert [r["entity"]["name"] for r in rows] == ["New Voice"], \
        "a long-standing guest who was on last week is not a new voice"


def test_by_topic_filters_on_the_resolved_tag():
    ents, rels, _by = _graph()
    ents[0]["investment_approach"] = ["trend_following"]
    scored = rk.score_all(ents, rels)
    rows = discover.by_topic(ents, scored, "investment_approach", "Trend Following")
    assert [e["name"] for e in rows] == ["Alice Smith"]


def test_coverage_gaps_report_weak_shows_and_thin_tags():
    ents, _rels, _by = _graph()
    pod = {"shows": [{"podcast": "Quiet Show", "coverage": 0.12, "episodes": 900},
                     {"podcast": "Good Show", "coverage": 0.91, "episodes": 100}]}
    gaps = discover.coverage_gaps(ents, pod)
    subjects = [g["subject"] for g in gaps]
    assert "Quiet Show" in subjects
    assert "Good Show" not in subjects
    assert any(g["kind"] == "thin taxonomy" for g in gaps)
    # The gap text must not assert a cause that was never verified per-show.
    quiet = next(g for g in gaps if g["subject"] == "Quiet Show")
    assert "editorial" not in quiet["detail"]


def test_build_returns_every_panel():
    ents, rels, _by = _provenanced()
    out = discover.build(ents, rels, {}, {"shows": []})
    for key in ("scored", "new_to_you", "cross_pollinators", "bridges",
                "discovered_firms", "emerging", "gaps", "degrees"):
        assert key in out


# ------------------------------------------------------------ live artefacts --
@pytest.mark.skipif(not config.INDEX_FILES["entities"].exists(),
                    reason="no committed catalog")
def test_live_graph_is_traversable():
    from zenith.index import load
    ents, rels = load("entities", []), load("relationships", [])
    by_name = {e["name"]: e for e in ents}
    if "Robert Carver" not in by_name or "Man Group" not in by_name:
        pytest.skip("seed entities missing")
    path = nw.path_between(ents, rels, by_name["Robert Carver"]["id"],
                           by_name["Man Group"]["id"])
    names = [by_name_id(ents, i) for i in path]
    assert names[0] == "Robert Carver" and names[-1] == "Man Group"
    assert "Man AHL" in names, "the chain should route through his old employer"
    assert quality.orphan_edges(ents, rels) == []


def by_name_id(entities, eid):
    return next((e["name"] for e in entities if e["id"] == eid), eid)


@pytest.mark.skipif(not config.INDEX_FILES["entities"].exists(),
                    reason="no committed catalog")
def test_live_ranking_is_coherent():
    from zenith.index import load
    ents, rels = load("entities", []), load("relationships", [])
    gi = (load("podcasts", {}) or {}).get("guests", {})
    scored = rk.score_all(ents, rels, gi)
    assert len(scored) == len(ents)
    ranks = sorted(r["rank"] for r in scored.values())
    assert ranks == list(range(1, len(ents) + 1)), "ranks must be a dense 1..N"
    for row in scored.values():
        assert 0.0 <= row["score"] <= 1.0
        assert abs(sum(row["components"].values()) - row["score"]) < 1e-9

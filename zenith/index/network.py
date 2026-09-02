"""INDEX Phase 3 — the knowledge graph as a picture.

A 2,149-node, 2,732-edge graph drawn all at once is a hairball that tells you
nothing, so this module never tries. Every view it produces is a BOUNDED
subgraph built for a specific question:

  ego(entity, hops)    what is this thing connected to?
  by_type(...)         how do the podcasts and their guests relate?
  top_connected(...)   what sits at the centre of the collected data?

LAYOUT IS REUSED, NOT REBUILT. ``mom/mvt/network.py`` already contains a
dependency-free Fruchterman-Reingold spring layout in plain NumPy, written for
exactly this problem (dozens of nodes, deterministic, no networkx). Importing it
keeps one implementation of the physics rather than two that can drift; the
alternative — copying forty lines of numerical code — would be worse.

WHAT AN EDGE MEANS HERE. Every edge is a stated fact with a provenance, not an
inferred similarity: a person hosts or appeared on a podcast, a brand is a
division of a parent, a person works or worked at a firm. Distance in the
picture is a layout artefact of the force simulation and carries no meaning
beyond "these are connected"; nothing here is a correlation or a score.
"""

from __future__ import annotations

import numpy as np

from ..mom.mvt.network import _force_layout

# Edge types drawn, and the spring weight each gets. Weight only affects LAYOUT
# (how tightly two nodes cluster), never interpretation: an employment link
# pulls harder than a loose topical one so the picture groups firms with their
# people, which is the reading that makes a directory graph legible.
EDGE_WEIGHTS: dict[str, float] = {
    "works_at": 1.0,
    "founded": 1.0,
    "subsidiary_of": 1.0,
    "hosts": 0.9,
    "publishes": 0.8,
    "worked_at": 0.6,
    "appeared_on": 0.5,
    "related_to": 0.4,
}

NODE_LIMIT = 120          # beyond this a force layout is an unreadable hairball


def _adjacency(entities: list[dict], relationships: list[dict]) -> dict[str, set[str]]:
    ids = {e["id"] for e in entities}
    adj: dict[str, set[str]] = {i: set() for i in ids}
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        if s in ids and t in ids and s != t:
            adj[s].add(t)
            adj[t].add(s)
    return adj


def degrees(entities: list[dict], relationships: list[dict]) -> dict[str, int]:
    """Edge count per entity — the graph's own measure of connectedness."""
    return {k: len(v) for k, v in _adjacency(entities, relationships).items()}


def ego_ids(entities: list[dict], relationships: list[dict], focus_id: str,
            hops: int = 1, limit: int = NODE_LIMIT) -> list[str]:
    """Ids within ``hops`` of ``focus_id``, breadth-first, bounded by ``limit``.

    Breadth-first rather than depth-first so a truncated result is still a
    complete picture of the near neighbourhood, instead of one arbitrary deep
    branch. Neighbours are visited in a stable order so the same focus always
    yields the same subgraph.
    """
    adj = _adjacency(entities, relationships)
    if focus_id not in adj:
        return []
    seen = {focus_id}
    frontier = [focus_id]
    for _ in range(max(1, hops)):
        nxt: list[str] = []
        for node in frontier:
            for nb in sorted(adj.get(node, ())):
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
                    if len(seen) >= limit:
                        return list(seen)
        frontier = nxt
        if not frontier:
            break
    return list(seen)


def top_connected_ids(entities: list[dict], relationships: list[dict],
                      limit: int = 60, entity_types: tuple[str, ...] | None = None
                      ) -> list[str]:
    """The most-connected entities, optionally restricted to certain types."""
    deg = degrees(entities, relationships)
    pool = [e for e in entities
            if not entity_types or e.get("entity_type") in entity_types]
    pool.sort(key=lambda e: (-deg.get(e["id"], 0), e.get("name", "").lower()))
    return [e["id"] for e in pool[:limit] if deg.get(e["id"], 0) > 0]


def build(entities: list[dict], relationships: list[dict], node_ids: list[str],
          seed: int = 0) -> dict | None:
    """Lay out the subgraph induced by ``node_ids``.

    Returns {nodes, edges, n, truncated} or None when there is nothing to draw.
    Node positions come from the shared spring layout; edges are returned as
    two rows each (source point, target point) so Altair can draw them as
    line segments with a shared ``edge_id``.
    """
    ids = list(dict.fromkeys(node_ids))[:NODE_LIMIT]
    if len(ids) < 2:
        return None
    by_id = {e["id"]: e for e in entities if e["id"] in set(ids)}
    ids = [i for i in ids if i in by_id]
    if len(ids) < 2:
        return None

    # Drop nodes that have no edge WITHIN this subgraph. They are well-connected
    # elsewhere (that is often why they were selected) but here they are
    # isolated, and in a force layout an isolated node is pushed away by
    # everything and pulled back by nothing — so it flies to the edge of the
    # canvas and squeezes the entire connected core into a corner. Removing
    # them is what makes the picture legible.
    inside = set(ids)
    linked: set[str] = set()
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        if s in inside and t in inside and s != t:
            linked.add(s)
            linked.add(t)
    ids = [i for i in ids if i in linked]
    if len(ids) < 2:
        return None
    pos_of = {eid: k for k, eid in enumerate(ids)}

    weights = np.zeros((len(ids), len(ids)), dtype=float)
    edges: list[dict] = []
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        if s not in pos_of or t not in pos_of or s == t:
            continue
        w = EDGE_WEIGHTS.get(str(r.get("type")), 0.4)
        i, j = pos_of[s], pos_of[t]
        weights[i, j] = max(weights[i, j], w)
        weights[j, i] = weights[i, j]
        edges.append({"source": s, "target": t, "type": str(r.get("type") or ""),
                      "weight": w})
    if not edges:
        return None

    coords = _force_layout(weights, iterations=160, seed=seed)
    # Rescale to a fixed box. The simulation's absolute scale depends on node
    # count, so without this a 12-node graph and a 90-node graph render at
    # wildly different zooms and the chart's axis range jumps between views.
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span < 1e-9] = 1.0
    coords = (coords - coords.min(axis=0)) / span * 2.0 - 1.0

    deg = {eid: int((weights[pos_of[eid]] > 0).sum()) for eid in ids}
    nodes = []
    for eid in ids:
        ent = by_id[eid]
        k = pos_of[eid]
        nodes.append({
            "id": eid,
            "name": ent.get("name", ""),
            "entity_type": ent.get("entity_type", ""),
            "primary_category": ent.get("primary_category", ""),
            "degree": deg[eid],
            "x": float(coords[k, 0]),
            "y": float(coords[k, 1]),
        })

    name_of = {n["id"]: n["name"] for n in nodes}
    for e in edges:
        e["source_name"] = name_of.get(e["source"], "")
        e["target_name"] = name_of.get(e["target"], "")
    return {"nodes": nodes, "edges": edges, "n": len(nodes),
            "truncated": len(node_ids) > len(ids)}


def edge_segments(graph: dict) -> list[dict]:
    """Edges as plottable point pairs — two rows per edge, sharing an id."""
    pos = {n["id"]: (n["x"], n["y"]) for n in graph["nodes"]}
    rows: list[dict] = []
    for i, e in enumerate(graph["edges"]):
        for end in ("source", "target"):
            x, y = pos[e[end]]
            rows.append({"edge_id": i, "x": x, "y": y, "type": e["type"],
                         "weight": e["weight"],
                         "label": f"{e['source_name']} → {e['target_name']}"})
    return rows


def path_between(entities: list[dict], relationships: list[dict],
                 a_id: str, b_id: str, max_hops: int = 6) -> list[str]:
    """Shortest path between two entities, as a list of ids (empty if none).

    This is what makes the graph answer a question rather than just look like
    one: "how is this researcher connected to that firm?" is a real query, and
    the answer is a chain of stated facts.
    """
    if a_id == b_id:
        return [a_id]
    adj = _adjacency(entities, relationships)
    if a_id not in adj or b_id not in adj:
        return []
    prev: dict[str, str] = {a_id: ""}
    frontier = [a_id]
    for _ in range(max_hops):
        nxt: list[str] = []
        for node in frontier:
            for nb in sorted(adj.get(node, ())):
                if nb in prev:
                    continue
                prev[nb] = node
                if nb == b_id:
                    path = [nb]
                    while prev[path[-1]]:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return []


def describe_path(entities: list[dict], relationships: list[dict],
                  path: list[str]) -> list[dict]:
    """Turn a path of ids into readable hops with the edge type on each."""
    by_id = {e["id"]: e for e in entities}
    typed: dict[tuple[str, str], str] = {}
    for r in relationships:
        s, t, ty = r.get("source"), r.get("target"), str(r.get("type") or "")
        typed.setdefault((s, t), ty)
        typed.setdefault((t, s), ty)
    out = []
    for a, b in zip(path, path[1:]):
        out.append({
            "from": by_id.get(a, {}).get("name", a),
            "to": by_id.get(b, {}).get("name", b),
            "type": typed.get((a, b), "related_to").replace("_", " "),
        })
    return out

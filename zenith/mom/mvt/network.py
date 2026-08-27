"""Phase 3 — relative-strength network (spec section 22): each instrument is
a node, its strongest pairwise relationships are edges, positioned by a
force-directed layout so tightly-related names cluster visually.

Computed LIVE, on demand, for a user-chosen (bounded) subset -- never
precomputed/committed, the same "reconstruct from the committed vectors,
don't persist a big derived artifact" pattern Phase 1's interactive pairwise
matrix already established (see pairwise.py's storage-architecture note).
A full ~1000-node graph would be unreadable and is explicitly optional per
the spec ("This should be optional if computationally expensive") -- this
module is only ever asked to lay out a few dozen nodes.

No new dependency: the layout is a small, dependency-free Fruchterman-
Reingold-style force simulation in plain NumPy (this repo already avoids
scipy; a full graph library like networkx would be a much heavier addition
than a ~40-line spring layout needs to justify).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import pairwise as pw


def _reconstruct_matrix(df, tickers: list[str], horizon: str, layer: str):
    """Same reconstruction pattern as mvt/view.py's _reconstruct_submatrix,
    duplicated here (rather than imported from view.py, which is a
    Streamlit-rendering module this one shouldn't depend on) so network.py
    stays independently testable without pulling in Streamlit."""
    sub = df[df["ticker"].isin(tickers)].drop_duplicates(subset="ticker").set_index("ticker").reindex(tickers)
    key = "total_return" if layer == "raw" else "residual_return"
    var_key = "total_var" if layer == "raw" else "resid_var"
    horizon_days = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252, "12_1": 231}[horizon]
    r = sub[key].apply(lambda d: (d or {}).get(horizon))
    var = sub[var_key]
    valid = r.notna() & var.notna()
    if valid.sum() < 2:
        return None, []
    order = [t for t in tickers if bool(valid.get(t, False))]
    r_arr = r.reindex(order).to_numpy(dtype=float)
    var_arr = var.reindex(order).to_numpy(dtype=float)
    D = pw.spread_matrix(r_arr, var_arr, horizon_days)
    return D, order


def _force_layout(edge_weights: np.ndarray, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """Fruchterman-Reingold-style spring layout: every pair repels (so
    unrelated nodes spread out), edges attract proportional to their
    weight (so strongly-related nodes cluster). Returns (N, 2) positions,
    roughly centered at the origin. Deterministic (fixed seed) so the same
    subset lays out the same way across reruns rather than jittering."""
    n = edge_weights.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n, 2))
    if n < 2:
        return pos
    k = 1.0 / np.sqrt(n)          # ideal spring length, standard FR heuristic
    for it in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]           # (n, n, 2)
        dist = np.sqrt((delta ** 2).sum(axis=-1)) + 1e-9    # (n, n)
        # repulsive force between every pair (Coulomb-like, ~1/dist)
        repulse = (k ** 2) / dist
        # attractive force along weighted edges (Hooke-like, ~dist)
        attract = (dist ** 2) / k * edge_weights
        disp_mag = repulse - attract
        disp = (delta / dist[..., None]) * disp_mag[..., None]
        step = disp.sum(axis=1)
        step_norm = np.sqrt((step ** 2).sum(axis=1, keepdims=True)) + 1e-9
        temp = max(0.05, 1.0 - it / iterations)   # cooling schedule
        pos += (step / step_norm) * np.minimum(step_norm, temp)
    pos -= pos.mean(axis=0)
    return pos


def build_network(df, tickers: list[str], horizon: str = "6m", layer: str = "residual",
                  edges_per_node: int = 3, iterations: int = 150) -> dict | None:
    """Nodes + edges + a 2D layout for the given (bounded) ticker subset.
    Each node keeps its top `edges_per_node` strongest ABSOLUTE relationships
    as edges (sparse by construction, so the graph stays readable even at
    30-50 nodes) -- both positive (co-moving) and negative (diverging)
    relationships are kept, since section 22 asks for clusters AND
    idiosyncratic movers, and a negative-only or positive-only graph would
    hide one of those."""
    D, order = _reconstruct_matrix(df, tickers, horizon, layer)
    if D is None or len(order) < 3:
        return None
    n = len(order)

    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        row = D[i].copy()
        row[i] = 0.0
        top_idx = np.argsort(-np.abs(row))[:edges_per_node]
        for j in top_idx:
            if row[j] == 0:
                continue
            edge_set.add((min(i, int(j)), max(i, int(j))))

    weight_mat = np.zeros((n, n))
    for i, j in edge_set:
        weight_mat[i, j] = weight_mat[j, i] = abs(D[i, j])
    max_w = weight_mat.max() if weight_mat.max() > 0 else 1.0
    weight_norm = weight_mat / max_w

    pos = _force_layout(weight_norm, iterations=iterations)

    scores = df.set_index("ticker").reindex(order)["normalized_score"]
    nodes = [{"ticker": order[i], "x": round(float(pos[i, 0]), 4), "y": round(float(pos[i, 1]), 4),
             "score": None if pd.isna(scores.iloc[i]) else round(float(scores.iloc[i]), 3)}
            for i in range(n)]
    edges = [{"source": order[i], "target": order[j], "weight": round(float(D[i, j]), 4)}
            for i, j in sorted(edge_set)]
    return {"nodes": nodes, "edges": edges, "horizon": horizon, "layer": layer, "n": n}

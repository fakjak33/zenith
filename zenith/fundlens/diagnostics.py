"""'Well-rounded' diagnostics: breadth, weakest pillar, weighting sensitivity, robust leaders."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .scoring import ScoreResult, composite, rank_order


@dataclass
class Diagnostics:
    breadth: pd.Series            # pillars at/above threshold
    n_pillars: int
    weakest: pd.Series            # label of the weakest pillar
    strongest: pd.Series
    scheme_ranks: pd.DataFrame    # rank under each alternative weighting
    rank_best: pd.Series
    rank_worst: pd.Series
    robust: pd.Series             # bool: top quintile under every scheme
    schemes: dict[str, dict[str, float]]


def weight_schemes(pw: dict[str, float], labels: dict[str, str]) -> dict[str, dict[str, float]]:
    active = [k for k, w in pw.items() if w > 0]
    out = {"Profile weights": dict(pw), "Equal pillars": {k: 1.0 for k in active}}
    for k in active:
        out[f"{labels[k]} x2"] = {j: (pw[j] * 2 if j == k else pw[j]) for j in active}
    return out


def diagnose(df: pd.DataFrame, s: ScoreResult, prof: dict) -> Diagnostics:
    thr = float(prof["scoring"]["breadth_threshold"])
    k = float(prof["scoring"]["shrinkage_k"])
    ps = s.pillar_scores[[p for p in s.pillar_weights if p in s.pillar_scores]]
    labels = s.pillar_labels
    breadth = (ps >= thr).sum(axis=1).where(ps.notna().any(axis=1))

    def lab(row, fn):
        r = row.dropna()
        if r.empty:
            return ""
        # deterministic: ties resolved by pillar order
        target = fn(r.values)
        return labels[next(c for c in r.index if r[c] == target)]

    weakest = ps.apply(lambda r: lab(r, min), axis=1) if len(ps) else pd.Series(dtype=str)
    strongest = ps.apply(lambda r: lab(r, max), axis=1) if len(ps) else pd.Series(dtype=str)

    schemes = weight_schemes(s.pillar_weights, labels)
    ranks = {}
    for name, w in schemes.items():
        tot = sum(w.values())
        w = {p: v / tot for p, v in w.items()}
        raw, _, adj = composite(s.pillar_scores, s.pillar_cov, w, k, s.optional)
        _, r = rank_order(df, adj, raw)
        ranks[name] = r
    sr = pd.DataFrame(ranks, index=df.index)
    n = int(s.rank.notna().sum())
    cut = max(1, math.ceil(n * 0.2)) if n else 0
    best, worst = sr.min(axis=1), sr.max(axis=1)
    robust = (worst <= cut) & s.rank.notna()
    return Diagnostics(breadth, ps.shape[1], weakest, strongest, sr, best, worst, robust, schemes)

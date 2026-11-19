"""Plain-data snapshot of a Comparison (for golden tests, the Claude skill and the app)."""
from __future__ import annotations

import math

import pandas as pd

from .engine import Comparison


def _r(v, n=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), n)


def results_dict(c: Comparison) -> dict:
    s, dg = c.scores, c.diag
    funds = []
    for t in s.order:
        funds.append({
            "rank": None if pd.isna(s.rank[t]) else int(s.rank[t]), "ticker": t, "name": c.df.at[t, "name"],
            "tier": s.tier[t], "composite": _r(s.adjusted[t]), "raw": _r(s.raw[t]), "coverage": _r(s.coverage[t]),
            "breadth": None if pd.isna(dg.breadth[t]) else int(dg.breadth[t]), "weakest": dg.weakest[t] or None,
            "rank_range": [None if pd.isna(dg.rank_best[t]) else int(dg.rank_best[t]),
                           None if pd.isna(dg.rank_worst[t]) else int(dg.rank_worst[t])],
            "robust": bool(dg.robust[t]),
            "pillars": {s.pillar_labels[p]: _r(s.pillar_scores.at[t, p], 2) for p in s.pillar_weights},
            "approved": None if c.approved_flag is None else bool(c.approved_flag.get(t, False)),
        })
    return {
        "kind": c.kind, "title": c.title, "asof": str(c.asof), "profile": c.meta["profile_id"],
        "profile_hash": c.meta["profile_hash"], "engine_version": c.meta["engine_version"], "files": c.meta["files"],
        "n_ranked": int(s.rank.notna().sum()), "pillar_weights": {s.pillar_labels[k]: _r(v) for k, v in s.pillar_weights.items()},
        "funds": funds,
        "excluded": c.excluded,
        "metrics": [{"metric": x.metric, "pillar": x.pillar, "status": x.status, "reason": x.reason,
                     "effective_weight": _r(x.effective_weight)} for x in s.statuses],
        "quality": [{"severity": e.severity, "area": e.area, "ticker": e.ticker, "message": e.message} for e in c.log.sorted()],
    }

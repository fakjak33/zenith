"""Analyst estimate-revision screen (post-forecast-revision drift) — pure math.

Composite of three free-data revision signals per stock:
  * est_rev_pct  — % change in the current-FY mean EPS estimate over ~30-90d
    (yfinance eps_trend: '30daysAgo'/'90daysAgo' vs 'current').
  * up_frac      — direction consistency: fraction of the last-quarter revisions
    that were UP (yfinance eps_revisions up/down counts).
  * net_reco     — net analyst recommendation upgrades minus downgrades over the
    last 30 days (yfinance upgrades_downgrades), scaled by count.

    composite rank = 0.5*rank(est_rev_pct) + 0.3*rank(up_frac) + 0.2*rank(net_reco)

Long = top decile (positive revision momentum), short = bottom decile. Drift is
attenuated post-publication (~50%) but persists via analyst underreaction
(Chen et al. 2020) — evidence tier B.
"""

from __future__ import annotations

from .common import assemble, pct_ranks

HORIZON_TD = 20
W = {"est_rev_pct": 0.5, "up_frac": 0.3, "net_reco": 0.2}


def build(rows: list[dict]) -> dict:
    """rows: [{ticker, name, sector, est_rev_pct, up_frac, net_reco}] with any
    of the three components possibly None. Composite ranks only over rows that
    have est_rev_pct (the anchor component); missing sub-components get a
    neutral 50 rank."""
    rows = [dict(r) for r in rows if r.get("est_rev_pct") is not None]
    if not rows:
        return {"screen": "revisions", "horizon_td": HORIZON_TD,
                "n": 0, "long": [], "short": [], "ranked": []}
    comp_ranks = {}
    for key in W:
        present = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
        if present:
            pr = pct_ranks([v for _, v in present])
            for (i, _), p in zip(present, pr):
                comp_ranks.setdefault(i, {})[key] = p
    for i, r in enumerate(rows):
        cr = comp_ranks.get(i, {})
        r["composite"] = round(sum(W[k] * cr.get(k, 50.0) for k in W), 1)
        r["r_est"], r["r_dir"], r["r_reco"] = (cr.get("est_rev_pct"),
                                               cr.get("up_frac"), cr.get("net_reco"))
    a = assemble(rows, "composite", higher_is_long=True)
    return {"screen": "revisions", "horizon_td": HORIZON_TD, **a}

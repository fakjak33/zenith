"""Short-interest / days-to-cover screen — pure math.

Per stock: short percent of float (SI%) and days-to-cover (DTC = short shares /
avg daily volume, yfinance shortRatio). High SI/DTC names have historically
underperformed (Rapach-Ringgenberg-Zhou 2016; Hong et al. days-to-cover), but
net of borrow fees the edge ~vanishes (Muravyev 2025), so the high-SI side is an
AVOID/short-watch list, not a tradable long-short. A squeeze-risk flag marks
high-SI names that are ALSO rising (crowded shorts that can rip).

  composite = 0.6*rank(SI%float) + 0.4*rank(DTC)   (higher = more shorted)
"""

from __future__ import annotations

from .common import assemble, pct_ranks, zscore

HORIZON_TD = 20
W = {"si_float": 0.6, "dtc": 0.4}
SQUEEZE_SI = 60.0       # SI%-float percentile above which a riser is squeeze-risk
SQUEEZE_MOM = 0.0       # positive 1m momentum


def build(rows: list[dict], si_float_history: list[float] | None = None) -> dict:
    """rows: [{ticker, name, sector, si_float, dtc, mom_1m}]. si_float in %
    (0-100), dtc in days. Returns ranked screen; SHORT/avoid = top decile."""
    rows = [dict(r) for r in rows if r.get("si_float") is not None
            or r.get("dtc") is not None]
    if not rows:
        return {"screen": "shortint", "horizon_td": HORIZON_TD,
                "n": 0, "long": [], "short": [], "ranked": [], "aggregate": None}
    ranks = {}
    for key in W:
        present = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
        if present:
            pr = pct_ranks([v for _, v in present])
            for (i, _), p in zip(present, pr):
                ranks.setdefault(i, {})[key] = p
    for i, r in enumerate(rows):
        cr = ranks.get(i, {})
        r["composite"] = round(sum(W[k] * cr.get(k, 50.0) for k in W), 1)
        r["si_pctile"] = cr.get("si_float")
        r["squeeze_risk"] = bool((cr.get("si_float") or 0) >= SQUEEZE_SI
                                 and (r.get("mom_1m") or -1) > SQUEEZE_MOM)
    # higher composite = more heavily shorted = the SHORT/avoid side
    a = assemble(rows, "composite", higher_is_long=False)
    # aggregate Rapach-style gauge: median SI%float z vs its own committed history
    med = sorted(r["si_float"] for r in rows if r.get("si_float") is not None)
    median_si = med[len(med) // 2] if med else None
    agg = None
    if median_si is not None:
        hist = (si_float_history or []) + [median_si]
        agg = {"median_si_float": round(median_si, 2),
               "z_vs_history": zscore(median_si, hist),
               "n_history": len(si_float_history or [])}
    return {"screen": "shortint", "horizon_td": HORIZON_TD, "aggregate": agg, **a}

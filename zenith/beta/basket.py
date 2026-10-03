"""CLEAN BETA basket: greedy, equal-weight, capped. No optimizer dependency.

Order of seating (each step respects the sector cap):
  1. incumbents that still pass every filter keep their seat (the buffer band:
     an incumbent's beta filter is BETA_PCT_EXIT, not BETA_PCT_ENTER), so a
     name does not churn out the quarter it slips from the 80th to 75th pct;
  2. each size bucket gets BETA_MIN_PER_SIZE_BUCKET names when it has
     candidates (so the basket is not all mega-cap or all small-cap);
  3. the rest fill by score minus a pairwise-correlation penalty against names
     already seated — correlated high-beta names are fewer real bets than
     names, and the penalty pushes the greedy walk toward diversifiers.

Equal weight, capped at BETA_NAME_CAP; with fewer than 1/cap names the
remainder is reported as uninvested rather than silently over-weighted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import (BETA_BASKET_MAX, BETA_BASKET_MIN, BETA_BASKET_TARGET, BETA_CORR_PENALTY,
                      BETA_CORR_THRESHOLD, BETA_MIN_PER_SIZE_BUCKET, BETA_NAME_CAP,
                      BETA_SECTOR_CAP, BETA_SIZE_BUCKETS)
from . import r
from .estimators import TRADING_DAYS, effective_bets, effective_dimension


def select(rows: list[dict], rets: pd.DataFrame, target: int = BETA_BASKET_TARGET) -> dict:
    """rows: screened rows (passing ones are candidates); rets: daily returns
    panel covering at least the candidates. Returns the basket document."""
    target = max(BETA_BASKET_MIN, min(BETA_BASKET_MAX, target))
    cands = [x for x in rows if x.get("passes")]
    cands.sort(key=lambda x: -(x.get("score") or 0.0))
    corr = rets[[x["ticker"] for x in cands if x["ticker"] in rets]].corr() if len(cands) else None
    sector_max = max(1, math.floor(BETA_SECTOR_CAP * target + 1e-9))
    chosen, why = _fill(cands, corr, target, sector_max)
    n = len(chosen)
    weights = cap_weights([c["sector"] for c in chosen])
    members = [{"ticker": c["ticker"], "name": c.get("name"), "sector": c["sector"],
                "size": c.get("size"), "weight": r(w, 4), "bswa": c.get("bswa"),
                "rho": c.get("rho"), "ivol": c.get("ivol"), "score": c.get("score"),
                "why": why[c["ticker"]]} for c, w in zip(chosen, weights)]
    st = stats(members, rets)
    inv = sum(weights)
    st["beta_invested"] = r(st["beta"] / inv, 3) if st.get("beta") is not None and inv > 0 else None
    return {"members": members, **st,
            "n": n, "target": target, "short": n < BETA_BASKET_MIN,
            "n_candidates": len(cands), "invested": r(sum(weights), 4),
            "sector_cap_names": sector_max}


def cap_weights(sectors: list[str]) -> list[float]:
    """Equal weight subject to the single-name and sector WEIGHT caps.
    Water-filling: start at min(1/n, name cap), scale any sector above its cap
    down to it, hand the freed weight to names that still have room under both
    caps, repeat. Whatever cannot be placed stays uninvested -- a short or
    sector-concentrated basket is reported as partly in cash, never quietly
    over-weighted."""
    n = len(sectors)
    if not n:
        return []
    w = np.full(n, min(1.0 / n, BETA_NAME_CAP))
    sec = np.array(sectors)
    for _ in range(50):
        for s_ in set(sectors):
            m = sec == s_
            tot = w[m].sum()
            if tot > BETA_SECTOR_CAP:
                w[m] *= BETA_SECTOR_CAP / tot
        free = 1.0 - w.sum()
        room = np.array([min(BETA_NAME_CAP - w[i],
                             BETA_SECTOR_CAP - w[sec == sec[i]].sum()) for i in range(n)])
        room = np.clip(room, 0.0, None)
        if free <= 1e-9 or room.sum() <= 1e-9:
            break
        w += room / room.sum() * min(free, room.sum()) * (room > 0)
    return [float(x) for x in w]


def _fill(cands: list[dict], corr, target: int, sector_max: int) -> tuple[list[dict], dict]:
    chosen: list[dict] = []
    sector_n: dict[str, int] = {}
    why: dict[str, str] = {}

    def can_seat(x):
        return sector_n.get(x["sector"], 0) < sector_max and x not in chosen

    def seat(x, reason):
        chosen.append(x)
        sector_n[x["sector"]] = sector_n.get(x["sector"], 0) + 1
        why[x["ticker"]] = reason

    for x in cands:                                     # 1. incumbents
        if len(chosen) >= target:
            break
        if x.get("incumbent") and can_seat(x):
            seat(x, "incumbent (buffer band)")
    for bucket, _ in BETA_SIZE_BUCKETS:                 # 2. size representation
        have = sum(1 for c in chosen if c.get("size") == bucket)
        for x in cands:
            if have >= BETA_MIN_PER_SIZE_BUCKET or len(chosen) >= target:
                break
            if x.get("size") == bucket and can_seat(x):
                seat(x, f"{bucket.lower()}-cap representation")
                have += 1
    while len(chosen) < target:                         # 3. penalized greedy fill
        best, best_adj, best_pen = None, -1e18, 0.0
        for x in cands:
            if not can_seat(x):
                continue
            pen = 0.0
            if corr is not None and chosen and x["ticker"] in corr:
                cs = [corr.at[x["ticker"], c["ticker"]] for c in chosen if c["ticker"] in corr]
                cs = [c for c in cs if np.isfinite(c)]
                if cs:
                    pen = BETA_CORR_PENALTY * max(0.0, max(cs) - BETA_CORR_THRESHOLD)
            adj = (x.get("score") or 0.0) - pen
            if adj > best_adj:
                best, best_adj, best_pen = x, adj, pen
        if best is None:
            break
        seat(best, "score" + (f" (corr penalty −{best_pen:.1f})" if best_pen > 0 else ""))
    return chosen, why


def stats(members: list[dict], rets: pd.DataFrame) -> dict:
    """Ex-ante beta, vol, effective bets, sector/size mix, avg pairwise corr."""
    if not members:
        return {}
    tick = [m["ticker"] for m in members if m["ticker"] in rets]
    w = np.array([m["weight"] or 0.0 for m in members if m["ticker"] in rets])
    beta = sum((m["weight"] or 0.0) * (m["bswa"] or 0.0) for m in members)
    out = {"beta": r(beta, 3)}
    sectors: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for m in members:
        sectors[m["sector"]] = sectors.get(m["sector"], 0.0) + (m["weight"] or 0.0)
        sizes[m.get("size") or "—"] = sizes.get(m.get("size") or "—", 0) + 1
    out["sectors"] = {k: r(v, 4) for k, v in sorted(sectors.items(), key=lambda kv: -kv[1])}
    out["sizes"] = sizes
    if len(tick) >= 2:
        panel = rets[tick].dropna(how="all").fillna(0.0)
        cov = panel.cov().values
        out["vol"] = r(math.sqrt(max(w @ cov @ w, 0.0) * TRADING_DAYS), 4)
        out["enb"] = r(effective_bets(w, cov), 2)
        c = panel.corr().values
        out["eff_dim"] = r(effective_dimension(c), 2)
        iu = np.triu_indices_from(c, k=1)
        out["avg_corr"] = r(float(np.nanmean(c[iu])), 3)
    return out

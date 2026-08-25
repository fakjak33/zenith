"""IDEAS select: candidate pool -> final ranked BUY/SELL lists.

Two stages, split for cost reasons (see riskreward.py's module docstring):

  * `rank_candidates()` -- runs the eight-group scorer + conviction +
    unusualness over the WHOLE scan universe using only already-committed
    data (no network), and narrows to a candidate pool by conviction/
    unusualness gates. Cheap, runs on ~1300 names.

  * `finalize()` -- takes the candidate pool AFTER compute.py has fetched
    fresh prices for just those names (ADV$ liquidity + riskreward.build),
    applies the remaining gates (liquidity floor, ETF slot cap, target/max
    count), classifies and narrates each survivor, and produces the final
    BUY/SELL lists. NEVER pads to a quota (spec section 1) -- a thin day
    shows fewer than the target and says so via `thin_day` on the result.
"""

from __future__ import annotations

from ..config import IDEAS_GATES
from . import groups, conviction, unusual, divergence, confluence, classify, narrative


def score_security(ticker: str, row: dict, regime_summary: dict) -> dict:
    """One security's full scoring bundle (everything downstream needs),
    computed entirely from already-fused panel data -- no network."""
    gscores = groups.score_all(row, regime_summary)
    conv = conviction.compute(gscores, regime_summary.get("label"))
    unu = unusual.compute(gscores, ticker)
    div = divergence.compute(gscores)
    net = conv["net_score"]
    conf = confluence.compute(gscores, net)
    side = "long" if net > 0 else ("short" if net < 0 else None)
    return {
        "ticker": ticker, "meta": row.get("meta", {}), "side": side,
        "net_score": net, "conviction": conviction.magnitude(net) if side else 0.0,
        "unusual": unu["unusual"], "coverage_n": conv["coverage_n"],
        "group_scores": gscores, "conviction_detail": conv, "unusual_detail": unu,
        "divergence": div, "confluence": conf,
    }


def rank_candidates(panel: dict[str, dict], regime_summary: dict,
                    top_n_per_side: int = 150) -> list[dict]:
    """Score the whole universe, apply the conviction/unusualness floors, and
    return the top `top_n_per_side` per direction by a 50/50 blend of the two
    -- the candidate pool that gets a fresh price fetch + riskreward.build in
    compute.py."""
    gates = IDEAS_GATES
    scored = []
    for t, row in panel.items():
        s = score_security(t, row, regime_summary)
        if s["side"] is None:
            continue
        if s["coverage_n"] < gates["min_coverage_n"]:
            continue
        if s["conviction"] < gates["min_conviction"] or s["unusual"] < gates["min_unusual"]:
            continue
        s["_rank_key"] = 0.5 * s["conviction"] + 0.5 * s["unusual"]
        scored.append(s)

    out = []
    for side in ("long", "short"):
        side_rows = sorted((s for s in scored if s["side"] == side),
                           key=lambda s: s["_rank_key"], reverse=True)
        out.extend(side_rows[:top_n_per_side])
    return out


def finalize(candidates: list[dict], riskreward_by_ticker: dict[str, dict],
            adv_usd_by_ticker: dict[str, float], catalyst_days_by_ticker: dict[str, int] | None = None
            ) -> dict:
    """Apply the liquidity floor + ETF slot cap + target/max count, classify
    and narrate every survivor, and return {buy: [...], sell: [...], thin_day:
    {long: bool, short: bool}}."""
    gates = IDEAS_GATES
    catalyst_days_by_ticker = catalyst_days_by_ticker or {}

    survivors = []
    for c in candidates:
        t = c["ticker"]
        adv = adv_usd_by_ticker.get(t)
        if adv is not None and adv < gates["min_adv_usd"]:
            continue
        rr = riskreward_by_ticker.get(t)
        if not rr or not rr.get("available"):
            continue
        if rr.get("rr_ratio") is not None and rr["rr_ratio"] < 1.0:
            continue      # a structurally poor risk/reward is disqualifying, regardless of conviction
        c = dict(c)
        c["riskreward"] = rr
        survivors.append(c)

    result = {"buy": [], "sell": [], "thin_day": {"long": False, "short": False}}
    for side, key in (("long", "buy"), ("short", "sell")):
        pool = sorted((s for s in survivors if s["side"] == side),
                     key=lambda s: s["_rank_key"], reverse=True)
        n_etf = 0
        picked = []
        for c in pool:
            if len(picked) >= gates["max_n_per_side"]:
                break
            is_etf = c["meta"].get("security_type") == "etf"
            if is_etf:
                if n_etf >= gates["max_etf_slots_per_side"]:
                    continue
                n_etf += 1
            picked.append(c)
        result["thin_day"][side] = len(picked) < gates["target_n_per_side"]

        for c in picked:
            t = c["ticker"]
            cls = classify.classify(side, c["group_scores"], c["divergence"],
                                    security_type=c["meta"].get("security_type", "stock"),
                                    catalyst_days_out=catalyst_days_by_ticker.get(t))
            nar = narrative.build(t, side, cls["opportunity_type"], c["group_scores"],
                                  c["divergence"], c["confluence"], c["riskreward"])
            idea = {**{k: v for k, v in c.items() if k != "_rank_key"}, **cls, "narrative": nar}
            result[key].append(idea)

    return result

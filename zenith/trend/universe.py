"""TREND FOLLOWING universes — inherited, never maintained here.

There is deliberately no ticker list in this package. Both universes are the
EXACT constituent functions the existing tabs call, so Trend Following stays
synchronized with them automatically: a Russell 1000 reconstitution or a change
to the ETF MOMENTUM universe shows up here on the next nightly run with no code
change.

  stocks  -> zenith.mom.universe.constituents()   (MOMENTUM: the Russell 1000,
             itself a pass-through to pretom.universe.russell1000()) + the
             committed data/mom/meta.json cache for sector / industry / market
             cap. Read-only: MOMENTUM owns and refreshes that cache.
  etfs    -> zenith.etfmom.universe.constituents()'s `included` rows (ETF
             MOMENTUM) + the SAME empirical leveraged/inverse backstop ETF
             MOMENTUM applies at compute time (etfmom.compute._leverage_exclusions),
             so the two ETF tabs score one identical set.
"""

from __future__ import annotations


def _dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    """First occurrence of each ticker wins. The inherited Russell 1000 feed
    has been seen to carry the same ticker twice (17 names on 2026-09-19); the
    ticker SET is still exactly the source tab's, just without double rows
    (which would double-count breadth and double-append events)."""
    seen, out = set(), []
    for r in rows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        out.append(r)
    return out, len(rows) - len(out)


def stocks() -> tuple[list[dict], dict]:
    from ..mom import load as mom_load
    from ..mom import universe as mom_universe
    rows, status = mom_universe.constituents()
    meta = mom_load("meta", {})
    out = []
    for u in rows:
        t = u["ticker"]
        m = meta.get(t, {})
        out.append({"ticker": t, "name": u.get("name") or m.get("name") or t,
                    "sector": m.get("sector") or u.get("sector") or "",
                    "industry": m.get("industry") or "", "mktcap": m.get("mktcap")})
    out, n_dup = _dedupe(out)
    return out, {"source": "mom.universe.constituents", **(status or {}), "n_duplicates_dropped": n_dup}


def etfs() -> tuple[list[dict], dict]:
    from ..etfmom import universe as etf_universe
    rows, status = etf_universe.constituents()
    out = [{k: u.get(k) for k in ("ticker", "name", "category", "asset_class", "region",
                                  "aum_m", "er")}
           for u in rows if u.get("included")]
    out, n_dup = _dedupe(out)
    return out, {"source": "etfmom.universe.constituents", "n_included": len(out),
                 "n_duplicates_dropped": n_dup}


def etf_leverage_exclusions(px: dict) -> dict[str, str]:
    """ETF MOMENTUM's own empirical leverage gate, applied to the same prices."""
    from ..etfmom.compute import _leverage_exclusions
    return _leverage_exclusions(px)


LOADERS = {"stocks": stocks, "etfs": etfs}

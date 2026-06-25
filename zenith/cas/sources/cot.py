"""CFTC Commitments of Traders (COT) — weekly futures positioning, free.

Uses the public CFTC Socrata dataset (Traders in Financial Futures + the legacy
disaggregated report) to get dealer / asset-manager / leveraged-fund /
managed-money net positioning for the major futures markets. This is the best
*free* proxy for dealer, CTA, hedge-fund and managed-futures positioning.

Returns market -> list of weekly records {date, dealer_net, asset_mgr_net,
lev_money_net, mgd_money_net, open_interest}. Degrades gracefully.
"""

from __future__ import annotations

import requests

from .. import store_cas

# CFTC Socrata — Traders in Financial Futures (combined) + Disaggregated (combined)
TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
DISAGG_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
TIMEOUT = 20


def _num(rec, *keys) -> float:
    for k in keys:
        v = rec.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def get_positioning(markets: list[str], weeks: int = 104,
                    max_age_hours: float = 36.0) -> tuple[dict[str, list], dict]:
    """Fetch recent COT weeks for the named markets (substring match on
    market_and_exchange_names). Cached ~weekly."""
    key = "cot"
    cached = store_cas.cache_get(key, max_age_hours)
    if cached is not None:
        return {m: cached.get(m, []) for m in markets}, {
            "ok": True, "n": sum(len(cached.get(m, [])) for m in markets),
            "source": "cftc(cache)"}

    out: dict[str, list] = {m: [] for m in markets}
    err = ""
    for url, fld in ((TFF_URL, "tff"), (DISAGG_URL, "disagg")):
        try:
            r = requests.get(url, params={"$limit": 50000, "$order": "report_date_as_yyyy_mm_dd DESC",
                                          "$where": "report_date_as_yyyy_mm_dd > '2023-01-01'"},
                             timeout=TIMEOUT)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            err = str(e)[:160]
            continue
        for rec in rows:
            name = (rec.get("market_and_exchange_names") or "").upper()
            for m in markets:
                if m.upper() in name and len(out[m]) < weeks:
                    out[m].append({
                        "date": rec.get("report_date_as_yyyy_mm_dd", "")[:10],
                        "dealer_net": _num(rec, "dealer_positions_long_all", "dealer_positions_long")
                                      - _num(rec, "dealer_positions_short_all", "dealer_positions_short"),
                        "asset_mgr_net": _num(rec, "asset_mgr_positions_long", "asset_mgr_positions_long_all")
                                         - _num(rec, "asset_mgr_positions_short", "asset_mgr_positions_short_all"),
                        "lev_money_net": _num(rec, "lev_money_positions_long", "lev_money_positions_long_all")
                                         - _num(rec, "lev_money_positions_short", "lev_money_positions_short_all"),
                        "mgd_money_net": _num(rec, "m_money_positions_long_all", "managed_money_longs")
                                         - _num(rec, "m_money_positions_short_all", "managed_money_shorts"),
                        "open_interest": _num(rec, "open_interest_all", "open_interest"),
                    })

    have = {m: v for m, v in out.items() if v}
    if have:
        store_cas.cache_put(key, out)
    return out, {"ok": bool(have), "n": sum(len(v) for v in have.values()),
                 "source": "cftc", "error": "" if have else (err or "no matching markets")}

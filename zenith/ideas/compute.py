"""IDEAS compute orchestrator.

  python -m zenith.ideas.compute --action auto          (nightly path)
  python -m zenith.ideas.compute --action fundamentals   (force a fuller fundamentals refresh)

Nightly path: gate on the trading calendar (after mom.yml, edge.yml, pead.yml
and fmom have already committed their day's artifacts -- see ideas.yml's cron),
build the fused universe/panel, refresh a slice of the committed fundamentals
cache, compute the market regime, score the whole universe (no network),
narrow to a candidate pool, fetch fresh prices for JUST that pool to build the
riskreward construction + liquidity check, finalize the BUY/SELL lists, and
write every artifact. Never pads the daily list to a quota (spec section 1).
"""

from __future__ import annotations

import argparse
import math
from datetime import date

from . import DISCLAIMER, save, load
from . import panel as ideas_panel
from . import select as ideas_select
from . import regime as ideas_regime
from . import fundamentals as ideas_fund
from . import valuation as ideas_val
from . import riskreward as ideas_riskreward
from ..pretom import calendar as cal
from ..cas.universe import master_etfs
from ..cas.sources import prices


def _scrub(obj):
    """Recursively replace non-finite floats with None -- json.dumps emits
    bare NaN/Infinity otherwise, which is not valid JSON (mom/compute.py's
    same guard, reused verbatim here since this package produces even more
    derived float fields)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _days_out(report_date: str | None, today: date) -> int | None:
    if not report_date:
        return None
    try:
        rd = date.fromisoformat(report_date)
    except (TypeError, ValueError):
        return None
    if rd < today:
        return None
    return len(cal.trading_days(today, rd)) - 1


def _catalyst_days_map(candidates: list[dict], today: date) -> dict[str, int]:
    out = {}
    for c in candidates:
        cat = c["group_scores"].get("catalyst", {})
        e = (cat.get("explain") or {})
        rd = e.get("upcoming_report_date") or e.get("recent_report_date")
        d = _days_out(rd, today)
        if d is not None:
            out[c["ticker"]] = d
    return out


def _fetch_prices(tickers: list[str], period: str, status: list[dict], label: str):
    px, st = prices.get_history(tickers, period=period)
    status.append({"segment": label, "ok": bool(px), "n": len(px),
                   "requested": len(tickers), "error": st.get("error", "")})
    return px


def run_auto(force: bool = False, candidate_pool: int = 150) -> dict:
    today = date.today()
    status: list[dict] = []
    if not cal.is_trading_day(today) and not force:
        save("status", {"date": today.isoformat(), "is_trading_day": False,
                        "disclaimer": DISCLAIMER,
                        "segments": [{"segment": "gate", "ok": True, "note": "non-trading day"}]})
        print(f"[ideas] {today} non-trading day -- no-op")
        return {"ok": True, "gated": True}

    universe = ideas_panel.build_universe()
    status.append({"segment": "universe", "ok": bool(universe), "n": len(universe)})
    tickers = [u["ticker"] for u in universe]

    etf_tickers = sorted({u["ticker"] for u in universe if u["security_type"] == "etf"} | set(master_etfs()))
    etf_px = _fetch_prices(etf_tickers, "2y", status, "etf_prices")

    fund_status = ideas_fund.refresh(tickers)
    status.append({"segment": "fundamentals", "ok": True, **fund_status})

    regime_summary = ideas_regime.compute_regime()
    status.append({"segment": "regime", "ok": bool(regime_summary), **regime_summary})

    panel = ideas_panel.build_panel(universe, etf_px=etf_px)
    status.append({"segment": "panel", "ok": True, "n": len(panel)})

    # snapshot this month's fundamentals into the own-history archive (idempotent per month)
    fund_cache = ideas_fund.get(tickers)
    added_snap = ideas_val.append_monthly_snapshot(fund_cache, today) if fund_cache else 0
    status.append({"segment": "valuation_history", "ok": True, "added": added_snap})

    candidates = ideas_select.rank_candidates(panel, regime_summary, top_n_per_side=candidate_pool)
    status.append({"segment": "candidates", "ok": True, "n": len(candidates)})

    cand_tickers = [c["ticker"] for c in candidates]
    cand_px = _fetch_prices(cand_tickers, "2y", status, "candidate_prices")

    rr_by, adv_by = {}, {}
    for c in candidates:
        t = c["ticker"]
        df = cand_px.get(t)
        if df is None or df.empty:
            continue
        tech = panel.get(t, {}).get("technicals", {})
        bg = (tech.get("breakout_grid") or {}).get("1m", {})
        rr_by[t] = ideas_riskreward.build(
            t, c["side"], df, breakout_confirmed=bg.get("confirmed"), state=tech.get("state"))
        try:
            adv_by[t] = float((df["close"] * df["volume"]).tail(63).mean())
        except Exception:
            adv_by[t] = None

    catalyst_days = _catalyst_days_map(candidates, today)
    result = ideas_select.finalize(candidates, rr_by, adv_by, catalyst_days)
    status.append({"segment": "select", "ok": True, "n_buy": len(result["buy"]),
                   "n_sell": len(result["sell"]), "thin_day": result["thin_day"]})

    ideas_doc = {
        "as_of": today.isoformat(), "disclaimer": DISCLAIMER,
        "regime": regime_summary, "thin_day": result["thin_day"],
        "n_buy": len(result["buy"]), "n_sell": len(result["sell"]),
        "buy": result["buy"], "sell": result["sell"],
        "coverage": {"n_universe": len(universe), "n_panel": len(panel),
                    "n_candidates": len(candidates)},
    }
    save("ideas", _scrub(ideas_doc), indent=2)

    from ..config import IDEAS_ARCHIVE_DIR
    import json as _json
    IDEAS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    (IDEAS_ARCHIVE_DIR / f"{today.isoformat()}.json").write_text(
        _json.dumps(_scrub(ideas_doc), indent=None, ensure_ascii=False), encoding="utf-8")

    cand_compact = [{
        "ticker": c["ticker"], "side": c["side"], "conviction": c["conviction"],
        "unusual": c["unusual"], "coverage_n": c["coverage_n"],
        "security_type": c["meta"].get("security_type"),
    } for c in candidates]
    save("candidates", _scrub({"as_of": today.isoformat(), "n": len(cand_compact), "rows": cand_compact}),
        indent=None)

    uni_scores = {t: {"conviction": None, "unusual": None} for t in tickers}
    for c in candidates:
        uni_scores[c["ticker"]] = {"conviction": c["conviction"], "unusual": c["unusual"],
                                   "side": c["side"]}
    save("universe_scores", _scrub({"as_of": today.isoformat(), "rows": uni_scores}), indent=None)

    save("status", {"date": today.isoformat(), "is_trading_day": True, "disclaimer": DISCLAIMER,
                    "segments": status})
    print(f"[ideas] {today} universe={len(universe)} candidates={len(candidates)} "
          f"buy={len(result['buy'])} sell={len(result['sell'])} thin_day={result['thin_day']} "
          f"regime={regime_summary.get('label')}")
    return {"ok": True, "n_buy": len(result["buy"]), "n_sell": len(result["sell"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "fundamentals"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--candidate-pool", type=int, default=150)
    args = ap.parse_args()
    if args.action == "fundamentals":
        universe = ideas_panel.build_universe()
        tickers = [u["ticker"] for u in universe]
        res = ideas_fund.refresh(tickers, max_per_run=len(tickers))
        print(f"[ideas] fundamentals full refresh: {res}")
    else:
        run_auto(force=args.force, candidate_pool=args.candidate_pool)


if __name__ == "__main__":
    main()

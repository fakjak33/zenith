"""NIGHT & DAY compute orchestrator.

  python -m zenith.nightday.compute --action auto

Cheap (prices only): decompose an ETF panel into overnight/intraday cumulative
series, rank the Russell 1000 by the overnight-momentum signal and tug-of-war
spread, snapshot decile members, and evaluate matured horizons.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from . import DISCLAIMER, save, load
from . import core, history
from ..pretom import calendar as cal
from ..pretom import universe as uni
from ..cas.sources import prices
from ..cas.universe import STYLE_GRID

PANEL_ETFS = ["SPY", "QQQ", "IWM", "MTUM", "VLUE", "QUAL", "USMV", "SPHB"]
SCREEN_DECILE = 0.05


def _panel_tickers() -> list[str]:
    extra = []
    try:
        for region in STYLE_GRID.values():
            for etf in region.values():
                if etf and etf not in extra:
                    extra.append(etf)
    except Exception:
        pass
    seen, out = set(), []
    for t in PANEL_ETFS + extra:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out[:16]


def _fetch(tickers, period, status, label):
    px, st = prices.get_history(tickers, period=period)
    miss = [t for t in tickers if t not in px]
    if len(miss) > max(5, len(tickers) // 20):
        time.sleep(15)
        px2, _ = prices.get_history(miss, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px)})
    return px


def run_auto(force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []
    if not cal.is_trading_day(today) and not force:
        save("status", {"date": today.isoformat(), "is_trading_day": False,
                        "segments": [{"segment": "gate", "ok": True, "note": "non-trading"}]})
        print(f"[nightday] {today} non-trading day — no-op")
        return {"ok": True, "gated": True}

    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    meta = {u["ticker"]: {"name": u.get("name", ""), "sector": u.get("sector", "")}
            for u in universe}
    tickers = [u["ticker"] for u in universe]
    panel_t = _panel_tickers()

    px = _fetch(sorted(set(tickers + panel_t + ["SPY"])), "2y", status, "prices")
    spy = px.get("SPY")
    spy_close = spy["close"] if spy is not None and not spy.empty else None

    # --- ETF panel: cumulative overnight vs intraday --------------------------
    panel = {}
    for t in panel_t:
        df = px.get(t)
        if df is None or df.empty:
            continue
        stats = core.stock_stats(df)
        series = core.cumulative_panel(df)
        if stats and series:
            panel[t] = {"name": meta.get(t, {}).get("name", t), "stats": stats,
                        "series": series}
    save("panel", {"as_of": today.isoformat(), "disclaimer": DISCLAIMER,
                   "etfs": panel})
    status.append({"segment": "panel", "ok": bool(panel), "n": len(panel)})

    # --- R1000 screen: overnight-momentum + tug-of-war ------------------------
    rows = []
    for t in tickers:
        df = px.get(t)
        if df is None or df.empty:
            continue
        s = core.stock_stats(df)
        if s and s.get("on_mom") is not None:
            rows.append({"ticker": t, **meta.get(t, {}), **s})
    rows.sort(key=lambda r: r["on_mom"], reverse=True)
    n = len(rows)
    cut = max(1, round(n * SCREEN_DECILE))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    longs = [{**r, "side": "long"} for r in rows[:cut]]
    shorts = [{**r, "side": "short"} for r in rows[-cut:]]
    tug_hi = sorted(rows, key=lambda r: r["tug"], reverse=True)[:cut]
    tug_lo = sorted(rows, key=lambda r: r["tug"])[:cut]
    screen = {"as_of": today.isoformat(), "disclaimer": DISCLAIMER, "n": n,
              "long": longs, "short": shorts, "tug_overnight": tug_hi,
              "tug_intraday": tug_lo, "ranked": rows[:200], "horizon_td": history.HORIZON_TD}
    save("screen", screen)
    status.append({"segment": "screen", "ok": bool(n), "n": n})

    # --- history snapshot + eval ---------------------------------------------
    added = history.append_rows(history.make_rows(screen, today.isoformat()))
    n_eval = 0
    if spy_close is not None:
        n_eval = history.evaluate_pending(px, spy_close, today)
    status.append({"segment": "history", "ok": True, "added": added, "evaluated": n_eval})

    save("status", {"date": today.isoformat(), "is_trading_day": True,
                    "disclaimer": DISCLAIMER, "segments": status})
    print(f"[nightday] {today} panel={len(panel)} screen={n} hist +{added} eval {n_eval}")
    return {"ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_auto(force=args.force)


if __name__ == "__main__":
    main()

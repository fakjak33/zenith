"""CAS compute orchestrator.

  python -m zenith.cas.compute --cadence daily|weekly|monthly

Pulls the right free data for the cadence, runs the signal families, aggregates
into consensus + overlap, evaluates contingency, and writes JSON under data/cas/.
Mirrors zenith.scrape: degrade-gracefully, per-segment status, archive merge.

Cadence:
  daily   — prices -> strategies, themes, regime, consensus, overlap, contingency
  weekly  — daily + COT positioning (flows)
  monthly — weekly + 13F filings
"""

from __future__ import annotations

import argparse
from datetime import date

from . import store_cas, consensus, overlap, registry, contingency
from .universe import all_etfs, COT_MAP
from .sources import prices, cot, finviz, fred, edgar13f, calendar as cal
from .signals import (strategies, strategies151, flows, themes, rebalance, regime,
                      factor_rotation)
from .universe import master_etfs
from .etf_master import category_of as master_category_of


def run(cadence: str = "daily") -> dict:
    status: list[dict] = []
    signals: list[dict] = []
    # pull the full master list once (superset of the core universe); the core
    # segments use the subset, the 151-model uses the whole thing
    master = master_etfs()
    core = list(all_etfs())

    # --- prices (always) ---
    px, st = prices.get_history(master)
    status.append({"segment": "prices", **st})
    core_px = {t: px[t] for t in core if t in px}

    # --- strategies (always, if prices) ---
    if core_px:
        try:
            s = strategies.compute(core_px)
            signals += s
            status.append({"segment": "strategies", "ok": True, "n": len(s)})
        except Exception as e:
            status.append({"segment": "strategies", "ok": False, "error": str(e)[:200]})

    # --- 151-strategies model across the master ETF list ---
    if px:
        try:
            s151 = strategies151.compute(px, category_of=master_category_of)
            signals += s151
            status.append({"segment": "strategies151", "ok": True, "n": len(s151),
                           "universe": len(px)})
        except Exception as e:
            status.append({"segment": "strategies151", "ok": False, "error": str(e)[:200]})

        # --- themes (always) ---
        groups, gst = finviz.get_groups("sector")
        status.append({"segment": "finviz", **gst})
        try:
            th = themes.compute(px, groups)
            signals += th
            store_cas.save("themes", th)
            status.append({"segment": "themes", "ok": True, "n": len(th)})
        except Exception as e:
            status.append({"segment": "themes", "ok": False, "error": str(e)[:200]})

        # --- factor rotation momentum (always) ---
        try:
            frm = factor_rotation.compute(px)
            signals += frm
            store_cas.save("factor_rotation", frm)
            store_cas.save("rotation", factor_rotation.rotation_by_timeframe(px))
            status.append({"segment": "factor_rotation", "ok": True, "n": len(frm)})
        except Exception as e:
            status.append({"segment": "factor_rotation", "ok": False, "error": str(e)[:200]})

    # --- regime (always) ---
    fred_data, fst = fred.get_series(list(fred.DEFAULT_SERIES))
    status.append({"segment": "fred", **fst})
    regime_summary = {}
    try:
        rsig, regime_summary = regime.compute(fred_data, px)
        signals += rsig
        status.append({"segment": "regime", "ok": True, "n": len(rsig)})
    except Exception as e:
        status.append({"segment": "regime", "ok": False, "error": str(e)[:200]})

    # --- rebalance calendar (always) ---
    events = cal.upcoming()
    store_cas.save("rebalance", events)
    rb = rebalance.compute(events)
    signals += rb
    status.append({"segment": "rebalance", "ok": True, "n": len(rb)})

    # --- flows / positioning (weekly+) ---
    if cadence in ("weekly", "monthly"):
        cot_data, cst = cot.get_positioning(list(COT_MAP.values()))
        status.append({"segment": "cot", **cst})
        try:
            fl = flows.compute(cot_data, px)
            signals += fl
            store_cas.save("positioning", {"cot": cot_data, "manual_dynamics": flows.MANUAL_DYNAMICS})
            status.append({"segment": "flows", "ok": True, "n": len(fl)})
        except Exception as e:
            status.append({"segment": "flows", "ok": False, "error": str(e)[:200]})
    elif px:
        # volume-flow proxy is cheap; include it daily too
        fl = flows.from_volume(px)
        signals += fl
        status.append({"segment": "flows(volume-proxy)", "ok": True, "n": len(fl)})

    # --- 13F (monthly) ---
    if cadence == "monthly":
        f13, est = edgar13f.recent_13f()
        status.append({"segment": "edgar13f", **est})
        store_cas.save("positioning", {**store_cas.load("positioning", {}), "edgar_13f": f13})

    # --- factor-momentum academic backtest (monthly; validates the FRM model) ---
    if cadence == "monthly":
        try:
            from .backtest import factor_momentum as bt
            bt_out = bt.run()
            status.append({"segment": "backtest", "ok": True,
                           "n": bt_out.get("n_factors", 0)})
        except Exception as e:
            status.append({"segment": "backtest", "ok": False, "error": str(e)[:200]})

    # --- aggregate ---
    cons = consensus.build(signals)
    ovl = overlap.build(signals)
    scen = contingency.evaluate(regime_summary, fred_data, ovl["ranked"])
    registry.load()           # ensure registry seeded

    day = date.today().isoformat()
    store_cas.save("signals", signals)
    store_cas.save("consensus", cons)
    # store overlap without the bulky matrix duplicated under each asset
    store_cas.save("overlap", {"ranked": ovl["ranked"], "matrix": ovl["matrix"]})
    store_cas.archive_signals(day, signals)

    # --- signal history + predictive hit-rate (history fetches deep 5y FRM prices;
    #     the committed price panel is built from the broad master px so the app's
    #     price overlays work without the gitignored cache) ---
    try:
        from .analytics import history
        h = history.run()
        store_cas.save("price_panel", history.build_price_panel(px))
        status.append({"segment": "history+hitrate", "ok": True, "n": h.get("n_assets", 0)})
    except Exception as e:
        status.append({"segment": "history+hitrate", "ok": False, "error": str(e)[:200]})

    status_out = {"date": day, "cadence": cadence, "n_signals": len(signals),
                  "n_assets": len(cons), "regime": regime_summary.get("label", "?"),
                  "segments": status}
    store_cas.save("status", status_out)

    print(f"[cas] {day} cadence={cadence} signals={len(signals)} assets={len(cons)} "
          f"regime={regime_summary.get('label','?')}")
    for s in status:
        flag = "ok " if s.get("ok", True) else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', s.get('n', ''))}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    return status_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default="daily", choices=["daily", "weekly", "monthly"])
    run(ap.parse_args().cadence)


if __name__ == "__main__":
    main()

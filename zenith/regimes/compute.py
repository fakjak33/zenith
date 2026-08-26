"""REGIMES compute orchestrator.

  python -m zenith.regimes.compute --action auto        (nightly path)
  python -m zenith.regimes.compute --action force       (ignore all TTLs, refetch everything)
  python -m zenith.regimes.compute --skip-phase2 --skip-phase3   (fast Phase-1-only smoke test)

Not trading-day-gated (macro releases land on weekends/holidays too, and the
classifier is monthly-cadence regardless of when it runs) — the Action's own
cron (Mon-Sat) is the only cadence control. Idempotent: a same-day rerun
overwrites today's journal row rather than duplicating it (history.py).

Pipeline: macro.build_panel() -> classify.classify_timeline() (the full
monthly reconstruction) -> classify.current() (this month's call +
explainability) -> momentum.regime_momentum() -> dimensions.latest_summary()
(the six secondary dimensions) -> history.segments()/transitions() -> Phase 2
(transition.build_tables/changes/crossasset/performance/analogs/accuracy) ->
Phase 3 (themes/scenarios/alerts) -> write every artifact + append the
journal. Phase 2/3 segments are wrapped individually in try/except and each
reported in status.json — a failure in, say, the AI theme's price pull must
never take down the core regime classification, which is the one thing every
other artifact and the TODAY-tab badge depends on.
"""

from __future__ import annotations

import argparse
import math
from datetime import date

from . import DISCLAIMER, save, load
from . import macro, classify, dimensions, momentum, history
from . import transition, changes, crossasset, performance, analogs, accuracy
from . import themes, scenarios, alerts as alerts_mod


def _scrub(obj):
    """Recursively replace non-finite floats with None — json.dumps emits
    bare NaN/Infinity, which is not valid JSON. Same helper every Zenith
    feature package that serializes pandas-derived floats carries (see
    mom/compute.py's identical function)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def run_auto(force: bool = False, skip_phase2: bool = False, skip_phase3: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []

    z_df, raw_df, fetch_status = macro.build_panel(force=force)
    status.append({"segment": "fetch", **fetch_status})

    timeline = classify.classify_timeline(z_df)
    status.append({"segment": "classify", "ok": not timeline.empty, "n_months": len(timeline)})

    current = classify.current(z_df, raw_df)
    mtm_series = None
    if timeline.empty:
        mtm = {"score": None, "narrative": "No data yet."}
    else:
        mtm = momentum.regime_momentum(timeline["growth_z"], timeline["infl_z"], current.get("regime"))
    current["momentum"] = mtm

    secondary = dimensions.latest_summary(z_df, raw_df, min_coverage=3)

    segs = history.segments(timeline)
    trans = history.transitions(segs)

    save("current", _scrub({"as_of": today.isoformat(), "disclaimer": DISCLAIMER, **current}))
    save("dimensions", _scrub({"as_of": today.isoformat(), "dimensions": secondary}))
    save("timeline", _scrub({"as_of": today.isoformat(), "n_months": len(timeline),
                             "months": history.timeline_to_records(timeline),
                             "segments": segs, "transitions": trans}), indent=None)

    change_score = {"score": None, "band": None}
    perf_tables = {"asset": {}, "by_regime": {}}
    if not timeline.empty and not skip_phase2:
        # ---------------------------------------------------------- transitions
        mtm_series = momentum.momentum_series(timeline["growth_z"], timeline["infl_z"], timeline["declared_regime"])
        tables = transition.build_tables(timeline, mtm_series)
        trans_for_current = transition.for_current(tables, current.get("regime"), mtm.get("score"))
        save("transitions", _scrub({"as_of": today.isoformat(), "tables": tables,
                                   "current": trans_for_current}), indent=None)
        status.append({"segment": "transitions", "ok": True})

        # ------------------------------------------------------------- changes
        deltas = changes.indicator_deltas(z_df)
        change_score = changes.regime_change_score(z_df)
        save("changes", _scrub({"as_of": today.isoformat(), "indicators": deltas,
                               "regime_change_score": change_score}), indent=None)
        status.append({"segment": "changes", "ok": True, "score": change_score.get("score")})

        # ---------------------------------------------------------- cross-asset
        try:
            trends = crossasset.context_trends()
            breadth = crossasset.sector_breadth()
            g_rising = current.get("growth", {}).get("rising")
            i_rising = current.get("inflation", {}).get("rising")
            conf = crossasset.confirmation(g_rising, i_rising, trends, breadth)
            div = crossasset.divergences(g_rising, i_rising, trends, breadth)
            save("crossasset", _scrub({"as_of": today.isoformat(), "trends": trends, "breadth": breadth,
                                      "confirmation": conf, "divergences": div}))
            status.append({"segment": "crossasset", "ok": True, "n_divergences": len(div)})
        except Exception as e:
            status.append({"segment": "crossasset", "ok": False, "error": str(e)[:200]})

        # --------------------------------------------------------- performance
        try:
            asset_perf = performance.build_performance_table(timeline["declared_regime"], universe="asset")
            factor_perf = performance.build_performance_table(timeline["declared_regime"], universe="factor")
            perf_tables = {"asset": asset_perf, "factor": factor_perf}
            save("performance", _scrub({"as_of": today.isoformat(), **perf_tables}), indent=None)
            status.append({"segment": "performance", "ok": True, "n_asset": len(asset_perf),
                           "n_factor": len(factor_perf)})
        except Exception as e:
            status.append({"segment": "performance", "ok": False, "error": str(e)[:200]})

        # -------------------------------------------------------------- analogs
        try:
            analog_doc = analogs.build(z_df)
            save("analogs", _scrub({"as_of": today.isoformat(), **analog_doc}))
            status.append({"segment": "analogs", "ok": True, "n": len(analog_doc.get("analogs", []))})
        except Exception as e:
            status.append({"segment": "analogs", "ok": False, "error": str(e)[:200]})

        # ------------------------------------------------------------- accuracy
        try:
            acc = accuracy.build(timeline["declared_regime"], tables)
            save("accuracy", _scrub({"as_of": today.isoformat(), **acc}))
            status.append({"segment": "accuracy", "ok": acc.get("available", False)})
        except Exception as e:
            status.append({"segment": "accuracy", "ok": False, "error": str(e)[:200]})

    if not timeline.empty and not skip_phase3:
        # --------------------------------------------------------------- themes
        try:
            theme_doc = themes.build(z_df)
            save("themes", _scrub({"as_of": today.isoformat(), "themes": theme_doc}))
            status.append({"segment": "themes", "ok": True})
        except Exception as e:
            status.append({"segment": "themes", "ok": False, "error": str(e)[:200]})

        # ------------------------------------------------------------ scenarios
        try:
            g_rising = current.get("growth", {}).get("rising")
            i_rising = current.get("inflation", {}).get("rising")
            scen_doc = scenarios.build(g_rising, i_rising, perf_tables.get("asset", {}))
            save("scenarios", _scrub({"as_of": today.isoformat(), **scen_doc}))
            status.append({"segment": "scenarios", "ok": True})
        except Exception as e:
            status.append({"segment": "scenarios", "ok": False, "error": str(e)[:200]})

    # ------------------------------------------------------------------ journal
    n_journal = history.append_journal({
        "regime": current.get("regime"), "raw_regime": current.get("raw_regime"),
        "confidence": current.get("confidence"), "transitioning": current.get("transitioning"),
        "momentum_score": mtm.get("score"), "change_score": change_score.get("score"),
    }, today)
    status.append({"segment": "journal", "ok": True, "n_rows_this_year": n_journal})

    # ------------------------------------------------------------------- alerts
    if not skip_phase3:
        try:
            journal_rows = history.load_journal(start_year=today.year - 1)
            triggered = alerts_mod.evaluate(journal_rows, current, change_score)
            save("alerts", _scrub({"as_of": today.isoformat(), "alerts": triggered}))
            status.append({"segment": "alerts", "ok": True, "n_triggered": len(triggered)})
        except Exception as e:
            status.append({"segment": "alerts", "ok": False, "error": str(e)[:200]})

    save("status", {"date": today.isoformat(), "disclaimer": DISCLAIMER, "segments": status})
    print(f"[regimes] {today} regime={current.get('regime')} confidence={current.get('confidence')} "
         f"months={len(timeline)} fetched={fetch_status.get('fetched')} "
         f"reused={fetch_status.get('reused')} change_score={change_score.get('score')}")
    return {"ok": True, "regime": current.get("regime")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "force"])
    ap.add_argument("--skip-phase2", action="store_true", help="skip transitions/changes/crossasset/"
                    "performance/analogs/accuracy (useful for a fast Phase-1-only smoke test)")
    ap.add_argument("--skip-phase3", action="store_true", help="skip themes/scenarios/alerts")
    args = ap.parse_args()
    run_auto(force=(args.action == "force"), skip_phase2=args.skip_phase2, skip_phase3=args.skip_phase3)


if __name__ == "__main__":
    main()

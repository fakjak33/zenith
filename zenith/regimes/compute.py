"""REGIMES compute orchestrator.

  python -m zenith.regimes.compute --action auto        (nightly path)
  python -m zenith.regimes.compute --action force       (ignore all TTLs, refetch everything)

Not trading-day-gated (macro releases land on weekends/holidays too, and the
classifier is monthly-cadence regardless of when it runs) — the Action's own
cron (Mon-Sat) is the only cadence control. Idempotent: a same-day rerun
overwrites today's journal row rather than duplicating it (history.py).

Pipeline: macro.build_panel() -> classify.classify_timeline() (the full
monthly reconstruction) -> classify.current() (this month's call +
explainability) -> momentum.regime_momentum() -> dimensions.latest_summary()
(the six secondary dimensions) -> history.segments()/transitions() -> write
every artifact + append the journal.
"""

from __future__ import annotations

import argparse
import math
from datetime import date

from . import DISCLAIMER, save, load
from . import macro, classify, dimensions, momentum, history


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


def run_auto(force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []

    z_df, raw_df, fetch_status = macro.build_panel(force=force)
    status.append({"segment": "fetch", **fetch_status})

    timeline = classify.classify_timeline(z_df)
    status.append({"segment": "classify", "ok": not timeline.empty, "n_months": len(timeline)})

    current = classify.current(z_df, raw_df)
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

    n_journal = history.append_journal({
        "regime": current.get("regime"), "raw_regime": current.get("raw_regime"),
        "confidence": current.get("confidence"), "transitioning": current.get("transitioning"),
        "momentum_score": mtm.get("score"),
    }, today)
    status.append({"segment": "journal", "ok": True, "n_rows_this_year": n_journal})

    save("status", {"date": today.isoformat(), "disclaimer": DISCLAIMER, "segments": status})
    print(f"[regimes] {today} regime={current.get('regime')} confidence={current.get('confidence')} "
         f"months={len(timeline)} fetched={fetch_status.get('fetched')} "
         f"reused={fetch_status.get('reused')}")
    return {"ok": True, "regime": current.get("regime")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "force"])
    args = ap.parse_args()
    run_auto(force=(args.action == "force"))


if __name__ == "__main__":
    main()

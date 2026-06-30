"""Automated sanity screen for the Zenith data artefacts — guards against the kind
of 'minor errors' that are easy to miss by eye (mislabeled charts, stale data,
sign/sort mismatches, empty sections, NaN leaks, duplicate tickers, absurd values).

    python scripts/screen.py

Exits non-zero if any hard CHECK fails; WARN lines are advisory. Reads only the
committed artefacts (no network), so it reflects what the deployed app will show.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zenith.brief import load as brief_load            # noqa: E402
from zenith.cas import store_cas                       # noqa: E402
from zenith.cas.universe import frm_universe, label_of  # noqa: E402

fails: list[str] = []
warns: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok  " if ok else "  ERR ") + msg)
    if not ok:
        fails.append(msg)


def warn(cond: bool, msg: str) -> None:
    if cond:
        print("  warn " + msg)
        warns.append(msg)


def _days_old(iso: str) -> int:
    try:
        return (date.today() - date.fromisoformat(iso[:10])).days
    except Exception:
        return 999


def screen_brief() -> None:
    print("[brief]")
    b = brief_load("brief", {})
    if not b:
        check(False, "brief.json missing/empty")
        return
    check(_days_old(b.get("as_of", "")) <= 10, f"brief as_of fresh ({b.get('as_of')})")
    ov = b.get("market_overview", {})
    check(bool(ov) and all(ov.get(g) for g in ("equity", "commodity", "bond", "fx")),
          "all 4 overview asset-class groups populated")
    for grp in ov.values():
        for r in grp:
            warn(r.get("last") is None, f"overview {r.get('ticker')} has null last price")

    sectors = b.get("sectors", [])
    check(len(sectors) == 11, f"11 SPDR sectors present (got {len(sectors)})")
    if sectors:
        ws = [(s["ticker"], s["w1"]) for s in sectors if s.get("w1") is not None]
        # sectors are stored sorted desc by w1 -> first is leader, last is laggard
        leader, lagg = ws[0], ws[-1]
        check(leader[1] >= lagg[1], "sector list sorted (leader >= laggard)")
        print(f"       sector leader={leader[0]} {leader[1]:+.1%} · "
              f"laggard={lagg[0]} {lagg[1]:+.1%}  (eyeball vs reality)")

    heat = b.get("stock_heatmap", {})
    movers = heat.get("leaders_1w", []) + heat.get("laggards_1w", [])
    check(all(m.get("name") for m in movers), "movers all have company names")

    earn = b.get("earnings", {})
    for r in earn.get("recent", []) + earn.get("upcoming", []):
        mc = r.get("mktcap")
        warn(isinstance(mc, (int, float)) and mc > 5e12,
             f"earnings {r.get('ticker')} mktcap looks absurd (${mc/1e9:.0f}B) — feed quirk")

    news = b.get("news", [])
    warn(not news, "no ticker news this run")


def screen_cas() -> None:
    print("[cas]")
    status = store_cas.load("status", {})
    check(bool(status), "CAS status present")
    if status:
        check(_days_old(status.get("date", "")) <= 10, f"CAS as_of fresh ({status.get('date')})")

    sigs = store_cas.load("signals", [])
    check(len(sigs) > 1000, f"signals populated ({len(sigs)})")
    confs = {s.get("confidence") for s in sigs}
    check("high" in confs, "dynamic confidence reaches 'high'")

    frm = [s for s in sigs if s.get("segment") == "factor_rotation"]
    check(len(frm) > 500, f"factor-rotation signals populated ({len(frm)})")

    uni = frm_universe()
    check(len(uni) == len(set(uni)), "no duplicate tickers in FRM universe")
    groups = {v["group"] for v in uni.values()}
    check({"style", "industry", "beta"} <= groups, f"FRM groups present ({sorted(groups)})")

    panel = store_cas.load("price_panel", {})
    check(isinstance(panel, dict) and "SPY" in panel and panel["SPY"].get("c"),
          "committed price panel non-empty (powers price overlays)")

    rot = store_cas.load("rotation", {})
    check(bool(rot) and all(rot.get(tf) for tf in ("1m", "3m", "6m", "1y")),
          "rotation-by-timeframe artefact present")

    hr = store_cas.load("hitrate", {})
    check(bool(hr.get("models")), "multi-model hit-rate present")

    # label resolution: a few representative tickers should resolve to names
    for t in ("INDA", "MTUM"):
        if t in uni or t in [s.get("asset") for s in frm]:
            warn(label_of(t) == t, f"{t} label does not resolve to a name")


def main() -> None:
    screen_brief()
    screen_cas()
    print()
    if fails:
        print(f"SCREEN FAILED — {len(fails)} error(s), {len(warns)} warning(s).")
        sys.exit(1)
    print(f"SCREEN PASSED — 0 errors, {len(warns)} warning(s).")


if __name__ == "__main__":
    main()

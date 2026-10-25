"""Phase 04 entry point: fetch (cached), compute, write data/vault/repro/<name>.json.

  .venv/Scripts/python -m zenith.vault.repro.run [--only vrp_spx strangle_mgmt ...]
                                                 [--refresh] [--sample 400]
"""

from __future__ import annotations

import argparse
import json
import time

from . import save
from . import data as D

ALL = ["vrp_spx", "cboe_short_vol", "cboe_tail_hedge", "strangle_mgmt", "earnings_drift",
       "expected_move", "vix_spikes"]


def _market(refresh: bool) -> dict:
    m = D.spx_vix_rf(refresh)
    return {k: v.loc["1990-01-01":] if k != "sptr" else v for k, v in m.items()}


def run_vrp(m):
    from . import vrp
    return vrp.build(vrp.compute(m["vix"], m["spx"]))


def run_cboe(m, refresh):
    from . import cboe
    names = cboe.SHORT_VOL + cboe.TAIL
    ix = {n: D.cboe_index(n, refresh) for n in names}
    res = cboe.compute(ix, m["sptr"], D.yahoo_close("^IRX", refresh) / 100.0, names)
    pput = res["stats"][("1988", "PPUT")]["cagr"]
    mix = cboe.matched_mix(pput, m["sptr"].loc["1988-01-04":cboe.END], D.yahoo_close("^IRX") / 100.0)
    return cboe.build_short_vol(res), cboe.build_tail_hedge(res, mix)


def run_strangle(m):
    from . import strangle
    return strangle.build(strangle.compute(m["spx"], m["vix"], m["rf"]))


def run_earnings(refresh: bool, n: int):
    import pandas as pd
    from . import earnings
    from ...pretom import universe as uni
    from ...pead import load as pead_load
    rows, _ = uni.russell1000()
    tickers = earnings.sample_universe(rows, n)
    spy = D.daily_panel(["SPY"], refresh)["SPY"]["close"]
    px = D.daily_panel(tickers, refresh)
    evs = []
    for k, t in enumerate(tickers):
        if t not in px:
            continue
        dates = D.earnings_dates(t, refresh)
        evs += earnings.ticker_events(px[t], spy, dates, t)
        if k % 50 == 0:
            print(f"[repro] earnings {k}/{len(tickers)} events={len(evs)}", flush=True)
    res = earnings.summarize(pd.DataFrame(evs))
    hist = pead_load("history", [])
    tracker = earnings.tracker_stats(hist if isinstance(hist, list) else hist.get("rows", []))
    return earnings.build(res, tracker)


def run_expected_move(m):
    from . import expected_move
    return expected_move.build(expected_move.compute(m["spx"], m["vix"]))


def run_vix_spikes(m):
    from . import vix_spikes
    return vix_spikes.build(vix_spikes.compute(m["spx"], m["vix"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sample", type=int, default=400)
    a = ap.parse_args()
    m = _market(a.refresh)
    for name in a.only:
        t0 = time.time()
        if name == "vrp_spx":
            outs = [run_vrp(m)]
        elif name in ("cboe_short_vol", "cboe_tail_hedge"):
            outs = [o for o in run_cboe(m, a.refresh) if o["name"] == name]
        elif name == "strangle_mgmt":
            outs = [run_strangle(m)]
        elif name == "earnings_drift":
            outs = [run_earnings(a.refresh, a.sample)]
        elif name == "expected_move":
            outs = [run_expected_move(m)]
        elif name == "vix_spikes":
            outs = [run_vix_spikes(m)]
        else:
            raise SystemExit(f"unknown repro {name}")
        for o in outs:
            save(o["name"], o)
            print(f"[repro] {o['name']}: {o['verdict']} ({time.time() - t0:.0f}s)\n  {o['our_value']}",
                  flush=True)
    print(json.dumps({"written": a.only, "at": D.stamp()}))


if __name__ == "__main__":
    main()

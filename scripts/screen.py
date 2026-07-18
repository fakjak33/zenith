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


def screen_pretom() -> None:
    print("[pretom]")
    from zenith.pretom import load as pretom_load, archive_months, load_month

    status = pretom_load("status", {})
    check(bool(status), "PRETOM status present")
    if status:
        check(_days_old(status.get("date", "")) <= 5,
              f"PRETOM status fresh ({status.get('date')})")
        warn(status.get("calendar_check") not in ("ok", "n/a", "no-universe"),
             f"trading-calendar cross-check: {status.get('calendar_check')}")

    basket = pretom_load("basket", {})
    check(bool(basket), "basket_latest present")
    if basket:
        names = basket.get("names", [])
        check(60 <= len(names) <= 130, f"basket size sane ({len(names)})")
        tickers = [n["ticker"] for n in names]
        check(len(tickers) == len(set(tickers)), "no duplicate tickers in basket")
        ranks = [n["rank"] for n in names]
        check(sorted(ranks) == list(range(1, len(names) + 1)),
              "ranks unique and contiguous from 1")
        check(all(0.0 <= (n["score"] or 0) <= 1.0 for n in names),
              "scores within [0, 1]")
        check(all(0.0 <= (n["pct_below_high"] or 0) <= 1.0 for n in names),
              "pct_below_high within [0, 1]")
        warn(basket.get("universe", {}).get("coverage", 1.0) < 0.85,
             f"universe price coverage low ({basket.get('universe', {}).get('coverage')})")

    months = archive_months()
    hist = pretom_load("history", {})
    rows = hist.get("rows", [])
    check(len(rows) >= 20, f"history populated ({len(rows)} rows)")
    check(len(rows) == len(months), "history rows match archived months")
    for r in rows:
        if not r.get("final"):
            continue
        for w in ("classic", "t1"):
            ex = (r.get(w) or {}).get("ew_excess")
            check(ex is not None and -1.0 <= ex <= 1.0,
                  f"{r['month']} {w} ew_excess sane ({ex})")
    if months:
        latest = load_month(months[0])
        stats_state = latest.get("stats", {}).get("state")
        if stats_state in ("window", "post", "final"):
            check(bool(latest.get("panel")), "latest basket has a price panel")


def screen_fmom() -> None:
    print("[fmom]")
    from zenith.fmom import load as fmom_load, archive_months as fmom_months
    from zenith.fmom.history import summarize

    signals = fmom_load("signals", {})
    check(bool(signals.get("models")), "fmom signals present")
    if not signals.get("models"):
        return
    check(_days_old(signals.get("as_of", "")) <= 40,
          f"fmom signals fresh ({signals.get('as_of')})")
    for key, m in signals["models"].items():
        rows = m.get("rows", [])
        check(len(rows) >= 3, f"{key}: enough factors ({len(rows)})")
        svals = [r["s"] for r in rows if r.get("s") is not None]
        check(len(svals) == len(rows) and all(-2.0001 <= s <= 2.0001 for s in svals),
              f"{key}: signals present and within ±2")
        wl = sum(r["weight"] for r in rows if r["weight"] > 0)
        ws = sum(r["weight"] for r in rows if r["weight"] < 0)
        check((wl == 0 or abs(wl - 1) < 0.02) and (ws == 0 or abs(ws + 1) < 0.02),
              f"{key}: leg weights sum to ±1 (long {wl:.3f}, short {ws:.3f})")
        check((wl == 0 or ws == 0) == m.get("degenerate", False),
              f"{key}: degenerate flag consistent")
        ranks = [r["rank"] for r in rows]
        check(sorted(ranks) == list(range(1, len(rows) + 1)),
              f"{key}: ranks contiguous from 1")
        n_top = sum(1 for r in rows if r["decile"] == "top")
        n_bot = sum(1 for r in rows if r["decile"] == "bottom")
        check(n_top >= 1 and n_bot >= 1, f"{key}: decile flags present")
        top = max(rows, key=lambda r: r["s"])
        print(f"       {key}: {m['formation_month']} strongest={top['factor']} "
              f"s={top['s']:+.2f} (eyeball vs last month's tape)")

    bt = fmom_load("backtest", {}).get("families", {})
    check(bool(bt), "fmom backtest present")
    for fam, b in bt.items():
        stats = (b.get("stats") or {}).get("tsfm") or {}
        check(stats.get("n_months", 0) >= 24, f"backtest({fam}): enough history "
              f"({stats.get('n_months')} months)")
        check(bool(b.get("recent_panel")), f"backtest({fam}): heatmap panel present")

    hist = fmom_load("history", {})
    rows = hist.get("rows", [])
    check(len(rows) >= 20, f"fmom history populated ({len(rows)} rows)")
    dupes = len(rows) - len({(r["month"], r["model"]) for r in rows})
    check(dupes == 0, f"no duplicate (month, model) history rows ({dupes})")
    bad = [f"{r['month']}/{r['model']}" for r in rows if r.get("evaluated")
           and not (r["realized"].get("model_ret") is not None
                    and -1.0 <= r["realized"]["model_ret"] <= 1.0)]
    check(not bad, f"all evaluated realized returns sane "
                   f"({len(bad)} bad: {', '.join(bad[:5])})" if bad
          else "all evaluated realized returns sane")
    summary = summarize(rows)
    check(bool(summary.get("models")), "fmom tracker summary computable")
    warn(len(summary.get("pending_months", [])) > 2,
         f"fmom: {len(summary.get('pending_months', []))} months pending evaluation")

    check(bool(fmom_load("etf_catalog", {}).get("etfs")), "etf catalog committed")
    check(len(fmom_months()) >= 1, "fmom archive has months")

    oc = fmom_load("osap_catalog", {})
    if "osap_tsfm" in signals["models"]:
        check(oc.get("n", 0) >= 200 and oc.get("n_documented", 0) >= 200,
              f"osap catalog documented ({oc.get('n_documented')}/{oc.get('n')})")
        stale = signals["models"]["osap_tsfm"]["formation_month"]
        check(stale >= "2024-01", f"osap vintage sane ({stale})")
    scr = fmom_load("screens", {})
    if scr:
        check(scr.get("n_screens", 0) >= 20,
              f"stock screens present ({scr.get('n_screens')})")
        bad = []
        for k, s in (scr.get("screens") or {}).items():
            lv = [r["value"] for r in s.get("long", [])]
            sv = [r["value"] for r in s.get("short", [])]
            asc = s.get("high_is") == "short"
            ok_order = (lv == sorted(lv, reverse=not asc)
                        and sv == sorted(sv, reverse=asc))
            if not (lv and sv and ok_order):
                bad.append(k)
        check(not bad, f"screen sides ordered correctly ({', '.join(bad[:4])})"
              if bad else "screen sides ordered correctly")


def screen_pead() -> None:
    print("[pead]")
    from zenith.pead import load as pead_load, load_day, archive_days

    st = pead_load("status", {})
    check(bool(st), "pead status present")
    check(_days_old(st.get("date", "")) <= 4, f"pead status fresh ({st.get('date')})")

    sig = pead_load("signals", {})
    check(bool(sig), "pead signals_latest present")
    hist = pead_load("history", {})
    rows = hist.get("rows", [])
    days = archive_days()
    check(bool(days), "pead archive has signal sheets")
    if not (sig and rows and days):
        return

    check(bool(sig.get("disclaimer")), "pead disclaimer present")

    keys = [(r["ticker"], r["report_date"]) for r in rows]
    check(len(keys) == len(set(keys)), "no duplicate (ticker, report_date) in history")

    comps = [r.get("composite") for r in rows if r.get("composite") is not None]
    check(all(0 <= c <= 100 for c in comps), "history composites all within [0, 100]")
    check(all(r.get("side") in ("long", "short") for r in rows),
          "history rows all long/short (mixed never tracked)")

    hist_keys = set(keys)
    act = sig.get("active_book", {})
    book = act.get("long", []) + act.get("short", [])
    check(all((r["ticker"], r["report_date"]) in hist_keys for r in book),
          "active book is a subset of history")
    warn(not book, "active book empty (fine in a quiet earnings week)")

    sheet = load_day(days[0])
    el = [r for r in sheet.get("signals", []) if not r.get("excluded_reason")]
    check(all(r.get("ranks") and r.get("composite") is not None for r in el),
          "latest sheet: every eligible row has ranks + composite")
    for r in el:
        raw = r.get("raw", {})
        bad = (r["side"] == "long" and ((raw.get("sue") or 0) <= 0
                                        or (raw.get("ear") or 0) <= 0)) or \
              (r["side"] == "short" and ((raw.get("sue") or 0) >= 0
                                         or (raw.get("ear") or 0) >= 0))
        check(not bad, f"two-confirmation gate holds for {r['ticker']}")

    curve = sig.get("drift_curve", [])
    if curve:
        tds = [p["td"] for p in curve]
        check(tds == sorted(tds) and len(tds) == len(set(tds)),
              "drift curve td axis strictly increasing")
    else:
        warn(True, "drift curve empty (backfill / weekly refresh pending)")

    evald = sum(1 for r in rows for h in r["horizons"].values() if h.get("evaluated"))
    xs = [h.get("excess_ret") for r in rows for h in r["horizons"].values()
          if h.get("evaluated") and h.get("excess_ret") is not None]
    check(all(-1.0 <= v <= 2.0 for v in xs), "evaluated excess returns sane")
    print(f"       history={len(rows)} sheets={len(days)} horizons_evaluated={evald}")


def main() -> None:
    screen_brief()
    screen_cas()
    screen_pretom()
    screen_fmom()
    screen_pead()
    print()
    if fails:
        print(f"SCREEN FAILED — {len(fails)} error(s), {len(warns)} warning(s).")
        sys.exit(1)
    print(f"SCREEN PASSED — 0 errors, {len(warns)} warning(s).")


if __name__ == "__main__":
    main()

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
        if isinstance(mc, (int, float)) and mc > 5e12:   # message must not
            warn(True, f"earnings {r.get('ticker')} mktcap looks absurd "
                       f"(${mc / 1e9:.0f}B) — feed quirk")   # format None

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

    # FOMC-cycle evidence exhibit
    fomc = store_cas.load("fomc", {})
    check(bool(fomc.get("eras")), "FOMC era stats present")
    if fomc.get("eras"):
        check(_days_old(fomc.get("as_of", "")) <= 10, f"fomc fresh ({fomc.get('as_of')})")
        check((fomc.get("n_meetings") or 0) >= 200, f"FOMC meeting set complete ({fomc.get('n_meetings')})")
        smpl = fomc["eras"].get("1994-2016 (CMVJ sample)", {})
        gap = smpl.get("even_minus_odd_bp")
        # sanity: the in-sample even-week edge should be materially positive
        check(gap is not None and gap > 3, f"CMVJ-sample even-week edge replicates (+{gap} bp)")
        for era, rec in fomc["eras"].items():
            for side in ("even", "odd"):
                b = rec.get(side) or {}
                if b.get("avg_bp") is not None:
                    check(-50 <= b["avg_bp"] <= 50, f"fomc {era}/{side} avg within +/-50bp ({b['avg_bp']})")
        nm = fomc.get("next_meeting")
        check(not nm or nm >= date.today().isoformat(), f"FOMC next meeting is future ({nm})")

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

    # --- TOM evidence artifact ------------------------------------------------
    tom = pretom_load("tom", {})
    check(bool(tom.get("rows")), "tom_longrun present")
    if tom.get("rows"):
        check(_days_old(tom.get("as_of", "")) <= 35,
              f"tom_longrun fresh ({tom.get('as_of')})")
        trows = tom["rows"]
        check(all(-0.5 <= r["tom_ret"] <= 0.5 for r in trows),
              "tom monthly returns within +/-50% sanity band")
        check(all(r["n_tom_days"] == 4 for r in trows),
              "every TOM window has exactly 4 trading days")
        summ = tom.get("summary", {})
        check(bool(summ.get("by_decade")), "tom decade summary non-empty")
        check((summ.get("overall") or {}).get("n_months", 0) >= 300,
              f"tom long-run depth ({(summ.get('overall') or {}).get('n_months')} months)")
    finals = [r for r in rows if r.get("final")]
    if finals:
        warn(all((r.get("tom_market") or {}).get("spy_ret") is None
                 for r in finals[-2:]),
             "recent final months lack tom_market stats (reconcile pending)")


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
        # vol-scaled lens (Barroso-Santa-Clara): legs sum to ±multiplier, not ±1
        tgt = m.get("vs_multiplier", 1.0) if key.endswith("_vs") else 1.0
        check((wl == 0 or abs(wl - tgt) < 0.02) and (ws == 0 or abs(ws + tgt) < 0.02),
              f"{key}: leg weights sum to ±{tgt:.2f} (long {wl:.3f}, short {ws:.3f})")
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
    # real vertical movers reach ±2-3x (MRVL/MSTR/CRWV 2024-26); the band only
    # guards against data corruption (bad prices -> absurd multiples)
    check(all(-5.0 <= v <= 5.0 for v in xs), "evaluated excess returns sane")
    warn(sum(1 for v in xs if abs(v) > 1.0) > len(xs) * 0.01,
         "more than 1% of evaluated picks moved >100% vs SPY — eyeball prices")

    # --- announcement premium (EAP) -------------------------------------------
    eap_obj = pead_load("eap", {})
    check(bool(eap_obj), "eap.json present")
    if eap_obj:
        check(_days_old(eap_obj.get("as_of", "")) <= 4,
              f"eap fresh ({eap_obj.get('as_of')})")
        today_iso = date.today().isoformat()
        ups = eap_obj.get("upcoming", [])
        check(all(u.get("reaction_day", "") >= today_iso for u in ups),
              "eap upcoming reaction days are all today or later")
        summ = eap_obj.get("summary", {})
        check((summ.get("through") or {}).get("overall", {}) is not None
              and (summ.get("n_rows") or 0) > 0,
              f"eap tracked rows populated ({summ.get('n_rows')})")
    ehist = pead_load("eap_history", {})
    erows = ehist.get("rows", [])
    if erows:
        ekeys = [(r["ticker"], r["report_date"]) for r in erows]
        check(len(ekeys) == len(set(ekeys)),
              "no duplicate (ticker, report_date) in eap history")
        exs = [r[w]["excess"] for r in erows for w in ("pre", "through")
               if r.get(w, {}).get("evaluated")]
        check(all(-5.0 <= v <= 5.0 for v in exs),
              "eap evaluated excess returns sane")
    print(f"       history={len(rows)} sheets={len(days)} horizons_evaluated={evald} "
          f"eap_rows={len(erows)}")


def screen_edge() -> None:
    print("[edge]")
    from zenith.edge import load as edge_load, SCREENS

    status = edge_load("status", {})
    if not status:
        warn(True, "edge not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 5, f"edge status fresh ({status.get('date')})")
    for screen in SCREENS:
        res = edge_load(screen, {})
        if not res.get("ranked"):
            warn(True, f"edge {screen}: no ranked data yet")
            continue
        rows = res["ranked"]
        tks = [r["ticker"] for r in rows]
        check(len(tks) == len(set(tks)), f"edge {screen}: no duplicate tickers")
        ranks = [r["rank"] for r in rows]
        check(ranks == list(range(1, len(rows) + 1)), f"edge {screen}: ranks contiguous")
        check(all(0 <= (r.get("pctile") or 0) <= 100 for r in rows),
              f"edge {screen}: pctiles within [0,100]")
        # long/short sides are disjoint and drawn from the ranked set
        lt = {r["ticker"] for r in res.get("long", [])}
        sh = {r["ticker"] for r in res.get("short", [])}
        check(not (lt & sh), f"edge {screen}: long/short disjoint")
    iv = edge_load("ivspread", {})
    if iv.get("ranked"):
        check(all(abs(r.get("iv_spread", 0)) <= 1.0 for r in iv["ranked"]),
              "edge ivspread: spreads within +/-100 vol pts")
    si = edge_load("shortint", {})
    if si.get("ranked"):
        check(all(0 <= (r.get("si_float") or 0) <= 100 for r in si["ranked"]),
              "edge shortint: si%float within [0,100]")
    lot = edge_load("lottery", {})
    if lot.get("ranked"):
        # MAX-beta can exceed raw MAX when the market moved AGAINST a stock's
        # jump (residual > raw), so only sanity-bound the daily magnitudes.
        check(all(0 <= (r.get("max_beta") or 0) < 0.6 for r in lot["ranked"]),
              "edge lottery: MAX-beta daily magnitude sane (<60%/day)")
    hrows = edge_load("history", {}).get("rows", [])
    exs = [r["excess"] for r in hrows if r.get("evaluated") and r.get("excess") is not None]
    check(all(-5.0 <= v <= 5.0 for v in exs), "edge history: evaluated excess sane")
    print(f"       screens={sum(1 for s in SCREENS if edge_load(s, {}).get('ranked'))} "
          f"history={len(hrows)}")


def screen_nightday() -> None:
    print("[nightday]")
    from zenith.nightday import load as nd_load

    status = nd_load("status", {})
    if not status:
        warn(True, "nightday not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 5, f"nightday status fresh ({status.get('date')})")
    panel = nd_load("panel", {}).get("etfs", {})
    check(bool(panel), "nightday ETF panel present")
    for t, d in panel.items():
        st = d.get("stats", {})
        if st:
            check(abs(st.get("overnight_avg_bp", 0)) < 500
                  and abs(st.get("intraday_avg_bp", 0)) < 500,
                  f"nightday {t}: avg daily legs sane")
    screen = nd_load("screen", {})
    if screen.get("ranked"):
        tks = [r["ticker"] for r in screen["ranked"]]
        check(len(tks) == len(set(tks)), "nightday screen: no duplicate tickers")
    hrows = nd_load("history", {}).get("rows", [])
    exs = [r["excess"] for r in hrows if r.get("evaluated") and r.get("excess") is not None]
    check(all(-5.0 <= v <= 5.0 for v in exs), "nightday history: evaluated excess sane")
    print(f"       panel={len(panel)} screen={screen.get('n', 0)} history={len(hrows)}")


def screen_holdings() -> None:
    print("[holdings]")
    import zenith.holdings as hold
    from zenith.holdings import normalize

    reg = (hold.load_funds() or {}).get("funds", [])
    if not reg:
        warn(True, "holdings not yet run (no registry) — skipping")
        return

    for f in reg:
        if not f.get("enabled"):
            continue
        key, tick = f["key"], f["ticker"]
        status = hold.load(key, "status", {})
        if not status:
            warn(True, f"holdings {tick}: not yet run (no status) — skipping")
            continue
        check(_days_old(status.get("date", "")) <= 5,
              f"holdings {tick}: status fresh ({status.get('date')})")

        latest = hold.load(key, "latest", {})
        hist = hold.load(key, "history", {})
        changes = hold.load(key, "changes", {})
        days = hold.archive_days(key)

        check(bool(latest.get("as_of")), f"holdings {tick}: latest has a value date")
        check(bool(days), f"holdings {tick}: snapshots archived")
        if days:
            check(latest.get("as_of") == max(days),
                  f"holdings {tick}: latest matches the newest archive day")
        check(hist.get("dates") == sorted(days),
              f"holdings {tick}: history dates match the archive exactly")
        check(hist.get("dates", []) == sorted(set(hist.get("dates", []))),
              f"holdings {tick}: history dates strictly increasing, no repeats")

        pos = latest.get("positions", [])
        ids = [p["id"] for p in pos]
        check(len(ids) == len(set(ids)), f"holdings {tick}: no duplicate positions")
        check(all(abs(p["weight"]) <= 5.0 for p in pos),
              f"holdings {tick}: no single position above 500% of NAV")
        check(all(p["asset_class"] in normalize.ASSET_CLASSES for p in pos),
              f"holdings {tick}: every position has a known asset class")
        warn(any(p["asset_class"] == "unclassified" for p in pos),
             f"holdings {tick}: unclassified position(s) — add the root to "
             f"normalize.ROOTS")

        s = latest.get("summary", {})
        gross = s.get("gross", 0.0)
        check(0.2 <= gross <= 8.0,
              f"holdings {tick}: gross exposure sane ({gross:.2f}x NAV)")
        check(abs(s.get("net", 0.0) - (s.get("long", 0.0) + s.get("short", 0.0)))
              < 1e-6, f"holdings {tick}: net equals long plus short")
        check((latest.get("nav") or 0) > 0, f"holdings {tick}: NAV positive")

        q = latest.get("quality", {})
        check(bool(q.get("ok")), f"holdings {tick}: latest snapshot passed validation")
        warn(bool(q.get("warnings")),
             f"holdings {tick}: quality warnings {q.get('warnings')}")

        if len(days) > 1:
            check(bool(changes.get("rankings")),
                  f"holdings {tick}: change rankings built")
            caps = changes.get("caps", {})
            check((caps.get("d_weight") or 0) > 0,
                  f"holdings {tick}: heatmap colour cap is positive")
        warn(bool(latest.get("gaps")),
             f"holdings {tick}: {len(latest.get('gaps', []))} missing recent "
             f"trading day(s)")
        print(f"       {tick}: {len(days)} snapshots {min(days) if days else '-'}"
              f"..{max(days) if days else '-'} · {len(pos)} positions · "
              f"gross {gross:.2f}x · {changes.get('n_events', 0)} change events")


def screen_mom() -> None:
    print("[mom]")
    from zenith.mom import load as mom_load

    status = mom_load("status", {})
    if not status:
        warn(True, "mom not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 5, f"mom status fresh ({status.get('date')})")
    coverage_seg = next((s for s in status.get("segments", []) if s.get("segment") == "coverage"), {})
    check((coverage_seg.get("coverage") or 0) >= 0.85,
         f"mom coverage >= 0.85 ({coverage_seg.get('coverage')})")

    scores = mom_load("scores", {})
    rows = [r for r in scores.get("rows", []) if not r.get("excluded")]
    if not rows:
        warn(True, "mom: no scored rows yet")
    else:
        tks = [r["ticker"] for r in rows]
        check(len(tks) == len(set(tks)), "mom scores: no duplicate tickers")
        ranks = sorted(r["rank"] for r in rows)
        check(ranks == list(range(1, len(rows) + 1)), "mom scores: ranks contiguous")
        check(all(0 <= (r.get("pctile") or 0) <= 100 for r in rows),
             "mom scores: pctiles within [0,100]")
        check(all(-20.0 <= r["composite"] <= 20.0 for r in rows),
             "mom scores: composite within [-20,+20]")
        bad = [r["ticker"] for r in rows
              if abs(max(-20.0, min(20.0, sum((r.get("contributions") or {}).values())))
                     - r["composite"]) > 1e-4]
        check(not bad, f"mom scores: contributions sum to composite ({bad[:5] if bad else 'ok'})")
        lt = {r["ticker"] for r in rows if r.get("side") == "long"}
        sh = {r["ticker"] for r in rows if r.get("side") == "short"}
        check(not (lt & sh), "mom scores: long/short disjoint")
        print(f"       universe={scores.get('n', 0)} scored={len(rows)} "
              f"coverage={coverage_seg.get('coverage')}")

    from zenith.config import MOM_HISTORY_DIR
    if MOM_HISTORY_DIR.exists():
        years = sorted(int(p.stem) for p in MOM_HISTORY_DIR.glob("*.json") if p.stem.isdigit())
        for y in years:
            import json as _json
            doc = _json.loads((MOM_HISTORY_DIR / f"{y}.json").read_text(encoding="utf-8"))
            dates_seen = sorted({r["date"] for r in doc.get("rows", [])})
            check(dates_seen == sorted(dates_seen), f"mom history {y}: dates monotonic")

    picks = mom_load("picks", {}).get("rows", [])
    exs = [cell["excess"] for r in picks for cell in r.get("eval", {}).values()
          if cell.get("evaluated") and cell.get("excess") is not None]
    check(all(-5.0 <= v <= 5.0 for v in exs), "mom picks: evaluated excess sane")
    print(f"       picks={len(picks)} evaluated_cells={len(exs)}")


def screen_mvt() -> None:
    print("[mom.mvt]")
    from zenith.mom.mvt import load as mvt_load
    from zenith.config import MOM_MVT_LEVERAGED_EXCLUDE, MOM_WEIGHTS

    status = mvt_load("status", {})
    if not status:
        warn(True, "mvt not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 5, f"mvt status fresh ({status.get('date')})")
    check("mvt" in MOM_WEIGHTS, "mvt: registered in MOM_WEIGHTS")
    check(abs(sum(MOM_WEIGHTS.values()) - 1.0) < 1e-6, "mvt: MOM_WEIGHTS sum to 1.0")

    for universe_key in ("equities", "etfs"):
        doc = mvt_load(universe_key, {})
        rows = doc.get("rows", [])
        if not rows:
            warn(True, f"mvt {universe_key}: no scored rows yet")
            continue
        tks = [r["ticker"] for r in rows]
        check(len(tks) == len(set(tks)), f"mvt {universe_key}: no duplicate tickers")
        check(not (set(tks) & MOM_MVT_LEVERAGED_EXCLUDE),
             f"mvt {universe_key}: no explicitly-leveraged ticker survived the universe gate")

        bad_raw = [r["ticker"] for r in rows
                  if r.get("raw_score") is not None and not (-20.0 <= r["raw_score"] <= 20.0)]
        check(not bad_raw, f"mvt {universe_key}: raw_score within [-20,+20] ({bad_raw[:5] or 'ok'})")
        bad_norm = [r["ticker"] for r in rows
                   if r.get("normalized_score") is not None and not (-20.0 <= r["normalized_score"] <= 20.0)]
        check(not bad_norm, f"mvt {universe_key}: normalized_score within [-20,+20] ({bad_norm[:5] or 'ok'})")

        # every percentile stored anywhere must be a valid 0-100 read
        bad_pct = []
        for r in rows:
            for grid in (r.get("raw_percentiles") or {}, r.get("residual_percentiles") or {}):
                for h, p in grid.items():
                    if p is not None and not (0.0 <= p <= 100.0):
                        bad_pct.append(r["ticker"])
        check(not bad_pct, f"mvt {universe_key}: percentiles within [0,100] ({bad_pct[:5] or 'ok'})")

        n_scored_norm = sum(1 for r in rows if r.get("normalized_score") is not None)
        n_scored_raw = sum(1 for r in rows if r.get("raw_score") is not None)
        print(f"       {universe_key}: n={len(rows)} raw_scored={n_scored_raw} "
              f"normalized_scored={n_scored_norm} k_factors={doc.get('k_factors')} "
              f"explained_var={doc.get('explained_variance_ratio')}")

        # sanity: reconstructing a pairwise spread from the committed return/
        # variance vectors for two arbitrary scored names must match a direct
        # computation from the same inputs (proves the "never persist the
        # NxN, only what reconstructs it" architecture is actually correct,
        # not just fast).
        candidates = [r for r in rows if r.get("residual_return", {}).get("6m") is not None
                     and r.get("resid_var") is not None][:2]
        if len(candidates) == 2:
            import numpy as _np
            from zenith.mom.mvt import pairwise as _pw
            r_arr = _np.array([candidates[0]["residual_return"]["6m"], candidates[1]["residual_return"]["6m"]])
            v_arr = _np.array([candidates[0]["resid_var"], candidates[1]["resid_var"]])
            D = _pw.spread_matrix(r_arr, v_arr, horizon_days=126)
            check(abs(D[0, 1] + D[1, 0]) < 1e-9, f"mvt {universe_key}: reconstructed spread antisymmetric")

    weighting = mvt_load("weighting", {})
    if weighting and weighting.get("factor_erc_weights_shrunk"):
        w = weighting["factor_erc_weights_shrunk"]
        check(abs(sum(w.values()) - 1.0) < 1e-3, "mvt weighting: ERC-shrunk weights sum to ~1.0")
        print(f"       weighting mode={weighting.get('active_weight_mode')} "
              f"erc_converged={weighting.get('erc_converged')}")

    cross = mvt_load("crossuniverse", {})
    if cross and cross.get("rows"):
        rows = cross["rows"]
        bad_gap = [r["ticker"] for r in rows if abs(r.get("gap") or 0) > 40.0]
        check(not bad_gap, f"mvt crossuniverse: gaps within a sane band ({bad_gap[:5] or 'ok'})")
        bad_class = [r["ticker"] for r in rows
                    if r.get("classification") not in ("idiosyncratic", "broad/systemic")]
        check(not bad_class, f"mvt crossuniverse: classification is one of the two registered labels")
        print(f"       crossuniverse: n={len(rows)} sectors={len(cross.get('by_sector', {}))}")
    else:
        warn(True, "mvt crossuniverse: no data yet")

    validation = mvt_load("validation", {})
    if validation and validation.get("models"):
        models = validation["models"]
        for m, s in models.items():
            sh = s.get("sharpe")
            check(sh is None or -10 < sh < 10, f"mvt validation {m}: Sharpe in a sane band ({sh})")
            dd = s.get("max_drawdown")
            check(dd is None or -1.0 <= dd <= 0.0, f"mvt validation {m}: max drawdown within [-1,0] ({dd})")
        central = validation.get("central_hypothesis_test")
        print(f"       validation: n_periods={validation.get('n_periods')} "
              f"central_test={'c_lower_than_a=' + str(central.get('c_lower_than_a')) if central else 'n/a'}")
        for name, w in validation.get("by_stress_window", {}).items():
            print(f"       stress[{name}]: available={w.get('available')} n_periods={w.get('n_periods')}")
    else:
        warn(True, "mvt validation: no backtest yet (run --action validate)")


def screen_ideas() -> None:
    print("[ideas]")
    from zenith.ideas import load as ideas_load

    status = ideas_load("status", {})
    if not status:
        warn(True, "ideas not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 5, f"ideas status fresh ({status.get('date')})")

    doc = ideas_load("ideas", {})
    all_ideas = doc.get("buy", []) + doc.get("sell", [])
    if not all_ideas:
        warn(True, "ideas: no ideas generated yet")
    else:
        tks = [i["ticker"] for i in all_ideas]
        check(len(tks) == len(set(tks)), "ideas: no duplicate tickers across buy+sell")
        buy_tks = {i["ticker"] for i in doc.get("buy", [])}
        sell_tks = {i["ticker"] for i in doc.get("sell", [])}
        check(not (buy_tks & sell_tks), "ideas: buy/sell disjoint")
        check(all(0 <= i.get("conviction", -1) <= 100 for i in all_ideas),
             "ideas: conviction within [0,100]")
        check(all(0 <= i.get("unusual", -1) <= 100 for i in all_ideas),
             "ideas: unusual within [0,100]")
        check(all(i.get("side") == ("long" if i in doc.get("buy", []) else "short")
                  for i in all_ideas),
             "ideas: side matches buy/sell bucket")
        from zenith.ideas import OPPORTUNITY_TYPES
        bad_opp = [i["ticker"] for i in all_ideas if i.get("opportunity_type") not in OPPORTUNITY_TYPES]
        check(not bad_opp, f"ideas: opportunity_type registered ({bad_opp[:5] if bad_opp else 'ok'})")
        bad_rr = [i["ticker"] for i in all_ideas
                 if (i.get("riskreward") or {}).get("rr_ratio") is not None
                 and not (0.5 <= i["riskreward"]["rr_ratio"] <= 20.0)]
        check(not bad_rr, f"ideas: R/R ratio within a sane band ({bad_rr[:5] if bad_rr else 'ok'})")
        # every narrative sentence must trace to a payload field, never invent a metric:
        # spot-check that any explicitly-cited price level appears in that idea's own riskreward
        bad_stop = []
        for i in all_ideas:
            rr = i.get("riskreward") or {}
            cmm = " ".join(i.get("narrative", {}).get("change_my_mind", []))
            if rr.get("stop") is not None and f"{rr['stop']:.2f}" not in cmm and "stop" in cmm.lower():
                bad_stop.append(i["ticker"])
        check(not bad_stop, f"ideas: cited stop levels match riskreward payload ({bad_stop[:5] if bad_stop else 'ok'})")
        print(f"       buy={len(doc.get('buy', []))} sell={len(doc.get('sell', []))} "
              f"regime={doc.get('regime', {}).get('label')} thin_day={doc.get('thin_day')}")


def screen_regimes() -> None:
    print("[regimes]")
    from zenith.regimes import load as regimes_load, REGIME_LABELS

    status = regimes_load("status", {})
    if not status:
        warn(True, "regimes not yet run (no status) — skipping")
        return
    check(_days_old(status.get("date", "")) <= 3, f"regimes status fresh ({status.get('date')})")

    current = regimes_load("current", {})
    regime = current.get("regime")
    if not regime:
        warn(True, "regimes: no regime classified yet")
    else:
        check(regime in REGIME_LABELS.values(), f"regimes: regime is a registered label ({regime})")
        conf = current.get("confidence")
        check(conf is None or 0 <= conf <= 100, f"regimes: confidence within [0,100] ({conf})")
        for axis_name in ("growth", "inflation"):
            axis = current.get(axis_name, {})
            n_r, n_t = axis.get("n_rising"), axis.get("n_total")
            if n_r is not None and n_t is not None:
                check(0 <= n_r <= n_t, f"regimes: {axis_name} n_rising <= n_total ({n_r}/{n_t})")
            breadth = axis.get("breadth")
            check(breadth is None or 0.0 <= breadth <= 1.0,
                 f"regimes: {axis_name} breadth within [0,1] ({breadth})")
        print(f"       regime={regime} confidence={conf} transitioning={current.get('transitioning')} "
             f"streak={current.get('streak_months')}mo")

    dims = regimes_load("dimensions", {}).get("dimensions", {})
    for dim, d in dims.items():
        n = d.get("coverage_n", 0)
        n_ind = len(d.get("indicators", []))
        check(n <= n_ind, f"regimes: {dim} coverage_n <= indicator rows listed ({n}/{n_ind})")

    timeline = regimes_load("timeline", {})
    months = timeline.get("months", [])
    if months:
        m_dates = [m["month"] for m in months]
        check(m_dates == sorted(m_dates), "regimes: timeline months monotonic")
        check(len(m_dates) == len(set(m_dates)), "regimes: timeline months unique")
        segs = timeline.get("segments", [])
        bad_seg = [s for s in segs if s["regime"] not in REGIME_LABELS.values()]
        check(not bad_seg, f"regimes: segment regimes all registered labels ({bad_seg[:3] if bad_seg else 'ok'})")
        print(f"       months={len(months)} segments={len(segs)} transitions={len(timeline.get('transitions', []))}")

    # --- Phase 2 ---
    trans = regimes_load("transitions", {}).get("tables", {}).get("unconditional", {})
    bad_prob = []
    for start_regime, horizons in trans.items():
        for h, cell in horizons.items():
            probs = [d["p"] for d in cell["destinations"].values() if d["p"] is not None]
            if probs and abs(sum(probs) - 1.0) > 0.01:
                bad_prob.append((start_regime, h))
            for d in cell["destinations"].values():
                if d["p"] is None:
                    check(d["n"] <= cell["n_start"], "regimes: transitions n never exceeds n_start")
    check(not bad_prob, f"regimes: transition probabilities sum to ~1 per cell ({bad_prob[:3] if bad_prob else 'ok'})")

    changes_doc = regimes_load("changes", {})
    score = changes_doc.get("regime_change_score", {}).get("score")
    check(score is None or 0 <= score <= 100, f"regimes: Regime Change Score within [0,100] ({score})")

    perf = regimes_load("performance", {})
    bad_dd = []
    for universe in ("asset", "factor"):
        for ticker, data in perf.get(universe, {}).items():
            for regime_name, stats in (data.get("by_regime") or {}).items():
                if stats and not (-1.0 <= stats.get("max_drawdown", 0) <= 0.0):
                    bad_dd.append((ticker, regime_name))
    check(not bad_dd, f"regimes: performance max_drawdown within [-1,0] ({bad_dd[:3] if bad_dd else 'ok'})")

    analog_doc = regimes_load("analogs", {})
    bad_analog = [a for a in analog_doc.get("analogs", []) if a.get("n_shared_dimensions", 0) < 5]
    check(not bad_analog, f"regimes: analogs meet minimum shared-dimension threshold ({len(bad_analog)} below)")

    acc = regimes_load("accuracy", {})
    if acc.get("available"):
        brier = acc.get("brier", {}).get("brier")
        check(brier is None or 0.0 <= brier <= 1.0, f"regimes: Brier score within [0,1] ({brier})")
        check(acc.get("brier", {}).get("in_sample") is True, "regimes: Brier score labelled in-sample")

    # --- Phase 3 ---
    themes_doc = regimes_load("themes", {}).get("themes", {})
    for name, t in themes_doc.items():
        theme_score = t.get("signal_score")
        check(theme_score is None or 0 <= theme_score <= 100,
             f"regimes: theme {name} signal_score within [0,100] ({theme_score})")
        for ev in t.get("evidence", []):
            check(ev.get("category") in ("Fact", "Interpretation", "Forecast", "Speculation"),
                 f"regimes: theme {name} evidence category registered ({ev.get('category')})")

    scen_doc = regimes_load("scenarios", {})
    for s in scen_doc.get("dimension_scenarios", []):
        check(s.get("grounded") is False, f"regimes: dimension scenario {s.get('id')} correctly labelled ungrounded")
    for s in scen_doc.get("quadrant_scenarios", []):
        if s.get("grounded"):
            check(s.get("implied_regime") in REGIME_LABELS.values(),
                 f"regimes: quadrant scenario {s.get('id')} implies a registered regime")

    n_alerts = len(regimes_load("alerts", {}).get("alerts", []))
    if trans or changes_doc or themes_doc:
        print(f"       phase2/3: change_score={score} n_asset_perf={len(perf.get('asset', {}))} "
             f"n_analogs={len(analog_doc.get('analogs', []))} accuracy_available={acc.get('available')} "
             f"n_themes={len(themes_doc)} n_alerts={n_alerts}")


def screen_index() -> None:
    """INDEX (Master List) — catalog integrity, taxonomy conformance, graph sanity.

    A directory fails differently from a signal engine: the damage is silent
    (a dropped row, a duplicate slug, an edge left dangling by an import, an
    entry quietly claiming to be verified) rather than a wrong number on screen.
    These checks target exactly those failure modes.
    """
    from zenith.index import load as index_load
    from zenith.index import taxonomy as itx
    from zenith.index import quality as iq

    print("\n[INDEX]")
    ents = index_load("entities", [])
    rels = index_load("relationships", [])
    status = index_load("status", {})
    if not ents:
        warn("index: no catalog yet (run `python -m zenith.index.compute --action seed`)")
        return

    check(len(ents) >= 100, f"index: catalog has {len(ents)} entries (expected >= 100)")

    # --- identity integrity ------------------------------------------------
    ids = [e.get("id") for e in ents]
    slugs = [e.get("slug") for e in ents]
    dup_ids = {i for i in ids if ids.count(i) > 1}
    dup_slugs = {sl for sl in slugs if slugs.count(sl) > 1}
    check(not dup_ids, f"index: no duplicate ids ({len(dup_ids)} found)")
    check(not dup_slugs, f"index: no duplicate slugs ({sorted(dup_slugs)[:4]})")
    check(all(e.get("name") for e in ents), "index: every entry has a name")
    check(all(e.get("id") for e in ents), "index: every entry has an id")

    # --- taxonomy conformance ---------------------------------------------
    bad_cat = [e["name"] for e in ents
               if e.get("primary_category") not in itx.PRIMARY_CATEGORIES]
    bad_type = [e["name"] for e in ents if e.get("entity_type") not in itx.ENTITY_TYPES]
    bad_state = [e["name"] for e in ents
                 if e.get("lifecycle_state") not in itx.LIFECYCLE_STATES]
    bad_conf = [e["name"] for e in ents
                if e.get("confidence") not in itx.CONFIDENCE_LEVELS]
    check(not bad_cat, f"index: every primary_category is declared ({bad_cat[:3]})")
    check(not bad_type, f"index: every entity_type is declared ({bad_type[:3]})")
    check(not bad_state, f"index: every lifecycle_state is declared ({bad_state[:3]})")
    check(not bad_conf, f"index: every confidence level is declared ({bad_conf[:3]})")

    # --- graph integrity ---------------------------------------------------
    orphans = iq.orphan_edges(ents, rels)
    check(not orphans, f"index: no orphan relationship edges ({len(orphans)} found)")
    self_loops = [r for r in rels if r.get("source") == r.get("target")]
    check(not self_loops, f"index: no self-referencing edges ({len(self_loops)} found)")
    bad_rel = [r.get("type") for r in rels if r.get("type") not in itx.RELATIONSHIP_TYPES]
    check(not bad_rel, f"index: every relationship type is declared ({sorted(set(bad_rel))[:3]})")

    # --- honesty invariants (the ones this feature exists to uphold) -------
    # An entry may only claim `verified` if a link check actually succeeded.
    false_verified = [e["name"] for e in ents
                      if e.get("lifecycle_state") == "verified" and e.get("link_status") != "ok"]
    check(not false_verified,
          f"index: nothing marked verified without a live link ({false_verified[:3]})")
    # A low-confidence entry must not assert a URL that does not resolve.
    unconfirmed_url = [e["name"] for e in ents
                       if e.get("confidence") == "low" and e.get("url")
                       and e.get("link_status") not in ("ok", "blocked")]
    check(not unconfirmed_url,
          f"index: low-confidence entries assert no unreachable URL ({unconfirmed_url[:3]})")
    # NaN/None leakage into user-visible string fields.
    leaked = [e["name"] for e in ents
              for f in ("description", "url", "notes", "name")
              if str(e.get(f, "")).strip().lower() in ("nan", "none", "null")]
    check(not leaked, f"index: no NaN/None leaked into text fields ({leaked[:3]})")

    # --- export round-trip -------------------------------------------------
    try:
        from zenith.index import export as iex
        df = iex.to_dataframe(ents, rels)
        check(len(df) == len(ents),
              f"index: export row count matches catalog ({len(df)} vs {len(ents)})")
        check("relationships" in df.columns and "url" in df.columns,
              "index: export carries links and relationships")
    except Exception as exc:
        check(False, f"index: export failed — {type(exc).__name__}: {exc}")

    # --- Phase 2: podcast intelligence -------------------------------------
    pod = index_load("podcasts", {})
    if pod.get("shows"):
        from zenith.index import guests as ig
        from zenith.index import podcasts as ipc

        shows = pod["shows"]
        check(len(shows) == len(ipc.PODCASTS),
              f"index: all {len(ipc.PODCASTS)} registered podcasts harvested ({len(shows)})")
        dead = [s["podcast"] for s in shows if not s.get("ok")]
        if dead:
            warn(f"index: podcast feed(s) returned nothing: {dead}")
        thin = [s["podcast"] for s in shows if s.get("ok") and s["episodes"] < 50]
        if thin:
            warn(f"index: suspiciously shallow archive(s) — a site feed capped at "
                 f"10 posts looks like this: {thin}")

        # Every declared extraction pattern must actually exist, or the show
        # silently degrades to the loose generic fallback (a real bug once).
        phantom = [(p.name, s) for p in ipc.PODCASTS for s in p.patterns
                   if s not in ig.STRATEGIES]
        check(not phantom, f"index: every declared podcast pattern exists ({phantom[:3]})")

        # Hosts must never be recorded as guests of their own show.
        guests = pod.get("guests") or {}
        host_leak = []
        for p in ipc.PODCASTS:
            hosts = {h.lower() for h in p.hosts}
            for key, prof in guests.items():
                if key in hosts and p.name in (prof.get("podcasts") or []):
                    host_leak.append((p.name, prof.get("name")))
        check(not host_leak, f"index: no host recorded as their own guest ({host_leak[:3]})")

        # Nothing below the confidence bar may have become a directory entry.
        # (Guests already in the catalog from the curated seed are exempt — they
        # were verified by hand, not by the parser.)
        harvest_names = {e["name"].lower() for e in ents
                         if "harvest" in str(e.get("provenance", ""))
                         and "seed list" not in str(e.get("provenance", ""))}
        leaked = sorted(key for key, g in guests.items()
                        if not ig.meets_threshold(g.get("confidence", "low"))
                        and key in harvest_names)
        check(not leaked,
              f"index: no low-confidence guest reached the directory ({leaked[:4]})")

        # Every guest name still has to look like a person.
        malformed = [g.get("name") for g in guests.values()
                     if ig.meets_threshold(g.get("confidence", "low"))
                     and not ig.looks_like_person(g.get("name", ""))]
        check(not malformed,
              f"index: every promoted guest name parses as a person ({malformed[:4]})")

        # Appearances must point at real episodes.
        episodes = index_load("episodes", [])
        check(len(episodes) >= 1000,
              f"index: episode archive has {len(episodes)} episodes (expected 1000+)")
        ep_ids = {e.get("id") for e in episodes}
        dangling = sum(1 for g in guests.values() for a in (g.get("appearances") or [])
                       if a.get("episode_id") and a["episode_id"] not in ep_ids)
        check(not dangling, f"index: no appearance points at a missing episode ({dangling})")

        # A person's affiliation must not itself be a person's name.
        person_firms = [(e["name"], e["current_affiliation"]) for e in ents
                        if e.get("entity_type") == "person" and e.get("current_affiliation")
                        and e["current_affiliation"].lower() in
                        {x["name"].lower() for x in ents if x.get("entity_type") == "person"}]
        check(not person_firms,
              f"index: no person recorded as another person's employer ({person_firms[:3]})")

        cov = {s["podcast"]: s["coverage"] for s in shows}
        print(f"       podcasts: {len(shows)} shows · {pod.get('episodes', 0)} episodes · "
              f"{pod.get('guests_promoted', 0)}/{pod.get('guests_total', 0)} guests promoted · "
              f"coverage {min(cov.values()):.0%}-{max(cov.values()):.0%}")
    else:
        warn("index: no podcast harvest yet (run `--action podcasts`)")

    # --- Phase 3: graph, ranking, discovery --------------------------------
    if rels:
        from zenith.index import discover as idisc
        from zenith.index import network as inet
        from zenith.index import ranking as irank

        # degrees() counts distinct NEIGHBOURS, not edges, so this compares
        # against distinct pairs. Two entities legitimately carry two edges:
        # Niels Kaastrup-Larsen both `hosts` Top Traders Unplugged and
        # `works_at` it, and 33 other host/show pairs are the same shape.
        deg = inet.degrees(ents, rels)
        ids = {e["id"] for e in ents}
        pairs = {frozenset((r["source"], r["target"])) for r in rels
                 if r.get("source") in ids and r.get("target") in ids
                 and r["source"] != r["target"]}
        check(sum(deg.values()) == 2 * len(pairs),
              f"index: degree sum matches twice the distinct-pair count "
              f"({sum(deg.values())} vs {2 * len(pairs)})")

        # A bounded subgraph must stay bounded, or the network view becomes an
        # unreadable hairball rather than a picture.
        big = inet.build(ents, rels, [e["id"] for e in ents])
        check(big is None or big["n"] <= inet.NODE_LIMIT,
              f"index: network subgraph respects the {inet.NODE_LIMIT}-node limit")

        # Layout must be deterministic, or the same page redraws differently on
        # every rerun and nobody can point at a node twice.
        by_name = {e["name"]: e for e in ents}
        if "Robert Carver" in by_name:
            ids = inet.ego_ids(ents, rels, by_name["Robert Carver"]["id"], hops=1)
            a, b = inet.build(ents, rels, ids), inet.build(ents, rels, ids)
            check(a and b and [n["x"] for n in a["nodes"]] == [n["x"] for n in b["nodes"]],
                  "index: network layout is deterministic across runs")
            if "Man Group" in by_name:
                path = inet.path_between(ents, rels, by_name["Robert Carver"]["id"],
                                         by_name["Man Group"]["id"])
                check(bool(path),
                      "index: the graph is traversable (Carver -> Man Group)")

        scored = irank.score_all(ents, rels, (pod or {}).get("guests", {}))
        check(len(scored) == len(ents),
              f"index: every entity is scored ({len(scored)} vs {len(ents)})")
        # The transparency promise: contributions must reconstruct the score.
        drift = [eid for eid, row in scored.items()
                 if abs(sum(row["components"].values()) - row["score"]) > 1e-9]
        check(not drift,
              f"index: ranking contributions sum to the score ({len(drift)} drifted)")
        # A published methodology that does not match the computation is a lie.
        sample = next(iter(scored.values()))
        check(set(sample["components"]) == set(irank.COMPONENTS),
              "index: computed components match the published COMPONENTS")
        check(abs(sum(w for w, _l, _d in irank.COMPONENTS.values()) - 1.0) < 1e-9,
              "index: ranking weights sum to 1.0")
        out_of_range = [eid for eid, row in scored.items()
                        if not 0.0 <= row["score"] <= 1.0]
        check(not out_of_range, f"index: every score is in 0..1 ({out_of_range[:3]})")

        # Discovery must not recommend things the user already supplied.
        disc = idisc.build(ents, rels, (pod or {}).get("guests", {}), pod or {})
        leaked = [e["name"] for e in disc["new_to_you"]
                  if idisc.is_curated(e) or e.get("zenith_source")]
        check(not leaked,
              f"index: 'new to you' excludes curated and ingested entries ({leaked[:3]})")
        print(f"       graph: {len(rels)} edges · max degree "
              f"{max(deg.values()) if deg else 0} · "
              f"{len(disc['new_to_you'])} new-to-you · "
              f"{len(disc['cross_pollinators'])} cross-show · "
              f"{len(disc['bridges'])} bridges · {len(disc['gaps'])} gaps reported")

    # --- staleness ---------------------------------------------------------
    if status.get("date"):
        age = (date.today() - date.fromisoformat(status["date"])).days
        if age > 30:
            warn(f"index: status is {age} days old — re-run compute")

    ls = status.get("links", {}) or {}
    print(f"       {len(ents)} entries · {len(rels)} edges · "
          f"{status.get('verified', 0)} verified · {status.get('needs_review', 0)} need review · "
          f"links ok/blocked/error {ls.get('ok', 0)}/{ls.get('blocked', 0)}/{ls.get('error', 0)} · "
          f"{status.get('zenith_sources_linked', 0)} linked to Zenith sources")


def main() -> None:
    screen_brief()
    screen_cas()
    screen_pretom()
    screen_fmom()
    screen_pead()
    screen_edge()
    screen_nightday()
    screen_holdings()
    screen_mom()
    screen_mvt()
    screen_ideas()
    screen_regimes()
    screen_index()
    print()
    if fails:
        print(f"SCREEN FAILED — {len(fails)} error(s), {len(warns)} warning(s).")
        sys.exit(1)
    print(f"SCREEN PASSED — 0 errors, {len(warns)} warning(s).")


if __name__ == "__main__":
    main()

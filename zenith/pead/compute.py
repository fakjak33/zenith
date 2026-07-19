"""PEAD compute orchestrator.

  python -m zenith.pead.compute --action auto|backfill [--quarters N]

auto (the daily weekday cron path) self-gates on the trading calendar:
  * non-trading day -> cheap no-op (status only)
  * otherwise scan the last 5 trading days for reaction days that have no
    archived signal sheet yet (normally just today; the backscan self-heals
    missed crons and sweeps Friday-after-hours/weekend reporters on Monday),
    build + archive each sheet, append eligible picks to the history, then
    evaluate matured horizons and refresh the active book.
backfill seeds ~N quarters of past signals from per-ticker yfinance earnings
history (current Russell 1000 membership — survivorship caveat recorded).
Dispatch-only; resumable via week-long per-ticker caches.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

from . import DISCLAIMER, save, load, archive_day, archive_days, load_day
from . import eap
from . import earnings as earn
from . import history as hist
from . import signals as sig
from ..pretom import calendar as cal
from ..pretom import universe as uni
from ..cas.sources import prices

BACKSCAN_TD = 5             # reaction days healed per run
RECENT_SHEET_DAYS = 5       # archive days surfaced in signals_latest
DRIFT_REFRESH_WEEKDAY = 0   # Monday: recompute the (price-heavy) drift curve


def _fetch_prices(tickers: list[str], period: str, status: list[dict],
                  label: str) -> dict:
    px, st = prices.get_history(tickers, period=period)
    missing = [t for t in tickers if t not in px]
    if len(missing) > max(5, len(tickers) // 20):     # retry once on a bad pull
        time.sleep(20)
        px2, _ = prices.get_history(missing, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px),
                   "requested": len(tickers), "error": st.get("error", "")})
    return px


def _pending_reaction_days(today: date, lookback: int = BACKSCAN_TD) -> list[date]:
    """Trading days in the backscan window without an archived sheet yet."""
    done = set(archive_days())
    start = today - timedelta(days=lookback * 2 + 4)
    win = [d for d in cal.trading_days(start, today)][-lookback:]
    return [d for d in win if d.isoformat() not in done]


def _collect_reporters(r_day: date, universe_by_ticker: dict,
                       status: list[dict]) -> list[dict]:
    """Nasdaq reporters mapping onto reaction day r_day, R1000 only, EPS
    resolved through the source ladder, deduped per ticker."""
    rows: dict[str, dict] = {}
    for c_day in earn.calendar_dates_for_reaction_day(r_day):
        day_rows = earn.nasdaq_day(c_day)
        status.append({"segment": f"nasdaq({c_day})",
                       "ok": bool(day_rows) or not cal.is_trading_day(c_day),
                       "n": len(day_rows)})
        for rep in day_rows:
            t = rep["ticker"]
            if t not in universe_by_ticker:
                continue
            if rep["time"] not in ("time-pre-market", "time-after-hours"):
                rep = {**rep, "time": earn.infer_time(t, rep["report_date"])
                       or rep["time"], "time_inferred": True}
            if sig.reaction_day(date.fromisoformat(rep["report_date"]),
                                rep["time"]) != r_day:
                continue
            old = rows.get(t)
            if old is None or (old.get("eps_actual") is None
                               and rep.get("eps_actual") is not None):
                rows[t] = rep
    return [earn.resolve_eps(r) for r in rows.values()]


def _load_pool() -> dict:
    return load("signals", {}).get("rank_pool") or {"days": {}}


def _eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("excluded_reason") is None]


def _sheet_obj(r_day: date, rows: list[dict], n_reporters: int,
               backfilled: bool) -> dict:
    el = _eligible(rows)
    return {"day": r_day.isoformat(), "built_at": date.today().isoformat(),
            "backfilled": backfilled, "n_reporters": n_reporters,
            "n_eligible": len(el),
            "n_long": sum(1 for r in el if r["side"] == "long"),
            "n_short": sum(1 for r in el if r["side"] == "short"),
            "signals": rows}


def _refresh_latest(pool: dict, status: list[dict], drift: list | None) -> dict:
    h = load("history", {"rows": []})
    rows = h.get("rows", [])
    act = hist.active_rows(rows)
    prev = load("signals", {})
    days = archive_days()
    recent, signals = [], []
    for d in days[:10]:
        sheet = load_day(d)
        recent.append({k: sheet.get(k) for k in
                       ("day", "n_reporters", "n_eligible", "n_long", "n_short")})
        if d in days[:RECENT_SHEET_DAYS]:
            signals.extend(sheet.get("signals", []))
    out = {
        "as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
        "recent_days": recent,
        "signals": signals,
        "active_book": {
            "long": [r for r in act if r["side"] == "long"],
            "short": [r for r in act if r["side"] == "short"],
        },
        "summary": hist.summarize(rows),
        "drift_curve": drift if drift is not None else prev.get("drift_curve", []),
        "drift_curve_as_of": (date.today().isoformat() if drift is not None
                              else prev.get("drift_curve_as_of")),
        "rank_pool": pool,
    }
    save("signals", out)
    return out


def _refresh_eap(upcoming_reps: list[dict], today: date,
                 status: list[dict]) -> None:
    """Write the EAP artifact: upcoming reporters + tracked-premium summary."""
    try:
        rows = eap.load("eap_history", {"rows": []}).get("rows", [])
        out = {"as_of": today.isoformat(), "disclaimer": DISCLAIMER,
               "survivorship_note": eap.SURVIVORSHIP_NOTE,
               "upcoming": eap.build_upcoming(upcoming_reps, rows, today),
               "summary": eap.summarize(rows)}
        save("eap", out)
        status.append({"segment": "eap_refresh", "ok": True,
                       "n": len(out["upcoming"]),
                       "tracked": out["summary"]["n_rows"]})
    except Exception as e:
        status.append({"segment": "eap_refresh", "ok": False,
                       "error": str(e)[:200]})


def _seed_eap_from_archives(status: list[dict]) -> int:
    """EAP rows for every reporter in every archived signal sheet (the premium
    is unconditional, so excluded/mixed reporters count too). Idempotent."""
    new = []
    for dstr in archive_days():
        sheet = load_day(dstr)
        for s in sheet.get("signals", []):
            try:
                new.append(eap.make_row({
                    "ticker": s["ticker"], "name": s.get("name", ""),
                    "report_date": s["report_date"],
                    "time": s.get("time", ""),
                    "mktcap": (s.get("context") or {}).get("mktcap"),
                }, backfilled=sheet.get("backfilled", False)))
            except (KeyError, ValueError):
                continue
    added = eap.append_rows(new)
    status.append({"segment": "eap_seed", "ok": True, "n": added,
                   "scanned": len(new)})
    return added


def run_auto(force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []

    if not cal.is_trading_day(today) and not force:
        out = {"date": today.isoformat(), "is_trading_day": False,
               "segments": [{"segment": "gate", "ok": True,
                             "note": "non-trading day"}]}
        save("status", out)
        print(f"[pead] {today} non-trading day — no-op")
        return out

    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    ubt = {u["ticker"]: u for u in universe}

    pending = _pending_reaction_days(today)
    h = load("history", {"rows": []})
    active_tk = sorted({r["ticker"] for r in hist.active_rows(h.get("rows", []))})

    # one combined pull: pending days' reporters + active book + EAP + SPY
    reporters_by_day: dict[date, list[dict]] = {}
    for r_day in pending:
        reporters_by_day[r_day] = _collect_reporters(r_day, ubt, status)
    try:
        upcoming_reps = earn.upcoming_calendar(7, ubt)
        status.append({"segment": "upcoming_calendar", "ok": True,
                       "n": len(upcoming_reps)})
    except Exception as e:
        upcoming_reps = []
        status.append({"segment": "upcoming_calendar", "ok": False,
                       "error": str(e)[:200]})
    tickers = sorted({rep["ticker"] for reps in reporters_by_day.values()
                      for rep in reps} | set(active_tk)
                     | set(eap.pending_tickers(today))) + ["SPY"]
    px = _fetch_prices(tickers, "1y", status, "prices") if len(tickers) > 1 else {}
    spy = px.get("SPY")
    spy_close = spy["close"] if spy is not None and not spy.empty else None

    pool = _load_pool()
    if spy_close is not None:
        for r_day in sorted(reporters_by_day):
            try:
                reps = reporters_by_day[r_day]
                revenues = {r["ticker"]: earn.quarterly_revenue(r["ticker"]) for r in reps}
                next_dates = {r["ticker"]: earn.next_earnings_date(r["ticker"], r_day)
                              for r in reps}
                rows = sig.build_signal_sheet(reps, px, spy_close, ubt, pool,
                                              r_day, revenues, next_dates)
                if r_day == today and not any(r["raw"].get("ear") is not None
                                              for r in rows) and rows:
                    status.append({"segment": f"sheet({r_day})", "ok": False,
                                   "error": "no reaction bars yet — retry tomorrow"})
                    continue
                sheet = _sheet_obj(r_day, rows, len(reps), backfilled=False)
                archive_day(r_day.isoformat(), sheet)
                added = hist.append_rows([hist.make_row(r) for r in _eligible(rows)])
                eap.append_rows([eap.make_row(rep) for rep in reps])
                pool = sig.update_pool(pool, rows, r_day)
                status.append({"segment": f"sheet({r_day})", "ok": True,
                               "n": len(rows), "eligible": sheet["n_eligible"],
                               "picks_added": added})
            except Exception as e:
                status.append({"segment": f"sheet({r_day})", "ok": False,
                               "error": str(e)[:200]})

        # evaluate matured horizons + refresh next-earnings on the active book
        try:
            filled = hist.evaluate_pending(px, spy_close, today)
            status.append({"segment": "evaluate", "ok": True, "n": filled})
        except Exception as e:
            status.append({"segment": "evaluate", "ok": False, "error": str(e)[:200]})
        try:
            n_eap = eap.evaluate_pending(px, spy_close, today)
            status.append({"segment": "eap_evaluate", "ok": True, "n": n_eap})
        except Exception as e:
            status.append({"segment": "eap_evaluate", "ok": False,
                           "error": str(e)[:200]})
        try:
            upcoming = {t: earn.next_earnings_date(t, today) for t in active_tk}
            n = hist.refresh_next_earnings(upcoming)
            status.append({"segment": "next_earn_refresh", "ok": True, "n": n})
        except Exception as e:
            status.append({"segment": "next_earn_refresh", "ok": False,
                           "error": str(e)[:200]})
    else:
        status.append({"segment": "prices", "ok": False, "error": "no SPY history"})

    # weekly (Monday) drift-curve refresh — price-heavy, needs all history tickers
    drift = None
    if spy_close is not None and (today.weekday() == DRIFT_REFRESH_WEEKDAY
                                  or not load("signals", {}).get("drift_curve")):
        try:
            h2 = load("history", {"rows": []})
            all_tk = sorted({r["ticker"] for r in h2.get("rows", [])}) + ["SPY"]
            px_all = _fetch_prices(all_tk, "3y", status, "prices(drift)")
            spy_all = px_all.get("SPY")
            if spy_all is not None and not spy_all.empty:
                drift = hist.drift_curve(h2.get("rows", []), px_all, spy_all["close"])
                status.append({"segment": "drift_curve", "ok": True, "n": len(drift)})
        except Exception as e:
            status.append({"segment": "drift_curve", "ok": False, "error": str(e)[:200]})

    _refresh_eap(upcoming_reps, today, status)
    _refresh_latest(pool, status, drift)
    h3 = load("history", {"rows": []})
    n_act = len(hist.active_rows(h3.get("rows", [])))
    calendar_check = "n/a"
    if spy_close is not None:
        calendar_check = "ok" if not cal.verify_against(spy_close.index[-90:]) else "mismatch"
    out = {"date": today.isoformat(), "is_trading_day": True,
           "reaction_days_processed": [d.isoformat() for d in pending],
           "n_history_rows": len(h3.get("rows", [])), "n_active": n_act,
           "universe": {"n": len(universe), "source": uni_status.get("source", "?")},
           "calendar_check": calendar_check, "segments": status}
    save("status", out)

    print(f"[pead] {today} pending={len(pending)} history={len(h3.get('rows', []))} "
          f"active={n_act}")
    for s in status:
        flag = "ok " if s.get("ok") else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', '')}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    return out


def run_backfill(quarters: int = 8, force: bool = False) -> dict:
    """Seed the tracker from per-ticker yfinance earnings history. Chronological
    so every sheet ranks against its own trailing pool. Resumable: per-ticker
    fetches are week-cached and existing archive days are skipped."""
    today = date.today()
    status: list[dict] = []
    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    if not universe:
        print("[pead] backfill aborted: no universe")
        return {"ok": False}
    ubt = {u["ticker"]: u for u in universe}
    start = today - timedelta(days=int(quarters * 95) + 30)

    # 1. per-ticker earnings history (cached; the slow part, ~1000 tickers)
    events: dict[date, list[dict]] = {}
    fetched = 0
    for k, u in enumerate(universe):
        t = u["ticker"]
        rows = earn.earnings_history(t)
        if rows:
            fetched += 1
        earn.polite_sleep(0.15)
        for rrow in rows:
            try:
                rep_d = date.fromisoformat(rrow["date"])
            except ValueError:
                continue
            if not (start <= rep_d < today - timedelta(days=2)):
                continue
            if rrow["eps_actual"] is None or rrow["eps_estimate"] is None:
                continue
            time_str = ("time-pre-market" if (rrow.get("hour") or 17) < 12
                        else "time-after-hours")
            r_day = sig.reaction_day(rep_d, time_str)
            events.setdefault(r_day, []).append({
                "ticker": t, "name": u.get("name", ""),
                "report_date": rep_d.isoformat(), "time": time_str,
                "eps_actual": rrow["eps_actual"],
                "eps_consensus": rrow["eps_estimate"],
                "n_estimates": None, "mktcap": None, "sue_source": "yf"})
        if (k + 1) % 100 == 0:
            print(f"[pead] earnings history {k + 1}/{len(universe)}")
    status.append({"segment": "earnings_history", "ok": fetched > len(universe) // 2,
                   "n": fetched, "requested": len(universe)})

    # 2. one bulk price pull
    px = _fetch_prices([u["ticker"] for u in universe] + ["SPY"], "3y",
                       status, "prices(3y)")
    spy = px.get("SPY")
    if spy is None or spy.empty:
        print("[pead] backfill aborted: no SPY history")
        return {"ok": False}
    spy_close = spy["close"]

    # 3. chronological sheets (pool grows as it goes)
    pool: dict = {"days": {}}
    done = written = 0
    for r_day in sorted(events):
        if load_day(r_day.isoformat()) and not force:
            done += 1
            continue
        reps = events[r_day]
        revenues = {r["ticker"]: earn.quarterly_revenue(r["ticker"]) for r in reps}
        rows = sig.build_signal_sheet(reps, px, spy_close, ubt, pool, r_day,
                                      revenues, {})
        sheet = _sheet_obj(r_day, rows, len(reps), backfilled=True)
        archive_day(r_day.isoformat(), sheet)
        hist.append_rows([hist.make_row(r, backfilled=True) for r in _eligible(rows)])
        pool = sig.update_pool(pool, rows, r_day)
        written += 1
    status.append({"segment": "sheets", "ok": True, "n": written, "skipped": done})

    # 4. next-earnings dates — per row (each pick's OWN next report), from the
    # same cached earnings histories; then evaluate matured horizons
    h = load("history", {"rows": []})
    for r in h.get("rows", []):
        if r.get("next_earn_date") is None:
            nxt = earn.next_earnings_date(r["ticker"],
                                          date.fromisoformat(r["report_date"]))
            if nxt:
                r["next_earn_date"] = nxt
    save("history", h)
    filled = hist.evaluate_pending(px, spy_close, today)
    status.append({"segment": "evaluate", "ok": True, "n": filled})

    # 5. EAP: seed announcement-premium rows from every archived sheet, then
    # evaluate both windows off the same bulk price pull
    _seed_eap_from_archives(status)
    n_eap = eap.evaluate_pending(px, spy_close, today)
    status.append({"segment": "eap_evaluate", "ok": True, "n": n_eap})
    try:
        upcoming_reps = earn.upcoming_calendar(7, ubt)
    except Exception:
        upcoming_reps = []
    _refresh_eap(upcoming_reps, today, status)

    drift = hist.drift_curve(load("history", {"rows": []}).get("rows", []),
                             px, spy_close)
    status.append({"segment": "drift_curve", "ok": bool(drift), "n": len(drift)})
    _refresh_latest(pool, status, drift)

    out = {"ok": True, "sheets_written": written, "horizons_filled": filled}
    print(f"[pead] backfill wrote {written} sheets, filled {filled} horizons")
    for s in status:
        flag = "ok " if s.get("ok") else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', '')}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    save("status", {"date": today.isoformat(), "is_trading_day": cal.is_trading_day(today),
                    "backfill": out, "segments": status})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "backfill"])
    ap.add_argument("--quarters", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.action == "backfill":
        run_backfill(args.quarters, args.force)
    else:
        run_auto(force=args.force)


if __name__ == "__main__":
    main()

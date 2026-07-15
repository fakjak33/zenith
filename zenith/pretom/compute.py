"""PRETOM compute orchestrator.

  python -m zenith.pretom.compute --action auto|backfill [--months N]

auto (the daily weekday cron path) self-gates on the trading calendar:
  * non-trading day                -> cheap no-op (status only)
  * tau-10 .. tau-4, no basket yet -> lock this month's basket (late-lock flag
                                      after tau-10, so one missed cron heals)
  * any non-final archived month   -> reconcile its window/post stats; any one
                                      successful run repairs all pending state
backfill reconstructs the last N completed months from history using current
Russell 1000 membership (survivorship caveat recorded on every row).
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from . import DISCLAIMER, save, archive_month, archive_months, load_month
from . import calendar as cal
from . import analytics as an
from . import universe as uni
from ..cas.sources import prices


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


def _build_month(month: tuple[int, int], lock_date: date, px: dict,
                 universe: list[dict], uni_status: dict,
                 div_flags: dict, backfilled: bool) -> dict:
    y, m = month
    sched = cal.month_schedule(y, m)
    basket = an.build_basket(px, universe, lock_date, div_flags)
    tickers = [b["ticker"] for b in basket]
    coverage = (sum(1 for u in universe if u["ticker"] in px) / len(universe)
                if universe else 0.0)
    k = cal.tau_offset(lock_date)
    return {
        "month": f"{y:04d}-{m:02d}",
        "locked_at": lock_date.isoformat(),
        "locked_late": bool(k is not None and k > cal.LOCK_AT),
        "backfilled": backfilled,
        "tau": sched["tau"].isoformat(),
        "schedule": cal.schedule_iso(sched),
        "universe": {"n": len(universe), "source": uni_status.get("source", "?"),
                     "coverage": round(coverage, 4),
                     "weights_source": uni_status.get("weights_source", "?")},
        "names": basket,
        "stats": {"state": "locked", "updated": date.today().isoformat(),
                  "final": False, "classic": {}, "t1": {}, "post": {}},
        "per_name_marks": {},
        "panel": {},
        "n_basket": len(tickers),
    }


def _update_stats(obj: dict, px: dict, spy, today: date) -> dict:
    """Fill/refresh window + post stats on an archived month in place."""
    s = {k: date.fromisoformat(v) for k, v in obj["schedule"].items()}
    names = obj["names"]
    classic = an.window_stats(names, px, s["win_start"], s["win_end_classic"], spy)
    t1 = an.window_stats(names, px, s["win_start"], s["win_end_t1"], spy)
    post = (an.post_stats(names, px, s["post_start"], s["post_end"], spy,
                          classic.get("ew_excess"))
            if today > s["post_start"] else {})
    obj["stats"] = {
        "state": ("final" if today >= s["post_end"] else
                  "post" if today > s["win_end_t1"] else
                  "window" if today >= s["win_start"] else "locked"),
        "updated": today.isoformat(),
        "final": today >= s["post_end"],
        "classic": classic, "t1": t1, "post": post,
    }
    obj["per_name_marks"] = an.per_name_marks(names, px, s)
    panel_start = min(date.fromisoformat(obj["locked_at"]), s["win_start"])
    obj["panel"] = an.build_panel(px, [n["ticker"] for n in names] + ["SPY"],
                                  panel_start, s["post_end"])
    return obj


def _rebuild_history() -> dict:
    rows = []
    for mstr in sorted(archive_months()):
        obj = load_month(mstr)
        if not obj:
            continue
        st = obj.get("stats", {})
        strip = lambda d: {k: v for k, v in (d or {}).items() if k != "daily"}
        rows.append({"month": mstr, "locked_at": obj.get("locked_at"),
                     "n": obj.get("n_basket", len(obj.get("names", []))),
                     "backfilled": obj.get("backfilled", False),
                     "final": st.get("final", False),
                     "classic": strip(st.get("classic")),
                     "t1": strip(st.get("t1")),
                     "post": strip(st.get("post"))})
    out = {"as_of": date.today().isoformat(),
           "survivorship_note": ("Backfilled months use CURRENT Russell 1000 "
                                 "membership; delisted losers are absent, which "
                                 "flatters loser recovery."),
           "disclaimer": DISCLAIMER, "rows": rows}
    save("history", out)
    return out


def _basket_with_div_flags(px: dict, universe: list[dict], lock_date: date,
                           flags: dict | None = None) -> tuple[list[dict], dict]:
    """Two-pass build: decile membership doesn't depend on the dividend flag,
    so rank once to find the decile, fetch flags for just those names, re-rank."""
    if flags is None:
        first = an.build_basket(px, universe, lock_date, {})
        flags = uni.dividend_payers([b["ticker"] for b in first])
    return an.build_basket(px, universe, lock_date, flags), flags


def run_auto() -> dict:
    today = date.today()
    status: list[dict] = []
    state = cal.state_for(today)

    if not state["is_trading_day"]:
        out = {"date": today.isoformat(), **_state_summary(state),
               "calendar_check": "n/a", "segments": [{"segment": "gate",
               "ok": True, "note": "non-trading day"}]}
        save("status", out)
        print(f"[pretom] {today} non-trading day — no-op")
        return out

    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    calendar_check = "n/a"      # set from SPY actuals when a reconcile runs

    # --- lock this month's basket (tau-10 through tau-4, once) --------------
    k = cal.tau_offset(today)
    mstr = f"{today.year:04d}-{today.month:02d}"
    if (universe and k is not None and cal.LOCK_AT <= k <= cal.WIN_END_CLASSIC
            and not load_month(mstr)):
        px = _fetch_prices([u["ticker"] for u in universe] + ["SPY"], "1y",
                           status, "prices(lock)")
        try:
            basket, flags = _basket_with_div_flags(px, universe, today)
            obj = _build_month((today.year, today.month), today, px, universe,
                               uni_status, flags, backfilled=False)
            obj["names"] = basket
            archive_month(mstr, obj)
            status.append({"segment": "lock", "ok": True, "n": len(basket),
                           "month": mstr, "late": obj["locked_late"]})
        except Exception as e:
            status.append({"segment": "lock", "ok": False, "error": str(e)[:200]})

    # --- reconcile every recent non-final month ------------------------------
    for mstr2 in archive_months()[:4]:
        obj = load_month(mstr2)
        if not obj or obj.get("stats", {}).get("final"):
            continue
        try:
            tickers = [n["ticker"] for n in obj["names"]] + ["SPY"]
            px = _fetch_prices(tickers, "6mo", status, f"prices({mstr2})")
            spy = px.get("SPY")
            if spy is None or spy.empty:
                raise ValueError("no SPY history")
            if date.fromisoformat(obj["schedule"]["win_start"]) <= today:
                _update_stats(obj, px, spy["close"], today)
            archive_month(mstr2, obj)
            calendar_check = ("ok" if not cal.verify_against(spy.index[-90:])
                              else "mismatch")
            status.append({"segment": f"reconcile({mstr2})", "ok": True,
                           "state": obj["stats"]["state"]})
        except Exception as e:
            status.append({"segment": f"reconcile({mstr2})", "ok": False,
                           "error": str(e)[:200]})

    # --- current basket pointer + history + status ---------------------------
    months = archive_months()
    if months:
        save("basket", load_month(months[0]))
    hist = _rebuild_history()
    out = {"date": today.isoformat(), **_state_summary(state),
           "calendar_check": calendar_check, "n_months": len(months),
           "segments": status}
    save("status", out)

    print(f"[pretom] {today} state={state['state']} month={state['month']} "
          f"tau_offset={state['tau_offset']} months={len(months)} "
          f"history_rows={len(hist['rows'])}")
    for s in status:
        flag = "ok " if s.get("ok") else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', '')}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    return out


def _state_summary(state: dict) -> dict:
    return {"state": state["state"], "month": state["month"],
            "tau_offset": state["tau_offset"],
            "days_to_open": state["days_to_open"],
            "days_to_close_classic": state["days_to_close_classic"],
            "days_to_close_t1": state["days_to_close_t1"],
            "schedule": cal.schedule_iso(state["schedule"])}


def run_backfill(n_months: int = 24, force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []
    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    if not universe:
        print("[pretom] backfill aborted: no universe")
        return {"ok": False}

    tickers = [u["ticker"] for u in universe] + ["SPY"]
    px = _fetch_prices(tickers, "5y", status, "prices(5y)")
    spy = px.get("SPY")
    if spy is None or spy.empty:
        print("[pretom] backfill aborted: no SPY history")
        return {"ok": False}

    # dividend flags once for the union of decile members (trailing-year proxy)
    months = cal.completed_months(n_months, today)
    deciles = set()
    for (y, m) in months:
        sched = cal.month_schedule(y, m)
        for b in an.build_basket(px, universe, sched["lock_by"], {}):
            deciles.add(b["ticker"])
    flags = uni.dividend_payers(sorted(deciles))
    status.append({"segment": "dividends", "ok": True, "n": len(deciles)})

    done = 0
    for (y, m) in months:
        mstr = f"{y:04d}-{m:02d}"
        if load_month(mstr) and not force:
            continue
        try:
            sched = cal.month_schedule(y, m)
            obj = _build_month((y, m), sched["lock_by"], px, universe,
                               uni_status, flags, backfilled=True)
            _update_stats(obj, px, spy["close"], today)
            archive_month(mstr, obj)
            done += 1
        except Exception as e:
            status.append({"segment": f"backfill({mstr})", "ok": False,
                           "error": str(e)[:200]})

    hist = _rebuild_history()
    print(f"[pretom] backfill wrote {done}/{len(months)} months "
          f"(history rows={len(hist['rows'])})")
    for s in status:
        flag = "ok " if s.get("ok") else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', '')}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    return {"ok": True, "written": done}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "backfill"])
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.action == "backfill":
        run_backfill(args.months, args.force)
    else:
        run_auto()


if __name__ == "__main__":
    main()

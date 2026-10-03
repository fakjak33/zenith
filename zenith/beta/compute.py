"""CLEAN BETA compute orchestrator.

  python -m zenith.beta.compute --action auto        (nightly path)
  python -m zenith.beta.compute --action screen      (force the monthly screen)
  python -m zenith.beta.compute --action rebalance   (force screen + basket rebalance)
  python -m zenith.beta.compute --action hedge       (hedge monitor only)

`auto` is self-healing rather than calendar-exact: it re-screens whenever the
stored screen is from an earlier month, rebalances whenever the stored basket
predates the current quarter-start month (config.BETA_REBALANCE_MONTHS), and
runs the hedge monitor every trading day. A missed run is caught up on the
next one, never skipped.

Each segment is fail-soft: an exception is recorded in status.json and the
remaining segments still run. A ticker that fails to download is logged in
the screen's coverage numbers, never fatal.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save, scrub
from . import basket as bk
from . import hedge as hg
from . import quality as q
from . import screen as sc
from ..cas.sources import prices
from ..config import (BETA_BENCHMARK, BETA_EARNINGS_DAYS, BETA_EXPORT_DIR, BETA_HEDGE_PERIOD,
                      BETA_LOOKBACK, BETA_OPTIONABLE_TTL_DAYS, BETA_PRICE_PERIOD,
                      BETA_REBALANCE_MONTHS)
from ..pretom import calendar as cal

HEDGE_TICKERS = ("SPY", "^VIX", "BTAL", "DBMF", "KMLM")


# ------------------------------------------------------------------ seams --
def _fetch_prices(tickers: list[str], period: str, status: list[dict], label: str) -> dict:
    """Monkeypatchable seam (mirrors trend.compute._fetch_prices)."""
    px, st = prices.get_history(tickers, period=period)
    missing = [t for t in tickers if t not in px]
    if len(missing) > max(5, len(tickers) // 20):
        time.sleep(15)
        px2, _ = prices.get_history(missing, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px), "requested": len(tickers),
                   "missing": sorted(t for t in tickers if t not in px)[:50],
                   "error": st.get("error", "")})
    return px


def _load_universe() -> tuple[list[dict], dict]:
    """The Russell 1000 exactly as TREND/MOMENTUM resolve it (with sector/mktcap)."""
    from ..trend.universe import stocks
    return stocks()


def _fundamentals(tickers: list[str]) -> dict:
    """Read-only view of IDEAS' committed fundamentals cache."""
    from ..ideas import fundamentals
    return fundamentals.get(tickers)


def _earnings(universe_by_ticker: dict) -> dict[str, str]:
    from ..pead import earnings
    return sc.earnings_flags(earnings.upcoming_calendar(BETA_EARNINGS_DAYS, universe_by_ticker))


def _optionable(tickers: list[str], today: date) -> dict[str, bool]:
    """Listed-options check for screen finalists only, TTL-cached in
    data/beta/optionable.json. Unknown (lookup failed) is left out of the
    result, i.e. assumed optionable and not penalized."""
    doc = load("optionable", {})
    todo = []
    for t in tickers:
        e = doc.get(t)
        if not e or (today - date.fromisoformat(e["asof"])).days > BETA_OPTIONABLE_TTL_DAYS:
            todo.append(t)
    if todo:
        try:
            import yfinance as yf
            for t in todo:
                try:
                    doc[t] = {"optionable": bool(yf.Ticker(t).options), "asof": today.isoformat()}
                except Exception:
                    pass
                time.sleep(0.1)
        except Exception:
            pass
        save("optionable", doc, indent=None)
    return {t: doc[t]["optionable"] for t in tickers if t in doc}


def _chains(ticker: str, today: date) -> tuple[float | None, list[dict]]:
    """SPY spot + put chains for expiries 30-200 DTE. Never raises."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        out = []
        for exp in tk.options or []:
            d = hg.dte(exp, today)
            if 30 <= d <= 200:
                try:
                    puts = tk.option_chain(exp).puts
                    out.append({"expiry": exp, "dte": d, "puts": puts})
                except Exception:
                    continue
                time.sleep(0.2)
        hist = tk.history(period="5d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        return spot, out
    except Exception:
        return None, []


# ------------------------------------------------------------ scheduling --
def quarter_key(today: date) -> str:
    """'YYYY-MM' of the most recent rebalance month at or before today."""
    y, m = today.year, today.month
    for _ in range(13):
        if m in BETA_REBALANCE_MONTHS:
            return f"{y:04d}-{m:02d}"
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return f"{today.year:04d}-{today.month:02d}"


def needs_screen(latest: dict, today: date) -> bool:
    return (latest.get("computed") or "")[:7] != today.strftime("%Y-%m")


def needs_rebalance(basket: dict, today: date) -> bool:
    return basket.get("quarter") != quarter_key(today)


# --------------------------------------------------------------- segments --
def run_screen(today: date, status: list[dict], rebalance: bool) -> dict:
    members, ust = _load_universe()
    status.append({"segment": "universe", "ok": bool(members), "n": len(members),
                   **{k: v for k, v in (ust or {}).items() if not isinstance(v, (list, dict))}})
    tickers = sorted({m["ticker"] for m in members})
    px = _fetch_prices(tickers + [BETA_BENCHMARK], BETA_PRICE_PERIOD, status, "prices")
    if BETA_BENCHMARK not in px:
        raise RuntimeError(f"no benchmark ({BETA_BENCHMARK}) prices")
    bench_ret = px[BETA_BENCHMARK]["close"].pct_change().dropna()
    as_of = bench_ret.index[-1].strftime("%Y-%m-%d")

    fund = _fundamentals(tickers)
    status.append({"segment": "fundamentals", "ok": bool(fund), "n": len(fund), "requested": len(tickers)})
    try:
        earn = _earnings({t: True for t in tickers})
        status.append({"segment": "earnings", "ok": True, "n": len(earn)})
    except Exception as e:
        earn = {}
        status.append({"segment": "earnings", "ok": False, "error": str(e)[:200]})
    shares = q.snapshot_shares(load("shares", {}), fund, today)
    save("shares", shares, indent=None)

    prior_basket = load("basket", {})
    incumbents = {m["ticker"] for m in prior_basket.get("members", [])}
    rows = sc.build_rows(members, px, bench_ret, fund, earn, shares, incumbents, today)
    # Listed-options check only for names that otherwise pass (dozens, not 1,000).
    finalists = [x["ticker"] for x in rows if x.get("passes")]
    opt = _optionable(finalists, today) if finalists else {}
    if any(v is False for v in opt.values()):
        rows = sc.build_rows(members, px, bench_ret, fund, earn, shares, incumbents, today, opt)
    for x in rows:
        x["optionable"] = opt.get(x["ticker"])

    doc = {"as_of": as_of, "computed": today.isoformat(), "benchmark": BETA_BENCHMARK,
           "disclaimer": DISCLAIMER, "n": len(rows),
           "n_liquid": sum(1 for x in rows if x.get("f_liquid")),
           "n_pass": sum(1 for x in rows if x.get("passes")),
           "n_priced": sum(1 for m in members if m["ticker"] in px),
           "filter_counts": {f: sum(1 for x in rows if x.get(f)) for f in ("f_liquid", *sc.FILTERS)},
           "rows": rows}
    save("latest", doc, indent=None)

    if rebalance or not prior_basket.get("members"):
        cand = [x["ticker"] for x in rows if x.get("passes")]
        rets = sc.returns_panel(px, cand, BETA_LOOKBACK)
        b = bk.select(rows, rets)
        b.update({"as_of": as_of, "quarter": quarter_key(today), "computed": today.isoformat(),
                  "prior": sorted(incumbents),
                  "added": sorted({m["ticker"] for m in b["members"]} - incumbents),
                  "removed": sorted(incumbents - {m["ticker"] for m in b["members"]})})
        save("basket", b)
        status.append({"segment": "basket", "ok": b["n"] > 0, "n": b["n"], "short": b["short"],
                       "beta": b.get("beta"), "eff_dim": b.get("eff_dim")})
        inb = {m["ticker"] for m in b["members"]}
    else:
        inb = {m["ticker"] for m in prior_basket.get("members", [])}
    for x in rows:
        x["in_basket"] = x["ticker"] in inb
    save("latest", doc, indent=None)
    export(doc, load("basket", {}), load("hedge", {}))
    return {"segment": "screen", "ok": doc["n_pass"] > 0, "n": doc["n"], "n_liquid": doc["n_liquid"],
            "n_pass": doc["n_pass"], "as_of": as_of}


def run_hedge(today: date, status: list[dict]) -> dict:
    b = load("basket", {})
    names = [m["ticker"] for m in b.get("members", [])]
    px = _fetch_prices(list(HEDGE_TICKERS), BETA_HEDGE_PERIOD, status, "hedge_prices")
    if "SPY" not in px:
        raise RuntimeError("no SPY prices")
    spy = px["SPY"]["close"]
    state: dict = {"as_of": spy.index[-1].strftime("%Y-%m-%d"), "computed": today.isoformat()}
    state["trend"] = hg.trend_gate(spy)
    state["trend_etfs"] = {t: hg.tsmom(px[t]["close"]) for t in ("DBMF", "KMLM") if t in px}
    if "^VIX" in px:
        state["vrp"] = hg.vrp(px["^VIX"]["close"], spy)
    if "BTAL" in px:
        state["btal"] = hg.btal_gate(px["BTAL"]["close"])
    if names:
        bpx = _fetch_prices(names, BETA_PRICE_PERIOD, status, "basket_prices")
        w = {m["ticker"]: m["weight"] or 0.0 for m in b["members"]}
        panel = sc.returns_panel(bpx, names, BETA_LOOKBACK).fillna(0.0)
        bret = (panel * pd.Series(w)).sum(axis=1) / max(sum(w.values()), 1e-9)
        state["vol_target"] = hg.vol_target(bret)
        state["basket"] = {"beta": b.get("beta"), "vol": b.get("vol"), "eff_dim": b.get("eff_dim"),
                           "n": b.get("n"), "quarter": b.get("quarter"),
                           "spy_equiv_per_mm": round((b.get("beta") or 0) * 1e6)}
    beta = b.get("beta") or 1.0
    budget = (state.get("vrp") or {}).get("put_budget") or 0.01
    spot, chains = _chains("SPY", today)
    if spot and chains:
        state["puts"] = hg.put_ladder(spot, chains, beta, budget)
        status.append({"segment": "options", "ok": bool(state["puts"]["rungs"]), "n_expiries": len(chains)})
    else:
        state["puts"] = {"rungs": [], "error": "SPY option chain unavailable this run"}
        status.append({"segment": "options", "ok": False})
    state["alerts"] = hg.alerts(state)
    save("hedge", state, indent=None)

    hist = load("hedge_history", {"rows": []})
    rows = [x for x in hist.get("rows", []) if x["date"] != state["as_of"]]
    v = state.get("vrp") or {}
    rows.append({"date": state["as_of"], "trend": state["trend"]["state"],
                 "spy": state["trend"].get("price"), "vix": v.get("vix"), "vrp": v.get("vrp"),
                 "vrp_pct": v.get("vrp_pct"), "vix_pct": v.get("vix_pct"),
                 "scale": (state.get("vol_target") or {}).get("scale"),
                 "btal": (state.get("btal") or {}).get("state"),
                 "put_budget": v.get("put_budget"),
                 "ladder_carry": (state.get("puts") or {}).get("ladder_carry_full")})
    save("hedge_history", {"rows": sorted(rows, key=lambda x: x["date"])}, indent=None)
    return {"segment": "hedge", "ok": True, "as_of": state["as_of"], "trend": state["trend"]["state"],
            "n_alerts": len(state["alerts"])}


def export(latest: dict, basket: dict, hedge_state: dict) -> None:
    """Point-in-time PARALLAX snapshot: data/beta/exports/<as_of>.{json,csv}."""
    as_of = latest.get("as_of")
    if not as_of:
        return
    import json
    BETA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    snap = {"as_of": as_of, "computed": latest.get("computed"), "benchmark": latest.get("benchmark"),
            "rows": latest.get("rows", []),
            "basket": {k: basket.get(k) for k in ("quarter", "members", "beta", "vol", "enb", "eff_dim")},
            "hedge": {k: v for k, v in (hedge_state or {}).items() if k not in ("vrp",)}
            | {"vrp": {k: v for k, v in ((hedge_state or {}).get("vrp") or {}).items() if k != "series"}}}
    (BETA_EXPORT_DIR / f"{as_of}.json").write_text(json.dumps(scrub(snap), ensure_ascii=False),
                                                     encoding="utf-8")
    cols = ["ticker", "name", "sector", "size", "mktcap", "price", "adv", "bars", "bswa", "ols",
            "vasicek", "rho", "r2", "ivol", "vol", "jumps", "mom", "beta_pct", "rho_pct", "ivol_pct",
            "quality", "quality_imputed", "earnings", "ipo", "f_liquid", "f_beta", "f_rho", "f_ivol",
            "f_events", "f_quality", "passes", "score", "rank", "in_basket"]
    with open(BETA_EXPORT_DIR / f"{as_of}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["as_of", *cols], extrasaction="ignore")
        w.writeheader()
        for x in latest.get("rows", []):
            w.writerow({"as_of": as_of, **scrub(x)})


# ------------------------------------------------------------------ driver --
def run(action: str = "auto", force: bool = False, today: date | None = None) -> dict:
    today = today or date.today()
    if action == "auto" and not cal.is_trading_day(today) and not force:
        save("status", {**load("status", {}), "checked": today.isoformat(), "is_trading_day": False})
        print(f"[beta] {today} non-trading day -- no-op")
        return {"ok": True, "gated": True}

    status: list[dict] = []
    latest, basket = load("latest", {}), load("basket", {})
    do_rebalance = action == "rebalance" or (action == "auto" and needs_rebalance(basket, today))
    do_screen = (action in ("screen", "rebalance") or do_rebalance
                 or (action == "auto" and needs_screen(latest, today)))
    results = {}
    if do_screen:
        try:
            results["screen"] = run_screen(today, status, rebalance=do_rebalance)
            status.append(results["screen"])
            print(f"[beta] screen: {results['screen']}")
        except Exception as e:
            status.append({"segment": "screen", "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            traceback.print_exc()
    if action in ("auto", "hedge", "rebalance", "screen"):
        try:
            results["hedge"] = run_hedge(today, status)
            status.append(results["hedge"])
            print(f"[beta] hedge: {results['hedge']}")
            if results.get("screen"):        # re-export so the snapshot carries today's hedge state
                export(load("latest", {}), load("basket", {}), load("hedge", {}))
        except Exception as e:
            status.append({"segment": "hedge", "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            traceback.print_exc()
    save("status", {"date": today.isoformat(), "checked": today.isoformat(), "is_trading_day": True,
                    "action": action, "screened": do_screen, "rebalanced": do_rebalance,
                    "disclaimer": DISCLAIMER, "segments": status})
    return {"ok": bool(results), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "screen", "rebalance", "hedge"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(action=args.action, force=args.force)


if __name__ == "__main__":
    main()

"""EDGE screens compute orchestrator.

  python -m zenith.edge.compute --action auto

Nightly weekday path: gate on the trading calendar, pull one price panel + the
cached info/options data, build the four screens, snapshot decile members into
history, evaluate matured horizons, and write the artifacts. Screens start live
(no backfill) — the per-screen tracker fills in as history accrues.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import numpy as np

from . import SCREENS, DISCLAIMER, save, load
from . import info as einfo
from . import options as eopt
from . import ivspread, revisions, shortint, lottery, history
from ..pretom import calendar as cal
from ..pretom import universe as uni
from ..cas.sources import prices

# Universe caps by liquidity (ADV$). The yfinance .info/option pulls are ~1-1.5s
# each and the cache is gitignored, so a cold run (local or CI) must finish in
# one shot; these caps keep it well under the Action's 50-min budget while
# covering the liquid names where each signal is most reliable.
SHORTINT_UNIVERSE = 700     # short interest: 1 .info call each
REVISIONS_UNIVERSE = 250    # 3 yfinance calls each — the runtime long pole
OPTIONS_UNIVERSE = 200      # option-chain pull


def _fetch_prices(tickers, period, status, label):
    px, st = prices.get_history(tickers, period=period)
    missing = [t for t in tickers if t not in px]
    if len(missing) > max(5, len(tickers) // 20):
        time.sleep(15)
        px2, _ = prices.get_history(missing, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px),
                   "requested": len(tickers)})
    return px


def _adv_dollar(df, lookback=63):
    try:
        s = (df["close"] * df["volume"]).dropna().tail(lookback)
        return float(s.mean()) if len(s) else 0.0
    except Exception:
        return 0.0


def _spot(df):
    try:
        return float(df["close"].dropna().iloc[-1])
    except Exception:
        return None


def _mom_1m(df):
    try:
        c = df["close"].dropna()
        return float(c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) > 21 else None
    except Exception:
        return None


def run_auto(force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []
    if not cal.is_trading_day(today) and not force:
        save("status", {"date": today.isoformat(), "is_trading_day": False,
                        "segments": [{"segment": "gate", "ok": True,
                                      "note": "non-trading day"}]})
        print(f"[edge] {today} non-trading day — no-op")
        return {"ok": True, "gated": True}

    universe, uni_status = uni.russell1000()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    if not universe:
        save("status", {"date": today.isoformat(), "segments": status})
        return {"ok": False}
    meta = {u["ticker"]: {"name": u.get("name", ""), "sector": u.get("sector", "")}
            for u in universe}
    tickers = [u["ticker"] for u in universe]

    px = _fetch_prices(tickers + ["SPY"], "1y", status, "prices")
    spy = px.get("SPY")
    spy_close = spy["close"] if spy is not None and not spy.empty else None
    if spy_close is None:
        status.append({"segment": "spy", "ok": False, "error": "no SPY"})
        save("status", {"date": today.isoformat(), "segments": status})
        return {"ok": False}

    priced = [t for t in tickers if t in px]
    by_adv = sorted(priced, key=lambda t: _adv_dollar(px[t]), reverse=True)
    si_universe = by_adv[:SHORTINT_UNIVERSE]
    opt_universe = by_adv[:OPTIONS_UNIVERSE]
    rev_universe = by_adv[:REVISIONS_UNIVERSE]

    # --- short interest (top ADV$, 1 .info call each) ------------------------
    si = einfo.short_interest(si_universe)
    hi_si = _build_shortint(si, px, meta, status)

    # --- IV spread (top ADV$) ------------------------------------------------
    _build_ivspread(opt_universe, px, meta, si, status)

    # --- revisions (top ADV$) ------------------------------------------------
    _build_revisions(rev_universe, meta, status)

    # --- lottery MAX-beta (full priced universe) -----------------------------
    _build_lottery(priced, px, spy_close, meta, status)

    # --- history: snapshot + evaluate ----------------------------------------
    added = 0
    for screen in SCREENS:
        res = load(screen, {})
        if res.get("ranked"):
            added += history.append_rows(
                history.make_rows(screen, res, today.isoformat(),
                                  res.get("horizon_td", 20)))
    n_eval = history.evaluate_pending(px, spy_close, today)
    status.append({"segment": "history", "ok": True, "added": added, "evaluated": n_eval})

    save("status", {"date": today.isoformat(), "is_trading_day": True,
                    "disclaimer": DISCLAIMER, "universe_n": len(universe),
                    "segments": status})
    print(f"[edge] {today} screens built; history +{added} eval {n_eval}")
    for s in status:
        print(f"  {'ok ' if s.get('ok') else 'ERR'} {s['segment']}: n={s.get('n', s.get('added',''))}"
              + (f" ({s.get('error')})" if s.get("error") else ""))
    return {"ok": True}


def _hard_to_borrow(si: dict) -> set[str]:
    """Names in the top-quartile of short percent of float (borrow-fee proxy)."""
    vals = [(t, d.get("si_float")) for t, d in si.items() if d.get("si_float") is not None]
    if not vals:
        return set()
    cut = float(np.percentile([v for _, v in vals], 75))
    return {t for t, v in vals if v >= cut}


def _build_shortint(si, px, meta, status):
    rows = []
    for t, d in si.items():
        if not d:
            continue
        rows.append({"ticker": t, **meta.get(t, {}),
                     "si_float": d.get("si_float"), "dtc": d.get("dtc"),
                     "mom_1m": _mom_1m(px[t]) if t in px else None,
                     "shares_short": d.get("shares_short")})
    hist_si = [r.get("median_si_float") for r in load("shortint", {}).get("_agg_history", [])]
    res = shortint.build(rows, si_float_history=[v for v in hist_si if v is not None])
    agg_hist = (load("shortint", {}).get("_agg_history", []) or [])[-250:]
    if res.get("aggregate"):
        agg_hist = agg_hist + [{"date": date.today().isoformat(),
                                "median_si_float": res["aggregate"]["median_si_float"]}]
    res["_agg_history"] = agg_hist
    res["as_of"] = date.today().isoformat()
    save("shortint", res)
    status.append({"segment": "shortint", "ok": bool(res.get("n")), "n": res.get("n", 0)})
    return si


def _build_ivspread(opt_universe, px, meta, si, status):
    spots = {t: _spot(px[t]) for t in opt_universe if t in px}
    raw = eopt.iv_spread_rows(opt_universe, spots)
    rows = [{"ticker": t, **meta.get(t, {}), "iv_spread": d["iv_spread"],
             "n_pairs": d.get("n_pairs"), "spot": d.get("spot")}
            for t, d in raw.items()]
    res = ivspread.build(rows, hard_to_borrow=_hard_to_borrow(si))
    res["as_of"] = date.today().isoformat()
    save("ivspread", res)
    status.append({"segment": "ivspread", "ok": bool(res.get("n")), "n": res.get("n", 0)})


def _build_revisions(rev_universe, meta, status):
    rev = einfo.revisions(rev_universe)
    rows = [{"ticker": t, **meta.get(t, {}), **d}
            for t, d in rev.items() if d and d.get("est_rev_pct") is not None]
    res = revisions.build(rows)
    res["as_of"] = date.today().isoformat()
    save("revisions", res)
    status.append({"segment": "revisions", "ok": bool(res.get("n")), "n": res.get("n", 0)})


def _build_lottery(priced, px, spy_close, meta, status):
    sub = {t: px[t] for t in priced if t in px}
    res = lottery.build(sub, spy_close, meta)
    res["as_of"] = date.today().isoformat()
    save("lottery", res)
    status.append({"segment": "lottery", "ok": bool(res.get("n")), "n": res.get("n", 0)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_auto(force=args.force)


if __name__ == "__main__":
    main()

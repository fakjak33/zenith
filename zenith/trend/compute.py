"""TREND FOLLOWING compute orchestrator.

  python -m zenith.trend.compute --action auto                 (nightly path)
  python -m zenith.trend.compute --action backfill             (one-off, 10y of history)
  python -m zenith.trend.compute --action auto --universe etfs (one universe only)

Nightly path: gate on the trading calendar -> resolve both universes from the
MOMENTUM / ETF MOMENTUM constituent functions (universe.py) -> ONE price pull
for the union -> per instrument, the full seven-speed EWMAC series (vectorized;
~1,300 bars x 7 speeds is milliseconds) -> today's row, the history rows dated
after the last stored day (so a missed night is gap-filled, never skipped),
and every event dated after it -> write latest / history / events /
recent_events / breadth / diagnostics / status.

Why recompute the full series nightly rather than update the EMAs
incrementally from stored state: the download dominates the cost, not the
math, and a full recompute is immune to the retroactive dividend
back-adjustments yfinance applies to past prices -- a persisted EMA state
would silently drift from the true series after every ex-date. (The
forecasts themselves barely move under such an adjustment, since the
vol-normalization is scale-invariant -- but "barely" is not "exactly".)

Each universe is fail-soft: an exception in one is recorded in its status and
the other still completes.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from datetime import date

import numpy as np
import pandas as pd

from . import DISCLAIMER, SPEED_KEYS, load, save
from . import events as ev
from . import history, structure, systems
from . import universe as trend_universe
from .ewmac import finite_or_none
from ..cas.sources import prices
from ..config import (TREND_BACKFILL_PERIOD, TREND_FORECAST_CAP, TREND_FORECAST_SCALARS,
                      TREND_FORECAST_TARGET_ABS, TREND_MIN_SPEEDS, TREND_NORMALIZATION,
                      TREND_PRICE_PERIOD, TREND_RECENT_CROSS_DAYS, TREND_RECENT_EVENT_DAYS,
                      TREND_SPARK_DAYS, TREND_SPEEDS, TREND_STALE_DAYS, TREND_UNIVERSES,
                      TREND_CASH_LIKE_VOL)
from ..edge.common import pct_ranks
from ..pretom import calendar as cal

_EVENT_FLUSH_EVERY = 250     # tickers between event-shard flushes (bounds backfill memory)


def _fetch_prices(tickers: list[str], period: str, status: list[dict], label: str) -> dict:
    """Monkeypatchable seam, mirroring mom.compute._fetch_prices: tests stub
    this rather than making a ~2,000-ticker network call."""
    px, st = prices.get_history(tickers, period=period)
    missing = [t for t in tickers if t not in px]
    if len(missing) > max(5, len(tickers) // 20):
        time.sleep(15)
        px2, _ = prices.get_history(missing, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px), "requested": len(tickers),
                   "error": st.get("error", "")})
    return px


def _load_universe(universe: str) -> tuple[list[dict], dict]:
    """Monkeypatchable seam around universe.LOADERS."""
    return trend_universe.LOADERS[universe]()


def _leverage(px: dict) -> dict[str, str]:
    """Monkeypatchable seam around ETF MOMENTUM's empirical leverage gate."""
    return trend_universe.etf_leverage_exclusions(px)


def _scrub(obj):
    """Recursively replace non-finite floats with None (valid JSON only)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _scrub(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _r(x, nd: int = 2):
    v = finite_or_none(x)
    return None if v is None else round(v, nd)


def _ago(s: pd.Series, k: int):
    return s.iloc[-1 - k] if len(s) > k else np.nan


# ------------------------------------------------------------ one instrument --
def process_ticker(ticker: str, close: pd.Series, as_of: pd.Timestamp,
                   since: str | None) -> dict:
    """Everything the run needs from one instrument. Returns
    {"row": latest-row fields, "encoded": history.Encoded, "events": [...],
     "abs_mean": per-speed mean |forecast| over its own history,
     "exclusion_reason": str | None}."""
    try:
        res = systems.build(close)
    except ValueError as e:                        # non-daily input
        return {"exclusion_reason": f"invalid_series({e})"}
    if res.forecast.empty:
        return {"exclusion_reason": "no_price_data"}
    last_bar = res.forecast.index[-1]
    if (as_of - last_bar).days > TREND_STALE_DAYS:
        return {"exclusion_reason": f"stale_price(last bar {last_bar.date()})"}

    score_now = finite_or_none(res.score.iloc[-1])
    evs = ev.detect(ticker, res.raw, res.score)
    encoded = history.encode_rows(res.forecast, res.raw, res.score).after(since)
    new_events = [e for e in evs if since is None or e["date"] > since]
    if score_now is None:
        return {"exclusion_reason": f"insufficient_history({len(res.forecast)} bars; "
                                    f"needs {TREND_SPEEDS[TREND_MIN_SPEEDS - 1][1]} for "
                                    f"{TREND_MIN_SPEEDS} speeds)",
                "encoded": encoded, "events": new_events, "bars": len(res.forecast)}

    f_now = [finite_or_none(x) for x in res.forecast.iloc[-1]]
    struct = structure.analyse(f_now, score_now)

    def f_ago(k: int):
        if len(res.forecast) <= k:
            return None
        return [_r(x, 1) for x in res.forecast.iloc[-1 - k]]

    non_x = [e for e in evs if e["type"] != "cross"]
    last_evt = non_x[-1] if non_x else None
    confirms = [e for e in evs if e["type"] == "confirmation"]
    last_conf = confirms[-1] if confirms else None
    lc = ev.last_cross(res.raw)
    x_dates = [v["date"] for v in lc.values() if v]
    spark = [_r(x, 1) for x in res.score.tail(TREND_SPARK_DAYS)]
    ann_vol = finite_or_none(close.pct_change(fill_method=None).tail(252).std() * np.sqrt(252))
    row = {
        "score": round(score_now, 2),
        "state": ev.state_label(score_now),
        "forecasts": [_r(x) for x in f_now],
        "directions": [None if not np.isfinite(x) else int(np.sign(x)) for x in res.raw.iloc[-1]],
        "f5": f_ago(5), "f20": f_ago(20), "f60": f_ago(60),
        "d1": _r(score_now - _ago(res.score, 1)), "d5": _r(score_now - _ago(res.score, 5)),
        "d20": _r(score_now - _ago(res.score, 20)),
        **struct,
        "partial": struct["n_valid"] < len(SPEED_KEYS),
        "last_cross": lc,
        "latest_cross_date": max(x_dates) if x_dates else None,
        "last_event": ({k: last_evt.get(k) for k in ("date", "type", "dir", "n_speeds", "from", "to")}
                       if last_evt else None),
        "last_confirmation": ({k: last_conf.get(k) for k in ("date", "dir", "n_speeds", "speeds", "horizon")}
                              if last_conf else None),
        "spark": spark,
        "ann_vol": None if ann_vol is None else round(ann_vol, 4),
        "cash_like": bool(ann_vol is not None and ann_vol < TREND_CASH_LIKE_VOL),
        "bars": len(res.forecast), "asof": last_bar.strftime("%Y-%m-%d"),
        "last_close": _r(close.iloc[-1], 4),
    }
    abs_mean = [finite_or_none(res.forecast[k].abs().mean()) for k in SPEED_KEYS]
    return {"row": row, "encoded": encoded, "events": new_events, "abs_mean": abs_mean,
            "exclusion_reason": None}


# --------------------------------------------------------------- one universe --
def _rank(rows: list[dict]) -> None:
    scored = [r for r in rows if r.get("score") is not None]
    scored.sort(key=lambda r: r["score"], reverse=True)
    pr = pct_ranks([r["score"] for r in scored])
    for i, (r, p) in enumerate(zip(scored, pr)):
        r["rank"], r["pctile"] = i + 1, p


def _diagnostics(rows: list[dict], abs_means: list[list], as_of: str) -> dict:
    """The CHECKS the methodology panel renders: is each speed's realized
    average |forecast| near the scalar's target of 10, and how correlated are
    the seven speeds across the cross-section today."""
    scored = [r for r in rows if r.get("score") is not None]
    f = pd.DataFrame([r["forecasts"] for r in scored], columns=list(SPEED_KEYS), dtype=float)
    am = pd.DataFrame(abs_means, columns=list(SPEED_KEYS), dtype=float)
    corr = f.corr(method="spearman") if len(f) >= 10 else pd.DataFrame()
    s = pd.Series([r["score"] for r in scored], dtype=float)
    return {
        "as_of": as_of, "normalization": TREND_NORMALIZATION,
        "scalars": {k: TREND_FORECAST_SCALARS[sp] for k, sp in zip(SPEED_KEYS, TREND_SPEEDS)},
        "target_abs": TREND_FORECAST_TARGET_ABS, "cap": TREND_FORECAST_CAP,
        "realized_abs_today": {k: _r(f[k].abs().mean()) for k in SPEED_KEYS} if len(f) else {},
        "realized_abs_longrun": {k: _r(am[k].mean()) for k in SPEED_KEYS} if len(am) else {},
        "pct_capped_today": {k: _r((f[k].abs() >= TREND_FORECAST_CAP - 1e-9).mean(), 4)
                             for k in SPEED_KEYS} if len(f) else {},
        "speed_correlation": ({a: {b: _r(corr.loc[a, b], 3) for b in SPEED_KEYS} for a in SPEED_KEYS}
                              if not corr.empty else {}),
        "distribution": ({"n": int(len(s)), "mean": _r(s.mean(), 3), "median": _r(s.median(), 3),
                          "std": _r(s.std(), 3), "pct_bull": _r((s >= 5).mean(), 4),
                          "pct_bear": _r((s < -5).mean(), 4)} if len(s) else {}),
    }


def run_universe(universe: str, members: list[dict], px: dict, today: date,
                 backfill: bool = False, exclusions: dict | None = None) -> dict:
    """Score one universe, persist everything, return a status segment."""
    exclusions = exclusions or {}
    if backfill:
        history.clear(universe)
    since = None if backfill else history.last_date(universe)

    bar_dates = [px[m["ticker"]].index[-1] for m in members
                 if m["ticker"] in px and px[m["ticker"]] is not None and not px[m["ticker"]].empty]
    if not bar_dates:
        raise RuntimeError(f"no price data for any {universe} member")
    as_of = max(bar_dates)
    as_of_iso = as_of.strftime("%Y-%m-%d")

    rows, encoded, pending_events, abs_means = [], {}, [], []
    recent_events: list[dict] = []
    recent_cut = (as_of - pd.tseries.offsets.BDay(TREND_RECENT_EVENT_DAYS)).strftime("%Y-%m-%d")
    events_added = 0
    for i, m in enumerate(members):
        t = m["ticker"]
        row = dict(m)
        df = px.get(t)
        if t in exclusions:
            out = {"exclusion_reason": exclusions[t]}
        elif df is None or df.empty or "close" not in df:
            out = {"exclusion_reason": "no_price_data"}
        else:
            out = process_ticker(t, df["close"], as_of, since)
        row.update(out.get("row") or {})
        row["excluded"] = out.get("row") is None
        row["exclusion_reason"] = out.get("exclusion_reason")
        if row["excluded"]:
            row["score"] = None
            row["bars"] = out.get("bars")
        rows.append(row)
        if out.get("encoded") is not None and len(out["encoded"]):
            encoded[t] = out["encoded"]
        if out.get("abs_mean"):
            abs_means.append(out["abs_mean"])
        for e in out.get("events") or []:
            if e["type"] != "cross":
                pending_events.append(e)
            if e["date"] > recent_cut:
                recent_events.append(e)
        if len(pending_events) and (i + 1) % _EVENT_FLUSH_EVERY == 0:
            events_added += history.append_events(universe, pending_events, today=today)
            pending_events = []
    events_added += history.append_events(universe, pending_events, today=today)

    _rank(rows)
    rows.sort(key=lambda r: (r.get("score") is None, -(r.get("score") or 0.0)))

    hist = history.append(universe, encoded, today=today)
    history.roll_closed_years(universe, today=today)

    # breadth history: merge by date, never rewriting a stored day
    b_doc = {"rows": []} if backfill else load(universe, "breadth", {"rows": []})
    have = {r["date"] for r in b_doc.get("rows", [])}
    b_doc["rows"] = sorted(b_doc.get("rows", []) + [b for b in hist["breadth"] if b["date"] not in have],
                           key=lambda r: r["date"])
    b_doc["as_of"] = as_of_iso
    save(universe, "breadth", _scrub(b_doc), indent=None)

    prior_recent = [] if backfill else load(universe, "recent_events", {}).get("rows", [])
    recent_rows = history.merge_recent(prior_recent, recent_events, as_of_iso,
                                       TREND_RECENT_EVENT_DAYS, TREND_RECENT_CROSS_DAYS)
    save(universe, "recent_events", {"as_of": as_of_iso, "fields": history.EVENT_FIELDS,
                                     "rows": _scrub(recent_rows)}, indent=None)

    n_scored = sum(1 for r in rows if r.get("score") is not None)
    save(universe, "latest", _scrub({
        "as_of": as_of_iso, "computed": today.isoformat(), "universe": universe,
        "disclaimer": DISCLAIMER, "speeds": list(SPEED_KEYS),
        "n": len(rows), "n_scored": n_scored,
        "n_partial": sum(1 for r in rows if r.get("partial")),
        "rows": rows}), indent=None)
    save(universe, "diagnostics", _scrub(_diagnostics(rows, abs_means, as_of_iso)))
    return {"segment": universe, "ok": n_scored > 0, "n": len(rows), "n_scored": n_scored,
            "as_of": as_of_iso, "history_dates_added": hist["dates_added"],
            "history_rows_added": hist["rows_added"], "events_added": events_added,
            "recent_events": len(recent_rows),
            "n_excluded": len(rows) - n_scored}


# ------------------------------------------------------------------ driver --
def run(action: str = "auto", universes: tuple = TREND_UNIVERSES, force: bool = False) -> dict:
    today = date.today()
    backfill = action == "backfill"
    if not backfill and not cal.is_trading_day(today) and not force:
        for u in universes:
            prior = load(u, "status", {})
            save(u, "status", {**prior, "checked": today.isoformat(), "is_trading_day": False})
        print(f"[trend] {today} non-trading day -- no-op")
        return {"ok": True, "gated": True}

    status: list[dict] = []
    members: dict[str, list[dict]] = {}
    for u in universes:
        try:
            rows, st = _load_universe(u)
            members[u] = rows
            status.append({"segment": f"universe_{u}", "ok": bool(rows), "n": len(rows),
                           **{k: v for k, v in (st or {}).items() if not isinstance(v, (list, dict))}})
        except Exception as e:
            status.append({"segment": f"universe_{u}", "ok": False, "error": str(e)[:300]})

    tickers = sorted({m["ticker"] for rows in members.values() for m in rows})
    period = TREND_BACKFILL_PERIOD if backfill else TREND_PRICE_PERIOD
    px = _fetch_prices(tickers, period, status, "prices") if tickers else {}

    results = {}
    for u in universes:
        if u not in members:
            continue
        seg_status = list(status)
        try:
            excl = {}
            if u == "etfs":
                etf_px = {m["ticker"]: px[m["ticker"]] for m in members[u] if m["ticker"] in px}
                excl = _leverage(etf_px)
                seg_status.append({"segment": "leverage_backstop", "ok": True, "n_excluded": len(excl)})
            n_priced = sum(1 for m in members[u] if m["ticker"] in px)
            cov = round(n_priced / len(members[u]), 4) if members[u] else 0.0
            seg_status.append({"segment": "coverage", "ok": cov >= 0.85, "n": n_priced,
                               "requested": len(members[u]), "coverage": cov})
            seg = run_universe(u, members[u], px, today, backfill=backfill, exclusions=excl)
            seg_status.append(seg)
            results[u] = seg
            print(f"[trend] {u}: as_of={seg['as_of']} n={seg['n']} scored={seg['n_scored']} "
                  f"hist+{seg['history_dates_added']}d/{seg['history_rows_added']}r "
                  f"events+{seg['events_added']}")
        except Exception as e:
            seg_status.append({"segment": u, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            traceback.print_exc()
        save(u, "status", _scrub({"date": today.isoformat(), "checked": today.isoformat(),
                                  "is_trading_day": True, "action": action,
                                  "disclaimer": DISCLAIMER, "segments": seg_status}))
    return {"ok": bool(results), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "backfill"])
    ap.add_argument("--universe", default="all", choices=["all", *TREND_UNIVERSES])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    universes = TREND_UNIVERSES if args.universe == "all" else (args.universe,)
    run(action=args.action, universes=universes, force=args.force)


if __name__ == "__main__":
    main()

"""Nightly VOL SCREENER build.

    python -m zenith.volscreen.compute --action auto          # the workflow
    python -m zenith.volscreen.compute --action force --limit 15
    python -m zenith.volscreen.compute --offline tests/fixtures/volscreen --tickers SPY AAPL

Pipeline per underlying: Cboe chain -> parity forward + IV from mid -> smiles,
constant-maturity term, delta / moneyness surface -> earnings event variance
(ex-earnings IV, implied move) -> liquidity -> realized-vol estimators ->
GARCH / GJR term forecasts -> VRP per tenor. Then, against the stored
history: IV rank / percentile, per-tenor sigma bands, VRP z / slope / state,
liquidity class; then the checklist; then correlation baskets.

`auto` runs once per trading day from 15:30 ET: delayed quotes are then
~15:15-15:30 live markets. After the bell market makers pull their quotes and
spreads balloon, so a late run is flagged `post-close` (liquidity spreads are
then not representative; IVs still are).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (EPHEMERIS_PX_DIR, THETA_FILES, VOLSCREEN_DELTAS, VOLSCREEN_FILES,
                      VOLSCREEN_LIQ, VOLSCREEN_STATE, VOLSCREEN_TENORS)
from . import (checklist, cboe, clusters, garch, history, iv, liquidity, load, realized, save,
               surface, universe, vrp)

TEN = list(VOLSCREEN_TENORS)
ROUND = 5


def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _r(x, n: int = ROUND):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def _rd(d: dict, n: int = ROUND) -> dict:
    return {str(k): _r(v, n) for k, v in d.items()}


def gate(action: str, status: dict) -> tuple[bool, str]:
    if action == "force":
        return True, "forced"
    from ..pretom.calendar import is_trading_day
    now = _now_et()
    if not is_trading_day(now.date()):
        return False, "not a trading day"
    if now.hour * 60 + now.minute < 15 * 60 + 30:
        return False, "before 15:30 ET"
    if status.get("quote_date") == now.date().isoformat() and status.get("ok"):
        return False, "already built for today"
    return True, "scheduled"


# ------------------------------------------------------------------ inputs --

def risk_free() -> float:
    try:
        from ..cas.sources.fred import get_series
        out, _ = get_series(["DTB3"], limit=5, cache_key="volscreen_fred")
        pts = out.get("DTB3") or []
        if pts:
            return float(pts[-1]["value"]) / 100.0
    except Exception:
        pass
    return 0.04


def price_bars(uni: list[dict], store) -> dict[str, pd.DataFrame]:
    """Daily OHLCV per ticker: the release-asset price store, yfinance for the rest."""
    out: dict[str, pd.DataFrame] = {}
    need_yf: dict[str, tuple[str, float]] = {}
    for u in uni:
        df = None
        if store is not None and u["kind"] != "index":
            try:
                df = store.bars(u["ticker"])
            except Exception:
                df = None
        if df is not None and len(df) >= 60:
            out[u["ticker"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        else:
            need_yf[u["ticker"]] = (u["yf"], u["scale"])
    if need_yf:
        try:
            from ..cas.sources.prices import get_history
            syms = sorted({v[0] for v in need_yf.values()})
            data, _ = get_history(syms, period="5y")
        except Exception:
            data = {}
        for t, (sym, sc) in need_yf.items():
            df = data.get(sym)
            if df is not None and len(df):
                d = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
                d[["open", "high", "low", "close"]] *= sc
                d.index = pd.to_datetime(d.index).tz_localize(None)
                out[t] = d
    return out


def session(qtime: datetime) -> str:
    m = qtime.hour * 60 + qtime.minute
    return "live" if 9 * 60 + 30 <= m < 16 * 60 + 5 else "post-close" if m >= 16 * 60 + 5 else "pre-open"


def with_today(df: pd.DataFrame | None, u: dict, qdate: date, final: bool = True) -> pd.DataFrame | None:
    """Append the chain payload's own daily bar when the store is a day behind
    (only once the day's bar is final: a 15:30 bar would understate the range)."""
    if not final or not u.get("close") or not u.get("open") or not u.get("high") or not u.get("low"):
        return df
    row = pd.DataFrame([[u["open"], u["high"], u["low"], u["close"], u.get("volume") or 0.0]],
                       columns=["open", "high", "low", "close", "volume"], index=[pd.Timestamp(qdate)])
    if df is None or not len(df):
        return row
    last = pd.Timestamp(df.index[-1]).normalize()
    if last < pd.Timestamp(qdate):
        return pd.concat([df, row])
    return df


def earnings_book(tickers: list[str], refresh_days: int = 7) -> dict[str, list[str]]:
    """Report dates per stock: committed cache, THETA's calendar, then yfinance weekly."""
    book = load("earnings", {}) or {}
    dates = book.get("dates", {})
    checked = book.get("checked", {})
    theta_dates = {}
    try:
        theta_dates = json.loads(THETA_FILES["earnings"].read_text(encoding="utf-8")).get("dates", {})
    except Exception:
        pass
    today = date.today()
    cutoff = (today - timedelta(days=3 * 366)).isoformat()
    for t in tickers:
        last = checked.get(t)
        stale = not last or (today - date.fromisoformat(last)).days >= refresh_days
        if t in theta_dates and not stale:
            continue
        if not stale and t in dates:
            continue
        got = theta_dates.get(t)
        if not got or stale:
            try:
                from ..vault.repro.data import earnings_dates
                got = earnings_dates(t) or got
            except Exception:
                pass
        if got:
            dates[t] = [d for d in got if d >= cutoff]
        checked[t] = today.isoformat()
    book = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "yfinance get_earnings_dates (weekly refresh) + THETA RUN calendar",
            "dates": {t: dates[t] for t in sorted(dates)}, "checked": checked}
    save("earnings", book)
    return book["dates"]


def next_event(dates: list[str], qdate: date) -> date | None:
    fut = [date.fromisoformat(d) for d in dates or [] if date.fromisoformat(d) >= qdate]
    return min(fut) if fut else None


def vol_indices() -> dict[str, pd.Series]:
    from ..stratidx.fetch import cboe_csv
    names = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "VXN", "RVX", "VXD", "VXEEM", "GVZ", "OVX"]
    return {n: cboe_csv(n) for n in names}


# ----------------------------------------------------------- one ticker ----

def _mask(bars: pd.DataFrame, dates: list[str]) -> pd.Series | None:
    if not dates:
        return None
    from ..theta.ivmodel import reaction_mask
    return reaction_mask(bars.index, dates)


def _garch_returns(bars: pd.DataFrame, mask) -> np.ndarray:
    r = np.log(bars["close"].astype(float)).diff().dropna()
    if mask is not None:
        m = mask.reindex(r.index, fill_value=False)
        sd = float(r[~m].std())
        if sd > 0:
            r = r.where(~m, r.clip(-3 * sd, 3 * sd))         # winsorise earnings jumps
    return r.to_numpy(float)


def analyse(meta: dict, js: dict, bars: pd.DataFrame | None, earn: list[str], r_free: float,
            warm: dict) -> tuple[dict, dict, list, dict]:
    """(row, history row, surface-history rows, garch warm starts) for one underlying."""
    t = meta["ticker"]
    und = cboe.underlying(js)
    qtime = cboe.quote_time(js)
    qdate = qtime.date()
    spot = und["spot"]
    chain = cboe.chain_frame(js, t)
    px = iv.price_chain(chain, spot, qtime, r_free)
    sl = surface.slices(px)
    if not sl:
        raise ValueError("no usable expiries")
    tm, ext = surface.term(sl)
    gd = surface.grid(sl, "delta", len(VOLSCREEN_DELTAS))
    gm = surface.grid(sl, "mny", 17)
    ev_date = next_event(earn, qdate) if meta["kind"] == "stock" else None
    ev = surface.event_variance(sl, ev_date)
    ev_T = ((pd.Timestamp(ev_date) - pd.Timestamp(qdate)).days + 0.5) / 365.0 if ev_date else None
    ivx = surface.ex_event_term(tm, ev, ev_T)
    fb = surface.front_back(sl)
    sk = surface.skew_metrics(gd)

    bars = with_today(bars, und, qdate, final=session(qtime) != "live")
    mask = _mask(bars, earn) if (bars is not None and meta["kind"] == "stock") else None
    rv = realized.estimator_set(bars, mask) if bars is not None else {}
    gfit, gtm, jtm, new_warm = None, {}, {}, {}
    if bars is not None and len(bars) > 200:
        rets = _garch_returns(bars, mask)
        gfit = garch.fit(rets, "garch", x0=(warm or {}).get("garch"))
        jfit = garch.fit(rets, "gjr", x0=(warm or {}).get("gjr"))
        if gfit.ok:
            gtm = garch.term_structure(gfit, TEN)
            new_warm["garch"] = gfit.theta
        if jfit.ok:
            jtm = garch.term_structure(jfit, TEN)
            new_warm["gjr"] = jfit.theta
        if not gtm and jtm:
            gtm = jtm
    if not gtm and rv.get("ewma"):
        gtm = {k: rv["ewma"] for k in TEN}                     # EWMA fallback, flat
    vsnap = vrp.snapshot(ivx, rv, gtm)
    vt = vrp.per_tenor(ivx, gtm)
    liq = liquidity.metrics(px, spot, bars)

    hist_move = None
    if ev_date is not None and bars is not None and earn:
        from ..theta.ivmodel import earnings_moves
        mv = earnings_moves(bars["close"], earn).dropna().tail(8)
        hist_move = float(mv.mean()) if len(mv) else None
    iv30 = tm.get(30)
    ratio = (tm[30] / tm[90]) if tm.get(90) and np.isfinite(tm[90]) and tm[90] > 0 else float("nan")
    ratio_x = (ivx[30] / ivx[90]) if ivx.get(90) and np.isfinite(ivx[90]) and ivx[90] > 0 else float("nan")
    kink = bool(ev is not None and ev_date is not None and (ev_date - qdate).days <= 90 and ev["e2"] > 0
                and np.isfinite(ratio_x) and ratio_x <= 1.02)
    prev = und.get("prev")
    row = {
        "ticker": t, "kind": meta["kind"], "name": meta.get("name", t), "sector": meta.get("sector", ""),
        "note": meta.get("note", ""), "spot": _r(spot, 4), "chg": _r(spot / prev - 1) if prev else None,
        "source": js.get("source", "cboe"), "quote_date": qdate.isoformat(),
        "iv": _rd(tm), "ivx": _rd(ivx), "ext": ext, "iv30_cboe": _r(und.get("iv30_cboe")),
        "front": _r(fb.get("front")), "front_dte": fb.get("front_dte"), "back": _r(fb.get("back")),
        "back_dte": fb.get("back_dte"), "ratio_30_90": _r(ratio), "ratio_30_90_x": _r(ratio_x),
        "slope_30_90": _r((tm.get(90, np.nan) - tm.get(30, np.nan))),
        "contango": bool(np.isfinite(ratio) and ratio <= 1.0), "kink_explained": kink,
        "cal_viol": surface.calendar_violations(sl),
        "rv": {k: _r(v) for k, v in rv.items()},
        "garch": _rd(gtm), "gjr": _rd(jtm),
        "garch_params": ({"alpha": _r(gfit.alpha, 4), "beta": _r(gfit.beta, 4), "rho": _r(gfit.rho, 4),
                          "lr_vol": _r(math.sqrt(gfit.var_lr * 252) / 100.0)} if gfit and gfit.ok else None),
        "vrp": {k: _r(v) for k, v in vsnap.items()}, "vrp_t": _rd(vt),
        "skew": {k: _r(v) for k, v in sk.items()},
        "earn": {"next": ev_date.isoformat() if ev_date else None,
                 "days": (ev_date - qdate).days if ev_date else None,
                 "move": _r(ev["move"]) if ev else None, "method": ev["method"] if ev else None,
                 "hist_move": _r(hist_move)},
        "em30": _r(spot * iv30 * math.sqrt(30 / 365.0), 3) if iv30 and np.isfinite(iv30) else None,
        "liq": {k: (_r(v) if isinstance(v, float) else v) for k, v in liq.items()},
        "slices": [[s["dte"], _r(s["atm"], 4)] for s in sl if s["dte"] <= 400],
    }
    hrow = {"ticker": t, "date": pd.Timestamp(qdate), "src": 0, "spot": spot,
            **{f"iv_{k}": tm.get(k) for k in TEN}, "ivx_30": ivx.get(30),
            "skew25": sk.get("skew25"), "bf25": sk.get("bf25"),
            "rv_cc20": rv.get("cc20"), "rv_yz20": rv.get("yz20"), "garch_30": gtm.get(30),
            "vrp_30": vsnap["vrp30"], "opt_vol": liq.get("opt_vol"), "oi": liq.get("oi"),
            "atm_spread": liq.get("atm_spread")}
    srows = [{"ticker": t, "date": pd.Timestamp(qdate), "tenor": k,
              **{f"d{j}": gd[i][j] for j in range(len(VOLSCREEN_DELTAS))}} for i, k in enumerate(TEN)]
    surf = {"delta": [[_r(x, 4) for x in rw] for rw in gd], "mny": [[_r(x, 4) for x in rw] for rw in gm],
            "F": _r(sl[0]["F"], 4), "spot": _r(spot, 4)}
    row["_surf"] = surf
    return row, hrow, srows, new_warm


# -------------------------------------------------------- history pass -----

def enrich(row: dict, h: pd.DataFrame, cross: dict) -> dict:
    """Ranks, sigma bands, VRP history stats + state, liquidity class, checklist."""
    h = h.sort_values("date")
    src = h["src"].fillna(0).astype(int)
    obs = int((src == 0).sum())
    s30 = h.set_index("date")["iv_30"]
    ivr, ivp, n = vrp.iv_rank(s30)
    row["ivr"], row["ivp"], row["ivr_n"] = _r(ivr), _r(ivp), n
    row["hist_obs"] = obs
    row["ivr_src"] = ("observed" if obs >= 252 else
                      "index-seeded" if (src == 1).any() else
                      "modeled" if (src == 2).any() else "building")
    row["iv_z"] = {str(k): _r(vrp.sigma_z(h.set_index("date")[f"iv_{k}"])) for k in TEN}
    row["iv_bands"] = {}
    for k in TEN:
        s = h.set_index("date")[f"iv_{k}"].dropna().iloc[-252:]
        if len(s) >= 20:
            row["iv_bands"][str(k)] = {"mean": _r(s.mean()), "sd": _r(s.std(ddof=1)),
                                       "lo": _r(s.min()), "hi": _r(s.max())}
    vs = h.set_index("date")["vrp_30"]
    stats = vrp.history_stats(vs)
    states = checklist.vrp_states(vs.to_numpy())
    row["vrp"].update({"z": _r(stats["z"]), "slope5": _r(stats["slope5"], 6), "slope10": _r(stats["slope10"], 6),
                       "n": stats["n"]})
    row["state"] = states[-1] if states else "NEUTRAL"
    row["state_days"] = checklist.days_in_state(states)
    tail = h.tail(60)
    code = {s: i for i, s in enumerate(checklist.STATES)}
    row["vrp_hist"] = {"d0": tail["date"].iloc[0].date().isoformat() if len(tail) else None,
                       "d": [d.strftime("%m%d") for d in tail["date"]],
                       "v": [_r(x, 4) for x in tail["vrp_30"]],
                       "s": [code[x] for x in states[-len(tail):]] if states else [],
                       "src": [int(x) for x in tail["src"].fillna(0)]}
    # skew percentile: own history when there is enough, else the cross-section
    sk = h.set_index("date")["skew25"].dropna()
    today_sk = (row.get("skew") or {}).get("skew25")
    if len(sk) >= 40 and today_sk is not None:
        row["skew"]["pct"] = _r(float((sk.iloc[:-1] < today_sk).mean()))
        row["skew"]["pct_src"] = "own"
    elif today_sk is not None and cross.get("skews"):
        arr = np.asarray(cross["skews"])
        row["skew"]["pct"] = _r(float((arr < today_sk).mean()))
        row["skew"]["pct_src"] = "cross-section"
    # liquidity history
    past = h.loc[(src == 0) & (h["date"] < h["date"].max()), "opt_vol"].tolist()
    liq = liquidity.with_history(row["liq"], past)
    live = cross.get("session") != "post-close"
    liq["stars"] = liquidity.stars(liq, use_spread=live)
    liq["spread_rated"] = live
    liq["class"] = liquidity.classify(liq, cross.get("turnover"))
    liq["score"] = liquidity.score(liq, use_spread=live)
    row["liq"] = {k: (_r(v) if isinstance(v, float) else v) for k, v in liq.items()}
    flat = {"stars": liq["stars"], "vrp30": row["vrp"].get("vrp30"), "iv_rv": row["vrp"].get("iv_rv"),
            "state": row["state"], "ivr": row["ivr"], "ivp": row["ivp"],
            "ratio_30_90": row.get("ratio_30_90"), "kink_explained": row.get("kink_explained"),
            "yz10": row["rv"].get("yz10"), "yz60": row["rv"].get("yz60"),
            "earn_days": row["earn"].get("days"), "skew_pct": row["skew"].get("pct"),
            "vrp_t": row.get("vrp_t", {})}
    row["check"] = checklist.evaluate(flat)
    return row


# ---------------------------------------------------------------- main -----

def run(action: str = "auto", limit: int | None = None, tickers: list[str] | None = None,
        offline: str | None = None, write_history: bool = True) -> dict:
    t0 = time.time()
    status = load("status", {}) or {}
    ok, why = gate("force" if offline else action, status)
    if not ok:
        print(f"[volscreen] skip: {why}")
        return {"skipped": why}

    store = None
    if not offline:
        try:
            from ..ephemeris.store_px import PxStore
            store = PxStore()
            if not store.available():
                store = None
        except Exception:
            store = None
    uni = universe.candidates(store)
    if tickers:
        known = {u["ticker"]: u for u in uni}
        meta = {r["ticker"]: r for r in universe._rows()}
        uni = [known.get(t) or {"ticker": t, "kind": "stock", "name": meta.get(t, {}).get("name", t),
                                "sector": meta.get(t, {}).get("sector", ""), "note": "", "yf": t, "scale": 1.0}
               for t in tickers]
    if limit:
        uni = uni[:limit]
    print(f"[volscreen] universe {len(uni)}")

    r_free = 0.04 if offline else risk_free()
    bars = {} if offline else price_bars(uni, store)
    earn = {} if offline else earnings_book([u["ticker"] for u in uni if u["kind"] == "stock"])
    vidx = {} if offline else vol_indices()

    hist_path = VOLSCREEN_FILES["term_local"] if not offline else None
    surf_path = VOLSCREEN_FILES["surf_local"] if not offline else None
    hist = history.load_term(hist_path)
    surf_hist = history.load_surf(surf_path)
    prev_rows, prev_surf = len(hist), len(surf_hist)
    expected = int(status.get("history_rows") or 0)
    if write_history and not offline and prev_rows < 0.9 * expected:
        # the release download failed: writing now would clobber a year of IV history
        raise SystemExit(f"[volscreen] IV history has {prev_rows} rows, status says {expected}; "
                         "refusing to run without it")
    warm_all = load("garch", {}) or {}

    fetch = cboe.Fetcher(offline_dir=Path(offline) if offline else None)
    rows, hrows, srows, failures, warm_new, sources = [], [], [], {}, {}, {"cboe": 0, "yfinance": 0}
    quote_ts, sess = None, None
    for i, u in enumerate(uni):
        t = u["ticker"]
        js = fetch.get(t)
        if js is None and fetch.tripped and u["kind"] != "index" and not offline:
            from . import yfchain
            js = yfchain.payload(t)
        if js is None:
            failures[t] = fetch.errors.get(t, "no chain")
            continue
        try:
            row, hrow, sr, wm = analyse(u, js, bars.get(t), earn.get(t, []), r_free, warm_all.get(t))
        except Exception as e:
            failures[t] = f"{type(e).__name__}: {str(e)[:100]}"
            continue
        sources["yfinance" if js.get("source") == "yfinance" else "cboe"] += 1
        quote_ts = quote_ts or cboe.quote_time(js).strftime("%Y-%m-%d %H:%M")
        sess = sess or session(cboe.quote_time(js))
        rows.append(row)
        hrows.append(hrow)
        srows.extend(sr)
        if wm:
            warm_new[t] = wm
        if (i + 1) % 25 == 0:
            print(f"[volscreen] {i + 1}/{len(uni)} ok={len(rows)} fail={len(failures)} "
                  f"{time.time() - t0:.0f}s")
    if not rows:
        save("status", {**status, "ok": False, "run_utc": _utc(), "error": "no chains",
                        "failures": failures})
        raise SystemExit("[volscreen] no chains parsed")

    # the CDN can serve a quiet symbol's file from a previous session: keep its true
    # date in the history, and flag it in the screen rather than pass it off as today's
    qdate = max(pd.Timestamp(h["date"]).date() for h in hrows)
    stale = [r["ticker"] for r in rows if r["quote_date"] < qdate.isoformat()]
    for r in rows:
        r["stale"] = r["ticker"] in stale
    today_df = pd.DataFrame(hrows)
    # seed tickers that have no history at all (index vol families, else THETA's model)
    theta_path = EPHEMERIS_PX_DIR / "theta.parquet"
    seeds = []
    have = set(hist["ticker"].unique()) if len(hist) else set()
    for u in uni:
        t = u["ticker"]
        if t in have or t not in set(today_df["ticker"]):
            continue
        b = bars.get(t)
        sd = history.seed_from_indices(t, vidx, b["close"] if b is not None else None)
        if sd.empty and not offline:
            sd = history.seed_from_theta(t, theta_path)
        if not sd.empty:
            seeds.append(sd[sd["date"] < pd.Timestamp(qdate)])
    hist = history.append(hist, pd.concat(seeds, ignore_index=True) if seeds else None)
    hist = history.append(hist, today_df)
    surf_hist = history.append_surf(surf_hist, pd.DataFrame(srows))

    # cross-sectional context for day-one proxies
    turn = [r["liq"].get("opt_vol", 0) / r["liq"]["oi"] for r in rows if r["liq"].get("oi")]
    cross = {"turnover": float(np.median(turn)) if turn else None, "session": sess,
             "skews": [r["skew"].get("skew25") for r in rows if r["skew"].get("skew25") is not None]}
    by_t = {t: g for t, g in hist.groupby("ticker")}
    rows = [enrich(r, by_t.get(r["ticker"], today_df[today_df["ticker"] == r["ticker"]]), cross) for r in rows]

    # core liquid basket
    core = sorted((r for r in rows if r["liq"].get("class") == "CORE" and r["liq"].get("stars", 0) >= 4),
                  key=lambda r: -(r["liq"].get("score") or 0))[: VOLSCREEN_LIQ["core_n"]]
    core_set = {r["ticker"] for r in core}
    for r in rows:
        r["core"] = r["ticker"] in core_set

    # correlation baskets over the liquid names
    closes = {r["ticker"]: bars[r["ticker"]]["close"] for r in rows
              if r["ticker"] in bars and r["liq"].get("stars", 0) >= 2 and r["kind"] != "index"}
    scores = {r["ticker"]: r["check"]["sell"]["score"] for r in rows}
    eligible = {r["ticker"] for r in rows if r["liq"].get("stars", 0) >= 3 and not r.get("note")}
    baskets = {}
    for w in (60, 120, 250):
        try:
            baskets[str(w)] = clusters.build(closes, scores, eligible, window=w)
        except Exception as e:
            baskets[str(w)] = {"window": w, "error": str(e)[:120], "clusters": [], "basket": []}
    lab = {}
    b120 = baskets.get("120", {})
    for c in b120.get("clusters", []):
        for m in c["members"]:
            lab[m] = c["id"]
    for r in rows:
        r["cluster"] = lab.get(r["ticker"])
        r["in_basket"] = r["ticker"] in set(b120.get("basket", []))

    surfaces = {r["ticker"]: r.pop("_surf") for r in rows}
    market = market_context(vidx, rows)
    as_of = qdate.isoformat()
    save("screen", {"as_of": as_of, "quote_ts": quote_ts, "session": sess, "generated": _utc(), "tenors": TEN,
                    "deltas": [d[0] for d in VOLSCREEN_DELTAS], "market": market,
                    "core": [r["ticker"] for r in core], "state_cfg": VOLSCREEN_STATE,
                    "rows": sorted(rows, key=lambda r: (-(r["liq"].get("score") or 0)))})
    save("surfaces", {"as_of": as_of, "tenors": TEN, "deltas": [d[0] for d in VOLSCREEN_DELTAS],
                      "mny": [round(0.80 + 0.025 * i, 3) for i in range(17)], "surf": surfaces})
    save("baskets", {"as_of": as_of, **baskets})
    save("universe", {"as_of": as_of, "rows": [{k: u[k] for k in ("ticker", "kind", "name", "sector", "note")}
                                                for u in uni]})
    save("garch", {**warm_all, **warm_new})
    sizes = {}
    if write_history and not offline:
        sizes["term"] = history.write(hist, VOLSCREEN_FILES["term_local"], prev_rows, history.TERM_COLS)
        sizes["surf"] = history.write(surf_hist, VOLSCREEN_FILES["surf_local"], prev_surf,
                                      ["tenor"] + history.SURF_COLS)
    st = {"ok": True, "quote_date": as_of, "quote_ts": quote_ts, "session": sess, "run_utc": _utc(),
          "n_universe": len(uni), "n_ok": len(rows), "failures": failures, "sources": sources,
          "breaker_tripped": fetch.tripped, "stale": stale, "seconds": round(time.time() - t0, 1),
          # only a WRITTEN history counts: the next run refuses to start with less
          "history_rows": (int(len(hist)) if sizes and os.environ.get("GITHUB_ACTIONS")
                           else int(status.get("history_rows") or 0)),
          "history_tickers": int(hist["ticker"].nunique()),
          "observed_days": int(hist.loc[hist["src"] == 0, "date"].nunique()),
          "seeded": len(seeds), "asset_bytes": sizes, "r_free": r_free}
    save("status", st)
    if sizes:                                   # tells the workflow there is something to publish
        flag = VOLSCREEN_FILES["status"].parent / "cache" / "built.flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(as_of, encoding="utf-8")
    print(f"[volscreen] done {len(rows)}/{len(uni)} in {st['seconds']}s; failures {len(failures)}")
    return st


def market_context(vidx: dict[str, pd.Series], rows: list[dict]) -> dict:
    def last(n):
        s = vidx.get(n)
        return float(s.iloc[-1]) if s is not None and len(s) else None
    def pct(n):
        s = vidx.get(n)
        if s is None or len(s) < 260:
            return None
        w = s.iloc[-252:]
        return float((w.iloc[:-1] < w.iloc[-1]).mean())
    vix, v3m = last("VIX"), last("VIX3M")
    vr = [r["vrp"].get("vrp30") for r in rows if r["vrp"].get("vrp30") is not None]
    st = [r.get("state") for r in rows]
    return {"vix": vix, "vix9d": last("VIX9D"), "vix3m": v3m, "vix6m": last("VIX6M"), "vix1y": last("VIX1Y"),
            "vvix": last("VVIX"), "vix_ratio": (vix / v3m) if vix and v3m else None, "vix_pct": pct("VIX"),
            "median_vrp30": float(np.median(vr)) if vr else None,
            "pct_all_clear": (st.count("ALL-CLEAR") / len(st)) if st else None,
            "states": {s: st.count(s) for s in checklist.STATES},
            "n_surge": sum(1 for r in rows if r["liq"].get("class") == "SURGE")}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="zenith.volscreen.compute")
    ap.add_argument("--action", default="auto", choices=["auto", "force"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--offline", default=None, help="directory of <TICKER>.json chain payloads")
    ap.add_argument("--no-history-write", action="store_true")
    ap.add_argument("--check", action="store_true", help="print RUN or SKIP for the gate and exit")
    a = ap.parse_args(argv)
    if a.check:
        ok, why = gate(a.action, load("status", {}) or {})
        print("RUN" if ok else "SKIP")
        return
    run(a.action, a.limit, a.tickers, a.offline, write_history=not a.no_history_write)


if __name__ == "__main__":
    main()

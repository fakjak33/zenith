"""MOMENTUM compute orchestrator.

  python -m zenith.mom.compute --action auto        (nightly path)
  python -m zenith.mom.compute --action meta         (force a full metadata refresh)
  python -m zenith.mom.compute --action rebuild      (re-seed the membership archive from git)
  python -m zenith.mom.compute --action backfill --weeks 156   (3y weekly history, run once locally)

Nightly path: gate on the trading calendar, pull the whole Russell 1000 +
SPY's daily OHLC, build the five factors per stock, combine them into the
composite, rank/decile/side the universe, aggregate by sector/industry,
compute diagnostics (factor correlation + any matured IC), append composite
history + decile picks, evaluate matured pick horizons, and write every
artifact. Views read only these committed artifacts (except the individual-
stock GMMA chart, which fetches on demand).
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date

import pandas as pd

from . import DISCLAIMER, FACTORS, save, load
from . import engine, factors, history
from . import universe as mom_universe
from .mvt import compute as mvt_compute
from .mvt import horizons as mvt_horizons
from .mvt import save as mvt_save, load as mvt_load
from ..pretom import calendar as cal
from ..cas.sources import prices
from ..edge.common import assemble
from ..config import MOM_WEIGHTS, MOM_WEIGHT_MODE


def _run_mvt(px: dict) -> dict:
    """Thin, monkeypatchable seam around mvt_compute.run_auto() -- mirrors
    `_fetch_prices` below exactly, for the same reason: mvt's OWN internal
    ETF universe/price pull is real network I/O (yfinance metadata +
    ~900-ticker OHLCV), and every offline test in this suite must be able to
    stub it out the same way they already stub _fetch_prices, rather than
    silently paying that cost (or requiring network) on every test run."""
    return mvt_compute.run_auto(px)


def _fetch_prices(tickers: list[str], period: str, status: list[dict], label: str):
    px, st = prices.get_history(tickers, period=period)
    missing = [t for t in tickers if t not in px]
    if len(missing) > max(5, len(tickers) // 20):
        time.sleep(15)
        px2, _ = prices.get_history(missing, period=period, max_age_hours=0)
        px.update(px2)
    status.append({"segment": label, "ok": bool(px), "n": len(px), "requested": len(tickers),
                   "error": st.get("error", "")})
    return px


def _scrub(obj):
    """Recursively replace non-finite floats with None -- json.dumps emits
    bare NaN/Infinity, which is not valid JSON and breaks every downstream
    reader. Every Zenith feature that serializes pandas-derived floats does
    this; MOMENTUM has more float-valued fields than most."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _build_rows(universe: list[dict], px: dict, meta: dict) -> list[dict]:
    """One row per universe constituent: priced-and-scored, priced-but-
    insufficient-history, or entirely unpriced. Every row survives into
    scores_latest.json (excluded rows carry composite=None + a reason) so a
    delisting or a data gap is visible, never silently dropped."""
    rows = []
    for u in universe:
        t = u["ticker"]
        m = meta.get(t, {})
        df = px.get(t)
        row = {
            "ticker": t, "name": u.get("name") or m.get("name") or t,
            "sector": m.get("sector") or u.get("sector") or "",
            "industry": m.get("industry") or "", "mktcap": m.get("mktcap"),
            "weight_pct": u.get("weight_pct"),
        }
        if df is None or df.empty:
            row["raw"] = None
            row["exclusion_reason"] = "no_price_data"
        else:
            raw = factors.build_all(df)
            row["raw"] = raw
            if raw is None:
                row["exclusion_reason"] = f"insufficient_history(<{factors.MIN_BARS}d)"
        rows.append(row)
    return rows


def _sector_aggregates(scored: list[dict]) -> dict:
    def _agg(key: str) -> dict:
        buckets: dict[str, list[float]] = {}
        for r in scored:
            k = r.get(key) or "Unknown"
            buckets.setdefault(k, []).append(r["composite"])
        out = {}
        for k, vals in buckets.items():
            s = pd.Series(vals, dtype=float)
            out[k] = {
                "n": int(len(s)), "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "pct_bullish": round(float((s >= 5).mean()), 4),
                "pct_bearish": round(float((s <= -5).mean()), 4),
                "n_extreme_bullish": int((s >= 15).sum()), "n_extreme_bearish": int((s <= -15).sum()),
                "dispersion": round(float(s.std()), 3) if len(s) > 1 else 0.0,
            }
        return out
    return {"sectors": _agg("sector"), "industries": _agg("industry")}


def _spearman_ic(pairs: list[tuple]) -> float | None:
    if len(pairs) < 8:
        return None
    df = pd.DataFrame(pairs, columns=["score", "excess"])
    ic = df["score"].corr(df["excess"], method="spearman")
    return None if pd.isna(ic) else round(float(ic), 4)


def _weighting_comparison(rows: list[dict], corr_matrix: dict, active_weights: dict) -> dict:
    """Declared vs equal vs equal-risk-contribution (ERC) factor weights,
    for the Momentum tab's weighting-evaluation lens (the second explicit
    ask alongside adding mvt: re-evaluate weighting by each input's actual
    contribution, not by how much overlapping history it carries). This
    does NOT change today's `composite` field (computed above with
    `active_weights`, "declared" by default) -- it is a parallel, fully
    transparent comparison, exactly like the existing `composite_equal_weight`
    field already is.

    ERC weights are shrunk 50% toward equal weight for the same reason
    mvt/score.py shrinks its horizon-level ERC weights (see that module's
    comment): pure risk-parity has no notion of signal quality, only of
    correlation, and can overweight a noisy-but-uncorrelated input. This is
    the repo's stated preference (spec section 33) for a transparent/simple
    method over an unshrunk "optimal" one when the latter isn't clearly
    statistically defensible on the available data."""
    scored = [r for r in rows if r.get("factor_scores")]
    n = len(scored)
    if n < 10 or not corr_matrix:
        return {"as_of": date.today().isoformat(), "n": n, "note": "insufficient data",
                "declared_weights": MOM_WEIGHTS, "active_weights": active_weights}

    factors_present = [f for f in FACTORS if f in corr_matrix]
    eq_weights = {f: 1.0 / len(factors_present) for f in factors_present}

    # None -> nan explicitly so the frame is float64, not object dtype
    # (an object column full of None triggers a pandas fillna downcasting
    # FutureWarning and is generally worth avoiding here regardless).
    corr_df = pd.DataFrame({a: {b: (corr_matrix[a].get(b) if corr_matrix[a].get(b) is not None else float("nan"))
                                for b in factors_present}
                            for a in factors_present}).fillna(0.0)
    erc = mvt_horizons.erc_weights(corr_df)
    erc_shrunk = {f: round(0.5 * erc["weights"].get(f, 0.0) + 0.5 * eq_weights[f], 6) for f in factors_present}

    # Mean ABSOLUTE per-factor contribution under each scheme -- "how much
    # does this factor typically move a stock's composite" (section 32).
    # Signed mean would be the wrong statistic here: xsec and mvt are both
    # percentile-rank-based transforms, which are mean-ZERO across the
    # universe by construction (half the names are above the median, half
    # below) -- a plain average would make them look like they contribute
    # "almost nothing" even when they swing individual stocks by +/-20,
    # simply because their ups and downs cancel out across the cross-
    # section. ts/breakout/speed/strength are NOT rank-based and can carry
    # a genuine market-wide tilt (e.g. broad bull market -> mostly positive
    # ts_score that day), so this asymmetry is a real, meaningful
    # distinction the table should surface, not an artifact to average away.
    def _avg_contribution(weights: dict) -> dict:
        totals = {f: 0.0 for f in factors_present}
        cnt = 0
        for r in scored:
            fs = r["factor_scores"]
            if not all(f in fs for f in factors_present):
                continue
            for f in factors_present:
                totals[f] += abs(20.0 * weights.get(f, 0.0) * fs[f])
            cnt += 1
        return {f: round(v / cnt, 4) for f, v in totals.items()} if cnt else totals

    return {
        "as_of": date.today().isoformat(),
        "n": n,
        "factors": factors_present,
        "declared_weights": {f: MOM_WEIGHTS.get(f) for f in factors_present},
        "equal_weights": eq_weights,
        "factor_erc_weights": erc["weights"],
        "factor_erc_weights_shrunk": erc_shrunk,
        "erc_converged": erc["converged"],
        "avg_contribution_declared": _avg_contribution(MOM_WEIGHTS),
        "avg_contribution_equal": _avg_contribution(eq_weights),
        "avg_contribution_erc_shrunk": _avg_contribution(erc_shrunk),
        "active_weight_mode": MOM_WEIGHT_MODE,
        "active_weights": active_weights,
        "correlation_matrix": corr_matrix,
    }


def _diagnostics(scored: list[dict], picks_rows: list[dict]) -> dict:
    """Predictive diagnostics per the plan's decision to ship IC/hit-rate/
    correlation now rather than a full portfolio backtest: factor
    correlation matrix (checks the weight-redundancy assumption), composite
    distribution/breadth, and rank information coefficient (Spearman, score
    at entry vs realized SPY-excess) per horizon -- for the composite and,
    once enough tagged picks exist, for each individual factor. Mostly
    empty/None on day one; fills in as `picks.json` accrues evaluated rows."""
    corr = engine.correlations(scored)
    comps = [r["composite"] for r in scored]
    dist = {}
    if comps:
        s = pd.Series(comps, dtype=float)
        dist = {"n": int(len(s)), "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
                "pct_bullish": round(float((s >= 5).mean()), 4), "pct_bearish": round(float((s <= -5).mean()), 4),
                "n_extreme_bullish": int((s >= 15).sum()), "n_extreme_bearish": int((s <= -15).sum())}
    ic_by_horizon, factor_ic_by_horizon = {}, {}
    for h in history.PICK_HORIZONS_TD:
        hk = str(h)
        pairs = [(r["composite"], r["eval"][hk]["excess"]) for r in picks_rows
                 if r["eval"][hk]["evaluated"] and r["eval"][hk]["excess"] is not None
                 and r.get("composite") is not None]
        ic_by_horizon[hk] = _spearman_ic(pairs)
        f_ic = {}
        for f in FACTORS:
            fpairs = [((r.get("factor_scores") or {}).get(f), r["eval"][hk]["excess"])
                      for r in picks_rows if r["eval"][hk]["evaluated"] and r["eval"][hk]["excess"] is not None
                      and r.get("factor_scores") and r["factor_scores"].get(f) is not None]
            f_ic[f] = _spearman_ic(fpairs)
        factor_ic_by_horizon[hk] = f_ic
    return {"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
            "correlation": corr, "distribution": dist,
            "ic_by_horizon": ic_by_horizon, "factor_ic_by_horizon": factor_ic_by_horizon,
            "hit_rate": history.summarize(picks_rows)}


def _write_artifacts(rows: list[dict], today: date, coverage: dict) -> dict:
    scored = [r for r in rows if r.get("composite") is not None]
    assemble(scored, "composite", higher_is_long=True, decile=0.1)   # mutates rank/pctile/decile/side in place

    scores_out = []
    detail_out = {}
    for r in rows:
        raw = r.get("raw") or {}
        out = {
            "ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "industry": r["industry"],
            "mktcap": r.get("mktcap"), "weight_pct": r.get("weight_pct"),
            "composite": r.get("composite"), "composite_equal_weight": r.get("composite_equal_weight"),
            "state": r.get("state"), "rank": r.get("rank"), "pctile": r.get("pctile"),
            "decile": r.get("decile"), "side": r.get("side"),
            "factor_scores": r.get("factor_scores"), "contributions": r.get("contributions"),
            "ts_grid": r.get("ts_grid"), "breakout_grid": r.get("breakout_grid"),
            "bars": raw.get("bars"), "excluded": r.get("composite") is None,
            "exclusion_reason": r.get("exclusion_reason"),
        }
        scores_out.append(out)
        if raw:
            detail_out[r["ticker"]] = {
                "speed": raw.get("speed_raw"), "strength": raw.get("strength_raw"),
                "asof": raw.get("asof"), "last_close": raw.get("last_close"),
            }

    scores_doc = {"as_of": today.isoformat(), "disclaimer": DISCLAIMER, "n": len(scores_out),
                 "n_scored": len(scored), "coverage": coverage, "rows": scores_out}
    save("scores", _scrub(scores_doc), indent=None)
    save("detail", _scrub({"as_of": today.isoformat(), "stocks": detail_out}), indent=None)
    save("sectors", _scrub({"as_of": today.isoformat(), **_sector_aggregates(scored)}))

    added_h = history.append_history(scored, today)
    pick_rows_new = history.make_pick_rows(scored, today.isoformat())
    added_p = history.append_picks(pick_rows_new)

    picks_doc = load("picks", {"rows": []})
    return {"scored": len(scored), "history_added": added_h, "picks_added": added_p,
            "picks_rows": picks_doc.get("rows", [])}


def run_auto(force: bool = False) -> dict:
    today = date.today()
    status: list[dict] = []
    if not cal.is_trading_day(today) and not force:
        save("status", {"date": today.isoformat(), "is_trading_day": False,
                        "disclaimer": DISCLAIMER,
                        "segments": [{"segment": "gate", "ok": True, "note": "non-trading day"}]})
        print(f"[mom] {today} non-trading day — no-op")
        return {"ok": True, "gated": True}

    universe, uni_status = mom_universe.constituents()
    status.append({"segment": "universe", "ok": bool(universe), **uni_status})
    tickers = [u["ticker"] for u in universe]
    mom_universe.membership_archive(tickers, today.isoformat())

    px = _fetch_prices(sorted(set(tickers + ["SPY"])), "5y", status, "prices")
    spy = px.get("SPY")
    spy_close = spy["close"] if spy is not None and not spy.empty else None
    n_priced = sum(1 for t in tickers if t in px)
    coverage = round(n_priced / len(tickers), 4) if tickers else 0.0
    status.append({"segment": "coverage", "ok": coverage >= 0.85, "n": n_priced,
                   "requested": len(tickers), "coverage": coverage})

    meta_status = mom_universe.refresh_metadata(tickers)
    status.append({"segment": "metadata", "ok": True, **meta_status})
    meta = load("meta", {})

    rows = _build_rows(universe, px, meta)
    n_scored_raw = sum(1 for r in rows if r.get("raw") is not None)
    status.append({"segment": "factors", "ok": n_scored_raw > 0, "n": n_scored_raw})

    engine.cross_sectional(rows)

    # --- Multivariate Trend: the 6th factor. Computed as a separate,
    # universe-wide pairwise pass (mvt/) and attached per-ticker BEFORE
    # composite() runs, exactly like ts/xsec's cross-sectional scores above.
    # Fail-soft: mvt_compute.run_auto() never raises (see its own module
    # docstring) -- a bad mvt night simply leaves mvt_score unset on every
    # row, and engine.composite() renormalizes the other five factors'
    # weights for any row missing it, rather than letting a missing 6th
    # factor drag every composite toward zero.
    mvt_result = _run_mvt(px)
    mvt_by_ticker = {r["ticker"]: r for r in mvt_result.get("equities", {}).get("rows", [])}
    for r in rows:
        mv = mvt_by_ticker.get(r["ticker"])
        if mv and mv.get("normalized_score") is not None:
            r["mvt_score"] = mv["normalized_score"]
            r["mvt_raw_score"] = mv.get("raw_score")
    status.append({"segment": "mvt", "ok": bool(mvt_by_ticker), "n": len(mvt_by_ticker),
                   **mvt_result.get("equities", {}).get("status", {})})

    # --- weighting mode: "declared" (default, always the headline) or "erc"
    # (equal-risk-contribution, from the PRIOR run's persisted weighting.json
    # so switching modes never uses same-day information -- no look-ahead).
    # See config.MOM_WEIGHT_MODE's docstring: this flips only when the user
    # reviews the comparison and decides to, never silently.
    active_weights = MOM_WEIGHTS
    if MOM_WEIGHT_MODE == "erc":
        prior = mvt_load("weighting", {})
        prior_erc = prior.get("factor_erc_weights_shrunk")
        if prior_erc and abs(sum(prior_erc.values()) - 1.0) < 1e-6:
            active_weights = prior_erc

    engine.composite(rows, weights=active_weights)

    result = _write_artifacts(rows, today, {"n_universe": len(tickers), "n_priced": n_priced,
                                            "coverage": coverage})
    status.append({"segment": "artifacts", "ok": True, "n": result["scored"]})

    n_eval = 0
    if spy_close is not None:
        n_eval = history.evaluate_pending(px, spy_close, today)
    status.append({"segment": "history", "ok": True, "history_added": result["history_added"],
                   "picks_added": result["picks_added"], "evaluated": n_eval})

    diag = _diagnostics([r for r in rows if r.get("composite") is not None], result["picks_rows"])
    save("diagnostics", _scrub(diag))
    status.append({"segment": "diagnostics", "ok": True, "flagged_pairs": len(diag["correlation"].get("flagged_pairs", []))})

    weighting_doc = _weighting_comparison(rows, diag["correlation"].get("matrix", {}), active_weights)
    mvt_save("weighting", _scrub(weighting_doc))
    status.append({"segment": "weighting", "ok": bool(weighting_doc.get("factor_erc_weights")),
                   "mode": MOM_WEIGHT_MODE})

    save("status", {"date": today.isoformat(), "is_trading_day": True, "disclaimer": DISCLAIMER,
                    "segments": status})
    print(f"[mom] {today} universe={len(tickers)} priced={n_priced} scored={result['scored']} "
          f"hist+{result['history_added']} picks+{result['picks_added']} eval={n_eval}")
    return {"ok": True, "scored": result["scored"]}


def run_backfill(weeks: int = 156, force: bool = False) -> dict:
    """Recompute the composite as-of each of the last `weeks` Fridays, using
    ONLY price data up to that date (no look-ahead), and append to the
    sharded history + pick tracker. Point-in-time membership is used where
    the archive reaches (see mom.universe.membership_asof); earlier dates
    reuse today's constituents and are survivorship-biased -- the UI marks
    the boundary at config.MOM_MEMBERSHIP_START rather than hiding it.
    Intended to run ONCE, locally (not in CI); the nightly Action only ever
    calls --action auto."""
    today = date.today()
    universe, _ = mom_universe.constituents()
    tickers = [u["ticker"] for u in universe]
    meta_all, _ = {}, None
    status: list[dict] = []
    px = _fetch_prices(sorted(set(tickers + ["SPY"])), "10y", status, "prices")
    spy = px.get("SPY")
    spy_close = spy["close"] if spy is not None and not spy.empty else None
    meta = load("meta", {})

    fridays = [d for d in pd.bdate_range(end=today, periods=weeks * 7 + 10, freq="B")
              if d.weekday() == 4]
    fridays = [d.date() for d in fridays if cal.is_trading_day(d.date())][-weeks:]

    total_added_h = total_added_p = 0
    for i, asof in enumerate(fridays):
        pit = mom_universe.membership_asof(asof.isoformat())
        uni_asof = [u for u in universe if u["ticker"] in pit] if pit else universe
        rows = []
        for u in uni_asof:
            t = u["ticker"]
            df = px.get(t)
            m = meta.get(t, {})
            row = {"ticker": t, "name": u.get("name") or t, "sector": m.get("sector") or u.get("sector") or "",
                  "industry": m.get("industry") or "", "mktcap": m.get("mktcap"), "weight_pct": u.get("weight_pct")}
            if df is None or df.empty:
                row["raw"] = None
            else:
                sliced = df.loc[:pd.Timestamp(asof)]
                row["raw"] = factors.build_all(sliced)
            rows.append(row)

        engine.cross_sectional(rows)
        engine.composite(rows)
        scored = [r for r in rows if r.get("composite") is not None]
        if not scored:
            continue
        assemble(scored, "composite", higher_is_long=True, decile=0.1)
        added_h = history.append_history(scored, asof, full=True)
        added_p = history.append_picks(history.make_pick_rows(scored, asof.isoformat()))
        total_added_h += added_h
        total_added_p += added_p
        print(f"[mom backfill] {i + 1}/{len(fridays)} {asof} n_scored={len(scored)} "
              f"hist+{added_h} picks+{added_p}")

    n_eval = history.evaluate_pending(px, spy_close, today) if spy_close is not None else 0
    print(f"[mom backfill] done. weeks={len(fridays)} hist+{total_added_h} picks+{total_added_p} eval={n_eval}")
    return {"ok": True, "weeks": len(fridays), "history_added": total_added_h,
            "picks_added": total_added_p, "evaluated": n_eval}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "backfill", "meta", "rebuild"])
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.action == "backfill":
        run_backfill(weeks=args.weeks, force=args.force)
    elif args.action == "meta":
        universe, _ = mom_universe.constituents()
        tickers = [u["ticker"] for u in universe]
        res = mom_universe.refresh_metadata(tickers, max_per_run=len(tickers))
        print(f"[mom] meta refresh: {res}")
    elif args.action == "rebuild":
        res = mom_universe.seed_membership_from_git()
        print(f"[mom] membership seeded from git: {res}")
    else:
        run_auto(force=args.force)


if __name__ == "__main__":
    main()

"""ETF MOMENTUM compute orchestrator.

  python -m zenith.etfmom.compute --action auto     (nightly path)

Nightly path: gate on the trading calendar, pull the whole ETF universe's
daily OHLC, build the same five per-instrument factors MOMENTUM builds, attach
the 6th (Multivariate Trend) from the artefact mom.yml already committed,
combine into the same -20..+20 composite, rank/decile/side, aggregate by asset
class and category, compute diagnostics, append composite history + decile
picks, evaluate matured pick horizons, and write every artefact. Views read
only these committed artefacts (except the individual-ETF GMMA chart, which
fetches that one ticker on demand).

Stage for stage this mirrors mom/compute.py, and where it differs the reason
is stated inline. The factor math itself is imported, never reimplemented.

There is deliberately no `--action backfill`. MOMENTUM has one, but here it
would be actively misleading: the 6th factor has no historical values to read,
so every backfilled row would be a five-factor composite plotted on the same
axis as today's six-factor one. The history chart honestly starts at one point
and fills in forward. (`--action meta` and `--action rebuild` are inapplicable
too -- metadata is mom.yml's job, and there is no git history of an ETF
universe file to re-seed point-in-time membership from.)
"""

from __future__ import annotations

import argparse
import math
from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save
from . import history as etf_history
from . import mvt_link
from . import universe as etf_universe
from ..cas.sources import prices
from ..config import ETFMOM_PRICE_PERIOD, MOM_WEIGHTS
from ..edge.common import assemble
from ..mom import FACTORS
from ..mom import engine, factors
from ..mom.mvt import panel as mvt_panel
from ..mom.mvt import universe as mvt_universe
from ..pretom import calendar as cal

_LEVERAGE_PREFIX = "empirical_leverage"


def _fetch_prices(tickers: list[str], period: str, status: list[dict], label: str):
    """Monkeypatchable seam, mirroring mom.compute._fetch_prices exactly --
    every offline test in this suite stubs this rather than making a real
    ~900-ticker network call."""
    px, st = prices.get_history(tickers, period=period)
    status.append({"segment": label, "ok": bool(px), "n": len(px), "requested": len(tickers),
                   "error": st.get("error", "")})
    return px


def _load_mvt(today: date):
    """Monkeypatchable seam around mvt_link.scores() -- keeps tests off the
    committed data/mom/mvt/ artefact, which belongs to a different package."""
    return mvt_link.scores(today=today)


def _scrub(obj):
    """Recursively replace non-finite floats with None -- json.dumps emits a
    bare NaN/Infinity token, which is not valid JSON and silently corrupts
    every downstream reader. Mirrors mom.compute._scrub; the same hazard
    applies here because the inputs are the same pandas-derived floats."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _leverage_exclusions(px: dict) -> dict[str, str]:
    """The empirical leveraged/inverse backstop ONLY.

    mvt_universe.duplicate_and_leverage_gate returns two kinds of verdict:
    `empirical_leverage_vs_<benchmark>(...)` and `near_duplicate_of:<keeper>`.
    ETF MOMENTUM wants the first and rejects the second -- near-duplicate
    clustering exists to stop a pairwise spread dividing by a near-zero spread
    vol, a problem a ranked list simply does not have, and collapsing
    SPY/VOO/IVV would hide the fund the user actually holds.

    The leverage half is kept because it is the only layer that catches a
    geared fund whose NAME doesn't self-describe and whose metadata cache
    hasn't caught up: correlated >= 0.95 with a broad benchmark AND carrying
    >= 2.5x its realized vol.

    Computed from OUR OWN panel, not from the committed mvt artefact's
    `empirical_exclusions` -- verified live, that dict currently holds 102
    entries and every one of them is a `near_duplicate_of:`, so reading it for
    leverage verdicts would be a silent no-op that looks like it works.
    """
    try:
        returns, _ = mvt_panel.build_return_panel(px, min_bars=60)
        if returns.empty:
            return {}
        verdicts = mvt_universe.duplicate_and_leverage_gate(returns, adv=mvt_panel.advdollar(px))
    except Exception:
        return {}
    return {t: r for t, r in verdicts.items() if r.startswith(_LEVERAGE_PREFIX)}


def _build_rows(universe: list[dict], px: dict, lev: dict[str, str],
                adv: dict[str, float]) -> list[dict]:
    """One row per included ETF: scored, priced-but-insufficient-history, or
    unpriced. Every row survives into scores_latest.json (excluded rows carry
    composite=None plus a stated reason) so a fund closure or a data gap stays
    visible rather than quietly shrinking the universe."""
    rows = []
    for u in universe:
        if not u.get("included"):
            continue
        t = u["ticker"]
        row = {k: u.get(k) for k in ("ticker", "name", "category", "asset_class",
                                     "region", "aum_m", "er")}
        row["adv_dollar"] = adv.get(t)
        df = px.get(t)
        if t in lev:
            row["raw"] = None
            row["exclusion_reason"] = lev[t]
        elif df is None or df.empty:
            row["raw"] = None
            row["exclusion_reason"] = "no_price_data"
        else:
            raw = factors.build_all(df)
            row["raw"] = raw
            if raw is None:
                row["exclusion_reason"] = f"insufficient_history(<{factors.MIN_BARS}d)"
        rows.append(row)
    return rows


def universe_benchmark(px: dict, tickers: list[str]) -> pd.Series | None:
    """An equal-weight price index of the scored ETF universe, used as the
    benchmark the decile pick tracker measures excess against.

    NOT SPY, and the difference is not cosmetic. This universe spans equities,
    bonds, commodities and currencies. Measured against SPY, a long Treasury-
    fund pick would show negative excess through every equity bull market and a
    long equity-fund pick positive excess through the same period, whether or
    not the momentum signal had any skill at all -- the information coefficient
    in diagnostics.json would then be measuring asset-class beta and reporting
    it as evidence for the engine. Since that IC is the ONLY thing that can
    promote this feature's evidence tier, getting the benchmark wrong would
    quietly invalidate the promotion mechanism.

    Equal-weight (not cap- or AUM-weight) because the ranking itself is
    equal-weight: every ETF gets one row and one percentile regardless of size.
    """
    closes = [px[t]["close"] for t in tickers
              if t in px and px[t] is not None and not px[t].empty and "close" in px[t]]
    if len(closes) < 20:
        return None
    frame = pd.concat(closes, axis=1).sort_index()
    # Normalize each series to its own first valid observation before averaging,
    # so a $500 share price doesn't dominate a $20 one.
    normed = frame / frame.apply(lambda s: s.loc[s.first_valid_index()] if s.first_valid_index() is not None else pd.NA)
    idx = normed.mean(axis=1, skipna=True).dropna()
    return idx if len(idx) > 60 else None


def _aggregates(scored: list[dict]) -> dict:
    """Breadth by asset class and by normalized Morningstar category -- the ETF
    analogue of mom's sector/industry split. Both levels partition the scored
    set exactly (screen_etfmom asserts this)."""
    def _agg(key: str) -> dict:
        buckets: dict[str, list[float]] = {}
        for r in scored:
            buckets.setdefault(r.get(key) or "Unknown", []).append(r["composite"])
        out = {}
        for k, vals in buckets.items():
            s = pd.Series(vals, dtype=float)
            out[k] = {
                "n": int(len(s)), "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "pct_bullish": round(float((s >= 5).mean()), 4),
                "pct_bearish": round(float((s <= -5).mean()), 4),
                "n_extreme_bullish": int((s >= 15).sum()),
                "n_extreme_bearish": int((s <= -15).sum()),
                "dispersion": round(float(s.std()), 3) if len(s) > 1 else 0.0,
            }
        return out
    return {"asset_classes": _agg("asset_class"), "categories": _agg("category")}


def _spearman_ic(pairs: list[tuple]) -> float | None:
    if len(pairs) < 8:
        return None
    df = pd.DataFrame(pairs, columns=["score", "excess"])
    ic = df["score"].corr(df["excess"], method="spearman")
    return None if pd.isna(ic) else round(float(ic), 4)


def _diagnostics(scored: list[dict], picks_rows: list[dict]) -> dict:
    """The same predictive diagnostics MOMENTUM ships: the factor correlation
    matrix (which CHECKS the weight-redundancy assumption rather than asserting
    it), composite distribution/breadth, and rank IC per horizon once picks
    mature. Mostly empty on day one; fills in as picks.json accrues."""
    corr = engine.correlations(scored)
    comps = [r["composite"] for r in scored]
    dist = {}
    if comps:
        s = pd.Series(comps, dtype=float)
        dist = {"n": int(len(s)), "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
                "pct_bullish": round(float((s >= 5).mean()), 4),
                "pct_bearish": round(float((s <= -5).mean()), 4),
                "n_extreme_bullish": int((s >= 15).sum()),
                "n_extreme_bearish": int((s <= -15).sum())}
    ic_by_horizon, factor_ic_by_horizon = {}, {}
    for h in etf_history.PICK_HORIZONS_TD:
        hk = str(h)
        pairs = [(r["composite"], r["eval"][hk]["excess"]) for r in picks_rows
                 if r["eval"][hk]["evaluated"] and r["eval"][hk]["excess"] is not None
                 and r.get("composite") is not None]
        ic_by_horizon[hk] = _spearman_ic(pairs)
        f_ic = {}
        for f in FACTORS:
            fpairs = [((r.get("factor_scores") or {}).get(f), r["eval"][hk]["excess"])
                      for r in picks_rows
                      if r["eval"][hk]["evaluated"] and r["eval"][hk]["excess"] is not None
                      and r.get("factor_scores") and r["factor_scores"].get(f) is not None]
            f_ic[f] = _spearman_ic(fpairs)
        factor_ic_by_horizon[hk] = f_ic
    return {"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
            "benchmark": "equal-weight ETF universe index (not SPY) -- see "
                         "compute.universe_benchmark for why",
            "correlation": corr, "distribution": dist,
            "ic_by_horizon": ic_by_horizon, "factor_ic_by_horizon": factor_ic_by_horizon,
            "hit_rate": etf_history.summarize(picks_rows)}


def _write_artifacts(rows: list[dict], today: date, coverage: dict, mvt_status: dict) -> dict:
    scored = [r for r in rows if r.get("composite") is not None]
    assemble(scored, "composite", higher_is_long=True, decile=0.1)   # sets rank/pctile/decile/side

    scores_out, detail_out = [], {}
    for r in rows:
        raw = r.get("raw") or {}
        fs = r.get("factor_scores") or {}
        scores_out.append({
            "ticker": r["ticker"], "name": r["name"], "category": r.get("category"),
            "asset_class": r.get("asset_class"), "region": r.get("region"),
            "aum_m": r.get("aum_m"), "er": r.get("er"), "adv_dollar": r.get("adv_dollar"),
            "composite": r.get("composite"),
            "composite_equal_weight": r.get("composite_equal_weight"),
            "state": r.get("state"), "rank": r.get("rank"), "pctile": r.get("pctile"),
            "decile": r.get("decile"), "side": r.get("side"),
            "factor_scores": fs, "contributions": r.get("contributions"),
            "ts_grid": r.get("ts_grid"), "breakout_grid": r.get("breakout_grid"),
            "bars": raw.get("bars"),
            # --- honesty fields. How many of the six factors actually went
            # into this row's composite, and -- when mvt was inherited across
            # one of mvt's own near-duplicate clusters -- which ticker it was
            # really measured on. Without these the table silently mixes two
            # different composites and one measured-elsewhere factor.
            "n_factors": len(fs) if fs else None,
            "mvt_source": r.get("mvt_source"),
            "excluded": r.get("composite") is None,
            "exclusion_reason": r.get("exclusion_reason"),
        })
        # detail carries ONLY what the GMMA chart needs as its offline
        # fallback. mom stores the whole speed_raw/strength_raw dicts, but the
        # view reads just ma_values out of them and every other field it needs
        # is already in scores_latest.json -- keeping parity here would cost
        # ~1.3MB of git history a day for data nothing reads.
        if raw:
            speed = raw.get("speed_raw") or {}
            detail_out[r["ticker"]] = {"ma_values": speed.get("ma_values"),
                                       "asof": raw.get("asof"),
                                       "last_close": raw.get("last_close")}

    save("scores", _scrub({"as_of": today.isoformat(), "disclaimer": DISCLAIMER,
                           "n": len(scores_out), "n_scored": len(scored),
                           "coverage": coverage, "mvt": mvt_status,
                           "rows": scores_out}), indent=None)
    save("detail", _scrub({"as_of": today.isoformat(), "etfs": detail_out}), indent=None)
    save("categories", _scrub({"as_of": today.isoformat(), **_aggregates(scored)}))

    added_h = etf_history.append_history(scored, today)
    added_p = etf_history.append_picks(etf_history.make_pick_rows(scored, today.isoformat()))
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
        print(f"[etfmom] {today} non-trading day -- no-op")
        return {"ok": True, "gated": True}

    universe, uni_status = etf_universe.constituents()
    included = [u for u in universe if u.get("included")]
    tickers = [u["ticker"] for u in included]
    status.append({"segment": "universe", "ok": bool(tickers), **uni_status})

    px = _fetch_prices(sorted(set(tickers)), ETFMOM_PRICE_PERIOD, status, "prices")
    n_priced = sum(1 for t in tickers if t in px)
    coverage = round(n_priced / len(tickers), 4) if tickers else 0.0
    status.append({"segment": "coverage", "ok": coverage >= 0.85, "n": n_priced,
                   "requested": len(tickers), "coverage": coverage})

    lev = _leverage_exclusions(px)
    status.append({"segment": "leverage_backstop", "ok": True, "n_excluded": len(lev),
                   "note": "empirical vol/correlation gate only; near-duplicates deliberately kept"})

    adv = mvt_panel.advdollar(px)
    rows = _build_rows(included, px, lev, adv)
    n_raw = sum(1 for r in rows if r.get("raw") is not None)
    status.append({"segment": "factors", "ok": n_raw > 0, "n": n_raw})

    engine.cross_sectional(rows)

    # --- The 6th factor, READ rather than recomputed (see mvt_link.py).
    # Fail-soft: a missing or too-stale artefact leaves mvt_score unset on
    # every row, and engine.composite() renormalizes the other five factors'
    # weights per row rather than imputing a neutral 0 (which would drag the
    # composite toward the middle and read as more confident than it is).
    mvt_scores, mvt_status = _load_mvt(today)
    for r in rows:
        hit = mvt_scores.get(r["ticker"])
        if hit:
            r["mvt_score"] = hit["score"]
            if hit["source"] != r["ticker"]:
                r["mvt_source"] = hit["source"]
    status.append({"segment": "mvt", "ok": bool(mvt_scores), **mvt_status})

    # Weights are MOMENTUM's, imported not copied -- the two tabs must stay
    # directly comparable (see config.py's ETFMOM block).
    engine.composite(rows, weights=MOM_WEIGHTS)

    result = _write_artifacts(rows, today, {"n_universe": len(tickers), "n_priced": n_priced,
                                            "coverage": coverage}, mvt_status)
    status.append({"segment": "artifacts", "ok": True, "n": result["scored"]})

    bench = universe_benchmark(px, [r["ticker"] for r in rows if r.get("composite") is not None])
    n_eval = etf_history.evaluate_pending(px, bench, today) if bench is not None else 0
    status.append({"segment": "history", "ok": True, "benchmark": "equal_weight_universe",
                   "benchmark_available": bench is not None,
                   "history_added": result["history_added"],
                   "picks_added": result["picks_added"], "evaluated": n_eval})

    diag = _diagnostics([r for r in rows if r.get("composite") is not None], result["picks_rows"])
    save("diagnostics", _scrub(diag))
    status.append({"segment": "diagnostics", "ok": True,
                   "flagged_pairs": len(diag["correlation"].get("flagged_pairs", []))})

    save("status", {"date": today.isoformat(), "is_trading_day": True, "disclaimer": DISCLAIMER,
                    "mvt_as_of": mvt_status.get("mvt_as_of"),
                    "mvt_stale_days": mvt_status.get("mvt_stale_days"),
                    "segments": status})
    n_inherit = sum(1 for r in rows if r.get("mvt_source"))
    n_five = sum(1 for r in rows if r.get("factor_scores")
                 and len(r["factor_scores"]) < len(FACTORS))
    print(f"[etfmom] {today} universe={len(tickers)} priced={n_priced} scored={result['scored']} "
          f"mvt_direct={mvt_status.get('direct')} mvt_inherited={n_inherit} five_factor={n_five} "
          f"hist+{result['history_added']} picks+{result['picks_added']} eval={n_eval}")
    return {"ok": True, "scored": result["scored"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_auto(force=args.force)


if __name__ == "__main__":
    main()

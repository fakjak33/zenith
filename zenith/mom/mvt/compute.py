"""Multivariate Trend compute orchestrator -- called from mom/compute.py's
run_auto(), once per universe (equities, ETFs).

Equities reuse the R1000+SPY price dict mom.compute.run_auto() already pulled
(period="5y") -- no second network cost. ETFs are a new pull here, chunked
by cas.sources.prices.get_history exactly like every other Zenith ETF pull.

Fail-soft by design (mirrors mom.compute's own posture): any stage failing
returns rows with mvt scores absent rather than raising, so a bad
Multivariate Trend night degrades gracefully -- the sixth factor drops out
of the coverage-renormalized Momentum composite (engine._weighted already
handles a missing input this way) rather than taking MOMENTUM down.
"""

from __future__ import annotations

from datetime import date

import math

from ...config import MOM_MVT_COV_WINDOW, MOM_MVT_MIN_BARS, MOM_MVT_VALIDATION_MONTHS
from ...cas.sources import prices
from . import DISCLAIMER, save
from . import panel as mvt_panel
from . import score as mvt_score
from . import universe as mvt_universe
from . import validate as mvt_validate
from . import history as mvt_history


def _scrub(obj):
    """Recursively replace non-finite floats with None before writing JSON
    -- json.dumps happily emits a bare NaN/Infinity token, which is not
    valid JSON and silently corrupts every downstream reader (this is the
    exact mechanism behind a real bug caught in production: one stock's
    interior data gap left a NaN return that serialized as literal `NaN`,
    reloaded as float('nan'), and poisoned every OTHER ticker's raw_score
    once summed across the pairwise matrix). Mirrors mom.compute._scrub
    exactly -- every Zenith feature that serializes pandas-derived floats
    does this; mvt's outputs are no exception, and this is the safety net
    UNDERNEATH the score.py fix that addresses the actual root cause."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def run_universe(universe_name: str, px: dict, min_bars: int = MOM_MVT_MIN_BARS,
                 cov_window: int = MOM_MVT_COV_WINDOW, adv: dict | None = None) -> dict:
    """Full pipeline for one universe's raw OHLCV dict: build the return
    panel, apply the empirical duplicate/leverage backstop, fit the factor
    model, score every name, and shape the result for `save()`. Returns a
    doc with `rows` keyed by ticker plus run diagnostics; never raises --
    any stage that can't produce a usable result returns an (honestly)
    empty `rows` list rather than propagating an exception into MOMENTUM's
    nightly run."""
    status: dict = {"universe": universe_name, "n_priced": len(px)}
    try:
        log_returns, panel_status = mvt_panel.build_return_panel(px, min_bars=min_bars)
        status["panel"] = panel_status
        if log_returns.empty or log_returns.shape[1] < 10:
            status["error"] = "insufficient_panel"
            return {"as_of": date.today().isoformat(), "rows": [], "status": status}

        adv = adv or mvt_panel.advdollar(px)
        empirical_excluded = mvt_universe.duplicate_and_leverage_gate(log_returns, adv=adv)
        status["empirical_excluded"] = len(empirical_excluded)
        keep_cols = [c for c in log_returns.columns if c not in empirical_excluded]
        log_returns = log_returns[keep_cols]
        if log_returns.shape[1] < 10:
            status["error"] = "insufficient_panel_after_empirical_gate"
            return {"as_of": date.today().isoformat(), "rows": [], "status": status,
                    "empirical_exclusions": empirical_excluded}

        result = mvt_score.compute_universe_scores(log_returns, cov_window=cov_window)
        if result is None:
            status["error"] = "fit_failed"
            return {"as_of": date.today().isoformat(), "rows": [], "status": status,
                    "empirical_exclusions": empirical_excluded}

        status.update({
            "n_scored": len(result["rows"]),
            "k_factors": result["k_factors"],
            "explained_variance_ratio": result["explained_variance_ratio"],
            "effective_factor_count": result["effective_factor_count"],
            "erc_horizon_weights": result["erc_horizon_weights"],
        })
        return {
            "as_of": date.today().isoformat(),
            "disclaimer": DISCLAIMER,
            "rows": result["rows"],
            "k_factors": result["k_factors"],
            "explained_variance_ratio": result["explained_variance_ratio"],
            "effective_factor_count": result["effective_factor_count"],
            "erc_horizon_weights": result["erc_horizon_weights"],
            "status": status,
            "empirical_exclusions": empirical_excluded,
        }
    except Exception as e:  # fail-soft: a bad mvt night must not take MOMENTUM down
        status["error"] = f"exception:{type(e).__name__}:{str(e)[:200]}"
        return {"as_of": date.today().isoformat(), "rows": [], "status": status}


def run_auto(px_equities: dict, meta_status_out: list | None = None) -> dict:
    """Called from mom.compute.run_auto(). `px_equities` is the SAME price
    dict mom.compute already pulled for the R1000 (+SPY) -- reused verbatim,
    no second equities pull. ETFs are pulled here (new network cost, chunked
    by prices.get_history exactly like every other Zenith ETF pull)."""
    segs: list[dict] = meta_status_out if meta_status_out is not None else []

    today = date.today()
    eq_px = {t: df for t, df in px_equities.items() if t != "SPY"}
    eq_result = run_universe("equities", eq_px)
    save("equities", _scrub(eq_result), indent=None)
    added_h = mvt_history.append_history(eq_result["rows"], "equities", today)
    segs.append({"segment": "mvt_equities", "ok": bool(eq_result["rows"]),
                "n": len(eq_result["rows"]), "history_added": added_h, **eq_result.get("status", {})})

    etf_rows, etf_universe_status = mvt_universe.etf_universe(refresh_meta=True)
    etf_tickers = sorted({r["ticker"] for r in etf_rows if r["included"]} | {"SPY", "QQQ", "IWM", "TLT", "GLD"})
    etf_px, price_status = prices.get_history(etf_tickers, period="3y")
    segs.append({"segment": "mvt_etf_universe", **etf_universe_status})
    segs.append({"segment": "mvt_etf_prices", "ok": bool(etf_px), "n": len(etf_px),
                "requested": len(etf_tickers), **{k: v for k, v in price_status.items() if k != "ok"}})

    etf_result = run_universe("etfs", etf_px)
    save("etfs", _scrub(etf_result), indent=None)
    added_h_etf = mvt_history.append_history(etf_result["rows"], "etfs", today)
    segs.append({"segment": "mvt_etfs", "ok": bool(etf_result["rows"]),
                "n": len(etf_result["rows"]), "history_added": added_h_etf, **etf_result.get("status", {})})

    save("status", _scrub({"date": date.today().isoformat(), "disclaimer": DISCLAIMER, "segments": segs}))
    return {"equities": eq_result, "etfs": etf_result}


def run_validation(px_equities: dict, n_months: int = MOM_MVT_VALIDATION_MONTHS) -> dict:
    """Phase 2 validation: Models A/B/C/D backtested with monthly
    rebalancing over the cached equity price history (see validate.py's own
    module docstring for the full honest scope-limitation writeup). This is
    NOT part of the nightly `--action auto` run -- it takes tens of minutes
    (each rebalance date re-fits the whole factor stack for ~1000 names)
    and answers a standing research question, not something that needs a
    fresh answer every night. Run it manually:
      python -m zenith.mom.mvt.compute --action validate [--months N]
    `px_equities` should include SPY (unlike run_auto's own `eq_px`, which
    strips it for the universe result) -- validate.summarize() needs it for
    the market-beta calculation.
    """
    backtest = mvt_validate.run_backtest(px_equities, n_months=n_months)
    report = mvt_validate.summarize(backtest, px_equities)
    save("validation", _scrub(report))
    print(f"[mvt.validate] {report.get('as_of')} n_periods={report.get('n_periods')} "
          f"central_test={report.get('central_hypothesis_test')}")
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto", choices=["auto", "validate"])
    ap.add_argument("--months", type=int, default=MOM_MVT_VALIDATION_MONTHS)
    args = ap.parse_args()

    if args.action == "validate":
        from ...cas.sources import prices as _prices
        from .. import universe as mom_universe
        universe, _ = mom_universe.constituents()
        tickers = sorted(set(u["ticker"] for u in universe) | {"SPY"})
        px, _ = _prices.get_history(tickers, period="5y")
        run_validation(px, n_months=args.months)
    else:
        # Standalone `--action auto` (outside mom.compute's own run_auto)
        # for ad-hoc runs -- pulls the R1000+SPY panel itself rather than
        # reusing one already pulled by a caller.
        from ...cas.sources import prices as _prices
        from .. import universe as mom_universe
        universe, _ = mom_universe.constituents()
        tickers = sorted(set(u["ticker"] for u in universe) | {"SPY"})
        px, _ = _prices.get_history(tickers, period="5y")
        run_auto(px)


if __name__ == "__main__":
    main()

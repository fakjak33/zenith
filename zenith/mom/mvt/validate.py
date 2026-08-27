"""Phase 2 — validation dashboard: Model A (traditional time-series trend) /
B (cross-sectional momentum) / C (multivariate trend, residual pairwise) /
D (combined, the full declared-weight Momentum composite) backtested with
MONTHLY rebalancing over whatever equity price history is already cached.

THE CENTRAL EMPIRICAL QUESTION (per the Quantica-inspired hypothesis this
whole feature exists to test, treated as a hypothesis, not a fact): does
Model C actually lower cross-instrument P&L correlation relative to Model A,
and does that translate into better portfolio-level risk-adjusted
performance? Nothing here is hard-coded toward a particular answer -- if C
does NOT reduce correlation, or reducing correlation does NOT help
portfolio-level Sharpe, this module reports that plainly.

HONEST SCOPE LIMITATIONS (read before trusting any number this produces):

  * History depth -- TWO separate limits, not one: the cached price pull
    itself covers ~5 years (config.MOM_MVT_VALIDATION_MONTHS=58 as built),
    but the factor computations built on top of it need their own ~1.8-2
    years of TRAILING history before they can score anything at all
    (mom.factors.MIN_BARS=460 trading days; mvt's own MOM_MVT_MIN_BARS +
    MOM_MVT_COV_WINDOW=504). A rebalance date early in the cached window
    simply doesn't have enough history BEHIND it yet to produce a decile
    portfolio. VERIFIED on a live run: a nominal 58-month backtest actually
    produced only 38 usable months (the earliest ~20 are silently excluded
    via "insufficient data" in performance_stats, not faked as zero
    performance) -- meaning the genuinely-covered range is roughly the most
    recent ~3 years, not 5. This is exactly why the "2022_drawdown" stress
    window (config.MOM_MVT_STRESS_WINDOWS) shows unavailable despite SPY
    price data technically reaching back to 2021-08: the window's dates
    fall entirely inside that unusable early stretch. Combined, none of
    this reaches 2008 or 2020. Reaching those would need a much deeper
    (15-20 year), much slower price pull for ~1000 names, AND would carry
    dramatically worse survivorship bias than the shorter window already
    has -- today's Russell 1000 constituent list projected back to 2008
    excludes every name that has since been removed, acquired, or
    delisted, which is exactly the bias this repo's own MOM_MEMBERSHIP_
    START note already flags for the live composite's own history. Rather
    than pretend to cover the Global Financial Crisis with data that can't
    honestly support that claim, this module simply doesn't try.
  * Stress windows (config.MOM_MVT_STRESS_WINDOWS) were identified
    EMPIRICALLY by finding the actual largest drawdown in the cached SPY
    series, not assumed from real-world calendar dates -- see the
    `identify_stress_windows` note below for how to regenerate them if the
    data changes.
  * Survivorship bias applies to the ENTIRE backtest window in the same way
    it applies to the live composite's own history: the universe used at
    every historical rebalance date is TODAY's Russell 1000 list (point-in-
    time membership only reaches back to config.MOM_MEMBERSHIP_START,
    2026-07-15), not the constituent list as it actually stood on that
    historical date.
  * Monthly rebalancing, not daily: a deliberate choice for computational
    tractability (each rebalance date re-fits the whole 6-factor stack for
    ~1000 names, ~30-60s of CPU) and because momentum signals of this
    horizon (1-12 months) are not meaningfully traded intraday anyway.
  * Cross-instrument P&L is a PROXY, not a live simulated portfolio: each
    instrument's per-period "P&L" is decile-position-sign x realized
    forward return (see `_pnl_matrix`), which captures whether the SIGNAL
    would have profited from that instrument, not a real trading P&L with
    costs, sizing, or execution. This is the right level of fidelity for
    the specific empirical question here (does the SIGNAL structure reduce
    correlation), and the wrong level for anything beyond that.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ...config import (
    MOM_MVT_COV_WINDOW, MOM_MVT_MIN_BARS, MOM_MVT_VALIDATION_MONTHS,
    MOM_MVT_VALIDATION_DECILE, MOM_MVT_VALIDATION_MODELS, MOM_MVT_STRESS_WINDOWS,
    MOM_WEIGHTS,
)
from .. import factors as mom_factors
from .. import engine as mom_engine
from . import panel as mvt_panel
from . import score as mvt_score

TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = 12  # monthly rebalancing

# "A_timeseries" -> "A" (the key _model_scores_asof's per-ticker dict uses).
MODEL_KEY = {m: m[0] for m in MOM_MVT_VALIDATION_MODELS}


def rebalance_dates(price_index: pd.DatetimeIndex, n_months: int) -> list:
    """Month-end trading dates present in `price_index`, the most recent
    `n_months` + 1 of them (the extra one supplies the LAST period's
    forward-return evaluation window)."""
    idx = pd.DatetimeIndex(sorted(set(price_index)))
    s = pd.Series(idx, index=idx)
    month_ends = s.groupby(idx.to_period("M")).max().sort_values()
    dates = list(month_ends)
    return dates[-(n_months + 1):] if len(dates) > n_months else dates


def _model_scores_asof(px_asof: dict[str, pd.DataFrame]) -> tuple[dict, float | None]:
    """Point-in-time model scores for every priced ticker, using ONLY the
    price history already sliced into `px_asof` (the caller is responsible
    for the point-in-time cut -- this function never looks past what it's
    handed). Mirrors mom.compute.run_backfill's own point-in-time pattern
    (factors.build_all -> engine.cross_sectional -> engine.composite) plus
    an mvt pass, but keeps everything in memory for research rather than
    writing to the live history/picks artifacts.

    Returns ({ticker: {"A":.., "B":.., "C":.., "D":..}}, effective_factor_count).
    """
    rows = []
    for t, df in px_asof.items():
        if df is None or df.empty:
            continue
        raw = mom_factors.build_all(df)
        rows.append({"ticker": t, "raw": raw})

    mom_engine.cross_sectional(rows)

    log_returns, _ = mvt_panel.build_return_panel(px_asof, min_bars=MOM_MVT_MIN_BARS)
    mvt_by_ticker: dict[str, dict] = {}
    eff_factors = None
    if not log_returns.empty and log_returns.shape[1] >= 10:
        result = mvt_score.compute_universe_scores(log_returns, cov_window=MOM_MVT_COV_WINDOW)
        if result is not None:
            mvt_by_ticker = {r["ticker"]: r for r in result["rows"]}
            eff_factors = result["effective_factor_count"]

    out: dict[str, dict] = {}
    for r in rows:
        if r.get("raw") is None:
            continue
        t = r["ticker"]
        mv = mvt_by_ticker.get(t)
        mvt_norm = mv.get("normalized_score") if mv else None

        a_score = 20.0 * (r.get("ts_score") or 0.0)
        b_score = 20.0 * (r.get("xsec_score") or 0.0)
        c_score = mvt_norm  # already on -20..+20

        row_d = {"raw": r["raw"], "ts_score": r.get("ts_score"), "xsec_score": r.get("xsec_score")}
        if mvt_norm is not None:
            row_d["mvt_score"] = mvt_norm
        mom_engine.composite([row_d], weights=MOM_WEIGHTS)
        d_score = row_d.get("composite")

        out[t] = {"A": a_score, "B": b_score, "C": c_score, "D": d_score}
    return out, eff_factors


def _forward_return(px_full: dict[str, pd.DataFrame], ticker: str, start, end) -> float | None:
    df = px_full.get(ticker)
    if df is None or df.empty:
        return None
    close = df["close"]
    try:
        s = close.loc[:start].iloc[-1]
        e = close.loc[:end].iloc[-1]
    except (IndexError, KeyError):
        return None
    if not (np.isfinite(s) and np.isfinite(e)) or s == 0:
        return None
    return float(e / s - 1.0)


def _decile_buckets(scores: dict[str, float | None], decile: float) -> tuple[set, set]:
    valid = {t: v for t, v in scores.items() if v is not None and np.isfinite(v)}
    if len(valid) < 20:
        return set(), set()
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    n = len(ordered)
    k = max(1, int(round(n * decile)))
    shorts = {t for t, _ in ordered[:k]}
    longs = {t for t, _ in ordered[-k:]}
    return longs, shorts


def run_backtest(px_full: dict[str, pd.DataFrame], n_months: int = MOM_MVT_VALIDATION_MONTHS,
                 decile: float = MOM_MVT_VALIDATION_DECILE, progress_every: int = 6) -> dict:
    """Walk the cached price history forward, monthly, computing all four
    models' scores + decile positions + realized forward returns at every
    rebalance date. Returns the raw per-period data (`periods`) that every
    other function in this module derives its statistics from -- computed
    ONCE, reused for every cut (overall / by regime / by stress window) so
    the expensive part (re-fitting six factors x ~1000 names x N months)
    only ever happens once per backtest run.
    """
    all_idx = sorted(set().union(*[set(df.index) for df in px_full.values() if df is not None]))
    dates = rebalance_dates(pd.DatetimeIndex(all_idx), n_months)
    if len(dates) < 3:
        return {"periods": [], "n_periods": 0, "error": "insufficient price history"}

    periods = []
    for i, asof in enumerate(dates[:-1]):
        px_asof = {t: df.loc[:asof] for t, df in px_full.items() if df is not None}
        px_asof = {t: df for t, df in px_asof.items() if not df.empty and len(df) >= MOM_MVT_MIN_BARS}
        scores, eff_factors = _model_scores_asof(px_asof)

        next_date = dates[i + 1]
        fwd = {t: _forward_return(px_full, t, asof, next_date) for t in scores}

        period = {"asof": asof.isoformat(), "next": next_date.isoformat(),
                 "effective_factor_count": eff_factors, "n_scored": len(scores)}
        for m in MOM_MVT_VALIDATION_MODELS:
            key = MODEL_KEY[m]  # "A_timeseries" -> "A", etc.
            m_scores = {t: v.get(key) for t, v in scores.items()}
            longs, shorts = _decile_buckets(m_scores, decile)
            period[m] = {
                "longs": sorted(longs), "shorts": sorted(shorts),
                "fwd_return_by_ticker": {t: fwd[t] for t in (longs | shorts) if fwd.get(t) is not None},
            }
        periods.append(period)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[mvt.validate] {i + 1}/{len(dates) - 1} periods done ({asof.date()})")

    return {"periods": periods, "n_periods": len(periods),
            "as_of": date.today().isoformat(), "n_months": n_months, "decile": decile}


# ------------------------------------------------------------- portfolios --
def portfolio_returns(periods: list[dict], model: str) -> list[float | None]:
    """Monthly long-short portfolio return per period: mean(long forward
    returns) - mean(short forward returns). None for a period where either
    bucket has no evaluable forward return (never silently treated as 0)."""
    out = []
    for p in periods:
        m = p[model]
        fwd = m["fwd_return_by_ticker"]
        long_rets = [fwd[t] for t in m["longs"] if t in fwd]
        short_rets = [fwd[t] for t in m["shorts"] if t in fwd]
        if not long_rets or not short_rets:
            out.append(None)
            continue
        out.append(float(np.mean(long_rets) - np.mean(short_rets)))
    return out


def performance_stats(returns: list[float | None]) -> dict:
    """CAGR / Sharpe / Sortino / max drawdown / vol / hit rate from a
    monthly long-short return series. All annualized off PERIODS_PER_YEAR=12.
    Returns an honest "insufficient data" marker rather than a
    divide-by-zero or a fabricated number when there's too little history."""
    r = np.array([x for x in returns if x is not None], dtype=float)
    if len(r) < 6:
        return {"n_periods": len(r), "note": "insufficient data (<6 periods)"}
    wealth = np.cumprod(1.0 + r)
    n_years = len(r) / PERIODS_PER_YEAR
    cagr = float(wealth[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 and wealth[-1] > 0 else None
    vol = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    # Epsilon rather than a bare > 0: a near-constant return series can have
    # a std that's floating-point noise (~1e-18) rather than exactly zero,
    # which would otherwise produce a nonsensical Sharpe in the 1e16 range
    # instead of correctly reporting "undefined" -- caught by a test using
    # a literal constant return series.
    r_std = r.std(ddof=1)
    sharpe = float(r.mean() / r_std * np.sqrt(PERIODS_PER_YEAR)) if r_std > 1e-9 else None
    downside = r[r < 0]
    downside_dev = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0
    sortino = float(r.mean() / downside_dev * np.sqrt(PERIODS_PER_YEAR)) if downside_dev > 1e-9 else None
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    max_dd = float(drawdown.min())
    hit_rate = float((r > 0).mean())
    return {"n_periods": len(r), "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "volatility": vol, "max_drawdown": max_dd, "hit_rate": hit_rate,
            "mean_monthly_return": float(r.mean())}


def turnover_stats(periods: list[dict], model: str) -> dict:
    """Month-over-month fraction of the long bucket that turns over, and the
    implied average holding period (1/turnover months) -- a low-turnover
    signal trades less and is more capacity-friendly, all else equal."""
    churns = []
    prev = None
    for p in periods:
        longs = set(p[model]["longs"])
        if prev and len(prev):
            churns.append(len(longs - prev) / len(prev))
        prev = longs
    if not churns:
        return {"avg_monthly_turnover": None, "avg_holding_period_months": None}
    avg_churn = float(np.mean(churns))
    holding = round(1.0 / avg_churn, 2) if avg_churn > 0 else None
    return {"avg_monthly_turnover": round(avg_churn, 4), "avg_holding_period_months": holding}


def market_beta(returns: list[float | None], spy_returns: list[float | None]) -> float | None:
    """OLS beta of the long-short portfolio's return series against SPY's
    same-period return -- section 18's "market beta" / "factor exposure"
    ask, computed directly rather than via a new dependency."""
    pairs = [(a, b) for a, b in zip(returns, spy_returns) if a is not None and b is not None]
    if len(pairs) < 6:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if b.std() < 1e-9:
        return None
    return float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))


# ------------------------------------------------------- P&L correlation --
def pnl_matrix(periods: list[dict], model: str) -> pd.DataFrame:
    """Instrument x period matrix of decile-position-sign x realized
    forward return -- see the module docstring's "Cross-instrument P&L is a
    PROXY" note for exactly what this does and doesn't represent. Zero for
    any (ticker, period) where the ticker wasn't in that model's long/short
    bucket that period (a name not held contributes zero P&L that period,
    the standard portfolio-attribution convention)."""
    tickers = sorted(set().union(*[set(p[model]["longs"]) | set(p[model]["shorts"]) for p in periods])) \
        if periods else []
    cols = [p["asof"] for p in periods]
    mat = pd.DataFrame(0.0, index=tickers, columns=cols)
    for p in periods:
        m = p[model]
        fwd = m["fwd_return_by_ticker"]
        for t in m["longs"]:
            if t in fwd:
                mat.loc[t, p["asof"]] = fwd[t]
        for t in m["shorts"]:
            if t in fwd:
                mat.loc[t, p["asof"]] = -fwd[t]
    return mat


def pairwise_correlation_stats(mat: pd.DataFrame, min_periods_held: int = 6) -> dict:
    """Average / median / 90th-percentile pairwise Pearson correlation
    across every instrument-pair's P&L series, restricted to instruments
    held (nonzero) in at least `min_periods_held` periods so a name that
    only ever appeared once doesn't contribute a degenerate correlation.
    THIS IS THE CENTRAL EMPIRICAL TEST: compare this across models A vs C."""
    if mat.empty:
        return {"n_instruments": 0, "n_pairs": 0, "avg_pairwise_corr": None,
                "median_pairwise_corr": None, "tail_pairwise_corr_p90": None}
    held = (mat != 0).sum(axis=1)
    keep = held[held >= min_periods_held].index
    sub = mat.loc[keep]
    if len(sub) < 3:
        return {"n_instruments": int(len(sub)), "n_pairs": 0, "avg_pairwise_corr": None,
                "median_pairwise_corr": None, "tail_pairwise_corr_p90": None}
    corr = sub.T.corr()
    iu = np.triu_indices_from(corr.to_numpy(), k=1)
    vals = corr.to_numpy()[iu]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n_instruments": int(len(sub)), "n_pairs": 0, "avg_pairwise_corr": None,
                "median_pairwise_corr": None, "tail_pairwise_corr_p90": None}
    return {
        "n_instruments": int(len(sub)), "n_pairs": int(len(vals)),
        "avg_pairwise_corr": round(float(np.mean(vals)), 4),
        "median_pairwise_corr": round(float(np.median(vals)), 4),
        "tail_pairwise_corr_p90": round(float(np.percentile(vals, 90)), 4),
    }


# --------------------------------------------------------- regime / stress --
def regime_buckets(periods: list[dict]) -> dict[str, list[dict]]:
    """Terciles of `effective_factor_count` (the same PCA-based diagnostic
    factors.effective_factor_count already computes at every rebalance
    date) -- LOW effective-factor-count = a macro-dominated, HIGH cross-
    instrument correlation regime; HIGH = a more idiosyncratic, LOW-
    correlation regime. Section 19's "does mvt behave differently under
    different correlation regimes" ask, using a diagnostic this feature
    already produces for free rather than a new regime classifier."""
    vals = [p["effective_factor_count"] for p in periods if p.get("effective_factor_count") is not None]
    if len(vals) < 9:
        return {}
    q1, q2 = np.percentile(vals, [33.3, 66.7])
    buckets: dict[str, list[dict]] = {"low_effective_factors_high_corr_regime": [],
                                      "mid_regime": [], "high_effective_factors_low_corr_regime": []}
    for p in periods:
        v = p.get("effective_factor_count")
        if v is None:
            continue
        if v <= q1:
            buckets["low_effective_factors_high_corr_regime"].append(p)
        elif v <= q2:
            buckets["mid_regime"].append(p)
        else:
            buckets["high_effective_factors_low_corr_regime"].append(p)
    return buckets


def stress_window_periods(periods: list[dict], start: str, end: str) -> list[dict]:
    return [p for p in periods if start <= p["asof"] <= end]


# ------------------------------------------------------------------ top level --
def _model_report(periods: list[dict], model: str, spy_returns: list[float | None]) -> dict:
    """`mean_monthly_return` in the returned stats dict IS the average
    realized long-short spread (mean(long fwd return) - mean(short fwd
    return), averaged across periods) -- performance_stats() already
    computes it from the same `rets` series, so it is not duplicated here."""
    rets = portfolio_returns(periods, model)
    stats = performance_stats(rets)
    stats["turnover"] = turnover_stats(periods, model)
    stats["market_beta"] = market_beta(rets, spy_returns)
    mat = pnl_matrix(periods, model)
    stats["pnl_correlation"] = pairwise_correlation_stats(mat)
    return stats


def summarize(backtest: dict, px_full: dict[str, pd.DataFrame]) -> dict:
    """The full validation report: per-model performance + P&L correlation,
    overall and conditioned on correlation regime and on the empirically-
    identified stress windows. This is the function mvt/compute.py's
    run_validation() calls and persists as data/mom/mvt/validation.json.

    Answers, as plainly as the data supports, the two questions section 18
    of the spec asks for -- WITHOUT assuming an answer:
      1. Does Model C (multivariate/residual) show lower avg pairwise P&L
         correlation than Model A (traditional time-series trend)?
      2. Does that (if true) show up as better risk-adjusted performance
         at the portfolio level?
    """
    periods = backtest.get("periods", [])
    if not periods:
        return {"as_of": date.today().isoformat(), "error": backtest.get("error", "no periods"),
                "n_periods": 0}

    spy_df = px_full.get("SPY")
    spy_returns = []
    if spy_df is not None and not spy_df.empty:
        close = spy_df["close"]
        for p in periods:
            spy_returns.append(_forward_return(px_full, "SPY", pd.Timestamp(p["asof"]),
                                                pd.Timestamp(p["next"])))
    else:
        spy_returns = [None] * len(periods)

    overall = {m: _model_report(periods, m, spy_returns) for m in MOM_MVT_VALIDATION_MODELS}

    regimes = regime_buckets(periods)
    by_regime = {}
    for regime_name, regime_periods in regimes.items():
        if len(regime_periods) < 6:
            continue
        idx_map = {p["asof"]: i for i, p in enumerate(periods)}
        regime_spy = [spy_returns[idx_map[p["asof"]]] for p in regime_periods]
        by_regime[regime_name] = {m: _model_report(regime_periods, m, regime_spy)
                                  for m in ("A_timeseries", "C_multivariate")}

    by_stress = {}
    idx_map = {p["asof"]: i for i, p in enumerate(periods)}
    for name, (start, end) in MOM_MVT_STRESS_WINDOWS.items():
        wp = stress_window_periods(periods, start, end)
        if len(wp) < 3:
            by_stress[name] = {"available": False, "n_periods": len(wp),
                              "note": "insufficient periods in the cached price history's coverage"}
            continue
        w_spy = [spy_returns[idx_map[p["asof"]]] for p in wp]
        models = {m: _model_report(wp, m, w_spy) for m in ("A_timeseries", "C_multivariate")}
        # A window can fall INSIDE the requested date range (len(wp)>=3 above)
        # while still producing zero usable decile portfolios, if it lands
        # in the early part of the backtest before enough TRAILING history
        # has accumulated for the factor computations themselves (a real
        # thing this module's own docstring warns about -- verified: the
        # default 58-month window only yields 38 usable months, and the
        # 2022 stress window falls entirely inside the unusable stretch).
        # Reported explicitly here rather than leaving the reader to notice
        # n_periods=0 buried inside a model's own stats dict.
        usable = any((models[m].get("n_periods") or 0) > 0 for m in models)
        by_stress[name] = {"available": bool(usable), "n_periods": len(wp), "window": [start, end],
                          "models": models,
                          "note": None if usable else (
                              "dates fall within the backtest's nominal range but before enough "
                              "TRAILING history had accumulated for the factor computations to "
                              "score anything -- see this module's docstring")}

    central_test = None
    a_corr = overall["A_timeseries"]["pnl_correlation"].get("avg_pairwise_corr")
    c_corr = overall["C_multivariate"]["pnl_correlation"].get("avg_pairwise_corr")
    if a_corr is not None and c_corr is not None:
        central_test = {
            "model_a_avg_pairwise_pnl_corr": a_corr,
            "model_c_avg_pairwise_pnl_corr": c_corr,
            "c_lower_than_a": bool(c_corr < a_corr),
            "difference": round(c_corr - a_corr, 4),
            "model_a_sharpe": overall["A_timeseries"].get("sharpe"),
            "model_c_sharpe": overall["C_multivariate"].get("sharpe"),
            "sharpe_improved_alongside_lower_corr":
                (bool(c_corr < a_corr) and
                 (overall["C_multivariate"].get("sharpe") or -999) > (overall["A_timeseries"].get("sharpe") or -999))
                if (overall["C_multivariate"].get("sharpe") is not None
                    and overall["A_timeseries"].get("sharpe") is not None) else None,
        }

    return {
        "as_of": date.today().isoformat(),
        "n_periods": len(periods),
        "period_range": [periods[0]["asof"], periods[-1]["asof"]],
        "models": overall,
        "central_hypothesis_test": central_test,
        "by_regime": by_regime,
        "by_stress_window": by_stress,
        "methodology_note": (
            f"Monthly rebalancing over the cached equity price history only (see this module's "
            f"docstring for exactly what that does and doesn't cover -- 2008 and 2020 are NOT "
            f"reached, and today's Russell 1000 list is projected backward, which is survivorship-"
            f"biased). The factor computations need ~2 years of trailing history before they can "
            f"score anything, so the effective usable window is shorter than the nominal one -- "
            f"this run covers {len(periods)} nominal months but only {overall['A_timeseries'].get('n_periods', 0)} "
            f"were actually usable (see any stress window marked unavailable for why). "
            f"Cross-instrument P&L is decile-position-sign x realized forward return, a "
            f"signal-level proxy, not a costed trading simulation. Nothing here is tuned to produce "
            f"a particular result."
        ),
    }

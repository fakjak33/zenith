"""Streamlit rendering for the EDGE SCREENS tab — four cross-sectional screens
behind a radio selector (CAS pattern). Reads only committed JSON (data/edge/).

Each screen leads with its evidence-strength rating (honest about the Muravyev
borrow-fee and Bali mispricing caveats), then a ranked long table, a short/
avoid table, a distribution of the raw measure, and a live per-screen tracker.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load
from . import history as ehist
from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section, stamp

SEGMENTS = ["IV spread", "Revisions", "Short interest", "Lottery"]

# per-screen presentation config
CFG = {
    "ivspread": {
        "seg": "IV spread", "tier": "C+", "tier_label": "options IV spread",
        "measure": "iv_spread_bp", "measure_label": "IV spread (bp)",
        "long_label": "Expensive calls (bullish tilt)",
        "short_label": "Expensive puts / hard-to-borrow",
        "tier_note": ("Real in-sample (+51bp/wk, Cremers-Weinbaum 2010) but "
                      "Muravyev-Pearson-Pollet (JFE 2025) show ~2/3 is a borrow-"
                      "fee proxy in hard-to-borrow names — 'a mirage not readily "
                      "exploitable'. Read the negative side as a short/borrow "
                      "flag more than a clean long."),
        "findings": [
            {"stat": "Stocks with relatively expensive CALLS beat expensive-PUT "
                     "stocks by ~51 bp/week — options lead the stock.",
             "cite": "Cremers & Weinbaum (2010)"},
            {"stat": "~2/3 of it is really the stock BORROW FEE; excluding high-"
                     "fee names cuts the predictability by two-thirds.",
             "cite": "Muravyev, Pearson & Pollet (2025)"}],
    },
    "revisions": {
        "seg": "Revisions", "tier": "B", "tier_label": "analyst revision drift",
        "measure": "composite", "measure_label": "revision composite (0-100)",
        "long_label": "Rising estimates (long drift)",
        "short_label": "Falling estimates (short drift)",
        "tier_note": ("Prices underreact to analyst EPS revisions and drift for "
                      "weeks (Chen et al. 2020; sell-side reco drift -9.1%/6mo). "
                      "Attenuated ~50% post-publication but alive."),
        "findings": [
            {"stat": "Post-forecast-revision drift: prices keep moving with "
                     "analyst EPS revisions for weeks (underreaction).",
             "cite": "Chen, Narayanamoorthy, Sougiannis & Zhou (2020)"},
            {"stat": "New SELL recommendations drift -9.1% over 6 months; the "
                     "short side is the stronger leg.",
             "cite": "Bradshaw et al."}],
    },
    "shortint": {
        "seg": "Short interest", "tier": "B", "tier_label": "short interest / DTC",
        "measure": "composite", "measure_label": "short-crowding (0-100)",
        "long_label": "Least-shorted (benign)",
        "short_label": "Most-shorted / avoid list",
        "tier_note": ("High short interest & days-to-cover predict weak returns "
                      "gross (Rapach-Ringgenberg-Zhou 2016; Hong et al.), but net "
                      "of borrow fees the edge ~vanishes (Muravyev 2025) — an "
                      "AVOID list with a squeeze-risk flag, not tradable alpha."),
        "findings": [
            {"stat": "Aggregate short interest is among the strongest known "
                     "return predictors (out-of-sample R2 ~13%/yr).",
             "cite": "Rapach, Ringgenberg & Zhou (2016)"},
            {"stat": "Days-to-cover sorts ~1.2%/mo gross — but net of borrow "
                     "fees anomaly returns fall to ~0.",
             "cite": "Hong et al. · Muravyev (2025)"}],
    },
    "lottery": {
        "seg": "Lottery", "tier": "B+", "tier_label": "lottery / MAX-beta",
        "measure": "max_beta_pct", "measure_label": "MAX-beta (%/day)",
        "long_label": "Low lottery demand (benign)",
        "short_label": "High MAX-beta (lottery short)",
        "tier_note": ("Raw MAX is subsumed by mispricing factors, but the beta-"
                      "purged MAXbeta survives modern models: -0.81%/mo (t=-3.62), "
                      "including large caps (Bali-Ince-Ozsoylev 2026). MAXbeta is "
                      "the headline; raw MAX shown alongside."),
        "findings": [
            {"stat": "High max-daily-return 'lottery' stocks underperform by "
                     ">1%/month (investors overpay for the jackpot).",
             "cite": "Bali, Cakici & Whitelaw (2011)"},
            {"stat": "Beta-purged MAXbeta survives mispricing factors: "
                     "-0.81%/mo, t=-3.62, even in large caps.",
             "cite": "Bali, Ince & Ozsoylev (2026)"}],
    },
}
_SCREEN_OF = {c["seg"]: k for k, c in CFG.items()}


def _grad(v, cap, color):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    a = max(0.0, min(1.0, abs(float(v)) / cap))
    r, g, b = (int(11 + (c - 11) * a) for c in color)
    return f"background-color: rgb({r},{g},{b}); color:#fff;"


def _table(rows: list[dict], measure: str, extra: list[str], key: str) -> None:
    if not rows:
        st.caption("— none —")
        return
    base = ["rank", "ticker", "name", "sector", measure, "pctile"]
    cols = [c for c in base + extra if any(c in r for r in rows)]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    cap = max(1e-9, df[measure].abs().max()) if measure in df else 1.0
    try:
        sty = df.style.map(lambda v: _grad(v, cap, (46, 196, 182)), subset=[measure])
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(430, 45 + 34 * len(df)))
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _hist(ranked: list[dict], measure: str, label: str) -> None:
    vals = [r.get(measure) for r in ranked if r.get(measure) is not None]
    if len(vals) < 5:
        return
    try:
        import altair as alt
        df = pd.DataFrame({label: vals})
        ch = (alt.Chart(df).mark_bar(opacity=0.85, color=THEME.teal)
              .encode(x=alt.X(f"{label}:Q", bin=alt.Bin(maxbins=40),
                              axis=alt.Axis(labelLimit=0)),
                      y=alt.Y("count()", title="stocks")))
        st.altair_chart(ch.properties(height=180), use_container_width=True)
    except Exception:
        pass


def _tracker(screen: str) -> None:
    summ = ehist.summarize(load("history", {}).get("rows", []))
    cell = summ.get(screen)
    if not cell:
        st.caption("Tracker fills in as horizons mature (screens start live, no "
                   "backfill).")
        return
    cols = st.columns(3)
    for i, side in enumerate(("long", "short", "all")):
        c = cell.get(side)
        if c:
            cols[i].metric(f"{side} n={c['n']}",
                           f"{c['avg']*100:+.2f}% avg",
                           f"{c['win_rate']:.0%} win", delta_color="off")


def _segment(seg: str) -> None:
    screen = _SCREEN_OF[seg]
    cfg = CFG[screen]
    res = load(screen, {})
    st.markdown(evidence_rating(cfg["tier"], cfg["tier_label"], cfg["tier_note"]),
                unsafe_allow_html=True)
    st.markdown(key_findings(cfg["findings"]), unsafe_allow_html=True)
    if not res.get("ranked"):
        st.info(f"No {seg} data yet — run `python -m zenith.edge.compute` or wait "
                "for the nightly Action.")
        return
    st.markdown(stamp(res.get("as_of", "?"), f"EDGE · {seg}"), unsafe_allow_html=True)
    st.caption(f"{res.get('n', 0)} names ranked · horizon {res.get('horizon_td','?')} "
               "trading days · tracked vs SPY, sign-adjusted (positive = worked).")

    measure = cfg["measure"]
    extra_long, extra_short = _extra_cols(screen)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(section(cfg["long_label"], 2), unsafe_allow_html=True)
        _table(res.get("long", []), measure, extra_long, key=f"{screen}_long")
    with c2:
        st.markdown(section(cfg["short_label"], 4), unsafe_allow_html=True)
        _table(res.get("short", []), measure, extra_short, key=f"{screen}_short")

    if screen == "shortint" and (res.get("aggregate") or {}).get("z_vs_history") is not None:
        agg = res["aggregate"]
        st.caption(f"Aggregate short-crowding gauge (Rapach et al.): median "
                   f"SI%float {agg['median_si_float']:.1f}%, z vs our history "
                   f"{agg['z_vs_history']:+.2f} (n={agg['n_history']}).")

    st.markdown(section(f"Distribution — {cfg['measure_label']}", 1),
                unsafe_allow_html=True)
    _hist(res.get("ranked", []), measure, cfg["measure_label"])

    st.markdown(section("Live tracker — does this screen work on our data?", 0),
                unsafe_allow_html=True)
    _tracker(screen)

    with st.expander(f"How the {seg} screen works — the research"):
        st.markdown(_METHOD[screen])


def _extra_cols(screen: str):
    if screen == "ivspread":
        return (["n_pairs"], ["borrow_flag", "n_pairs"])
    if screen == "revisions":
        return (["r_est", "r_dir", "r_reco"], ["r_est", "r_dir", "r_reco"])
    if screen == "shortint":
        return (["si_float", "dtc"], ["si_float", "dtc", "squeeze_risk"])
    if screen == "lottery":
        return (["max_raw_pct", "beta"], ["max_raw_pct", "beta"])
    return ([], [])


_METHOD = {
    "ivspread": (
        "**Measure**: for each stock, on the nearest standard expiry (>=7 days "
        "out), the open-interest-weighted mean of (ATM call implied-vol - ATM "
        "put implied-vol) across strikes within +/-5% of spot. Positive = "
        "expensive calls, a bullish options tilt (Cremers & Weinbaum 2010, "
        "JFQA — deviations from put-call parity predict the stock ~1 week out, "
        "+51 bp/week).\n\n"
        "**The big caveat** (Muravyev, Pearson & Pollet 2025, JFE): most of the "
        "signal is the stock's BORROW FEE showing up in option prices, not "
        "free directional information — remove high-fee names and ~2/3 of the "
        "predictability disappears, and it's 'not readily exploitable'. So the "
        "hard-to-borrow flag marks negative-spread names that are also heavily "
        "shorted (double-counted risk), and this screen is C+ — real but weak "
        "net of costs."),
    "revisions": (
        "**Composite** = 0.5*rank(current-FY mean EPS estimate change over "
        "~90d) + 0.3*rank(share of last-30d revisions that were UP) + "
        "0.2*rank(net analyst upgrades - downgrades, 30d). Prices underreact to "
        "revisions and drift in their direction for weeks — analyst "
        "underreaction feeds the post-forecast-revision drift (Chen, "
        "Narayanamoorthy, Sougiannis & Zhou 2020, JBFA). New SELL "
        "recommendations drift further (-9.1%/6mo) than buys.\n\n"
        "**Caveat**: like most published anomalies the effect is ~50% smaller "
        "post-publication (McLean-Pontiff). All three inputs are free yfinance "
        "estimate fields, which are sometimes stale/sparse — thin coverage is "
        "flagged by fewer names ranked. Tier B."),
    "shortint": (
        "**Composite** = 0.6*rank(short % of float) + 0.4*rank(days-to-cover). "
        "High short interest and days-to-cover have predicted weak returns "
        "(Rapach, Ringgenberg & Zhou 2016 — aggregate SI is a top market "
        "predictor, ~13% OOS R-squared; Hong et al. — DTC ~1.2%/mo cross-"
        "sectionally).\n\n"
        "**Why it's an avoid-list, not alpha** (Muravyev 2025, JF): net of the "
        "borrow fees you pay to short these names, the anomaly return is about "
        "zero. So the most-shorted decile is framed as an AVOID / short-watch "
        "list. The **squeeze-risk flag** marks the dangerous subset — names "
        "that are heavily shorted AND already rising (crowded shorts that can "
        "rip). The aggregate gauge z-scores median SI%float vs our own history."),
    "lottery": (
        "**MAX-beta** (the headline): estimate each stock's market beta over a "
        "trailing 60-day window, strip the systematic part (residual = return - "
        "beta*market), then average the 5 largest DAILY RESIDUAL returns over "
        "the last ~21 days. High MAX-beta = strongest idiosyncratic 'lottery' "
        "demand = the SHORT screen.\n\n"
        "Raw MAX (Bali, Cakici & Whitelaw 2011) predicted >1%/mo underperformance "
        "but is largely subsumed by modern mispricing factors. Bali, Ince & "
        "Ozsoylev (2026) show the beta-purged MAX-beta SURVIVES: a high-minus-"
        "low spread of -0.81%/mo (t=-3.62), and unlike raw MAX it works even in "
        "the largest, most liquid names. Both are shown; MAX-beta drives the "
        "ranking. Tier B+."),
}


def today_badge() -> str | None:
    """Chip on the TODAY tab flagging the strongest fresh short-side names."""
    lot = load("lottery", {})
    shorts = lot.get("short", [])
    if not shorts or lot.get("as_of", "") < _stale_cut():
        return None
    n = len(shorts)
    color = THEME.coral
    # Shared chip helper -- see the note in pretom/view.py::today_badge.
    return uc.chip(f"EDGE — {n} lottery-short & short-interest names flagged",
                   color=color, sub="see EDGE SCREENS tab")


def _stale_cut() -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=5)).isoformat()


def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(section("EDGE SCREENS — four cross-sectional signals, rated", 3,
                        help="Four ranked long/short screens from the empirical "
                             "cross-section, each with an honest evidence tier. "
                             "Pick a screen below."), unsafe_allow_html=True)
    seg = st.radio("Screen", SEGMENTS, horizontal=True, label_visibility="collapsed")
    st.caption({"IV spread": "Option-implied direction (put-call parity deviations).",
                "Revisions": "Analyst EPS-estimate revision drift.",
                "Short interest": "Short interest & days-to-cover crowding.",
                "Lottery": "Beta-purged MAX lottery-demand short screen."}[seg])
    _segment(seg)

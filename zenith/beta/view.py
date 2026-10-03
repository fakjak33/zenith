"""CLEAN BETA tab — quality high-beta screener + convex hedge monitor.

Reads only committed artefacts under data/beta/. Four views on one radio:
Screener (sort / filter the whole Russell 1000 by Quality-Beta Score, with a
pass/fail column per filter so it is clear WHY a name is out) -> Basket (the
quarterly 30-50 name candidate basket and how many real bets it is) -> Hedge
Monitor (trend gate, vol target, VRP, BTAL, advisory put ladder, alerts) ->
Playbook (what each hedge buys you, with the evidence). Column headers carry
the tab's [P]/[S]/[E] data-confidence tags.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import (BETA_BSWA_DELTA, BETA_BSWA_LAMBDA, BETA_CORR_PENALTY, BETA_CORR_THRESHOLD,
                      BETA_EARNINGS_DAYS, BETA_FILES, BETA_HARVEST_MULTIPLE, BETA_IVOL_MAX_PCT,
                      BETA_JUMP_SIGMA, BETA_MAX_JUMP_DAYS, BETA_MIN_ADV_USD, BETA_MIN_MKTCAP,
                      BETA_MIN_PRICE, BETA_NAME_CAP, BETA_PCT_ENTER, BETA_PCT_EXIT, BETA_PUT_BUDGET,
                      BETA_QUALITY_MIN_PCT, BETA_RHO_MIN, BETA_SCORE_WEIGHTS, BETA_SECTOR_CAP,
                      BETA_TREND_EXPOSURE, BETA_VIX_HARVEST_PCT, BETA_VOL_TARGET, THEME)
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import CONF, DISCLAIMER, load, tag

SUBVIEWS = ["Screener", "Basket", "Hedge Monitor", "Playbook"]

_EVIDENCE_NOTE = ("Betting-against-beta is one of the most replicated anomalies (US, 20 international markets, "
                  "bonds, futures): naive high beta has earned LESS than its beta implies. The clean-beta "
                  "refinement — keep high-beta names whose beta is correlation, not idiosyncratic noise — "
                  "rests on Liu, Stambaugh & Yuan (2018) and is less replicated out of sample. The fundamental "
                  "quality layer here is a free-data proxy. Hedge guidance is evidence-graded per item in the "
                  "Playbook.")

_FINDINGS = [
    {"stat": "High-beta assets earn lower risk-adjusted returns across US and 20 international equity markets.",
     "cite": "Frazzini & Pedersen (2014)"},
    {"stat": "The beta anomaly vanishes once overpriced high-IVOL stocks are excluded.",
     "cite": "Liu, Stambaugh & Yuan (2018)"},
    {"stat": "Slope-winsorized, age-decayed beta forecasts future beta better than OLS, Vasicek or Bloomberg.",
     "cite": "Welch (2022)"},
    {"stat": "Rolling protective puts have underperformed simply holding less equity.",
     "cite": "Israelov (2019)"},
]

COLS = {
    "Rank": "Position by Quality-Beta Score among liquid names.",
    "Ticker": "Exchange ticker.", "Name": "Company name.", "Sector": "Sector (MOMENTUM metadata).",
    "Size": "Market-cap bucket: Mega ≥ $200B, Large ≥ $50B, Mid ≥ $10B, Small below.",
    f"Score {tag('score')}": ("Quality-Beta Score, 0–100: weighted mean of percentiles — "
                              + ", ".join(f"{k} {v:.0%}" for k, v in BETA_SCORE_WEIGHTS.items())
                              + ". IVOL enters inverted (low IVOL scores high)."),
    f"bswa β {tag('bswa')}": "Welch slope-winsorized, age-decayed beta vs SPY (~2 years, half-life ~4 months).",
    f"OLS β {tag('ols')}": "Plain 252-day OLS beta vs SPY, for comparison.",
    f"Vasicek β {tag('vasicek')}": "OLS beta shrunk toward the universe mean (for the PARALLAX grid).",
    f"β pct {tag('bswa')}": "bswa beta percentile within the liquid universe.",
    f"ρ {tag('rho')}": "252-day correlation of daily returns with SPY.",
    f"R² {tag('r2')}": "Systematic share of variance (ρ²).",
    f"IVOL {tag('ivol')}": "Annualized standard deviation of market-model residuals.",
    f"IVOL pct {tag('ivol')}": "IVOL percentile (higher = noisier).",
    f"Events {tag('jumps')}": f"Days in the last 252 with |residual| > {BETA_JUMP_SIGMA:g}σ.",
    f"Quality {tag('quality')}": "Quality/mispricing percentile: ROA, gross margin, FCF margin, accruals, "
                                 "leverage, 12-1 momentum (+ net issuance once 12 months of share data exist). "
                                 "An italic [E] means too few fundamentals — neutral 50.",
    f"Earnings {tag('earnings')}": f"Scheduled report within {BETA_EARNINGS_DAYS} days of the screen date.",
    "β✓": f"bswa β pct ≥ {BETA_PCT_ENTER:g} (incumbents: ≥ {BETA_PCT_EXIT:g}).",
    "ρ✓": f"ρ ≥ {BETA_RHO_MIN}.", "IVOL✓": "IVOL not in the top tercile.",
    "Events✓": f"≤ {BETA_MAX_JUMP_DAYS} event days, no earnings soon, not a recent IPO, no manual flag.",
    "Quality✓": f"Quality ≥ {BETA_QUALITY_MIN_PCT:g} (not worst quintile).",
    "Pass": "Passes every filter — a basket candidate.", "Basket": "In the current quarter's basket.",
    "Why out": "Every filter this name fails.",
}


# ================================================================ data ====
def _bust() -> str:
    return "|".join(str(p.stat().st_mtime_ns) if p.exists() else "-"
                    for p in (BETA_FILES["latest"], BETA_FILES["basket"], BETA_FILES["hedge"]))


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    return {k: load(k, {}) for k in ("latest", "basket", "hedge", "hedge_history", "status")}


def _ck(v) -> str:
    return "✓" if v else "✗"


@st.cache_data(ttl=600, show_spinner=False)
def _frame(cache_bust: str = "") -> pd.DataFrame:
    rows = _artefacts(cache_bust)["latest"].get("rows", [])
    recs = []
    for x in rows:
        if not x.get("f_liquid"):
            continue
        recs.append({
            "Rank": x.get("rank"), "Ticker": x["ticker"], "Name": x.get("name"), "Sector": x.get("sector"),
            "Size": x.get("size"), f"Score {tag('score')}": x.get("score"),
            f"bswa β {tag('bswa')}": x.get("bswa"), f"OLS β {tag('ols')}": x.get("ols"),
            f"Vasicek β {tag('vasicek')}": x.get("vasicek"), f"β pct {tag('bswa')}": x.get("beta_pct"),
            f"ρ {tag('rho')}": x.get("rho"), f"R² {tag('r2')}": x.get("r2"),
            f"IVOL {tag('ivol')}": x.get("ivol"), f"IVOL pct {tag('ivol')}": x.get("ivol_pct"),
            f"Events {tag('jumps')}": x.get("jumps"),
            f"Quality {tag('quality')}": x.get("quality"), "_q_imputed": x.get("quality_imputed"),
            f"Earnings {tag('earnings')}": x.get("earnings") or "",
            "β✓": _ck(x.get("f_beta")), "ρ✓": _ck(x.get("f_rho")), "IVOL✓": _ck(x.get("f_ivol")),
            "Events✓": _ck(x.get("f_events")), "Quality✓": _ck(x.get("f_quality")),
            "Pass": bool(x.get("passes")), "Basket": bool(x.get("in_basket")),
            "Why out": x.get("fail_reasons") or "",
        })
    return pd.DataFrame(recs)


# ================================================================ badge ====
def today_badge() -> str | None:
    try:
        h = load("hedge", {})
        lt = load("latest", {})
        if not h and not lt:
            return None
        tg = (h.get("trend") or {}).get("state", "—")
        v = h.get("vrp") or {}
        b = load("basket", {})
        color = {"ON": THEME.teal, "PARTIAL": THEME.mustard, "REDUCED": THEME.coral}.get(tg, THEME.muted)
        vrp = f"VRP {v['vrp_pct']:.0f}th pct" if v.get("vrp_pct") is not None else "VRP —"
        return uc.chip(f"CLEAN BETA — trend gate {tg} · {vrp} · basket {b.get('n', 0)} names "
                       f"β {b.get('beta') or 0:.2f} · {len(h.get('alerts') or [])} alert(s)",
                       color=color, sub="see CLEAN BETA tab")
    except Exception:
        return None


# ================================================================= main ====
def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating("B", "BAB is robust; the clean-beta refinement is newer", _EVIDENCE_NOTE),
                unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    art = _artefacts(_bust())
    if not art["latest"].get("rows") and not art["hedge"]:
        st.markdown(stamp("—", "CLEAN BETA"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.beta.compute --action rebalance` once; the nightly "
                "`--action auto` then re-screens monthly, rebalances quarterly and monitors hedges daily.")
        _methodology()
        return
    st.markdown(stamp(art["latest"].get("as_of", "—"), "CLEAN BETA · screen"), unsafe_allow_html=True)
    _methodology()
    sub = st.radio("View", SUBVIEWS, horizontal=True, key="beta_sub", label_visibility="collapsed")
    {"Screener": _screener, "Basket": _basket, "Hedge Monitor": _hedge,
     "Playbook": _playbook}[sub](art)


def _methodology() -> None:
    with st.expander("Methodology — clean beta, the score, the basket, the hedges"):
        st.markdown(
            f"**Beta engine — bswa (Welch 2022).** Each daily stock return is winsorized into the band between "
            f"(1−δ)·r_m and (1+δ)·r_m with δ = {BETA_BSWA_DELTA:g}, which bounds any single day's implied beta "
            f"to [{1 - BETA_BSWA_DELTA:g}, {1 + BETA_BSWA_DELTA:g}]. Then weighted least squares on SPY with "
            f"weights exp(−{BETA_BSWA_LAMBDA:g}·age in years) — a ~4-month half-life. One-off events (an "
            "earnings gap, a takeover pop) cannot dominate the estimate.\n\n"
            f"**Liquid universe.** Russell 1000, price ≥ ${BETA_MIN_PRICE:g}, market cap ≥ "
            f"${BETA_MIN_MKTCAP / 1e9:g}B, 63-day average dollar volume ≥ ${BETA_MIN_ADV_USD / 1e6:g}M, listed "
            "options (checked for finalists).\n\n"
            f"**Filters.** bswa β ≥ {BETA_PCT_ENTER:g}th percentile (an incumbent stays until it falls below the "
            f"{BETA_PCT_EXIT:g}th — buffer bands cut turnover) · ρ ≥ {BETA_RHO_MIN} · IVOL percentile < "
            f"{BETA_IVOL_MAX_PCT:.0f} · ≤ {BETA_MAX_JUMP_DAYS} days with a >{BETA_JUMP_SIGMA:g}σ residual, no "
            f"earnings within {BETA_EARNINGS_DAYS} days, not a recent IPO · quality ≥ "
            f"{BETA_QUALITY_MIN_PCT:g}th percentile.\n\n"
            "**Quality-Beta Score.** " + " + ".join(f"{v:.0%} {k}" for k, v in BETA_SCORE_WEIGHTS.items())
            + ", each a within-universe percentile (IVOL inverted; events = 100 − 25 per event day).\n\n"
            f"**Basket.** Equal weight (cap {BETA_NAME_CAP:.0%}), sector cap {BETA_SECTOR_CAP:.0%}. Seats go "
            "to passing incumbents first, then each size bucket gets representation, then the rest fill by "
            f"score minus {BETA_CORR_PENALTY:g} points per unit of correlation above {BETA_CORR_THRESHOLD} "
            "with any name already seated. Effective bets = (Σλ)²/Σλ² of the members' correlation eigenvalues "
            "(N uncorrelated names = N, N identical names = 1); variance ENB (Meucci) is the inverse "
            "Herfindahl of each principal component's share of basket variance.\n\n"
            "**Cadence.** Screen re-estimated monthly, basket rebalanced quarterly (Jan/Apr/Jul/Oct), hedge "
            "monitor daily.\n\n**Data-confidence tags.** "
            + " · ".join(f"**[{k}]** {v}" for k, v in CONF.items()))


# ============================================================= screener ====
def _screener(art: dict) -> None:
    df = _frame(_bust())
    lt = art["latest"]
    fc = lt.get("filter_counts", {})
    st.markdown(uc.numeric_slab([
        {"label": "Universe", "value": lt.get("n", 0), "sub": f"{lt.get('n_priced', 0)} priced"},
        {"label": "Liquid", "value": lt.get("n_liquid", 0)},
        {"label": "High β", "value": fc.get("f_beta", 0), "sub": f"≥ {BETA_PCT_ENTER:g}th pct"},
        {"label": "Pass all", "value": lt.get("n_pass", 0), "color": THEME.teal},
    ]), unsafe_allow_html=True)
    if df.empty:
        st.info("No liquid names in the latest screen.")
        return
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    sectors = sorted(df["Sector"].dropna().unique())
    pick = c1.multiselect("Sector", sectors, key="beta_sectors")
    passes = c2.toggle("Passes all filters", value=False, key="beta_pass")
    basket_only = c3.toggle("In basket", value=False, key="beta_inb")
    min_score = c4.slider("Min score", 0, 100, 0, key="beta_min")
    q = st.text_input("Ticker / name search", key="beta_q", placeholder="e.g. NVDA")
    v = df
    if pick:
        v = v[v["Sector"].isin(pick)]
    if passes:
        v = v[v["Pass"]]
    if basket_only:
        v = v[v["Basket"]]
    score_col = f"Score {tag('score')}"
    v = v[v[score_col].fillna(0) >= min_score]
    if q:
        ql = q.strip().lower()
        v = v[v["Ticker"].str.lower().str.contains(ql) | v["Name"].fillna("").str.lower().str.contains(ql)]
    st.markdown(section(f"Quality-Beta ranking — {len(v)} names", 0,
                        help="Click any column header to sort. ✓/✗ columns show each filter; 'Why out' "
                             "lists every failure."), unsafe_allow_html=True)
    show = v.drop(columns=["_q_imputed"])
    fmt = {c: "{:.2f}" for c in show.columns if c.split(" ")[0] in ("bswa", "OLS", "Vasicek", "ρ", "R²")}
    fmt.update({f"IVOL {tag('ivol')}": "{:.1%}", score_col: "{:.1f}", f"Quality {tag('quality')}": "{:.0f}",
                f"β pct {tag('bswa')}": "{:.0f}", f"IVOL pct {tag('ivol')}": "{:.0f}"})
    st.dataframe(show.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True,
                 height=min(720, 38 + 35 * len(show)), column_config=uc.colcfg(show.columns, COLS))
    n_imp = int(v["_q_imputed"].fillna(False).sum())
    if n_imp:
        st.caption(f"{n_imp} name(s) show quality 50 [E]: too few fundamentals to score, so neutral — never "
                   "a fail.")
    st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                       file_name=f"clean_beta_{lt.get('as_of', '')}.csv", mime="text/csv")
    excl = [x for x in lt.get("rows", []) if not x.get("f_liquid")]
    if excl:
        with st.expander(f"{len(excl)} names outside the liquid universe"):
            st.dataframe(pd.DataFrame([{"Ticker": x["ticker"], "Name": x.get("name"),
                                        "Reason": x.get("excluded_reason")} for x in excl]),
                         use_container_width=True, hide_index=True)


# =============================================================== basket ====
def _basket(art: dict) -> None:
    b = art["basket"]
    if not b.get("members"):
        st.info("No basket yet — it is built on the first screen and rebalanced quarterly.")
        return
    st.markdown(uc.numeric_slab([
        {"label": f"Names {tag('score')}", "value": b["n"], "sub": f"quarter {b.get('quarter', '—')}"},
        {"label": f"Basket β {tag('basket_beta')}", "value": f"{b.get('beta') or 0:.2f}",
         "color": THEME.mustard,
         "sub": f"incl. cash · invested names {b.get('beta_invested') or 0:.2f} · {b.get('invested') or 0:.0%} invested"},
        {"label": f"Ex-ante vol {tag('basket_vol')}", "value": uc.fmt_pct(b.get("vol"), 0, signed=False),
         "sub": "252d covariance"},
        {"label": f"Effective bets {tag('enb')}", "value": f"{b.get('eff_dim') or 0:.1f}", "color": THEME.teal,
         "sub": f"of {b['n']} names · avg ρ {b.get('avg_corr') or 0:.2f}"},
        {"label": f"Variance ENB {tag('enb')}", "value": f"{b.get('enb') or 0:.2f}",
         "sub": "share of basket variance per factor"},
    ]), unsafe_allow_html=True)
    if b.get("short"):
        st.warning(f"{b['n']} names seated of {b.get('n_candidates', 0)} that pass every filter — below the "
                   "30-name floor. Passing names are concentrated in few sectors, and the 15% sector cap "
                   "limits how many each sector can seat. The basket is not padded with failing names or "
                   f"over-weighted sectors; the rest is uninvested ({1 - (b.get('invested') or 0):.0%}).")
    st.caption("Effective bets = (Σλ)²/Σλ² over the eigenvalues of the members' correlation matrix: how many "
               "independent names the basket behaves like. Correlated high-beta names are far fewer bets than "
               "names. Variance ENB (Meucci) asks how many factors drive the basket's VARIANCE — near 1 for any "
               "long-only high-beta basket, because the market factor is almost all of it. That is the point of "
               "the sleeve, and why hedging happens at the index.")
    sec = pd.DataFrame([{"Sector": k, "Weight": v} for k, v in (b.get("sectors") or {}).items()])
    if not sec.empty:
        st.markdown(section("Sector weights", 1, help=f"Sector cap {BETA_SECTOR_CAP:.0%}."),
                    unsafe_allow_html=True)

        def build(alt):
            return (alt.Chart(sec).mark_bar(color=THEME.navy).encode(
                x=alt.X("Weight:Q", axis=alt.Axis(format=".0%"), title=None),
                y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=0)),
                tooltip=["Sector", alt.Tooltip("Weight:Q", format=".1%")])
                .properties(height=max(160, 26 * len(sec))))
        uc.render_chart(build, fallback=sec)
    st.markdown(section("Members", 2), unsafe_allow_html=True)
    mdf = pd.DataFrame([{"Ticker": m["ticker"], "Name": m.get("name"), "Sector": m["sector"],
                         "Size": m.get("size"), "Weight": m.get("weight"), f"bswa β {tag('bswa')}": m.get("bswa"),
                         f"ρ {tag('rho')}": m.get("rho"), f"IVOL {tag('ivol')}": m.get("ivol"),
                         f"Score {tag('score')}": m.get("score"), "Seated by": m.get("why")}
                        for m in b["members"]])
    st.dataframe(mdf.style.format({"Weight": "{:.1%}", f"bswa β {tag('bswa')}": "{:.2f}",
                                   f"ρ {tag('rho')}": "{:.2f}", f"IVOL {tag('ivol')}": "{:.1%}",
                                   f"Score {tag('score')}": "{:.1f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    if b.get("added") or b.get("removed"):
        st.caption(f"Last rebalance — added: {', '.join(b.get('added') or []) or 'none'} · removed: "
                   f"{', '.join(b.get('removed') or []) or 'none'}")


# ================================================================ hedge ====
def _hedge(art: dict) -> None:
    h = art["hedge"]
    if not h:
        st.info("No hedge-monitor data yet.")
        return
    tg = h.get("trend") or {}
    vt = h.get("vol_target") or {}
    v = h.get("vrp") or {}
    bt = h.get("btal") or {}
    color = {"ON": THEME.teal, "PARTIAL": THEME.mustard, "REDUCED": THEME.coral}.get(tg.get("state"), THEME.muted)
    st.markdown(uc.state_banner(color, f"Trend gate {tg.get('state', '—')}",
                                f"suggested sleeve exposure {tg.get('exposure', 1):.0%} · as of {h.get('as_of')}"),
                unsafe_allow_html=True)
    for a in h.get("alerts") or []:
        (st.error if a["level"] == "high" else st.warning)(a["text"])
    net = (tg.get("exposure") or 1.0) * (vt.get("scale") or 1.0)
    st.markdown(uc.numeric_slab([
        {"label": f"SPY 12-1 {tag('trend')}", "value": uc.fmt_pct(tg.get("ret_12_1"), 1),
         "sub": f"vs 10m MA {uc.fmt_pct(tg.get('vs_ma'), 1)}"},
        {"label": f"Vol scale {tag('vol_scale')}", "value": f"{vt.get('scale') or 0:.2f}",
         "sub": f"target {BETA_VOL_TARGET:.0%} / forecast {uc.fmt_pct(vt.get('forecast'), 0, signed=False)}"},
        {"label": "Net exposure", "value": f"{net:.0%}", "color": color, "sub": "trend × vol scale"},
        {"label": f"VRP {tag('vrp')}", "value": f"{(v.get('vrp') or 0) * 100:+.1f}",
         "sub": f"{v.get('vrp_pct') or 0:.0f}th pct (5y) · VIX {v.get('vix') or 0:.1f}"},
        {"label": f"BTAL {tag('btal')}", "value": bt.get("state", "—"),
         "color": THEME.coral if bt.get("state") == "HEADWIND" else THEME.text,
         "sub": f"12-1 {uc.fmt_pct(bt.get('ret_12_1'), 1)}"},
    ], min_width=140), unsafe_allow_html=True)

    s = (v.get("series") or {})
    if s.get("dates"):
        st.markdown(section("Volatility risk premium — VIX minus forecast realized vol", 3,
                            help="Above zero = implied vol is pricing more than realized vol has been "
                                 "delivering (protection is expensive). Low/negative = protection is cheap. "
                                 "Size puts on THIS, not on the VIX level."), unsafe_allow_html=True)
        vdf = pd.DataFrame({"date": pd.to_datetime(s["dates"]), "VRP (vol pts)": [x * 100 if x is not None
                                                                                 else None for x in s["vrp"]]})

        def build(alt):
            base = alt.Chart(vdf).mark_area(line={"color": THEME.navy}, color=THEME.navy, opacity=0.35).encode(
                x=alt.X("date:T", title=None), y=alt.Y("VRP (vol pts):Q", title="VRP, vol points"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("VRP (vol pts):Q", format=".1f")])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.mustard).encode(y="y:Q")
            return (base + zero).properties(height=220)
        uc.render_chart(build, fallback=vdf)

    p = h.get("puts") or {}
    lo, hi = BETA_PUT_BUDGET
    st.markdown(section(f"Advisory put ladder — SPY 25Δ/10Δ put spreads {tag('puts')}", 4,
                        help="Three staggered expiries, rolled monthly so one rung is always near 2-4 months. "
                             "Budget scales from " f"{lo:.1%} to {hi:.1%} a year as the VRP percentile falls."),
                unsafe_allow_html=True)
    if p.get("rungs"):
        st.caption(f"Basket β {p.get('beta'):.2f} · put budget {p.get('budget', 0):.2%}/yr · full-coverage "
                   f"carry {p.get('ladder_carry_full', 0):.2%}/yr → suggested coverage "
                   f"{p.get('coverage') or 0:.0%} of beta-weighted notional · SPY {p.get('spot')}")
        pdf = pd.DataFrame([{"Expiry": x["expiry"], "DTE": x["dte"], "Long K": x["long_strike"],
                             "Long Δ": x["long_delta"], "Short K": x["short_strike"], "Short Δ": x["short_delta"],
                             "OTM": x["long_otm"], "Cost $": x["cost"], "Cost % spot": x["cost_pct_spot"],
                             "Max ×": x["max_multiple"], f"Harvest at {BETA_HARVEST_MULTIPLE:g}× $": x["harvest_at"],
                             "Contracts / $1M": x.get("contracts_per_mm")} for x in p["rungs"]])
        st.dataframe(pdf.style.format({"Long K": "{:.0f}", "Short K": "{:.0f}",
                                       "Long Δ": "{:.2f}", "Short Δ": "{:.2f}", "OTM": "{:+.1%}",
                                       "Cost $": "{:.2f}", "Cost % spot": "{:.2%}", "Max ×": "{:.1f}",
                                       f"Harvest at {BETA_HARVEST_MULTIPLE:g}× $": "{:.2f}",
                                       "Contracts / $1M": "{:.2f}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)
        st.caption(f"Monetize: when a rung's value reaches {BETA_HARVEST_MULTIPLE:g}× its entry cost, or VIX "
                   f"reaches its {BETA_VIX_HARVEST_PCT:g}th percentile, harvest part of it and restrike lower. "
                   "Advisory only — no positions are tracked. Mid prices from yfinance; real fills will be worse.")
    else:
        st.info(p.get("error") or "No usable put spreads in the chain this run.")

    te = h.get("trend_etfs") or {}
    if te:
        st.markdown(section("Trend-follower check — managed-futures ETFs", 5), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"ETF": k, "12-1": x.get("ret_12_1"), "vs 10m MA": x.get("vs_ma")}
                                   for k, x in te.items()]).style.format({"12-1": "{:+.1%}",
                                                                          "vs 10m MA": "{:+.1%}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)
    hist = pd.DataFrame(art["hedge_history"].get("rows", []))
    if len(hist) > 1:
        with st.expander("Hedge-state history"):
            st.dataframe(hist.iloc[::-1], use_container_width=True, hide_index=True)


# ============================================================= playbook ====
_PLAYBOOK = [
    ("Trend gate (primary, inducible)", "B+",
     "Cut beta when SPY's 12-1 month return and its 10-month average both turn negative. Cheap when quiet; "
     "historically strongest in extreme equity years, because slow bear markets give it time to act. Weakness: "
     "whipsaws and V-shaped crashes; trend was weak for ~4 years after 2008.",
     "Hurst, Ooi & Pedersen (2017); Moskowitz, Ooi & Pedersen (2012)",
     f"Exposure {BETA_TREND_EXPOSURE['ON']:.0%} / {BETA_TREND_EXPOSURE['PARTIAL']:.0%} / "
     f"{BETA_TREND_EXPOSURE['REDUCED']:.0%} for ON / PARTIAL / REDUCED."),
    ("Vol targeting (risk control)", "B",
     "Scale exposure by target ÷ forecast vol. Keeps the sleeve's risk steady, de-risks into vol spikes. "
     "In-sample alpha (Moreira & Muir) mostly fails out of sample — treat it as risk control, not a return source.",
     "Moreira & Muir (2017); Cederburg, O'Doherty, Wang & Yan (2020)",
     f"Target {BETA_VOL_TARGET:.0%}, scale capped at 1.0 (no leverage by default)."),
    ("Index put spreads (small crash sleeve)", "B",
     "Rolling protective puts have historically lost to simply holding less equity; they win only in sharp "
     "crashes that land before expiry. So keep them small, use spreads to cut carry, and stagger expiries.",
     "Israelov (2019), Pathetic Protection",
     f"Budget {BETA_PUT_BUDGET[0]:.1%}–{BETA_PUT_BUDGET[1]:.1%} a year, 25Δ long / 10Δ short, 2–4 months."),
    ("Size by VRP, not by VIX", "B",
     "Low implied vol is not the same as cheap puts. What matters is implied minus subsequently realized vol. "
     "Buy more protection when the VRP is low, not merely when the VIX is low.",
     "Israelov & Nielsen (2015), Still Not Cheap", "Budget slides toward the top as VRP percentile falls."),
    ("Hedge at the index, not single names", "A-",
     "Index options embed a correlation-risk premium that a high-correlation basket is exactly exposed to, and "
     "single-name puts on high-IVOL stocks are the most overpriced. Beta-weight SPY (or sector-ETF) puts.",
     "Driessen, Maenhout & Vilkov (2009); Cao & Han (2013)", "Contracts per $1M = β × $1M ÷ (100 × SPY)."),
    ("Monetize, don't flip", "B-",
     "When puts pay off, harvest part and restrike lower. Do not switch to selling vol in high-vol regimes — "
     "that leaves a long-beta book short gamma exactly when it matters most.",
     "Spec hedge philosophy; Israelov (2019)",
     f"Harvest at {BETA_HARVEST_MULTIPLE:g}× entry or VIX ≥ {BETA_VIX_HARVEST_PCT:g}th pct."),
    ("BTAL as a regime gate", "C+",
     "BTAL (long low-beta, short high-beta) is a live betting-against-beta factor. When it trends up, the "
     "high-beta sleeve is fighting the tape. Hong & Sraer: high beta is most overpriced when disagreement is high.",
     "Frazzini & Pedersen (2014); Hong & Sraer (2016)", "12-1 return and 200-day trend both up = HEADWIND."),
    ("The null to beat", "A",
     "If the clean-beta stock sleeve does not beat SPY futures levered to the same ex-ante beta with the same "
     "overlay, after costs, stock selection is adding cost, not value. That test belongs to PARALLAX, fed by "
     "this tab's monthly export.", "Spec Phase 3", "Snapshots in data/beta/exports/."),
]


def _playbook(art: dict) -> None:
    st.markdown(section("Hedge playbook — what each defense buys you", 0,
                        help="Inducible defenses (trend, vol target) are the primary hedge; constitutive "
                             "defenses (puts) are a small, always-on crash sleeve."), unsafe_allow_html=True)
    for title, grade, body, cite, rule in _PLAYBOOK:
        st.markdown(evidence_rating(grade, title, body), unsafe_allow_html=True)
        st.caption(f"Rule here: {rule} · Source: {cite}")

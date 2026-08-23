"""MOMENTUM tab — Russell 1000 multi-factor stock momentum engine.

Reads only committed JSON (data/mom/*), with one deliberate exception: the
individual-stock GMMA chart fetches that ONE ticker's price history on
demand (cached), because a 500-day x 7-MA panel for all ~1000 stocks would
be a multi-MB daily rewrite (see mom/__init__.py's note on artifact size).

Five sub-views on one radio, so the user can move Russell 1000 -> ranked
list -> one factor -> one stock -> that stock's history without leaving the
tab: Overview (breadth + heatmap) -> Rankings (filterable table) -> Factors
(one panel each + correlation check) -> Sectors (where is momentum
concentrated) -> Stock (full drill-down).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME, MOM_MEMBERSHIP_START, MOM_WEIGHTS, MOM_HORIZON_WEIGHTS, MOM_STATES
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import DISCLAIMER, SURVIVORSHIP_NOTE, FACTORS, FACTOR_LABELS, HORIZONS, HORIZON_LABELS, load
from . import history as mom_history

SUBVIEWS = ["Overview", "Rankings", "Factors", "Sectors", "Stock"]

COLS = {
    "Rank": "Position in the priced Russell 1000, ranked by composite momentum score.",
    "Ticker": "Exchange ticker.",
    "Company": "Company name.",
    "Sector": "GICS sector (Wikipedia components table, best-effort).",
    "Industry": "GICS industry, from a rolling yfinance metadata cache (fills in over ~1 week).",
    "Score": "Composite momentum score: -20 (extreme bearish) to +20 (extreme bullish).",
    "State": "Categorical read of the composite score (see the methodology expander for bands).",
    "TS": "Time-series momentum: this stock's own trailing return, vol-adjusted, blended across 6 horizons.",
    "Breakout": "Has the daily close broken its own trailing high/low, blended across 6 horizons.",
    "XSec": "Cross-sectional momentum: this stock's momentum rank vs the rest of the Russell 1000.",
    "Speed": "Trend speed: how fast the moving-average (GMMA) structure is changing.",
    "Strength": "Momentum strength: are MA slopes rising/accelerating and is the trend statistically smooth.",
    "Mkt Cap": "Market capitalization, from the rolling metadata cache (may be blank for recently-added tickers).",
}

_EVIDENCE_NOTE = ("Momentum is one of the most replicated cross-sectional anomalies in equities "
                  "(Jegadeesh & Titman 1993; Moskowitz, Ooi & Pedersen 2012), but it also crashes "
                  "hard and somewhat predictably in panic states (Daniel & Moskowitz 2016; Barroso "
                  "& Santa-Clara 2015), and large-cap momentum (this universe) is its most crowded "
                  "corner. Promotion to A- requires our OWN out-of-sample information coefficient — "
                  "see the Factors tab, which fills in as picks accrue.")

_FINDINGS = [
    {"stat": "Momentum is among the most replicated cross-sectional anomalies in equities.",
     "cite": "Jegadeesh & Titman (1993)"},
    {"stat": "Time-series momentum earns significant excess returns across asset classes.",
     "cite": "Moskowitz, Ooi & Pedersen (2012)"},
    {"stat": "Scaling by a strategy's own realized volatility materially raises momentum's Sharpe.",
     "cite": "Barroso & Santa-Clara (2015)"},
    {"stat": "Momentum crashes are partly forecastable — they cluster after market declines, in high-vol states.",
     "cite": "Daniel & Moskowitz (2016)"},
]


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    """Streamlit runs EVERY tab body on EVERY rerun, so an uncached read here
    would be paid on every click anywhere in the app (holdings/view.py's
    pattern)."""
    return {
        "scores": load("scores", {}), "detail": load("detail", {}),
        "sectors": load("sectors", {}), "diagnostics": load("diagnostics", {}),
        "status": load("status", {}),
    }


def today_badge() -> str | None:
    """TODAY-tab chip: only speaks up when there's a real breadth read."""
    try:
        scores = load("scores", {})
        rows = [r for r in scores.get("rows", []) if not r.get("excluded")]
        if not rows:
            return None
        n_bull = sum(1 for r in rows if (r.get("composite") or 0) >= 10)
        n_bear = sum(1 for r in rows if (r.get("composite") or 0) <= -10)
        if not n_bull and not n_bear:
            return None
        color = THEME.teal if n_bull >= n_bear else THEME.coral
        return uc.chip(f"MOMENTUM — {n_bull} strong-bullish / {n_bear} strong-bearish "
                       f"in the Russell 1000", color=color, sub="see MOMENTUM tab")
    except Exception:
        return None


# ------------------------------------------------------------------ helpers --
def _priced_df(scores: dict) -> pd.DataFrame:
    rows = [r for r in scores.get("rows", []) if not r.get("excluded")]
    return pd.DataFrame(rows)


def _state_color(state: str | None) -> str:
    if not state:
        return THEME.muted
    if "BULLISH" in state:
        return THEME.teal
    if "BEARISH" in state:
        return THEME.coral
    return THEME.mustard


def _picked_ticker(event) -> str | None:
    rows = (event or {}).get("selection", {}).get("pick", [])
    return rows[0].get("ticker") if rows else None


# -------------------------------------------------------------------- main --
def render() -> None:
    st.caption(DISCLAIMER)
    status = load("status", {})
    art = _artefacts(cache_bust=str(status.get("date", "")))
    scores = art["scores"]
    rows = scores.get("rows", [])

    st.markdown(evidence_rating("B+", "replicated, but crowded and crash-prone", _EVIDENCE_NOTE),
                unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    if not rows:
        st.markdown(stamp("—", "MOMENTUM"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.mom.compute --action auto` to populate the "
                "Russell 1000 momentum scores.")
        return

    st.markdown(stamp(scores.get("as_of", "—"), "MOMENTUM"), unsafe_allow_html=True)

    with st.expander("Methodology — five factors, one transparent composite"):
        weights_txt = " · ".join(f"{FACTOR_LABELS[k]} {v:.0%}" for k, v in MOM_WEIGHTS.items())
        horizons_txt = " · ".join(f"{HORIZON_LABELS[h]} {w:.0%}" for h, w in MOM_HORIZON_WEIGHTS.items())
        st.markdown(
            "Every factor is normalized to **[-1, +1]**; the composite is "
            "`20 x sum(weight_i x factor_i)`, so it always lands in **[-20, +20]**.\n\n"
            f"**Factor weights:** {weights_txt} — not equal: time-series and cross-sectional "
            "momentum carry the most independent information (Moskowitz-Ooi-Pedersen decompose "
            "the premium into distinct components), breakout is a non-linear read of the same "
            "path as time-series so it's downweighted, and speed/strength (both MA-derived, the "
            "likeliest redundant pair) split .15/.20. The **Factors** tab renders the live factor "
            "correlation matrix so this assumption is checked, not just asserted.\n\n"
            f"**Horizon weights** (shared by time-series/breakout/cross-sectional): {horizons_txt} "
            "— tilted toward the slower legs; the 1-month leg is kept small because very "
            "short-horizon returns are known to reverse, not persist (Jegadeesh 1990).\n\n"
            "**Prices** are yfinance daily closes, split- AND dividend-adjusted — the only "
            "self-consistent choice for both return math and trailing-high/low breakout levels, "
            "since an unadjusted series jumps at every split. One honest side effect: dividend "
            "adjustment lowers historical prices, so a high-yield name reaches a \"new adjusted "
            "high\" marginally more easily than a non-payer.\n\n"
            f"**Signal states:** {' · '.join(f'{t:+.0f} {l}' for t, l in MOM_STATES)}.\n\n"
            f"{SURVIVORSHIP_NOTE}"
        )

    sub = st.radio("View", SUBVIEWS, horizontal=True, label_visibility="collapsed", key="mom_sub")
    if sub == "Overview":
        _overview(scores, art["diagnostics"])
    elif sub == "Rankings":
        _rankings(scores)
    elif sub == "Factors":
        _factors(scores, art["diagnostics"])
    elif sub == "Sectors":
        _sectors(art["sectors"])
    else:
        _stock(scores, art["detail"])


# --------------------------------------------------------------- overview --
def _overview(scores: dict, diagnostics: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored names yet.")
        return
    n = len(df)
    med = float(df["composite"].median())
    pct_bull = float((df["composite"] >= 5).mean())
    n_ext_bull = int((df["composite"] >= 15).sum())
    n_ext_bear = int((df["composite"] <= -15).sum())
    breadth_color = THEME.teal if pct_bull >= 0.5 else THEME.coral
    st.markdown(uc.state_banner(breadth_color, "BREADTH",
                                f"{pct_bull:.0%} of the priced Russell 1000 is bullish (score >= +5) — "
                                f"median score {med:+.1f}"), unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Universe (scored)", "value": str(n), "color": THEME.text},
        {"label": "Median score", "value": f"{med:+.1f}", "color": THEME.text},
        {"label": "% Bullish (>=+5)", "value": f"{pct_bull:.0%}", "color": THEME.teal},
        {"label": "Extreme bullish (>=+15)", "value": str(n_ext_bull), "color": THEME.teal},
        {"label": "Extreme bearish (<=-15)", "value": str(n_ext_bear), "color": THEME.coral},
        {"label": "Last calculated", "value": scores.get("as_of", "—"), "color": THEME.muted},
    ]), unsafe_allow_html=True)

    excluded_n = scores.get("n", 0) - scores.get("n_scored", 0)
    if excluded_n:
        st.caption(f"{excluded_n} Russell 1000 names excluded — insufficient price history "
                   "(recently listed) or no price data returned this run.")

    st.markdown(section("Russell 1000 — momentum heatmap", 4,
                        help="Every scored stock, sorted by composite score. Click a cell to open "
                             "its full breakdown on the Stock tab."), unsafe_allow_html=True)
    n_cols = st.slider("Heatmap columns", 20, 60, 40, key="mom_hm_cols",
                       help="Fewer columns = taller cells, easier to read on a phone.")
    grid = df.sort_values("composite", ascending=False).reset_index(drop=True)
    grid["col"] = grid.index % n_cols
    grid["row"] = grid.index // n_cols

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["ticker"], on="click",
                                   clear="dblclick", empty=False)
        return (alt.Chart(grid).mark_rect(stroke=THEME.bg, strokeWidth=0.5).encode(
            x=alt.X("col:O", title=None, axis=None),
            y=alt.Y("row:O", title=None, axis=None, sort="descending"),
            color=alt.Color("composite:Q", title="Score", scale=uc.diverging_scale(20, alt),
                            legend=alt.Legend(labelLimit=0)),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.92)),
            tooltip=["ticker", "name", "sector",
                     alt.Tooltip("composite:Q", format="+.1f", title="Composite"),
                     alt.Tooltip("rank:Q", title="Rank"), "state"],
        ).add_params(pick).properties(height=max(240, 16 * (grid["row"].max() + 1))))

    event = uc.render_chart(build, fallback=grid[["ticker", "composite", "sector"]],
                            on_select="rerun", key="mom_heatmap")
    picked = _picked_ticker(event)
    if picked:
        st.session_state["mom_sub"] = "Stock"
        st.session_state["mom_pick"] = picked
        st.rerun()

    st.markdown(section("Score distribution", 1), unsafe_allow_html=True)
    hist_df = df[["composite"]].copy()

    def build_hist(alt):
        return (alt.Chart(hist_df).mark_bar().encode(
            x=alt.X("composite:Q", bin=alt.Bin(maxbins=40), title="Composite score"),
            y=alt.Y("count():Q", title="Stocks"),
            color=alt.Color("composite:Q", scale=uc.diverging_scale(20, alt), legend=None,
                            bin=alt.Bin(maxbins=40)),
        ).properties(height=200))

    uc.render_chart(build_hist, fallback=hist_df)


# --------------------------------------------------------------- rankings --
def _rankings(scores: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored names yet.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("Search ticker or company", "", key="mom_rank_q")
    sectors = sorted(df["sector"].dropna().unique().tolist())
    sec_pick = c2.multiselect("Sector", sectors, default=[], key="mom_rank_sec")
    preset = c3.selectbox("Preset", ["All", "Top 10", "Top 25", "Top 50", "Bottom 10", "Bottom 25", "Bottom 50"],
                          key="mom_rank_preset")

    with st.expander("More filters"):
        f1, f2, f3 = st.columns(3)
        score_min, score_max = f1.slider("Composite score", -20.0, 20.0, (-20.0, 20.0), key="mom_rank_score")
        factor_pick = f2.selectbox("Factor filter", ["(none)"] + list(FACTORS),
                                   format_func=lambda k: "(none)" if k == "(none)" else FACTOR_LABELS[k],
                                   key="mom_rank_factor")
        factor_min = f3.slider("Min factor score", -1.0, 1.0, -1.0, key="mom_rank_factor_min") \
            if factor_pick != "(none)" else None
        breakout_only = st.checkbox("Only names with a fresh breakout (any horizon, confirmed)",
                                    key="mom_rank_breakout")

    view = df[(df["composite"] >= score_min) & (df["composite"] <= score_max)].copy()
    if sec_pick:
        view = view[view["sector"].isin(sec_pick)]
    if q.strip():
        ql = q.strip().lower()
        view = view[view["ticker"].str.lower().str.contains(ql) | view["name"].str.lower().str.contains(ql)]
    if factor_pick != "(none)" and factor_min is not None:
        view = view[view["factor_scores"].apply(lambda d: (d or {}).get(factor_pick, -1.0) >= factor_min)]
    if breakout_only:
        def _has_break(grid):
            return any((g or {}).get("confirmed") and (g or {}).get("state") in ("break_up", "break_down")
                      for g in (grid or {}).values())
        view = view[view["breakout_grid"].apply(_has_break)]

    view = view.sort_values("composite", ascending=False)
    if preset.startswith("Top"):
        view = view.head(int(preset.split()[1]))
    elif preset.startswith("Bottom"):
        view = view.sort_values("composite", ascending=True).head(int(preset.split()[1]))
        view = view.sort_values("composite", ascending=False)

    st.caption(f"{len(view)} of {len(df)} scored names")

    disp = pd.DataFrame({
        "Rank": view["rank"], "Ticker": view["ticker"], "Company": view["name"],
        "Sector": view["sector"], "Score": view["composite"], "State": view["state"],
        "TS": view["factor_scores"].apply(lambda d: (d or {}).get("ts")),
        "Breakout": view["factor_scores"].apply(lambda d: (d or {}).get("breakout")),
        "XSec": view["factor_scores"].apply(lambda d: (d or {}).get("xsec")),
        "Speed": view["factor_scores"].apply(lambda d: (d or {}).get("speed")),
        "Strength": view["factor_scores"].apply(lambda d: (d or {}).get("strength")),
        "Mkt Cap": view["mktcap"],
    })
    try:
        sty = (disp.style
               .map(lambda v: uc.grad_diverging(v, 20.0), subset=["Score"])
               .map(lambda v: uc.grad_diverging(v, 1.0), subset=["TS", "Breakout", "XSec", "Speed", "Strength"])
               .format({"Score": "{:+.1f}", "TS": "{:+.2f}", "Breakout": "{:+.2f}", "XSec": "{:+.2f}",
                        "Speed": "{:+.2f}", "Strength": "{:+.2f}",
                        "Mkt Cap": lambda v: uc.fmt_money(v)}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                    height=min(600, 45 + 34 * len(disp)),
                    column_config=uc.colcfg(disp.columns, COLS), key="mom_rank_table")
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown(section("Leaderboards", 3), unsafe_allow_html=True)
    lead_factor = st.selectbox("Leaderboard factor", ["Composite"] + list(FACTORS),
                               format_func=lambda k: "Composite" if k == "Composite" else FACTOR_LABELS[k],
                               key="mom_lead_factor")
    if lead_factor == "Composite":
        top = df.nlargest(15, "composite")[["ticker", "composite"]].rename(columns={"composite": "v"})
        cap = 20.0
    else:
        vv = df["factor_scores"].apply(lambda d: (d or {}).get(lead_factor))
        tmp = df.assign(v=vv).nlargest(15, "v")[["ticker", "v"]]
        top, cap = tmp, 1.0
    uc.hbar(top, x="v", y="ticker", cap=cap, title=f"Top 15 — {lead_factor if lead_factor=='Composite' else FACTOR_LABELS[lead_factor]}",
           fmt="+.2f" if lead_factor != "Composite" else "+.1f")


# ----------------------------------------------------------------- factors --
def _factors(scores: dict, diagnostics: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored names yet.")
        return
    pick = st.radio("Factor", list(FACTORS), horizontal=True, label_visibility="collapsed",
                    format_func=lambda k: FACTOR_LABELS[k], key="mom_factor_pick")
    df = df.assign(_fv=df["factor_scores"].apply(lambda d: (d or {}).get(pick)))
    st.caption(FACTOR_LABELS[pick])

    grid = df.sort_values("_fv", ascending=False).reset_index(drop=True)
    n_cols = 40
    grid["col"] = grid.index % n_cols
    grid["row"] = grid.index // n_cols

    def build(alt):
        return (alt.Chart(grid).mark_rect(stroke=THEME.bg, strokeWidth=0.5).encode(
            x=alt.X("col:O", title=None, axis=None), y=alt.Y("row:O", title=None, axis=None, sort="descending"),
            color=alt.Color("_fv:Q", title=FACTOR_LABELS[pick], scale=uc.diverging_scale(1.0, alt),
                            legend=alt.Legend(labelLimit=0)),
            tooltip=["ticker", "name", alt.Tooltip("_fv:Q", format="+.2f", title=FACTOR_LABELS[pick]),
                     alt.Tooltip("composite:Q", format="+.1f", title="Composite")],
        ).properties(height=max(200, 16 * (grid["row"].max() + 1))))
    uc.render_chart(build, fallback=grid[["ticker", "_fv"]])

    top = grid.nlargest(15, "_fv")[["ticker", "_fv"]]
    uc.hbar(top, x="_fv", y="ticker", cap=1.0, title=f"Top 15 — {FACTOR_LABELS[pick]}", fmt="+.2f")

    if pick in ("ts", "breakout", "xsec"):
        st.markdown(section(f"{FACTOR_LABELS[pick]} — horizon grid", 2), unsafe_allow_html=True)
        top20 = df.sort_values("_fv", ascending=False).head(25)
        grid_key = "ts_grid" if pick in ("ts", "xsec") else "breakout_grid"
        val_key = "ts_signal" if pick == "ts" else ("xsec_signal" if pick == "xsec" else "b")
        cols = {}
        for h in HORIZONS:
            cols[HORIZON_LABELS[h]] = top20["ts_grid" if pick != "breakout" else "breakout_grid"].apply(
                lambda g, h=h: (g or {}).get(h, {}).get(val_key))
        hgrid = pd.DataFrame({"Ticker": top20["ticker"].values, **{k: v.values for k, v in cols.items()}})
        try:
            sty = hgrid.style
            for c in hgrid.columns[1:]:
                sty = sty.map(lambda v: uc.grad_diverging(v, 1.0), subset=[c])
            sty = sty.format({c: "{:+.2f}" for c in hgrid.columns[1:]})
            st.dataframe(sty, use_container_width=True, hide_index=True, height=min(430, 45 + 34 * len(hgrid)))
        except Exception:
            st.dataframe(hgrid, use_container_width=True, hide_index=True)

    st.markdown(section("Factor correlation — checking the redundancy assumption", 5,
                        help="Spearman correlation of the five factor scores across the priced "
                             "universe. Pairs above 0.85 are flagged: the app's weights already "
                             "tilt down the pair we expect to be most redundant (speed/strength); "
                             "this is the check on that call, not an assertion of it."),
                unsafe_allow_html=True)
    corr = diagnostics.get("correlation", {})
    matrix = corr.get("matrix", {})
    if matrix:
        cdf = pd.DataFrame(matrix).reindex(index=list(FACTORS), columns=list(FACTORS))
        cdf.index = [FACTOR_LABELS[k] for k in FACTORS]
        cdf.columns = [FACTOR_LABELS[k] for k in FACTORS]
        try:
            sty = cdf.style.map(lambda v: uc.grad_diverging(v, 1.0)).format("{:+.2f}")
            st.dataframe(sty, use_container_width=True, height=min(300, 45 + 34 * len(cdf)))
        except Exception:
            st.dataframe(cdf, use_container_width=True)
        flagged = corr.get("flagged_pairs", [])
        if flagged:
            msg = " · ".join(f"{FACTOR_LABELS[p['a']]} vs {FACTOR_LABELS[p['b']]}: {p['corr']:+.2f}"
                             for p in flagged)
            st.markdown(uc.state_banner(THEME.mustard, "REDUNDANCY FLAG", msg), unsafe_allow_html=True)
        else:
            st.caption("No factor pair currently exceeds the 0.85 redundancy threshold.")
    else:
        st.caption("Correlation matrix not yet available (needs at least 10 scored names).")

    st.markdown(section("Predictive diagnostics — information coefficient", 4,
                        help="Spearman rank correlation between the composite (and, once tagged "
                             "history exists, each factor) at pick time and the realized SPY-excess "
                             "return at each horizon. Fills in as picks.json accumulates evaluated "
                             "rows — thin or empty on a freshly-launched tab."), unsafe_allow_html=True)
    ic = diagnostics.get("ic_by_horizon", {})
    if any(v is not None for v in ic.values()):
        st.markdown(uc.numeric_slab([
            {"label": f"IC @ {h}td", "value": (f"{v:+.3f}" if v is not None else "—"),
             "color": THEME.teal if (v or 0) > 0 else THEME.coral}
            for h, v in ic.items()
        ]), unsafe_allow_html=True)
    else:
        st.caption("No evaluated picks yet — the composite's information coefficient will appear "
                  "here once the first decile picks reach their +20 trading-day horizon.")


# ----------------------------------------------------------------- sectors --
def _sectors(sectors_doc: dict) -> None:
    secs = sectors_doc.get("sectors", {})
    inds = sectors_doc.get("industries", {})
    if not secs:
        st.info("No sector data yet.")
        return
    st.markdown(section("Sector momentum", 2,
                        help="Mean/median composite score, breadth, and dispersion per GICS sector."),
                unsafe_allow_html=True)
    sdf = pd.DataFrame(secs).T.reset_index().rename(columns={"index": "Sector"})
    sdf = sdf.sort_values("mean", ascending=False)
    disp = pd.DataFrame({
        "Sector": sdf["Sector"], "N": sdf["n"], "Mean": sdf["mean"], "Median": sdf["median"],
        "% Bullish": sdf["pct_bullish"], "% Bearish": sdf["pct_bearish"],
        "Extreme Bull": sdf["n_extreme_bullish"], "Extreme Bear": sdf["n_extreme_bearish"],
        "Dispersion": sdf["dispersion"],
    })
    try:
        sty = (disp.style.map(lambda v: uc.grad_diverging(v, 10.0), subset=["Mean", "Median"])
               .format({"Mean": "{:+.2f}", "Median": "{:+.2f}", "% Bullish": "{:.0%}", "% Bearish": "{:.0%}",
                        "Dispersion": "{:.2f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True, height=min(430, 45 + 34 * len(disp)))
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    bar = sdf[["Sector", "mean"]].rename(columns={"mean": "v"}) if "Sector" in sdf else sdf[["index", "mean"]]
    bar = pd.DataFrame({"Sector": sdf["Sector"], "v": sdf["mean"]})
    uc.hbar(bar, x="v", y="Sector", cap=10.0, title="Mean composite score by sector", fmt="+.2f")

    st.markdown(section("Industry momentum — top / bottom 15", 5), unsafe_allow_html=True)
    idf = pd.DataFrame(inds).T.reset_index().rename(columns={"index": "Industry"})
    if not idf.empty:
        idf = idf.sort_values("mean", ascending=False)
        top = idf.head(15)[["Industry", "mean"]].rename(columns={"mean": "v"})
        bot = idf.tail(15)[["Industry", "mean"]].rename(columns={"mean": "v"})
        c1, c2 = st.columns(2)
        with c1:
            uc.hbar(top, x="v", y="Industry", cap=10.0, title="Strongest industries", fmt="+.2f")
        with c2:
            uc.hbar(bot, x="v", y="Industry", cap=10.0, title="Weakest industries", fmt="+.2f")


# ------------------------------------------------------------------- stock --
def _explain(row: dict) -> list[str]:
    """Plain-English signal explanation built from stored flags — no free text,
    every bullet traces to a stored number."""
    out = []
    grid = row.get("ts_grid") or {}
    best_h, best_dev = None, 0.0
    for h, g in grid.items():
        p = g.get("pctile")
        if p is not None and abs(p - 50) > best_dev:
            best_dev, best_h = abs(p - 50), h
    if best_h:
        p = grid[best_h]["pctile"]
        out.append(f"{HORIZON_LABELS[best_h]} return in the {p:.0f}th percentile of the Russell 1000")
    bgrid = row.get("breakout_grid") or {}
    for h in ("12m", "6m", "3m"):
        g = bgrid.get(h) or {}
        if g.get("state") in ("break_up", "break_down") and g.get("days_since_break") is not None:
            direction = "high" if g["state"] == "break_up" else "low"
            days = g["days_since_break"]
            when = "today" if days == 0 else f"{days}d ago"
            out.append(f"Broke its {HORIZON_LABELS[h]} {direction} ({when}"
                      f"{', confirmed' if g.get('confirmed') else ''})")
            break
    fs = row.get("factor_scores") or {}
    if fs.get("speed", 0) is not None and abs(fs.get("speed", 0)) > 0.4:
        out.append("Moving averages " + ("bullishly aligned with fresh crossovers"
                                        if fs["speed"] > 0 else "bearishly aligned with fresh crossovers"))
    if fs.get("strength", 0) is not None:
        if fs["strength"] > 0.3:
            out.append("Trend strength accelerating with a statistically smooth path")
        elif fs["strength"] < -0.3:
            out.append("Trend deteriorating — slopes decelerating or reversing")
    return out or ["No standout single signal — the composite reflects a blend of moderate readings."]


def _stock(scores: dict, detail: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored names yet.")
        return
    tickers = sorted(df["ticker"].tolist())
    sel_key = "mom_stock_pick"
    preselect = st.session_state.get("mom_pick")
    if preselect and preselect in tickers:
        st.session_state[sel_key] = preselect
        st.session_state.pop("mom_pick", None)
    if st.session_state.get(sel_key) not in tickers:
        st.session_state[sel_key] = tickers[0]
    ticker = st.selectbox("Stock", tickers, key=sel_key,
                          format_func=lambda t: f"{t} — {df.loc[df['ticker']==t,'name'].iloc[0]}")
    row = df[df["ticker"] == ticker].iloc[0].to_dict()

    comp, state = row["composite"], row["state"]
    st.markdown(uc.state_banner(_state_color(state), f"{ticker} — {row['name']}",
                                f"{comp:+.1f} · {state}"), unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Composite", "value": f"{comp:+.1f}", "color": _state_color(state)},
        {"label": "Rank", "value": f"#{int(row['rank'])} / {len(df)}", "color": THEME.text},
        {"label": "Percentile", "value": f"{row.get('pctile', 0):.0f}th", "color": THEME.text},
        {"label": "Sector", "value": row.get("sector") or "—", "color": THEME.muted},
        {"label": "Equal-weight composite", "value": f"{row.get('composite_equal_weight', 0):+.1f}",
         "color": THEME.muted, "sub": "for comparison against the tilted weighting"},
    ]), unsafe_allow_html=True)

    st.markdown(section("Factor breakdown", 3,
                        help="Each factor's [-1,+1] score times its weight times 20 — these five "
                             "numbers sum exactly to the composite above."), unsafe_allow_html=True)
    contrib = row.get("contributions") or {}
    cdf = pd.DataFrame({"Factor": [FACTOR_LABELS[k] for k in FACTORS],
                        "Contribution": [contrib.get(k, 0.0) for k in FACTORS]})
    uc.hbar(cdf, x="Contribution", y="Factor", cap=8.0, title="Contribution to composite", fmt="+.1f")
    fs = row.get("factor_scores") or {}
    st.caption(" · ".join(f"{FACTOR_LABELS[k]}: {fs.get(k, 0):+.2f}" for k in FACTORS))

    st.markdown(section("Why this score", 1), unsafe_allow_html=True)
    for line in _explain(row):
        st.markdown(f"- {line}")

    stocks = detail.get("stocks", {})
    d = stocks.get(ticker, {})
    speed, strength = d.get("speed") or {}, d.get("strength") or {}

    st.markdown(section("Trend structure (GMMA)", 5,
                        help="Fetched live for this one ticker (not part of the nightly committed "
                             "artifact) — falls back to the committed moving-average snapshot if the "
                             "live fetch is unavailable."), unsafe_allow_html=True)
    _gmma_chart(ticker, speed)

    st.markdown(section("Breakout levels", 2), unsafe_allow_html=True)
    bgrid = row.get("breakout_grid") or {}
    bdf = pd.DataFrame([
        {"Horizon": HORIZON_LABELS[h], "State": (bgrid.get(h) or {}).get("state", "n/a"),
         "Trailing High": (bgrid.get(h) or {}).get("high"), "Trailing Low": (bgrid.get(h) or {}).get("low"),
         "Days Since Break": (bgrid.get(h) or {}).get("days_since_break"),
         "Confirmed": (bgrid.get(h) or {}).get("confirmed")}
        for h in HORIZONS
    ])
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.markdown(section("Historical composite score", 4,
                        help=SURVIVORSHIP_NOTE), unsafe_allow_html=True)
    _history_chart(ticker)


def _gmma_chart(ticker: str, speed_detail: dict) -> None:
    from ..cas.sources import prices
    from . import factors as mom_factors

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch(t: str):
        px, st_ = prices.get_history([t], period="2y", max_age_hours=6.0)
        df = px.get(t)
        return df if df is not None and not df.empty else None

    df = None
    try:
        df = _fetch(ticker)
    except Exception:
        df = None

    if df is None or len(df) < 60:
        ma_vals = (speed_detail or {}).get("ma_values", {})
        if not ma_vals:
            st.caption("Live price fetch unavailable and no committed moving-average snapshot yet.")
            return
        st.caption("Live fetch unavailable — showing the committed moving-average snapshot only.")
        mdf = pd.DataFrame({"Period": list(ma_vals.keys()), "MA": list(ma_vals.values())})
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        return

    periods = mom_factors.MA_PERIODS
    mas = mom_factors.moving_averages(df["close"], periods)
    chart_df = pd.DataFrame({"date": df.index, "price": df["close"].values})
    for p in periods:
        chart_df[f"MA{p}"] = mas[p].values
    tail = chart_df.tail(400)
    show = st.multiselect("Show averages", [f"MA{p}" for p in periods],
                          default=[f"MA{p}" for p in (21, 50, 200)], key=f"mom_gmma_{ticker}")

    def build(alt):
        base = alt.Chart(tail).encode(x=alt.X("date:T", title=None))
        layers = [base.mark_line(color=THEME.text, strokeWidth=1.5).encode(
            y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False)))]
        palette = [THEME.teal, THEME.mustard, THEME.coral, THEME.mauve, THEME.navy, THEME.mint, THEME.orange]
        for i, p in enumerate(periods):
            col = f"MA{p}"
            if col in show:
                layers.append(base.mark_line(color=palette[i % len(palette)], strokeWidth=1.2).encode(
                    y=alt.Y(f"{col}:Q", title="Price", scale=alt.Scale(zero=False))))
        return alt.layer(*layers).properties(height=340)

    uc.render_chart(build, fallback=tail[["date", "price"]])


def _history_chart(ticker: str) -> None:
    series = mom_history.series_for(ticker)
    if not series:
        st.caption("No historical score yet — accrues daily once the Action has run a few times, "
                  "or after a one-time `--action backfill`.")
        return
    hdf = pd.DataFrame(series)

    def build(alt):
        rule_df = pd.DataFrame({"y": [t for t, _ in MOM_STATES]})
        rules = alt.Chart(rule_df).mark_rule(color=THEME.grid, strokeDash=[2, 2]).encode(y="y:Q")
        boundary = alt.Chart(pd.DataFrame({"x": [MOM_MEMBERSHIP_START]})).mark_rule(
            color=THEME.mustard, strokeDash=[4, 3], strokeWidth=2).encode(x="x:T")
        line = alt.Chart(hdf).mark_line(color=THEME.teal, point=True).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("composite:Q", title="Composite score", scale=alt.Scale(domain=[-20, 20])),
            tooltip=["date", alt.Tooltip("composite:Q", format="+.1f"), "state"],
        )
        return (rules + line + boundary).properties(height=280)

    uc.render_chart(build, fallback=hdf[["date", "composite"]])
    st.caption("Mustard dashed line = point-in-time Russell 1000 membership tracking begins "
              f"({MOM_MEMBERSHIP_START}). Scores before it reuse today's constituents and are "
              "survivorship-biased.")

    full_rows = [r for r in series if r.get("factor_scores")]
    if full_rows:
        st.markdown(section("Historical factor breakdown (weekly)", 3), unsafe_allow_html=True)
        fdf = pd.DataFrame([{"Date": r["date"], **{FACTOR_LABELS[k]: (r["factor_scores"] or {}).get(k)
                                                   for k in FACTORS}, "Composite": r["composite"]}
                            for r in full_rows])
        st.dataframe(fdf, use_container_width=True, hide_index=True, height=min(400, 45 + 34 * len(fdf)))

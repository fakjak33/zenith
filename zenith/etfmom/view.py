"""ETF MOMENTUM tab — the momentum engine over Zenith's full ETF universe.

Reads only committed JSON (data/etfmom/*), with one deliberate exception: the
individual-ETF GMMA chart fetches that ONE ticker's price history on demand
(cached), because a 500-day x 7-MA panel for ~900 funds would be a multi-MB
daily rewrite.

Five sub-views on one radio, mirroring the MOMENTUM tab so the two read the
same way: Overview (breadth + heatmap) -> Rankings (filterable table) ->
Factors (one panel each + the redundancy check) -> Categories (where momentum
is concentrated, by asset class and Morningstar category) -> ETF (drill-down).

EVERY widget key here is prefixed `etfmom_`. Streamlit keys are session-global
and both momentum tabs render on every rerun, so an unprefixed key would
collide with the MOMENTUM tab's identically-named widget and the two would
silently drive each other.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME, MOM_WEIGHTS, MOM_HORIZON_WEIGHTS, MOM_STATES
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import (DISCLAIMER, UNIVERSE_NOTE, XSEC_CAVEAT, FACTORS, FACTOR_LABELS,
               HORIZONS, HORIZON_LABELS, load)
from . import history as etf_history

SUBVIEWS = ["Overview", "Rankings", "Factors", "Categories", "ETF"]

COLS = {
    "Rank": "Position in the priced ETF universe, ranked by composite momentum score.",
    "Ticker": "Exchange ticker.",
    "Fund": "Fund name.",
    "Asset Class": "Coarse asset class, rolled up mechanically from the Morningstar category.",
    "Category": "Morningstar category, normalized (the 'US Fund ' prefix is stripped so the "
                "catalog and the metadata cache agree on one spelling).",
    "Score": "Composite momentum score: -20 (extreme bearish) to +20 (extreme bullish). Same "
             "scale, same weights and same factors as the MOMENTUM tab, so the two are directly "
             "comparable.",
    "State": "Categorical read of the composite score (see the methodology expander for bands).",
    "TS": "Time-series momentum: this fund's own trailing return, vol-adjusted, across 6 horizons.",
    "Breakout": "Has the daily close broken its own trailing high/low, blended across 6 horizons.",
    "XSec": "Cross-sectional momentum: this fund's rank against every other ETF here. Read it "
            "within an asset class — see the caveat above the table.",
    "Speed": "Trend speed: how fast the moving-average (GMMA) structure is changing.",
    "Strength": "Momentum strength: are MA slopes rising/accelerating and is the trend smooth.",
    "MVT": "Multivariate trend: relative strength against peers once common factors are removed. "
           "Read from the MOMENTUM engine's nightly ETF pass.",
    "ADV $": "Trailing 21-day average daily dollar volume. The liquidity measure with full "
             "coverage — AUM and expense ratio are only known for the catalogued subset.",
    "AUM $m": "Fund assets, from the committed Morningstar catalog (574 of 912 names; blank "
              "otherwise — nothing is imputed).",
    "ER": "Expense ratio, same source and same partial coverage as AUM.",
    "Factors": "How many of the six factors went into this row's composite. '5' means the "
               "multivariate-trend factor was unavailable for this fund and the other five "
               "weights were renormalized.",
}

_EVIDENCE_NOTE = (
    "The factor math is MOMENTUM's, unchanged, and the time-series leg is if anything BETTER "
    "evidenced across asset classes than within the equity cross-section: Moskowitz, Ooi & "
    "Pedersen (2012) test time-series momentum on 58 futures markets spanning equities, bonds, "
    "currencies and commodities, and Asness, Moskowitz & Pedersen (2013) find momentum "
    "essentially everywhere. Two things hold this a notch below MOMENTUM's B+. First, the "
    "cross-sectional leg pools instruments that are not comparable — ranking a Treasury fund "
    "against a single-country equity fund is not the experiment the cross-sectional literature "
    "ran. Second, these funds are not independent bets: dozens track near-identical indices, so "
    "a breadth reading here is a softer statement than the same number across the Russell 1000. "
    "Promotion to B+ requires this engine's OWN out-of-sample information coefficient, measured "
    "against the equal-weight ETF universe rather than SPY — see the Factors sub-view, which "
    "fills in as picks accrue."
)

_FINDINGS = [
    {"stat": "Time-series momentum earns significant excess returns across 58 futures markets "
             "spanning four asset classes.",
     "cite": "Moskowitz, Ooi & Pedersen (2012)"},
    {"stat": "Momentum and value premia appear jointly across eight markets and asset classes.",
     "cite": "Asness, Moskowitz & Pedersen (2013)"},
    {"stat": "Scaling by a strategy's own realized volatility materially raises momentum's Sharpe.",
     "cite": "Barroso & Santa-Clara (2015)"},
    {"stat": "Momentum crashes are partly forecastable — they cluster after market declines, "
             "in high-volatility states.",
     "cite": "Daniel & Moskowitz (2016)"},
]


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    """Streamlit runs EVERY tab body on EVERY rerun, so an uncached read here
    would be paid on every click anywhere in the app."""
    return {"scores": load("scores", {}), "detail": load("detail", {}),
            "categories": load("categories", {}), "diagnostics": load("diagnostics", {}),
            "status": load("status", {})}


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
        # Leading asset class by mean composite -- the allocation-level read
        # that is the whole reason this tab exists alongside the stock one.
        by_class: dict[str, list[float]] = {}
        for r in rows:
            by_class.setdefault(r.get("asset_class") or "Unknown", []).append(r["composite"])
        ranked = sorted(((k, sum(v) / len(v)) for k, v in by_class.items() if len(v) >= 5),
                        key=lambda kv: -kv[1])
        lead = f" · strongest: {ranked[0][0]} ({ranked[0][1]:+.1f})" if ranked else ""
        color = THEME.teal if n_bull >= n_bear else THEME.coral
        return uc.chip(f"ETF MOMENTUM — {n_bull} strong-bullish / {n_bear} strong-bearish "
                       f"across {len(rows)} ETFs{lead}", color=color, sub="see ETF MOMENTUM tab")
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


def _mvt_banner(scores: dict, status: dict) -> None:
    """The 6th factor comes from a DIFFERENT nightly workflow (mom.yml). If it
    is missing or stale the composites are still valid — engine.composite
    renormalizes the other five weights — but the user must be told, because a
    five-factor composite plotted on the same axis as a six-factor one is a
    real discontinuity, not a rounding difference."""
    mvt = scores.get("mvt") or {}
    stale = mvt.get("mvt_stale_days")
    err = mvt.get("error")
    if err:
        st.markdown(uc.state_banner(
            THEME.coral, "MULTIVARIATE TREND UNAVAILABLE",
            f"The 6th factor could not be read this run ({err}). Every composite below is a "
            "five-factor score with the remaining weights renormalized — internally consistent, "
            "but not directly comparable to a six-factor run or to the MOMENTUM tab."),
            unsafe_allow_html=True)
    elif stale:
        st.markdown(uc.state_banner(
            THEME.mustard, "MULTIVARIATE TREND IS STALE",
            f"The 6th factor is {stale} day(s) old (computed {mvt.get('mvt_as_of')} by the "
            "MOMENTUM job). The other five factors are current."), unsafe_allow_html=True)


# -------------------------------------------------------------------- main --
def render() -> None:
    st.caption(DISCLAIMER)
    status = load("status", {})
    art = _artefacts(cache_bust=str(status.get("date", "")))
    scores = art["scores"]
    rows = scores.get("rows", [])

    st.markdown(evidence_rating("B", "documented across asset classes; this mixed-asset "
                                     "cross-section is unproven", _EVIDENCE_NOTE),
                unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    if not rows:
        st.markdown(stamp("—", "ETF MOMENTUM"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.etfmom.compute --action auto` to populate "
                "the ETF momentum scores.")
        return

    st.markdown(stamp(scores.get("as_of", "—"), "ETF MOMENTUM"), unsafe_allow_html=True)
    _mvt_banner(scores, status)

    with st.expander("Methodology — the same six factors, a different universe"):
        weights_txt = " · ".join(f"{FACTOR_LABELS[k]} {v:.0%}" for k, v in MOM_WEIGHTS.items())
        horizons_txt = " · ".join(f"{HORIZON_LABELS[h]} {w:.0%}"
                                  for h, w in MOM_HORIZON_WEIGHTS.items())
        st.markdown(
            "This tab runs **the identical engine** as MOMENTUM — the factor and composite code "
            "is imported from that package, not reimplemented — over the ETF universe instead of "
            "the Russell 1000. An ETF's `+12.4` and a stock's `+12.4` mean the same thing.\n\n"
            "Every factor is normalized to **[-1, +1]**; the composite is "
            "`20 x sum(weight_i x factor_i)`, so it always lands in **[-20, +20]**.\n\n"
            f"**Factor weights:** {weights_txt} — reused verbatim from MOMENTUM. Forking them "
            "would make the two tabs quietly non-comparable, which is the main thing having both "
            "is for.\n\n"
            f"**Horizon weights** (shared by time-series/breakout/cross-sectional): {horizons_txt} "
            "— tilted toward the slower legs; the 1-month leg is kept small because very "
            "short-horizon returns are known to reverse, not persist (Jegadeesh 1990).\n\n"
            "**The 6th factor is read, not recomputed.** The MOMENTUM job already scores this same "
            "ETF universe with its pairwise/residual-momentum engine every night, so this tab "
            "reads that artefact. Where that engine collapses near-identical funds into one "
            "(SPY/VOO/IVV), the surviving fund's score is **inherited** by its cluster and "
            "labelled as such — those funds are excluded there precisely because they correlate "
            "at 0.99 or above. Anything still without a score gets a five-factor composite with "
            "the remaining weights renormalized, never an imputed neutral zero.\n\n"
            f"**Cross-sectional caveat.** {XSEC_CAVEAT}\n\n"
            f"**Signal states:** {' · '.join(f'{t:+.0f} {l}' for t, l in MOM_STATES)}.\n\n"
            f"**Universe.** {UNIVERSE_NOTE}\n\n"
            "**Prices** are yfinance daily closes, split- AND distribution-adjusted — the only "
            "self-consistent choice for both return math and trailing-high/low breakout levels.\n\n"
            "The full pairwise relative-strength matrix for these same ETFs lives on the "
            "**MOMENTUM** tab, under *Multivariate Trend → ETFs* — it is not duplicated here."
        )

    sub = st.radio("View", SUBVIEWS, horizontal=True, label_visibility="collapsed",
                   key="etfmom_sub")
    if sub == "Overview":
        _overview(scores)
    elif sub == "Rankings":
        _rankings(scores)
    elif sub == "Factors":
        _factors(scores, art["diagnostics"])
    elif sub == "Categories":
        _categories(art["categories"])
    else:
        _etf(scores, art["detail"])


# --------------------------------------------------------------- overview --
def _overview(scores: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored funds yet.")
        return
    n = len(df)
    med = float(df["composite"].median())
    pct_bull = float((df["composite"] >= 5).mean())
    n_ext_bull = int((df["composite"] >= 15).sum())
    n_ext_bear = int((df["composite"] <= -15).sum())
    n_six = int((df["n_factors"] == 6).sum()) if "n_factors" in df else 0
    breadth_color = THEME.teal if pct_bull >= 0.5 else THEME.coral
    st.markdown(uc.state_banner(breadth_color, "BREADTH",
                                f"{pct_bull:.0%} of the priced ETF universe is bullish "
                                f"(score >= +5) — median score {med:+.1f}"), unsafe_allow_html=True)
    st.caption("Breadth here counts funds, not independent bets: near-duplicate funds are "
               "deliberately kept so you can find the one you hold, and a crowded category "
               "occupies more of the cross-section than a thin one.")
    st.markdown(uc.numeric_slab([
        {"label": "Universe (scored)", "value": str(n), "color": THEME.text},
        {"label": "Median score", "value": f"{med:+.1f}", "color": THEME.text},
        {"label": "% Bullish (>=+5)", "value": f"{pct_bull:.0%}", "color": THEME.teal},
        {"label": "Extreme bullish (>=+15)", "value": str(n_ext_bull), "color": THEME.teal},
        {"label": "Extreme bearish (<=-15)", "value": str(n_ext_bear), "color": THEME.coral},
        {"label": "Six-factor rows", "value": f"{n_six} / {n}", "color": THEME.muted,
         "sub": "the rest run on five, renormalized"},
    ]), unsafe_allow_html=True)

    excluded_n = scores.get("n", 0) - scores.get("n_scored", 0)
    if excluded_n:
        st.caption(f"{excluded_n} funds in the universe are not scored — insufficient price "
                   "history (a fund younger than ~1.8 years cannot have a 400-day moving-average "
                   "slope), no price data returned this run, or caught by the leveraged/inverse "
                   "backstop. They stay in the artefact with a stated reason rather than being "
                   "dropped.")

    st.markdown(section("Asset-class momentum — the allocation read", 2,
                        help="Mean composite by asset class. This is the question an ETF momentum "
                             "engine can answer that a stock one cannot."), unsafe_allow_html=True)
    ac = (df.groupby("asset_class")["composite"].agg(["mean", "count"]).reset_index()
          .rename(columns={"mean": "v"}).sort_values("v", ascending=False))
    ac = ac[ac["count"] >= 3]
    if not ac.empty:
        uc.hbar(ac, x="v", y="asset_class", cap=10.0,
                title="Mean composite score by asset class", fmt="+.2f")

    st.markdown(section("ETF universe — momentum heatmap", 4,
                        help="Every scored fund, sorted by composite score. Click a cell to open "
                             "its full breakdown on the ETF tab."), unsafe_allow_html=True)
    n_cols = st.slider("Heatmap columns", 20, 60, 40, key="etfmom_hm_cols",
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
            tooltip=["ticker", "name", "asset_class", "category",
                     alt.Tooltip("composite:Q", format="+.1f", title="Composite"),
                     alt.Tooltip("rank:Q", title="Rank"), "state"],
        ).add_params(pick).properties(height=max(240, 16 * (grid["row"].max() + 1))))

    event = uc.render_chart(build, fallback=grid[["ticker", "composite", "asset_class"]],
                            on_select="rerun", key="etfmom_heatmap")
    picked = _picked_ticker(event)
    if picked:
        st.session_state["etfmom_sub"] = "ETF"
        st.session_state["etfmom_pick"] = picked
        st.rerun()

    st.markdown(section("Score distribution", 1), unsafe_allow_html=True)
    hist_df = df[["composite"]].copy()

    def build_hist(alt):
        return (alt.Chart(hist_df).mark_bar().encode(
            x=alt.X("composite:Q", bin=alt.Bin(maxbins=40), title="Composite score"),
            y=alt.Y("count():Q", title="ETFs"),
            color=alt.Color("composite:Q", scale=uc.diverging_scale(20, alt), legend=None,
                            bin=alt.Bin(maxbins=40)),
        ).properties(height=200))

    uc.render_chart(build_hist, fallback=hist_df)


# --------------------------------------------------------------- rankings --
def _rankings(scores: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored funds yet.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("Search ticker or fund name", "", key="etfmom_rank_q")
    classes = sorted(df["asset_class"].dropna().unique().tolist())
    class_pick = c2.multiselect("Asset class", classes, default=[], key="etfmom_rank_class")
    preset = c3.selectbox("Preset", ["All", "Top 10", "Top 25", "Top 50",
                                     "Bottom 10", "Bottom 25", "Bottom 50"],
                          key="etfmom_rank_preset")

    with st.expander("More filters"):
        f1, f2, f3 = st.columns(3)
        score_min, score_max = f1.slider("Composite score", -20.0, 20.0, (-20.0, 20.0),
                                         key="etfmom_rank_score")
        factor_pick = f2.selectbox("Factor filter", ["(none)"] + list(FACTORS),
                                   format_func=lambda k: "(none)" if k == "(none)" else FACTOR_LABELS[k],
                                   key="etfmom_rank_factor")
        factor_min = f3.slider("Min factor score", -1.0, 1.0, -1.0, key="etfmom_rank_factor_min") \
            if factor_pick != "(none)" else None
        # Liquidity matters far more here than in a large-cap stock screen --
        # the tail of this universe is genuinely thin. Display-only: it never
        # changes a score or a percentile, only what the table shows.
        adv_series = pd.to_numeric(df.get("adv_dollar"), errors="coerce")
        adv_max = float(adv_series.max()) if adv_series.notna().any() else 0.0
        min_adv = st.slider("Minimum average daily dollar volume ($m)", 0.0,
                            max(1.0, round(adv_max / 1e6, 1)), 0.0, key="etfmom_rank_adv",
                            help="Display filter only — it does not change any score or rank.")
        cats = sorted(df["category"].dropna().unique().tolist())
        cat_pick = st.multiselect("Morningstar category", cats, default=[],
                                  key="etfmom_rank_cat")
        breakout_only = st.checkbox("Only funds with a fresh breakout (any horizon, confirmed)",
                                    key="etfmom_rank_breakout")

    view = df[(df["composite"] >= score_min) & (df["composite"] <= score_max)].copy()
    if class_pick:
        view = view[view["asset_class"].isin(class_pick)]
    if cat_pick:
        view = view[view["category"].isin(cat_pick)]
    if q.strip():
        ql = q.strip().lower()
        view = view[view["ticker"].str.lower().str.contains(ql)
                    | view["name"].str.lower().str.contains(ql)]
    if factor_pick != "(none)" and factor_min is not None:
        view = view[view["factor_scores"].apply(
            lambda d: (d or {}).get(factor_pick, -1.0) >= factor_min)]
    if min_adv > 0:
        view = view[pd.to_numeric(view["adv_dollar"], errors="coerce").fillna(0.0) >= min_adv * 1e6]
    if breakout_only:
        def _has_break(grid):
            return any((g or {}).get("confirmed")
                       and (g or {}).get("state") in ("break_up", "break_down")
                       for g in (grid or {}).values())
        view = view[view["breakout_grid"].apply(_has_break)]

    view = view.sort_values("composite", ascending=False)
    if preset.startswith("Top"):
        view = view.head(int(preset.split()[1]))
    elif preset.startswith("Bottom"):
        view = view.sort_values("composite", ascending=True).head(int(preset.split()[1]))
        view = view.sort_values("composite", ascending=False)

    st.caption(f"{len(view)} of {len(df)} scored funds")
    st.caption(XSEC_CAVEAT)

    disp = pd.DataFrame({
        "Rank": view["rank"], "Ticker": view["ticker"], "Fund": view["name"],
        "Asset Class": view["asset_class"], "Category": view["category"],
        "Score": view["composite"], "State": view["state"],
        "TS": view["factor_scores"].apply(lambda d: (d or {}).get("ts")),
        "Breakout": view["factor_scores"].apply(lambda d: (d or {}).get("breakout")),
        "XSec": view["factor_scores"].apply(lambda d: (d or {}).get("xsec")),
        "Speed": view["factor_scores"].apply(lambda d: (d or {}).get("speed")),
        "Strength": view["factor_scores"].apply(lambda d: (d or {}).get("strength")),
        "MVT": view["factor_scores"].apply(lambda d: (d or {}).get("mvt")),
        "ADV $": view["adv_dollar"], "AUM $m": view["aum_m"], "ER": view["er"],
        # The honesty column: without it the table silently mixes six-factor
        # and five-factor composites as if they were the same measurement.
        "Factors": view["n_factors"],
    })
    fcols = ["TS", "Breakout", "XSec", "Speed", "Strength", "MVT"]
    try:
        sty = (disp.style
               .map(lambda v: uc.grad_diverging(v, 20.0), subset=["Score"])
               .map(lambda v: uc.grad_diverging(v, 1.0), subset=fcols)
               .format({"Score": "{:+.1f}", **{c: "{:+.2f}" for c in fcols},
                        "ADV $": lambda v: uc.fmt_money(v),
                        "AUM $m": lambda v: "—" if pd.isna(v) else f"{v:,.0f}",
                        "ER": lambda v: "—" if pd.isna(v) else f"{v:.2f}%",
                        "Factors": lambda v: "—" if pd.isna(v) else ("6" if v == 6 else f"{int(v)} (no mvt)")}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(600, 45 + 34 * len(disp)),
                     column_config=uc.colcfg(disp.columns, COLS), key="etfmom_rank_table")
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown(section("Leaderboards", 3), unsafe_allow_html=True)
    lead_factor = st.selectbox("Leaderboard factor", ["Composite"] + list(FACTORS),
                               format_func=lambda k: "Composite" if k == "Composite" else FACTOR_LABELS[k],
                               key="etfmom_lead_factor")
    if lead_factor == "Composite":
        top = df.nlargest(15, "composite")[["ticker", "composite"]].rename(columns={"composite": "v"})
        cap, fmt = 20.0, "+.1f"
    else:
        vv = df["factor_scores"].apply(lambda d: (d or {}).get(lead_factor))
        top = df.assign(v=vv).nlargest(15, "v")[["ticker", "v"]]
        cap, fmt = 1.0, "+.2f"
    label = "Composite" if lead_factor == "Composite" else FACTOR_LABELS[lead_factor]
    uc.hbar(top, x="v", y="ticker", cap=cap, title=f"Top 15 — {label}", fmt=fmt)


# ----------------------------------------------------------------- factors --
def _factors(scores: dict, diagnostics: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored funds yet.")
        return
    pick = st.radio("Factor", list(FACTORS), horizontal=True, label_visibility="collapsed",
                    format_func=lambda k: FACTOR_LABELS[k], key="etfmom_factor_pick")
    df = df.assign(_fv=df["factor_scores"].apply(lambda d: (d or {}).get(pick)))
    st.caption(FACTOR_LABELS[pick])
    if pick == "xsec":
        st.caption(XSEC_CAVEAT)

    grid = df.dropna(subset=["_fv"]).sort_values("_fv", ascending=False).reset_index(drop=True)
    if grid.empty:
        st.info(f"No fund carries a {FACTOR_LABELS[pick]} score this run.")
        return
    n_cols = 40
    grid["col"] = grid.index % n_cols
    grid["row"] = grid.index // n_cols

    def build(alt):
        return (alt.Chart(grid).mark_rect(stroke=THEME.bg, strokeWidth=0.5).encode(
            x=alt.X("col:O", title=None, axis=None),
            y=alt.Y("row:O", title=None, axis=None, sort="descending"),
            color=alt.Color("_fv:Q", title=FACTOR_LABELS[pick],
                            scale=uc.diverging_scale(1.0, alt), legend=alt.Legend(labelLimit=0)),
            tooltip=["ticker", "name", "asset_class",
                     alt.Tooltip("_fv:Q", format="+.2f", title=FACTOR_LABELS[pick]),
                     alt.Tooltip("composite:Q", format="+.1f", title="Composite")],
        ).properties(height=max(200, 16 * (grid["row"].max() + 1))))
    uc.render_chart(build, fallback=grid[["ticker", "_fv"]])

    top = grid.nlargest(15, "_fv")[["ticker", "_fv"]]
    uc.hbar(top, x="_fv", y="ticker", cap=1.0, title=f"Top 15 — {FACTOR_LABELS[pick]}", fmt="+.2f")

    if pick in ("ts", "breakout", "xsec"):
        st.markdown(section(f"{FACTOR_LABELS[pick]} — horizon grid", 2), unsafe_allow_html=True)
        top25 = grid.head(25)
        val_key = "ts_signal" if pick == "ts" else ("xsec_signal" if pick == "xsec" else "b")
        src_col = "breakout_grid" if pick == "breakout" else "ts_grid"
        cols = {HORIZON_LABELS[h]: top25[src_col].apply(
            lambda g, h=h: (g or {}).get(h, {}).get(val_key)) for h in HORIZONS}
        hgrid = pd.DataFrame({"Ticker": top25["ticker"].values,
                              **{k: v.values for k, v in cols.items()}})
        try:
            sty = hgrid.style
            for c in hgrid.columns[1:]:
                sty = sty.map(lambda v: uc.grad_diverging(v, 1.0), subset=[c])
            sty = sty.format({c: "{:+.2f}" for c in hgrid.columns[1:]})
            st.dataframe(sty, use_container_width=True, hide_index=True,
                         height=min(430, 45 + 34 * len(hgrid)))
        except Exception:
            st.dataframe(hgrid, use_container_width=True, hide_index=True)

    st.markdown(section("Factor correlation — checking the redundancy assumption", 5,
                        help="Spearman correlation of the six factor scores across the priced ETF "
                             "universe. Pairs above 0.85 are flagged. This is the check on the "
                             "weighting assumption, not an assertion of it — and it is worth "
                             "reading separately from MOMENTUM's, because the ETF cross-section "
                             "may well have a different redundancy structure than the R1000 did."),
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
            st.markdown(uc.state_banner(THEME.mustard, "REDUNDANCY FLAG", msg),
                        unsafe_allow_html=True)
        else:
            st.caption("No factor pair currently exceeds the 0.85 redundancy threshold.")
    else:
        st.caption("Correlation matrix not yet available (needs at least 10 scored funds).")

    # mvt coverage is a real caveat on the matrix above, not a footnote: an
    # inherited score is REPEATED across every member of a near-duplicate
    # cluster, which mildly inflates mvt's apparent cross-sectional agreement
    # with the other factors.
    mvt = scores.get("mvt") or {}
    n_with = int(df["factor_scores"].apply(lambda d: "mvt" in (d or {})).sum())
    n_inherited = int(df["mvt_source"].notna().sum()) if "mvt_source" in df else 0
    st.caption(
        f"Multivariate trend present on {n_with} of {len(df)} scored funds "
        f"({mvt.get('direct', 0)} measured directly, {n_inherited} inherited from a fund it "
        "correlates with at 0.99 or above). Inherited scores repeat across cluster members, "
        "which mildly inflates this factor's apparent correlation with the others — read the "
        "matrix with that in mind.")

    st.markdown(section("Predictive diagnostics — information coefficient", 4,
                        help="Spearman rank correlation between the composite (and each factor) at "
                             "pick time and the realized excess return at each horizon. Excess is "
                             "measured against an EQUAL-WEIGHT INDEX OF THIS ETF UNIVERSE, not "
                             "SPY: against SPY a long Treasury-fund pick would fail through every "
                             "bull market regardless of signal skill, and the IC would be "
                             "measuring asset-class beta rather than the engine."),
                unsafe_allow_html=True)
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


# -------------------------------------------------------------- categories --
def _categories(categories_doc: dict) -> None:
    """The ETF analogue of MOMENTUM's Sectors view. Two levels: a coarse asset
    class rolled up mechanically from Morningstar's own category vocabulary,
    then the categories themselves."""
    classes = categories_doc.get("asset_classes", {})
    cats = categories_doc.get("categories", {})
    if not classes:
        st.info("No category data yet.")
        return

    st.markdown(section("Asset-class momentum", 2,
                        help="Mean/median composite, breadth and dispersion per asset class. The "
                             "asset class is rolled up from the Morningstar category by a fixed "
                             "documented rule list — see etfmom/universe.py."),
                unsafe_allow_html=True)
    cdf = pd.DataFrame(classes).T.reset_index().rename(columns={"index": "Asset Class"})
    cdf = cdf.sort_values("mean", ascending=False)
    disp = pd.DataFrame({
        "Asset Class": cdf["Asset Class"], "N": cdf["n"], "Mean": cdf["mean"],
        "Median": cdf["median"], "% Bullish": cdf["pct_bullish"], "% Bearish": cdf["pct_bearish"],
        "Extreme Bull": cdf["n_extreme_bullish"], "Extreme Bear": cdf["n_extreme_bearish"],
        "Dispersion": cdf["dispersion"],
    })
    try:
        sty = (disp.style.map(lambda v: uc.grad_diverging(v, 10.0), subset=["Mean", "Median"])
               .format({"Mean": "{:+.2f}", "Median": "{:+.2f}", "% Bullish": "{:.0%}",
                        "% Bearish": "{:.0%}", "Dispersion": "{:.2f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(430, 45 + 34 * len(disp)))
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    uc.hbar(pd.DataFrame({"Asset Class": cdf["Asset Class"], "v": cdf["mean"]}),
            x="v", y="Asset Class", cap=10.0, title="Mean composite score by asset class",
            fmt="+.2f")

    st.markdown(section("Category momentum", 5,
                        help="Morningstar categories, normalized. Within an asset class this is "
                             "the like-for-like comparison the universe-wide cross-sectional "
                             "factor cannot give you."), unsafe_allow_html=True)
    idf = pd.DataFrame(cats).T.reset_index().rename(columns={"index": "Category"})
    if idf.empty:
        return
    idf = idf.sort_values("mean", ascending=False)
    min_n = st.slider("Minimum funds per category", 1, 20, 3, key="etfmom_cat_min_n",
                      help="Thin categories swing wildly on one fund; raise this to see only "
                           "categories with enough members to mean something.")
    shown = idf[idf["n"] >= min_n]
    if shown.empty:
        st.caption("No category meets that minimum.")
        return
    c1, c2 = st.columns(2)
    with c1:
        uc.hbar(shown.head(15)[["Category", "mean"]].rename(columns={"mean": "v"}),
                x="v", y="Category", cap=10.0, title="Strongest categories", fmt="+.2f")
    with c2:
        uc.hbar(shown.tail(15)[["Category", "mean"]].rename(columns={"mean": "v"}),
                x="v", y="Category", cap=10.0, title="Weakest categories", fmt="+.2f")

    cdisp = pd.DataFrame({
        "Category": shown["Category"], "N": shown["n"], "Mean": shown["mean"],
        "Median": shown["median"], "% Bullish": shown["pct_bullish"],
        "% Bearish": shown["pct_bearish"], "Dispersion": shown["dispersion"],
    })
    try:
        sty = (cdisp.style.map(lambda v: uc.grad_diverging(v, 10.0), subset=["Mean", "Median"])
               .format({"Mean": "{:+.2f}", "Median": "{:+.2f}", "% Bullish": "{:.0%}",
                        "% Bearish": "{:.0%}", "Dispersion": "{:.2f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(500, 45 + 34 * len(cdisp)))
    except Exception:
        st.dataframe(cdisp, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------- etf --
def _explain(row: dict) -> list[str]:
    """Plain-English signal explanation built from stored numbers — every
    bullet traces to a field rendered elsewhere on the same page."""
    out = []
    grid = row.get("ts_grid") or {}
    best_h, best_dev = None, 0.0
    for h, g in grid.items():
        p = g.get("pctile")
        if p is not None and abs(p - 50) > best_dev:
            best_dev, best_h = abs(p - 50), h
    if best_h:
        p = grid[best_h]["pctile"]
        out.append(f"{HORIZON_LABELS[best_h]} return in the {p:.0f}th percentile of the ETF universe")
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
    if fs.get("speed") is not None and abs(fs["speed"]) > 0.4:
        out.append("Moving averages " + ("bullishly" if fs["speed"] > 0 else "bearishly")
                   + " aligned with fresh crossovers")
    if fs.get("strength") is not None:
        if fs["strength"] > 0.3:
            out.append("Trend strength accelerating with a statistically smooth path")
        elif fs["strength"] < -0.3:
            out.append("Trend deteriorating — slopes decelerating or reversing")
    if row.get("n_factors") == 5:
        out.append("Scored on five factors — multivariate trend was unavailable for this fund, "
                   "and the remaining weights were renormalized rather than filling in a zero")
    return out or ["No standout single signal — the composite reflects a blend of moderate readings."]


def _etf(scores: dict, detail: dict) -> None:
    df = _priced_df(scores)
    if df.empty:
        st.info("No scored funds yet.")
        return
    tickers = sorted(df["ticker"].tolist())
    sel_key = "etfmom_etf_pick"
    preselect = st.session_state.get("etfmom_pick")
    if preselect and preselect in tickers:
        st.session_state[sel_key] = preselect
        st.session_state.pop("etfmom_pick", None)
    if st.session_state.get(sel_key) not in tickers:
        st.session_state[sel_key] = tickers[0]
    ticker = st.selectbox("ETF", tickers, key=sel_key,
                          format_func=lambda t: f"{t} — {df.loc[df['ticker']==t,'name'].iloc[0]}")
    row = df[df["ticker"] == ticker].iloc[0].to_dict()

    comp, state = row["composite"], row["state"]
    st.markdown(uc.state_banner(_state_color(state), f"{ticker} — {row['name']}",
                                f"{comp:+.1f} · {state}"), unsafe_allow_html=True)

    aum = row.get("aum_m")
    er = row.get("er")
    st.markdown(uc.numeric_slab([
        {"label": "Composite", "value": f"{comp:+.1f}", "color": _state_color(state)},
        {"label": "Rank", "value": f"#{int(row['rank'])} / {len(df)}", "color": THEME.text},
        {"label": "Percentile", "value": f"{row.get('pctile', 0):.0f}th", "color": THEME.text},
        {"label": "Asset class", "value": row.get("asset_class") or "—", "color": THEME.muted},
        {"label": "Category", "value": row.get("category") or "—", "color": THEME.muted},
        {"label": "ADV $", "value": uc.fmt_money(row.get("adv_dollar")), "color": THEME.muted},
        {"label": "AUM / ER", "color": THEME.muted,
         "value": (f"{aum:,.0f}m" if aum is not None and pd.notna(aum) else "—")
                  + " / " + (f"{er:.2f}%" if er is not None and pd.notna(er) else "—"),
         "sub": "catalogued subset only"},
        {"label": "Equal-weight composite",
         "value": f"{row.get('composite_equal_weight', 0):+.1f}", "color": THEME.muted,
         "sub": "for comparison against the tilted weighting"},
    ]), unsafe_allow_html=True)

    st.markdown(section("Factor breakdown", 3,
                        help="Each factor's [-1,+1] score times its weight times 20 — these "
                             "numbers sum exactly to the composite above."), unsafe_allow_html=True)
    contrib = row.get("contributions") or {}
    cdf = pd.DataFrame({"Factor": [FACTOR_LABELS[k] for k in FACTORS],
                        "Contribution": [contrib.get(k, 0.0) for k in FACTORS]})
    uc.hbar(cdf, x="Contribution", y="Factor", cap=8.0, title="Contribution to composite",
            fmt="+.1f")
    fs = row.get("factor_scores") or {}
    st.caption(" · ".join(f"{FACTOR_LABELS[k]}: {fs[k]:+.2f}" if k in fs
                          else f"{FACTOR_LABELS[k]}: n/a" for k in FACTORS))

    # Provenance of the 6th factor, stated per row rather than assumed.
    src = row.get("mvt_source")
    if "mvt" not in fs:
        st.caption("**Multivariate trend: not available for this fund this run.** Its weight was "
                   "redistributed across the other five factors rather than filled in as a "
                   "neutral zero — so this composite is a five-factor score.")
    elif src and pd.notna(src):
        st.caption(f"**Multivariate trend inherited from {src}.** The pairwise engine excludes "
                   f"{ticker} as a near-duplicate of {src} (their daily returns correlate at 0.99 "
                   "or above), so the relative-strength geometry it measured for "
                   f"{src} is {ticker}'s by construction. It was not measured on {ticker} itself.")
    else:
        st.caption("Multivariate trend measured directly for this fund. The full pairwise matrix "
                   "is on the MOMENTUM tab under *Multivariate Trend → ETFs*.")

    st.markdown(section("Why this score", 1), unsafe_allow_html=True)
    for line in _explain(row):
        st.markdown(f"- {line}")

    d = (detail.get("etfs") or {}).get(ticker, {})

    st.markdown(section("Trend structure (GMMA)", 5,
                        help="Fetched live for this one ticker (not part of the nightly committed "
                             "artefact) — falls back to the committed moving-average snapshot if "
                             "the live fetch is unavailable."), unsafe_allow_html=True)
    _gmma_chart(ticker, d)

    st.markdown(section("Breakout levels", 2), unsafe_allow_html=True)
    bgrid = row.get("breakout_grid") or {}
    st.dataframe(pd.DataFrame([
        {"Horizon": HORIZON_LABELS[h], "State": (bgrid.get(h) or {}).get("state", "n/a"),
         "Trailing High": (bgrid.get(h) or {}).get("high"),
         "Trailing Low": (bgrid.get(h) or {}).get("low"),
         "Days Since Break": (bgrid.get(h) or {}).get("days_since_break"),
         "Confirmed": (bgrid.get(h) or {}).get("confirmed")}
        for h in HORIZONS
    ]), use_container_width=True, hide_index=True)

    st.markdown(section("Historical composite score", 4, help=UNIVERSE_NOTE),
                unsafe_allow_html=True)
    _history_chart(ticker)


def _gmma_chart(ticker: str, etf_detail: dict) -> None:
    from ..cas.sources import prices
    from ..mom import factors as mom_factors

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch(t: str):
        px, _ = prices.get_history([t], period="2y", max_age_hours=6.0)
        df = px.get(t)
        return df if df is not None and not df.empty else None

    try:
        df = _fetch(ticker)
    except Exception:
        df = None

    if df is None or len(df) < 60:
        ma_vals = (etf_detail or {}).get("ma_values") or {}
        if not ma_vals:
            st.caption("Live price fetch unavailable and no committed moving-average snapshot yet.")
            return
        st.caption("Live fetch unavailable — showing the committed moving-average snapshot only.")
        st.dataframe(pd.DataFrame({"Period": list(ma_vals.keys()), "MA": list(ma_vals.values())}),
                     use_container_width=True, hide_index=True)
        return

    periods = mom_factors.MA_PERIODS
    mas = mom_factors.moving_averages(df["close"], periods)
    chart_df = pd.DataFrame({"date": df.index, "price": df["close"].values})
    for p in periods:
        chart_df[f"MA{p}"] = mas[p].values
    tail = chart_df.tail(400)
    show = st.multiselect("Show averages", [f"MA{p}" for p in periods],
                          default=[f"MA{p}" for p in (21, 50, 200)],
                          key=f"etfmom_gmma_{ticker}")

    def build(alt):
        base = alt.Chart(tail).encode(x=alt.X("date:T", title=None))
        layers = [base.mark_line(color=THEME.text, strokeWidth=1.5).encode(
            y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False)))]
        palette = [THEME.teal, THEME.mustard, THEME.coral, THEME.mauve, THEME.navy,
                   THEME.mint, THEME.orange]
        for i, p in enumerate(periods):
            col = f"MA{p}"
            if col in show:
                layers.append(base.mark_line(color=palette[i % len(palette)],
                                             strokeWidth=1.2).encode(
                    y=alt.Y(f"{col}:Q", title="Price", scale=alt.Scale(zero=False))))
        return alt.layer(*layers).properties(height=340)

    uc.render_chart(build, fallback=tail[["date", "price"]])


def _history_chart(ticker: str) -> None:
    series = etf_history.series_for(ticker)
    if not series:
        st.caption("No historical score yet — this accrues daily once the Action has run a few "
                   "times. There is deliberately no backfill: the multivariate-trend factor has "
                   "no historical values, so a backfilled score would be a five-factor number "
                   "plotted on the same axis as today's six-factor one.")
        return
    hdf = pd.DataFrame(series)

    def build(alt):
        rules = alt.Chart(pd.DataFrame({"y": [t for t, _ in MOM_STATES]})).mark_rule(
            color=THEME.grid, strokeDash=[2, 2]).encode(y="y:Q")
        line = alt.Chart(hdf).mark_line(color=THEME.teal, point=True).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("composite:Q", title="Composite score", scale=alt.Scale(domain=[-20, 20])),
            tooltip=["date", alt.Tooltip("composite:Q", format="+.1f"), "state"],
        )
        return (rules + line).properties(height=280)

    uc.render_chart(build, fallback=hdf[["date", "composite"]])

    full_rows = [r for r in series if r.get("factor_scores")]
    if full_rows:
        st.markdown(section("Historical factor breakdown (weekly)", 3), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Date": r["date"], **{FACTOR_LABELS[k]: (r["factor_scores"] or {}).get(k)
                                   for k in FACTORS}, "Composite": r["composite"]}
            for r in full_rows]), use_container_width=True, hide_index=True,
            height=min(400, 45 + 34 * len(full_rows)))

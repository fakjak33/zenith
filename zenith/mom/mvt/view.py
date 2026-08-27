"""Multivariate Trend sub-tab: Equities / ETFs pairwise relative-strength
engine. Rendered from mom/view.py's SUBVIEWS radio as one more panel inside
the existing MOMENTUM tab -- everything else in mom/view.py is untouched.

Reads only the committed mvt artifacts (data/mom/mvt/*.json). The pairwise
matrix is reconstructed ON DEMAND from the per-ticker return/variance
vectors those artifacts carry (see pairwise.py's storage-architecture note)
-- never from a persisted NxN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ... import ui_charts as uc
from ...config import THEME
from ...ui_theme import evidence_rating, key_findings, section
from . import DISCLAIMER, HORIZONS, HORIZON_LABELS, load
from . import pairwise as pw

SUB_UNIVERSES = ["Equities", "ETFs"]
_KEY = {"Equities": "equities", "ETFs": "etfs"}

_EVIDENCE_NOTE = (
    "Residual (common-factor-removed) pairwise momentum is a published, replicated anomaly "
    "(Blitz, Huij & Martens 2011, \"Residual Momentum\"), but THIS composite -- this universe, "
    "this factor count, this horizon blend -- has no out-of-sample record of its own yet. "
    "Promotion to a higher tier requires this app's own validation diagnostics (planned), not a "
    "backtest tuned to look good. A synthetic-panel test run while building this measured the "
    "raw pairwise score at ~0.9+ Spearman-redundant with plain cross-sectional momentum, and the "
    "residual score at meaningfully less -- see the methodology panel below for the numbers."
)

_FINDINGS = [
    {"stat": "Residual (idiosyncratic) momentum persists after controlling for common risk factors.",
     "cite": "Blitz, Huij & Martens (2011)"},
    {"stat": "A naive pairwise relative-return spread is arithmetically dominated by each name's own "
             "return -- it does not, by itself, inject correlation structure into the signal.",
     "cite": "measured on a synthetic panel while building this feature"},
]


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    return {"equities": load("equities", {}), "etfs": load("etfs", {}),
           "status": load("status", {})}


def _rows_df(doc: dict) -> pd.DataFrame:
    rows = doc.get("rows", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _breadth_bands(scores: pd.Series) -> dict:
    return {
        "Strong positive (>=10)": int((scores >= 10).sum()),
        "Positive (2 to 10)": int(((scores >= 2) & (scores < 10)).sum()),
        "Neutral (-2 to 2)": int(((scores > -2) & (scores < 2)).sum()),
        "Negative (-10 to -2)": int(((scores <= -2) & (scores > -10)).sum()),
        "Strong negative (<=-10)": int((scores <= -10).sum()),
    }


def render() -> None:
    st.caption(DISCLAIMER)
    status = load("status", {})
    art = _artefacts(cache_bust=str(status.get("date", "")))

    st.markdown(evidence_rating("C+", "new, unproven -- promotion requires this app's own diagnostics",
                                _EVIDENCE_NOTE), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    sub = st.radio("Universe", SUB_UNIVERSES, horizontal=True, label_visibility="collapsed",
                   key="mvt_universe_sub")
    doc = art[_KEY[sub]]
    df = _rows_df(doc)

    with st.expander("How is this calculated?"):
        _methodology(doc)

    if df.empty:
        st.info(f"No Multivariate Trend data yet for {sub}. Run "
                "`python -m zenith.mom.compute --action auto` to populate it.")
        err = (doc.get("status") or {}).get("error")
        if err:
            st.caption(f"Last run status: {err}")
        return

    st.caption(f"{sub}: {len(df)} names scored · as of {doc.get('as_of', '—')} · "
              f"{doc.get('k_factors', '?')} statistical factors explain "
              f"{doc.get('explained_variance_ratio', 0):.0%} of common variance · "
              f"effective factor count {doc.get('effective_factor_count', '—')}")

    _universe_breadth(df)
    _leaders_laggards(df)
    _distribution(df)
    st.markdown(section("Relative-strength matrix", 3,
                        help="Pairwise spread for a chosen subset, reconstructed on demand from "
                             "the committed return/variance vectors -- never a full NxN dump."),
                unsafe_allow_html=True)
    _matrix(df)
    st.markdown(section("Instrument detail", 3), unsafe_allow_html=True)
    _instrument_detail(df)


# --------------------------------------------------------------- breadth --
def _universe_breadth(df: pd.DataFrame) -> None:
    st.markdown(section("Universe breadth", 3), unsafe_allow_html=True)
    bands = _breadth_bands(df["normalized_score"].dropna())
    items = [{"label": k, "value": str(v)} for k, v in bands.items()]
    st.markdown(uc.numeric_slab(items), unsafe_allow_html=True)


# ------------------------------------------------------- leaders/laggards --
def _consensus(consistency: dict | None) -> float | None:
    """Takes the `consistency` SUB-dict directly (this is called via
    `.apply()` on the consistency column itself, not on the whole row --
    a real bug here previously looked for a nested row["consistency"]
    inside the sub-dict, which never exists, silently returning None for
    every row and leaving the "timeframe consensus" leaderboards empty)."""
    return (consistency or {}).get("agreement")


def _leaders_laggards(df: pd.DataFrame) -> None:
    st.markdown(section("Leaders & laggards", 3), unsafe_allow_html=True)
    valid = df.dropna(subset=["normalized_score"])
    if valid.empty:
        return
    c1, c2 = st.columns(2)
    with c1:
        top = valid.nlargest(15, "normalized_score")[["ticker", "normalized_score"]].rename(
            columns={"normalized_score": "v"})
        uc.hbar(top, x="v", y="ticker", cap=20.0, title="Top 15 leaders", fmt="+.1f")
    with c2:
        bot = valid.nsmallest(15, "normalized_score")[["ticker", "normalized_score"]].rename(
            columns={"normalized_score": "v"})
        uc.hbar(bot, x="v", y="ticker", cap=20.0, title="Bottom 15 laggards", fmt="+.1f")

    # pd.to_numeric coerces the None/float mix _consensus() can return into a
    # clean float64 column -- an object-dtype column (what a raw Series of
    # None/float naturally is) makes nlargest/nsmallest raise a TypeError.
    consensus = valid.assign(
        agreement=pd.to_numeric(valid["consistency"].apply(_consensus), errors="coerce"))
    c3, c4 = st.columns(2)
    with c3:
        strong = consensus.dropna(subset=["agreement"]).nlargest(10, "agreement")[["ticker", "agreement"]]
        strong = strong.rename(columns={"agreement": "v"})
        uc.hbar(strong, x="v", y="ticker", cap=1.0, title="Strongest timeframe consensus", fmt=".0%")
    with c4:
        divergent = consensus.dropna(subset=["agreement"]).nsmallest(10, "agreement")[["ticker", "agreement"]]
        divergent = divergent.rename(columns={"agreement": "v"})
        uc.hbar(divergent, x="v", y="ticker", cap=1.0, title="Most divergent across timeframes", fmt=".0%")

    raw_vs_norm = valid.dropna(subset=["raw_score"]).assign(
        gap=lambda d: (d["raw_score"] - d["normalized_score"]).abs())
    if not raw_vs_norm.empty:
        st.markdown(uc.note_strip("Most independent trend", [
            f"{r.ticker}: raw {r.raw_score:+.1f} vs residual {r.normalized_score:+.1f} "
            f"(gap {r.gap:.1f}) -- its apparent trend is largely NOT shared with the broad move"
            for r in raw_vs_norm.nlargest(5, "gap").itertuples()
        ]), unsafe_allow_html=True)


# ---------------------------------------------------------- distribution --
def _distribution(df: pd.DataFrame) -> None:
    st.markdown(section("Score distribution", 3), unsafe_allow_html=True)
    valid = df.dropna(subset=["normalized_score", "raw_score"])
    if valid.empty:
        return

    def _hist_builder(series: pd.Series):
        d = pd.DataFrame({"score": series})

        def build(alt):
            return alt.Chart(d).mark_bar(color=THEME.teal).encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Score",
                       axis=alt.Axis(labelLimit=0)),
                y=alt.Y("count():Q", title="Count", axis=alt.Axis(labelLimit=0)),
            ).properties(height=220)
        return build

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Raw / naive score")
        uc.render_chart(_hist_builder(valid["raw_score"]), fallback=valid[["ticker", "raw_score"]])
    with c2:
        st.caption("Normalized / residual score")
        uc.render_chart(_hist_builder(valid["normalized_score"]), fallback=valid[["ticker", "normalized_score"]])


# --------------------------------------------------------------- matrix --
def _reconstruct_submatrix(df: pd.DataFrame, tickers: list[str], horizon: str, layer: str):
    # De-dupe while preserving order FIRST -- a repeated ticker (the default
    # top-10/bottom-10 selection overlaps when the universe has <20 scored
    # names, and a user could paste a duplicate into the search box) would
    # otherwise leave `sub`'s index non-unique, making `valid.get(t)` return
    # a multi-row Series instead of a scalar bool and crash the truth-value
    # check below -- a real bug caught via the view's own render test.
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]
    sub = df[df["ticker"].isin(tickers)].drop_duplicates(subset="ticker").set_index("ticker").reindex(tickers)
    key = "total_return" if layer == "raw" else "residual_return"
    var_key = "total_var" if layer == "raw" else "resid_var"
    horizon_days = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252, "12_1": 231}[horizon]
    r = sub[key].apply(lambda d: (d or {}).get(horizon))
    var = sub[var_key]
    valid = r.notna() & var.notna()
    if valid.sum() < 2:
        return None, []
    order = [t for t in tickers if bool(valid.get(t, False))]
    r_arr = r.reindex(order).to_numpy(dtype=float)
    var_arr = var.reindex(order).to_numpy(dtype=float)
    D = pw.spread_matrix(r_arr, var_arr, horizon_days)
    return D, order


def _matrix(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("Search / pick tickers (comma-separated, blank = top+bottom 10)", "",
                      key="mvt_matrix_q")
    horizon = c2.selectbox("Horizon", HORIZONS, index=1, format_func=lambda h: HORIZON_LABELS[h],
                           key="mvt_matrix_h")
    layer = c3.selectbox("Layer", ["residual", "raw"],
                         format_func=lambda k: "Residual (feeds Momentum)" if k == "residual" else "Raw / naive",
                         key="mvt_matrix_layer")

    if q.strip():
        picks = [t.strip().upper() for t in q.split(",") if t.strip()]
        tickers = [t for t in picks if t in set(df["ticker"])]
    else:
        valid = df.dropna(subset=["normalized_score"])
        combined = (valid.nlargest(10, "normalized_score")["ticker"].tolist()
                   + valid.nsmallest(10, "normalized_score")["ticker"].tolist())
        seen = set()
        tickers = [t for t in combined if not (t in seen or seen.add(t))]  # top/bottom can overlap if n<20

    if len(tickers) < 2:
        st.info("Pick at least 2 valid tickers to render the matrix.")
        return

    D, order = _reconstruct_submatrix(df, tickers, horizon, layer)
    if D is None or len(order) < 2:
        st.info("Not enough data for that horizon/selection.")
        return

    mat = pd.DataFrame(D, index=order, columns=order)
    try:
        sty = mat.style.map(lambda v: uc.grad_diverging(v, 3.0)).format("{:+.2f}")
        st.dataframe(sty, use_container_width=True)
    except Exception:
        st.dataframe(mat, use_container_width=True)
    st.caption("Row beats column when positive (vol-normalized spread). Reconstructed live from "
              "committed return/variance vectors -- the full pairwise matrix is never persisted.")


# ------------------------------------------------------------- drilldown --
def _instrument_detail(df: pd.DataFrame) -> None:
    tickers = sorted(df["ticker"].unique())
    picked = st.selectbox("Select an instrument", tickers, key="mvt_detail_ticker")
    row = df[df["ticker"] == picked].iloc[0].to_dict()

    c1, c2 = st.columns(2)
    c1.metric("Normalized (residual) score", f"{row.get('normalized_score'):+.1f}"
             if row.get("normalized_score") is not None else "—")
    c2.metric("Raw (naive) score", f"{row.get('raw_score'):+.1f}"
             if row.get("raw_score") is not None else "—")

    raw_pct = row.get("raw_percentiles") or {}
    resid_pct = row.get("residual_percentiles") or {}
    tbl_rows = []
    for h in HORIZONS:
        if h not in raw_pct and h not in resid_pct:
            continue
        tbl_rows.append({"Horizon": HORIZON_LABELS[h],
                         "Raw peer pctile": raw_pct.get(h), "Residual peer pctile": resid_pct.get(h)})
    if tbl_rows:
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    consistency = row.get("consistency") or {}
    if consistency:
        st.markdown(uc.numeric_slab([
            {"label": "Bullish horizons", "value": str(consistency.get("n_bullish", 0))},
            {"label": "Bearish horizons", "value": str(consistency.get("n_bearish", 0))},
            {"label": "Timeframe agreement",
             "value": f"{consistency.get('agreement', 0):.0%}" if consistency.get("agreement") is not None else "—"},
            {"label": "Acceleration (1M-12M pctile)",
             "value": f"{consistency.get('acceleration'):+.1f}" if consistency.get("acceleration") is not None else "—"},
        ]), unsafe_allow_html=True)

    horizon = st.selectbox("Strongest/weakest relationships — horizon", HORIZONS, index=1,
                           format_func=lambda h: HORIZON_LABELS[h], key="mvt_detail_horizon")
    peers = [t for t in df["ticker"].tolist() if t != picked]
    sample_peers = peers if len(peers) <= 300 else list(np.random.default_rng(0).choice(peers, 300, replace=False))
    universe_for_sw = [picked] + list(sample_peers)
    D, order = _reconstruct_submatrix(df, universe_for_sw, horizon, "residual")
    if D is not None and picked in order:
        i = order.index(picked)
        sw = pw.strongest_weakest(D, order, i, top=5)
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Strongest pairwise relationships (residual)")
            st.dataframe(pd.DataFrame(sw["strongest"]), hide_index=True, use_container_width=True)
        with c4:
            st.caption("Weakest pairwise relationships (residual)")
            st.dataframe(pd.DataFrame(sw["weakest"]), hide_index=True, use_container_width=True)


# ---------------------------------------------------------- methodology --
def _methodology(doc: dict) -> None:
    st.markdown(
        "**In plain language:**\n\n"
        "1. Every instrument is compared against every peer in its universe.\n"
        "2. A vol-normalized relative return (\"spread\") is computed for every pair, at six horizons.\n"
        "3. A statistical factor model (PCA) separates each instrument's return into a "
        "COMMON component (shared market/sector movement) and a RESIDUAL (idiosyncratic) component.\n"
        "4. The pairwise spreads are recomputed on the residual returns -- this is what actually "
        "asks \"is this instrument trending on its own\" rather than \"is it just moving with everything else\".\n"
        "5. Each instrument's peer-beaten percentile at each horizon is converted to a -20..+20 score.\n"
        "6. Horizons are combined on DISJOINT return increments (0-1M/1-3M/3-6M/6-9M/9-12M), not the "
        "nested 1M/3M/6M/9M/12M windows directly, so a 12M reading doesn't triple-count last month's move.\n"
        "7. The result is one more input to the overall Momentum score, alongside the five existing factors."
    )
    st.markdown("---")
    st.markdown(
        "**Advanced / methodology detail:**\n\n"
        "- Two scores are shown: **Raw** (vol-normalized total-return spread -- measured on a "
        "synthetic panel to be ~0.9+ Spearman-redundant with plain cross-sectional momentum, i.e. "
        "arithmetically dominated by each name's own return) and **Normalized** (the same pairwise "
        "construction on PCA RESIDUAL returns -- measurably less redundant, and what feeds the "
        "Momentum composite).\n"
        f"- This run: **{doc.get('k_factors', '—')}** statistical factors retained, explaining "
        f"**{doc.get('explained_variance_ratio', 0):.0%}** of common variance; effective factor "
        f"count (Shannon-entropy based, section 20 of the spec) = **{doc.get('effective_factor_count', '—')}** "
        "-- a low reading means recent behavior is explained by fewer dominant statistical factors; "
        "this is a descriptive regime diagnostic, NOT a signal on its own.\n"
        "- Horizon weights for the normalized score use equal-risk-contribution (ERC), shrunk 50% "
        "toward equal weight (pure risk-parity has no notion of signal quality, only correlation, "
        "and can overweight a noisy-but-uncorrelated horizon).\n"
        "- ETF universe excludes leveraged/inverse funds via an explicit list, a name regex, and an "
        "empirical vol/correlation backstop; near-duplicate instruments (e.g. SPY vs VOO) collapse "
        "to their most liquid member.\n"
        "- Minimum history and covariance-estimation-window requirements are documented in "
        "config.MOM_MVT_MIN_BARS / MOM_MVT_COV_WINDOW.\n"
        "- No look-ahead: the factor model and every horizon return use only trailing data as of "
        "the run date."
    )

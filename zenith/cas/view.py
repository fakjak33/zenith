"""Streamlit rendering for the CAS tab. Kept out of app.py so the viewer stays
thin. Reuses the Zenith theme/section helpers; uses altair (ships with Streamlit)
for the overlap heatmap."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import store_cas, registry, contingency
from .universe import label_of
from ..ui_theme import section, help_badge
from . import DISCLAIMER
from .help_text import HELP

_STATE_COLOR = {"buy": "#2ec4b6", "neutral": "#b8b8b8", "sell": "#ff5a3c"}

SEGMENT_TABS = ["Overview", "Strategies", "Flows & positioning", "Themes",
                "Factor Rotation", "Asset detail", "History & hit-rate", "Rebalance",
                "Contingency", "BCT consensus", "Models & notes"]

# short hover help per segment tab
_TAB_HELP = {
    "Overview": HELP["overlap"], "Strategies": "Per-asset technical/statistical strategy battery.",
    "Flows & positioning": HELP["flows"], "Themes": HELP["themes"],
    "Factor Rotation": HELP["frm"], "Asset detail": "Every CAS signal for one ticker, in one place.",
    "History & hit-rate": HELP["hitrate"],
    "Rebalance": HELP["rebalance"], "Contingency": "Pre-planned playbooks that arm on triggers.",
    "BCT consensus": HELP["consensus"], "Models & notes": "Tune model weights and log research.",
}

# per-column header help ("?" on table headers) + legends
COLS = {
    "net": "Average signal across all of an asset's models (-1 sell … +1 buy).",
    "mean_strength": "Average absolute strength of the aligned signals — how strong, ignoring direction.",
    "buy_count": "How many of the asset's models are flashing buy.",
    "sell_count": "How many models are flashing sell.",
    "n_segments": "How many distinct segments (strategies / themes / flows / …) weighed in.",
    "signal": HELP["signal"], "state": HELP["state"], "family": "The specific model / indicator.",
    "segment": HELP["segment"], "confidence": HELP["confidence"], "horizon": HELP["horizon"],
    "percentile": "Where today's reading sits in its own history (1.0 = highest ever).",
    "composite": "Blended BCT ensemble score for the asset.", "behavioral": HELP["behavioral"],
    "physical": HELP["physical"], "entanglement": HELP["entanglement"],
    "signal_variety": HELP["signal_variety"], "entropy": HELP["entropy"],
    "rationale": "Plain-English reason for the signal.", "label": "Human-readable name.",
    "group": HELP.get("frm_group", ""), "region": "Region the ETF targets.",
}


def _signals_df() -> pd.DataFrame:
    rows = store_cas.load("signals", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _colcfg(cols) -> dict:
    """st.column_config map giving each column a hover '?' from COLS."""
    return {c: st.column_config.Column(help=COLS[c]) for c in cols if c in COLS}


def _price_series(ticker: str):
    """Daily close for a ticker from the cached CAS price panel (no network)."""
    import io
    cache = store_cas.cache_get("prices_2y", 999) or store_cas.cache_get("prices_5y", 999) or {}
    j = cache.get(ticker)
    if not j:
        return None
    try:
        return pd.read_json(io.StringIO(j), orient="split")["close"]
    except Exception:
        return None


def _buy_sell_panels(sub: pd.DataFrame, n: int = 12, cols=None) -> None:
    """Side-by-side Top buys / Top sells tables from a signals sub-frame."""
    if sub.empty:
        st.caption("No signals.")
        return
    cols = cols or ["asset", "label", "family", "signal", "confidence", "rationale"]
    cols = [c for c in cols if c in sub.columns]
    s = sub.sort_values("signal", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("🟢 **Strongest buys**")
        st.dataframe(s.head(n)[cols], use_container_width=True, hide_index=True,
                     height=min(440, 45 + 35 * n), column_config=_colcfg(cols))
    with c2:
        st.caption("🔴 **Strongest sells**")
        st.dataframe(s.tail(n).iloc[::-1][cols], use_container_width=True, hide_index=True,
                     height=min(440, 45 + 35 * n), column_config=_colcfg(cols))


def _signal_bar(rows: list[dict], value: str = "signal", label: str = "label",
                height_per: int = 24, dom=(-1, 1)) -> None:
    if not rows:
        st.caption("No data.")
        return
    df = pd.DataFrame(rows)
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{value}:Q", title=None, scale=alt.Scale(domain=list(dom))),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            color=alt.condition(alt.datum[value] >= 0, alt.value("#2ec4b6"), alt.value("#ff5a3c")),
            tooltip=[c for c in df.columns if c in (label, value, "region", "group", "family")],
        ).properties(height=max(160, height_per * len(df))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render() -> None:
    status = store_cas.load("status", {})
    st.caption(DISCLAIMER)
    if not status:
        st.info("No CAS data yet. Run `python -m zenith.cas.compute --cadence weekly` "
                "(or wait for the scheduled Action) to populate signals.")
        return
    regime = status.get("regime", "?")
    st.caption(f"As of {status.get('date','?')} · cadence {status.get('cadence','?')} · "
               f"{status.get('n_signals',0)} signals · {status.get('n_assets',0)} assets · "
               f"regime: **{regime}**")

    seg = st.radio("CAS segment", SEGMENT_TABS, horizontal=True, label_visibility="collapsed")
    if _TAB_HELP.get(seg):
        st.caption(f"ⓘ {_TAB_HELP[seg]}")
    df = _signals_df()

    if seg == "Overview":
        _overview(df)
    elif seg == "Strategies":
        _segment(df, "strategies", "Per-asset strategy signals (101-alpha + classics)",
                 help_key="signal")
    elif seg == "Flows & positioning":
        _flows(df)
    elif seg == "Themes":
        _themes(df)
    elif seg == "Factor Rotation":
        _factor_rotation(df)
    elif seg == "Asset detail":
        _asset_detail(df)
    elif seg == "History & hit-rate":
        _history()
    elif seg == "Rebalance":
        _rebalance()
    elif seg == "Contingency":
        _contingency()
    elif seg == "BCT consensus":
        _consensus()
    elif seg == "Models & notes":
        _registry()


def _overview(df: pd.DataFrame) -> None:
    cons = store_cas.load("consensus", [])
    ovl = store_cas.load("overlap", {})
    ranked = ovl.get("ranked", [])

    # --- strongest buys vs sells (the headline read) ---
    st.markdown(section("Strongest signals — buys vs sells", 0, help=HELP["consensus"]),
                unsafe_allow_html=True)
    if cons:
        cdf = pd.DataFrame(cons)
        cdf["label"] = cdf["asset"].map(label_of)
        show = ["asset", "label", "state", "composite", "confidence", "signal_variety"]
        c1, c2 = st.columns(2)
        with c1:
            st.caption("🟢 **Top buys** (highest composite)")
            st.dataframe(cdf.sort_values("composite", ascending=False).head(15)[show],
                         use_container_width=True, hide_index=True, height=560,
                         column_config=_colcfg(show))
        with c2:
            st.caption("🔴 **Top sells** (lowest composite)")
            st.dataframe(cdf.sort_values("composite").head(15)[show],
                         use_container_width=True, hide_index=True, height=560,
                         column_config=_colcfg(show))

    # --- cross-segment heatmap: top buys + top sells × segment ---
    st.markdown(section("Cross-segment heatmap — conviction names × segment", 2, help=HELP["signal"]),
                unsafe_allow_html=True)
    st.caption("Mean signal per segment for the strongest buys (top) and sells (bottom). "
               "Hover any cell for detail; green = buy, red = sell.")
    _segment_heatmap(df, cons)

    # --- overlap leaderboard with column help + legend ---
    st.markdown(section("Highest-conviction overlap (signals aligned across segments)", 4,
                        help=HELP["overlap"]), unsafe_allow_html=True)
    st.caption("**net** = average signal across the asset's models · **mean_strength** = average "
               "absolute strength of the aligned ones · **n_segments** = how many segments agree.")
    if ranked:
        top = pd.DataFrame(ranked).head(20)
        top["label"] = top["asset"].map(label_of)
        cols = ["asset", "label", "asset_class", "net", "buy_count", "sell_count",
                "n_segments", "mean_strength"]
        st.dataframe(top[cols], use_container_width=True, height=400, hide_index=True,
                     column_config=_colcfg(cols))


def _segment_heatmap(df: pd.DataFrame, cons: list[dict]) -> None:
    if df.empty:
        st.caption("No signals.")
        return
    seg_mean = df.groupby(["asset", "segment"])["signal"].mean().reset_index()
    if cons:
        cdf = pd.DataFrame(cons).sort_values("composite", ascending=False)
        picks = cdf["asset"].head(15).tolist() + cdf["asset"].tail(15).tolist()
    else:
        picks = (df.groupby("asset")["signal"].apply(lambda s: s.abs().mean())
                 .sort_values(ascending=False).head(30).index.tolist())
    sub = seg_mean[seg_mean["asset"].isin(picks)].copy()
    sub["label"] = sub["asset"].map(lambda t: f"{t} · {label_of(t)}"[:28])
    order = [f"{t} · {label_of(t)}"[:28] for t in picks]
    try:
        import altair as alt
        ch = (alt.Chart(sub).mark_rect().encode(
            x=alt.X("segment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("label:N", title=None, sort=order),
            color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                            legend=alt.Legend(title="signal")),
            tooltip=["asset", "segment", alt.Tooltip("signal:Q", format="+.2f")],
        ).properties(height=max(300, 24 * len(picks))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(sub.pivot_table(index="label", columns="segment", values="signal"),
                     use_container_width=True)


def _segment(df: pd.DataFrame, segment: str, title: str, help_key: str | None = None) -> None:
    st.markdown(section(title, 0, help=HELP.get(help_key) if help_key else None),
                unsafe_allow_html=True)
    if df.empty:
        st.caption("No signals.")
        return
    sub = df[df["segment"] == segment].copy()
    if sub.empty:
        st.caption("No signals in this segment.")
        return
    sub["label"] = sub["asset"].map(label_of)

    _buy_sell_panels(sub)

    st.markdown(section("All signals", 2), unsafe_allow_html=True)
    st.caption("Tip: open the **Asset detail** tab to see every model for a single ticker together.")
    assets = sorted(sub["asset"].unique())
    pick = st.multiselect("Filter assets", assets, default=[])
    if pick:
        sub = sub[sub["asset"].isin(pick)]
    cols = ["asset", "label", "family", "state", "signal", "horizon", "confidence", "rationale"]
    st.dataframe(sub[cols].sort_values("signal", ascending=False),
                 use_container_width=True, height=520, hide_index=True, column_config=_colcfg(cols))


def _flows(df: pd.DataFrame) -> None:
    _segment(df, "flows", "Flows & positioning (COT proxy + volume proxy)")
    st.markdown(section("Dynamics with no free feed (proxy / manual)", 2), unsafe_allow_html=True)
    from .signals.flows import MANUAL_DYNAMICS
    st.caption("These institutional dynamics are paywalled (SpotGamma, EPFR, etc.). "
               "They appear here as placeholders — wire a paid feed or hand-enter later:")
    st.write(", ".join(d.replace("_", " ") for d in MANUAL_DYNAMICS))


def _themes(df: pd.DataFrame) -> None:
    st.markdown(section("Themes — sector / industry / theme rotation", 0, help=HELP["themes"]),
                unsafe_allow_html=True)
    sub = df[df["segment"] == "themes"].copy() if not df.empty else pd.DataFrame()
    if sub.empty:
        st.caption("No theme signals yet. Run the CAS compute.")
        return
    sub["label"] = sub["asset"].map(label_of)
    st.caption("Families: **relative_strength** (3m/6m vs SPY) · **rs_short** (1m) · **rs_long** (6m) "
               "· **trend_vs_200dma** (absolute trend). Tickers show their descriptive label.")
    _buy_sell_panels(sub)

    st.markdown(section("Theme signals — asset × family", 2, help=HELP["themes"]),
                unsafe_allow_html=True)
    _theme_heatmap(sub)

    st.markdown(section("All theme signals", 3), unsafe_allow_html=True)
    cols = ["asset", "label", "family", "state", "signal", "horizon", "confidence", "rationale"]
    have = [c for c in cols if c in sub.columns]
    st.dataframe(sub[have].sort_values("signal", ascending=False), use_container_width=True,
                 height=420, hide_index=True, column_config=_colcfg(have))

    _expanded_heatmap()


def _theme_heatmap(sub: pd.DataFrame) -> None:
    fams = ["relative_strength", "rs_short", "rs_long", "trend_vs_200dma"]
    rs = sub[sub["family"].isin(fams)].copy()
    if rs.empty:
        st.caption("No relative-strength signals.")
        return
    strength = (rs[rs["family"] == "relative_strength"].set_index("asset")["signal"]
                .abs().sort_values(ascending=False))
    top = strength.head(25).index.tolist()
    rs = rs[rs["asset"].isin(top)]
    rs["row"] = rs["asset"].map(lambda t: f"{t} · {label_of(t)}"[:26])
    try:
        import altair as alt
        ch = (alt.Chart(rs).mark_rect().encode(
            x=alt.X("family:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("row:N", title=None),
            color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1])),
            tooltip=["asset", "family", alt.Tooltip("signal:Q", format="+.2f"), "rationale"],
        ).properties(height=max(300, 24 * len(top))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(rs.pivot_table(index="row", columns="family", values="signal"),
                     use_container_width=True)


def _expanded_heatmap() -> None:
    """Composite signal for EVERY tracked factor/style/industry/region-sector (the full
    Factor-Rotation universe) — the user's 'all factors, not just the 13' view."""
    from .universe import frm_tag, REGION_LABEL
    frm = [s for s in store_cas.load("signals", [])
           if s.get("segment") == "factor_rotation" and s.get("family") == "frm_composite"]
    if not frm:
        return
    rows = []
    for s in frm:
        tag = frm_tag(s["asset"]) or {}
        rows.append({"asset": s["asset"], "label": tag.get("label", s["asset"]),
                     "group": tag.get("group", ""), "signal": s["signal"],
                     "region": REGION_LABEL.get(tag.get("region"), tag.get("region", ""))})
    rdf = pd.DataFrame(rows)
    st.markdown(section("Expanded composite — every tracked factor, style, industry & theme", 4,
                        help="The full Factor-Rotation universe (~120 ETFs), not just the 13 core "
                             "styles. Green = bullish composite, red = bearish."),
                unsafe_allow_html=True)
    styles = rdf[rdf["group"] == "style"]
    if not styles.empty:
        st.caption("**Styles × region**")
        try:
            import altair as alt
            ch = (alt.Chart(styles).mark_rect().encode(
                x=alt.X("region:N", title=None),
                y=alt.Y("label:N", title=None),
                color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1])),
                tooltip=["asset", "label", "region", alt.Tooltip("signal:Q", format="+.2f")],
            ).properties(height=300))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.dataframe(styles.pivot_table(index="label", columns="region", values="signal"),
                         use_container_width=True)
    ind = rdf[rdf["group"] == "industry"].sort_values("signal", ascending=False)
    if not ind.empty:
        st.caption("**Industries — top & bottom by composite**")
        _signal_bar(pd.concat([ind.head(18), ind.tail(18)]).to_dict("records"))
    rs = rdf[rdf["group"] == "region_sector"].sort_values("signal", ascending=False)
    if not rs.empty:
        st.caption("**Region-sectors**")
        _signal_bar(rs.to_dict("records"))


def _factor_rotation(df: pd.DataFrame) -> None:
    from .universe import frm_tag, REGION_LABEL
    st.markdown(section("Factor Rotation Momentum", 0, help=HELP["frm"]), unsafe_allow_html=True)
    st.caption("Style factors + industries + region-sectors traded via US-listed ETFs. "
               "Time-series momentum (each ETF on its own trend) + cross-sectional rotation "
               "(ranked within peer groups). Gupta & Kelly (2019) + Man Group style-trend.")

    sub = df[df["segment"] == "factor_rotation"].copy() if not df.empty else pd.DataFrame()
    if not sub.empty:
        tags = sub["asset"].map(lambda t: frm_tag(t) or {})
        sub["group"] = [d.get("group", "") for d in tags]
        sub["label"] = [d.get("label", "") for d in tags]
        sub["region"] = [REGION_LABEL.get(d.get("region"), d.get("region", "")) for d in tags]

        st.caption("**frm_ts_mom** = own-trend (time-series) momentum · **frm_cs_region/peer** = "
                   "ranked vs peers (cross-sectional) · **frm_composite** = 0.6·TS + 0.3·CS-region "
                   "+ 0.1·CS-peer. Buys/sells below use the composite.")
        comp = sub[sub["family"] == "frm_composite"]
        _buy_sell_panels(comp, cols=["asset", "group", "label", "region", "signal", "confidence"])

        st.markdown(section("Style composite — factor × region", 2, help=HELP["frm_composite"]),
                    unsafe_allow_html=True)
        styles = sub[(sub["family"] == "frm_composite") & (sub["group"] == "style")]
        _frm_heatmap(styles)

        st.markdown(section("All factor-rotation signals", 4, help=HELP["frm_ts_mom"]),
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        groups = sorted(g for g in sub["group"].unique() if g)
        gpick = c1.multiselect("Filter group", groups, default=[], help=HELP["frm_group"])
        fpick = c2.multiselect("Filter family", sorted(sub["family"].unique()), default=[])
        view = sub
        if gpick:
            view = view[view["group"].isin(gpick)]
        if fpick:
            view = view[view["family"].isin(fpick)]
        st.dataframe(view[["asset", "group", "label", "region", "family", "state", "signal",
                           "confidence", "rationale"]].sort_values("signal", ascending=False),
                     use_container_width=True, height=440)
    else:
        st.caption("No factor-rotation signals yet. Run `python -m zenith.cas.compute` to populate.")

    _frm_backtest()


def _frm_heatmap(comp: pd.DataFrame) -> None:
    if comp.empty:
        st.caption("No composite style signals.")
        return
    try:
        import altair as alt
        chart = (alt.Chart(comp).mark_rect().encode(
            x=alt.X("region:N", title=None),
            y=alt.Y("label:N", title=None),
            color=alt.Color("signal:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
                            legend=alt.Legend(title="composite")),
            tooltip=["asset", "label", "region", "signal", "state"],
        ).properties(height=320))
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(comp.pivot_table(index="label", columns="region", values="signal",
                                      aggfunc="first"), use_container_width=True)


def _frm_backtest() -> None:
    st.markdown(section("Academic backtest — Factor Momentum Everywhere (replication)", 0,
                        help=HELP["backtest"]), unsafe_allow_html=True)
    bt = store_cas.load("backtest", {})
    if not bt:
        st.caption("No backtest yet. Run `python -m zenith.cas.backtest.factor_momentum` "
                   "(or wait for the monthly Action).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Factors tested", bt.get("n_factors", 0), help=HELP["ar1"])
    c2.metric("Positive / Significant AR(1)",
              f"{bt.get('n_positive', 0)} / {bt.get('n_significant', 0)}", help=HELP["t_stat"])
    c3.metric("Combined TS Sharpe", bt.get("combined_ts_sharpe", "—"), help=HELP["ts_sharpe"])
    st.caption(f"As of {bt.get('as_of','?')} · {bt.get('source','')} · "
               "replicates the paper's autocorrelation + TS factor-momentum result on the "
               "publicly available academic factor set.")
    facts = bt.get("factors", {})
    if facts:
        rows = [{"factor": k, **v} for k, v in facts.items()]
        cols = ["factor", "region", "ar1", "t_stat", "positive", "significant",
                "ts_sharpe", "n_months"]
        st.dataframe(pd.DataFrame(rows)[cols].sort_values("ts_sharpe", ascending=False),
                     use_container_width=True, height=360)


def _price_signal_overlay(ticker: str, hist_rows: list[dict] | None = None) -> None:
    price = _price_series(ticker)
    hist = (hist_rows if hist_rows is not None
            else [h for h in store_cas.load("history", []) if h.get("asset") == ticker])
    try:
        import altair as alt
    except Exception:
        if price is not None:
            st.line_chart(price.tail(252))
        return
    layers = []
    if price is not None and len(price) > 5:
        pdf = price.tail(252).reset_index()
        pdf.columns = ["date", "close"]
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf["price %"] = (pdf["close"] / pdf["close"].iloc[0] - 1.0) * 100
        layers.append(alt.Chart(pdf).mark_line(color="#b8b8b8").encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price %:Q", axis=alt.Axis(title="price % (rebased)")),
            tooltip=["date:T", alt.Tooltip("price %:Q", format="+.1f")]))
    if hist:
        hdf = pd.DataFrame(hist)
        hdf["date"] = pd.to_datetime(hdf["date"])
        layers.append(alt.Chart(hdf).mark_line(point=False).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("signal:Q", scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(title="signal")),
            color=alt.Color("family:N", legend=alt.Legend(title="signal")),
            tooltip=["date:T", "family:N", alt.Tooltip("signal:Q", format="+.2f")]))
    if not layers:
        st.caption("No cached price or signal history for this ticker.")
        return
    st.altair_chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=320),
                    use_container_width=True)
    st.caption("Grey = the underlying's price (rebased to 0%). Coloured = its CAS signal(s). "
               "Look for the signal leading turns in price.")


def _asset_detail(df: pd.DataFrame) -> None:
    from .universe import frm_tag, REGION_LABEL
    st.markdown(section("Asset detail — every signal for one ticker", 0,
                        help="All CAS signals for a single underlying, across every segment, in "
                             "one place — with its price and signal history."), unsafe_allow_html=True)
    if df.empty:
        st.caption("No signals yet. Run the CAS compute.")
        return
    assets = sorted(df["asset"].unique())
    default = "SPY" if "SPY" in assets else assets[0]
    pick = st.selectbox("Ticker", assets, index=assets.index(default),
                        help="Pick any underlying that appears in the CAS tables.")
    one = df[df["asset"] == pick].copy()
    one["label"] = one["asset"].map(label_of)
    tag = frm_tag(pick) or {}
    desc = label_of(pick)
    if tag:
        desc += f" · {tag.get('group','')} / {REGION_LABEL.get(tag.get('region'), tag.get('region',''))}"
    st.markdown(f"### {pick} — {desc}")

    rec = {c["asset"]: c for c in store_cas.load("consensus", [])}.get(pick, {})
    if rec:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Consensus", rec.get("state", "?"),
                  f"{rec.get('composite', 0):+.2f}", help=COLS["composite"])
        m2.metric("Signal variety", rec.get("signal_variety", 0), help=COLS["signal_variety"])
        m3.metric("Entropy", rec.get("entropy", 0), help=COLS["entropy"])
        m4.metric("Confidence", rec.get("confidence", "—"))

    st.markdown(section("Signals by family (this ticker)", 2, help=HELP["signal"]),
                unsafe_allow_html=True)
    _signal_bar(one[["family", "signal"]].to_dict("records"), label="family", height_per=22)

    st.markdown(section("Price & signal history", 4, help=HELP["history"]), unsafe_allow_html=True)
    _price_signal_overlay(pick)

    st.markdown(section("All signals for this ticker", 1), unsafe_allow_html=True)
    cols = ["segment", "family", "state", "signal", "horizon", "confidence", "rationale"]
    have = [c for c in cols if c in one.columns]
    st.dataframe(one[have].sort_values(["segment", "signal"], ascending=[True, False]),
                 use_container_width=True, height=460, hide_index=True, column_config=_colcfg(have))


def _history() -> None:
    from .universe import frm_tag, REGION_LABEL
    st.markdown(section("Signal history — how signals evolved, vs price", 0, help=HELP["history"]),
                unsafe_allow_html=True)
    st.caption("**frm_ts_mom** = own-trend momentum · **frm_cs_region** = ranked vs peers "
               "(cross-sectional) · **frm_composite** = the blend (0.6 TS + 0.3 CS-region + "
               "0.1 CS-peer). Grey line = the asset's price (rebased) so you can see if the signal "
               "leads the move.")
    hist = store_cas.load("history", [])
    if hist:
        hdf = pd.DataFrame(hist)
        assets = sorted(hdf["asset"].unique())
        c1, c2 = st.columns([1, 2])
        pick = c1.selectbox("Asset", assets,
                            index=assets.index("MTUM") if "MTUM" in assets else 0)
        fams = sorted(hdf[hdf["asset"] == pick]["family"].unique())
        chosen = c2.multiselect("Signals to compare", fams, default=fams,
                                help="Compare time-series vs cross-sectional vs the composite.")
        tag = frm_tag(pick) or {}
        st.caption(f"{label_of(pick)} · {tag.get('group','')} / "
                   f"{REGION_LABEL.get(tag.get('region'), tag.get('region',''))}")
        rows = hdf[(hdf["asset"] == pick) & (hdf["family"].isin(chosen))].to_dict("records")
        _price_signal_overlay(pick, hist_rows=rows)
    else:
        st.caption("No signal history yet — it backfills on the next CAS run.")

    # --- hit-rate, selectable across models ---
    st.markdown(section("Predictive hit-rate — did the signal call the move?", 2, help=HELP["hitrate"]),
                unsafe_allow_html=True)
    hr = store_cas.load("hitrate", {})
    if not hr or "models" not in hr:
        st.caption("No hit-rate yet. Run the CAS compute to backfill it.")
        return
    models = hr["models"]
    mkey = st.selectbox("Model / signal", list(models),
                        format_func=lambda k: models[k].get("label", k),
                        help="Pick any backfillable CAS model to see its predictive hit-rate.")
    m = models[mkey]
    st.caption(f"{m.get('label','')} · {hr.get('n_assets',0)} assets · as of {hr.get('as_of','?')}. "
               "Each number is the % of directional calls that matched the ACTUAL forward return "
               "(50% = coin-flip; higher = predictive). " + hr.get("note", ""))
    byh = m.get("by_horizon", {})
    cols = st.columns(4)
    for i, h in enumerate(["1m", "3m", "6m", "12m"]):
        rec = byh.get(h, {})
        rate = rec.get("hit_rate")
        cols[i].metric(f"{h} hit-rate", "—" if rate is None else f"{rate:.0%}",
                       help=f"n = {rec.get('n', 0)} observations")

    st.markdown(section("Hit-rate by asset group", 4,
                        help="Rows = factor / industry / region-sector groups; columns = forward "
                             "horizons. Each cell = % of that group's directional calls that were right."),
                unsafe_allow_html=True)
    bg = m.get("by_group", {})
    rows = []
    for grp, d in bg.items():
        row = {"group": grp}
        for h in ("1m", "3m", "6m", "12m"):
            r = d.get(h, {})
            row[h] = None if r.get("hit_rate") is None else round(r["hit_rate"], 3)
        rows.append(row)
    if rows:
        cc = {h: st.column_config.NumberColumn(h, help=f"{h} forward hit-rate (fraction).")
              for h in ("1m", "3m", "6m", "12m")}
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=cc)


def _rebalance() -> None:
    st.markdown(section("Upcoming rebalance & key dates", 0), unsafe_allow_html=True)
    events = store_cas.load("rebalance", [])
    if events:
        cc = {"days_until": st.column_config.Column(help="Calendar days until the event."),
              "kind": st.column_config.Column(help="Event type (opex / index_rebal / period_end)."),
              "note": st.column_config.Column(help="Why it matters for flows.")}
        st.dataframe(pd.DataFrame(events)[["date", "days_until", "event", "kind", "note"]],
                     use_container_width=True, height=420, hide_index=True, column_config=cc)
    else:
        st.caption("No upcoming events loaded.")


def _contingency() -> None:
    st.markdown(section("Contingency playbook", 0), unsafe_allow_html=True)
    for s in contingency.load():
        flag = "🔴 ACTIVE" if s.get("active") else "⚪ standby"
        with st.expander(f"{flag} — {s['name']}  ·  {s.get('status','')}",
                         expanded=bool(s.get("active"))):
            st.markdown(f"**Trigger:** {s['trigger']}")
            st.markdown(f"**Proactive:** {s['proactive']}")
            st.markdown(f"**Reactive:** {s['reactive']}")


def _consensus() -> None:
    st.markdown(section("Behavioral Consensus Theory — ensemble view", 0), unsafe_allow_html=True)
    st.caption("BCT-inspired ensemble (honest interpretation, not literal physics): "
               "behavioral = positioning/mean-reversion extremes · physical = trend/regime/vol · "
               "entanglement = cross-horizon agreement. Variety/entropy = Ashby's-Law analogue.")
    cons = store_cas.load("consensus", [])
    if cons:
        cdf = pd.DataFrame(cons)
        cdf.insert(1, "label", cdf["asset"].map(label_of))
        st.dataframe(cdf, use_container_width=True, height=560, hide_index=True,
                     column_config=_colcfg(list(cdf.columns)))


def _registry() -> None:
    st.markdown(section("Models & research notes", 0,
                        help="Tune how much each model counts toward the consensus, and log "
                             "research that should shape the models."), unsafe_allow_html=True)
    reg = registry.load()

    st.markdown("**Add research — paste an abstract, link a URL, or upload a paper**")
    fam = st.selectbox("Family this informs", sorted(reg["weights"].keys()),
                       help="Which model family the research bears on.")
    title = st.text_input("Note title / paper")
    url = st.text_input("Source URL (optional)",
                        help="A link to the paper/insight. The /zenith-research Claude skill can "
                             "later turn a flagged note into a signal-module scaffold.")
    upload = st.file_uploader("Upload a paper (PDF/txt, optional)", type=["pdf", "txt", "md"])
    abstract = st.text_area("Paste abstract or summary", height=150)
    w = st.slider("Set weight for this family", 0.0, 2.0, float(reg["weights"].get(fam, 1.0)), 0.05)
    if st.button("Save note + weight"):
        src = url.strip()
        body = abstract.strip()
        if upload is not None:
            src = src or upload.name
            if not body:
                try:
                    body = upload.getvalue().decode("utf-8", "ignore")[:4000]
                except Exception:
                    body = f"(uploaded {upload.name}; binary — process with /zenith-research)"
        status = "pending-review" if (src or upload) and not body.strip() else ""
        registry.add_note(fam, title or (src or "(untitled)"), body,
                          weight_adjustment=w, source=src, status=status)
        st.success(f"Saved note against '{fam}' (weight {w}). "
                   + ("Flagged pending-review for the /zenith-research skill." if status else ""))

    st.markdown(section("Current family weights", 2), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"family": k, "weight": v} for k, v in reg["weights"].items()]),
                 use_container_width=True, height=280)

    st.markdown(section("Research notes log", 4), unsafe_allow_html=True)
    if reg["notes"]:
        for n in reg["notes"][:30]:
            flag = f"  ·  🟡 {n['status']}" if n.get("status") else ""
            st.markdown(f"**{n['title']}**  ·  _{n['family']}_  ·  {n['ts']}{flag}")
            if n.get("source"):
                st.caption(f"source: {n['source']}")
            if n.get("abstract"):
                st.caption(n["abstract"][:600])
    else:
        st.caption("No notes yet.")

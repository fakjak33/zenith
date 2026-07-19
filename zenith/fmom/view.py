"""Streamlit rendering for the FACTOR MOMENTUM tab — visual, Zenith-themed.

Reads only committed JSON (data/fmom/*). Chart sizing rule everywhere: factors
live on the Y axis with `labelLimit=0` and an explicit per-row pixel height,
so labels are never rotated, squeezed or truncated no matter how many factors
a family carries.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load, load_month, archive_months
from .catalog import (AQR65, AQR_DEFS, AQR_SUPPLEMENTAL, CHAR_META, ETF_PROXIES,
                      MAN_STYLES, OSAP_CHAR_MAP)
from .history import MODELS, family_of, summarize
from ..config import THEME
from ..ui_theme import section, stamp

ROW_PX = 22                      # px per factor row in bar charts
HM_PX = 20                       # px per factor row in heatmaps
TRIM_AT = 60                     # families larger than this default to top/bottom 25
TRIM_N = 25

AQR_NAME = {r["abbr"]: r["name"] for r in AQR65 + AQR_SUPPLEMENTAL}
AQR_ROW = {r["abbr"]: r for r in AQR65 + AQR_SUPPLEMENTAL}

FAMILY_META = {
    "etf": ("ETF PROXIES", "Real, tradable factor ETFs — monthly returns in "
                           "excess of SPY. The long leg is directly buyable."),
    "man": ("MAN 13 STYLES", "13 broad equity styles as equal-weight composites "
                             "of tagged strategic-beta ETFs, minus SPY — after "
                             "Man AHL's style-trend work. Membership is shown "
                             "below the ranked table."),
    "aqr": ("AQR 65 FACTORS", "Gupta & Kelly's academic factor set via published "
                              "long-short returns (Ken French sorts + AQR "
                              "QMJ/BAB/HML-Devil). Factors without a free "
                              "monthly-updating source are listed in the "
                              "coverage panel; the Russell 1000 replication "
                              "layer shows current holdings per factor."),
    "osap": ("OSAP 212 PREDICTORS", "The full Open Source Asset Pricing "
                                    "(Chen-Zimmermann) anomaly library — 212 "
                                    "published long-short predictors with "
                                    "documented definitions. The dataset "
                                    "updates ~once a year, so these signals "
                                    "form on the latest published month; the "
                                    "paper finds factor momentum persists at "
                                    "1-60 month horizons, so they still carry "
                                    "information. Not in the monthly tracker."),
}

SORT_MODES = {
    "Strongest LONG first": lambda rows: sorted(rows, key=lambda r: -r["s"]),
    "Strongest SHORT first": lambda rows: sorted(rows, key=lambda r: r["s"]),
    "Biggest |signal| first": lambda rows: sorted(rows, key=lambda r: -abs(r["s"])),
}

MODEL_LABEL = {
    "etf_tsfm": "ETF · TSFM", "etf_csfm": "ETF · CSFM",
    "man_tsfm": "Man · TSFM", "man_csfm": "Man · CSFM",
    "aqr_tsfm": "AQR · TSFM", "aqr_csfm": "AQR · CSFM",
}

COLS = {
    "Rank": "Position when sorted by TSFM signal (1 = strongest).",
    "Factor": "Factor / style name. The ETF family shows the proxy ticker too.",
    "Proxy": "The tradable ETF standing in for the factor.",
    "1m return": "Previous month's factor return (ETF family: in excess of "
                 "SPY). This is the entire 1-1 formation window.",
    "De-meaned": "The 1-month return minus the cross-factor average — the "
                 "CSFM formation return.",
    "Vol (ann)": "Trailing 36-month annualized volatility of the factor's "
                 "monthly returns (min 24 months), excluding the formation "
                 "month. The signal's denominator.",
    "TSFM s": "Gupta & Kelly Eq. 1: previous-month return / trailing vol, "
              "capped at ±2. Positive = the model is long next month.",
    "CSFM s": "Same construction on the de-meaned return: long only the "
              "relative outperformers.",
    "Weight": "Signal-weighted portfolio weight. Longs sum to +1, shorts "
              "to −1 ($1 long / $1 short construction).",
    "Leg": "Which side of the $1/$1 portfolio holds the factor next month.",
    "Decile": "★ top / ▼ bottom decile of factors by signal strength.",
    "Source month": "The month whose return forms this factor's signal. AQR "
                    "workbook factors (QMJ, BAB, HML-Devil) publish 1-2 months "
                    "behind Ken French, so they can trail the others.",
}


# ---------------------------------------------------------------- helpers ---
def _grad(v, cap: float, color: tuple[int, int, int]) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    a = max(0.0, min(1.0, abs(float(v)) / cap))
    r, g, b = (int(11 + (c - 11) * a) for c in color)
    return f"background-color: rgb({r},{g},{b}); color: #fff;"


def _grad_diverging(v, cap=0.05):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return (_grad(v, cap, (46, 196, 182)) if float(v) >= 0
            else _grad(v, cap, (255, 90, 60)))


def _colcfg(cols) -> dict:
    return {c: st.column_config.Column(help=COLS[c]) for c in cols if c in COLS}


def _fmt_pct(v):
    return "" if v is None or pd.isna(v) else f"{v:+.1%}"


def _chip(label: str, color: str) -> str:
    return (f'<span style="display:inline-block; font-family:{THEME.font_display}; '
            f'font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; '
            f'color:{color}; border:1px solid {color}; padding:0.1rem 0.55rem; '
            f'margin:0 0.4rem 0.4rem 0;">{label}</span>')


def _freshness_chips(signals: dict) -> None:
    chips = []
    for fam, meta in (signals.get("sources") or {}).items():
        label, _ = FAMILY_META.get(fam, (fam.upper(), ""))
        lm = meta.get("last_month") or "?"
        note = " · annual release" if meta.get("annual") else ""
        chips.append(_chip(f"{label} · through {lm}{note}",
                           THEME.mauve if meta.get("annual") else THEME.teal))
        cov = meta.get("coverage")
        if cov:
            chips.append(_chip(f"AQR coverage: {cov['defined']} defined · "
                               f"{cov['live']} live · {cov['replicable']} "
                               f"replicated", THEME.mustard))
        missing = meta.get("missing") or []
        if missing and fam != "aqr":     # aqr gaps live in the coverage panel
            chips.append(_chip(f"{fam}: no data for {', '.join(missing)}",
                               THEME.coral))
    for fam, (label, blurb) in FAMILY_META.items():
        if fam not in (signals.get("sources") or {}):
            chips.append(_chip(f"{label} · coming soon", THEME.muted))
    st.markdown("".join(chips), unsafe_allow_html=True)


def _decile_mark(d: str) -> str:
    return {"top": "★", "bottom": "▼"}.get(d, "")


def _trim_rows(rows: list[dict], show_all: bool,
               n: int = TRIM_N) -> tuple[list[dict], bool]:
    """For big families (OSAP's 212), default charts to the n strongest longs +
    n strongest shorts by signal. Returns (rows, trimmed?)."""
    if show_all or len(rows) <= TRIM_AT:
        return rows, False
    by_s = sorted(rows, key=lambda r: -r["s"])
    keep = {id(r) for r in by_s[:n]} | {id(r) for r in by_s[-n:]}
    return [r for r in rows if id(r) in keep], True


def _factor_info(fam: str, factor: str, osap_cat: dict) -> dict:
    """definition / recipe / char / extra context for one factor, per family."""
    if fam == "etf":
        p = next((p for p in ETF_PROXIES if p["factor"] == factor), {})
        return {"name": p.get("name", factor), "definition": p.get("definition"),
                "recipe": p.get("recipe"), "char": p.get("char"),
                "proxy": p.get("ticker")}
    if fam == "man":
        s = MAN_STYLES.get(factor, {})
        return {"name": factor, "definition": s.get("definition"),
                "recipe": s.get("recipe"), "char": s.get("char")}
    if fam == "aqr":
        row = AQR_ROW.get(factor, {})
        d = AQR_DEFS.get(factor, (None, None))
        return {"name": row.get("name", factor), "definition": d[0],
                "recipe": d[1], "char": row.get("yf_char"),
                "replicated": bool(row.get("yf_char"))}
    if fam == "osap":
        rec = (osap_cat.get("signals") or {}).get(factor, {})
        sign = rec.get("sign")
        side = ("higher values of the signal predict HIGHER returns (long the top)"
                if (sign or 0) > 0 else
                "higher values predict LOWER returns (long the bottom)"
                if (sign or 0) < 0 else None)
        recipe = rec.get("definition")
        if recipe and side:
            recipe = f"{recipe} — {side}."
        meta = " · ".join(x for x in (
            rec.get("category"),
            f"{rec.get('authors')} ({rec.get('year')})" if rec.get("authors") else None,
            f"orig. sample {rec.get('sample_start')}–{rec.get('sample_end')}"
            if rec.get("sample_start") else None) if x)
        return {"name": rec.get("name", factor), "definition": meta or None,
                "recipe": recipe, "char": OSAP_CHAR_MAP.get(factor)}
    return {"name": factor}


# ------------------------------------------------------------------ charts --
def _signal_bars(rows: list[dict], lens: str, key: str) -> None:
    """Horizontal diverging signal bars, top/bottom decile outlined mustard."""
    skey = "s"                                 # rows are per-model already
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption("No signals.")
        return
    df["decile_mark"] = df["decile"].map(_decile_mark)

    def _label(r):
        if isinstance(r.get("proxy"), str):
            return f"{r['factor']} ({r['proxy']})"
        if r["factor"] in AQR_NAME:
            return f"{r['factor']} · {AQR_NAME[r['factor']][:26]}"
        return str(r["factor"])
    df["label"] = df.apply(_label, axis=1)
    order = df.sort_values(skey, ascending=False)["label"].tolist()
    try:
        import altair as alt
        base = alt.Chart(df)
        bars = base.mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X(f"{skey}:Q", title="signal s (capped at ±2)",
                    scale=alt.Scale(domain=[-2.2, 2.2])),
            y=alt.Y("label:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=0, labelFontSize=11)),
            color=alt.condition(f"datum.{skey} >= 0",
                                alt.value(THEME.teal), alt.value(THEME.coral)),
            stroke=alt.condition("datum.decile !== ''",
                                 alt.value(THEME.mustard), alt.value(None)),
            strokeWidth=alt.condition("datum.decile !== ''",
                                      alt.value(2), alt.value(0)),
            opacity=alt.condition("datum.decile !== ''",
                                  alt.value(1.0), alt.value(0.8)),
            tooltip=["factor", "proxy",
                     alt.Tooltip("r_1m:Q", title="1m return", format="+.2%"),
                     alt.Tooltip("r_1m_dm:Q", title="de-meaned", format="+.2%"),
                     alt.Tooltip("vol_ann:Q", title="vol (ann)", format=".1%"),
                     alt.Tooltip(f"{skey}:Q", title="signal", format="+.3f"),
                     alt.Tooltip("weight:Q", format="+.3f"), "leg", "decile"],
        )
        zero = base.mark_rule(color=THEME.muted).encode(x=alt.datum(0))
        caps = alt.Chart(pd.DataFrame({"x": [-2.0, 2.0]})).mark_rule(
            strokeDash=[4, 4], color=THEME.muted, opacity=0.6).encode(x="x:Q")
        ch = (bars + zero + caps).properties(height=max(240, ROW_PX * len(df)))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df[["factor", skey, "weight", "leg", "decile"]],
                     hide_index=True)


def _lens_scatter(ts_rows: list[dict], cs_rows: list[dict]) -> None:
    """s_ts vs s_cs per factor — where the two lenses agree and disagree."""
    ts = {r["factor"]: r for r in ts_rows}
    cs = {r["factor"]: r for r in cs_rows}
    recs = []
    for f in set(ts) & set(cs):
        s1, s2 = ts[f]["s"], cs[f]["s"]
        agree = ("both long" if s1 > 0 and s2 > 0 else
                 "both short" if s1 <= 0 and s2 <= 0 else "disagree")
        recs.append({"factor": f, "s_ts": s1, "s_cs": s2, "agree": agree,
                     "r_1m": ts[f]["r_1m"]})
    if not recs:
        return
    df = pd.DataFrame(recs)
    try:
        import altair as alt
        lim = 2.2
        pts = alt.Chart(df).mark_circle(size=160, opacity=0.9).encode(
            x=alt.X("s_ts:Q", title="TSFM signal (own past return)",
                    scale=alt.Scale(domain=[-lim, lim])),
            y=alt.Y("s_cs:Q", title="CSFM signal (relative to peers)",
                    scale=alt.Scale(domain=[-lim, lim])),
            color=alt.Color("agree:N", title=None,
                            scale=alt.Scale(domain=["both long", "both short",
                                                    "disagree"],
                                            range=[THEME.teal, THEME.coral,
                                                   THEME.mustard])),
            tooltip=["factor", alt.Tooltip("s_ts:Q", format="+.3f"),
                     alt.Tooltip("s_cs:Q", format="+.3f"),
                     alt.Tooltip("r_1m:Q", title="1m return", format="+.2%")],
        )
        text = alt.Chart(df).mark_text(dy=-12, fontSize=10,
                                       color=THEME.text).encode(
            x="s_ts:Q", y="s_cs:Q", text="factor:N")
        diag = alt.Chart(pd.DataFrame({"x": [-lim, lim], "y": [-lim, lim]})
                         ).mark_line(strokeDash=[4, 4], color=THEME.muted,
                                     opacity=0.5).encode(x="x:Q", y="y:Q")
        rules = (alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
                     color=THEME.grid).encode(x="v:Q") +
                 alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
                     color=THEME.grid).encode(y="v:Q"))
        st.altair_chart((rules + diag + pts + text).properties(height=430),
                        use_container_width=True)
        st.caption("On the dashed diagonal the two lenses agree exactly. "
                   "Off-diagonal mustard points are factors the time-series "
                   "model and the cross-sectional model treat differently — "
                   "usually months when most factors moved the same way.")
    except Exception:
        st.dataframe(df, hide_index=True)


def _ranked_table(ts_rows: list[dict], cs_rows: list[dict], key: str) -> None:
    cs_by_factor = {r["factor"]: r for r in cs_rows}
    recs = []
    for r in sorted(ts_rows, key=lambda x: x["rank"]):
        c = cs_by_factor.get(r["factor"], {})
        recs.append({
            "Rank": r["rank"],
            "Factor": r["factor"],
            "Proxy": r.get("proxy") or "—",
            "1m return": r["r_1m"], "De-meaned": r["r_1m_dm"],
            "Vol (ann)": r["vol_ann"],
            "TSFM s": r["s"], "CSFM s": c.get("s"),
            "Weight": r["weight"], "Leg": r["leg"],
            "Decile": _decile_mark(r["decile"]),
            "Source month": r.get("source_month", ""),
        })
    df = pd.DataFrame(recs)
    try:
        sty = (df.style
               .map(lambda v: _grad_diverging(v, 0.06),
                    subset=["1m return", "De-meaned"])
               .map(lambda v: _grad_diverging(v, 2.0),
                    subset=["TSFM s", "CSFM s"])
               .map(lambda v: _grad_diverging(v, 1.0), subset=["Weight"])
               .format({"1m return": _fmt_pct, "De-meaned": _fmt_pct,
                        "Vol (ann)": lambda v: "" if pd.isna(v) else f"{v:.1%}",
                        "TSFM s": "{:+.3f}", "CSFM s": "{:+.3f}",
                        "Weight": "{:+.3f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(640, 45 + 35 * len(df)),
                     column_config=_colcfg(df.columns), key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _perf_heatmap(recent: list[dict], order: list[str]) -> None:
    df = pd.DataFrame(recent)
    if df.empty:
        st.caption("Heatmap appears after the first compute run.")
        return
    cap = max(0.02, float(df["ret"].abs().quantile(0.95)))
    try:
        import altair as alt
        n = df["factor"].nunique()
        ch = (alt.Chart(df).mark_rect().encode(
            x=alt.X("d:O", title=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
            y=alt.Y("factor:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=0, labelFontSize=11)),
            color=alt.Color("ret:Q",
                            scale=alt.Scale(scheme="redyellowgreen",
                                            domain=[-cap, cap], clamp=True),
                            legend=alt.Legend(title="monthly ret", format="+.1%")),
            tooltip=["factor", "d", alt.Tooltip("ret:Q", format="+.2%")],
        ).properties(height=max(240, HM_PX * n)))
        st.altair_chart(ch, use_container_width=True)
        st.caption("Rows are sorted by the current TSFM signal — momentum says "
                   "the green rows near the top should stay green next month.")
    except Exception:
        st.dataframe(df.pivot(index="factor", columns="d", values="ret"))


def _backtest_section(bt: dict, key: str = "fmom_bt") -> None:
    series = pd.DataFrame(bt.get("series") or [])
    if series.empty:
        st.caption("Backtest appears after the first compute run.")
        return
    series["d"] = pd.to_datetime(series["d"])
    logscale = st.toggle("log scale", value=False, key=f"{key}_log")
    long = series.melt("d", value_vars=[c for c in ("tsfm", "csfm", "ew")
                                        if c in series],
                       var_name="model", value_name="ret").dropna()
    long["growth"] = (long.sort_values("d").groupby("model")["ret"]
                      .transform(lambda r: (1.0 + r).cumprod()))
    label = {"tsfm": "TSFM (1-1)", "csfm": "CSFM (1-1)",
             "ew": "EW all factors (untimed)"}
    long["model"] = long["model"].map(label)
    try:
        import altair as alt
        hover = alt.selection_point(fields=["d"], nearest=True,
                                    on="mouseover", empty=False)
        lines = alt.Chart(long).mark_line(strokeWidth=2.5,
                                          interpolate="monotone").encode(
            x=alt.X("d:T", title=None),
            y=alt.Y("growth:Q", title="growth of $1",
                    scale=alt.Scale(type="log" if logscale else "linear")),
            color=alt.Color("model:N", title=None,
                            scale=alt.Scale(domain=list(label.values()),
                                            range=[THEME.teal, THEME.navy,
                                                   THEME.muted])),
            tooltip=["model", alt.Tooltip("d:T", title="month"),
                     alt.Tooltip("growth:Q", format=".3f", title="$1 grew to"),
                     alt.Tooltip("ret:Q", format="+.2%", title="month")],
        )
        pts = lines.mark_point(size=60).transform_filter(hover)
        rule = alt.Chart(long).mark_rule(color=THEME.grid).encode(
            x="d:T").add_params(hover).transform_filter(hover)
        st.altair_chart((lines + pts + rule).properties(height=380),
                        use_container_width=True)
    except Exception:
        st.line_chart(long.pivot_table(index="d", columns="model",
                                       values="growth"))

    stats = bt.get("stats") or {}
    cols = st.columns(3)
    for col, key in zip(cols, ("tsfm", "csfm", "ew")):
        s = stats.get(key) or {}
        with col:
            st.markdown(f"**{label.get(key, key)}**")
            if s.get("sharpe") is None:
                st.caption("insufficient history")
                continue
            a, b = st.columns(2)
            a.metric("Sharpe", f"{s['sharpe']:.2f}",
                     help="Annualized mean / vol of monthly model returns. "
                          "Paper: 1-1 TSFM ≈ 0.84 on 65 academic factors.")
            b.metric("Ann. return", _fmt_pct(s.get("ann_ret")))
            a.metric("Max drawdown", _fmt_pct(s.get("max_dd")).lstrip("+"))
            b.metric("Hit rate", f"{s.get('hit_rate', 0):.0%}",
                     help="Share of months the model made money.")
    st.caption(f"Backtest {bt.get('start')} → {bt.get('end')} on the same "
               "panels the live signals use — published/ETF returns only, no "
               "survivorship-biased reconstruction. Gross of costs.")


def _tracker(hist: dict, family: str) -> None:
    rows = hist.get("rows", [])
    if not rows:
        st.caption("No tracked months yet. Run "
                   "`python -m zenith.fmom.compute --action backfill`.")
        return
    summary = summarize(rows)
    models = summary.get("models", {})
    if models:
        recs = [{"Model": MODEL_LABEL.get(m, m),
                 "Months": v["n_months"], "Live": v["n_live"],
                 "Hit rate": v["hit_rate"], "Avg month": v["avg_ret"],
                 "Cumulative": v["cum_ret"], "Through": v["last_month"]}
                for m, v in models.items()]
        df = pd.DataFrame(recs)
        try:
            sty = (df.style
                   .map(lambda v: _grad_diverging(v, 0.02), subset=["Avg month"])
                   .map(lambda v: _grad_diverging(v, 1.0), subset=["Cumulative"])
                   .map(lambda v: _grad(v, 1.0, (46, 196, 182)),
                        subset=["Hit rate"])
                   .format({"Hit rate": "{:.0%}", "Avg month": _fmt_pct,
                            "Cumulative": _fmt_pct}))
            st.dataframe(sty, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)

    curve = pd.DataFrame(summary.get("curve") or [])
    fam_models = [m for m in MODELS if family_of(m) == family]
    fam_curve = curve[curve["model"].isin(fam_models)] if not curve.empty else curve
    if not fam_curve.empty:
        # trailing window only — the deep history lives in the backtest above,
        # and 250 ordinal month labels would squeeze the axis into mush
        months_avail = sorted(fam_curve["month"].unique())
        window = months_avail[-36:]
        fam_curve = fam_curve[fam_curve["month"].isin(window)].copy()
        fam_curve["growth"] = (fam_curve.sort_values("month")
                               .groupby("model")["ret"]
                               .transform(lambda r: (1.0 + r).cumprod()))
        fam_curve["label"] = fam_curve["model"].map(MODEL_LABEL)
        if len(months_avail) > 36:
            st.caption(f"Showing the trailing 36 of {len(months_avail)} tracked "
                       "months (growth of $1 restarts at the window start); the "
                       "full-history curve is in the backtest section above.")
        try:
            import altair as alt
            ch = (alt.Chart(fam_curve).mark_line(strokeWidth=2.5,
                                                 point=True).encode(
                x=alt.X("month:O", title=None,
                        axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                y=alt.Y("growth:Q", title="growth of $1 following the model"),
                color=alt.Color("label:N", title=None,
                                scale=alt.Scale(range=[THEME.teal, THEME.navy])),
                strokeDash=alt.StrokeDash("backfilled:N",
                                          title="backfilled",
                                          scale=alt.Scale(domain=[False, True],
                                                          range=[[1, 0], [5, 4]])),
                tooltip=["label", "month",
                         alt.Tooltip("ret:Q", format="+.2%", title="month ret"),
                         alt.Tooltip("cum:Q", format="+.1%", title="cumulative"),
                         "backfilled"],
            ).properties(height=330))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.dataframe(fam_curve, hide_index=True)

        bars = fam_curve[fam_curve["month"].isin(window[-24:])].assign(
            pct=lambda d: d["ret"] * 100)
        try:
            import altair as alt
            ch = (alt.Chart(bars).mark_bar().encode(
                x=alt.X("month:O", title=None,
                        axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                xOffset="label:N",
                y=alt.Y("pct:Q", title="model return per month (%)"),
                color=alt.Color("label:N", title=None,
                                scale=alt.Scale(range=[THEME.teal, THEME.navy])),
                tooltip=["label", "month",
                         alt.Tooltip("pct:Q", format="+.2f", title="%")],
            ).properties(height=260))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            pass

    pending = summary.get("pending_months") or []
    if pending:
        st.caption(f"Pending evaluation (next month's factor returns not "
                   f"published yet): {', '.join(pending)}.")


def _buy_view(ts_block: dict, catalog: dict) -> None:
    """Long leg translated into plain long ETF positions."""
    rows = [r for r in ts_block.get("rows", []) if r["weight"] > 0 and r.get("proxy")]
    if not rows:
        st.caption("The model holds no longs this month (degenerate month).")
        return
    total = sum(r["weight"] for r in rows)
    cat = {e["ticker"]: e for e in (catalog.get("etfs") or [])}
    recs = []
    for r in sorted(rows, key=lambda x: -x["weight"]):
        c = cat.get(r["proxy"], {})
        recs.append({"Factor": r["factor"], "Ticker": r["proxy"],
                     "Long-only weight": r["weight"] / total,
                     "Signal": r["s"], "Decile": _decile_mark(r["decile"]),
                     "AUM $m": c.get("aum_m"), "Exp. ratio": c.get("er"),
                     "Category": c.get("category", "")})
    df = pd.DataFrame(recs)
    try:
        sty = (df.style
               .map(lambda v: _grad(v, 0.35, (46, 196, 182)),
                    subset=["Long-only weight"])
               .format({"Long-only weight": "{:.1%}", "Signal": "{:+.3f}",
                        "AUM $m": lambda v: "" if v is None or pd.isna(v) else f"{v:,.0f}",
                        "Exp. ratio": lambda v: "" if v is None or pd.isna(v) else f"{v:.2f}%"}))
        st.dataframe(sty, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)
    shorts = [f"{r['factor']} ({r['proxy']})" for r in ts_block.get("rows", [])
              if r["weight"] < 0]
    if shorts:
        st.caption("Short leg (avoid/underweight rather than short, if "
                   "long-only): " + " · ".join(shorts))


# ------------------------------------------------------------ stock finder --
def _screen_table(screen: dict, leg: str, key: str) -> None:
    rows = screen.get(leg) or []
    if not rows:
        st.caption("no names")
        return
    df = pd.DataFrame([{"Ticker": r["ticker"], "Name": r["name"],
                        screen["label"]: r["value"],
                        "Mkt cap $B": r.get("mktcap_b")} for r in rows])
    try:
        sty = df.style.format({screen["label"]: "{:.4g}",
                               "Mkt cap $B": lambda v: "" if v is None or pd.isna(v)
                               else f"{v:,.1f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(420, 45 + 35 * len(df)), key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key)


def _stock_finder(fam: str, ts_rows: list[dict], cs_rows: list[dict],
                  show_all: bool) -> None:
    """Per-factor: definition, screen recipe, and (when computable) the actual
    Russell 1000 names ranking highest on each side of the factor."""
    screens = load("screens", {})
    osap_cat = load("osap_catalog", {}) if fam == "osap" else {}
    cs_by = {r["factor"]: r for r in cs_rows}
    etf_holdings = screens.get("etf_holdings") or {}

    sort_label = st.selectbox("Sort factors by", list(SORT_MODES),
                              key=f"finder_sort_{fam}")
    ordered = SORT_MODES[sort_label](list(ts_rows))
    shown = ordered if (show_all or len(ordered) <= TRIM_AT) else ordered[:30]
    if len(shown) < len(ordered):
        st.caption(f"Showing the first {len(shown)} of {len(ordered)} factors "
                   "under this sort — flip 'show all factors' above to expand.")
    if screens.get("as_of"):
        st.caption(f"Stock screens as of {screens['as_of']} on the "
                   f"{(screens.get('universe') or {}).get('n', '?')}-name "
                   "Russell 1000 (current constituents — a screen, not advice).")
    else:
        st.caption("Run `python -m zenith.fmom.compute --action replicate` to "
                   "build the stock screens.")

    for r in shown:
        info = _factor_info(fam, r["factor"], osap_cat)
        cs = cs_by.get(r["factor"], {})
        side = "LONG" if r["s"] > 0 else "SHORT"
        title = (f"{_decile_mark(r['decile'])} {r['factor']}"
                 + (f" — {info['name']}" if info.get("name") not in (None, r["factor"]) else "")
                 + f"  ·  {side} s={r['s']:+.2f}")
        with st.expander(title):
            a, b, c = st.columns(3)
            a.metric("TSFM s", f"{r['s']:+.2f}",
                     help="Previous-month return / trailing vol, capped ±2.")
            b.metric("CSFM s", f"{cs.get('s'):+.2f}" if cs.get("s") is not None
                     else "—", help="Same signal after de-meaning across factors.")
            c.metric("1m return", _fmt_pct(r.get("r_1m")),
                     help=f"Formation month: {r.get('source_month', '?')}.")
            if info.get("definition"):
                st.markdown(f"**What it is:** {info['definition']}")
            if info.get("recipe"):
                st.markdown(f"**How to screen for it:** {info['recipe']}")
            ck = info.get("char")
            screen = (screens.get("screens") or {}).get(ck) if ck else None
            if screen:
                meta = CHAR_META.get(ck, {})
                st.markdown(f"**Russell 1000 names on this factor** "
                            f"({meta.get('label', ck)} — {meta.get('desc', '')})")
                lcol, scol = st.columns(2)
                with lcol:
                    st.markdown("*Fits the LONG side*")
                    _screen_table(screen, "long", key=f"scr_{fam}_{r['factor']}_l")
                with scol:
                    st.markdown("*Fits the SHORT side*")
                    _screen_table(screen, "short", key=f"scr_{fam}_{r['factor']}_s")
            elif ck:
                st.caption("Screen not built yet — run the replicate action.")
            else:
                st.caption("No single computable characteristic for this factor "
                           "— use the recipe above to screen manually.")
            if info.get("replicated") and (load("holdings", {}).get("factors")
                                           or {}).get(r["factor"]):
                st.caption("Full six-bin holdings for this factor are in the "
                           "'Factor drill-down & holdings' section above.")
            proxy = info.get("proxy")
            th = etf_holdings.get(proxy) if proxy else None
            if th:
                st.markdown(f"*Top holdings of the {proxy} proxy:* " + " · ".join(
                    f"{h['ticker']} ({h['weight']:.1%})" for h in th[:10]))


# ------------------------------------------------------------- aqr sections --
def _aqr_coverage_panel() -> None:
    with st.expander("Factor coverage — all 65, and where each one comes from"):
        recs = []
        for r in AQR65:
            src = {"sort": f"French sorts ({r['ds']})",
                   "factor": f"French factor ({r['ds']})",
                   "aqr": f"AQR workbook ({r['aqr_key']})"}.get(
                       r["live_source"], "OSAP / backtest-only")
            recs.append({"Abbr": r["abbr"], "Factor": r["name"],
                         "Theme": r["family"],
                         "Live source": src if r["live_source"] else "—",
                         "Replicated (R1000)": "✓" if r["yf_char"] else "—",
                         "Note": r.get("note") or ""})
        st.dataframe(pd.DataFrame(recs), use_container_width=True,
                     hide_index=True, height=45 + 35 * len(recs))
        st.caption("Live = a free source that updates monthly (Ken French "
                   "publishes with a ~1-2 month lag; AQR monthly workbooks). "
                   "Factors needing point-in-time accounting data have no free "
                   "live feed — they appear in the OSAP deep-history backtest "
                   "and, where marked, in the Russell 1000 replication layer.")


def _aqr_drilldown(ts_rows: list[dict], backtest: dict) -> None:
    holdings = load("holdings", {})
    per_factor = backtest.get("per_factor") or {}
    live = [r["factor"] for r in ts_rows]
    replicated = list((holdings.get("factors") or {}).keys())
    options = sorted(set(live) | set(replicated))
    if not options:
        st.caption("Run the compute action to populate the drill-down.")
        return
    pick = st.selectbox("Factor", options, key="fmom_aqr_drill",
                        format_func=lambda a: f"{a} — {AQR_NAME.get(a, a)}")
    row = AQR_ROW.get(pick, {})
    sig = next((r for r in ts_rows if r["factor"] == pick), None)
    pf = per_factor.get(pick) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current TSFM s", f"{sig['s']:+.2f}" if sig else "—",
              help="No live monthly source for this factor." if not sig else
                   "Previous-month return / trailing vol, capped ±2.")
    c2.metric("AR(1)", f"{pf['ar1']:+.3f}" if pf.get("ar1") is not None else "—",
              help="Monthly return autocorrelation — the persistence factor "
                   "momentum harvests. Paper average: +0.11.")
    c3.metric("AR(1) t-stat", f"{pf['t_stat']:+.2f}"
              if pf.get("t_stat") is not None else "—",
              help="|t| > 1.96 = significant at 5%. The paper found 49 of 65 "
                   "significant.")
    c4.metric("Months of history", pf.get("n_months") or "—")
    if row.get("note"):
        st.caption(f"Note: {row['note']}.")

    recent = [r for r in (backtest.get("recent_panel") or [])
              if r["factor"] == pick]
    if recent:
        df = pd.DataFrame(recent)
        try:
            import altair as alt
            ch = (alt.Chart(df).mark_bar().encode(
                x=alt.X("d:O", title=None,
                        axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                y=alt.Y("ret:Q", title="monthly long-short return",
                        axis=alt.Axis(format="+.0%")),
                color=alt.condition("datum.ret >= 0", alt.value(THEME.teal),
                                    alt.value(THEME.coral)),
                tooltip=["d", alt.Tooltip("ret:Q", format="+.2%")],
            ).properties(height=180))
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.dataframe(df, hide_index=True)

    hf = (holdings.get("factors") or {}).get(pick)
    if hf:
        st.markdown(f"**Current Russell 1000 holdings** — characteristic "
                    f"`{hf['char']}`, long the {hf['long_side']} side, "
                    f"coverage {hf['coverage']:.0%} of the universe "
                    f"(as of {holdings.get('as_of', '?')})")
        long_tab, short_tab = st.columns(2)
        for col, keys, title in ((long_tab, hf["long_bins"], "LONG bins"),
                                 (short_tab, hf["short_bins"], "SHORT bins")):
            with col:
                st.markdown(f"*{title}* — {' + '.join(keys)}")
                rows = []
                for k in keys:
                    for m in (hf["bins"].get(k) or [])[:25]:
                        rows.append({"Bin": k, "Ticker": m["ticker"],
                                     "Name": m.get("name", ""),
                                     "Weight in bin": m["w"],
                                     "Characteristic": m["char"]})
                if rows:
                    df = pd.DataFrame(rows)
                    try:
                        sty = (df.style
                               .map(lambda v: _grad(v, 0.08, (46, 196, 182)),
                                    subset=["Weight in bin"])
                               .format({"Weight in bin": "{:.2%}",
                                        "Characteristic": "{:.4g}"}))
                        st.dataframe(sty, use_container_width=True,
                                     hide_index=True, height=420)
                    except Exception:
                        st.dataframe(df, use_container_width=True,
                                     hide_index=True, height=420)
                else:
                    st.caption("empty bin")
        st.caption("Value-weighted within each bin (top 25 names shown per "
                   "bin). Factor = 0.5×(LargeHigh + SmallHigh) − "
                   "0.5×(LargeLow + SmallLow). Holdings only — this layer "
                   "produces no backtest (current membership would bias one).")
    elif row.get("yf_char"):
        st.caption("Replicable factor — run `python -m zenith.fmom.compute "
                   "--action replicate` to build current holdings.")


# ------------------------------------------------------------------ render --
def render() -> None:
    st.caption(DISCLAIMER)
    from ..ui_theme import key_findings
    st.markdown(key_findings([
        {"stat": "Equity factors persist month to month: 59 of 65 academic "
                 "factors had positive return autocorrelation, 49 of them "
                 "statistically significant.",
         "cite": "Gupta & Kelly (2019)"},
        {"stat": "Buy last month's winning factors, sell the losers, rebuild "
                 "monthly: 0.84 Sharpe in the paper — 0.75 in our free-data "
                 "AQR/French replication (1926–2026), the honest gap shown in "
                 "the backtest tab.",
         "cite": "Gupta & Kelly (2019) · this app's replication"},
        {"stat": "Factor momentum sidestepped the 2009 momentum crash (+18% "
                 "while stock momentum lost 31%) and is a rare diversifier "
                 "(~0.05 correlation to index trend-following).",
         "cite": "Gupta & Kelly (2019) · Man AHL (2026)"},
    ]), unsafe_allow_html=True)
    signals = load("signals", {})
    if not signals or not signals.get("models"):
        st.info("No signals yet. Run `python -m zenith.fmom.compute --action "
                "backfill` then `--action auto` (or wait for the monthly Action).")
        return
    st.markdown(stamp(signals.get("as_of", "?"), "FACTOR MOMENTUM"),
                unsafe_allow_html=True)
    _freshness_chips(signals)

    with st.expander("How factor momentum works — the research in 60 seconds"):
        st.markdown(
            "**The finding** (Gupta & Kelly 2019, *Factor Momentum Everywhere*): "
            "equity factors are strongly persistent month to month — 59 of their "
            "65 academic factors had positive return autocorrelation. A strategy "
            "that simply buys factors that made money LAST month and sells those "
            "that lost — rebuilt monthly — earned a **0.84 Sharpe**, beating stock "
            "momentum, industry momentum and every individual factor. It also "
            "sidestepped the 2009 momentum crash (+18% while stock momentum lost "
            "31%).\n\n"
            "**The signal (1-1)**: each factor's previous-month return is divided "
            "by its trailing 3-year volatility and capped at ±2 — that's `s` in "
            "every chart below. Positive `s` → long next month, negative → short, "
            "bigger |s| → bigger weight.\n\n"
            "**TSFM vs CSFM**: the time-series model (TSFM) judges every factor "
            "against ITSELF — all factors can be long at once. The cross-"
            "sectional model (CSFM) first subtracts the average factor return "
            "that month, so it is always long the relative winners and short the "
            "relative losers. The paper finds them >0.90 correlated, with TSFM "
            "the purer signal.\n\n"
            "**Portfolio**: signal-weighted, $1 long / $1 short, held one month, "
            "reformed monthly. Man AHL (2026) shows this style-momentum family "
            "is a rare diversifier: ~0.05 correlation to index trend-following, "
            "positive in 4 of its 5 worst drawdowns.")

    models = signals["models"]
    fams = sorted({family_of(k) for k in models},
                  key=lambda f: list(FAMILY_META).index(f))
    labels = {f: FAMILY_META[f][0] for f in fams}
    fam = st.selectbox("Factor family", fams,
                       format_func=lambda f: labels[f], key="fmom_family")
    st.caption(FAMILY_META[fam][1])

    ts = models.get(f"{fam}_tsfm") or {}
    cs = models.get(f"{fam}_csfm") or {}
    ts_rows, cs_rows = ts.get("rows", []), cs.get("rows", [])
    if not ts_rows:
        st.info("This family has no computed signals yet.")
        return
    annual = ts.get("annual", False)

    # --- KPI row -------------------------------------------------------------
    top = ts_rows[0]
    bottom = ts_rows[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data vintage" if annual else "Formation month",
              ts.get("formation_month", "?"),
              help=("This dataset updates about once a year; signals form on "
                    "its latest published month. Gupta & Kelly find factor "
                    "momentum persists at 1-60 month formation horizons, so an "
                    "older signal still carries documented information.")
              if annual else
              "The single month whose returns form the current 1-1 signal. "
              "Positions notionally hold through the following month.")
    c2.metric("Factors", ts.get("n_factors", 0),
              help="Factors with enough history for a signal (vol needs 24+ "
                   "months).")
    c3.metric("Long / short (TSFM)", f"{ts.get('n_long', 0)} / {ts.get('n_short', 0)}",
              help="TSFM can be long everything after a broad up month — "
                   "CSFM is always half long, half short.")
    c4.metric("Strongest", top["factor"],
              help=f"Highest signal: s = {top['s']:+.2f} "
                   f"(1m return {top['r_1m']:+.1%}).")
    c5.metric("Weakest", bottom["factor"],
              help=f"Lowest signal: s = {bottom['s']:+.2f} "
                   f"(1m return {bottom['r_1m']:+.1%}).")
    if ts.get("degenerate") or cs.get("degenerate"):
        st.caption("⚠ one-sided month: every signal shares the same sign in at "
                   "least one lens, so that model runs long-only (or short-only) "
                   "this month.")

    # --- signal bars ----------------------------------------------------------
    show_all = False
    if len(ts_rows) > TRIM_AT:
        show_all = st.toggle(f"show all {len(ts_rows)} factors "
                             f"(default: strongest {TRIM_N} each side)",
                             value=False, key=f"fmom_showall_{fam}")
    ts_view, trimmed = _trim_rows(ts_rows, show_all)
    cs_view, _ = _trim_rows(cs_rows, show_all)
    st.markdown(section("Current signals — the 1-1 formation", 1,
                        help="Gupta & Kelly Eq. 1 on the latest completed "
                             "month. Mustard outline = top/bottom decile. "
                             "Dashed rules = the ±2 signal cap."),
                unsafe_allow_html=True)
    if trimmed:
        st.caption(f"Charts show the {TRIM_N} strongest longs + {TRIM_N} "
                   f"strongest shorts of {len(ts_rows)} factors; the ranked "
                   "table and stock finder cover everything.")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**TSFM** — vs its own history "
                    f"({ts.get('n_long', 0)}L / {ts.get('n_short', 0)}S)")
        _signal_bars(ts_view, "tsfm", key="bars_ts")
    with right:
        st.markdown(f"**CSFM** — vs the other factors "
                    f"({cs.get('n_long', 0)}L / {cs.get('n_short', 0)}S)")
        _signal_bars(cs_view, "csfm", key="bars_cs")

    # --- lens comparison -------------------------------------------------------
    st.markdown(section("Two lenses, one month", 2,
                        help="Each factor's TSFM signal against its CSFM "
                             "signal."), unsafe_allow_html=True)
    _lens_scatter(ts_view, cs_view)

    # --- ranked table ----------------------------------------------------------
    st.markdown(section("All factors, ranked", 3,
                        help="Full signal table — every factor, sortable. "
                             "★ / ▼ mark the top and bottom signal deciles. "
                             "Click a column header to sort."),
                unsafe_allow_html=True)
    _ranked_table(ts_rows, cs_rows, key=f"rank_{fam}")

    # --- stock finder ----------------------------------------------------------
    st.markdown(section("Find the stocks behind a factor", 4,
                        help="Sort factors by signal strength, open one, and "
                             "see what it means, how to screen for it, and — "
                             "where the characteristic is computable — the "
                             "actual Russell 1000 names on each side."),
                unsafe_allow_html=True)
    _stock_finder(fam, ts_rows, cs_rows, show_all)

    memberships = ((signals.get("sources") or {}).get(fam) or {}).get("members")
    if memberships:
        with st.expander("Style composite membership — which ETFs make each style"):
            cat = {e["ticker"]: e for e in (load("etf_catalog", {}).get("etfs") or [])}
            recs = [{"Style": style, "Members": len(tks),
                     "ETFs": " · ".join(
                         f"{t} ({(cat.get(t, {}).get('name') or t)[:32]})"
                         for t in tks)}
                    for style, tks in memberships.items()]
            st.dataframe(pd.DataFrame(recs), use_container_width=True,
                         hide_index=True)
            st.caption("Composites are equal-weight across live members each "
                       "month (3+ required), minus SPY. 'Principle-based' maps "
                       "to Morningstar's committee/rules/earnings/liquidity "
                       "screens — the closest free-data stand-in for Man's "
                       "category.")

    # --- performance heatmap ---------------------------------------------------
    backtest = (load("backtest", {}).get("families") or {}).get(fam) or {}
    st.markdown(section("Factor performance — trailing 24 months", 5,
                        help="Monthly factor returns, rows sorted by current "
                             "TSFM signal (strongest at top)."),
                unsafe_allow_html=True)
    order = [r["factor"] for r in ts_view]
    shown_set = set(order)
    recent = [r for r in (backtest.get("recent_panel") or [])
              if r["factor"] in shown_set]
    _perf_heatmap(recent, order)

    # --- backtest ---------------------------------------------------------------
    st.markdown(section("Backtest — TSFM vs CSFM vs untimed", 5,
                        help="Growth of $1 walking the panel monthly with no "
                             "look-ahead: signals at month-end, held one month."),
                unsafe_allow_html=True)
    _backtest_section(backtest, key=f"bt_{fam}")
    if fam == "osap":
        st.caption("The OSAP backtest uses the PUBLISHED long-short returns "
                   "(which include delisted stocks — no survivorship bias). "
                   "OSAP models are not part of the monthly tracker below: the "
                   "annual data cadence would leave picks pending for a year.")

    # --- tracker ----------------------------------------------------------------
    st.markdown(section("Model tracker — every call scored", 0,
                        help="Each formation month's picks are stored, then "
                             "scored against the NEXT month's realized factor "
                             "returns. Dashed = backfilled, solid = live. "
                             "OSAP models are excluded (annual data)."),
                unsafe_allow_html=True)
    if fam == "osap":
        st.caption("OSAP models are not tracked monthly — the summary below "
                   "covers the three live families.")
    _tracker(load("history", {}), fam)

    # --- AQR coverage + drill-down ----------------------------------------------
    if fam == "aqr":
        st.markdown(section("Factor drill-down & holdings", 1,
                            help="Pick any factor: persistence stats, recent "
                                 "returns, and — for replicated factors — the "
                                 "actual Russell 1000 names in each bin."),
                    unsafe_allow_html=True)
        _aqr_coverage_panel()
        _aqr_drilldown(ts_rows, backtest)

    # --- buy view (ETF family) ---------------------------------------------------
    if fam == "etf":
        st.markdown(section("What would you actually buy", 1,
                            help="The TSFM long leg renormalized to a plain "
                                 "long-only ETF portfolio, with fund facts from "
                                 "the strategic-beta catalog."),
                    unsafe_allow_html=True)
        _buy_view(ts, load("etf_catalog", {}))

    # --- archive ------------------------------------------------------------------
    months = archive_months()
    if months:
        st.markdown(section("Open a past month", 2), unsafe_allow_html=True)
        pick = st.selectbox("Formation month", months, key="fmom_month")
        old = load_month(pick)
        old_ts = (old.get("models") or {}).get(f"{fam}_tsfm")
        if old_ts and pick != ts.get("formation_month"):
            _signal_bars(old_ts.get("rows", []), "tsfm", key=f"bars_{pick}")

    st.caption("Caveats: model returns are gross of trading costs, borrow and "
               "slippage; ETF families start at fund inception (~2013 for most "
               "single-factor ETFs) plus a 2-year vol warm-up; signals form on "
               "completed calendar months only. Long-short factor momentum is a "
               "research lens, not a portfolio recommendation.")

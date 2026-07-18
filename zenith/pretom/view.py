"""Streamlit rendering for the PRETOM tab — colorful, visual, Zenith-themed.

Reads only committed JSON (data/pretom/*). The countdown banner is computed
live from the pure trading calendar so it is never a day stale; artefact
freshness is shown separately via the 'data as of' stamp.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load, load_month, archive_months
from . import calendar as cal
from ..config import THEME
from ..ui_theme import section, stamp

_STATE_META = {
    "FORMING":     (THEME.mustard, "FORMING"),
    "LOCKED":      (THEME.teal, "BASKET LOCKED"),
    "WINDOW_OPEN": (THEME.coral, "WINDOW OPEN"),
    "POST":        (THEME.mauve, "POST / REVERSAL"),
    "CLOSED":      (THEME.muted, "CLOSED"),
}

COLS = {
    "Rank": "Composite rank inside the basket (1 = strongest short candidate).",
    "Score": "0-1 blend: 50% distance below 52w high, 30% liquidity, 20% index "
             "weight, +0.05 non-dividend-payer bonus.",
    "Below 52w high": "How far the last price sits under the trailing 252-day "
                      "high — the paper's strongest 'dispensability' sort.",
    "12-2 mom": "Classic momentum: cumulative return months -12 to -2. Shown "
                "for reference; the 52w-high sort is the primary signal.",
    "ADV $m": "63-day average daily dollar volume (millions). Liquid losers "
              "carry the selling pressure, so higher ranks UP.",
    "Weight %": "Weight in the Russell 1000 (VONE holdings) — proxy for how "
                "prevalent the name is across funds that may sell for cash.",
    "Div": "Paid a dividend in the trailing year? Non-payers are more "
           "'dispensable' (84% of the paper's loser decile are non-payers).",
    "Px @ lock": "Last close when the basket was locked.",
    "Classic EW xs": "Equal-weight basket return minus SPY over [tau-9, tau-4]. "
                     "Negative = the short thesis paid.",
    "T+1 EW xs": "Equal-weight excess return over the [tau-9, tau-3] span "
                 "(adds the post-2024 T+1 marginal selling day).",
    "Classic CW xs": "Cap-weighted (index-weight) excess return, classic window.",
    "T+1 CW xs": "Cap-weighted excess return, T+1 span.",
    "% neg end": "Share of basket names below their entry price at the window "
                 "close.",
    "% neg any": "Share of names that traded below entry at ANY close in the "
                 "window — the 'was there a shortable dip' rate.",
    "Post EW xs": "Equal-weight excess return over the post window "
                  "[tau-3, tau+3], where the paper finds the bounce.",
    "Reversal": "Fraction of the window loss recovered post-window "
                "(paper: ~0.7). High reversal = pressure, not fundamentals.",
}


# ---------------------------------------------------------------- helpers ---
def _grad(v, cap: float, color: tuple[int, int, int]) -> str:
    """Panel-black -> `color` ramp, saturating at |v| = cap."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    a = max(0.0, min(1.0, abs(float(v)) / cap))
    r, g, b = (int(11 + (c - 11) * a) for c in color)
    return f"background-color: rgb({r},{g},{b}); color: #fff;"


def _grad_coral(v, cap=0.6):
    return _grad(v, cap, (255, 90, 60))


def _grad_teal(v, cap=1.0):
    return _grad(v, cap, (46, 196, 182))


def _grad_diverging(v, cap=0.05):
    """Teal for gains, coral for losses (matches the Brief's convention)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return _grad_teal(v, cap) if float(v) >= 0 else _grad_coral(v, cap)


def _colcfg(cols) -> dict:
    return {c: st.column_config.Column(help=COLS[c]) for c in cols if c in COLS}


def _fmt_pct(v):
    return "" if v is None or pd.isna(v) else f"{v:+.1%}"


def _live_state() -> dict:
    return cal.state_for(date.today())


def _banner(state: dict) -> None:
    color, label = _STATE_META.get(state["state"], _STATE_META["CLOSED"])
    s = state["schedule"]
    k = state["tau_offset"]
    if state["state"] == "FORMING":
        lock_in = max(state["days_to_open"] - 1, 0)
        msg = (f"basket locks in {lock_in} trading day{'s' * (lock_in != 1)} "
               f"({s['lock_by']}) · window opens {s['win_start']}")
    elif state["state"] == "LOCKED":
        msg = (f"short window opens in {state['days_to_open']} trading "
               f"day{'s' * (state['days_to_open'] != 1)} ({s['win_start']})")
    elif state["state"] == "WINDOW_OPEN":
        if state["t1_tail"]:
            msg = (f"T+1 extension day (τ−3) · classic window closed "
                   f"{s['win_end_classic']} (τ−4) · T+1 close today")
        else:
            msg = (f"day {k + 10} of 6 · classic close {s['win_end_classic']} "
                   f"(τ−4) · T+1 close {s['win_end_t1']} (τ−3)")
    elif state["state"] == "POST":
        msg = f"reversal watch — the bounce window runs through {s['post_end']} (τ+3)"
    else:
        msg = f"next window opens {s['win_start']}"
    st.markdown(
        f'<div style="font-family:{THEME.font_display}; font-size:1.25rem; '
        f'letter-spacing:0.14em; text-transform:uppercase; color:#000; '
        f'background:linear-gradient(90deg, {color}, {THEME.panel}); '
        f'border-left:6px solid {color}; padding:0.45rem 0.9rem; '
        f'margin:0.2rem 0 0.8rem 0;">'
        f'<b>{label}</b> · {state["month"]} · {msg}</div>',
        unsafe_allow_html=True)


def today_badge() -> str | None:
    """Small alert chip surfaced on the TODAY tab when the window is near."""
    state = _live_state()
    stt, k = state["state"], state["tau_offset"]
    if stt == "LOCKED":
        text = (f"▲ PRETOM basket locked — short window opens in "
                f"{state['days_to_open']} trading day{'s' * (state['days_to_open'] != 1)}")
    elif stt == "WINDOW_OPEN":
        if state["t1_tail"]:
            text = "▲ PRETOM T+1 extension day (τ−3) — classic window closed"
        else:
            text = (f"▲ PRETOM WINDOW OPEN — day {k + 10} of 6 "
                    f"(classic closes in {state['days_to_close_classic']} td)")
    elif stt == "FORMING" and state["days_to_open"] <= 4:
        text = f"PRETOM window opens in {state['days_to_open']} trading days"
    elif stt == "POST":
        text = "PRETOM post-window — reversal watch"
    else:
        return None
    color = _STATE_META[stt][0]
    return (f'<div style="display:inline-block; font-family:{THEME.font_display}; '
            f'font-size:0.95rem; letter-spacing:0.1em; text-transform:uppercase; '
            f'color:{color}; border:1px solid {color}; padding:0.15rem 0.6rem; '
            f'margin:0 0 0.6rem 0;">{text} · see PRETOM tab</div>')


# ------------------------------------------------------------------ charts --
def _basket_table(names: list[dict], key: str) -> None:
    df = pd.DataFrame(names)
    if df.empty:
        st.caption("Empty basket.")
        return
    disp = pd.DataFrame({
        "Rank": df["rank"], "Ticker": df["ticker"], "Name": df["name"],
        "Sector": df["sector"], "Score": df["score"],
        "Below 52w high": df["pct_below_high"], "12-2 mom": df["mom_12_2"],
        "ADV $m": (df["adv_dollar"] / 1e6).round(1),
        "Weight %": df["weight_pct"],
        "Div": df["div_payer"].map({True: "✓", False: "—", None: "?"}),
        "Px @ lock": df["px_lock"],
    })
    try:
        sty = (disp.style
               .map(_grad_teal, subset=["Score"])
               .map(_grad_coral, subset=["Below 52w high"])
               .map(lambda v: _grad_diverging(v, 0.5), subset=["12-2 mom"])
               .format({"Below 52w high": _fmt_pct, "12-2 mom": _fmt_pct,
                        "Score": "{:.3f}", "Weight %": "{:.3f}",
                        "ADV $m": "{:,.1f}", "Px @ lock": "{:,.2f}"}))
        st.dataframe(sty, use_container_width=True, hide_index=True, height=520,
                     column_config=_colcfg(disp.columns), key=key)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True, height=520)


def _score_bar(names: list[dict], top: int = 20) -> None:
    df = pd.DataFrame(names[:top])
    if df.empty:
        return
    df["label"] = df["ticker"] + " · " + df["name"].str.slice(0, 28)
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("score:Q", title="composite short score",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=0)),
            color=alt.Color("score:Q", scale=alt.Scale(scheme="reds",
                                                       domain=[0.3, 1.0]), legend=None),
            tooltip=["ticker", "name", "sector",
                     alt.Tooltip("score:Q", format=".3f"),
                     alt.Tooltip("pct_below_high:Q", title="below 52w high",
                                 format=".1%"),
                     alt.Tooltip("weight_pct:Q", title="index weight %",
                                 format=".3f")],
        ).properties(height=max(200, 24 * len(df))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df[["ticker", "name", "score"]], hide_index=True)


def _dispensability_scatter(names: list[dict]) -> None:
    df = pd.DataFrame(names)
    if df.empty or df["adv_dollar"].isna().all():
        return
    df = df.assign(adv_m=df["adv_dollar"] / 1e6,
                   weight=pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.01))
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_circle(opacity=0.85).encode(
            x=alt.X("pct_below_high:Q", title="below 52-week high",
                    axis=alt.Axis(format="%")),
            y=alt.Y("adv_m:Q", title="avg daily $ volume ($m, log)",
                    scale=alt.Scale(type="log")),
            size=alt.Size("weight:Q", title="index weight",
                          scale=alt.Scale(range=[25, 600]), legend=None),
            color=alt.Color("score:Q", scale=alt.Scale(scheme="reds",
                                                       domain=[0.3, 1.0]),
                            legend=alt.Legend(title="score")),
            tooltip=["ticker", "name", "sector",
                     alt.Tooltip("pct_below_high:Q", format=".1%",
                                 title="below high"),
                     alt.Tooltip("adv_m:Q", format=",.0f", title="ADV $m"),
                     alt.Tooltip("score:Q", format=".3f")],
        ).properties(height=380))
        st.altair_chart(ch, use_container_width=True)
        st.caption("Up-and-right = deep below its high AND liquid — exactly where the "
                   "paper finds the month-end selling pressure lands. Bubble size = "
                   "index weight (fund prevalence).")
    except Exception:
        pass


def _window_lines(stats: dict) -> None:
    daily = (stats.get("t1", {}) or {}).get("daily") or \
            (stats.get("classic", {}) or {}).get("daily") or []
    if not daily:
        st.caption("Window return series appears once the window opens.")
        return
    df = pd.DataFrame(daily)
    long = df.melt("d", value_vars=[c for c in ("ew", "cw", "spy") if c in df],
                   var_name="series", value_name="ret")
    long["ret"] *= 100
    long["series"] = long["series"].map({"ew": "Basket (equal-weight)",
                                         "cw": "Basket (cap-weight)",
                                         "spy": "SPY"})
    try:
        import altair as alt
        ch = (alt.Chart(long).mark_line(interpolate="monotone", strokeWidth=2.5,
                                        point=True).encode(
            x=alt.X("d:T", title=None),
            y=alt.Y("ret:Q", title="% since window open"),
            color=alt.Color("series:N", legend=alt.Legend(title=None),
                            scale=alt.Scale(range=[THEME.coral, THEME.orange,
                                                   THEME.navy])),
            tooltip=["series", alt.Tooltip("d:T", title="day"),
                     alt.Tooltip("ret:Q", format="+.2f", title="% chg")],
        ).properties(height=330))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.line_chart(df.set_index("d"))


def _name_heatmap(panel: dict, names: list[dict], sched: dict, top: int) -> None:
    order = [n["ticker"] for n in names[:top] if n["ticker"] in panel]
    recs = []
    for t in order:
        d = pd.to_datetime(panel[t]["d"])
        c = pd.Series(panel[t]["c"], index=d, dtype=float)
        r = c.pct_change().dropna() * 100
        r = r[r.index >= pd.Timestamp(sched["win_start"])]
        recs += [{"ticker": t, "d": i.date().isoformat(), "ret": round(float(v), 2)}
                 for i, v in r.items()]
    if not recs:
        st.caption("Heatmap appears once the window opens.")
        return
    df = pd.DataFrame(recs)
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_rect().encode(
            x=alt.X("d:O", title=None),
            y=alt.Y("ticker:N", sort=order, title=None),
            color=alt.Color("ret:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                     domain=[-4, 4]),
                            legend=alt.Legend(title="daily %")),
            tooltip=["ticker", "d", alt.Tooltip("ret:Q", format="+.2f")],
        ).properties(height=max(220, 16 * len(order))))
        st.altair_chart(ch, use_container_width=True)
        st.caption("Daily % moves per basket name — red days are the short thesis "
                   "working. Columns after τ−3 are the reversal window.")
    except Exception:
        pass


def _marks_bar(marks: dict, names: list[dict]) -> None:
    rows = []
    for n in names:
        m = marks.get(n["ticker"])
        if m and m.get("t1_ret") is not None:
            rows.append({"label": n["ticker"], "ret": m["t1_ret"] * 100,
                         "min": (m.get("min_ret_in_window") or 0) * 100})
    if not rows:
        return
    df = pd.DataFrame(sorted(rows, key=lambda r: r["ret"]))
    try:
        import altair as alt
        ch = (alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("ret:Q", title="% return over the window (negative = short win)"),
            y=alt.Y("label:N", sort="x", title=None),
            color=alt.Color("ret:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                     domain=[-10, 10]), legend=None),
            tooltip=["label", alt.Tooltip("ret:Q", format="+.1f", title="window %"),
                     alt.Tooltip("min:Q", format="+.1f", title="worst mark %")],
        ).properties(height=max(220, 15 * len(df))))
        st.altair_chart(ch, use_container_width=True)
    except Exception:
        st.dataframe(df, hide_index=True)


# ----------------------------------------------------------------- tracker --
def _hist_frame(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        c, t1, p = r.get("classic") or {}, r.get("t1") or {}, r.get("post") or {}
        recs.append({
            "Month": r["month"], "N": r.get("n"),
            "Classic EW xs": c.get("ew_excess"), "Classic CW xs": c.get("cw_excess"),
            "T+1 EW xs": t1.get("ew_excess"), "T+1 CW xs": t1.get("cw_excess"),
            "% neg end": t1.get("pct_negative_end") or c.get("pct_negative_end"),
            "% neg any": t1.get("pct_negative_any") or c.get("pct_negative_any"),
            "Post EW xs": p.get("ew_excess"), "Reversal": p.get("reversal_ratio"),
            "Final": "✓" if r.get("final") else "…",
            "Backfilled": "✓" if r.get("backfilled") else "live",
        })
    return pd.DataFrame(recs)


def _tracker(history: dict) -> None:
    rows = [r for r in history.get("rows", []) if r.get("classic")]
    if not rows:
        st.caption("No tracked baskets yet.")
        return
    hf = _hist_frame(rows)
    final = hf[hf["Final"] == "✓"]

    def _mean(col):
        s = pd.to_numeric(final[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None

    mc, mt = _mean("Classic EW xs"), _mean("T+1 EW xs")
    hit = pd.to_numeric(final["Classic EW xs"], errors="coerce").dropna()
    hit_rate = float((hit < 0).mean()) if len(hit) else None
    # reversal ("share of the window loss recovered") only means something in
    # months where the window actually produced a loss to recover
    paid = final[pd.to_numeric(final["Classic EW xs"], errors="coerce") < 0]
    rev = pd.to_numeric(paid["Reversal"], errors="coerce").dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Months tracked", len(hf),
              help="Locked baskets with stats (live + backfilled).")
    c2.metric("Avg classic EW xs", _fmt_pct(mc),
              help=COLS["Classic EW xs"] + " Average across final months.")
    c3.metric("Avg T+1 EW xs", _fmt_pct(mt), help=COLS["T+1 EW xs"])
    c4.metric("Short hit-rate", _fmt_pct(hit_rate).lstrip("+") if hit_rate is not None else "",
              help="Share of months where the basket LAGGED SPY during the "
                   "classic window — i.e. the short paid.")
    c5.metric("Avg reversal", f"{rev.mean():.2f}" if len(rev) else "—",
              help=COLS["Reversal"])

    # monthly excess-return bars, classic vs T+1
    long = hf.melt("Month", value_vars=["Classic EW xs", "T+1 EW xs"],
                   var_name="window", value_name="xs").dropna()
    long["xs"] *= 100
    try:
        import altair as alt
        ch = (alt.Chart(long).mark_bar().encode(
            x=alt.X("Month:O", title=None),
            xOffset="window:N",
            y=alt.Y("xs:Q", title="basket excess return vs SPY (%)"),
            color=alt.Color("window:N",
                            scale=alt.Scale(range=[THEME.coral, THEME.mustard]),
                            legend=alt.Legend(title=None)),
            tooltip=["Month", "window", alt.Tooltip("xs:Q", format="+.2f")],
        ).properties(height=300))
        st.altair_chart(ch, use_container_width=True)
        st.caption("Bars BELOW zero = the basket lagged the market during the window "
                   "= the short bias paid that month.")
    except Exception:
        st.dataframe(long, hide_index=True)

    xs_cols = ["Classic EW xs", "Classic CW xs", "T+1 EW xs", "T+1 CW xs",
               "Post EW xs"]
    fmt = {c: _fmt_pct for c in xs_cols}
    fmt |= {c: (lambda v: "" if v is None or pd.isna(v) else f"{v:.0%}")
            for c in ("% neg end", "% neg any")}
    fmt["Reversal"] = lambda v: "" if v is None or pd.isna(v) else f"{v:.2f}"
    try:
        sty = (hf.style
               .map(lambda v: _grad_diverging(v, 0.05), subset=xs_cols)
               .map(lambda v: _grad_teal(v, 1.0), subset=["% neg end", "% neg any"])
               .format(fmt))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(560, 45 + 35 * len(hf)),
                     column_config=_colcfg(hf.columns))
    except Exception:
        st.dataframe(hf, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ render --
def render() -> None:
    st.caption(DISCLAIMER)
    state = _live_state()
    _banner(state)

    with st.expander("How PRETOM works — the research in 60 seconds"):
        st.markdown(
            "**The finding** (Nathan, Suominen & Tasa 2026): U.S. momentum returns "
            "concentrate in six trading days ending four days before month-end "
            "(**[τ−9, τ−4]**, τ = last trading day). $1 in the winners-minus-losers "
            "trade during only those days grew to **$18.78** over 1980–2025, vs "
            "$2.37 the whole rest of the month.\n\n"
            "**The mechanism**: institutions need settled cash at month-end and sell "
            "their most *dispensable* holdings — salient losers, non-dividend-payers, "
            "names with embedded tax losses. The hit lands on **losers** (−7.9 bps/day "
            "vs market), not winners, and concentrates in **liquid** losers where "
            "institutions actually trade.\n\n"
            "**The screen**: sorting by **distance below the 52-week high** captures "
            "dispensability even better than classic momentum ($45.72 vs $18.78 per "
            "$1) — so that's the primary rank here, blended with liquidity and index "
            "weight. The bottom decile of the Russell 1000 is locked as the short-bias "
            "basket at τ−10, before the window opens.\n\n"
            "**Two windows tracked**: the classic [τ−9, τ−4] and the [τ−9, τ−3] span — "
            "the May-2024 T+1 settlement change let sellers wait one day longer, "
            "shifting the marginal selling day from τ−4 to τ−3.\n\n"
            "**After the window**: ~70% of the underperformance reverses by τ+3 "
            "(it's price pressure, not news) — so the tracker also measures the "
            "bounce, and shorts are typically covered by τ−3/τ−4, not held.\n\n"
            "*Backfilled history uses today's Russell 1000 membership — delisted "
            "losers are missing, so historical stats are slightly flattered.*")

    basket = load("basket", {})
    if not basket:
        st.info("No basket yet. Run `python -m zenith.pretom.compute --action backfill` "
                "then `--action auto` (or wait for the scheduled Action).")
        return
    st.markdown(stamp(basket.get("stats", {}).get("updated",
                                                  basket.get("locked_at", "?")),
                      "PRETOM"), unsafe_allow_html=True)

    # --- current basket ------------------------------------------------------
    names = basket.get("names", [])
    uni_info = basket.get("universe", {})
    st.markdown(section(f"Short-bias basket — {basket.get('month', '?')} "
                        f"(locked {basket.get('locked_at', '?')})", 2,
                        help="Bottom decile of the Russell 1000 by distance below "
                             "the 52-week high, composite-ranked. Click any column "
                             "header to sort."), unsafe_allow_html=True)
    if names:
        df = pd.DataFrame(names)
        nonpayer = (df["div_payer"] == False).mean()  # noqa: E712
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Names", len(df), help="Basket size — one decile of the universe.")
        c2.metric("Median below high", f"−{df['pct_below_high'].median():.0%}",
                  help=COLS["Below 52w high"])
        c3.metric("Median ADV", f"${df['adv_dollar'].median() / 1e6:,.0f}m",
                  help=COLS["ADV $m"])
        c4.metric("Non-payers", _fmt_pct(nonpayer).lstrip("+"),
                  help=COLS["Div"] + " Paper's loser decile: 84% non-payers.")
        c5.metric("Universe coverage", _fmt_pct(uni_info.get("coverage")).lstrip("+"),
                  help="Share of the Russell 1000 with usable price history "
                       f"(source: {uni_info.get('source', '?')}).")
        if basket.get("locked_late"):
            st.caption("⚠ this basket locked later than τ−10 (a run was missed); "
                       "ranks use data through the late lock date.")

        _basket_table(names, key="basket_now")
        cA, cB = st.columns([1, 1])
        with cA:
            st.markdown(section("Top 20 composite shorts", 4), unsafe_allow_html=True)
            _score_bar(names)
        with cB:
            st.markdown(section("Dispensability map", 3,
                                help="The paper's two key axes: depth below the "
                                     "52-week high and liquidity."),
                        unsafe_allow_html=True)
            _dispensability_scatter(names)

    # --- live window monitor -------------------------------------------------
    stats = basket.get("stats", {})
    if stats.get("classic") or stats.get("t1"):
        st.markdown(section("Window monitor — basket vs SPY", 1,
                            help="Cumulative return since the window opened. The "
                                 "short thesis works when the basket lines sit "
                                 "BELOW SPY."), unsafe_allow_html=True)
        _window_lines(stats)
        c, t1, p = (stats.get("classic") or {}), (stats.get("t1") or {}), (stats.get("post") or {})
        if c.get("ew_excess") is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Classic EW xs", _fmt_pct(c.get("ew_excess")),
                      help=COLS["Classic EW xs"])
            m2.metric("T+1 EW xs", _fmt_pct(t1.get("ew_excess")),
                      help=COLS["T+1 EW xs"])
            m3.metric("% neg at any close", _fmt_pct(t1.get("pct_negative_any")
                                                     or c.get("pct_negative_any")).lstrip("+"),
                      help=COLS["% neg any"])
            m4.metric("Post reversal", f"{p['reversal_ratio']:.2f}"
                      if p.get("reversal_ratio") is not None else "—",
                      help=COLS["Reversal"])
        hm_top = st.slider("Heatmap depth (top N by rank)", 10, len(names) or 10,
                           min(30, len(names) or 10), 5, key="hm_top")
        _name_heatmap(basket.get("panel", {}), names, basket.get("schedule", {}),
                      hm_top)
        if basket.get("per_name_marks"):
            with st.expander("Per-name window returns (ranked)"):
                _marks_bar(basket["per_name_marks"], names)

    # --- performance tracker -------------------------------------------------
    st.markdown(section("Basket tracker — every month scored", 0,
                        help="Each locked basket's window and post-window results, "
                             "both windows, equal- and cap-weighted, vs SPY."),
                unsafe_allow_html=True)
    _tracker(load("history", {}))

    # --- archive browser ------------------------------------------------------
    months = archive_months()
    if months:
        st.markdown(section("Open a past basket", 5), unsafe_allow_html=True)
        pick = st.selectbox("Month", months, key="pretom_month")
        old = load_month(pick)
        if old and pick != basket.get("month"):
            _basket_table(old.get("names", []), key=f"basket_{pick}")
            _window_lines(old.get("stats", {}))
            if old.get("per_name_marks"):
                _marks_bar(old["per_name_marks"], old.get("names", []))

    st.caption("Caveats: backfilled baskets use current Russell 1000 membership "
               "(survivorship bias flatters recovery); prices are yfinance "
               "best-effort; weights are VONE ETF holdings. The window return is "
               "measured close-to-close from the day before the window opens. "
               "Not investment advice — shorting can lose more than 100%.")

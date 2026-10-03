"""STATS and READ sub-views of the EPHEMERIS tab."""

from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import help_badge, section
from . import START_BALANCE
from . import report as R
from . import stats as S

_GREY = "#555555"


def _fmt_pct(x, d=1, signed=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def _pp(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.1f}pp"


def _filters(d: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    with st.expander("FILTERS — mode, dates, timeframe, horizon, class, indicators, conviction", expanded=False):
        c1, c2, c3 = st.columns(3)
        modes = c1.multiselect("Mode", sorted(d["mode"].unique()), key="eph_f_mode")
        tfs = c2.multiselect("Timeframe", sorted(d["timeframe"].unique()), key="eph_f_tf")
        hs = c3.multiselect("Horizon", sorted(d["horizon"].unique()), key="eph_f_h")
        c4, c5, c6 = st.columns(3)
        cls = c4.multiselect("Asset class", sorted(d["asset_class"].dropna().unique()), key="eph_f_cls")
        ind = c5.multiselect("Indicator set", sorted(d["indicator_set"].unique()), key="eph_f_ind")
        conv = c6.multiselect("Conviction", ["Low", "Medium", "High"], key="eph_f_conv")
        lo, hi = d["ts"].min().date(), d["ts"].max().date()
        rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key="eph_f_dates")
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
    f = S.filter_df(d, modes=modes, timeframes=tfs, horizons=hs, classes=cls, indicator_sets=ind,
                    convictions=conv, start=start, end=end)
    label = " · ".join(x for x in [",".join(modes), ",".join(tfs), ",".join(map(str, hs)), ",".join(cls),
                                   ",".join(ind), ",".join(conv)] if x)
    return f, label


def _sigcol(v, z, small: bool) -> str:
    """Colour an edge only when it is statistically distinguishable from zero."""
    if small or z is None or not np.isfinite(z) or abs(z) < S.Z95:
        return _GREY if small else THEME.text
    return THEME.teal if v > 0 else THEME.coral


def _headline(h: dict, dd: dict) -> None:
    b, r = h["base"], h["rule"]
    small = h["n"] < S.MIN_N
    sig = lambda z: "" if not np.isfinite(z) else (" · significant" if abs(z) >= S.Z95 else " · not significant")
    st.markdown(uc.numeric_slab([
        {"label": "Hit rate", "value": _fmt_pct(h["hit"]),
         "sub": f"95% CI {_fmt_pct(h['hit_lo'], 0)}–{_fmt_pct(h['hit_hi'], 0)} · n={h['n']}",
         "color": _GREY if small else THEME.text},
        {"label": "Edge vs base rate", "value": _pp(b["edge"]),
         "sub": f"base {_fmt_pct(b['base'])} · p={b['p_value']:.2f}{sig(b['z'])}" if np.isfinite(b["z"]) else "—",
         "color": _sigcol(b["edge"], b["z"], small)},
        {"label": "Edge vs trend rule", "value": _fmt_pct(r["mean"], 2, True),
         "sub": f"per call, paired · p={r['p_value']:.2f}{sig(r['z'])}" if np.isfinite(r["z"]) else "per call",
         "color": _sigcol(r["mean"], r["z"], small)},
        {"label": "Expectancy", "value": f"{'+' if h['expectancy'] >= 0 else '−'}${abs(h['expectancy']):,.1f}",
         "sub": f"per call · total {'+' if h['total_pnl'] >= 0 else '−'}${abs(h['total_pnl']):,.0f}",
         "color": THEME.teal if h["expectancy"] >= 0 else THEME.coral},
    ], min_width=150), unsafe_allow_html=True)
    pf = h["profit_factor"]
    st.markdown(uc.numeric_slab([
        {"label": "Profit factor", "value": "∞" if not np.isfinite(pf) else f"{pf:.2f}"},
        {"label": "Avg R", "value": "—" if not np.isfinite(h["avg_r"]) else f"{h['avg_r']:+.2f}R",
         "sub": f"{h['n_r']} calls with a stop"},
        {"label": "Win / loss size", "value": "—" if not np.isfinite(h["win_loss"]) else f"{h['win_loss']:.2f}×"},
        {"label": "Max drawdown", "value": f"−${abs(dd['max_dd']):,.0f}",
         "sub": f"{dd['max_dd_pct'] * 100:.1f}% · longest {dd['longest_dd']} calls"},
        {"label": "Streaks", "value": f"{h['longest_win']}W / {h['longest_loss']}L", "sub": "longest"},
    ], min_width=130), unsafe_allow_html=True)
    st.caption(f"Benchmarks on these same charts — coin flip 50% (your band at n={h['n']}: "
               f"{_fmt_pct(h['coin'][0], 0)}–{_fmt_pct(h['coin'][1], 0)}) · always-long hit "
               f"{_fmt_pct(h['always_long_hit'])} · trend rule hit {_fmt_pct(h['rule_hit'])} on its "
               f"{h['rule_n']} trades. Skill = beating the base rate and the rule, not beating 50%.")


def _equity(eq: pd.DataFrame) -> None:
    if eq.empty:
        return

    def build(alt):
        base = alt.Chart(eq).encode(x=alt.X("i:Q", title="call #"))
        line = base.mark_line(color=THEME.teal, strokeWidth=2).encode(
            y=alt.Y("equity:Q", title="paper equity ($)", scale=alt.Scale(zero=False)),
            tooltip=["i", alt.Tooltip("equity:Q", format="$,.0f"), alt.Tooltip("ts:T", format="%Y-%m-%d")])
        start = alt.Chart(pd.DataFrame({"y": [START_BALANCE]})).mark_rule(color="#444", strokeDash=[4, 4]).encode(y="y:Q")
        resets = base.transform_filter("datum.reset").mark_rule(color=THEME.mustard).encode(
            tooltip=[alt.Tooltip("i:Q", title="reset before call")])
        top = (start + line + resets).properties(height=240)
        under = base.mark_area(color=THEME.coral, opacity=0.55).encode(
            y=alt.Y("dd_pct:Q", title="drawdown", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("dd_pct:Q", format=".1%")]).properties(height=110)
        return alt.vconcat(top, under).resolve_scale(x="shared")
    uc.render_chart(build, fallback=eq)


def _heatmap(t: pd.DataFrame) -> None:
    if t.empty:
        return
    t = t.copy()
    t["edge_pp"] = t["edge"] * 100
    t["label"] = t.apply(lambda r: f"{r['edge_pp']:+.0f}pp\nn={int(r['n'])}", axis=1)
    t["hz"] = t["horizon"].astype(int)
    cap = max(5.0, float(np.nanmax(np.abs(t["edge_pp"]))) if len(t) else 5.0)

    def build(alt):
        order = [x for x in ["1H", "4H", "Daily", "Weekly", "Monthly"] if x in set(t["timeframe"])]
        rows = max(1, len(order))
        base = alt.Chart(t).encode(
            x=alt.X("hz:O", title="horizon (candles)", axis=alt.Axis(labelAngle=0),
                    scale=alt.Scale(paddingInner=0.06)),
            y=alt.Y("timeframe:N", title=None, sort=order,
                    scale=alt.Scale(domain=order, paddingInner=0.08),
                    axis=alt.Axis(labelLimit=0, labelOverlap=False, labelFontSize=13)))
        rect = base.mark_rect(stroke="#000", strokeWidth=2).encode(
            color=alt.Color("edge_pp:Q", scale=uc.diverging_scale(cap, alt), legend=None),
            opacity=alt.condition("datum.small_n", alt.value(0.25), alt.value(1.0)),
            tooltip=["timeframe", "hz", "n", alt.Tooltip("hit:Q", format=".1%"),
                     alt.Tooltip("base:Q", format=".1%"), alt.Tooltip("edge_pp:Q", format="+.1f", title="edge pp"),
                     alt.Tooltip("exp_ret:Q", format="+.2%", title="avg return"), "small_n"])
        txt = base.mark_text(fontSize=12, color="#fff", lineBreak="\n", lineHeight=15).encode(text="label:N")
        return (rect + txt).properties(height=64 * rows)
    uc.render_chart(build, fallback=t)


def _table(t: pd.DataFrame, cols: list[str], title: str) -> None:
    if t.empty:
        return
    show = t[cols + ["n", "hit", "hit_lo", "hit_hi", "base", "edge", "edge_lo", "edge_hi", "exp_ret",
                     "rule_edge", "small_n"]].copy()
    show.columns = [c.replace("rg_", "").replace("_", " ") for c in cols] + [
        "n", "Hit", "Hit lo", "Hit hi", "Base", "Edge", "Edge lo", "Edge hi", "Avg ret", "vs rule", "n<30"]

    def grey(row):
        return ["color:#666" if row["n<30"] else "" for _ in row]
    st.markdown(f"**{title}**")
    st.dataframe(show.style.apply(grey, axis=1).format({
        "Hit": "{:.1%}", "Hit lo": "{:.0%}", "Hit hi": "{:.0%}", "Base": "{:.1%}", "Edge": "{:+.1%}",
        "Edge lo": "{:+.1%}", "Edge hi": "{:+.1%}", "Avg ret": "{:+.2%}", "vs rule": "{:+.2%}"}, na_rep="—")
        .map(lambda v: uc.grad_diverging(v, 0.15) if isinstance(v, float) else "", subset=["Edge"]),
        hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(show)))


def render_stats(repo, player: dict) -> None:
    raw = repo.guesses_df(player["id"])
    d = S.prepare(raw)
    if d.empty:
        st.info("No calls yet — play some charts in PLAY or DAILY FIVE and the dashboard fills itself.")
        return
    f, label = _filters(d)
    if f.empty:
        st.warning("No calls match these filters.")
        return
    h = S.headline(f)
    modes = f["mode"].unique()
    resets = repo.resets(player["id"], modes[0]) if len(modes) == 1 else []
    eq = S.equity_curve(f, resets) if len(modes) == 1 else S.equity_curve(f, [])
    dd = S.drawdown_stats(eq)
    if h["n"] < S.MIN_N:
        st.markdown(uc.state_banner(THEME.mustard, "Small sample",
                                    f"{h['n']} calls — numbers are shown greyed and no claims are made below "
                                    f"{S.MIN_N}."), unsafe_allow_html=True)
    _headline(h, dd)

    st.markdown(section("Equity curve & drawdown", 0,
                        help="Paper account per call. Mustard rules mark account resets (history is kept)."),
                unsafe_allow_html=True)
    if len(modes) > 1:
        st.caption("Several modes selected — the curve is their combined P&L from one $10,000 start; "
                   "filter to one mode for its own account and resets.")
    _equity(eq)

    st.markdown(section("Where your edge is — timeframe × horizon", 2,
                        help="Colour = hit rate minus the base rate. Faded cells have n < 30."),
                unsafe_allow_html=True)
    _heatmap(S.breakdown(f, ["timeframe", "horizon"]))

    st.markdown(section("Breakdowns", 4, help="Grey rows: n < 30 — read nothing into them."),
                unsafe_allow_html=True)
    tabs = st.tabs(["Asset class", "Sector", "Regime", "Indicator set", "Long vs short"])
    with tabs[0]:
        _table(S.breakdown(f, "asset_class"), ["asset_class"], "By asset class")
    with tabs[1]:
        sec = f[f["sector"].fillna("") != ""]
        if sec.empty:
            st.caption("Sector is known for Russell 1000 stocks only — none in this selection.")
        else:
            _table(S.breakdown(sec, "sector"), ["sector"], "By sector (Russell 1000)")
    with tabs[2]:
        for col, ttl in (("rg_trend", "Trend at decision"), ("rg_vol", "Volatility regime"),
                         ("rg_dist_high", "Distance from 52-week high"), ("rg_rsi", "RSI bucket")):
            _table(S.breakdown(f, col), [col], ttl)
    with tabs[3]:
        _table(S.breakdown(f, "indicator_set"), ["indicator_set"],
               "By indicator set — does adding RSI actually help you?")
    with tabs[4]:
        ls = S.long_short(f)
        if ls:
            rows = [{"Side": s.upper(), "n": ls[s]["n"], "Hit": ls[s]["hit"][0], "CI lo": ls[s]["hit"][1],
                     "CI hi": ls[s]["hit"][2], "Avg ret": ls[s]["exp_ret"]} for s in ("long", "short")]
            st.dataframe(pd.DataFrame(rows).style.format({"Hit": "{:.1%}", "CI lo": "{:.0%}", "CI hi": "{:.0%}",
                                                          "Avg ret": "{:+.2%}"}, na_rep="—"),
                         hide_index=True, use_container_width=True)
            st.caption(f"You call UP {ls['long_share']:.0%} of the time; the tape went up "
                       f"{ls['base']:.0%} of the time on these charts (bias {_pp(ls['bias'])}).")

    st.markdown(section("Calibration — does conviction mean anything?", 3,
                        help="Implied P(win): Low 55%, Medium 65%, High 75%. Brier: lower is better; a coin scores 0.25."),
                unsafe_allow_html=True)
    cal = S.calibration(f)
    t = cal["table"]
    if not t.empty:
        def build(alt):
            bars = alt.Chart(t).mark_bar(color=THEME.navy).encode(
                x=alt.X("conviction:N", sort=["Low", "Medium", "High"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("hit:Q", title="hit rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                opacity=alt.condition("datum.small_n", alt.value(0.3), alt.value(1.0)),
                tooltip=["conviction", "n", alt.Tooltip("hit:Q", format=".1%")])
            err = alt.Chart(t).mark_rule(color="#fff").encode(
                x=alt.X("conviction:N", sort=["Low", "Medium", "High"]), y="hit_lo:Q", y2="hit_hi:Q")
            imp = alt.Chart(t).mark_tick(color=THEME.mustard, thickness=3, size=40).encode(
                x=alt.X("conviction:N", sort=["Low", "Medium", "High"]), y="implied:Q")
            return (bars + err + imp).properties(height=220)
        c1, c2 = st.columns([2, 1])
        with c1:
            uc.render_chart(build, fallback=t)
        with c2:
            mono = cal["monotonic"]
            st.markdown(uc.numeric_slab([
                {"label": "Brier score", "value": "—" if not np.isfinite(cal["brier"]) else f"{cal['brier']:.3f}",
                 "sub": "coin = 0.250", "color": THEME.teal if cal["brier"] < 0.25 else THEME.coral},
                {"label": "High > Med > Low?", "value": {True: "YES", False: "NO", None: "n<30"}[mono],
                 "color": {True: THEME.teal, False: THEME.coral, None: _GREY}[mono]},
            ], min_width=120), unsafe_allow_html=True)
            st.caption("Mustard ticks: the probability each conviction level implies.")

    st.markdown(section("Stops & targets", 5), unsafe_allow_html=True)
    stp = S.stops_analysis(f)
    items = []
    if stp.get("stopped_n"):
        items.append({"label": "Stopped, then reversed", "value": _fmt_pct(stp["stopped_then_reversed"], 0),
                      "sub": f"of {stp['stopped_n']} stop-outs", "color": _GREY if stp["stopped_n"] < S.MIN_N else None})
    if "left_on_table" in stp:
        items.append({"label": "MFE left on the table", "value": _fmt_pct(stp["left_on_table"], 2),
                      "sub": f"avg best {_fmt_pct(stp['avg_mfe'], 2)} vs captured {_fmt_pct(stp['avg_captured'], 2, True)}"})
    items.append({"label": "Ambiguous fills", "value": str(stp.get("ambiguous_n", 0)),
                  "sub": "stop + target in one candle → scored as stop"})
    st.markdown(uc.numeric_slab(items, min_width=170), unsafe_allow_html=True)
    hs = stp.get("hindsight")
    if hs is not None and len(hs):
        st.markdown("**Optimal stop in HINDSIGHT** " + help_badge(
            "Replays every call with a k×ATR stop and no target, assuming the stop was hit whenever the "
            "horizon's worst point reached it. Order within the horizon is unknown, so this is an "
            "approximation — and hindsight. Use it to form a hypothesis, then test it going forward."),
            unsafe_allow_html=True)
        st.dataframe(hs.drop(columns=["best"]).style.format(
            {"exp_ret": "{:+.2%}", "hit": "{:.0%}", "stopped": "{:.0%}"})
            .apply(lambda r: [f"background:{THEME.teal}33" if hs.loc[r.name, "best"] else "" for _ in r], axis=1),
            hide_index=True, use_container_width=True)

    st.markdown(section("Export", 1), unsafe_allow_html=True)
    export_buttons(raw, player, "stats")
    with st.popover("Reset a paper account"):
        st.caption("Balance returns to $10,000. History is kept and the reset is marked on the curve.")
        m = st.radio("Account", ["practice", "daily"], horizontal=True, key="eph_reset_mode")
        if st.button("RESET", key="eph_reset_go"):
            repo.reset_account(player["id"], m)
            st.success(f"{m} account reset.")


def export_buttons(raw: pd.DataFrame, player: dict, where: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Full history — CSV", raw.to_csv(index=False).encode(),
                       file_name=f"ephemeris_{player['handle']}.csv", mime="text/csv",
                       key=f"eph_csv_{where}", use_container_width=True)
    buf = io.BytesIO()
    out = raw.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
    out.to_parquet(buf, index=False)
    c2.download_button("Full history — Parquet", buf.getvalue(), file_name=f"ephemeris_{player['handle']}.parquet",
                       mime="application/octet-stream", key=f"eph_pq_{where}", use_container_width=True)


def render_read(repo, player: dict) -> None:
    d = S.prepare(repo.guesses_df(player["id"]))
    if d.empty:
        st.info("No calls yet — the Read writes itself once you have played.")
        return
    f, label = _filters(d)
    rep = R.build(f)
    colours = {"summary": THEME.text, "strengths": THEME.teal, "weaknesses": THEME.coral,
               "notes": THEME.mustard, "drills": THEME.navy}
    titles = {"summary": "Summary", "strengths": "Strengths", "weaknesses": "Weaknesses",
              "notes": "Worth knowing", "drills": "Next drill"}
    for key in ("summary", "strengths", "weaknesses", "notes", "drills"):
        items = rep.get(key) or []
        if not items:
            if key in ("strengths", "weaknesses") and len(f) >= S.MIN_N:
                st.caption(f"{titles[key]}: none with a 95% interval that excludes zero yet.")
            continue
        st.markdown(f'<div style="font-family:{THEME.font_display};letter-spacing:.14em;color:{colours[key]};'
                    f'font-size:1.15rem;margin:.6rem 0 .2rem 0;">{titles[key].upper()}</div>', unsafe_allow_html=True)
        for it in items:
            st.markdown(f'<div style="border-left:4px solid {colours[key]};padding:.35rem .7rem;margin:.25rem 0;'
                        f'background:rgba(0,0,0,.3);font-size:.9rem;line-height:1.45;">{html.escape(it["text"])}</div>',
                        unsafe_allow_html=True)
    st.caption("Generated from deterministic rules on every visit — claims need n ≥ 30 and a 95% interval "
               "that excludes zero.")
    st.download_button("Download the Read (HTML — print to PDF from your browser)",
                       R.to_html(rep, player["display"], label).encode(), file_name=f"ephemeris_read_{player['handle']}.html",
                       mime="text/html", key="eph_read_dl")

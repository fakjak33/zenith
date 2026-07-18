"""Streamlit rendering for the PEAD tab — colorful, visual, Zenith-themed.

Reads only committed JSON (data/pead/*). Every signal component is shown with
its own 0-100 rank, a tooltip explaining the formula + citation, and the
composite is decomposable per stock. Shorts are visually flagged as the
historically weaker side.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load, load_day, archive_days
from . import history as ph
from .signals import WEIGHTS, CONVICTION, POOL_MIN
from ..pretom import calendar as cal
from ..config import THEME
from ..ui_theme import section, stamp

HORIZON_LABEL = {"h5": "+5 td", "h20": "+20 td", "h60": "+60 td",
                 "next_earn": "to next earnings"}

COLS = {
    "Comp": "Composite signal strength 0-100: 30% SUE rank + 30% EAR rank + "
            "15% abnormal volume + 15% 52w-high anchor + 10% revenue surprise. "
            "≥60 = conviction tier.",
    "Side": "LONG = EPS beat AND positive market-excess reaction; SHORT = miss "
            "AND negative reaction (the two-confirmation gate). Shorts drift "
            "less reliably and carry borrow costs (Martineau 2021).",
    "SUE": "Standardized unexpected earnings rank: (actual − consensus EPS) / "
           "pre-announcement price, ranked 0-100 vs the trailing quarter's "
           "announcements (Bernard & Thomas 1989; Livnat & Mendenhall 2006).",
    "EAR": "Earnings announcement return rank: stock close-to-close move over "
           "the reaction session minus SPY, ranked 0-100. The market's own "
           "verdict — historically drifts MORE than SUE and the combo is the "
           "strongest documented pairing (Brandt et al. 2008).",
    "AVOL": "Abnormal announcement volume rank: log(reaction-day volume / "
            "60-day median). High unexplained volume = opinion divergence = "
            "stronger drift (Garfinkel & Sokobin 2006). Never direction-flipped.",
    "ANCHOR": "52-week-high anchoring rank: good news NEAR the high (bad news "
              "FAR from it) is where investors underreact most, so the drift "
              "runs longest (George, Hwang & Li 2015).",
    "RUE": "Revenue surprise rank (time-series proxy: latest YoY quarterly "
           "revenue growth vs its own trailing pattern — analyst revenue "
           "consensus isn't free). Drift is stronger when revenue confirms "
           "the EPS beat (Jegadeesh & Livnat 2006). Neutral 50 when missing.",
    "EAR raw": "Raw market-excess reaction: stock minus SPY, close before the "
               "announcement to the reaction-day close.",
    "Surprise": "EPS surprise vs consensus: actual − forecast, as % of the "
                "consensus magnitude.",
    "EPS a/c": "Actual / consensus EPS. Source: 'nasdaq' = Nasdaq calendar "
               "consensus snapshot, 'yf' = Yahoo estimate, 'tsq' = last year's "
               "same quarter (seasonal proxy — weakest).",
    "Gap held": "Did the reaction candle close in the signal's direction "
                "(close ≥ open for longs)? Practitioner confirmation, not part "
                "of the score.",
    "Mom": "Is 12-1 month price momentum aligned with the signal direction? "
           "Context only (Chan, Jegadeesh & Lakonishok 1996).",
    "Cap": "Market-cap tier (small <$5B, mid $5-25B, large >$25B). Drift is "
           "historically strongest in smaller names — context, not score.",
    "Next earn": "Next scheduled report. 25-30% of the drift concentrates in "
                 "the 3 days around it (Bernard & Thomas 1990); the tracked "
                 "exit is the close the day BEFORE its reaction session.",
    "Entry": "Open of the first session after the reaction day (the standard "
             "academic implementation: let the gap settle, enter next open).",
    "Run xs": "Running sign-adjusted excess return vs SPY since entry: "
              "positive = the pick is working (for shorts, positive means the "
              "stock is LAGGING SPY).",
    "Win rate": "Share of evaluated picks with positive sign-adjusted excess "
                "return at that horizon (positive = the pick worked).",
    "Avg xs": "Average sign-adjusted excess return vs SPY at that horizon.",
}


# ---------------------------------------------------------------- helpers ---
def _grad(v, cap: float, color: tuple[int, int, int]) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    a = max(0.0, min(1.0, abs(float(v)) / cap))
    r, g, b = (int(11 + (c - 11) * a) for c in color)
    return f"background-color: rgb({r},{g},{b}); color: #fff;"


def _grad_teal(v, cap=1.0):
    return _grad(v, cap, (46, 196, 182))


def _grad_coral(v, cap=1.0):
    return _grad(v, cap, (255, 90, 60))


def _grad_diverging(v, cap=0.05):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return _grad_teal(v, cap) if float(v) >= 0 else _grad_coral(v, cap)


def _grad_score(v):
    """0-100 composite ramp: panel -> mustard -> teal above conviction."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if float(v) >= CONVICTION:
        return _grad_teal(float(v) / 100.0, 1.0)
    return _grad(float(v) / 100.0, 1.0, (255, 200, 87))


def _colcfg(cols) -> dict:
    return {c: st.column_config.Column(help=COLS[c]) for c in cols if c in COLS}


def _fmt_pct(v):
    return "" if v is None or pd.isna(v) else f"{v:+.1%}"


def _side_txt(side: str) -> str:
    return "▲ LONG" if side == "long" else "▼ SHORT" if side == "short" else "· mixed"


def _latest_sheet(latest: dict) -> dict:
    days = archive_days()
    return load_day(days[0]) if days else {}


def today_badge() -> str | None:
    """Chip on the TODAY tab when the freshest sheet has live signals."""
    days = archive_days()
    if not days:
        return None
    sheet = load_day(days[0])
    n_l, n_s = sheet.get("n_long", 0), sheet.get("n_short", 0)
    if not (n_l or n_s):
        return None
    d = sheet.get("day", "")
    if d < (date.today() - timedelta(days=4)).isoformat():   # stale sheet
        return None
    color = THEME.teal if n_l >= n_s else THEME.coral
    text = f"◆ PEAD: {n_l} long / {n_s} short signal{'s' * ((n_l + n_s) != 1)} ({d})"
    return (f'<div style="display:inline-block; font-family:{THEME.font_display}; '
            f'font-size:0.95rem; letter-spacing:0.1em; text-transform:uppercase; '
            f'color:{color}; border:1px solid {color}; padding:0.15rem 0.6rem; '
            f'margin:0 0 0.6rem 0;">{text} · see PEAD tab</div>')


def _banner(latest: dict) -> None:
    days = archive_days()
    sheet = load_day(days[0]) if days else {}
    n_el = sheet.get("n_eligible", 0)
    live = bool(n_el)
    color = THEME.teal if live else THEME.mustard
    label = "SIGNALS LIVE" if live else "QUIET TAPE"
    d = sheet.get("day", "—")
    msg = (f"{sheet.get('n_long', 0)} long · {sheet.get('n_short', 0)} short of "
           f"{sheet.get('n_reporters', 0)} R1000 reporters on {d}"
           if sheet else "no signal sheets yet")
    st.markdown(
        f'<div style="font-family:{THEME.font_display}; font-size:1.25rem; '
        f'letter-spacing:0.14em; text-transform:uppercase; color:#000; '
        f'background:linear-gradient(90deg, {color}, {THEME.panel}); '
        f'border-left:6px solid {color}; padding:0.45rem 0.9rem; '
        f'margin:0.2rem 0 0.8rem 0;">'
        f'<b>{label}</b> · {msg}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- signals ---
def _surprise_pct(r: dict) -> float | None:
    a, c = r.get("eps_actual"), r.get("eps_consensus")
    if a is None or c is None or c == 0:
        return None
    return (a - c) / abs(c)


def _sheet_frame(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        ranks, raw, ctx, fl = (r.get("ranks") or {}), r.get("raw", {}), \
            r.get("context", {}), r.get("flags", {})
        recs.append({
            "Ticker": r["ticker"], "Name": (r.get("name") or "")[:28],
            "Side": _side_txt(r["side"]),
            "Comp": r.get("composite"),
            "SUE": ranks.get("sue"), "EAR": ranks.get("ear"),
            "AVOL": ranks.get("avol"), "ANCHOR": ranks.get("anchor"),
            "RUE": ranks.get("rue"),
            "Surprise": _surprise_pct(r), "EAR raw": raw.get("ear"),
            "EPS a/c": (f"{r.get('eps_actual')} / {r.get('eps_consensus')} "
                        f"({r.get('sue_source', '?')})"),
            "Gap held": {True: "✓", False: "✗", None: "?"}[fl.get("gap_held")],
            "Mom": {True: "✓", False: "✗", None: "?"}[fl.get("momentum_aligned")],
            "Cap": ctx.get("cap_tier") or "?",
            "Next earn": ctx.get("next_earn_date") or "?",
        })
    return pd.DataFrame(recs)


def _signal_table(rows: list[dict], key: str) -> None:
    df = _sheet_frame(rows)
    if df.empty:
        st.caption("No eligible signals on this sheet.")
        return
    try:
        sty = (df.style
               .map(_grad_score, subset=["Comp"])
               .map(lambda v: _grad_teal((v or 0) / 100, 1.0),
                    subset=["SUE", "EAR", "AVOL", "ANCHOR", "RUE"])
               .map(lambda v: _grad_diverging(v, 0.08), subset=["EAR raw"])
               .map(lambda v: ("color: #2ec4b6;" if isinstance(v, str) and "LONG" in v
                               else "color: #ff5a3c; font-weight: 700;"
                               if isinstance(v, str) and "SHORT" in v else ""),
                    subset=["Side"])
               .format({"Comp": "{:.0f}", "SUE": "{:.0f}", "EAR": "{:.0f}",
                        "AVOL": "{:.0f}", "ANCHOR": "{:.0f}", "RUE": "{:.0f}",
                        "Surprise": _fmt_pct, "EAR raw": _fmt_pct}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(520, 45 + 35 * len(df)),
                     column_config=_colcfg(df.columns), key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _component_bars(row: dict) -> None:
    ranks = row.get("ranks") or {}
    if not ranks:
        return
    labels = {"sue": "SUE — EPS surprise", "ear": "EAR — price reaction",
              "avol": "AVOL — abnormal volume", "anchor": "ANCHOR — 52w high",
              "rue": "RUE — revenue surprise"}
    df = pd.DataFrame([{"component": f"{labels[k]}  (w {WEIGHTS[k]:.0%})",
                        "rank": ranks.get(k), "weight": WEIGHTS[k],
                        "contrib": round(WEIGHTS[k] * (ranks.get(k) or 50), 1)}
                       for k in WEIGHTS])
    try:
        import altair as alt
        base = alt.Chart(df)
        ch = (base.mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("rank:Q", title="component rank (0-100, direction-adjusted)",
                    scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("component:N", sort=list(df["component"]), title=None,
                    axis=alt.Axis(labelLimit=0)),
            color=alt.Color("rank:Q", scale=alt.Scale(scheme="tealblues",
                                                      domain=[0, 100]), legend=None),
            tooltip=["component", alt.Tooltip("rank:Q", format=".0f"),
                     alt.Tooltip("contrib:Q", title="points into composite")],
        ) + base.mark_rule(strokeDash=[4, 3], color=THEME.mustard).encode(
            x=alt.datum(50))).properties(height=190)
        st.altair_chart(ch, use_container_width=True)
        st.caption(f"Composite {row.get('composite')} = weighted blend of the five "
                   f"ranks (dashed line = neutral 50). Pool n = "
                   f"{row.get('rank_pool_n', '?')}.")
    except Exception:
        st.dataframe(df, hide_index=True)


def _excluded_table(rows: list[dict]) -> None:
    ex = [r for r in rows if r.get("excluded_reason")]
    if not ex:
        return
    reasons = {"mixed": "signals disagree (beat w/ selloff or miss w/ rally)",
               "weak_reaction": "reaction < 0.5% either way",
               "no_eps": "no reported EPS", "no_consensus": "no consensus estimate",
               "no_prices": "no usable price bars", "zero_surprise": "EPS exactly in line",
               "penny": "price ≤ $5", "illiquid": "median $ volume < $5m"}
    df = pd.DataFrame([{"Ticker": r["ticker"], "Name": (r.get("name") or "")[:30],
                        "Why excluded": reasons.get(r["excluded_reason"],
                                                    r["excluded_reason"]),
                        "Surprise": _surprise_pct(r),
                        "EAR raw": (r.get("raw") or {}).get("ear")} for r in ex])
    sty = df.style.map(lambda v: _grad_diverging(v, 0.08),
                       subset=["Surprise", "EAR raw"]) \
        .format({"Surprise": _fmt_pct, "EAR raw": _fmt_pct})
    st.dataframe(sty, use_container_width=True, hide_index=True,
                 height=min(400, 45 + 35 * len(df)))


# ------------------------------------------------------------- active book --
def _book_frame(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        run = r.get("running") or {}
        recs.append({
            "Ticker": r["ticker"], "Side": _side_txt(r["side"]),
            "Comp": r.get("composite"),
            "Entry": r.get("entry_date") or "pending",
            "Entry px": r.get("entry_open"),
            "Days held": (len(cal.trading_days(date.fromisoformat(r["entry_date"]),
                                               date.today())) - 1
                          if r.get("entry_date") else None),
            "Run xs": run.get("excess_ret"),
            "Raw ret": run.get("ret"),
            "Next earn": r.get("next_earn_date") or "?",
            "Cap": r.get("cap_tier") or "?",
            "Backfilled": "✓" if r.get("backfilled") else "live",
        })
    df = pd.DataFrame(recs)
    return df.sort_values("Comp", ascending=False) if not df.empty else df


def _book_table(rows: list[dict], key: str) -> None:
    df = _book_frame(rows)
    if df.empty:
        st.caption("No open picks on this side.")
        return
    try:
        sty = (df.style
               .map(_grad_score, subset=["Comp"])
               .map(lambda v: _grad_diverging(v, 0.10), subset=["Run xs", "Raw ret"])
               .format({"Comp": "{:.0f}", "Entry px": "{:,.2f}",
                        "Run xs": _fmt_pct, "Raw ret": _fmt_pct}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(520, 45 + 35 * len(df)),
                     column_config=_colcfg(df.columns), key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


# -------------------------------------------------------------- drift curve --
def _drift_chart(curve: list[dict], as_of: str | None) -> None:
    if not curve:
        st.caption("Drift curve appears after the backfill (or first weekly refresh).")
        return
    df = pd.DataFrame(curve)
    recs = []
    for _, r in df.iterrows():
        for side in ("long", "short"):
            v = r.get(f"{side}_avg_excess")
            if v is not None and not pd.isna(v):
                recs.append({"td": r["td"], "side": side.upper(),
                             "xs": v * 100, "n": r.get(f"n_{side}")})
    long_df = pd.DataFrame(recs)
    try:
        import altair as alt
        base = alt.Chart(long_df)
        lines = base.mark_line(interpolate="monotone", strokeWidth=3).encode(
            x=alt.X("td:Q", title="trading days since the announcement"),
            y=alt.Y("xs:Q", title="avg cumulative excess return vs SPY (%)"),
            color=alt.Color("side:N", scale=alt.Scale(domain=["LONG", "SHORT"],
                                                      range=[THEME.teal, THEME.coral]),
                            legend=alt.Legend(title=None)),
            tooltip=["side", "td", alt.Tooltip("xs:Q", format="+.2f"),
                     alt.Tooltip("n:Q", title="picks")])
        rules = alt.Chart(pd.DataFrame({"td": [5, 20, 60]})).mark_rule(
            strokeDash=[5, 4], color=THEME.mustard, opacity=0.7).encode(x="td:Q")
        st.altair_chart((lines + rules).properties(height=380),
                        use_container_width=True)
        n_l = int(long_df[long_df["side"] == "LONG"]["n"].max() or 0)
        n_s = int(long_df[long_df["side"] == "SHORT"]["n"].max() or 0)
        st.caption(f"The classic PEAD 'staircase': every tracked pick's excess-vs-SPY "
                   f"path, averaged by day since announcement (sign-adjusted: up = "
                   f"the picks worked, both sides). Dashed rules mark the +5/+20/+60 "
                   f"td exits. Cohorts: {n_l} long / {n_s} short."
                   + (f" Refreshed {as_of}." if as_of else ""))
    except Exception:
        st.line_chart(df.set_index("td"))


# ------------------------------------------------------------------ tracker --
def _tracker(history: dict) -> None:
    rows = history.get("rows", [])
    if not rows:
        st.caption("No tracked picks yet — run the backfill.")
        return
    summ = ph.summarize(rows)
    cells = summ.get("cells", {})

    h20l = cells.get("h20|long|all", {})
    h20s = cells.get("h20|short|all", {})
    h60l = cells.get("h60|long|all", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Picks tracked", summ["n_rows"],
              help="All history rows (live + backfilled), long + short.")
    c2.metric("Live (non-backfilled)", summ["n_live"],
              help="Signals generated by the daily pipeline after launch.")
    c3.metric("Long win rate +20td", _fmt_pct(h20l.get("win_rate")).lstrip("+")
              if h20l else "—", help=COLS["Win rate"])
    c4.metric("Long avg xs +20td", _fmt_pct(h20l.get("avg")) if h20l else "—",
              help=COLS["Avg xs"])
    c5.metric("Short win rate +20td", _fmt_pct(h20s.get("win_rate")).lstrip("+")
              if h20s else "—",
              help=COLS["Win rate"] + " Shorts are the historically weaker side.")
    c6.metric("Long avg xs +60td", _fmt_pct(h60l.get("avg")) if h60l else "—",
              help=COLS["Avg xs"])

    # win-rate grid: horizon x (side · tercile)
    recs = []
    for key, c in cells.items():
        h, side, terc = key.split("|")
        if terc == "all":
            continue
        recs.append({"horizon": HORIZON_LABEL[h], "group": f"{side} · {terc}",
                     "side": side, "win": c["win_rate"] * 100, "avg": c["avg"] * 100,
                     "n": c["n"]})
    grid = pd.DataFrame(recs)
    if not grid.empty:
        try:
            import altair as alt
            order = [f"{s} · {t}" for s in ("long", "short")
                     for t in ("high", "mid", "low")]
            ch = (alt.Chart(grid).mark_rect(stroke=THEME.grid).encode(
                x=alt.X("horizon:N", sort=list(HORIZON_LABEL.values()), title=None,
                        axis=alt.Axis(labelLimit=0)),
                y=alt.Y("group:N", sort=order, title="side · score tercile",
                        axis=alt.Axis(labelLimit=0)),
                color=alt.Color("win:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                         domain=[30, 70]),
                                legend=alt.Legend(title="win %")),
                tooltip=["group", "horizon",
                         alt.Tooltip("win:Q", format=".0f", title="win rate %"),
                         alt.Tooltip("avg:Q", format="+.2f", title="avg xs %"),
                         alt.Tooltip("n:Q", title="picks")],
            ) + alt.Chart(grid).mark_text(fontWeight="bold").encode(
                x=alt.X("horizon:N", sort=list(HORIZON_LABEL.values())),
                y=alt.Y("group:N", sort=order),
                text=alt.Text("win:Q", format=".0f"),
                color=alt.value("#000"))).properties(height=250)
            st.altair_chart(ch, use_container_width=True)
            st.caption("Win rate (%) by exit horizon, side and composite-score tercile "
                       "(high ≥60, mid 45-60, low <45). If the model works, the HIGH "
                       "tercile rows should be greenest — and research says the long "
                       "side should beat the short side.")
        except Exception:
            st.dataframe(grid, hide_index=True)

    # distribution of +20td outcomes
    vals = [(r["horizons"]["h20"]["excess_ret"], r["side"]) for r in rows
            if r["horizons"].get("h20", {}).get("evaluated")
            and r["horizons"]["h20"].get("excess_ret") is not None]
    if vals:
        dist = pd.DataFrame(vals, columns=["xs", "side"])
        dist["xs"] *= 100
        try:
            import altair as alt
            ch = (alt.Chart(dist).mark_bar(opacity=0.8).encode(
                x=alt.X("xs:Q", bin=alt.Bin(maxbins=40),
                        title="+20 td sign-adjusted excess return (%)"),
                y=alt.Y("count():Q", title="picks"),
                color=alt.Color("side:N", scale=alt.Scale(domain=["long", "short"],
                                                          range=[THEME.teal, THEME.coral]),
                                legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("count():Q", title="picks")],
            ).properties(height=260))
            st.altair_chart(ch, use_container_width=True)
            st.caption("Right of zero = the pick beat SPY in its direction. The fat "
                       "right tail is what PEAD is about; the left tail is why "
                       "position sizing matters.")
        except Exception:
            pass

    # full stats table (all cells)
    stat_rows = []
    for key, c in cells.items():
        h, side, terc = key.split("|")
        stat_rows.append({"Horizon": HORIZON_LABEL[h], "Side": side,
                          "Tercile": terc, "N": c["n"],
                          "Win rate": c["win_rate"], "Avg xs": c["avg"],
                          "Median": c["median"], "Std": c["std"],
                          "Min": c["min"], "Max": c["max"]})
    sf = pd.DataFrame(stat_rows)
    if not sf.empty:
        with st.expander("Full descriptive statistics (every horizon × side × tercile)"):
            sty = (sf.style
                   .map(lambda v: _grad_teal(v, 1.0), subset=["Win rate"])
                   .map(lambda v: _grad_diverging(v, 0.05),
                        subset=["Avg xs", "Median"])
                   .format({"Win rate": lambda v: f"{v:.0%}",
                            "Avg xs": _fmt_pct, "Median": _fmt_pct,
                            "Std": "{:.3f}", "Min": _fmt_pct, "Max": _fmt_pct}))
            st.dataframe(sty, use_container_width=True, hide_index=True,
                         height=min(560, 45 + 35 * len(sf)))


# ------------------------------------------------------------------ render --
def render() -> None:
    st.caption(DISCLAIMER)
    latest = load("signals", {})
    _banner(latest)

    with st.expander("How PEAD works — the research in 90 seconds"):
        st.markdown(
            "**The anomaly** (Ball & Brown 1968; Bernard & Thomas 1989/90): after an "
            "earnings surprise, prices keep drifting the same direction for ~60 "
            "trading days. Top-vs-bottom surprise deciles earned ~3.2% abnormal "
            "over the following quarter — Fama called it the 'granddaddy of "
            "underreaction events'.\n\n"
            "**Two confirmations, not one.** The EPS surprise (**SUE**) and the "
            "market's own reaction (**EAR** — the stock's move vs SPY through the "
            "reaction close) carry *independent* information; the drift after both "
            "agree is roughly TWICE either alone (Brandt et al. 2008: combined "
            "≈12.5%/yr abnormal historically). That's exactly the 'beat + closes "
            "up' pattern this tab formalizes — a signal only fires when both agree.\n\n"
            "**Three amplifiers** complete the 0-100 composite: abnormal announcement "
            "volume (opinion divergence — Garfinkel & Sokobin 2006), the 52-week-high "
            "anchor (underreaction is strongest near the high — George, Hwang & Li "
            "2015), and a time-series revenue surprise (Jegadeesh & Livnat 2006).\n\n"
            "**Entries and exits**: enter the next session's open (let the gap "
            "settle); the tracker scores every pick at +5, +20 and +60 trading days "
            "AND at the close before the *next* report — because 25-30% of the "
            "drift concentrates in the 3 days around it. The stats table shows which "
            "exit has actually worked.\n\n"
            "**Honest caveat** (Martineau 2021 'Rest in Peace PEAD'): in LARGE caps "
            "the drift has mostly been arbitraged away since ~2006 (2025 papers "
            "dispute this — the debate is measurement-dependent). This screener "
            "leans on the strongest surviving form (SUE+EAR agreement), flags cap "
            "tiers, and tracks its own live results so you can judge.")

    if not latest:
        st.info("No PEAD data yet. Run `python -m zenith.pead.compute --action "
                "backfill` then `--action auto` (or wait for the scheduled Action).")
        return
    st.markdown(stamp(latest.get("as_of", "?"), "PEAD"), unsafe_allow_html=True)

    # --- fresh signals -------------------------------------------------------
    days = archive_days()
    sheet = load_day(days[0]) if days else {}
    rows = sheet.get("signals", [])
    eligible = [r for r in rows if r.get("excluded_reason") is None]
    st.markdown(section(f"Fresh signals — reaction day {sheet.get('day', '—')}", 0,
                        help="Russell 1000 companies whose report's first tradeable "
                             "session just closed, scored 0-100. Only rows passing "
                             "every gate (incl. the two-confirmation direction "
                             "gate) become tracked picks."), unsafe_allow_html=True)
    pool_n = min((r.get("rank_pool_n") or 0 for r in eligible), default=0)
    if eligible and pool_n < POOL_MIN:
        st.caption(f"⚠ thin ranking pool (n={pool_n} < {POOL_MIN}) — component "
                   "ranks are less stable until more announcements accumulate.")
    _signal_table(eligible, key="sheet_now")

    if eligible:
        pick = st.selectbox("Decompose a signal", [r["ticker"] for r in eligible],
                            key="pead_decompose")
        row = next(r for r in eligible if r["ticker"] == pick)
        _component_bars(row)
    with st.expander(f"Reporters that did NOT qualify "
                     f"({sum(1 for r in rows if r.get('excluded_reason'))})"):
        _excluded_table(rows)

    # --- active books --------------------------------------------------------
    hist_obj = load("history", {})
    act = ph.active_rows(hist_obj.get("rows", []))
    longs = [r for r in act if r["side"] == "long"]
    shorts = [r for r in act if r["side"] == "short"]
    st.markdown(section("Active book — open drift positions", 1,
                        help="Every tracked pick still inside its drift window "
                             "(before both the +60 td and next-earnings exits). "
                             "Sorted by composite."), unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    run_l = [r["running"]["excess_ret"] for r in longs if r.get("running")]
    run_s = [r["running"]["excess_ret"] for r in shorts if r.get("running")]
    b1.metric("Open longs", len(longs))
    b2.metric("Long book running xs", _fmt_pct(sum(run_l) / len(run_l))
              if run_l else "—", help=COLS["Run xs"])
    b3.metric("Open shorts", len(shorts),
              help="Shorts are flagged — historically the weaker, harder side.")
    b4.metric("Short book running xs", _fmt_pct(sum(run_s) / len(run_s))
              if run_s else "—", help=COLS["Run xs"])
    t_long, t_short = st.tabs([f"▲ LONG book ({len(longs)})",
                               f"▼ SHORT book ({len(shorts)}) — weaker side"])
    with t_long:
        _book_table(longs, key="book_long")
    with t_short:
        st.caption("Short-side drift decayed most in the recent decades and real "
                   "shorting carries borrow costs + unlimited-loss risk (not "
                   "modeled). Treat as an avoid/hedge list first.")
        _book_table(shorts, key="book_short")

    # --- drift curve ---------------------------------------------------------
    st.markdown(section("Drift curve — the PEAD staircase", 4,
                        help="Average cumulative excess-vs-SPY path of all tracked "
                             "picks, by trading day since the announcement."),
                unsafe_allow_html=True)
    _drift_chart(latest.get("drift_curve", []), latest.get("drift_curve_as_of"))

    # --- tracker -------------------------------------------------------------
    st.markdown(section("Pick tracker — every signal scored", 2,
                        help="Sign-adjusted excess returns vs SPY at each exit "
                             "horizon: positive = the pick worked."),
                unsafe_allow_html=True)
    _tracker(hist_obj)

    # --- archive browser -----------------------------------------------------
    if days:
        st.markdown(section("Open a past signal sheet", 5), unsafe_allow_html=True)
        pick_day = st.selectbox("Reaction day", days, key="pead_day")
        old = load_day(pick_day)
        if old and pick_day != sheet.get("day"):
            _signal_table([r for r in old.get("signals", [])
                           if r.get("excluded_reason") is None],
                          key=f"sheet_{pick_day}")
            with st.expander("Excluded reporters that day"):
                _excluded_table(old.get("signals", []))

    st.caption("Caveats: consensus EPS is the Nasdaq calendar snapshot (not a "
               "point-in-time I/B/E/S history); revenue surprise is a time-series "
               "proxy; backfilled picks use current Russell 1000 membership "
               "(survivorship); prices are yfinance best-effort; SPY is the market "
               "proxy. Sign convention everywhere: positive excess = the pick "
               "worked, including shorts. Not investment advice.")

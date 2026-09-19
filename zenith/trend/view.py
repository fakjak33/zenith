"""TREND FOLLOWING tab — the seven-speed EWMAC dashboard.

Reads only committed artefacts under data/trend/ (plus MOMENTUM / ETF MOMENTUM's
own committed scores for the side-by-side comparison), with one deliberate
exception shared with the MOMENTUM tabs: the Detail view's price chart fetches
that ONE ticker's prices on demand (cached), because a daily price panel for
~2,000 instruments is not worth committing.

Layout: a cross-universe header (Stocks vs ETFs), then Stocks | ETFs, each with
the same four views on one radio -- Overview (what is trending, at which
speed) -> All Trends (screen / sort the whole universe) -> Triggers (what just
changed) -> Detail (why this score, how it developed). Radios rather than
st.tabs because clicking a dot, cell or row elsewhere must be able to open
Detail programmatically. The universe body runs inside @st.fragment: app.py
executes all sixteen tabs on every rerun, and this tab's interactions should
not pay for the other fifteen.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from .. import ui_charts as uc
from ..config import (MOM_MEMBERSHIP_START, MOM_STATES, THEME, TREND_CONFIRM_MIN_SPEEDS,
                      TREND_CONFIRM_WINDOW, TREND_FORECAST_SCALARS, TREND_RECENT_CROSS_DAYS,
                      TREND_FILES, TREND_SCALAR_EXTRAPOLATED, TREND_SPEEDS, TREND_UNIVERSES, TREND_VOL_SPAN,
                      TREND_CASH_LIKE_VOL, TREND_BAND_HYSTERESIS)
from ..ui_theme import evidence_rating, key_findings, section, stamp
from . import (DISCLAIMER, SPEED_KEYS, SPEED_NAMES, SPEED_SHORT, UNIVERSE_LABELS, load)
from . import events as ev
from . import history, structure, table, viz
from .structure import STRUCTURE_COLORS, STRUCTURE_HELP, STRUCTURES

SUBVIEWS = ["Overview", "All Trends", "Triggers", "Detail"]
GROUP_COL = {"stocks": "sector", "etfs": "asset_class"}
GROUP_LABEL = {"stocks": "Sector", "etfs": "Asset class"}

_EVIDENCE_NOTE = ("Time-series trend following is one of the best-documented return sources across "
                  "asset classes — significant in 58 futures markets (Moskowitz, Ooi & Pedersen 2012) "
                  "and positive in every decade since 1880 (Hurst, Ooi & Pedersen 2017). The evidence "
                  "is strongest for diversified multi-asset portfolios (the ETF side here) and weaker, "
                  "noisier and more crash-prone for single stocks. A score describes the trend that HAS "
                  "happened; it is not a forecast of returns.")

_FINDINGS = [
    {"stat": "Time-series momentum is significant across 58 equity, bond, currency and commodity markets.",
     "cite": "Moskowitz, Ooi & Pedersen (2012)"},
    {"stat": "Trend following has been profitable in every decade since 1880, including major bear markets.",
     "cite": "Hurst, Ooi & Pedersen (2017)"},
    {"stat": "CTA returns load on trend signals at several frequencies at once — daily, weekly and monthly.",
     "cite": "Baltas & Kosowski (2013)"},
    {"stat": "Vol-normalized EWMAC forecasts put every speed and instrument on one scale (avg |f| ≈ 10, cap 20).",
     "cite": "Carver (2015), Systematic Trading"},
]

COLS = {
    "Rank": "Position in this universe by Trend Score.",
    "Ticker": "Exchange ticker.",
    "Name": "Company / fund name.",
    "Score": "Trend Score: equal-weight mean of the seven EWMAC forecasts, -20 (max bearish) to +20 (max bullish).",
    "Structure": "What the relationship between the seven speeds says (see the legend on Overview).",
    **{SPEED_SHORT[k]: f"{SPEED_SHORT[k]} EWMAC ({SPEED_NAMES[k]}): vol-normalized forecast, -20..+20."
       for k in SPEED_KEYS},
    "Term structure": ("The seven forecasts as bars, fast (left) to slow (right). Each bar is a LEVEL "
                       "on -20..+20: empty = -20, half = 0, full = +20. The colored speed columns "
                       "carry the sign."),
    "60d": "Trend Score over the last 60 trading days.",
    "Δ5d": "Change in Trend Score over 5 trading days.",
    "Δ20d": "Change in Trend Score over 20 trading days.",
    "Accel": "Fast-group minus slow-group forecast: + = short-term trend running ahead of the long-term one.",
    "Agree": "Speeds on the same side as the score, out of the valid speeds.",
    "Last event": "Date of the most recent trigger / upgrade / downgrade / confirmation.",
    "Momentum": "The same asset's MOMENTUM (or ETF MOMENTUM) composite, also on -20..+20.",
    "Trend − Mom": "Trend Score minus Momentum composite.",
    "Sector": "GICS sector.", "Asset class": "Asset-class rollup of the Morningstar category.",
}


# ================================================================ data ====
@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(universe: str, cache_bust: str = "") -> dict:
    return {
        "latest": load(universe, "latest", {}),
        "recent": load(universe, "recent_events", {}),
        "breadth": load(universe, "breadth", {}),
        "diagnostics": load(universe, "diagnostics", {}),
        "status": load(universe, "status", {}),
    }


@st.cache_data(ttl=600, show_spinner=False)
def _momentum(universe: str, cache_bust: str = "") -> dict:
    """{ticker: composite} from the matching MOMENTUM tab (read-only)."""
    try:
        if universe == "stocks":
            from ..mom import load as mload
        else:
            from ..etfmom import load as mload
        doc = mload("scores", {})
        return {r["ticker"]: r.get("composite") for r in doc.get("rows", [])
                if r.get("composite") is not None}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=24)
def _shard(universe: str, year: int, cache_bust: str = "") -> dict:
    return history.read_shard(universe, year)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def _events_all(universe: str, cache_bust: str = "") -> list[dict]:
    return history.events_for(universe)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def _prices(ticker: str) -> pd.DataFrame | None:
    from ..cas.sources import prices
    px, _ = prices.get_history([ticker], period="10y", max_age_hours=6.0)
    df = px.get(ticker)
    return df if df is not None and not df.empty else None


def _bust(art: dict) -> str:
    return str(art["latest"].get("as_of", "")) + str(art["status"].get("date", ""))


def _scored(art: dict) -> list[dict]:
    return [r for r in art["latest"].get("rows", []) if not r.get("excluded")]


@st.cache_data(ttl=600, show_spinner=False)
def _frame(universe: str, cache_bust: str = "") -> pd.DataFrame:
    art = _artefacts(universe, cache_bust)
    df = table.frame(_scored(art), group_col=GROUP_COL[universe])
    if df.empty:
        return df
    mom = _momentum(universe, cache_bust)
    df["momentum"] = df["ticker"].map(mom).astype(float)
    df["trend_minus_mom"] = df["score"] - df["momentum"]
    df["agree"] = np.where(df["score"] >= 0, df["n_bull"], df["n_bear"])
    return df


def _summary(universe: str) -> dict | None:
    art = _artefacts(universe, _status_bust(universe))
    rows = _scored(art)
    if not rows:
        return None
    s = np.array([r["score"] for r in rows], dtype=float)
    speed_bull = []
    for j in range(len(SPEED_KEYS)):
        v = [r["forecasts"][j] for r in rows if r["forecasts"][j] is not None]
        speed_bull.append(sum(1 for x in v if x > 0) / len(v) if v else None)
    return {"label": UNIVERSE_LABELS[universe], "n": len(rows), "as_of": art["latest"].get("as_of", ""),
            "pct_bull": float((s >= 5).mean()), "pct_bear": float((s < -5).mean()),
            "median": float(np.median(s)), "breadth": speed_bull,
            "persist_up": sum(1 for r in rows if r["structure"] == "Persistent Uptrend"),
            "persist_down": sum(1 for r in rows if r["structure"] == "Persistent Downtrend")}


def _status_bust(universe: str) -> str:
    """Cache key for this universe's artefacts: the files' modification times.
    A date alone is not enough -- a same-day re-run (or a forced one)
    rewrites the artefacts without changing any date field."""
    files = TREND_FILES[universe]
    return "|".join(str(files[k].stat().st_mtime_ns) if files[k].exists() else "-"
                    for k in ("latest", "status", "recent_events"))


def _recent_events(art: dict) -> list[dict]:
    return [history.decode_event(r) for r in art["recent"].get("rows", [])]


# ================================================================ badge ====
def today_badge() -> str | None:
    """TODAY chip: breadth of both universes + today's confirmations."""
    try:
        parts, n_conf, color = [], 0, THEME.teal
        for u in TREND_UNIVERSES:
            doc = load(u, "latest", {})
            rows = [r for r in doc.get("rows", []) if not r.get("excluded")]
            if not rows:
                continue
            up = sum(1 for r in rows if r.get("structure") == "Persistent Uptrend")
            dn = sum(1 for r in rows if r.get("structure") == "Persistent Downtrend")
            parts.append(f"{UNIVERSE_LABELS[u]} {up}↑/{dn}↓")
            as_of = doc.get("as_of")
            n_conf += sum(1 for r in load(u, "recent_events", {}).get("rows", [])
                          if r[0] == as_of and r[2] == "confirmation")
            if dn > up:
                color = THEME.coral
        if not parts:
            return None
        return uc.chip(f"TREND — persistent trends: {' · '.join(parts)} · "
                       f"{n_conf} multi-speed confirmation(s) today", color=color,
                       sub="see TREND FOLLOWING tab")
    except Exception:
        return None


# ================================================================= main ====
def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating("B+", "century-long, multi-asset; weaker on single stocks",
                                _EVIDENCE_NOTE), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    summaries = [_summary(u) for u in TREND_UNIVERSES]
    if not any(summaries):
        st.markdown(stamp("—", "TREND FOLLOWING"), unsafe_allow_html=True)
        st.info("No data yet. Run `python -m zenith.trend.compute --action backfill` once (10 years of "
                "history), then the nightly `--action auto` keeps it current.")
        return

    st.markdown(section("Stocks vs ETFs — trend breadth at a glance", 0,
                        help="Share of each universe with a bullish (>= +5) or bearish (< -5) Trend "
                             "Score, and the share bullish at each of the seven speeds. A falling "
                             "right-hand side with a rising left-hand side is a market whose short-term "
                             "trend has turned before its long-term one."), unsafe_allow_html=True)
    st.markdown(viz.universe_panel([s for s in summaries if s]), unsafe_allow_html=True)

    _methodology()

    uni = st.radio("Universe", list(TREND_UNIVERSES), horizontal=True, key="trend_uni",
                   format_func=lambda u: UNIVERSE_LABELS[u].upper(), label_visibility="collapsed")
    _universe_body(uni)


def _methodology() -> None:
    with st.expander("Methodology — seven EWMAC speeds, one equal-weight Trend Score"):
        scal = " · ".join(f"{f}/{s}: {TREND_FORECAST_SCALARS[(f, s)]}"
                          + ("*" if (f, s) in TREND_SCALAR_EXTRAPOLATED else "")
                          for f, s in TREND_SPEEDS)
        st.markdown(
            "For each instrument's **daily** split- and dividend-adjusted close `p`:\n\n"
            "1. `EMA_N = p.ewm(span=N, adjust=True)` — exponential, not simple, averages for the seven "
            "pairs 2/8, 4/16, 8/32, 16/64, 32/128, 64/256, 128/512. A speed stays blank until the series "
            "has as many bars as its slow span.\n"
            f"2. `σ` = exponentially-weighted standard deviation of daily **price changes** (span "
            f"{TREND_VOL_SPAN}), floored at the 5th percentile of its own trailing 500 days.\n"
            "3. `raw = (EMA_fast − EMA_slow) / σ`: how many daily standard deviations apart the two "
            "averages are. It is unitless, so a $20 ETF and a $900 stock are directly comparable, and "
            "a split or dividend adjustment leaves it unchanged.\n"
            f"4. `forecast = clip(raw × scalar, −20, +20)`. Scalars: {scal}. The scalars are fixed "
            "values published by Carver, so each speed averages |forecast| ≈ 10. (*128/512 is not "
            "published; it is extrapolated at the published ladder's own √2-per-doubling ratio.)\n"
            "5. **Trend Score = (f₁ + … + f₇) / 7**: equal weight, with no tilt toward fast or slow "
            "speeds and no diversification multiplier. +20 means all seven speeds are at maximum "
            "bullish strength. Assets with only 4–6 valid speeds (young listings) are scored on those "
            "speeds and flagged *partial*.\n\n"
            "**Direction** of a speed is simply fast EMA above (bullish) or below (bearish) slow EMA. "
            "A **crossover** is the day that flips. That flip is taken from the unrounded signal, and "
            "every one is stored.\n\n"
            f"**States** use the same bands as MOMENTUM: {' · '.join(f'{t:+.0f} {l}' for t, l in MOM_STATES)}. "
            "**Triggers** mark the score entering BULLISH (≥ +5) or BEARISH (< −5). **Upgrades** and "
            "**downgrades** mark any other band crossing. Band events carry a "
            f"±{TREND_BAND_HYSTERESIS:g}-point hysteresis buffer. A move to a stronger band fires at the "
            "line itself, and a move back toward neutral fires only once the score clears the edge by "
            f"{TREND_BAND_HYSTERESIS:g}, so a score hovering on a line is one event, not a daily pair. "
            "A **multi-speed confirmation** is "
            f"≥ {TREND_CONFIRM_MIN_SPEEDS} speeds crossing the same way within {TREND_CONFIRM_WINDOW} "
            "trading days, with all of them still on that side.\n\n"
            "**Not a recommendation.** A positive score says the asset HAS trended up at those "
            "horizons. It is a trend signal, not a forecast of future returns.")
        uni = st.session_state.get("trend_uni", "stocks")
        diag = _artefacts(uni, _status_bust(uni))["diagnostics"]
        if diag.get("realized_abs_longrun"):
            st.markdown(section(f"Calibration check — {UNIVERSE_LABELS[uni]}", 4,
                                help="If a speed's realized average |forecast| drifts far from the target "
                                     "of 10, its fixed scalar no longer fits this universe. This is shown "
                                     "as a check and never silently re-fitted."),
                        unsafe_allow_html=True)
            cdf = pd.DataFrame({
                "Speed": [SPEED_SHORT[k] for k in SPEED_KEYS],
                "Scalar": [diag["scalars"].get(k) for k in SPEED_KEYS],
                "Avg |f| (history)": [diag["realized_abs_longrun"].get(k) for k in SPEED_KEYS],
                "Avg |f| (today)": [diag["realized_abs_today"].get(k) for k in SPEED_KEYS],
                "% capped at ±20 today": [diag["pct_capped_today"].get(k) for k in SPEED_KEYS],
            })

            def build(alt):
                base = alt.Chart(cdf).encode(x=alt.X("Speed:N", sort=None, title=None,
                                                     axis=alt.Axis(labelLimit=0, labelAngle=0)))
                bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=THEME.navy,
                                     size=26).encode(
                    y=alt.Y("Avg |f| (history):Q", title="Average |forecast|",
                            scale=alt.Scale(domain=[0, 20])),
                    tooltip=list(cdf.columns))
                target = alt.Chart(pd.DataFrame({"y": [10]})).mark_rule(
                    color=THEME.mustard, strokeWidth=2).encode(y="y:Q")
                return (bars + target).properties(height=190)
            uc.render_chart(build, fallback=cdf)
            st.caption("Mustard line = Carver's target of 10. Bars = each speed's average |forecast| "
                       "across every asset's full history.")
            st.dataframe(cdf.style.format({"Avg |f| (history)": "{:.1f}", "Avg |f| (today)": "{:.1f}",
                                           "% capped at ±20 today": "{:.0%}"}, na_rep="—"),
                         use_container_width=True, hide_index=True)
            corr = diag.get("speed_correlation") or {}
            if corr:
                cm = pd.DataFrame([{"a": SPEED_SHORT[a], "b": SPEED_SHORT[b], "rho": corr[a][b]}
                                   for a in SPEED_KEYS for b in SPEED_KEYS])

                def build_corr(alt):
                    order = [SPEED_SHORT[k] for k in SPEED_KEYS]
                    return (alt.Chart(cm).mark_rect(stroke=THEME.bg, strokeWidth=2).encode(
                        x=alt.X("a:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
                        y=alt.Y("b:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
                        color=alt.Color("rho:Q", scale=uc.diverging_scale(1.0, alt), title="Spearman ρ"),
                        tooltip=["a", "b", alt.Tooltip("rho:Q", format=".2f")],
                    ).properties(height=240))
                st.markdown(section("How correlated are the seven speeds today?", 5,
                                    help="Cross-sectional Spearman correlation of today's forecasts. "
                                         "Adjacent speeds overlap heavily by construction, and 2/8 vs "
                                         "128/512 is where the independent information lives."),
                            unsafe_allow_html=True)
                uc.render_chart(build_corr, fallback=cm)


# ======================================================== universe body ====
@st.fragment
def _universe_body(universe: str) -> None:
    art = _artefacts(universe, _status_bust(universe))
    latest = art["latest"]
    if not latest.get("rows"):
        st.info(f"No {UNIVERSE_LABELS[universe]} data yet — run "
                f"`python -m zenith.trend.compute --universe {universe}`.")
        return
    st.markdown(stamp(latest.get("as_of", "—"), f"TREND FOLLOWING · {UNIVERSE_LABELS[universe]}"),
                unsafe_allow_html=True)
    _status_line(universe, art)
    key = f"trend_sub_{universe}"
    pending = st.session_state.pop(f"trend_nav_{universe}", None)
    if pending in SUBVIEWS:
        st.session_state[key] = pending
    sub = st.radio("View", SUBVIEWS, horizontal=True, key=key, label_visibility="collapsed")
    if sub == "Overview":
        _overview(universe, art)
    elif sub == "All Trends":
        _all_trends(universe, art)
    elif sub == "Triggers":
        _triggers(universe, art)
    else:
        _detail(universe, art)


def _navigate(universe: str, sub: str, consumed_key: str, **state) -> None:
    """Switch sub-view from a click. The clicked chart/table's selection is
    cleared first -- a selection persists in widget state, and without this,
    coming back to the originating view would re-fire the old click and bounce
    straight back out. Reruns only this fragment when possible (a fragment
    rerun), or the app when the click was processed during a full-app run
    (where a fragment-scoped rerun is not allowed)."""
    st.session_state.pop(consumed_key, None)
    for k, v in state.items():
        st.session_state[k] = v
    # The sub-view radio is already instantiated in this run, so its key
    # cannot be written now; hand the target over and apply it before the
    # radio is created on the rerun.
    st.session_state[f"trend_nav_{universe}"] = sub
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _open_detail(universe: str, ticker: str, consumed_key: str) -> None:
    _navigate(universe, "Detail", consumed_key, **{f"trend_pick_{universe}": ticker})


def _status_line(universe: str, art: dict) -> None:
    latest = art["latest"]
    n, n_sc = latest.get("n", 0), latest.get("n_scored", 0)
    reasons: dict[str, int] = {}
    for r in latest.get("rows", []):
        if r.get("excluded"):
            k = (r.get("exclusion_reason") or "unknown").split("(")[0]
            reasons[k] = reasons.get(k, 0) + 1
    src = "MOMENTUM (Russell 1000)" if universe == "stocks" else "ETF MOMENTUM"
    sync = ""
    mom = _momentum(universe, _bust(art))
    if mom:
        ours = {r["ticker"] for r in latest.get("rows", [])}
        theirs = set(mom)
        both = len(ours & theirs)
        sync = f" · universe sync with {src}: {both}/{len(theirs)} of its scored names present"
    excl = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
    st.caption(f"{n_sc} of {n} {src} constituents scored"
               + (f" · {latest.get('n_partial', 0)} partial (4–6 speeds)" if latest.get("n_partial") else "")
               + (f" · excluded: {excl}" if excl else "") + sync)


# ============================================================ overview ====
def _overview(universe: str, art: dict) -> None:
    df = _frame(universe, _bust(art))
    if df.empty:
        st.info("No scored names yet.")
        return
    n = len(df)
    pct_bull, pct_bear = float((df["score"] >= 5).mean()), float((df["score"] < -5).mean())
    med = float(df["score"].median())
    counts = df["structure"].value_counts()
    evs = _recent_events(art)
    as_of = art["latest"].get("as_of")
    wk = (pd.Timestamp(as_of) - pd.tseries.offsets.BDay(5)).strftime("%Y-%m-%d") if as_of else ""
    recent = [e for e in evs if e["date"] > wk and e["type"] != "cross"]
    n_bt = sum(1 for e in recent if e["type"] == "trigger" and e["dir"] > 0)
    n_brt = sum(1 for e in recent if e["type"] == "trigger" and e["dir"] < 0)
    n_conf = sum(1 for e in recent if e["type"] == "confirmation")
    color = THEME.teal if pct_bull >= pct_bear else THEME.coral
    st.markdown(uc.state_banner(color, "TREND BREADTH",
                                f"{pct_bull:.0%} bullish · {pct_bear:.0%} bearish · median {med:+.1f}"),
                unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Scored", "value": str(n), "color": THEME.text},
        {"label": "Median score", "value": f"{med:+.1f}", "color": THEME.text,
         "sub": f"mean {df['score'].mean():+.1f}"},
        {"label": "Persistent up", "value": str(int(counts.get("Persistent Uptrend", 0))), "color": THEME.teal,
         "sub": "≥6 of 7 speeds bullish"},
        {"label": "Persistent down", "value": str(int(counts.get("Persistent Downtrend", 0))),
         "color": THEME.coral, "sub": "≥6 of 7 speeds bearish"},
        {"label": "Fast/slow disagree", "value": str(int(df["disagree"].sum())), "color": THEME.mustard,
         "sub": "emerging or deteriorating"},
        {"label": "Bullish triggers · 5d", "value": str(n_bt), "color": THEME.teal},
        {"label": "Bearish triggers · 5d", "value": str(n_brt), "color": THEME.coral},
        {"label": "Confirmations · 5d", "value": str(n_conf), "color": THEME.mint,
         "sub": f"≥{TREND_CONFIRM_MIN_SPEEDS} speeds, same way"},
    ], min_width=130), unsafe_allow_html=True)

    st.markdown(section("Where the universe sits on the trend spectrum", 0,
                        help="Every scored asset placed on the -20..+20 scale. Bars are counts per "
                             "integer score; the white line is the median."), unsafe_allow_html=True)
    st.markdown(viz.spectrum_strip(df["score"].tolist()), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(section("Breadth by speed — the market's own term structure", 4,
                            help="Share of assets whose fast EMA is above the slow EMA at each speed. "
                                 "Falling from left to right means short-term trends are weaker than "
                                 "long-term ones (a pullback in an uptrend). Rising from left to right "
                                 "means short-term trends are recovering first."),
                    unsafe_allow_html=True)
        bdf = pd.DataFrame({"Speed": [SPEED_SHORT[k] for k in SPEED_KEYS],
                            "Name": [SPEED_NAMES[k] for k in SPEED_KEYS],
                            "Bullish": [float((df[f"f_{k}"] > 0).sum() / max(1, df[f"f_{k}"].notna().sum()))
                                        for k in SPEED_KEYS]})
        bdf["tilt"] = (bdf["Bullish"] - 0.5) * 40

        def build_b(alt):
            base = alt.Chart(bdf).encode(x=alt.X("Speed:N", sort=None, title="fast → slow",
                                                 axis=alt.Axis(labelAngle=0, labelLimit=0)))
            bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=30).encode(
                y=alt.Y("Bullish:Q", title="% bullish", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("tilt:Q", scale=uc.diverging_scale(20, alt), legend=None),
                tooltip=["Speed", "Name", alt.Tooltip("Bullish:Q", format=".1%")])
            labels = base.mark_text(dy=-8, color=THEME.text, fontSize=11).encode(
                y="Bullish:Q", text=alt.Text("Bullish:Q", format=".0%"))
            half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=THEME.muted).encode(y="y:Q")
            return (bars + labels + half).properties(height=260)
        uc.render_chart(build_b, fallback=bdf)
    with c2:
        st.markdown(section("Structure mix", 1,
                            help="Click a bar to open All Trends filtered to that structure."),
                    unsafe_allow_html=True)
        sdf = pd.DataFrame({"Structure": list(STRUCTURES),
                            "Count": [int(counts.get(s, 0)) for s in STRUCTURES],
                            "What it means": [STRUCTURE_HELP[s] for s in STRUCTURES]})

        def build_s(alt):
            pick = alt.selection_point(name="pick", fields=["Structure"], on="click", empty=False)
            return (alt.Chart(sdf).mark_bar(cornerRadiusEnd=3, size=22).encode(
                y=alt.Y("Structure:N", sort=list(STRUCTURES), title=None, axis=alt.Axis(labelLimit=0)),
                x=alt.X("Count:Q", title="assets"),
                color=alt.Color("Structure:N", scale=alt.Scale(domain=list(STRUCTURES),
                                                               range=[STRUCTURE_COLORS[s] for s in STRUCTURES]),
                                legend=None),
                opacity=alt.condition(pick, alt.value(1.0), alt.value(0.85)),
                tooltip=["Structure", "Count", "What it means"],
            ).add_params(pick).properties(height=260))
        evt = uc.render_chart(build_s, fallback=sdf, on_select="rerun", key=f"trend_struct_{universe}")
        picked = ((evt or {}).get("selection", {}).get("pick") or [{}])[0].get("Structure")
        if picked:
            _navigate(universe, "All Trends", f"trend_struct_{universe}",
                      **{f"trend_f_struct_{universe}": [picked]})

    _breadth_timeline(universe, art)
    _structure_map(universe, df)
    _group_heatmap(universe, df)
    _trend_vs_momentum(universe, df)


def _breadth_timeline(universe: str, art: dict) -> None:
    rows = art["breadth"].get("rows", [])
    if len(rows) < 20:
        return
    st.markdown(section("How breadth at each speed has evolved", 2,
                        help="Each row is one speed; color is the share of the universe bullish at that "
                             "speed that day (teal > 50%, coral < 50%). A wave that starts in the top "
                             "rows and travels down is a trend change spreading from fast speeds to "
                             "slow ones."), unsafe_allow_html=True)
    span = st.select_slider("Window", ["6M", "1Y", "2Y", "5Y", "All"], value="2Y",
                            key=f"trend_bt_span_{universe}")
    days = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": 10**6}[span]
    rr = rows[-days:]
    step = max(1, len(rr) // 400)                     # at most ~400 columns x 7 rows
    rr = rr[::step]
    nxt = {r["date"]: (rr[i + 1]["date"] if i + 1 < len(rr) else r["date"]) for i, r in enumerate(rr)}
    long = pd.DataFrame([{"date": r["date"], "date2": nxt[r["date"]], "Speed": SPEED_SHORT[k],
                          "Bullish": r["speed_bull"][j],
                          "tilt": None if r["speed_bull"][j] is None else (r["speed_bull"][j] - 0.5) * 40}
                         for r in rr for j, k in enumerate(SPEED_KEYS)])
    med = pd.DataFrame([{"date": r["date"], "Median score": r.get("median")} for r in rr])

    def build(alt):
        order = [SPEED_SHORT[k] for k in SPEED_KEYS]
        # explicit x/x2: a rect on a continuous time axis has no width of its
        # own, and without x2 every cell overdraws the whole row
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %y")), x2="date2:T",
            y=alt.Y("Speed:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
            color=alt.Color("tilt:Q", scale=uc.diverging_scale(20, alt), legend=None),
            tooltip=[alt.Tooltip("date:T"), "Speed", alt.Tooltip("Bullish:Q", format=".0%")],
        ).properties(height=170)
        line = alt.Chart(med).mark_line(color=THEME.text, strokeWidth=1.5).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %y")),
            y=alt.Y("Median score:Q", title="median score", scale=alt.Scale(domain=[-20, 20])),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("Median score:Q", format="+.1f")],
        ).properties(height=110)
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.grid).encode(y="y:Q")
        return alt.vconcat(heat, zero + line, spacing=6).resolve_scale(x="shared")
    uc.render_chart(build, fallback=med)
    if rr and rr[0]["date"] < MOM_MEMBERSHIP_START and universe == "stocks":
        st.caption(f"Universe-level history before {MOM_MEMBERSHIP_START} uses today's Russell 1000 "
                   "constituents (survivorship-biased breadth). Each single stock's own signal history "
                   "has no such bias.")


def _structure_map(universe: str, df: pd.DataFrame) -> None:
    st.markdown(section("Trend Structure Map — fast vs slow", 3,
                        help="x = average forecast of the three slowest speeds (32/128, 64/256, 128/512); "
                             "y = the three fastest (2/8, 4/16, 8/32). Top-right: trending up at every "
                             "horizon. Top-left: only the short-term trend has turned up. Bottom-right: "
                             "a long uptrend whose short-term trend has rolled over. Click a dot for "
                             "its detail."), unsafe_allow_html=True)
    m = df.dropna(subset=["fast", "slow"])[["ticker", "name", "group", "score", "fast", "slow",
                                            "structure", "n_bull"]].copy()
    if m.empty:
        return
    quad = pd.DataFrame({"x": [15, -15, -15, 15], "y": [19, 19, -19, -19],
                         "t": ["PERSISTENT UP", "EMERGING UP", "PERSISTENT DOWN", "DETERIORATING UP"],
                         "c": [THEME.teal, THEME.mint, THEME.coral, THEME.mustard]})

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["ticker"], on="click", empty=False)
        axis = alt.Chart(pd.DataFrame({"v": [0]}))
        pts = alt.Chart(m).mark_circle(size=46, stroke="#6b6b6b", strokeWidth=0.6).encode(
            x=alt.X("slow:Q", title="slow speeds (32/128 · 64/256 · 128/512) →",
                    scale=alt.Scale(domain=[-20, 20])),
            y=alt.Y("fast:Q", title="fast speeds (2/8 · 4/16 · 8/32) →", scale=alt.Scale(domain=[-20, 20])),
            color=alt.Color("score:Q", scale=uc.diverging_scale(20, alt), title="Trend Score"),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.85)),
            tooltip=["ticker", "name", "group", alt.Tooltip("score:Q", format="+.1f"),
                     alt.Tooltip("fast:Q", format="+.1f"), alt.Tooltip("slow:Q", format="+.1f"),
                     "structure", alt.Tooltip("n_bull:Q", title="speeds bullish")],
        ).add_params(pick)
        labels = alt.Chart(quad).mark_text(fontSize=11, opacity=0.8, font="Space Mono").encode(
            x="x:Q", y="y:Q", text="t:N", color=alt.Color("c:N", scale=None))
        return (axis.mark_rule(color=THEME.muted).encode(x="v:Q")
                + axis.mark_rule(color=THEME.muted).encode(y="v:Q")
                + labels + pts).properties(height=520)
    evt = uc.render_chart(build, fallback=m, on_select="rerun", key=f"trend_map_{universe}")
    picked = ((evt or {}).get("selection", {}).get("pick") or [{}])[0].get("ticker")
    if picked:
        _open_detail(universe, picked, f"trend_map_{universe}")


def _group_heatmap(universe: str, df: pd.DataFrame) -> None:
    gl = GROUP_LABEL[universe]
    st.markdown(section(f"{gl} × speed — where each trend lives", 5,
                        help=f"Average forecast of every {gl.lower()} at each speed, plus its average "
                             "Trend Score. Click a cell to screen that group in All Trends."),
                unsafe_allow_html=True)
    g = df.groupby("group")
    rows = []
    for name, sub in g:
        if not name:
            name = "Unknown"
        for k in SPEED_KEYS:
            rows.append({"group": name, "Speed": SPEED_SHORT[k], "v": sub[f"f_{k}"].mean(), "n": len(sub)})
        rows.append({"group": name, "Speed": "SCORE", "v": sub["score"].mean(), "n": len(sub)})
    hm = pd.DataFrame(rows)
    hm["label"] = hm["v"].apply(lambda v: "" if pd.isna(v) else ("0" if round(v) == 0 else f"{round(v):+d}"))
    order = (hm[hm["Speed"] == "SCORE"].sort_values("v", ascending=False)["group"].tolist())
    cols = [SPEED_SHORT[k] for k in SPEED_KEYS] + ["SCORE"]

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["group"], on="click", empty=False)
        base = alt.Chart(hm).encode(
            x=alt.X("Speed:N", sort=cols, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0, orient="top")),
            y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)))
        rect = base.mark_rect(stroke=THEME.bg, strokeWidth=2).encode(
            color=alt.Color("v:Q", scale=uc.diverging_scale(12, alt), title="avg forecast"),
            tooltip=[alt.Tooltip("group:N", title=gl), "Speed", alt.Tooltip("v:Q", format="+.1f"),
                     alt.Tooltip("n:Q", title="assets")]).add_params(pick)
        text = base.mark_text(fontSize=10, color=THEME.text).encode(text="label:N")
        return (rect + text).properties(height=max(200, 26 * len(order)))
    evt = uc.render_chart(build, fallback=hm, on_select="rerun", key=f"trend_grp_{universe}")
    picked = ((evt or {}).get("selection", {}).get("pick") or [{}])[0].get("group")
    if picked:
        _navigate(universe, "All Trends", f"trend_grp_{universe}",
                  **{f"trend_f_group_{universe}": [picked]})


def _trend_vs_momentum(universe: str, df: pd.DataFrame) -> None:
    m = df.dropna(subset=["momentum"])
    if len(m) < 10:
        return
    src = "MOMENTUM" if universe == "stocks" else "ETF MOMENTUM"
    # Spearman = Pearson on ranks. (Series.corr(method="spearman") imports
    # scipy, which is not a Zenith dependency.)
    rho = m["score"].rank().corr(m["momentum"].rank())
    st.markdown(section(f"Trend Following vs {src}", 1,
                        help=f"Same asset, two -20..+20 scores. {src} blends six factors, including "
                             "cross-sectional ranks. Trend Following is purely the asset's own "
                             "seven-speed trend. Points far off the diagonal are where the two methods "
                             "disagree. Click a dot for its detail."), unsafe_allow_html=True)

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["ticker"], on="click", empty=False)
        diag = alt.Chart(pd.DataFrame({"x": [-20, 20], "y": [-20, 20]})).mark_line(
            color=THEME.grid, strokeWidth=1).encode(x="x:Q", y="y:Q")
        pts = alt.Chart(m).mark_circle(size=40, stroke="#6b6b6b", strokeWidth=0.6).encode(
            x=alt.X("momentum:Q", title=f"{src} composite", scale=alt.Scale(domain=[-20, 20])),
            y=alt.Y("score:Q", title="Trend Score", scale=alt.Scale(domain=[-20, 20])),
            color=alt.Color("trend_minus_mom:Q", scale=uc.diverging_scale(15, alt), title="Trend − Mom"),
            tooltip=["ticker", "name", alt.Tooltip("score:Q", format="+.1f", title="Trend"),
                     alt.Tooltip("momentum:Q", format="+.1f", title="Momentum"), "structure"],
        ).add_params(pick)
        return (diag + pts).properties(height=380)
    evt = uc.render_chart(build, fallback=m[["ticker", "score", "momentum"]], on_select="rerun",
                          key=f"trend_vsmom_{universe}")
    st.caption(f"Rank correlation between the two scores today: ρ = {rho:.2f} across {len(m)} assets.")
    picked = ((evt or {}).get("selection", {}).get("pick") or [{}])[0].get("ticker")
    if picked:
        _open_detail(universe, picked, f"trend_vsmom_{universe}")


# ========================================================== all trends ====
_PRESETS = ["All", "Top 25", "Top 50", "Bottom 25", "Bottom 50", "Persistent trends (up & down)",
            "Emerging uptrends", "Deteriorating uptrends", "Fast/slow disagreement",
            "Fresh confirmation (≤5d)", "Big movers (|Δ5d| ≥ 4)"]


def _apply_preset(df: pd.DataFrame, preset: str, as_of: str | None) -> pd.DataFrame:
    if preset == "Persistent trends (up & down)":
        return df[df["structure"].isin(["Persistent Uptrend", "Persistent Downtrend"])]
    if preset == "Emerging uptrends":
        return df[df["structure"] == "Emerging Uptrend"]
    if preset == "Deteriorating uptrends":
        return df[df["structure"] == "Deteriorating Uptrend"]
    if preset == "Fast/slow disagreement":
        return df[df["disagree"].fillna(False).astype(bool)]
    if preset == "Fresh confirmation (≤5d)" and as_of:
        cut = (pd.Timestamp(as_of) - pd.tseries.offsets.BDay(5)).strftime("%Y-%m-%d")
        return df[df["last_confirmation"].apply(lambda c: isinstance(c, dict) and (c.get("date") or "") > cut)]
    if preset == "Big movers (|Δ5d| ≥ 4)":
        return df[df["d5"].abs() >= 4]
    return df


def _all_trends(universe: str, art: dict) -> None:
    df = _frame(universe, _bust(art))
    if df.empty:
        st.info("No scored names yet.")
        return
    gcol, glabel = "group", GROUP_LABEL[universe]
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    q = c1.text_input("Search ticker or name", "", key=f"trend_q_{universe}")
    groups = c2.multiselect(glabel, sorted(g for g in df["group"].dropna().unique() if g),
                            key=f"trend_f_group_{universe}")
    structs = c3.multiselect("Structure", list(STRUCTURES), key=f"trend_f_struct_{universe}")
    preset = c4.selectbox("Preset", _PRESETS, key=f"trend_preset_{universe}")
    s1, s2, s3, s4 = st.columns([2, 1, 2, 2])
    sort_key = s1.selectbox("Sort by", list(table.SORTS), key=f"trend_sort_{universe}")
    flip = s2.toggle("Reverse", key=f"trend_sortrev_{universe}")
    n_cash = int(df["cash_like"].fillna(False).sum()) if "cash_like" in df else 0
    hide_cash = s2.toggle(f"Hide cash-like ({n_cash})", value=universe == "etfs", disabled=not n_cash,
                          key=f"trend_hidecash_{universe}",
                          help=f"T-bill / ultrashort funds (1-year annualized volatility below "
                               f"{TREND_CASH_LIKE_VOL:.1%}) drift up with almost no volatility, so a "
                               "vol-normalized EWMAC correctly pins them near +20. Flagged, never "
                               "re-scored; hidden here by default so they don't crowd the top of the "
                               "ETF screen.")
    mode = s3.radio("Display", ["Table", "Heatmap"], horizontal=True, key=f"trend_mode_{universe}")
    with s4.popover("More filters", use_container_width=True):
        rng = st.slider("Trend Score", -20.0, 20.0, (-20.0, 20.0), 0.5, key=f"trend_rng_{universe}")
        agree = st.slider("Min speeds agreeing with the score's direction", 0, 7, 0,
                          key=f"trend_agree_{universe}")
        hide_partial = st.checkbox("Hide partial rows (fewer than 7 valid speeds)",
                                   key=f"trend_hidepart_{universe}")

    view = table.filter_rows(df, query=q, groups=groups, group_col=gcol, structures=structs,
                             score_range=rng if rng != (-20.0, 20.0) else None,
                             hide_partial=hide_partial, min_bull=agree or None,
                             hide_cash_like=hide_cash)
    view = _apply_preset(view, preset, art["latest"].get("as_of"))
    _, desc = table.SORTS[sort_key]
    view = table.sort(view, sort_key, ascending=(desc if flip else not desc))
    if preset.startswith("Top"):
        view = table.sort(view, "Trend Score").head(int(preset.split()[1]))
    elif preset.startswith("Bottom"):
        view = table.sort(view, "Trend Score", ascending=True).head(int(preset.split()[1]))
    st.caption(f"{len(view)} of {len(df)} scored · sorted by {sort_key}"
               f"{' (reversed)' if flip else ''}"
               + (f" · {n_cash} cash-like hidden" if hide_cash and n_cash else ""))
    if view.empty:
        st.info("Nothing matches these filters.")
        return
    if mode == "Heatmap":
        _heatmap(universe, view)
    else:
        _table(universe, view, glabel)
    st.download_button("Download this view (CSV)", _export(view).to_csv(index=False).encode("utf-8"),
                       file_name=f"trend_{universe}_{art['latest'].get('as_of', '')}.csv",
                       mime="text/csv", key=f"trend_dl_{universe}")


def _export(view: pd.DataFrame) -> pd.DataFrame:
    out = view[["rank", "ticker", "name", "group", "score", "state", "structure", "d5", "d20", "slope",
                "n_bull", "momentum"]].copy()
    for k in SPEED_KEYS:
        out[f"ewmac_{k}"] = view[f"f_{k}"]
    return out


def _table(universe: str, view: pd.DataFrame, glabel: str) -> None:
    speed_cols = [SPEED_SHORT[k] for k in SPEED_KEYS]
    disp = pd.DataFrame({
        "Rank": view["rank"].astype("Int64"), "Ticker": view["ticker"], "Name": view["name"],
        glabel: view["group"], "Score": view["score"], "Structure": view["structure"],
        **{SPEED_SHORT[k]: view[f"f_{k}"] for k in SPEED_KEYS},
        "Term structure": view["forecasts"].apply(lambda v: [x if x is not None else 0.0 for x in v]),
        "60d": view["spark"].apply(lambda v: [x for x in (v or []) if x is not None]),
        "Δ5d": view["d5"], "Δ20d": view["d20"], "Accel": view["slope"],
        "Agree": view.apply(lambda r: f"{int(r['agree'])}/{int(r['n_valid'])}", axis=1),
        "Last event": view["last_event_date"],
        "Momentum": view["momentum"], "Trend − Mom": view["trend_minus_mom"],
    }).reset_index(drop=True)
    cfg = uc.colcfg(disp.columns, COLS)
    cfg["Term structure"] = st.column_config.BarChartColumn(
        "Term structure", help=COLS["Term structure"], y_min=-20, y_max=20, width="small")
    cfg["60d"] = st.column_config.LineChartColumn("60d", help=COLS["60d"], y_min=-20, y_max=20,
                                                  width="small")
    cfg["Ticker"] = st.column_config.TextColumn("Ticker", help=COLS["Ticker"], pinned=True)
    num = ["Score", *speed_cols, "Δ5d", "Δ20d", "Accel", "Momentum", "Trend − Mom"]
    try:
        sty = (disp.style
               .map(lambda v: uc.grad_diverging(v, 20.0), subset=["Score", *speed_cols, "Momentum"])
               .map(lambda v: uc.grad_diverging(v, 8.0), subset=["Δ5d", "Δ20d", "Trend − Mom"])
               .map(lambda v: uc.grad_diverging(v, 15.0), subset=["Accel"])
               .format({c: "{:+.1f}" for c in num}, na_rep="—"))
        data = sty
    except Exception:
        data = disp
    evt = st.dataframe(data, use_container_width=True, hide_index=True, column_config=cfg,
                       height=min(720, 40 + 35 * len(disp)), on_select="rerun",
                       selection_mode="single-row", key=f"trend_table_{universe}")
    rows = (evt or {}).get("selection", {}).get("rows", []) if evt else []
    if rows:
        _open_detail(universe, disp.iloc[rows[0]]["Ticker"], f"trend_table_{universe}")
    st.caption("Select a row to open its Detail view. Speed columns are colored teal (bullish) to "
               "coral (bearish) by strength, so reading across a row shows the term structure.")


def _heatmap(universe: str, view: pd.DataFrame) -> None:
    n_show = st.select_slider("Rows", [25, 50, 100, 250, 500], value=100, key=f"trend_hm_n_{universe}")
    page = view.head(n_show)
    order = page["ticker"].tolist()
    cols = [SPEED_SHORT[k] for k in SPEED_KEYS] + ["SCORE"]
    long = pd.DataFrame([{"ticker": r.ticker, "name": r.name, "structure": r.structure,
                          "Speed": SPEED_SHORT[k], "v": getattr(r, f"f_{k}")}
                         for r in page.itertuples() for k in SPEED_KEYS]
                        + [{"ticker": r.ticker, "name": r.name, "structure": r.structure,
                            "Speed": "SCORE", "v": r.score} for r in page.itertuples()])
    row_h = 16 if n_show <= 100 else 11

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["ticker"], on="click", empty=False)
        base = alt.Chart(long).encode(
            x=alt.X("Speed:N", sort=cols, title=None, axis=alt.Axis(labelAngle=0, orient="top", labelLimit=0)),
            y=alt.Y("ticker:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=0, labelFontSize=10 if n_show <= 100 else 8)))
        rect = base.mark_rect(stroke=THEME.bg, strokeWidth=1).encode(
            color=alt.Color("v:Q", scale=uc.diverging_scale(20, alt), title="forecast / score"),
            tooltip=["ticker", "name", "structure", "Speed", alt.Tooltip("v:Q", format="+.1f")],
        ).add_params(pick)
        return rect.properties(height=row_h * len(order), width=alt.Step(64))
    evt = uc.render_chart(build, fallback=page[["ticker", "score"]], on_select="rerun",
                          key=f"trend_hm_{universe}")
    picked = ((evt or {}).get("selection", {}).get("pick") or [{}])[0].get("ticker")
    if picked:
        _open_detail(universe, picked, f"trend_hm_{universe}")
    st.caption("Rows follow the current sort. Read across a row: a solid teal row trends up at every "
               "horizon, and a row teal only on the left has turned up short-term only. Click a row to "
               "open its detail.")


# ============================================================= triggers ====
_TYPE_FILTERS = {
    "Bullish triggers": lambda e: e["type"] == "trigger" and e["dir"] > 0,
    "Bearish triggers": lambda e: e["type"] == "trigger" and e["dir"] < 0,
    "Signal upgrades": lambda e: e["type"] == "upgrade",
    "Signal downgrades": lambda e: e["type"] == "downgrade",
    "Multi-speed confirmations": lambda e: e["type"] == "confirmation",
    "Single-speed crossovers": lambda e: e["type"] == "cross",
}


def _triggers(universe: str, art: dict) -> None:
    evs = _recent_events(art)
    if not evs:
        st.info("No recent signal changes recorded yet.")
        return
    df = _frame(universe, _bust(art))
    names = dict(zip(df["ticker"], df["name"])) if not df.empty else {}
    fc = dict(zip(df["ticker"], df["forecasts"])) if not df.empty else {}
    as_of = art["latest"].get("as_of")

    c1, c2, c3, c4 = st.columns([1, 3, 1, 2])
    window = c1.selectbox("Window", ["1 day", "5 days", "20 days", "60 days"], index=1,
                          key=f"trend_tw_{universe}")
    types = c2.multiselect("Event types", list(_TYPE_FILTERS),
                           default=[t for t in _TYPE_FILTERS if t != "Single-speed crossovers"],
                           key=f"trend_tt_{universe}")
    direction = c3.selectbox("Direction", ["All", "Bullish", "Bearish"], key=f"trend_td_{universe}")
    sort = c4.selectbox("Sort", ["Most recent", "Largest score change", "Most speeds changing"],
                        key=f"trend_ts_{universe}")
    d1, d2, d3 = st.columns([2, 3, 2])
    min_speeds = d1.slider("Min speeds involved", 1, 7, 1, key=f"trend_tms_{universe}",
                           help="Applies to crossovers and confirmations. Score events have no speeds.")
    horizons = d2.multiselect("Horizon", ["short", "medium", "long", "composite"],
                              default=["short", "medium", "long", "composite"],
                              key=f"trend_th_{universe}",
                              help="short = 2/8–8/32, medium = 16/64, long = 32/128–128/512, "
                                   "composite = the overall score. A confirmation takes the horizon "
                                   "of its slowest speed.")
    tq = d3.text_input("Ticker", "", key=f"trend_tq_{universe}")

    n_days = int(window.split()[0])
    cut = (pd.Timestamp(as_of) - pd.tseries.offsets.BDay(n_days)).strftime("%Y-%m-%d")
    sel = [e for e in evs if e["date"] > cut]
    if types:
        sel = [e for e in sel if any(_TYPE_FILTERS[t](e) for t in types)]
    if direction != "All":
        sel = [e for e in sel if (e["dir"] > 0) == (direction == "Bullish")]
    sel = [e for e in sel if e["type"] not in ("cross", "confirmation") or e["n_speeds"] >= min_speeds]
    sel = [e for e in sel if (e.get("horizon") or "composite") in horizons]
    if tq.strip():
        sel = [e for e in sel if tq.strip().upper() in e["ticker"]]
    if n_days > TREND_RECENT_CROSS_DAYS and "Single-speed crossovers" in types:
        st.caption(f"Single-speed crossovers are kept for the last {TREND_RECENT_CROSS_DAYS} trading days "
                   "here (every crossover ever is in the Detail view's history).")

    def _chg(e):
        a, b = e.get("score_after"), e.get("score_before")
        return abs(a - b) if a is not None and b is not None else 0.0
    if sort == "Largest score change":
        sel.sort(key=_chg, reverse=True)
    elif sort == "Most speeds changing":
        sel.sort(key=lambda e: (e["n_speeds"], e["date"]), reverse=True)
    else:
        sel.sort(key=lambda e: (e["date"], e["type"] == "confirmation", e["n_speeds"]), reverse=True)

    def _count(fn):
        return sum(1 for e in sel if fn(e))
    st.markdown(uc.numeric_slab([
        {"label": "Events", "value": str(len(sel)), "color": THEME.text, "sub": f"last {window}"},
        {"label": "Bullish triggers", "value": str(_count(_TYPE_FILTERS["Bullish triggers"])), "color": THEME.teal},
        {"label": "Bearish triggers", "value": str(_count(_TYPE_FILTERS["Bearish triggers"])), "color": THEME.coral},
        {"label": "Upgrades / downgrades", "value": f"{_count(_TYPE_FILTERS['Signal upgrades'])} / "
                                                     f"{_count(_TYPE_FILTERS['Signal downgrades'])}",
         "color": THEME.text},
        {"label": "Multi-speed confirmations", "value": str(_count(_TYPE_FILTERS["Multi-speed confirmations"])),
         "color": THEME.mint},
    ], min_width=140), unsafe_allow_html=True)

    _event_timeline(evs, types, direction)

    st.markdown(section("Feed — what just changed", 2,
                        help="Multi-speed confirmations are drawn larger with a gradient border so they "
                             "stand out from single-speed flips. The dots show which of the seven speeds "
                             "took part, fast (left) to slow (right). The small ribbon is the asset's "
                             "seven forecasts today."), unsafe_allow_html=True)
    show = sel[:60]
    st.markdown("".join(viz.event_card(e, names.get(e["ticker"], ""), fc.get(e["ticker"]))
                        for e in show), unsafe_allow_html=True)
    if len(sel) > len(show):
        st.caption(f"Showing the first {len(show)} of {len(sel)}. Every event is in the table below.")

    st.markdown(section("All matching events", 4), unsafe_allow_html=True)
    tdf = pd.DataFrame([{"Date": e["date"], "Ticker": e["ticker"], "Name": names.get(e["ticker"], ""),
                         "Event": viz.event_title(e),
                         "Direction": "Bullish" if e["dir"] > 0 else "Bearish",
                         "Speeds": ", ".join(SPEED_SHORT[k] for k in e["speeds"]) or "composite",
                         "# speeds": e["n_speeds"], "Horizon": e.get("horizon"),
                         "Score before": e.get("score_before"), "Score after": e.get("score_after"),
                         "From": e.get("from"), "To": e.get("to")} for e in sel])
    if tdf.empty:
        st.info("No events match these filters.")
        return
    evt = st.dataframe(tdf.style.format({"Score before": "{:+.1f}", "Score after": "{:+.1f}"}, na_rep="—"),
                       use_container_width=True, hide_index=True, height=min(520, 40 + 35 * len(tdf)),
                       on_select="rerun", selection_mode="single-row", key=f"trend_evtable_{universe}")
    rows = (evt or {}).get("selection", {}).get("rows", []) if evt else []
    if rows and tdf.iloc[rows[0]]["Ticker"] in names:
        _open_detail(universe, tdf.iloc[rows[0]]["Ticker"], f"trend_evtable_{universe}")


def _event_timeline(evs: list[dict], types: list[str], direction: str) -> None:
    fs = [_TYPE_FILTERS[t] for t in types] or list(_TYPE_FILTERS.values())
    use = [e for e in evs if e["type"] != "cross" and any(f(e) for f in fs)]
    if direction != "All":
        use = [e for e in use if (e["dir"] > 0) == (direction == "Bullish")]
    if not use:
        return
    agg: dict[tuple, int] = {}
    for e in use:
        k = (e["date"], "Bullish" if e["dir"] > 0 else "Bearish")
        agg[k] = agg.get(k, 0) + 1
    # float, not int64: an int64 column reaches Vega through Arrow as BigInt,
    # which its scales cannot use (the y-domain silently becomes infinite)
    tl = pd.DataFrame([{"date": d, "Direction": s, "events": float(n if s == "Bullish" else -n)}
                       for (d, s), n in agg.items()])
    st.markdown(section("Signal changes per day", 5,
                        help="Bullish events (triggers, upgrades, bullish confirmations) above the line; "
                             "bearish ones below. Spikes are broad, market-wide trend turns."),
                unsafe_allow_html=True)

    def build(alt):
        bars = alt.Chart(tl).mark_bar(size=5).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
            # one bullish (+) and one bearish (-) bar per day: nothing to stack
            y=alt.Y("events:Q", title="events", stack=None),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Bullish", "Bearish"],
                                                            range=[THEME.teal, THEME.coral]),
                            legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("date:T"), "Direction", alt.Tooltip("events:Q", format=".0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=THEME.muted).encode(y="y:Q")
        return (bars + zero).properties(height=220)
    uc.render_chart(build, fallback=tl)


# =============================================================== detail ====
def _detail(universe: str, art: dict) -> None:
    df = _frame(universe, _bust(art))
    if df.empty:
        st.info("No scored names yet.")
        return
    tickers = df.sort_values("ticker")["ticker"].tolist()
    names = dict(zip(df["ticker"], df["name"]))
    sel_key, pick_key = f"trend_detail_pick_{universe}", f"trend_pick_{universe}"
    pre = st.session_state.pop(pick_key, None)
    if pre in names:
        st.session_state[sel_key] = pre
    if st.session_state.get(sel_key) not in names:
        # default: the top-ranked asset that is not a cash-like carry fund
        pool = df[~df["cash_like"].fillna(False).astype(bool)] if "cash_like" in df else df
        st.session_state[sel_key] = (pool if len(pool) else df).iloc[0]["ticker"]
    ticker = st.selectbox("Asset", tickers, key=sel_key, format_func=lambda t: f"{t} — {names.get(t, '')}")
    row = df[df["ticker"] == ticker].iloc[0].to_dict()
    score, state, struct = row["score"], row["state"], row["structure"]
    col = THEME.teal if score >= 5 else (THEME.coral if score < -5 else THEME.mustard)
    st.markdown(uc.state_banner(col, f"{ticker} — {row['name']}",
                                f"Trend Score {score:+.1f} · {state} · {struct}"), unsafe_allow_html=True)
    mom = row.get("momentum")
    st.markdown(uc.numeric_slab([
        {"label": "Trend Score", "value": f"{score:+.1f}", "color": col},
        {"label": "Rank", "value": f"#{int(row['rank'])}", "color": THEME.text, "sub": f"of {len(df)}"},
        {"label": "Speeds bullish", "value": f"{int(row['n_bull'])}/{int(row['n_valid'])}",
         "color": THEME.text, "sub": "partial history" if row.get("partial") else ""},
        {"label": "Δ 5 days", "value": _fmt1(row.get("d5")), "color": THEME.text,
         "sub": f"20 days: {_fmt1(row.get('d20'))}"},
        {"label": "Accel (fast − slow)", "value": _fmt1(row.get("slope")), "color": THEME.text},
        {"label": "Momentum", "value": _fmt1(mom), "color": THEME.muted,
         "sub": "MOMENTUM composite" if universe == "stocks" else "ETF MOMENTUM composite"},
    ], min_width=120), unsafe_allow_html=True)
    if row.get("cash_like"):
        st.info(f"Cash-like instrument: 1-year annualized volatility is {row.get('ann_vol', 0):.2%}. A "
                "vol-normalized EWMAC divides a tiny steady drift by an even tinier volatility, so its "
                "trend reads as maximal. The score is correct under the method, but it describes "
                "carry, not a market trend.")
    st.markdown(viz.structure_chip(struct) + f'<span style="color:{THEME.muted};font-size:0.8rem;'
                f'margin-left:0.6rem;">{STRUCTURE_HELP.get(struct, "")} · {GROUP_LABEL[universe]}: '
                f'{row.get("group") or "—"}</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown(section("Trend ladder — seven speeds, fast → slow", 0,
                            help="Each bar is one speed's forecast today. The mustard tick shows where "
                                 "it stood 5 days ago and the grey dashed tick 20 days ago, so you can "
                                 "see which speeds are strengthening or fading."), unsafe_allow_html=True)
        st.markdown(viz.trend_ladder(row["forecasts"], row.get("f5"), row.get("f20"), row.get("last_cross")),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(section("Term structure — then and now", 4,
                            help="The seven forecasts as a curve across speeds, today vs 5, 20 and 60 "
                                 "trading days ago. A curve high on the left and low on the right means "
                                 "the trend is emerging. High on the right and low on the left means it "
                                 "is deteriorating. A flat high curve means it is persistent."),
                    unsafe_allow_html=True)
        _term_curve(row)

    st.markdown(section("Why this score", 1,
                        help="Each speed contributes exactly its forecast ÷ 7. These bars sum to the "
                             "Trend Score."), unsafe_allow_html=True)
    n_valid = max(1, int(row["n_valid"]))
    cdf = pd.DataFrame({"Speed": [f"{SPEED_SHORT[k]} · {SPEED_NAMES[k]}" for k in SPEED_KEYS],
                        "Contribution": [(v / n_valid) if v is not None else 0.0 for v in row["forecasts"]]})
    def build_c(alt):
        base = alt.Chart(cdf).encode(
            y=alt.Y("Speed:N", sort=None, title=None, axis=alt.Axis(labelLimit=0)),
            x=alt.X("Contribution:Q", title=f"contribution to the score (sum = {score:+.2f})",
                    scale=alt.Scale(domain=[-20 / 7, 20 / 7]), axis=alt.Axis(format="+.1f")))
        bars = base.mark_bar(cornerRadiusEnd=3, size=18).encode(
            color=alt.Color("Contribution:Q", scale=uc.diverging_scale(20 / 7, alt), legend=None),
            tooltip=["Speed", alt.Tooltip("Contribution:Q", format="+.2f")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=THEME.muted).encode(x="x:Q")
        return (bars + zero).properties(height=230)
    uc.render_chart(build_c, fallback=cdf)

    _detail_history(universe, ticker, row)


def _fmt1(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:+.1f}"


def _term_curve(row: dict) -> None:
    order = [SPEED_SHORT[k] for k in SPEED_KEYS]
    recs = []
    for lbl, key in (("60 days ago", "f60"), ("20 days ago", "f20"), ("5 days ago", "f5"), ("Today", "forecasts")):
        vals = row.get(key)
        if not isinstance(vals, (list, tuple)):
            continue
        for k, v in zip(order, vals):
            if v is not None:
                recs.append({"Speed": k, "Forecast": v, "When": lbl})
    cdf = pd.DataFrame(recs)
    if cdf.empty:
        return
    whens = ["Today", "5 days ago", "20 days ago", "60 days ago"]

    def build(alt):
        color = alt.Color("When:N", scale=alt.Scale(domain=whens,
                                                    range=[THEME.text, THEME.mustard, "#8a8a8a", "#4a4a4a"]),
                          legend=alt.Legend(orient="top", title=None))
        lines = alt.Chart(cdf).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
            x=alt.X("Speed:N", sort=order, title="fast → slow", axis=alt.Axis(labelAngle=0, labelLimit=0)),
            y=alt.Y("Forecast:Q", scale=alt.Scale(domain=[-20, 20]), title="forecast"),
            color=color, order=alt.Order("When:N"),
            tooltip=["When", "Speed", alt.Tooltip("Forecast:Q", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.muted).encode(y="y:Q")
        return (zero + lines).properties(height=330)
    uc.render_chart(build, fallback=cdf)


def _load_series(universe: str, ticker: str, years_back: int | None) -> pd.DataFrame:
    ys = history.years(universe)
    if years_back is not None and ys:
        ys = [y for y in ys if y >= ys[-1] - years_back]
    bust = _status_bust(universe)
    return history.series_for(universe, ticker, start_year=ys[0] if ys else None,
                              shard_loader=lambda y: _shard(universe, y, bust))


def _detail_history(universe: str, ticker: str, row: dict) -> None:
    st.markdown(section("How the trend developed", 3,
                        help="Everything below comes from the persisted daily history: the score and "
                             "all seven forecasts exactly as the system recorded them each day. Drag "
                             "across the bottom strip to zoom every panel."), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    span = c1.radio("Range", ["1Y", "3Y", "5Y", "All"], index=1, horizontal=True, key=f"trend_dspan_{universe}")
    marks = c2.pills("Crossover markers", [SPEED_SHORT[k] for k in SPEED_KEYS], selection_mode="multi",
                     default=[SPEED_SHORT[k] for k in ("16_64", "32_128", "64_256")],
                     key=f"trend_dmarks_{universe}",
                     help="Which speeds' crossovers to mark on the price. ▲ bullish, ▼ bearish. "
                          "Larger markers are slower speeds.")
    ema_pair = c3.selectbox("EMA pair on price", ["none"] + [SPEED_SHORT[k] for k in SPEED_KEYS], index=4,
                            key=f"trend_dema_{universe}")
    years_back = {"1Y": 1, "3Y": 3, "5Y": 5, "All": None}[span]
    h = _load_series(universe, ticker, years_back)
    if h.empty:
        st.caption("No persisted history for this asset yet. It accrues nightly, or after a one-time "
                   "`--action backfill`.")
        return
    days = {"1Y": 252, "3Y": 756, "5Y": 1260, "All": None}[span]
    if days:
        h = h.tail(days)

    px = None
    try:
        px = _prices(ticker)
    except Exception:
        px = None
    _history_chart(h, px, marks or [], ema_pair, ticker)
    _flip_table(universe, ticker, h, row)
    _mom_compare(universe, ticker, h)
    _event_log(universe, ticker)


def _regime_runs(h: pd.DataFrame) -> pd.DataFrame:
    fcols = [f"f{k}" for k in SPEED_KEYS]
    reg = [structure.regime_of([None if pd.isna(v) else v for v in vals])
           for vals in h[fcols].itertuples(index=False)]
    s = pd.Series(reg, index=h.index)
    runs = []
    start = s.index[0]
    for i in range(1, len(s) + 1):
        if i == len(s) or s.iloc[i] != s.iloc[i - 1]:
            runs.append({"start": start, "end": s.index[i - 1], "regime": s.iloc[i - 1]})
            if i < len(s):
                start = s.index[i]
    return pd.DataFrame(runs)


_REGIME_LABEL = {"all_bull": "All speeds bullish", "all_bear": "All speeds bearish",
                 "disagree": "Fast vs slow disagree", "mixed": "Mixed"}


def _history_chart(h: pd.DataFrame, px: pd.DataFrame | None, marks: list[str], ema_pair: str,
                   ticker: str) -> None:
    h = h.copy()
    h["date"] = h.index
    runs = _regime_runs(h)
    runs = runs[runs["regime"] != "mixed"].copy()
    runs["Regime"] = runs["regime"].map(_REGIME_LABEL)
    reg_domain = ["All speeds bullish", "All speeds bearish", "Fast vs slow disagree"]
    reg_range = [THEME.teal, THEME.coral, THEME.mustard]
    order = [SPEED_SHORT[k] for k in SPEED_KEYS]
    nxt = pd.Series(list(h.index[1:]) + [h.index[-1] + pd.tseries.offsets.BDay(1)], index=h.index)
    ribbon = pd.DataFrame([{"date": d, "date2": nxt[d], "Speed": SPEED_SHORT[k], "f": v}
                           for k in SPEED_KEYS
                           for d, v in h[f"f{k}"].dropna().items()])
    # crossovers from the persisted direction bits
    xs = []
    for k in SPEED_KEYS:
        if SPEED_SHORT[k] not in marks:
            continue
        b = h[f"bull_{k}"].dropna()
        flips = b[(b != b.shift()) & b.shift().notna()]
        for d, v in flips.items():
            xs.append({"date": d, "Speed": SPEED_SHORT[k], "dir": "Bullish crossover" if v > 0 else
                       "Bearish crossover", "size": 40 + 22 * SPEED_KEYS.index(k)})
    xdf = pd.DataFrame(xs)

    price = None
    if px is not None:
        c = px["close"]
        c = c[(c.index >= h.index[0]) & (c.index <= h.index[-1])]
        price = pd.DataFrame({"date": c.index, "price": c.values})
        if ema_pair != "none":
            k = SPEED_KEYS[[SPEED_SHORT[x] for x in SPEED_KEYS].index(ema_pair)]
            fs, ss = TREND_SPEEDS[SPEED_KEYS.index(k)]
            from .ewmac import ema
            full = px["close"]
            price["EMA fast"] = ema(full, fs).reindex(c.index).values
            price["EMA slow"] = ema(full, ss).reindex(c.index).values
        if not xdf.empty:
            xdf = xdf.merge(price[["date", "price"]], on="date", how="inner")

    def build(alt):
        brush = alt.selection_interval(encodings=["x"], name="zoom")
        xs_ = alt.X("date:T", title=None, scale=alt.Scale(domain={"param": "zoom"}),
                    axis=alt.Axis(format="%b %y"))
        panels = []
        if price is not None and not price.empty:
            bands = alt.Chart(runs).mark_rect(opacity=0.13).encode(
                x=alt.X("start:T", scale=alt.Scale(domain={"param": "zoom"}), title=None), x2="end:T",
                color=alt.Color("Regime:N", scale=alt.Scale(domain=reg_domain, range=reg_range),
                                legend=alt.Legend(orient="top", title="Background", labelLimit=0)),
                tooltip=["Regime", alt.Tooltip("start:T"), alt.Tooltip("end:T")])
            line = alt.Chart(price).mark_line(color=THEME.text, strokeWidth=1.5).encode(
                x=xs_, y=alt.Y("price:Q", title=f"{ticker} price", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=",.2f")])
            layers = [bands, line]
            if ema_pair != "none":
                emas = price.melt(id_vars=["date"], value_vars=["EMA fast", "EMA slow"],
                                  var_name="EMA", value_name="value").dropna()
                layers.append(alt.Chart(emas).mark_line(strokeWidth=1.3).encode(
                    x=xs_, y="value:Q",
                    color=alt.Color("EMA:N", scale=alt.Scale(domain=["EMA fast", "EMA slow"],
                                                             range=[THEME.mustard, THEME.navy]),
                                    legend=alt.Legend(orient="top", title=f"{ema_pair} EMAs")),
                    tooltip=["EMA", alt.Tooltip("value:Q", format=",.2f")]))
            if not xdf.empty:
                layers.append(alt.Chart(xdf).mark_point(filled=True, opacity=0.95, stroke=THEME.bg,
                                                        strokeWidth=1).encode(
                    x=xs_, y="price:Q",
                    shape=alt.Shape("dir:N", scale=alt.Scale(
                        domain=["Bullish crossover", "Bearish crossover"],
                        range=["triangle-up", "triangle-down"]), legend=None),
                    fill=alt.Fill("dir:N", scale=alt.Scale(domain=["Bullish crossover", "Bearish crossover"],
                                                           range=[THEME.teal, THEME.coral]),
                                  legend=alt.Legend(orient="top", title="Crossovers")),
                    size=alt.Size("size:Q", legend=None, scale=None),
                    tooltip=[alt.Tooltip("date:T"), "Speed", "dir", alt.Tooltip("price:Q", format=",.2f")]))
            panels.append(alt.layer(*layers).resolve_scale(color="independent")
                          .properties(height=300))
        rib = alt.Chart(ribbon).mark_rect().encode(
            x=xs_, x2="date2:T", y=alt.Y("Speed:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
            color=alt.Color("f:Q", scale=uc.diverging_scale(20, alt), title="forecast"),
            tooltip=[alt.Tooltip("date:T"), "Speed", alt.Tooltip("f:Q", format="+.0f")],
        ).properties(height=140)
        panels.append(rib)
        rule_df = pd.DataFrame({"y": [t for t, _ in MOM_STATES if t > -20]})
        rules = alt.Chart(rule_df).mark_rule(color=THEME.grid).encode(y="y:Q")
        sline = alt.Chart(h).mark_area(line={"color": THEME.text, "strokeWidth": 1.5}, opacity=0.35).encode(
            x=xs_, y=alt.Y("score:Q", title="Trend Score", scale=alt.Scale(domain=[-20, 20])),
            color=alt.value(THEME.navy),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("score:Q", format="+.1f")])
        panels.append((rules + sline).properties(height=200))
        nav = alt.Chart(h).mark_area(color=THEME.navy, opacity=0.5).encode(
            x=alt.X("date:T", title="drag to zoom · double-click to reset"),
            y=alt.Y("score:Q", title=None, axis=None, scale=alt.Scale(domain=[-20, 20]))
        ).add_params(brush).properties(height=46)
        panels.append(nav)
        return alt.vconcat(*panels, spacing=8).resolve_scale(color="independent")
    uc.render_chart(build, fallback=h[["score"]].reset_index())
    if price is None:
        st.caption("Live price fetch unavailable. Showing the persisted signal history only.")
    else:
        st.caption("Top: price, with background bands for all-bullish, all-bearish and fast-vs-slow "
                   "disagreement stretches, plus the chosen crossovers. Middle: the signal ribbon, one "
                   "row per speed. Bottom: the Trend Score.")


def _flip_table(universe: str, ticker: str, h: pd.DataFrame, row: dict) -> None:
    st.markdown(section("Crossover record by speed", 5,
                        help="From the persisted history in the selected range: how often each speed "
                             "flips between bullish and bearish, how long a run typically lasts, and "
                             "when it last turned."), unsafe_allow_html=True)
    rawlike = pd.DataFrame({k: h[f"bull_{k}"] for k in SPEED_KEYS}, index=h.index)
    fs = ev.flip_stats(rawlike)
    lc = row.get("last_cross") or {}
    recs = []
    for j, k in enumerate(SPEED_KEYS):
        s, c = fs.get(k) or {}, lc.get(k) or {}
        fv = row["forecasts"][j]
        recs.append({
            "Speed": SPEED_SHORT[k], "Name": SPEED_NAMES[k],
            "Now": ("Bullish" if fv > 0 else "Bearish") if fv is not None else "n/a",
            "Forecast": fv,
            "Turned on": c.get("date"), "Bars ago": c.get("bars_ago"),
            "Crossovers": s.get("n_crosses"),
            "Bull→bear": s.get("n_bull_to_bear"), "Bear→bull": s.get("n_bear_to_bull"),
            "Per year": s.get("per_year"), "Avg run (bars)": s.get("avg_run"),
        })
    fdf = pd.DataFrame(recs)
    for c in ("Forecast", "Bars ago", "Crossovers", "Bull→bear", "Bear→bull", "Per year", "Avg run (bars)"):
        fdf[c] = pd.to_numeric(fdf[c], errors="coerce")
    try:
        sty = fdf.style.map(lambda v: uc.grad_diverging(v, 20.0), subset=["Forecast"]).format(
            {"Forecast": "{:+.1f}", "Per year": "{:.1f}", "Avg run (bars)": "{:.0f}", "Bars ago": "{:.0f}",
             "Crossovers": "{:.0f}", "Bull→bear": "{:.0f}", "Bear→bull": "{:.0f}"}, na_rep="—")
        st.dataframe(sty, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(fdf, use_container_width=True, hide_index=True)


def _mom_compare(universe: str, ticker: str, h: pd.DataFrame) -> None:
    try:
        if universe == "stocks":
            from ..mom import history as mh
        else:
            from ..etfmom import history as mh
        ms = mh.series_for(ticker)
    except Exception:
        ms = []
    if not ms:
        return
    mdf = pd.DataFrame(ms)
    mdf["date"] = pd.to_datetime(mdf["date"])
    mdf = mdf[mdf["date"] >= h.index[0]]
    if mdf.empty:
        return
    # MOMENTUM's own history may be only weeks long; window the comparison to
    # the overlap (plus some context) so its line is not a sliver at the edge.
    start = max(h.index[0], mdf["date"].min() - pd.tseries.offsets.BDay(60))
    h = h[h.index >= start]
    both = pd.concat([pd.DataFrame({"date": h.index, "score": h["score"].values, "Series": "Trend Score"}),
                      pd.DataFrame({"date": mdf["date"], "score": mdf["composite"],
                                    "Series": "Momentum composite"})])
    st.markdown(section("Trend Following vs Momentum — same asset, same scale", 2,
                        help="Both scores run -20..+20 on one axis. Momentum's history is only as long "
                             "as that tab has been recording."), unsafe_allow_html=True)

    def build(alt):
        return (alt.Chart(both).mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=18)).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[-20, 20]), title="score"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Trend Score", "Momentum composite"],
                                                        range=[THEME.teal, THEME.mauve]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("date:T"), "Series", alt.Tooltip("score:Q", format="+.1f")],
        ).properties(height=220))
    uc.render_chart(build, fallback=both)
    st.caption(f"Momentum history available from {mdf['date'].min().date()} "
               f"({len(mdf)} recorded points); the window starts 60 trading days earlier for context.")


def _event_log(universe: str, ticker: str) -> None:
    evs = [e for e in _events_all(universe, _status_bust(universe)) if e["ticker"] == ticker]
    if not evs:
        return
    st.markdown(section("Signal-change log", 0,
                        help="Every trigger, upgrade, downgrade and multi-speed confirmation this asset "
                             "has recorded, newest first."), unsafe_allow_html=True)
    ldf = pd.DataFrame([{"Date": e["date"], "Event": viz.event_title(e),
                         "Speeds": ", ".join(SPEED_SHORT[k] for k in e["speeds"]) or "composite",
                         "Score before": e.get("score_before"), "Score after": e.get("score_after"),
                         "From": e.get("from"), "To": e.get("to")} for e in reversed(evs)])
    n_bb = sum(1 for e in evs if e["type"] == "trigger" and e["dir"] < 0)
    n_bl = sum(1 for e in evs if e["type"] == "trigger" and e["dir"] > 0)
    st.caption(f"{len(evs)} recorded events since {evs[0]['date']} · {n_bl} bullish and {n_bb} bearish "
               "triggers (Trend Score entering BULLISH / BEARISH).")
    st.dataframe(ldf.style.format({"Score before": "{:+.1f}", "Score after": "{:+.1f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True, height=min(420, 40 + 35 * len(ldf)))

"""VOL SCREENER page — liquidity, volatility screen, checklist, correlation baskets.

Reads committed JSON (data/volscreen/*) plus, for one selected ticker, its IV
history from the release asset. Five sections on one radio (key `vs_sub`,
pre-seedable in tests) and a SELL / BUY lens (key `vs_mode`).
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from .. import ui_kit as uk
from ..config import THEME as T
from ..config import VOLSCREEN_FILES, VOLSCREEN_STATE, VOLSCREEN_TERM_ASSET
from ..ui_theme import evidence_rating, help_badge, key_findings, section, stamp
from ..viz import volviz
from . import DISCLAIMER, glossary, load

SUBVIEWS = ["LIQUIDITY", "SCREEN", "CHECKLIST", "BASKETS", "LEARN"]
MODES = ["SELL PREMIUM", "BUY PREMIUM"]

_FINDINGS = [
    {"stat": "Index options have priced implied variance above subsequently realized variance in most "
             "periods: the variance risk premium is large, negative for buyers and time-varying.",
     "cite": "Carr & Wu (2009); Bollerslev, Tauchen & Zhou (2009)"},
    {"stat": "Stocks whose implied vol sits far above their historical vol have the lowest subsequent "
             "straddle returns — the cross-section of the HV-IV gap predicts option returns.",
     "cite": "Goyal & Saretto (2009)"},
    {"stat": "The slope of the VIX term structure carries the variance premium: steep contango = calm "
             "regime where short volatility has earned; inversion marks stress.",
     "cite": "Johnson (2017); Simon & Campasano (2014)"},
    {"stat": "Cross-sectionally the sign flips: names with UPWARD-sloping IV term structures earn higher "
             "straddle returns (front-rich names are the better sales).",
     "cite": "Vasquez (2017)"},
]
_EVIDENCE = ("The index variance premium is among the best-documented effects in options (A-grade for "
             "SPX). Single-name premia are smaller and noisier; liquidity stars and the VRP state machine "
             "are our own rules on top, not tested edges. Scores rank candidates; they are not forecasts.")
STAR = "★"


# ------------------------------------------------------------------ data --

@st.cache_data(ttl=900, show_spinner=False)
def _data() -> dict:
    return {"screen": load("screen", {}) or {}, "baskets": load("baskets", {}) or {},
            "status": load("status", {}) or {}}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def ticker_history(ticker: str) -> pd.DataFrame:
    """One ticker's stored IV history (release asset; empty when unavailable)."""
    from ..ephemeris.store_px import ensure_local
    from .history import read_ticker
    p = VOLSCREEN_FILES["term_local"]
    if not p.exists():
        p = ensure_local(VOLSCREEN_TERM_ASSET)
    return read_ticker(p, ticker)


def today_badge() -> str | None:
    try:
        sc = load("screen", {}) or {}
        rows = sc.get("rows") or []
        if not rows:
            return None
        m = sc.get("market") or {}
        clear = sum(1 for r in rows if r.get("state") == "ALL-CLEAR")
        a = sum(1 for r in rows if (r.get("check") or {}).get("sell", {}).get("grade") == "A")
        vix = m.get("vix")
        sub = f"{a} A-grade sells · VIX {vix:.1f}" if vix else f"{a} A-grade sells"
        return uc.chip(f"VOL SCREENER — {clear}/{len(rows)} ALL-CLEAR", color=T.teal, sub=sub)
    except Exception:
        return None


# --------------------------------------------------------------- helpers --

def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _p(x, d: int = 1) -> str:
    return f"{x * 100:.{d}f}" if _ok(x) else "—"


def _stars(n) -> str:
    n = int(n or 1)
    return STAR * n + "·" * (4 - n)


def _grade_color(g: str) -> str:
    return {"A": T.teal, "B": T.mint, "C": T.amber, "D": T.orange, "F": T.coral}.get(g, T.muted)


def _mode_key() -> str:
    return "buy" if st.session_state.get("vs_mode") == MODES[1] else "sell"


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Flat table of the screen (decimal vols; formatting happens per view)."""
    mk = _mode_key()
    out = []
    for r in rows:
        liq, rv, v = r.get("liq", {}), r.get("rv", {}), r.get("vrp", {})
        ck = (r.get("check") or {}).get(mk, {})
        out.append({
            "TICKER": r["ticker"], "NAME": r.get("name", ""), "KIND": r.get("kind", ""),
            "SECTOR": r.get("sector", ""), "CORE": bool(r.get("core")),
            "LIQ ★": _stars(liq.get("stars")), "_stars": liq.get("stars") or 1, "CLASS": liq.get("class", ""),
            "OPT VOL": liq.get("opt_vol"), "VOL 20D": liq.get("opt_vol20"), "OPEN INT": liq.get("oi"),
            "ATM SPRD": liq.get("atm_spread"), "25Δ SPRD": liq.get("d25_spread"),
            "EXP": liq.get("n_exp"), "STRIKES": liq.get("strikes"), "ADV $": liq.get("adv_usd"),
            "SURGE x": liq.get("surge_x"),
            "SPOT": r.get("spot"), "IV30": r.get("iv", {}).get("30"), "IVx30": r.get("ivx", {}).get("30"),
            "FRONT": r.get("front"), "BACK": r.get("back"), "30/90": r.get("ratio_30_90"),
            "IVR": r.get("ivr"), "IVP": r.get("ivp"), "σ-Z": (r.get("iv_z") or {}).get("30"),
            "RV YZ20": rv.get("yz20"), "GARCH30": (r.get("garch") or {}).get("30"), "HAR30": rv.get("har30"),
            "VRP30": v.get("vrp30"), "IV/RV": v.get("iv_rv"), "HV-IV": v.get("gs"), "VRP Z": v.get("z"),
            "SLOPE5": v.get("slope5"), "STATE": r.get("state", ""), "SKEW25": (r.get("skew") or {}).get("skew25"),
            "EARN": (r.get("earn") or {}).get("days"), "EARN MOVE": (r.get("earn") or {}).get("move"),
            "EM30": r.get("em30"), "SCORE": ck.get("score"), "GRADE": ck.get("grade", ""),
            "HIST": r.get("ivr_src", "") + (" · stale quote" if r.get("stale") else ""),
        })
    df = pd.DataFrame(out)
    text = {"TICKER", "NAME", "KIND", "SECTOR", "CORE", "LIQ ★", "CLASS", "STATE", "GRADE", "HIST"}
    for c in df.columns:
        if c not in text:
            df[c] = pd.to_numeric(df[c], errors="coerce")          # None -> NaN (blank, sortable)
    return df


def _colcfg(cols) -> dict:
    hm = glossary.help_map()
    pct = {"IV30", "IVx30", "FRONT", "BACK", "RV YZ20", "GARCH30", "HAR30", "EARN MOVE"}
    out = {}
    for c in cols:
        h = hm.get(c)
        if c in pct:
            out[c] = st.column_config.NumberColumn(c, help=h, format="%.3f")
        elif c in ("ATM SPRD", "25Δ SPRD"):
            out[c] = st.column_config.NumberColumn(c, help=h, format="%.3f")
        elif c in ("IVR", "IVP"):
            out[c] = st.column_config.ProgressColumn(c, help=h, min_value=0.0, max_value=1.0, format="%.2f")
        elif c == "SCORE":
            out[c] = st.column_config.ProgressColumn(c, help=h, min_value=0, max_value=100, format="%.0f")
        elif c in ("VRP30", "SKEW25"):
            out[c] = st.column_config.NumberColumn(c, help=h, format="%+.3f")
        elif c in ("OPT VOL", "VOL 20D", "OPEN INT", "ADV $"):
            out[c] = st.column_config.NumberColumn(c, help=h, format="compact")
        elif h:
            out[c] = st.column_config.Column(c, help=h)
    return out


# ------------------------------------------------------------ signature --

def _signature(sc: dict) -> None:
    rows = sc.get("rows", [])
    core = [r for r in rows if r.get("core")] or rows[:24]
    st.markdown(section("Term structure ribbon — the core liquid basket", 0,
                        help="Each tile is one underlying's ATM implied-vol term structure, 7 to 365 days "
                             "(log axis). Teal = contango, coral = backwardation, dashed amber = the curve with "
                             "the earnings event removed. The dot is the VRP state."), unsafe_allow_html=True)
    st.markdown(volviz.term_ribbon(core[:36], sc.get("tenors", [7, 14, 30, 60, 90, 180, 365])),
                unsafe_allow_html=True)


def _kpis(sc: dict) -> None:
    m = sc.get("market") or {}
    rows = sc.get("rows", [])
    vr = m.get("vix_ratio")
    items = [
        {"label": "VIX", "value": f"{m['vix']:.2f}" if _ok(m.get("vix")) else "—",
         "sub": f"1y pct {_p(m.get('vix_pct'), 0)}" if _ok(m.get("vix_pct")) else ""},
        {"label": "VIX / VIX3M", "value": f"{vr:.2f}" if _ok(vr) else "—",
         "sub": "contango" if _ok(vr) and vr < 1 else "backwardation" if _ok(vr) else "",
         "color": T.teal if _ok(vr) and vr < 1 else T.coral},
        {"label": "MEDIAN VRP30", "value": f"{_p(m.get('median_vrp30'), 2)} pts",
         "color": T.teal if _ok(m.get("median_vrp30")) and m["median_vrp30"] > 0 else T.coral},
        {"label": "ALL-CLEAR", "value": f"{_p(m.get('pct_all_clear'), 0)}%",
         "sub": f"{(m.get('states') or {}).get('ALARM', 0)} alarm · {(m.get('states') or {}).get('CONFIRMED', 0)} confirmed"},
        {"label": "SURGE NAMES", "value": str(m.get("n_surge", 0)), "color": T.amber, "sub": "unusual option volume"},
        {"label": "SCREENED", "value": str(len(rows)), "sub": f"{len(sc.get('core', []))} in core basket"},
    ]
    st.markdown(uk.kpis(items), unsafe_allow_html=True)


# -------------------------------------------------------------- sections --

def _liquidity(sc: dict, df: pd.DataFrame) -> None:
    st.markdown(section("Liquidity master list", 1, help=glossary.help("stars")), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    classes = c1.multiselect("Class", ["CORE", "SURGE", "ACTIVE", "FADING", "THIN"],
                             default=["CORE", "SURGE", "ACTIVE", "FADING"], key="vs_liq_cls")
    kinds = c2.multiselect("Kind", ["index", "etf", "stock"], default=["index", "etf", "stock"], key="vs_liq_kind")
    min_st = c3.select_slider("Min ★", options=[1, 2, 3, 4], value=2, key="vs_liq_min")
    d = df[df["CLASS"].isin(classes) & df["KIND"].isin(kinds) & (df["_stars"] >= min_st)]
    cols = ["TICKER", "NAME", "KIND", "LIQ ★", "CLASS", "OPT VOL", "VOL 20D", "OPEN INT", "ATM SPRD",
            "25Δ SPRD", "EXP", "STRIKES", "ADV $", "SURGE x"]
    st.dataframe(d.sort_values(["_stars", "OPT VOL"], ascending=False)[cols], hide_index=True,
                 width="stretch", height=420, column_config=_colcfg(cols))
    core = sc.get("core", [])
    st.markdown(section("Core liquid basket", 2,
                        help="4★ CORE names ranked by the continuous liquidity score: the tightest, deepest, "
                             "most consistently traded option markets in the universe."), unsafe_allow_html=True)
    if core:
        by = {r["ticker"]: r for r in sc.get("rows", [])}
        st.markdown("".join(uc.chip(t, T.teal, f"{_p((by[t]['liq'] or {}).get('atm_spread'), 1)}% sprd")
                            for t in core if t in by), unsafe_allow_html=True)
    else:
        st.caption("No 4★ CORE names yet — CORE needs a few sessions of volume history to confirm stability.")
    surge = df[df["CLASS"] == "SURGE"].sort_values("SURGE x", ascending=False)
    st.markdown(section("Surge — unusual option volume", 3,
                        help="Today's option volume against the name's own 20-day median (before 5 sessions "
                             "of history: turnover vs open interest relative to the universe). Usually an "
                             "event: check the earnings column."), unsafe_allow_html=True)
    if len(surge):
        s = surge[["TICKER", "NAME", "SURGE x", "OPT VOL", "OPEN INT", "EARN", "IV30", "IVR"]].head(20)
        st.dataframe(s, hide_index=True, width="stretch", column_config=_colcfg(s.columns))
    else:
        st.caption("No surge names today.")
    b = df[df["OPT VOL"].notna() & (df["OPT VOL"] > 0) & df["ATM SPRD"].notna()].copy()
    if len(b):
        st.markdown(section("Liquidity map — volume against cost", 4), unsafe_allow_html=True)
        b["SPRD %"] = (b["ATM SPRD"].clip(upper=0.3) * 100).round(2)
        dom = list(volviz.CLASS_COLORS)

        def build(alt):
            pts = alt.Chart(b).mark_circle(opacity=0.85, stroke=T.bg, strokeWidth=0.6).encode(
                x=alt.X("OPT VOL:Q", scale=alt.Scale(type="log"), title="OPTION VOLUME (LOG)"),
                y=alt.Y("SPRD %:Q", scale=alt.Scale(type="sqrt", reverse=True), title="ATM SPREAD % (TIGHT ↑)"),
                size=alt.Size("OPEN INT:Q", scale=alt.Scale(type="sqrt", range=[12, 520]), legend=None),
                color=alt.Color("CLASS:N", scale=alt.Scale(domain=dom, range=[volviz.CLASS_COLORS[c] for c in dom]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["TICKER", "CLASS", "LIQ ★", alt.Tooltip("OPT VOL:Q", format=","),
                         alt.Tooltip("OPEN INT:Q", format=","), "SPRD %"])
            lab = alt.Chart(b[b["_stars"] >= 4]).mark_text(dx=8, align="left", fontSize=8, color=T.muted).encode(
                x="OPT VOL:Q", y="SPRD %:Q", text="TICKER:N")
            return (pts + lab).properties(height=380)
        uc.render_chart(build, b[["TICKER", "CLASS", "OPT VOL", "SPRD %"]], width="stretch")
        st.caption("Up and to the right = deep and cheap to trade. Dot size = open interest; 4★ names labelled.")


PRESETS = {
    "IVR ≥ 50": lambda d: d["IVR"] >= 0.5,
    "VRP z ≥ 1": lambda d: d["VRP Z"] >= 1,
    "VRP > 0": lambda d: d["VRP30"] > 0,
    "Contango": lambda d: d["30/90"] <= 1.0,
    "No earnings ≤ 45d": lambda d: d["EARN"].isna() | (d["EARN"] > 45) | (d["EARN"] < 0),
    "≥ 3★": lambda d: d["_stars"] >= 3,
    "ALL-CLEAR": lambda d: d["STATE"] == "ALL-CLEAR",
    "BUY: IVR ≤ 20": lambda d: d["IVR"] <= 0.2,
    "BUY: VRP ≤ 0": lambda d: d["VRP30"] <= 0,
}


def _screen(sc: dict, df: pd.DataFrame) -> None:
    st.markdown(section("Volatility screen", 1, help="Every vol column has hover help. Vols are decimals "
                        "(0.25 = 25%)."), unsafe_allow_html=True)
    default = ["≥ 3★", "VRP > 0"] if _mode_key() == "sell" else ["≥ 3★", "BUY: VRP ≤ 0"]
    pick = st.multiselect("Presets (all must hold)", list(PRESETS), default=default, key="vs_presets")
    d = df.copy()
    for p in pick:
        try:
            d = d[PRESETS[p](d).fillna(False)]
        except Exception:
            pass
    cols = ["TICKER", "GRADE", "SCORE", "LIQ ★", "IV30", "IVx30", "FRONT", "BACK", "30/90", "IVR", "IVP", "σ-Z",
            "RV YZ20", "GARCH30", "HAR30", "VRP30", "IV/RV", "HV-IV", "VRP Z", "SLOPE5", "STATE", "SKEW25",
            "EARN", "EARN MOVE", "EM30", "HIST"]
    st.caption(f"{len(d)} of {len(df)} underlyings pass.")
    st.dataframe(d.sort_values("SCORE", ascending=False)[cols], hide_index=True, width="stretch",
                 height=440, column_config=_colcfg(cols))
    p = d[d["IVR"].notna() & d["VRP30"].notna()].copy()
    if len(p):
        def build(alt):
            dom = list(volviz.STATE_COLORS)
            return alt.Chart(p).mark_circle(opacity=0.85, stroke=T.bg, strokeWidth=0.5).encode(
                x=alt.X("IVR:Q", title="IV RANK", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("VRP30:Q", title="VRP30 (IVx − GARCH)"),
                size=alt.Size("_stars:Q", legend=None, scale=alt.Scale(range=[30, 260])),
                color=alt.Color("STATE:N", scale=alt.Scale(domain=dom, range=[volviz.STATE_COLORS[s] for s in dom]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["TICKER", "STATE", "IVR", "VRP30", "IV30", "LIQ ★", "GRADE"],
            ).properties(height=340) + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                strokeDash=[3, 3], color=T.faint).encode(y="y:Q")
        st.markdown(section("Where the premium is — IV rank vs VRP", 2), unsafe_allow_html=True)
        uc.render_chart(build, p[["TICKER", "IVR", "VRP30"]], width="stretch")
        st.caption("Upper-right = rich on both counts (high rank, implied above forecast realized). "
                   "Dot size = liquidity stars.")
    _detail(sc, d if len(d) else df, key="vs_ticker_screen")


def _term_chart(r: dict, hist: pd.DataFrame, tenors: list[int]) -> None:
    rows = []
    for t in tenors:
        rows.append({"tenor": t, "series": "IV today", "v": r.get("iv", {}).get(str(t))})
        rows.append({"tenor": t, "series": "IV ex-earnings", "v": r.get("ivx", {}).get(str(t))})
        rows.append({"tenor": t, "series": "GARCH forecast RV", "v": (r.get("garch") or {}).get(str(t))})
    if len(hist):
        h = hist[hist["src"] == 0].sort_values("date")
        for lag, lab in ((5, "IV −5 sessions"), (21, "IV −21 sessions")):
            if len(h) > lag:
                row = h.iloc[-1 - lag]
                for t in tenors:
                    rows.append({"tenor": t, "series": lab, "v": row.get(f"iv_{t}")})
    d = pd.DataFrame(rows).dropna()
    if d.empty:
        return
    gap = pd.DataFrame({"tenor": tenors, "lo": [(r.get("garch") or {}).get(str(t)) for t in tenors],
                        "hi": [r.get("ivx", {}).get(str(t)) for t in tenors]}).dropna()
    dom = ["IV today", "IV ex-earnings", "GARCH forecast RV", "IV −5 sessions", "IV −21 sessions"]
    rng = [T.teal, T.amber, T.violet, T.muted, T.faint]

    def build(alt):
        band = alt.Chart(gap).mark_area(opacity=0.14, color=T.teal).encode(
            x=alt.X("tenor:Q", scale=alt.Scale(type="log"), title="DAYS"), y="lo:Q", y2="hi:Q")
        lines = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=26)).encode(
            x=alt.X("tenor:Q", scale=alt.Scale(type="log"), title="DAYS"),
            y=alt.Y("v:Q", title="VOL", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(orient="bottom")),
            strokeDash=alt.condition(alt.FieldOneOfPredicate("series", ["IV ex-earnings", "IV −5 sessions",
                                                                        "IV −21 sessions"]),
                                     alt.value([4, 3]), alt.value([1, 0])),
            tooltip=["series", "tenor", alt.Tooltip("v:Q", format=".2%")])
        return (band + lines).properties(height=280)
    uc.render_chart(build, d, width="stretch")
    st.caption("Shaded = VRP by tenor (ex-earnings IV over the GARCH forecast of realized vol for the same horizon).")


def _rv_bars(r: dict) -> None:
    rv = r.get("rv", {})
    names = [("CC20", "cc20"), ("PARKINSON", "pk20"), ("GARMAN-KLASS", "gk20"), ("ROGERS-SATCHELL", "rs20"),
             ("YANG-ZHANG 10", "yz10"), ("YANG-ZHANG 20", "yz20"), ("YANG-ZHANG 60", "yz60"), ("EWMA .94", "ewma"),
             ("HAR 30D FCST", "har30")]
    d = pd.DataFrame([{"est": n, "v": rv.get(k)} for n, k in names]).dropna()
    g = (r.get("garch") or {}).get("30")
    if _ok(g):
        d = pd.concat([d, pd.DataFrame([{"est": "GARCH 30D FCST", "v": g}])])
    iv30 = r.get("ivx", {}).get("30")
    if d.empty:
        return

    def build(alt):
        bars = alt.Chart(d).mark_bar(height=11).encode(
            y=alt.Y("est:N", sort=None, title=None), x=alt.X("v:Q", title="ANNUALISED VOL", axis=alt.Axis(format="%")),
            color=alt.condition(alt.datum.v > (iv30 or 0), alt.value(T.coral), alt.value(T.teal)),
            tooltip=["est", alt.Tooltip("v:Q", format=".2%")])
        rule = alt.Chart(pd.DataFrame({"x": [iv30]})).mark_rule(color=T.amber, strokeDash=[4, 2]).encode(x="x:Q")
        return (bars + rule).properties(height=26 * len(d))
    uc.render_chart(build, d, width="stretch")
    st.caption("Realized-vol estimators against ex-earnings IV30 (dashed amber). Coral = realized above implied.")


def _vrp_chart(r: dict) -> None:
    vh = r.get("vrp_hist") or {}
    if not vh.get("d"):
        return
    from .checklist import STATES
    # compact form: d0 = first full date, d = MMDD stamps (year rolls forward on wrap), s = state codes
    year = int((vh.get("d0") or "2000")[:4])
    dates, prev = [], None
    for md in vh["d"]:
        if prev is not None and md < prev:
            year += 1
        dates.append(pd.Timestamp(f"{year}-{md[:2]}-{md[2:]}"))
        prev = md
    states = [STATES[i] if isinstance(i, int) else i for i in (vh.get("s") or [])] or ["NEUTRAL"] * len(dates)
    d = pd.DataFrame({"date": dates, "vrp": vh["v"], "state": states, "src": vh.get("src") or [0] * len(dates)})
    d["end"] = d["date"].shift(-1).fillna(d["date"].iloc[-1] + pd.Timedelta(days=1))
    dom = list(volviz.STATE_COLORS)

    def build(alt):
        bands = alt.Chart(d).mark_rect(opacity=0.18).encode(
            x="date:T", x2="end:T", color=alt.Color("state:N", scale=alt.Scale(domain=dom,
                                                    range=[volviz.STATE_COLORS[s] for s in dom]),
                                                    legend=alt.Legend(orient="bottom", title=None)))
        line = alt.Chart(d).mark_line(color=T.text, strokeWidth=1.4).encode(
            x=alt.X("date:T", title=None), y=alt.Y("vrp:Q", title="VRP30", axis=alt.Axis(format="%")),
            tooltip=["date:T", alt.Tooltip("vrp:Q", format="+.2%"), "state"])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=T.faint, strokeDash=[3, 3]).encode(y="y:Q")
        return (bands + zero + line).properties(height=200)
    uc.render_chart(build, d[["date", "vrp", "state"]], width="stretch")
    if any(s != 0 for s in d["src"]):
        st.caption("Earlier points come from the seed (Cboe vol index or modeled IV) until observed history accrues.")


def _detail(sc: dict, df: pd.DataFrame, key: str) -> None:
    rows = {r["ticker"]: r for r in sc.get("rows", [])}
    opts = [t for t in df["TICKER"].tolist() if t in rows] or list(rows)
    if not opts:
        return
    t = st.selectbox("Underlying detail", opts, key=key)
    r = rows[t]
    st.markdown(section(f"{t} — {html.escape(r.get('name', ''))}", 3), unsafe_allow_html=True)
    v, e = r.get("vrp", {}), r.get("earn", {})
    st.markdown(uk.kpis([
        {"label": "SPOT", "value": f"{r.get('spot', 0):,.2f}", "sub": f"{_p(r.get('chg'), 2)}% day" if _ok(r.get('chg')) else ""},
        {"label": "IV30 / IVx30", "value": f"{_p(r['iv'].get('30'))} / {_p(r['ivx'].get('30'))}",
         "sub": f"Cboe iv30 {_p(r.get('iv30_cboe'))}"},
        {"label": "IVR · IVP", "value": f"{_p(r.get('ivr'), 0)} · {_p(r.get('ivp'), 0)}", "sub": r.get("ivr_src", "")},
        {"label": "VRP30", "value": f"{_p(v.get('vrp30'), 2)} pts", "sub": f"z {v.get('z'):+.2f}" if _ok(v.get('z')) else "",
         "color": T.teal if _ok(v.get("vrp30")) and v["vrp30"] > 0 else T.coral},
        {"label": "STATE", "value": r.get("state", "—"), "sub": f"{r.get('state_days', 0)} sessions",
         "color": volviz.STATE_COLORS.get(r.get("state"), T.muted)},
        {"label": "EARNINGS", "value": e.get("next") or "none",
         "sub": (f"implied ±{_p(e.get('move'))}% · hist {_p(e.get('hist_move'))}%" if e.get("move") else "")},
    ]), unsafe_allow_html=True)
    if r.get("note"):
        st.markdown(uc.note_strip("Read with care", [f"{t} is a {r['note']} product: its implied vol and VRP are "
                                                     "not comparable to an unlevered underlying's."]),
                    unsafe_allow_html=True)
    hist = ticker_history(t)
    c1, c2 = st.columns([3, 2])
    with c1:
        _term_chart(r, hist, sc.get("tenors", [7, 14, 30, 60, 90, 180, 365]))
    with c2:
        _rv_bars(r)
    _vrp_chart(r)


def _checklist(sc: dict, df: pd.DataFrame) -> None:
    mk = _mode_key()
    st.markdown(section("Checklist — ranked master list", 1,
                        help="Each underlying is scored 0-100 on the checklist below (weights in "
                             "config.VOLSCREEN_CHECK_WEIGHTS). Higher = more of the evidence lines up for "
                             + ("selling" if mk == "sell" else "buying") + " premium."), unsafe_allow_html=True)
    min_st = st.select_slider("Min ★", options=[1, 2, 3, 4], value=3, key="vs_chk_min")
    d = df[df["_stars"] >= min_st].sort_values("SCORE", ascending=False)
    cols = ["TICKER", "GRADE", "SCORE", "LIQ ★", "STATE", "VRP30", "IV/RV", "IVR", "30/90", "SLOPE5", "EARN"]
    st.dataframe(d[cols], hide_index=True, width="stretch", height=360, column_config=_colcfg(cols))
    rows = {r["ticker"]: r for r in sc.get("rows", [])}
    opts = [t for t in d["TICKER"] if t in rows] or list(rows)
    if not opts:
        return
    t = st.selectbox("Checklist card", opts, key="vs_ticker_check")
    r = rows[t]
    ck = (r.get("check") or {}).get(mk, {})
    c1, c2, c3 = st.columns([1, 2.2, 1.4])
    with c1:
        g = ck.get("grade", "—")
        st.markdown(uk.radial_gauge(ck.get("score", 0) or 0, 0, 100, f"GRADE {g}", color=_grade_color(g),
                                    size=150, fmt="{:.0f}"), unsafe_allow_html=True)
        st.caption(f"{t} · {'SELL' if mk == 'sell' else 'BUY'} lens")
    with c2:
        sym = {"pass": ("✓", T.teal), "warn": ("~", T.amber), "fail": ("✗", T.coral)}
        from .checklist import text as rule_text
        txt = rule_text(mk)
        lines = []
        for it in ck.get("items", []):
            s, c = sym[it["status"]]
            label, why = txt.get(it["key"], (it["key"], ""))
            it = {**it, "label": it.get("label", label), "why": it.get("why", why)}
            lines.append(f'<div style="display:flex;gap:0.7rem;padding:0.32rem 0;border-bottom:1px solid {T.line}">'
                         f'<span style="color:{c};width:1rem;font-weight:600">{s}</span>'
                         f'<span style="flex:1"><span style="color:{T.text}">{html.escape(it["label"])}</span>'
                         f'<br><span style="color:{T.muted};font-size:0.72rem">{html.escape(it["why"])}</span></span>'
                         f'<span style="color:{c};font-size:0.78rem;white-space:nowrap">{html.escape(str(it["value"]))}</span>'
                         f'</div>')
        st.markdown("".join(lines), unsafe_allow_html=True)
    with c3:
        st.markdown(volviz.state_diagram(r.get("state", "NEUTRAL"), r.get("state_days", 0), 280),
                    unsafe_allow_html=True)
        s = VOLSCREEN_STATE
        st.caption(f"Slope = OLS over {s['slope_days']} sessions, standardised by {s['z_window']}-session "
                   f"VRP change volatility. Practitioner heuristic — thresholds in config.")
    _vrp_chart(r)


def _baskets(sc: dict) -> None:
    b = _data()["baskets"]
    st.markdown(section("Uncorrelated baskets", 1,
                        help="Average-linkage clustering of daily-return correlations (Ledoit-Wolf shrunk). "
                             "One best-scoring ≥3★ name per cluster, never two above ρ 0.6."), unsafe_allow_html=True)
    w = st.radio("Correlation window", ["60", "120", "250"], index=1, horizontal=True, key="vs_win")
    bk = b.get(w) or {}
    if not bk.get("clusters"):
        st.info("No basket for this window yet.")
        return
    rows = {r["ticker"]: r for r in sc.get("rows", [])}
    basket = bk.get("basket", [])
    st.markdown(uk.kpis([
        {"label": "CLUSTERS", "value": str(bk.get("k")), "sub": f"{len(bk.get('tickers', []))} names"},
        {"label": "BASKET", "value": str(len(basket)), "sub": "one per cluster, ρ ≤ 0.6"},
        {"label": "EFFECTIVE BETS", "value": f"{bk.get('eff_basket', 0):.1f}" if _ok(bk.get("eff_basket")) else "—",
         "sub": f"universe {bk.get('eff_universe', 0):.1f}" if _ok(bk.get("eff_universe")) else "",
         "color": T.teal},
        {"label": "SHRINKAGE", "value": f"{bk.get('shrinkage', 0):.2f}", "sub": "Ledoit-Wolf intensity"},
    ]), unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1])
    with c1:
        colors = {t: volviz.STATE_COLORS.get(rows.get(t, {}).get("state"), T.muted) for t in bk["tickers"]}
        st.markdown(volviz.circular_dendrogram(bk["merges"], bk["tickers"], bk["order"], bk["labels"], colors,
                                               set(basket)), unsafe_allow_html=True)
        st.caption("Rim colour = cluster; label colour = VRP state; ◆ = in the basket.")
    with c2:
        mk = _mode_key()
        out = []
        for t in basket:
            r = rows.get(t, {})
            iv30 = (r.get("ivx") or {}).get("30")
            out.append({"TICKER": t, "CLUSTER": next((c["id"] for c in bk["clusters"] if t in c["members"]), None),
                        "GRADE": (r.get("check") or {}).get(mk, {}).get("grade"),
                        "SCORE": (r.get("check") or {}).get(mk, {}).get("score"),
                        "IVx30": iv30, "STATE": r.get("state"), "_inv": 1 / iv30 if _ok(iv30) and iv30 > 0 else np.nan})
        bd = pd.DataFrame(out)
        if len(bd):
            bd["RISK WT"] = bd["_inv"] / bd["_inv"].sum()
            cols = ["TICKER", "CLUSTER", "GRADE", "SCORE", "IVx30", "STATE", "RISK WT"]
            st.dataframe(bd[cols], hide_index=True, width="stretch", column_config={
                **_colcfg(cols), "RISK WT": st.column_config.ProgressColumn(
                    "RISK WT", min_value=0.0, max_value=float(bd["RISK WT"].max() or 1), format="%.2f",
                    help="Inverse-IV weight: equal volatility budget per name (context only).")})
        vix = (sc.get("market") or {}).get("vix")
        if _ok(vix):
            band = 25 if vix < 15 else 30 if vix < 20 else 35 if vix < 30 else 40 if vix < 40 else 50
            st.markdown(uc.note_strip("Sizing context (not a position manager)", [
                f"VIX {vix:.1f}: a common practitioner guideline deploys about {band}% of buying power in "
                "premium-selling positions at this level, scaling up as VIX rises (tastylive convention).",
                "Inverse-IV risk weights give each name the same volatility budget; keep each position small.",
                "One name per cluster keeps the book from being one bet in several tickers."]),
                unsafe_allow_html=True)
    corr = np.array(bk.get("corr") or [])
    if corr.size:
        order = bk["order"]
        names = [bk["tickers"][i] for i in order]
        keep = [i for i, t in zip(order, names) if (rows.get(t, {}).get("liq") or {}).get("stars", 0) >= 3][:70]
        if len(keep) >= 4:
            lab = [bk["tickers"][i] for i in keep]
            sub = corr[np.ix_(keep, keep)]
            hm = pd.DataFrame([{"a": lab[i], "b": lab[j], "rho": float(sub[i, j])}
                               for i in range(len(keep)) for j in range(len(keep))])
            st.markdown(section("Seriated correlation heatmap (≥3★)", 2), unsafe_allow_html=True)

            def build(alt):
                return alt.Chart(hm).mark_rect().encode(
                    x=alt.X("a:N", sort=lab, title=None, axis=alt.Axis(labelFontSize=7, labelAngle=-90)),
                    y=alt.Y("b:N", sort=lab, title=None, axis=alt.Axis(labelFontSize=7)),
                    color=alt.Color("rho:Q", scale=uc.diverging_scale(1.0, alt), legend=alt.Legend(title="ρ")),
                    tooltip=["a", "b", alt.Tooltip("rho:Q", format=".2f")]).properties(height=560)
            uc.render_chart(build, hm, width="stretch")


def _learn() -> None:
    st.markdown(section("How every number is made", 1), unsafe_allow_html=True)
    with st.expander("Implied vol, term structure and the earnings kink", expanded=True):
        st.markdown("IVs are recomputed from mid quotes against the put-call-parity forward "
                    "F = K + e^{rT}(C − P), which absorbs dividends and borrow. Constant maturities are "
                    "interpolated linearly in total variance (w = σ²T), the convention behind the VIX:")
        st.latex(glossary.METRICS["iv30"][2])
        st.markdown("An expiry spanning an earnings report carries the event variance ε² on top of the "
                    "diffusive variance. With two post-event expiries the forward variance between them "
                    "is event-free; ε² is what is left in the first one:")
        st.latex(glossary.METRICS["ivx30"][2])
        st.latex(glossary.METRICS["earn_move"][2])
    with st.expander("Realized volatility — six estimators and two forecasts"):
        for k in ("cc20", "pk20", "gk20", "rs20", "yz20", "ewma", "har30", "garch30"):
            lab, h, f, c = glossary.METRICS[k]
            st.markdown(f"**{lab}** — {h} *{c}*")
            if f:
                st.latex(f)
        st.markdown("GARCH and GJR-GARCH are fitted nightly by Gaussian quasi-maximum likelihood (pure "
                    "numpy Nelder-Mead on a stationarity-preserving reparametrisation). The h-day average "
                    "variance forecast mean-reverts geometrically to the long-run level:")
        st.latex(r"E\left[\bar\sigma^2_{t,h}\right] = \bar\sigma^2 + (\sigma^2_{t+1} - \bar\sigma^2)"
                 r"\frac{1 - \rho^h}{(1-\rho)h},\quad \rho = \alpha + \beta + \gamma/2")
    with st.expander("The volatility risk premium and its state machine"):
        st.latex(glossary.METRICS["vrp30"][2])
        st.markdown("**Slope is the alarm, VRP is the confirmation, the direction of the slope is the "
                    "all-clear.** The VRP30 slope (OLS, vol points per session) is standardised by the "
                    "volatility of daily VRP changes; a sharply negative standardised slope while VRP is "
                    "still positive raises ALARM, VRP at or below zero CONFIRMS, and a slope that turns and "
                    "stays up moves through RECOVERING to ALL-CLEAR. This is a practitioner heuristic, not a "
                    "published result — every threshold is exposed in config.VOLSCREEN_STATE.")
        st.markdown(volviz.state_diagram("NEUTRAL", 0, 300), unsafe_allow_html=True)
    with st.expander("Term structure: what the research does and does not say"):
        st.markdown("- **Time series (index level):** a steep VIX contango is the calm regime in which "
                    "short volatility has historically been paid; backwardation marks stress (Johnson 2017; "
                    "Simon & Campasano 2014). The checklist rewards contango for this reason.\n"
                    "- **Cross-section (single names):** Vasquez (2017) finds names with *upward-sloping* IV "
                    "term structures earn *higher* straddle returns — i.e. among stocks, front-rich (inverted) "
                    "names are the better sales, unless an event explains the inversion. The checklist treats "
                    "an earnings-explained inversion as neutral and flags only unexplained backwardation.")
    with st.expander("Liquidity rating and classes"):
        st.markdown(glossary.help("stars") + " " + glossary.help("class"))
    with st.expander("References"):
        for a, t in glossary.REFERENCES:
            st.markdown(f"- **{a}** — {t}")


# ----------------------------------------------------------------- page --

def render() -> None:
    d = _data()
    sc, status = d["screen"], d["status"]
    rows = sc.get("rows") or []
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    st.markdown(evidence_rating("B", "VRP strong for indices, weaker for single names", _EVIDENCE),
                unsafe_allow_html=True)
    if not rows:
        st.info("No screen yet — the nightly VOL SCREENER job has not produced data/volscreen/screen.json.")
        _learn()
        st.caption(DISCLAIMER)
        return
    st.markdown(stamp(f"{sc.get('as_of')} · quotes {sc.get('quote_ts', '')} ET", "VOL SCREENER"),
                unsafe_allow_html=True)
    notes = []
    if sc.get("session") == "post-close":
        notes.append("Quotes were captured after the close, when market makers widen or pull them: today's "
                     "star ratings use volume, open interest and strike depth only, and the spread columns "
                     "overstate real trading costs (IVs are unaffected).")
    obs = status.get("observed_days", 0)
    if obs < 252:
        notes.append(f"IV history building: {obs}/252 observed sessions. IV rank and percentile use index-"
                     "seeded (Cboe vol indices) or MODELED history until a full year accrues.")
    stale = [r["ticker"] for r in rows if r.get("stale")]
    if stale:
        notes.append(f"{len(stale)} chain(s) were served from an earlier session by Cboe's CDN and carry that "
                     f"session's quotes: {', '.join(stale[:12])}{' …' if len(stale) > 12 else ''}.")
    if notes:
        st.markdown(uc.note_strip("Read first", notes), unsafe_allow_html=True)
    st.radio("Lens", MODES, horizontal=True, key="vs_mode")
    _kpis(sc)
    _signature(sc)
    df = _frame(rows)
    sub = st.radio("Section", SUBVIEWS, horizontal=True, key="vs_sub", label_visibility="collapsed")
    if sub == "LIQUIDITY":
        _liquidity(sc, df)
    elif sub == "SCREEN":
        _screen(sc, df)
    elif sub == "CHECKLIST":
        _checklist(sc, df)
    elif sub == "BASKETS":
        _baskets(sc)
    else:
        _learn()
    st.caption(DISCLAIMER)

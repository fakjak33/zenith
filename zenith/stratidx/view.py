"""STRATEGY INDICES page — which option strategies the market is paying.

Reads committed JSON (data/stratidx/*). Sections on one radio (key `si_sub`):
OVERVIEW (what's-working matrix, environment read, vol context, breadth),
INDICES (table + chart + per-index card with methodology), LEARN.
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
from ..ui_theme import evidence_rating, key_findings, section, stamp
from ..viz import volviz
from . import DISCLAIMER, load
from .registry import FAMILIES, INDICES, methodology_url

SUBVIEWS = ["OVERVIEW", "INDICES", "LEARN"]
_FINDINGS = [
    {"stat": "Over 1986-2018 the put-write PUT index earned a return close to the S&P 500 with about two-"
             "thirds of its volatility — the variance premium at work.", "cite": "Bondarenko (2019), Cboe study"},
    {"stat": "A buy-write on the S&P 500 matched the index's return with lower volatility over its first "
             "fifteen years.", "cite": "Whaley (2002)"},
    {"stat": "Time-series momentum (own past 12-month return) is positive across asset classes and has held "
             "for a century of data.", "cite": "Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017)"},
    {"stat": "Put protection has cost more in calm years than it repaid in crashes for the S&P 500.",
     "cite": "Israelov (2019), 'Pathetic Protection'"},
]
TONE = {"pos": T.teal, "neg": T.coral, "neu": T.amber}


@st.cache_data(ttl=900, show_spinner=False)
def _data() -> dict:
    return {k: load(k, {}) or {} for k in ("indices", "regime", "vol_context", "breadth", "methodology", "status")}


def today_badge() -> str | None:
    try:
        r = load("regime", {}) or {}
        m = (r.get("matrix") or {}).get("PUT-WRITE", {}).get("3M", {})
        x = m.get("x_cash")
        if x is None:
            return None
        return uc.chip(f"STRATEGY INDICES — short vol {'paid' if x > 0 else 'not paid'}",
                       color=T.teal if x > 0 else T.coral, sub=f"put-write {x * 100:+.1f}% vs T-bills, 3M")
    except Exception:
        return None


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _pc(x, d=1) -> str:
    return f"{x * 100:+.{d}f}%" if _ok(x) else "—"


def _overview(d: dict) -> None:
    reg, vc, br = d["regime"], (d["vol_context"].get("series") or {}), d["breadth"]
    mat = reg.get("matrix") or {}
    st.markdown(section("What's working — strategy family × window", 0,
                        help="Average return of each family's indices over the window, minus T-bills (or the "
                             "S&P 500 total return). Teal = the strategy beat the benchmark."), unsafe_allow_html=True)
    vs = st.radio("Excess over", ["T-BILLS", "S&P 500 TR"], horizontal=True, key="si_vs")
    key = "x_cash" if vs == "T-BILLS" else "x_sptr"
    wins = reg.get("windows", ["1M", "3M", "6M", "1Y"])
    vals, sub = {}, {}
    for fam in FAMILIES:
        for w in wins:
            cell = (mat.get(fam) or {}).get(w) or {}
            vals[(fam, w)] = cell.get(key)
            if _ok(cell.get("ret")):
                sub[(fam, w)] = f"raw {cell['ret'] * 100:+.1f}%"
    cap = max([abs(v) for v in vals.values() if _ok(v)] + [0.02])
    st.markdown(volviz.heat_matrix(list(FAMILIES), wins, vals, cap, sub), unsafe_allow_html=True)
    env = reg.get("environment") or []
    if env:
        st.markdown(section("Environment read", 1), unsafe_allow_html=True)
        st.markdown("".join(
            f'<div style="border-left:3px solid {TONE.get(e["tone"], T.muted)};padding:0.35rem 0.7rem;'
            f'margin:0 0 0.35rem 0;background:rgba(0,0,0,0.28);font-size:0.84rem">{html.escape(e["text"])}</div>'
            for e in env), unsafe_allow_html=True)
    st.markdown(section("Volatility context", 2,
                        help="Cboe's own indices: the VIX term structure (9 days to 1 year), VVIX (vol of "
                             "VIX), SKEW (tail demand), COR1M (implied correlation) and DSPX (dispersion)."),
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        pts = [(lab, days, (vc.get(k) or {}).get("last")) for lab, days, k in
               (("9D", 9, "VIX9D"), ("30D", 30, "VIX"), ("3M", 93, "VIX3M"), ("6M", 183, "VIX6M"), ("1Y", 365, "VIX1Y"))]
        st.markdown(volviz.vix_curve(pts, 320, 130), unsafe_allow_html=True)
        st.caption("VIX term structure today. Upward (teal) = contango, the calm regime.")
    with c2:
        items = []
        for k, lab in (("VIX", "VIX"), ("VVIX", "VVIX"), ("SKEW", "SKEW"), ("COR1M", "IMPLIED CORR 1M"),
                       ("DSPX", "DISPERSION")):
            s = vc.get(k) or {}
            if not s:
                continue
            p = s.get("pct1y")
            items.append({"label": lab, "value": f"{s.get('last', 0):.1f}",
                          "sub": f"1y pct {p * 100:.0f}" if _ok(p) else "",
                          "color": T.coral if _ok(p) and p > 0.8 else T.teal if _ok(p) and p < 0.3 else T.text})
        st.markdown(uk.kpis(items), unsafe_allow_html=True)
    if br.get("date"):
        st.markdown(section("Breadth — % of S&P 500 above its moving averages", 3,
                            help="The S5TW / S5FI / S5TH family, computed here from today's constituents "
                                 "(survivorship bias: past members that left the index are missing)."),
                    unsafe_allow_html=True)
        b = pd.DataFrame({"date": pd.to_datetime(br["date"]), "20-DAY": br["s5tw"], "50-DAY (S5FI)": br["s5fi"],
                          "200-DAY": br["s5th"]}).melt("date", var_name="sma", value_name="pct")
        last = {k: br[k][-1] for k in ("s5tw", "s5fi", "s5th")}
        st.markdown(uk.kpis([{"label": lab, "value": f"{last[k] * 100:.0f}%" if _ok(last[k]) else "—",
                              "color": T.teal if _ok(last[k]) and last[k] >= 0.5 else T.coral}
                             for k, lab in (("s5tw", "ABOVE 20D"), ("s5fi", "ABOVE 50D"), ("s5th", "ABOVE 200D"))]),
                    unsafe_allow_html=True)

        def build(alt):
            lines = alt.Chart(b).mark_line(strokeWidth=1.4).encode(
                x=alt.X("date:T", title=None), y=alt.Y("pct:Q", title="% OF MEMBERS", axis=alt.Axis(format="%"),
                                                       scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("sma:N", scale=alt.Scale(domain=["20-DAY", "50-DAY (S5FI)", "200-DAY"],
                                                          range=[T.violet, T.teal, T.amber]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["date:T", "sma", alt.Tooltip("pct:Q", format=".0%")])
            rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[3, 3], color=T.faint).encode(y="y:Q")
            return (rule + lines).properties(height=230)
        uc.render_chart(build, b, width="stretch")


def _table(ind: dict) -> pd.DataFrame:
    out = []
    for t, v in ind.items():
        p, s = v.get("perf", {}), v.get("signals", {})
        out.append({"INDEX": t, "FAMILY": v.get("family"), "NAME": v.get("name"), "3M": p.get("r3m"),
                    "YTD": p.get("ytd"), "1Y": p.get("r1y"), "3Y ANN": p.get("a3y"), "5Y ANN": p.get("a5y"),
                    "SHARPE 1Y": p.get("sharpe1y"), "SHARPE 3Y": p.get("sharpe3y"), "DD": p.get("dd"),
                    "MAX DD": p.get("maxdd"), "vs S&P 1Y": (v.get("rel_sptr") or {}).get("1Y"),
                    "TSMOM": {1: "▲", -1: "▼"}.get((s.get("tsmom") or {}).get("sign"), "·"),
                    "XS 12M": (s.get("xs") or {}).get("r12"), "EWMAC": s.get("ewmac"),
                    "DONCHIAN": (s.get("donchian") or {}).get("state"), "EMA": s.get("ema"),
                    "TREND": s.get("composite")})
    return pd.DataFrame(out)


def _indices(d: dict) -> None:
    ind = (d["indices"].get("indices") or {})
    sp = d["indices"].get("sptr") or {}
    if not ind:
        st.info("No index data yet.")
        return
    df = _table(ind)
    fams = st.multiselect("Families", list(FAMILIES), default=list(FAMILIES), key="si_fams")
    df = df[df["FAMILY"].isin(fams)]
    pct = {"3M", "YTD", "1Y", "3Y ANN", "5Y ANN", "DD", "MAX DD", "vs S&P 1Y"}
    cfg = {c: st.column_config.NumberColumn(c, format="percent") for c in pct}
    cfg["TREND"] = st.column_config.ProgressColumn("TREND", min_value=-1.0, max_value=1.0, format="%.2f",
                                                   help="Blend of TSMOM, EWMAC, cross-sectional rank and Donchian.")
    cfg["XS 12M"] = st.column_config.ProgressColumn("XS 12M", min_value=0.0, max_value=1.0, format="%.2f",
                                                    help="Percentile of 12M return across the tracked indices.")
    cfg["EWMAC"] = st.column_config.NumberColumn("EWMAC", format="%.1f", help="Seven-speed EMA crossover, -20..+20.")
    st.dataframe(df.sort_values("TREND", ascending=False), hide_index=True, width="stretch", column_config=cfg,
                 height=420)
    st.markdown(section("Rebased performance (log scale) vs the S&P 500 total return", 1), unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    pick = c1.multiselect("Indices", list(ind), default=[t for t in ("PUT", "BXM", "CNDR", "PPUT") if t in ind],
                          key="si_pick")
    yrs = c2.select_slider("Years", options=[1, 3, 5, 10, 15], value=5, key="si_yrs")
    rows = []
    start = pd.Timestamp.today() - pd.DateOffset(years=yrs)
    series = {t: ind[t]["monthly"] for t in pick}
    if sp.get("monthly"):
        series["S&P 500 TR"] = sp["monthly"]
    for t, m in series.items():
        s = pd.Series(m["v"], index=pd.to_datetime(m["d"])).dropna()
        s = s[s.index >= start]
        if len(s) < 2:
            continue
        s = s / s.iloc[0] * 100
        dd = s / s.cummax() - 1
        rows += [{"date": d0, "index": t, "level": v, "dd": x} for d0, v, x in zip(s.index, s.values, dd.values)]
    if rows:
        p = pd.DataFrame(rows)
        dom = list(series)
        rng = [T.cat[i % len(T.cat)] for i in range(len(dom) - 1)] + [T.muted]

        def build(alt):
            return alt.Chart(p).mark_line(strokeWidth=1.5).encode(
                x=alt.X("date:T", title=None), y=alt.Y("level:Q", scale=alt.Scale(type="log", zero=False),
                                                       title="GROWTH OF 100"),
                color=alt.Color("index:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(orient="bottom")),
                strokeDash=alt.condition(alt.datum.index == "S&P 500 TR", alt.value([4, 3]), alt.value([1, 0])),
                tooltip=["index", "date:T", alt.Tooltip("level:Q", format=".1f")]).properties(height=300)
        uc.render_chart(build, p, width="stretch")

        def build_dd(alt):
            return alt.Chart(p).mark_area(opacity=0.35, line=True).encode(
                x=alt.X("date:T", title=None), y=alt.Y("dd:Q", title="DRAWDOWN", axis=alt.Axis(format="%"), stack=None),
                color=alt.Color("index:N", scale=alt.Scale(domain=dom, range=rng), legend=None),
                tooltip=["index", "date:T", alt.Tooltip("dd:Q", format=".1%")]).properties(height=160)
        uc.render_chart(build_dd, p, width="stretch")
    st.markdown(section("Index card", 2), unsafe_allow_html=True)
    t = st.selectbox("Index", list(ind), key="si_card")
    v = ind[t]
    p, s = v["perf"], v["signals"]
    links = (d["methodology"].get("links") or {}).get(t, {})
    url = links.get("url") or methodology_url(t)
    st.markdown(f'<div style="font-size:0.95rem;letter-spacing:0.12em">{html.escape(t)} · '
                f'{html.escape(v["name"])} <span style="color:{T.muted};font-size:0.7rem">{html.escape(v["family"])}'
                f'</span></div><div style="color:{T.muted};font-size:0.8rem;margin:0.3rem 0 0.6rem 0;line-height:1.5">'
                f'{html.escape(v["rule"])}</div>'
                f'<div style="font-size:0.72rem"><a href="{html.escape(url)}" target="_blank" rel="noopener">'
                f'Official Cboe methodology (PDF) ↗</a>'
                + (f' · <a href="{html.escape(links["dashboard"])}" target="_blank" rel="noopener">Cboe dashboard ↗</a>'
                   if links.get("dashboard") else "")
                + ("" if links.get("ok", True) else f' <span style="color:{T.amber}">(link check failed)</span>')
                + "</div>", unsafe_allow_html=True)
    st.markdown(uk.kpis([
        {"label": "3M", "value": _pc(p.get("r3m")), "color": T.teal if (p.get("r3m") or 0) >= 0 else T.coral},
        {"label": "1Y", "value": _pc(p.get("r1y")), "sub": f"vs S&P {_pc((v.get('rel_sptr') or {}).get('1Y'))}"},
        {"label": "SHARPE 1Y / 3Y", "value": f"{p.get('sharpe1y') or 0:.2f} / {p.get('sharpe3y') or 0:.2f}"},
        {"label": "DRAWDOWN", "value": _pc(p.get("dd")), "sub": f"max {_pc(p.get('maxdd'))}",
         "color": T.coral if (p.get("dd") or 0) < -0.05 else T.text},
        {"label": "SINCE", "value": p.get("inception", "")[:4], "sub": f"5Y ann {_pc(p.get('a5y'))}"},
    ]), unsafe_allow_html=True)
    chips = []
    ts = s.get("tsmom") or {}
    if ts.get("sign") is not None:
        chips.append(uc.chip(f"TSMOM 12-1 {'▲' if ts['sign'] > 0 else '▼'}", T.teal if ts["sign"] > 0 else T.coral,
                             _pc(ts.get("ret"))))
    if _ok(s.get("ewmac")):
        chips.append(uc.chip(f"EWMAC {s['ewmac']:+.1f}", T.teal if s["ewmac"] > 0 else T.coral, "of ±20"))
    dn = s.get("donchian") or {}
    if dn.get("state"):
        chips.append(uc.chip(f"DONCHIAN {dn['state']}", {"BREAKOUT": T.teal, "BREAKDOWN": T.coral}.get(dn["state"], T.muted),
                             f"20d position {dn.get('pos20', 0) * 100:.0f}%"))
    if s.get("ema"):
        chips.append(uc.chip(f"EMA 50/200 {s['ema']}", {"UP": T.teal, "DOWN": T.coral}.get(s["ema"], T.amber)))
    xs = (s.get("xs") or {}).get("r12")
    if _ok(xs):
        chips.append(uc.chip(f"XS RANK {xs * 100:.0f}th pct", T.cyan, "12M across indices"))
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.markdown(uk.sparkline_svg(v.get("spark", []), T.teal, 420, 46, fill=True), unsafe_allow_html=True)


def _learn() -> None:
    st.markdown(section("The indices and the signals", 1), unsafe_allow_html=True)
    with st.expander("Every index's construction (from its Cboe methodology)", expanded=True):
        for t, (fam, name, rule, _) in INDICES.items():
            st.markdown(f"- **{t}** · {name} *({fam.lower()})* — {rule} "
                        f"[methodology]({methodology_url(t)})")
    with st.expander("Trend overlays"):
        st.markdown(
            "- **Time-series momentum (12-1)** — the sign of the index's own return from 12 months to 1 month "
            "ago; the size is that return over its volatility (Moskowitz, Ooi & Pedersen 2012).\n"
            "- **Cross-sectional rank** — where the index's 6- and 12-month returns rank among all tracked "
            "strategies (Jegadeesh & Titman 1993 applied to strategies).\n"
            "- **Donchian** — a close above the prior 55-day high is a BREAKOUT, below the 55-day low a "
            "BREAKDOWN; the 20-day position shows where it sits in its recent channel (Turtle rules).\n"
            "- **EWMAC** — seven EMA-crossover speeds scaled by volatility and averaged, −20..+20 (the same "
            "engine as the TREND FOLLOWING page).\n"
            "- **TREND** — the equal-weight blend of the four, −1..+1.")
    with st.expander("Reading the environment"):
        st.markdown(
            "- Put-write beating T-bills = the variance premium is being paid.\n"
            "- Condor beating put-write = a range-bound tape (both short wings paid); the reverse = direction "
            "mattered.\n- Put protection beating the S&P 500 = tails were priced too cheap.\n"
            "- Contango in VIX9D → VIX1Y = calm regime; inversion = stress.\n"
            "- High SKEW = expensive downside; high implied correlation (COR1M) = macro-driven, index-level "
            "risk; high dispersion (DSPX) = single-name stories dominate.")
    with st.expander("References"):
        st.markdown("- Whaley, R. (2002). Return and risk of CBOE Buy Write Monthly Index. *Journal of Derivatives* 10(2).\n"
                    "- Bondarenko, O. (2019). Historical performance of put-writing strategies. Cboe.\n"
                    "- Moskowitz, Ooi & Pedersen (2012). Time series momentum. *Journal of Financial Economics* 104(2).\n"
                    "- Hurst, Ooi & Pedersen (2017). A century of evidence on trend-following investing. "
                    "*Journal of Portfolio Management* 44(1).\n"
                    "- Jegadeesh & Titman (1993). Returns to buying winners and selling losers. *Journal of Finance* 48(1).\n"
                    "- Israelov, R. (2019). Pathetic protection: the elusive benefits of protective puts. "
                    "*Journal of Alternative Investments* 21(3).")


def render() -> None:
    d = _data()
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    st.markdown(evidence_rating("B", "documented premia, descriptive overlays",
                                "Put-write and buy-write premia are long-documented on Cboe's own data; trend "
                                "overlays on strategy returns describe the current environment and carry the "
                                "general time-series-momentum evidence, not a tested edge of their own."),
                unsafe_allow_html=True)
    if not (d["indices"].get("indices")):
        st.info("No strategy-index data yet — the nightly STRATEGY INDICES job fills data/stratidx/.")
        _learn()
        st.caption(DISCLAIMER)
        return
    st.markdown(stamp(d["indices"].get("as_of", ""), "STRATEGY INDICES"), unsafe_allow_html=True)
    sub = st.radio("Section", SUBVIEWS, horizontal=True, key="si_sub", label_visibility="collapsed")
    if sub == "OVERVIEW":
        _overview(d)
    elif sub == "INDICES":
        _indices(d)
    else:
        _learn()
    st.caption(DISCLAIMER)

"""LAB step 06 ATTRIBUTION and the verdict card (shown in 07 OUT-OF-SAMPLE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import evidence_rating, section
from .. import ledger as lg
from ..attrib import core as A
from ..attrib import verdict as V
from ..attrib.benchmarks import BENCHMARKS, cash_index, sixty_forty, suggest, to_returns
from ..engine import metrics
from ..robust.sensitivity import plateau, to_grid
from ..spec import StrategySpec, run_spec
from . import common as C

GOALS = ["Absolute return", "Beat buy-and-hold", "Crisis alpha / diversifier", "Income / carry"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cash() -> pd.Series:
    return cash_index()


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _factors() -> dict:
    from ...cas.backtest.factor_data import load_french_factors
    return load_french_factors()


def bench_returns(key: str, spec: StrategySpec, ins: pd.DataFrame) -> tuple[pd.Series, str]:
    """Raw (not vol-matched) benchmark returns on the strategy's bars, and its label."""
    idx = ins.index
    if key == "bh" or spec.src != "Market":
        return ins["close"].pct_change().fillna(0.0), f"buy & hold {spec.name}"
    if key == "SPY":
        return to_returns(C._market("SPY")["close"], idx), "SPY"
    if key == "6040":
        return sixty_forty(C._market("SPY")["close"], C._market("IEF")["close"], idx), "60/40"
    return to_returns(_cash(), idx), "cash"


def _pick_benchmark(spec: StrategySpec) -> str:
    goal = st.session_state.get("bt_goal", GOALS[0])
    sug = suggest(goal, spec.name, spec.direction, spec.src)
    if spec.src != "Market":
        st.caption("Synthetic markets have no real benchmark history — using buy & hold of the synthetic path.")
        return "bh"
    keys = [s.key for s in sug]
    why = {s.key: s.why for s in sug}
    k = st.radio("benchmark", keys, key="bt_bench", horizontal=True,
                 format_func=lambda x: ("★ " if x == keys[0] else "") + BENCHMARKS[x])
    st.caption(f"Goal: {goal.lower()} (set in DEFINE). ★ suggested — {why[keys[0]]}." +
               (f" This one: {why[k]}." if k != keys[0] and why.get(k) else ""))
    return k


# ================================================================ ATTRIBUTION
def causal_labels(bench_ret: pd.Series, periods: int, vix: pd.Series | None = None) -> pd.DataFrame:
    """Same dimensions as ``attrib.core.regime_labels`` but every label is known on its own day."""
    from ..regime import detect
    px = (1 + bench_ret.fillna(0)).cumprod()
    df = pd.DataFrame({"close": px})
    n_trend = max(10, int(round(periods * 200 / 252)))
    w = max(5, int(round(periods * 63 / 252)))
    out = pd.DataFrame({"trend": detect("trend", df, length=n_trend),
                        "volatility": detect("vol", df, length=w),
                        "drawdown": detect("drawdown", df).map({"in drawdown": "drawdown >10%",
                                                                 "near highs": "near highs"})}, index=px.index)
    if vix is not None and len(vix):
        v = vix.reindex(px.index.union(vix.index)).ffill().reindex(px.index)
        out["VIX"] = pd.cut(v, [0, 15, 20, 30, 1e9], labels=["<15", "15–20", "20–30", "30+"], right=False).astype(object)
    return out


def step(spec: StrategySpec, ins: pd.DataFrame, res, aux: dict) -> None:
    st.markdown(section("ATTRIBUTION — where did the returns come from?", 2), unsafe_allow_html=True)
    key = _pick_benchmark(spec)
    b, blabel = bench_returns(key, spec, ins)
    r, p = res.returns, res.periods
    cash = None
    if spec.src == "Market":
        cash = to_returns(_cash(), ins.index)
    if key == "cash":
        # vs cash the question is "excess return after removing equity beta"
        spy = to_returns(C._market("SPY")["close"], ins.index)
        ab = A.alpha_beta(r, spy, p, rf=cash)
        ex = (r - cash).dropna()
        ab["info_ratio"] = float(ex.mean() / ex.std(ddof=1) * np.sqrt(p)) if ex.std(ddof=1) > 1e-12 else np.nan
        blabel = "cash (after SPY beta)"
        reg_b = spy
    else:
        ab = A.alpha_beta(r, b, p, rf=cash if key != "bh" or spec.src == "Market" else None)
        reg_b = b
    C.memo()[f"attr|{spec.hash()}"] = {"alpha_ann": ab["alpha_ann"], "alpha_t": ab["alpha_t"], "beta": ab["beta"],
                                       "beta_share": ab["beta_share"], "bench": blabel}

    tone = lambda t: T.teal if t >= 2 else T.amber if t >= 1 else T.coral  # noqa: E731
    bs = ab["beta_share"]
    st.markdown(uk.kpis([
        {"label": "ALPHA / YR", "value": f"{ab['alpha_ann'] * 100:+.1f}%", "color": tone(ab["alpha_t"]),
         "sub": f"t = {ab['alpha_t']:.1f} (Newey-West)"},
        {"label": "BETA", "value": f"{ab['beta']:.2f}", "sub": f"t = {ab['beta_t']:.1f} · R² {ab['r2']:.2f}"},
        {"label": "RETURN FROM BETA", "value": "—" if not np.isfinite(bs) else f"{bs * 100:.0f}%",
         "color": T.teal if np.isfinite(bs) and bs < 0.5 else T.amber if np.isfinite(bs) and bs < 0.8 else T.coral,
         "sub": "of the average return"},
        {"label": "INFO RATIO", "value": f"{ab['info_ratio']:.2f}",
         "sub": "excess over cash" if key == "cash" else f"vs {blabel}"},
        {"label": "CORRELATION", "value": f"{ab['corr']:.2f}"},
    ]), unsafe_allow_html=True)
    if np.isfinite(bs) and bs >= 0.8:
        C.banner(T.coral, "MOSTLY BETA", f"most of the return is {blabel} exposure you could buy for a few basis "
                 "points — the timing adds little")

    tabs = st.tabs(["DECOMPOSITION", "UP / DOWN", "REGIMES", "LONG / SHORT", "YEARS", "ROLLING BETA", "FACTORS"])
    with tabs[0]:
        eq = (1 + r).cumprod()
        beq = (1 + ab["beta_ret"]).cumprod()
        heq = (1 + ab["hedged"]).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=beq.index, y=beq, name=f"beta part ({ab['beta']:.2f} × {blabel.split(' (')[0]})",
                                 line=dict(color=T.amber, width=1.2)))
        fig.add_trace(go.Scatter(x=heq.index, y=heq, name="beta-hedged (alpha + noise)",
                                 line=dict(color=T.violet, width=1.4)))
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name="strategy", line=dict(color=T.teal, width=1.8)))
        fig.update_yaxes(type="log", title_text="growth of 1")
        up.show(fig, 400, "strategy = beta part + what's left", key="bt_at_dec")
        st.caption("If the violet line is flat or falling, the strategy's return came from holding the benchmark, "
                   "not from skill.")
    with tabs[1]:
        cap = A.capture(r, reg_b)
        m = cap["months"]
        st.markdown(uk.kpis([
            {"label": "UP CAPTURE", "value": f"{cap['up_capture'] * 100:.0f}%",
             "sub": f"avg {cap['up_mean'] * 100:+.2f}%/mo in up months"},
            {"label": "DOWN CAPTURE", "value": f"{cap['down_capture'] * 100:.0f}%",
             "color": T.teal if cap["down_capture"] < 0.5 else T.amber if cap["down_capture"] < 1 else T.coral,
             "sub": f"avg {cap['down_mean'] * 100:+.2f}%/mo in down months"},
            {"label": "CONVEXITY γ", "value": f"{cap['gamma']:.2f}",
             "color": T.teal if cap["gamma_t"] > 2 else T.muted, "sub": f"t = {cap['gamma_t']:.1f} · + = smile"},
        ]), unsafe_allow_html=True)
        fig = go.Figure()
        col = np.where(m["b"] >= 0, T.teal, T.coral)
        fig.add_trace(go.Scatter(x=m["b"] * 100, y=m["r"] * 100, mode="markers", name="month",
                                 marker=dict(color=col, size=6, opacity=0.7),
                                 text=[d.strftime("%Y-%m") for d in m.index],
                                 hovertemplate="%{text}<br>bench %{x:.1f}% · strat %{y:.1f}%<extra></extra>"))
        xs = np.linspace(m["b"].min(), m["b"].max(), 60)
        c0, c1, c2 = cap["tm_coef"]
        fig.add_trace(go.Scatter(x=xs * 100, y=(c0 + c1 * xs + c2 * xs ** 2) * 100, name="quadratic fit",
                                 line=dict(color=T.violet, width=2)))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.add_vline(x=0, line=dict(color=T.line))
        fig.update_xaxes(title_text=f"{blabel} month %", showspikes=False)
        fig.update_yaxes(title_text="strategy month %", showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 420, "monthly returns vs benchmark", key="bt_at_sc")
    with tabs[2]:
        mode = st.radio("labels", ["causal (known at the time)", "ex-post (hindsight)"], horizontal=True,
                        key="bt_at_regmode",
                        help="Causal labels use only data available on each day (expanding thresholds), so a "
                             "difference between regimes is something you could have traded on. Ex-post labels "
                             "split volatility into terciles of the WHOLE sample — fine for describing where P&L "
                             "came from, but they peek at the future.")
        vix_s = None
        if spec.src == "Market":
            vix_s = aux.get("vix")
            if vix_s is None:
                try:
                    vix_s = C._market("^VIX")["close"]
                except Exception:  # noqa: BLE001
                    vix_s = None
        if mode.startswith("causal"):
            labels = causal_labels(reg_b, p, vix_s)
        else:
            labels = A.regime_labels(reg_b, p, vix_s)
        tbl = A.by_regime(r, labels, p)
        if tbl.empty:
            st.caption("Not enough history to split by regime.")
        else:
            dims = list(dict.fromkeys(tbl["dimension"]))
            fig = make_subplots(rows=1, cols=len(dims), subplot_titles=[d.upper() for d in dims])
            for i, d in enumerate(dims, 1):
                t = tbl[tbl.dimension == d]
                fig.add_trace(go.Bar(x=t["regime"], y=t["sharpe"],
                                     marker_color=[T.teal if v > 0 else T.coral for v in t["sharpe"]],
                                     text=[f"{v * 100:.0f}% time" for v in t["time"]], textposition="outside",
                                     textfont=dict(size=9), showlegend=False), 1, i)
            fig.update_annotations(font=dict(size=10, color=T.muted))
            fig.update_yaxes(title_text="Sharpe", col=1)
            fig.update_layout(hovermode="closest")
            up.show(fig, 340, f"sharpe by regime ({mode.split(' (')[0]} labels)", key="bt_at_reg")
            show = tbl.copy()
            for k in ("time", "ann_return", "share_of_pnl"):
                show[k] = show[k].map(lambda v: C.fmt("CAGR", v))
            show["sharpe"] = show["sharpe"].map(lambda v: C.fmt("x", v))
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("Regimes come from the benchmark's own trend, volatility and drawdown (and VIX). "
                       + ("Causal labels: each day's label used only what was known that day (BUILD › REGIMES "
                          "has more detectors)." if mode.startswith("causal") else
                          "Ex-post labels are set with hindsight: they describe where the P&L was made, they are "
                          "not a trading signal."))
    with tabs[3]:
        lt = A.legs(r, res.position, p)
        fig = go.Figure(go.Bar(x=lt["leg"], y=lt["ann_contrib"] * 100,
                               marker_color=[T.teal, T.coral, T.faint],
                               text=[f"{v * 100:.0f}% of time" for v in lt["time"]], textposition="outside"))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.update_yaxes(title_text="% per year")
        fig.update_layout(hovermode="closest")
        up.show(fig, 320, "return contribution by side held", key="bt_at_legs")
        if spec.direction == "both" and lt.loc[1, "ann_contrib"] < 0 < lt.loc[0, "ann_contrib"]:
            C.banner(T.amber, "SHORTS COST", "the short side loses money — consider long-only and compare in BUILD")
    with tabs[4]:
        y = A.yearly(r, reg_b)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=y.index, y=y["benchmark"] * 100, name=blabel, marker_color=up.fill(T.amber, 0.6)))
        fig.add_trace(go.Bar(x=y.index, y=y["strategy"] * 100, name="strategy", marker_color=T.teal))
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(title_text="% return")
        up.show(fig, 340, "calendar-year returns", key="bt_at_yr")
    with tabs[5]:
        rb = A.rolling_beta(r, reg_b, max(12, p))
        fig = go.Figure(go.Scatter(x=rb.index, y=rb, line=dict(color=T.violet, width=1.4), name="1-yr beta"))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.add_hline(y=ab["beta"], line=dict(color=T.faint, dash="dot"), annotation_text="full-period",
                      annotation_font=dict(size=9, color=T.muted))
        up.show(fig, 300, f"rolling 1-year beta to {blabel}", key="bt_at_rb")
    with tabs[6]:
        _factor_tab(spec, r)


def _factor_tab(spec: StrategySpec, r: pd.Series) -> None:
    if spec.src != "Market":
        st.caption("Factor regressions need real dates — switch to Market data.")
        return
    key = f"ff|{spec.hash()}"
    if st.button("RUN FAMA-FRENCH 5 + MOMENTUM REGRESSION", key="bt_ffrun",
                 help="Downloads Ken French's free factor library once (cached for a week)."):
        with st.spinner("loading factor library"):
            C.memo()[key] = A.factor_regression(r, _factors(), "US")
    out = C.memo().get(key)
    if out is None:
        st.caption("Regress monthly returns on market, size, value, profitability, investment and momentum: a "
                   "trend strategy that loads heavily on WML is mostly the momentum factor.")
        return
    if not out.get("ok"):
        C.banner(T.amber, "NO FACTORS", "factor library unavailable or too little overlap (need 24+ months)")
        return
    names, coef, se, t = out["names"][1:], out["coef"][1:], out["se"][1:], out["t"][1:]
    fig = go.Figure(go.Bar(x=coef, y=names, orientation="h",
                           marker_color=[T.teal if abs(v) >= 2 else T.faint for v in t],
                           error_x=dict(type="data", array=1.96 * se, color=T.muted, thickness=1, width=4)))
    fig.add_vline(x=0, line=dict(color=T.text, width=1))
    fig.update_yaxes(autorange="reversed", showspikes=False)
    fig.update_xaxes(title_text="loading (95% CI)", showspikes=False)
    fig.update_layout(hovermode="closest")
    up.show(fig, 320, f"factor loadings · {out['start']} → {out['end']} · R² {out['r2']:.2f}", key="bt_at_ff")
    st.markdown(uk.kpis([
        {"label": "FACTOR ALPHA / YR", "value": f"{out['alpha_ann'] * 100:+.1f}%",
         "color": T.teal if out["t"][0] >= 2 else T.amber if out["t"][0] >= 1 else T.coral,
         "sub": f"t = {out['t'][0]:.1f} after 6 factors"},
        {"label": "R²", "value": f"{out['r2']:.2f}", "sub": f"{out['n']} months"},
    ]), unsafe_allow_html=True)


# ================================================================ VERDICT
def _latest(prefix: str, h: str):
    hits = [v for k, v in C.memo().items() if k.startswith(f"{prefix}|{h}|") or k == f"{prefix}|{h}"]
    return hits[-1] if hits else None


def facts(spec: StrategySpec, ins: pd.DataFrame, full: pd.DataFrame, locked, res, n_trades: int, n_trials: int,
          aux: dict) -> dict:
    h = spec.hash()
    f = {"hypothesis": bool(spec.notes.strip()), "trades": n_trades, "years": len(res.returns) / res.periods,
         "dsr": res.stats["DSR"], "trials": n_trials, "n_filters": len(spec.filters),
         "sharpe_2x": run_spec(spec, ins, aux, cost_mult=2.0).stats["Sharpe"],
         "attr": C.memo().get(f"attr|{h}")}
    s = _latest("sens", h)
    if s is not None:
        pl = plateau(to_grid(s["df"], "Sharpe"))
        if pl.get("ok"):
            f["sensitivity"] = {"robustness": pl["robustness"], "share_pos": float((s["df"]["Sharpe"] > 0).mean())}
    pm = _latest("perm", h)
    if pm is not None:
        f["perm_p"] = pm["p"]
    nz = _latest("noise", h)
    if nz is not None:
        med = nz.groupby("k")["Sharpe"].median()
        if med.get(0.0, 0) > 0:
            f["noise_kept"] = float(med.get(0.25, np.nan) / med[0.0])
        else:
            f["noise_kept"] = 0.0
    wf = _latest("wf", h)
    if wf is not None:
        f["wf_eff"] = wf["efficiency"]
    ab = _latest("abl", h)
    if ab is not None:
        f["filters"] = list(ab["verdict"].iloc[1:])
    led = C.ledger()
    rev = lg.revealed(led, spec.family())
    if rev and locked is not None:
        if rev["hash"] != h:
            f["oos"] = {"contaminated": True}
        else:
            rr = run_spec(spec, full, aux).returns
            si = metrics.sharpe(rr[rr.index < locked], res.periods)
            so = metrics.sharpe(rr[rr.index >= locked], res.periods)
            f["oos"] = {"kept": so / si if si > 1e-9 else (so if so <= 0 else 0.0), "oos_sharpe": so,
                        "is_sharpe": si}
    return f


def card(f: dict) -> None:
    checks = V.build(f)
    tier, note = V.grade(checks)
    st.markdown(section("VERDICT", 0), unsafe_allow_html=True)
    st.markdown(evidence_rating(tier, {"A": "robust edge", "B": "promising, not proven",
                                       "C": "not yet evidence of an edge"}[tier], note), unsafe_allow_html=True)
    color = {V.PASS: T.teal, V.WARN: T.amber, V.FAIL: T.coral, V.NONE: T.faint}
    rows = "".join(
        f'<div style="display:flex; gap:0.8rem; align-items:baseline; border-top:1px solid {T.line}; '
        f'padding:0.32rem 0;"><span style="min-width:4.6rem; font-size:0.62rem; letter-spacing:0.16em; '
        f'color:{color[c.status]};">{c.status}</span><span style="min-width:12rem; font-size:0.78rem; '
        f'color:{T.text};">{C.esc(c.name)}</span><span style="font-size:0.74rem; color:{T.muted};">'
        f'{C.esc(c.detail)}</span></div>' for c in checks)
    st.markdown(f'<div style="margin:0.2rem 0 0.8rem 0;">{rows}</div>', unsafe_allow_html=True)

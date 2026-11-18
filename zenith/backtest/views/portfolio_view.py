"""BACKTEST / PORTFOLIO — combine saved strategies into one vol-targeted book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import key_findings, section
from .. import library
from ..engine import metrics
from ..portfolio import combine
from ..spec import run_spec, split_holdout
from . import common as C

_FINDINGS = [
    {"stat": "Combining imperfectly correlated strategies raises the Sharpe of the whole above the average of its "
             "parts; the diversification multiplier restores the risk the blend gave away.",
     "cite": "Carver, Systematic Trading (2015)"},
    {"stat": "Trend following across many markets earned positive returns in each decade for over a century, while "
             "any single market had long losing stretches.", "cite": "Hurst, Ooi & Pedersen (2017)"},
]
DEMO = [("EWMAC 16/64 · SPY", "SPY", "ema"), ("EWMAC 16/64 · TLT", "TLT", "ema"), ("EWMAC 16/64 · GLD", "GLD", "ema"),
        ("TSMOM 252 · DBC", "DBC", "tsmom")]


@st.cache_data(ttl=3600, show_spinner=False)
def _component(spec_json: str) -> tuple[pd.Series, float]:
    from ..spec import StrategySpec
    spec = StrategySpec.from_json(spec_json)
    ins, _ = split_holdout(C.load(spec), spec.holdout)
    res = run_spec(spec, ins, C.load_aux(spec))
    return res.returns, res.periods


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    store = C.saved()
    if len(store) < 2:
        C.banner(T.amber, "SAVE STRATEGIES FIRST", "save two or more strategies from LAB › BUILD (the SAVE box), or "
                 "load the demo set to see how this page works")
        if st.button("LOAD DEMO SET — trend on SPY, TLT, GLD, DBC", key="pf_demo"):
            from ..spec import StrategySpec
            for nm, tkr, rule in DEMO:
                library.save(store, nm, StrategySpec(name=tkr, rule=rule, params={}))
            st.rerun()
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    pick = c1.multiselect("strategies", list(store), default=list(store)[: min(6, len(store))], key="pf_pick")
    method = c2.radio("weights", ["equal", "inverse_vol"], key="pf_w", format_func=lambda m: m.replace("_", " "),
                      help="Equal weights fit nothing. Inverse-vol uses each strategy's trailing year only.")
    window = c3.slider("lookback (bars)", 63, 756, 252, 21, key="pf_win")
    if len(pick) < 2:
        st.caption("Pick at least two.")
        return

    rets, periods = {}, 252
    for nm in pick:
        try:
            r, periods = _component(library.spec_of(store[nm]).to_json())
            rets[nm] = r
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "SKIPPED", f"{nm}: {exc}")
    if len(rets) < 2:
        return
    df = pd.DataFrame(rets).dropna()
    if len(df) < 2 * window:
        C.banner(T.amber, "SHORT OVERLAP", f"only {len(df)} common in-sample bars — the strategies' dates barely overlap")
    out = combine(df, method, window)
    port = out["returns"]
    ps = metrics.summary(port, periods=periods)
    comp = {k: metrics.summary(v, periods=periods) for k, v in df.items()}
    avg_sr = float(np.mean([c["Sharpe"] for c in comp.values()]))
    st.caption(f"In-sample only: each strategy is cut at its own holdout, and the book uses the "
               f"{df.index[0].date()} → {df.index[-1].date()} overlap.")
    st.markdown(uk.kpis([
        {"label": "PORTFOLIO SHARPE", "value": f"{ps['Sharpe']:.2f}", "color": T.teal if ps["Sharpe"] > avg_sr else T.amber,
         "sub": f"avg component {avg_sr:.2f}"},
        {"label": "DIVERSIFICATION", "value": f"{out['div_ratio']:.2f}×", "sub": "Σ wσ ÷ σ_portfolio"},
        {"label": "IDM (LATEST)", "value": f"{out['idm'].iloc[-1]:.2f}", "sub": "capped at 2.5"},
        {"label": "VOL", "value": f"{ps['Vol'] * 100:.1f}%"},
        {"label": "MAX DD", "value": f"{ps['MaxDD'] * 100:.1f}%", "color": T.coral,
         "sub": f"worst component {min(c['MaxDD'] for c in comp.values()) * 100:.1f}%"},
        {"label": "CAGR", "value": f"{ps['CAGR'] * 100:.1f}%"},
    ]), unsafe_allow_html=True)

    tabs = st.tabs(["EQUITY", "CORRELATION", "RISK", "WEIGHTS & IDM", "STATS"])
    with tabs[0]:
        fig = go.Figure()
        for i, (k, v) in enumerate(df.items()):
            eq = (1 + v).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=k, line=dict(color=T.cat[(i + 1) % len(T.cat)], width=1),
                                     opacity=0.75))
        peq = (1 + port).cumprod()
        fig.add_trace(go.Scatter(x=peq.index, y=peq, name="PORTFOLIO", line=dict(color=T.teal, width=2.4)))
        fig.update_yaxes(type="log", title_text="growth of 1")
        up.show(fig, 440, "portfolio vs components", key="pf_eq")
    with tabs[1]:
        c = out["corr"]
        fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=list(c.columns), y=list(c.index), zmin=-1, zmax=1,
                                   colorscale=up.DIVERGING, text=np.round(c.to_numpy(), 2), texttemplate="%{text}",
                                   xgap=1, ygap=1, colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange="reversed", showspikes=False)
        fig.update_xaxes(showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 120 + 40 * len(c), "return correlation", key="pf_corr")
        off = c.to_numpy()[np.triu_indices(len(c), 1)]
        st.caption(f"Average pairwise correlation {off.mean():.2f}. Below ~0.3 is where combining pays most.")
    with tabs[2]:
        rc = out["risk_contrib"]
        w = out["weights"].iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(rc.index), y=w.values * 100, name="weight", marker_color=up.fill(T.amber, 0.6)))
        fig.add_trace(go.Bar(x=list(rc.index), y=rc.values * 100, name="risk contribution", marker_color=T.violet))
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(title_text="%")
        up.show(fig, 340, "capital weight vs share of portfolio risk", key="pf_rc")
    with tabs[3]:
        wdf = out["weights"]
        fig = go.Figure()
        for i, k in enumerate(wdf.columns):
            fig.add_trace(go.Scatter(x=wdf.index, y=wdf[k], stackgroup="w", name=k,
                                     line=dict(width=0.5, color=T.cat[(i + 1) % len(T.cat)])))
        up.show(fig, 300, "weights through time", key="pf_w_t")
        fig = go.Figure(go.Scatter(x=out["idm"].index, y=out["idm"], line=dict(color=T.teal, width=1.4)))
        fig.update_yaxes(title_text="IDM")
        up.show(fig, 240, "instrument diversification multiplier", key="pf_idm")
    with tabs[4]:
        keys = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "Skew"]
        tbl = pd.DataFrame({"PORTFOLIO": [C.fmt(k, ps[k]) for k in keys],
                            **{n: [C.fmt(k, comp[n][k]) for k in keys] for n in comp}}, index=keys)
        st.dataframe(tbl, use_container_width=True)
    st.markdown(section("SAVED STRATEGIES", 0), unsafe_allow_html=True)
    for nm, rec in list(store.items()):
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{C.esc(nm)} · {rec['saved_at']} · hash {rec['hash']}")
        if c2.button("remove", key=f"pf_del_{rec['hash']}_{nm}"):
            library.delete(store, nm)
            st.rerun()

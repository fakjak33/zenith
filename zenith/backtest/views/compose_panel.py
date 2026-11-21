"""BUILD panels for composed strategies: the BLEND breakdown and the
regime-filter luck test (SharpeEdge: benchmark a regime filter against random
filters with the same exposure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from ..builder import random_mask_test, regime_labels
from ..config import caps
from ..spec import StrategySpec, run_spec, with_aux
from . import common as C


def blend_tab(spec: StrategySpec, ins: pd.DataFrame, aux: dict) -> None:
    st.markdown(section("BLEND — what each signal contributes", 2,
                        help="Carver: blending imperfectly correlated forecasts reduces noise; the forecast "
                             "diversification multiplier (FDM) scales the average back up so the blend is not "
                             "timid."), unsafe_allow_html=True)
    strat = spec.strategy()
    frame = with_aux(ins, aux, spec.needs())
    comp = strat.components(frame)
    if comp.shape[1] < 2:
        st.caption("Add at least two signals with a non-zero weight.")
        return
    w = np.array([x[2] for x in strat.rules()], dtype=float)
    w = w / w.sum()
    corr = comp.corr()
    fdm = strat.fdm(frame)
    singles = {}
    for (name, s, _), part in zip(strat.rules(), strat.parts):
        sp = StrategySpec.from_dict({**spec.to_dict(), "mode": "single", "rule": part["rule"],
                                     "params": part.get("params") or {}})
        singles[name] = run_spec(sp, ins, aux).stats["Sharpe"]
    blend_sr = run_spec(spec, ins, aux).stats["Sharpe"]
    avg_corr = (corr.to_numpy().sum() - len(corr)) / max(len(corr) * (len(corr) - 1), 1)
    st.markdown(uk.kpis([
        {"label": "BLEND SHARPE", "value": f"{blend_sr:.2f}", "color": T.teal if blend_sr > max(singles.values())
         else T.amber, "sub": f"best single {max(singles.values()):.2f}"},
        {"label": "FDM", "value": f"{fdm:.2f}×", "sub": "diversification multiplier"},
        {"label": "AVG CORRELATION", "value": f"{avg_corr:+.2f}", "sub": "between forecasts",
         "color": T.teal if avg_corr < 0.5 else T.amber},
    ]), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        labels = [n.split("·", 1)[1][:28] for n in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale=up.DIVERGING,
                                   text=corr.map(lambda v: f"{v:.2f}").to_numpy(), texttemplate="%{text}",
                                   showscale=False, hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showspikes=False)
        fig.update_xaxes(showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 320, "forecast correlation", key="bt_bl_corr")
    with c2:
        fig = go.Figure(go.Bar(x=[n.split("·", 1)[1][:28] for n in singles], y=list(singles.values()),
                               marker_color=T.violet, name="alone"))
        fig.add_hline(y=blend_sr, line=dict(color=T.teal, width=2, dash="dash"),
                      annotation_text=f"blend {blend_sr:.2f}", annotation_font=dict(color=T.teal, size=10))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_yaxes(title_text="Sharpe")
        up.show(fig, 320, "each signal alone vs the blend", key="bt_bl_sr")
    contrib = comp.mul(w, axis=1) * fdm
    m = contrib.resample("ME").mean()
    fig = go.Figure()
    for i, col in enumerate(m.columns):
        fig.add_trace(go.Bar(x=m.index, y=m[col], name=col.split("·", 1)[1][:32],
                             marker_color=T.cat[i % len(T.cat)]))
    fig.update_layout(barmode="relative", hovermode="x unified")
    fig.update_yaxes(title_text="forecast contribution")
    up.show(fig, 320, "who drives the position · monthly average contribution to the forecast", key="bt_bl_ctb")
    st.caption("Trials: the blend is one strategy, but each component you tried before blending counted too.")


def luck_test(spec: StrategySpec, ins: pd.DataFrame, aux: dict) -> None:
    reg = spec.regime or {}
    if not reg.get("on") or reg.get("use") != "filter":
        return
    with st.expander("IS THE REGIME FILTER BETTER THAN LUCK?", expanded=False):
        st.caption("SharpeEdge's benchmark: compare your regime filter with many random filters that keep the "
                   "same exposure and the same spell lengths (your on/off pattern, slid in time). If random "
                   "timing does as well, the regime adds nothing.")
        n = min(1000, caps().permutations * 5)
        key = f"luck|{spec.hash()}|{n}"
        if st.button(f"RUN · {n} random filters", key="bt_luck"):
            base = StrategySpec.from_dict({**spec.to_dict(), "regime": {}})
            r = run_spec(base, ins, aux).returns
            lab = regime_labels(reg, with_aux(ins, aux, spec.needs()))
            C.memo()[key] = random_mask_test(r, lab, set(reg.get("allow") or []), n=n,
                                             periods=run_spec(base, ins, aux).periods)
        out = C.memo().get(key)
        if out is None:
            return
        st.markdown(uk.kpis([
            {"label": "FILTERED SHARPE", "value": f"{out['actual']:.2f}", "sub": f"unfiltered {out['unfiltered']:.2f}"},
            {"label": "P-VALUE", "value": f"{out['p']:.3f}", "sub": "share of random filters ≥ yours",
             "color": T.teal if out["p"] < 0.05 else T.amber if out["p"] < 0.2 else T.coral},
            {"label": "EXPOSURE", "value": f"{out['exposure'] * 100:.0f}%", "sub": "time in the market"},
        ]), unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=out["sims"], nbinsx=50, marker_color=up.fill(T.violet, 0.6),
                                     marker_line=dict(width=0), name="random filters"))
        fig.add_vline(x=out["actual"], line=dict(color=T.teal, width=2),
                      annotation_text="your filter", annotation_font=dict(color=T.teal, size=10))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text="Sharpe", showspikes=False)
        up.show(fig, 280, "sharpe of random same-exposure filters", key="bt_luck_h")
        st.caption("Approximation: costs of switching are ignored for every filter alike.")

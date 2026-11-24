"""Regime monitoring inside the LAB: detector controls, chart ribbons and the
BUILD › REGIMES tab (performance by regime, transitions, spell lengths)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_charts as uc
from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import glossary as G
from ..regime import DETECTORS, by_regime, current, detect, spells, transition_matrix, two_way
from . import common as C

MAX_SPELLS = 600


def palette() -> list[str]:
    return [T.teal, T.amber, T.coral, T.violet, getattr(T, "mint", T.teal), T.text, T.muted]


def color_map(labels: pd.Series, order: tuple = ()) -> dict:
    keys = [k for k in order if k in set(labels.dropna())] + sorted(set(labels.dropna()) - set(order))
    pal = palette()
    return {k: pal[i % len(pal)] for i, k in enumerate(keys)}


def controls(prefix: str = "bt_rg", default: str = "vol", src: str = "Market") -> tuple[str, dict, int]:
    keys = [k for k, d in DETECTORS.items() if src == "Market" or not d.needs]
    if st.session_state.get(f"{prefix}_det") not in keys:
        st.session_state[f"{prefix}_det"] = default if default in keys else keys[0]
    key = st.selectbox("regime detector", keys, key=f"{prefix}_det", format_func=lambda k: DETECTORS[k].label,
                       help=G.REGIME_HELP)
    d = DETECTORS[key]
    st.caption(d.help + (f" · {d.cite}" if d.cite else ""))
    params = {}
    if d.params:
        cols = st.columns(min(3, len(d.params)))
        for i, (pn, pt) in enumerate(d.params.items()):
            with cols[i % len(cols)]:
                params[pn] = C._num(pn, pt[:4], f"{prefix}_{key}_{pn}", help=pt[4])
    mb = int(st.slider("minimum spell (bars)", 1, 20, 3, key=f"{prefix}_minbars",
                       help="Hysteresis: a new regime only counts once it has lasted this many bars in a row. "
                            "Kills one-bar flip-flops (SharpeEdge's minimum-duration rule)."))
    return key, params, mb


def labels_for(spec, df: pd.DataFrame, key: str, params: dict, min_bars: int) -> pd.Series:
    """Detector labels on ``df`` (memoised per session)."""
    mk = ("regime", spec.family(), key, tuple(sorted(params.items())), min_bars, len(df), str(df.index[-1]))
    memo = C.memo()
    if mk in memo:
        return memo[mk]
    d = DETECTORS[key]
    frame = df
    if "vix" in d.needs and spec.src == "Market":
        try:
            frame = df.assign(vix=C._market("^VIX")["close"].reindex(df.index).ffill())
        except Exception:  # noqa: BLE001
            pass
    with st.spinner(f"labelling regimes · {d.label}"):
        lab = detect(key, frame, min_bars=min_bars, **params)
    memo[mk] = lab
    return lab


def ribbon(fig: go.Figure, labels: pd.Series, order: tuple = (), opacity: float = 0.13,
           legend: bool = True) -> go.Figure:
    """Shade the background of ``fig`` by regime (one rectangle per spell)."""
    sp = spells(labels)
    if sp.empty:
        return fig
    if len(sp) > MAX_SPELLS:
        sp = spells(labels.where(labels.groupby((labels != labels.shift()).cumsum()).transform("size") >= 5)
                    .ffill())
    cmap = color_map(labels, order)
    idx = labels.index
    pos = {d: i for i, d in enumerate(idx)}
    shapes = list(fig.layout.shapes or [])
    for _, row in sp.iterrows():
        j = pos.get(row["end"], len(idx) - 1)
        x1 = idx[min(j + 1, len(idx) - 1)]
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=row["start"], x1=x1, y0=0, y1=1,
                           fillcolor=up.fill(cmap[row["label"]], opacity), line=dict(width=0), layer="below"))
    fig.update_layout(shapes=shapes)
    if legend:
        for k, c in cmap.items():
            fig.add_trace(go.Scatter(x=[idx[0]], y=[None], mode="markers", name=f"▮ {k}",
                                     marker=dict(symbol="square", size=10, color=up.fill(c, 0.6)),
                                     hoverinfo="skip"))
    return fig


def ribbon_settings(spec, prefix: str = "bt_rg") -> tuple[bool, tuple | None]:
    """The BUILD toolbar: ribbon toggle + a popover with the detector settings."""
    c1, c2 = st.columns([1, 3])
    on = c1.toggle("regime ribbon", key="bt_ribbon", help="Shade EQUITY and PRICE charts by market regime. "
                   + G.REGIME_HELP)
    with c2.popover("REGIME SETTINGS", use_container_width=False):
        cfg = controls(prefix, src=spec.src)
    return on, cfg


def regimes_tab(spec, ins: pd.DataFrame, res, b: pd.Series, cfg: tuple) -> None:
    key, params, mb = cfg
    d = DETECTORS[key]
    lab = labels_for(spec, ins, key, params, mb)
    r, p = res.returns, res.periods
    if lab.dropna().empty:
        C.banner(T.amber, "NO LABELS", f"{d.label} has no labels here (needs {', '.join(d.needs) or 'more history'}).")
        return
    tbl = by_regime(r, lab, p, bench=b, order=d.order)
    now, run = current(lab)
    sp = spells(lab)
    st.markdown(uk.kpis([
        {"label": "CURRENT REGIME", "value": (now or "—").upper(), "sub": f"for {run} bars",
         "color": color_map(lab, d.order).get(now, T.text)},
        {"label": "SPELLS", "value": f"{len(sp)}", "sub": f"avg {sp['bars'].mean():.0f} bars"},
        {"label": "BEST REGIME", "value": (tbl.sort_values("sharpe").iloc[-1]["regime"].upper() if not tbl.empty
                                           else "—"), "sub": "highest Sharpe", "color": T.teal},
        {"label": "WORST REGIME", "value": (tbl.sort_values("sharpe").iloc[0]["regime"].upper() if not tbl.empty
                                            else "—"), "sub": "lowest Sharpe", "color": T.coral},
    ]), unsafe_allow_html=True)
    if "hmm" == key:
        cut = int(len(ins) * params.get("fit_pct", 50) / 100)
        st.caption(f"HMM fitted on bars before {ins.index[min(cut, len(ins) - 1)].date()}; labels after that are "
                   "the model running forward on data it never saw.")

    eq = (1 + r).cumprod()
    fig = go.Figure(go.Scatter(x=eq.index, y=eq, name="strategy", line=dict(color=T.teal, width=1.6)))
    beq = (1 + b).cumprod()
    fig.add_trace(go.Scatter(x=beq.index, y=beq, name="buy & hold (vol-matched)", line=dict(color=T.amber, width=1)))
    ribbon(fig, lab.reindex(eq.index), d.order)
    fig.update_yaxes(type="log", title_text="growth of 1")
    up.show(fig, 380, f"equity shaded by regime · {d.label}", key="bt_rg_eq")

    if tbl.empty:
        st.caption("Not enough bars in any regime.")
        return
    cmap = color_map(lab, d.order)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tbl["regime"], y=tbl["sharpe"], name="strategy", showlegend=False,
                         marker_color=[cmap.get(k, T.teal) for k in tbl["regime"]],
                         error_y=dict(type="data", symmetric=False, array=(tbl["ci_hi"] - tbl["sharpe"]).clip(lower=0),
                                      arrayminus=(tbl["sharpe"] - tbl["ci_lo"]).clip(lower=0), color=T.muted,
                                      thickness=1)))
    if "bench_sharpe" in tbl:
        fig.add_trace(go.Scatter(x=tbl["regime"], y=tbl["bench_sharpe"], mode="markers", name="buy & hold",
                                 marker=dict(symbol="diamond-open", size=11, color=T.text, line=dict(width=2))))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text="Sharpe (95% CI)")
    up.show(fig, 320, "sharpe by regime · whiskers = 95% bootstrap interval", key="bt_rg_bar")

    show = pd.DataFrame({
        "regime": tbl["regime"], "time": tbl["time"].map(lambda v: f"{v * 100:.0f}%"),
        "ann. return": tbl["ann_return"].map(lambda v: f"{v * 100:.1f}%"),
        "vol": tbl["vol"].map(lambda v: f"{v * 100:.1f}%"),
        "Sharpe": tbl["sharpe"].map(lambda v: C.fmt("x", v)),
        "95% CI": [f"{a:.2f} … {z:.2f}" if np.isfinite(a) else "—" for a, z in zip(tbl["ci_lo"], tbl["ci_hi"])],
        "hit rate": tbl["hit"].map(lambda v: f"{v * 100:.0f}%"),
        "b&h Sharpe": tbl.get("bench_sharpe", pd.Series(np.nan, index=tbl.index)).map(lambda v: C.fmt("x", v)),
        "spells": tbl["spells"], "avg bars": tbl["avg_bars"].map(lambda v: f"{v:.0f}" if np.isfinite(v) else "—"),
    })
    st.dataframe(show, use_container_width=True, hide_index=True, column_config=uc.colcfg(show.columns, {
        "time": "Share of in-sample bars spent in this regime.",
        "Sharpe": G.metric_help("Sharpe"),
        "95% CI": "Range the true Sharpe probably lies in (block bootstrap). If it spans 0, the regime difference "
                  "may be noise.",
        "hit rate": G.metric_help("HitRate"),
        "spells": "How many separate times the market entered this regime.",
        "avg bars": "Average length of one spell, in bars. Short spells = hard to trade on.",
    }))
    st.caption("Each return is assigned to the regime known at the previous close — the label you could have "
               "acted on. A regime where the CI sits clearly below zero is a candidate for a filter (Phase 2 "
               "regime gating) — test it, don't just switch it off.")

    c1, c2 = st.columns(2)
    with c1:
        tm = transition_matrix(lab, d.order)
        if not tm.empty:
            fig = go.Figure(go.Heatmap(z=tm.to_numpy(), x=list(tm.columns), y=list(tm.index), zmin=0, zmax=1,
                                       colorscale=up.SEQUENTIAL if hasattr(up, "SEQUENTIAL") else "Teal",
                                       text=tm.map(lambda v: f"{v * 100:.1f}%").to_numpy(), texttemplate="%{text}",
                                       textfont=dict(size=10), showscale=False,
                                       hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", title_text="today", showspikes=False)
            fig.update_xaxes(title_text="next bar", showspikes=False, side="top")
            fig.update_layout(hovermode="closest")
            up.show(fig, 300, "transition matrix · P(next | today)", key="bt_rg_tm")
            st.caption("High numbers on the diagonal = sticky regimes (good: a filter has time to act).")
    with c2:
        fig = go.Figure()
        for k, c in cmap.items():
            x = sp.loc[sp["label"] == k, "bars"]
            if len(x):
                fig.add_trace(go.Box(x=x, name=k, marker_color=c, boxpoints=False, orientation="h"))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text="spell length (bars)", type="log", showspikes=False)
        up.show(fig, 300, "how long each regime lasts", key="bt_rg_dur")

    with st.expander("CROSS TWO REGIMES — 2-D heatmap"):
        others = [k for k in DETECTORS if k != key and (spec.src == "Market" or not DETECTORS[k].needs)]
        k2 = st.selectbox("second detector", others, key="bt_rg_k2", format_func=lambda k: DETECTORS[k].label)
        lab2 = labels_for(spec, ins, k2, {}, mb)
        hm = two_way(r, lab, lab2, p)
        if hm.empty:
            st.caption("Not enough overlap.")
        else:
            cap = float(np.nanmax(np.abs(hm.to_numpy()))) if np.isfinite(hm.to_numpy()).any() else 1.0
            fig = go.Figure(go.Heatmap(z=hm.to_numpy(), x=[str(c) for c in hm.columns], y=[str(i) for i in hm.index],
                                       zmin=-cap, zmax=cap, colorscale=up.DIVERGING,
                                       text=hm.map(lambda v: "" if pd.isna(v) else f"{v:.2f}").to_numpy(),
                                       texttemplate="%{text}", showscale=False,
                                       hovertemplate="%{y} × %{x}: Sharpe %{z:.2f}<extra></extra>"))
            fig.update_xaxes(showspikes=False, title_text=DETECTORS[k2].label)
            fig.update_yaxes(showspikes=False, title_text=d.label)
            fig.update_layout(hovermode="closest")
            up.show(fig, 320, "sharpe by regime pair (cells with < 20 bars blank)", key="bt_rg_2d")


def section_header() -> None:
    st.markdown(section("REGIMES — how the strategy behaves in each market state", 3, help=G.REGIME_HELP),
                unsafe_allow_html=True)

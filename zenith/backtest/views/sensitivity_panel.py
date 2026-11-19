"""Reusable parameter-sensitivity panel for every BACKTEST tool.

A tool describes the knobs it can sweep (``Knob``) and a ``run(values) -> stats`` function;
this module does the rest: axis pickers, value editors, the grid run (parallel, capped), the
plateau analysis and the charts — 1-D line with neighbourhood mean (and 2x costs when the tool
supports it), HEATMAP, 3D SURFACE, GRADIENT and 2x COSTS for two knobs.

LAB keeps its own trial-ledger wiring but draws through ``render_results`` too, so every tool
reads the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from ..config import caps
from ..robust.runner import pmap
from ..robust.sensitivity import plateau, to_grid
from . import common as C


@dataclass
class Knob:
    """One sweepable setting. ``lo``/``hi`` bound numeric knobs; ``choices`` makes it categorical."""
    name: str
    label: str
    current: object
    lo: float | None = None
    hi: float | None = None
    integer: bool = False
    log: bool = False
    choices: list = field(default_factory=list)
    fmt: Callable[[object], str] = str

    def defaults(self, n: int) -> list:
        if self.choices:
            return list(self.choices)
        lo, hi = float(self.lo), float(self.hi)
        cur = float(self.current) if self.current is not None else (lo + hi) / 2
        a, b = max(lo, cur / 3 if cur > 0 else lo), min(hi, cur * 3 if cur > 0 else hi)
        if b - a < (hi - lo) * 0.2:
            a, b = lo, hi
        vals = np.geomspace(max(a, 1e-9), b, n) if self.log and a > 0 else np.linspace(a, b, n)
        if self.integer:
            return sorted({int(round(v)) for v in vals})
        return sorted({float(round(v, 4)) for v in vals})


def _editor(k: Knob, key: str, n: int) -> list:
    vals = k.defaults(n)
    if k.choices:
        pick = st.multiselect(f"{k.label} values", k.choices, default=vals, key=f"{key}_{k.name}",
                              format_func=k.fmt)
        return pick or vals
    txt = st.text_input(f"{k.label} values", ", ".join(str(v) for v in vals), key=f"{key}_{k.name}_{n}",
                        help="Comma-separated. Defaults to a spread around the current value.")
    out = []
    for t in txt.split(","):
        try:
            v = float(t)
        except ValueError:
            continue
        if k.lo is not None:
            v = min(max(v, float(k.lo)), float(k.hi))
        out.append(int(round(v)) if k.integer else v)
    return sorted(set(out)) or vals


def grid_run(run: Callable[[dict], dict], kx: Knob, xs: list, ky: Knob | None, ys: list | None,
             metrics: list[str], progress=None, **kw) -> pd.DataFrame:
    """Long table x[, y] + metrics. Failing cells become NaN holes, never crashes."""
    combos = [(x, y) for y in (ys or [None]) for x in xs]

    def one(c):
        x, y = c
        vals = {kx.name: x}
        if ky is not None:
            vals[ky.name] = y
        try:
            s = run(vals, **kw)
        except Exception:  # noqa: BLE001
            s = {}
        return {"x": x, "y": y, **{m: s.get(m, np.nan) for m in metrics}}

    return pd.DataFrame(pmap(one, combos, progress))


def _fmtv(metric: str, v) -> str:
    return C.fmt(metric, v) if metric in ("CAGR", "Vol", "MaxDD", "HitRate") else (
        "—" if v is None or not np.isfinite(v) else f"{v:,.2f}")


def render_results(df: pd.DataFrame, df2: pd.DataFrame | None, metric: str, kx_label: str,
                   ky_label: str | None, higher_is_better: bool, cur_x, key: str, log_x: bool = True,
                   cur_y=None) -> dict | None:
    """KPIs + charts for a finished sweep. Returns the plateau dict (or None)."""
    grid = to_grid(df, metric)
    pl = plateau(grid, higher_is_better=higher_is_better)
    if not pl.get("ok"):
        C.banner(T.coral, "NO RESULT", "every cell failed — check the ranges")
        return None
    rob = pl["robustness"]
    tone = T.teal if rob >= 0.75 else T.amber if rob >= 0.5 else T.coral
    kp = [
        {"label": "PEAK", "value": _fmtv(metric, pl["peak"]["value"]),
         "sub": f"{kx_label}={pl['peak']['x']}" + (f" · {ky_label}={pl['peak']['y']}" if ky_label else "")},
        {"label": "PLATEAU PICK", "value": _fmtv(metric, pl["plateau"]["value"]), "color": T.teal,
         "sub": f"{kx_label}={pl['plateau']['x']}" + (f" · {ky_label}={pl['plateau']['y']}" if ky_label else "")},
        {"label": "ROBUSTNESS", "value": f"{rob:.2f}", "color": tone, "sub": "neighbourhood ÷ peak · 1 = flat top"},
    ]
    if "Sharpe" in df:
        sp = float((df["Sharpe"] > 0).mean())
        kp.append({"label": "CELLS WITH SHARPE > 0", "value": f"{sp * 100:.0f}%",
                   "color": T.teal if sp > 0.7 else T.amber if sp > 0.4 else T.coral, "sub": "of the grid"})
    st.markdown(uk.kpis(kp), unsafe_allow_html=True)
    if rob < 0.5:
        C.banner(T.coral, "SPIKE", "the best cell stands alone; its neighbours are much worse — classic overfit shape")
    pick = pl["plateau"]
    on_edge = pick["x"] in (grid.columns[0], grid.columns[-1]) or (
        ky_label and pick["y"] in (grid.index[0], grid.index[-1]))
    if on_edge and grid.size > 2:
        C.banner(T.amber, "EDGE", "the plateau sits on the edge of the grid — the best region may lie beyond it; "
                 "widen the range before trusting the pick")

    numeric_x = all(isinstance(v, (int, float, np.integer, np.floating)) for v in grid.columns)
    if not ky_label:
        s = grid.iloc[0]
        sm = pl["smoothed"].iloc[0]
        fig = go.Figure()
        if df2 is not None:
            s2 = to_grid(df2, metric).iloc[0]
            fig.add_trace(go.Scatter(x=s2.index, y=s2.values, name="2× costs",
                                     line=dict(color=T.coral, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=metric, mode="lines+markers",
                                 line=dict(color=T.teal, width=1.6), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=sm.index, y=sm.values, name="neighbourhood mean",
                                 line=dict(color=T.violet, width=2)))
        fig.add_trace(go.Scatter(x=[pl["peak"]["x"]], y=[pl["peak"]["value"]], mode="markers", name="peak",
                                 marker=dict(symbol="x", size=11, color=T.amber)))
        fig.add_trace(go.Scatter(x=[pick["x"]], y=[s.get(pick["x"])], mode="markers", name="plateau pick",
                                 marker=dict(symbol="diamond", size=12, color=T.teal,
                                             line=dict(color=T.text, width=1))))
        if cur_x is not None and numeric_x:
            fig.add_vline(x=cur_x, line=dict(color=T.faint, dash="dash", width=1),
                          annotation_text="current", annotation_font=dict(size=9, color=T.muted))
        fig.add_hline(y=0, line=dict(color=T.line))
        lx = log_x and numeric_x and min(grid.columns) > 0
        fig.update_xaxes(type="log" if lx else ("linear" if numeric_x else "category"), title_text=kx_label)
        fig.update_yaxes(title_text=metric)
        fig.update_layout(hovermode="x")
        up.show(fig, 380, f"{metric} across {kx_label}", key=f"{key}_1d", scale_toggle=False)
        return pl

    tabs = st.tabs(["HEATMAP", "3D SURFACE", "GRADIENT", "2× COSTS"])
    z = grid.to_numpy(float)
    xl, yl = [str(v) for v in grid.columns], [str(v) for v in grid.index]
    zf = z[np.isfinite(z)]
    if higher_is_better and zf.size and zf.min() >= 0:
        zmin, zmax, zmid, scale = float(zf.min()), float(zf.max()), None, up.SEQUENTIAL
    else:
        zmax = float(np.nanmax(np.abs(z))) or 1.0
        zmin, zmid, scale = -zmax, 0.0, up.DIVERGING
    with tabs[0]:
        reg = pl["region"].to_numpy()
        hm = dict(z=z, x=xl, y=yl, zmin=zmin, zmax=zmax, colorscale=scale,
                  text=np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")(z),
                  texttemplate="%{text}", textfont=dict(size=9), xgap=1, ygap=1,
                  colorbar=dict(title=dict(text=metric, font=dict(size=10)), thickness=10))
        if zmid is not None:
            hm["zmid"] = zmid
        fig = go.Figure(go.Heatmap(**hm))
        ri, rj = np.where(reg)
        fig.add_trace(go.Scatter(x=[xl[j] for j in rj], y=[yl[i] for i in ri], mode="markers", name="plateau region",
                                 marker=dict(symbol="square-open", size=22, color=T.text, line=dict(width=1))))
        for p, sym, col, nm in ((pl["peak"], "x", T.amber, "peak"), (pick, "diamond", T.teal, "plateau pick")):
            fig.add_trace(go.Scatter(x=[str(p["x"])], y=[str(p["y"])], mode="markers", name=nm,
                                     marker=dict(symbol=sym, size=14, color=col, line=dict(color=T.bg, width=1))))
        if cur_x is not None and cur_y is not None and str(cur_x) in xl and str(cur_y) in yl:
            fig.add_trace(go.Scatter(x=[str(cur_x)], y=[str(cur_y)], mode="markers", name="current",
                                     marker=dict(symbol="circle-open", size=16, color=T.amber, line=dict(width=2))))
        fig.update_xaxes(title_text=kx_label, type="category", showspikes=False)
        fig.update_yaxes(title_text=ky_label, type="category", showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 460, f"{metric} · outlined = plateau region", key=f"{key}_2d")
    with tabs[1]:
        fig = go.Figure(go.Surface(z=z, x=list(range(len(xl))), y=list(range(len(yl))),
                                   colorscale=scale if higher_is_better else scale[::-1],
                                   contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                                   showscale=False,
                                   hovertemplate=f"{kx_label} %{{x}} · {ky_label} %{{y}} · {metric} %{{z:.2f}}"
                                                 "<extra></extra>"))
        fig.update_layout(scene=dict(xaxis=dict(title=kx_label, tickvals=list(range(len(xl))), ticktext=xl),
                                     yaxis=dict(title=ky_label, tickvals=list(range(len(yl))), ticktext=yl),
                                     zaxis=dict(title=metric), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
                          margin=dict(l=0, r=0, t=10, b=0))
        up.show(fig, 560, f"{metric} surface — drag to rotate", key=f"{key}_3d")
    with tabs[2]:
        g = pl["gradient"].to_numpy(float)
        fig = go.Figure(go.Heatmap(z=g, x=xl, y=yl, colorscale=up.SEQUENTIAL, xgap=1, ygap=1,
                                   colorbar=dict(thickness=10)))
        fig.update_xaxes(title_text=kx_label, type="category", showspikes=False)
        fig.update_yaxes(title_text=ky_label, type="category", showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 420, "gradient magnitude — bright = the result changes fast here", key=f"{key}_grad")
        st.caption("Pick settings in a dark, flat area: small changes there don't change the answer.")
    with tabs[3]:
        if df2 is None:
            st.caption("Turn on '2× costs' and re-run." if key else "")
        else:
            z2 = to_grid(df2, metric).to_numpy(float)
            hm2 = dict(z=z2, x=xl, y=yl, zmin=zmin, zmax=zmax, colorscale=scale, xgap=1, ygap=1,
                       colorbar=dict(thickness=10))
            if zmid is not None:
                hm2["zmid"] = zmid
            fig = go.Figure(go.Heatmap(**hm2))
            fig.update_xaxes(title_text=kx_label, type="category", showspikes=False)
            fig.update_yaxes(title_text=ky_label, type="category", showspikes=False)
            fig.update_layout(hovermode="closest")
            up.show(fig, 420, f"{metric} at double costs", key=f"{key}_2c")
    return pl


def panel(knobs: list[Knob], run: Callable[..., dict], key: str, metrics: list[str],
          lower_better: tuple = ("Turnover", "Vol", "AnnVol%", "TotalCost%"), state: str = "",
          supports_costs: bool = False, apply: Callable[[dict], None] | None = None,
          title: str = "SENSITIVITY — look for plateaus, not peaks", note: str = "",
          higher_is_better: dict | None = None) -> None:
    """Full sensitivity UI for one tool. ``state`` is a fingerprint of everything else the run
    depends on (so results are marked stale when the strategy changes)."""
    st.markdown(section(title, 1, help="Sweep one or two settings and look at the SHAPE. A broad high "
                        "plateau is evidence of a real effect; an isolated spike is fitted noise."),
                unsafe_allow_html=True)
    if note:
        st.caption(note)
    by = {k.name: k for k in knobs}
    names = list(by)
    c1, c2, c3, c4 = st.columns(4)
    nx = c1.selectbox("x", names, format_func=lambda n: by[n].label, key=f"{key}_x")
    ny = c2.selectbox("y", ["—"] + [n for n in names if n != nx], key=f"{key}_y",
                      format_func=lambda n: n if n == "—" else by[n].label)
    ky = None if ny == "—" else by[ny]
    kx = by[nx]
    n = c3.slider("points per axis", 3, caps().grid_side, 7 if ky else 11, key=f"{key}_n")
    metric = c4.selectbox("metric", metrics, key=f"{key}_m")
    xs = _editor(kx, f"{key}_xv", n)
    ys = _editor(ky, f"{key}_yv", n) if ky else None
    cost2 = st.toggle("also run at 2× costs", value=False, key=f"{key}_c2") if supports_costs else False
    n_runs = len(xs) * (len(ys) if ys else 1) * (2 if cost2 else 1)
    mk = f"{key}|{state}|{nx}|{ny}|{xs}|{ys}|{cost2}"
    if st.button(f"RUN SWEEP · {n_runs} backtests", type="primary", key=f"{key}_go"):
        df = grid_run(run, kx, xs, ky, ys, metrics, progress=C.progress_bar("sweeping"))
        df2 = grid_run(run, kx, xs, ky, ys, metrics, cost_mult=2.0) if cost2 else None
        C.memo()[mk] = {"df": df, "df2": df2}
    res = C.memo().get(mk)
    if res is None:
        st.caption("Not run yet for these settings.")
        return
    # drawdowns are stored negative, so "higher" (closer to 0) is already better
    hib = (higher_is_better or {}).get(metric, metric not in lower_better)
    pl = render_results(res["df"], res["df2"], metric, kx.label, ky.label if ky else None, hib,
                        kx.current, key, log_x=kx.log, cur_y=ky.current if ky else None)
    if pl and apply is not None:
        pick = pl["plateau"]
        vals = {kx.name: pick["x"]} | ({ky.name: pick["y"]} if ky else {})
        if st.button("USE PLATEAU CENTRE (" + ", ".join(f"{by[k].label}={v}" for k, v in vals.items()) + ")",
                     key=f"{key}_apply"):
            apply(vals)
    with st.expander("grid table"):
        st.dataframe(res["df"], use_container_width=True, hide_index=True)

"""LAB steps 03–05: SENSITIVITY, ROBUSTNESS, FILTERS.

Heavy jobs run on a button and are memoised per spec hash + settings, so
moving around the page never re-runs them; changing the strategy marks a
result stale instead of silently showing numbers for a different spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import ledger as lg
from ..config import caps
from ..engine import metrics
from ..engine.montecarlo import bootstrap, bootstrap_paths
from ..robust.ablation import ablation
from ..robust.sensitivity import METRICS, default_values, plateau, sweep, to_grid
from ..robust.shuffle import SYNTH_KINDS, noise_curve, permutation_test, synthetic_suite
from ..robust.walkforward import walk_forward
from ..spec import StrategySpec, run_spec, split_holdout
from . import common as C

LOWER_BETTER = {"MaxDD": False, "Turnover": False}   # MaxDD is negative: higher (closer to 0) is better


def _param_choices(spec: StrategySpec) -> list[str]:
    return list(spec.strategy().__class__.params)


def _values_editor(spec: StrategySpec, param: str, key: str, n: int) -> list:
    vals = default_values(spec, param, n)
    txt = st.text_input(f"{param} values", ", ".join(str(v) for v in vals), key=f"{key}_{param}_{n}",
                        help="Comma-separated. Defaults to a geometric spread from ¼× to 4× the current value.")
    out = []
    for t in txt.split(","):
        try:
            v = float(t)
            out.append(int(v) if float(v).is_integer() and isinstance(vals[0], int) else v)
        except ValueError:
            continue
    return sorted(set(out)) or vals


def _stale_or(key: str, label: str):
    res = C.memo().get(key)
    if res is None:
        st.caption(f"Not run yet for this exact strategy. {label}")
    return res


# =================================================================== SENSITIVITY
def sensitivity(spec: StrategySpec, ins: pd.DataFrame, aux: dict) -> None:
    st.markdown(section("PARAMETER SENSITIVITY — look for plateaus, not peaks", 1), unsafe_allow_html=True)
    params = _param_choices(spec)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    px = c1.selectbox("x parameter", params, index=params.index(spec.strategy().spectrum_param)
                      if spec.strategy().spectrum_param in params else 0, key="bt_sx")
    py = c2.selectbox("y parameter", ["—"] + [p for p in params if p != px], key="bt_sy")
    py = None if py == "—" else py
    n = c3.slider("points per axis", 3, caps().grid_side, 9 if py else 15, key="bt_sn")
    metric = c4.selectbox("metric", METRICS, key="bt_sm")
    xs = _values_editor(spec, px, "bt_sxv", n)
    ys = _values_editor(spec, py, "bt_syv", n) if py else None
    cost2 = st.toggle("also run at 2× costs", value=True, key="bt_s2c")
    n_runs = len(xs) * (len(ys) if ys else 1) * (2 if cost2 else 1)
    key = f"sens|{spec.hash()}|{px}|{py}|{xs}|{ys}|{cost2}"

    if st.button(f"RUN SWEEP · {n_runs} backtests", type="primary", key="bt_srun"):
        df, hashes = sweep(spec, ins, px, xs, py, ys, aux=aux, progress=C.progress_bar("sweeping"))
        df2 = sweep(spec, ins, px, xs, py, ys, aux=aux, cost_mult=2.0)[0] if cost2 else None
        lg.record(C.ledger(), spec.family(), hashes)
        C.memo()[key] = {"df": df, "df2": df2}
    res = _stale_or(key, "Each cell is one trial and is added to the trial counter.")
    if res is None:
        return

    df, df2 = res["df"], res["df2"]
    hib = LOWER_BETTER.get(metric, True)
    grid = to_grid(df, metric)
    pl = plateau(grid, higher_is_better=hib)
    if not pl.get("ok"):
        C.banner(T.coral, "NO RESULT", "every cell failed — check the parameter ranges")
        return
    cur_x = spec.params.get(px)
    rob = pl["robustness"]
    tone = T.teal if rob >= 0.75 else T.amber if rob >= 0.5 else T.coral
    sharpe_pos = float((df["Sharpe"] > 0).mean())
    st.markdown(uk.kpis([
        {"label": "PEAK", "value": C.fmt(metric, pl["peak"]["value"]),
         "sub": f"{px}={pl['peak']['x']}" + (f" · {py}={pl['peak']['y']}" if py else "")},
        {"label": "PLATEAU PICK", "value": C.fmt(metric, pl["plateau"]["value"]), "color": T.teal,
         "sub": f"{px}={pl['plateau']['x']}" + (f" · {py}={pl['plateau']['y']}" if py else "")},
        {"label": "ROBUSTNESS", "value": f"{rob:.2f}", "color": tone,
         "sub": "neighbourhood ÷ peak · 1 = flat top"},
        {"label": "CELLS WITH SHARPE > 0", "value": f"{sharpe_pos * 100:.0f}%",
         "color": T.teal if sharpe_pos > 0.7 else T.amber if sharpe_pos > 0.4 else T.coral, "sub": "of the grid"},
    ]), unsafe_allow_html=True)
    if rob < 0.5:
        C.banner(T.coral, "SPIKE", "the best cell stands alone; its neighbours are much worse — classic overfit shape")
    elif sharpe_pos < 0.5:
        C.banner(T.amber, "THIN", "most of the grid loses money; the edge depends on the setting")

    pick = pl["plateau"]
    edge = pick["x"] in (grid.columns[0], grid.columns[-1]) or (py and pick["y"] in (grid.index[0], grid.index[-1]))
    if edge:
        C.banner(T.amber, "EDGE", "the plateau sits on the edge of the grid — the best region may lie beyond it; "
                 "widen the range before trusting the pick")
    if st.button(f"USE PLATEAU CENTRE ({px}={pick['x']}" + (f", {py}={pick['y']})" if py else ")"),
                 key="bt_use_plat"):
        upd = {C.param_key(spec.rule, px): pick["x"]}
        if py:
            upd[C.param_key(spec.rule, py)] = pick["y"]
        C.queue(**upd)

    if not py:
        s = grid.iloc[0]
        sm = pl["smoothed"].iloc[0]
        fig = go.Figure()
        if df2 is not None:
            s2 = to_grid(df2, metric).iloc[0]
            fig.add_trace(go.Scatter(x=s2.index, y=s2.values, name="2× costs", line=dict(color=T.coral, width=1,
                                                                                            dash="dot")))
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=metric, mode="lines+markers",
                                 line=dict(color=T.teal, width=1.6), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=sm.index, y=sm.values, name="neighbourhood mean",
                                 line=dict(color=T.violet, width=2)))
        fig.add_trace(go.Scatter(x=[pl["peak"]["x"]], y=[pl["peak"]["value"]], mode="markers", name="peak",
                                 marker=dict(symbol="x", size=11, color=T.amber)))
        fig.add_trace(go.Scatter(x=[pick["x"]], y=[s.get(pick["x"])], mode="markers", name="plateau pick",
                                 marker=dict(symbol="diamond", size=12, color=T.teal, line=dict(color=T.text,
                                                                                               width=1))))
        if cur_x is not None:
            fig.add_vline(x=cur_x, line=dict(color=T.faint, dash="dash", width=1),
                          annotation_text="current", annotation_font=dict(size=9, color=T.muted))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.update_xaxes(type="log", title_text=px)
        fig.update_yaxes(title_text=metric)
        fig.update_layout(hovermode="x")
        up.show(fig, 380, f"{metric} across {px}", key="bt_s1d")
        return

    tabs = st.tabs(["HEATMAP", "3D SURFACE", "GRADIENT", "2× COSTS"])
    z = grid.to_numpy(float)
    xl, yl = [str(v) for v in grid.columns], [str(v) for v in grid.index]
    zmax = float(np.nanmax(np.abs(z))) or 1.0
    scale = up.DIVERGING if hib else up.DIVERGING[::-1]
    with tabs[0]:
        reg = pl["region"].to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=xl, y=yl, zmid=0, zmin=-zmax, zmax=zmax, colorscale=up.DIVERGING,
                                   text=np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")(z),
                                   texttemplate="%{text}", textfont=dict(size=9), xgap=1, ygap=1,
                                   colorbar=dict(title=dict(text=metric, font=dict(size=10)), thickness=10)))
        ri, rj = np.where(reg)
        fig.add_trace(go.Scatter(x=[xl[j] for j in rj], y=[yl[i] for i in ri], mode="markers", name="plateau region",
                                 marker=dict(symbol="square-open", size=22, color=T.text, line=dict(width=1))))
        for p, sym, col, nm in ((pl["peak"], "x", T.amber, "peak"), (pick, "diamond", T.teal, "plateau pick")):
            fig.add_trace(go.Scatter(x=[str(p["x"])], y=[str(p["y"])], mode="markers", name=nm,
                                     marker=dict(symbol=sym, size=14, color=col, line=dict(color=T.bg, width=1))))
        fig.update_xaxes(title_text=px, type="category", showspikes=False)
        fig.update_yaxes(title_text=py, type="category", showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 460, f"{metric} · outlined = plateau region", key="bt_s2d")
    with tabs[1]:
        fig = go.Figure(go.Surface(z=z, x=list(range(len(xl))), y=list(range(len(yl))), colorscale=scale,
                                   cmid=0, contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                                   showscale=False, hovertemplate=f"{px} %{{x}} · {py} %{{y}} · {metric} %{{z:.2f}}"
                                                                  "<extra></extra>"))
        fig.update_layout(scene=dict(xaxis=dict(title=px, tickvals=list(range(len(xl))), ticktext=xl),
                                     yaxis=dict(title=py, tickvals=list(range(len(yl))), ticktext=yl),
                                     zaxis=dict(title=metric), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
                          margin=dict(l=0, r=0, t=10, b=0))
        up.show(fig, 560, f"{metric} surface — drag to rotate", key="bt_s3d")
    with tabs[2]:
        g = pl["gradient"].to_numpy(float)
        fig = go.Figure(go.Heatmap(z=g, x=xl, y=yl, colorscale=up.SEQUENTIAL, xgap=1, ygap=1,
                                   colorbar=dict(thickness=10)))
        fig.update_xaxes(title_text=px, type="category", showspikes=False)
        fig.update_yaxes(title_text=py, type="category", showspikes=False)
        fig.update_layout(hovermode="closest")
        up.show(fig, 420, "gradient magnitude — bright = the result changes fast here", key="bt_sgrad")
        st.caption("Pick parameters in a dark, flat area: small changes there don't change the answer.")
    with tabs[3]:
        if df2 is None:
            st.caption("Turn on '2× costs' and re-run.")
        else:
            z2 = to_grid(df2, metric).to_numpy(float)
            fig = go.Figure(go.Heatmap(z=z2, x=xl, y=yl, zmid=0, zmin=-zmax, zmax=zmax, colorscale=up.DIVERGING,
                                       xgap=1, ygap=1, colorbar=dict(thickness=10)))
            fig.update_xaxes(title_text=px, type="category", showspikes=False)
            fig.update_yaxes(title_text=py, type="category", showspikes=False)
            fig.update_layout(hovermode="closest")
            up.show(fig, 420, f"{metric} at double costs", key="bt_s2c_map")


# =================================================================== ROBUSTNESS
ROBUST = ["PERMUTATION", "NOISE", "SYNTHETIC", "WALK-FORWARD", "MONTE CARLO", "OTHER INSTRUMENTS"]


def robustness(spec: StrategySpec, ins: pd.DataFrame, aux: dict, base) -> None:
    sub = st.segmented_control("test", ROBUST, default="PERMUTATION", key="bt_rsub",
                               label_visibility="collapsed") or "PERMUTATION"
    {"PERMUTATION": _perm, "NOISE": _noise, "SYNTHETIC": _synth, "WALK-FORWARD": _wf,
     "MONTE CARLO": _mc, "OTHER INSTRUMENTS": _others}[sub](spec, ins, aux, base)


def _perm(spec, ins, aux, base):
    st.markdown(section("PERMUTATION — does it beat markets with the time structure shuffled out?", 3),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n = c1.slider("shuffles", 20, caps().permutations, min(100, caps().permutations), 10, key="bt_pn")
    mode = c2.radio("compare", ["fixed parameters", "re-optimise every shuffle"], key="bt_pm", horizontal=True,
                    help="Re-optimising is the honest test when you tuned the parameter: luck gets to tune too.")
    grid = None
    if mode.startswith("re-opt"):
        params = _param_choices(spec)
        p = st.selectbox("parameter tuned", params, key="bt_pp")
        grid = (p, _values_editor(spec, p, "bt_ppv", 6))
    runs = n * (len(grid[1]) if grid else 1)
    key = f"perm|{spec.hash()}|{n}|{grid}"
    if st.button(f"RUN · {runs} backtests", type="primary", key="bt_prun"):
        C.memo()[key] = permutation_test(spec, ins, n=n, grid=grid, aux=aux, progress=C.progress_bar("shuffling"))
    res = _stale_or(key, "Masters' bar permutation: same bars, random order.")
    if res is None:
        return
    p = res["p"]
    tone, lbl = (T.teal, "PASS") if p < 0.01 else (T.amber, "MARGINAL") if p < 0.05 else (T.coral, "FAIL")
    C.banner(tone, lbl, f"p = {p:.3f} · real Sharpe {res['real']:.2f} beat {np.mean(res['perms'] < res['real']) * 100:.0f}% "
                        f"of {res['n']} shuffles" + (" (each re-optimised)" if res["optimised"] else ""))
    fig = go.Figure(go.Histogram(x=res["perms"], nbinsx=40, marker_color=up.fill(T.violet, 0.6),
                                 marker_line=dict(width=0), name="shuffled"))
    fig.add_vline(x=res["real"], line=dict(color=T.teal, width=2), annotation_text="REAL",
                  annotation_font=dict(color=T.teal, size=10))
    fig.update_xaxes(title_text="best Sharpe" if res["optimised"] else "Sharpe", showspikes=False)
    fig.update_layout(hovermode="closest", showlegend=False)
    up.show(fig, 340, "sharpe on shuffled markets vs the real one", key="bt_phist")


def _noise(spec, ins, aux, base):
    st.markdown(section("NOISE — how fast does the edge decay when prices are jittered?", 3), unsafe_allow_html=True)
    seeds = st.slider("seeds per level", 4, 40, 12, key="bt_ns")
    key = f"noise|{spec.hash()}|{seeds}"
    if st.button(f"RUN · {5 * seeds + 1} backtests", type="primary", key="bt_nrun"):
        C.memo()[key] = noise_curve(spec, ins, seeds=seeds, aux=aux, progress=C.progress_bar("adding noise"))
    df = _stale_or(key, "Each level adds Gaussian noise of k × the bar return σ.")
    if df is None:
        return
    g = df.groupby("k")["Sharpe"]
    med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(hi.index) + list(lo.index[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                             fillcolor=up.fill(T.teal, 0.15), line=dict(width=0), name="10–90%"))
    fig.add_trace(go.Scatter(x=med.index, y=med, mode="lines+markers", name="median", line=dict(color=T.teal)))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_xaxes(title_text="noise k (× bar σ)")
    fig.update_yaxes(title_text="Sharpe")
    up.show(fig, 340, "sharpe vs noise", key="bt_ncurve")
    keep = med.get(0.25, np.nan) / med.get(0.0, np.nan) if med.get(0.0, 0) > 0 else np.nan
    if np.isfinite(keep):
        C.banner(T.teal if keep > 0.6 else T.amber if keep > 0.3 else T.coral, f"{keep * 100:.0f}% KEPT",
                 "share of the Sharpe left at k = 0.25; a fragile rule loses most of it")


def _synth(spec, ins, aux, base):
    st.markdown(section("SYNTHETIC MARKETS — known answers", 3), unsafe_allow_html=True)
    seeds = st.slider("paths per market", 3, 20, 6, key="bt_ys")
    key = f"synth|{spec.hash()}|{seeds}"
    if st.button(f"RUN · {len(SYNTH_KINDS) * seeds} backtests", type="primary", key="bt_yrun"):
        C.memo()[key] = synthetic_suite(spec, seeds=seeds, progress=C.progress_bar("simulating"))
    df = _stale_or(key, "A random walk ('gbm') has no edge to find: a rule that 'wins' there is fitting noise.")
    if df is None:
        return
    fig = go.Figure()
    for i, k in enumerate(SYNTH_KINDS):
        s = df[df.market == k]["Sharpe"]
        fig.add_trace(go.Box(y=s, name=k, marker_color=T.cat[i % len(T.cat)], boxpoints="all", jitter=0.4,
                             pointpos=0, line=dict(width=1)))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_layout(showlegend=False, hovermode="closest")
    up.show(fig, 360, "sharpe by market type", key="bt_ybox")
    rw = df[df.market == "gbm"]["Sharpe"].median()
    C.banner(T.teal if abs(rw) < 0.3 else T.coral, "RANDOM WALK", f"median Sharpe {rw:.2f} on pure noise "
             "(should be near 0)")


def _wf(spec, ins, aux, base):
    st.markdown(section("WALK-FORWARD — re-tune on the past, trade the next slice", 3), unsafe_allow_html=True)
    params = _param_choices(spec)
    c1, c2 = st.columns(2)
    p = c1.selectbox("parameter to re-tune", params, key="bt_wp")
    folds = c2.slider("folds", 3, 8, 5, key="bt_wf")
    vals = _values_editor(spec, p, "bt_wv", 7)
    key = f"wf|{spec.hash()}|{p}|{folds}|{vals}"
    if st.button(f"RUN · ≈{(folds + 1) * len(vals) + folds} backtests", type="primary", key="bt_wrun"):
        out = walk_forward(spec, ins, p, vals, n_folds=folds, aux=aux, progress=C.progress_bar("walking forward"))
        lg.record(C.ledger(), spec.family(), out["hashes"])
        C.memo()[key] = out
    res = _stale_or(key, "Each fold picks the plateau centre on its training data only.")
    if res is None:
        return
    eff = res["efficiency"]
    st.markdown(uk.kpis([
        {"label": "WALK-FORWARD SHARPE", "value": f"{res['wf_sharpe']:.2f}", "color": T.teal},
        {"label": "BEST IN-SAMPLE SHARPE", "value": f"{res['is_best_sharpe']:.2f}", "sub": "hindsight pick"},
        {"label": "EFFICIENCY", "value": "—" if not np.isfinite(eff) else f"{eff * 100:.0f}%",
         "color": T.teal if np.isfinite(eff) and eff > 0.6 else T.amber if np.isfinite(eff) and eff > 0.3 else T.coral,
         "sub": "walk-forward ÷ hindsight"},
    ]), unsafe_allow_html=True)
    oos = res["oos"]
    eq = (1 + oos).cumprod()
    fig = go.Figure(go.Scatter(x=eq.index, y=eq, name="stitched walk-forward", line=dict(color=T.teal, width=1.6)))
    fig.update_yaxes(type="log", title_text="growth of 1")
    up.show(fig, 300, "stitched out-of-fold equity", key="bt_wfeq")
    st.dataframe(res["folds"], use_container_width=True, hide_index=True)


def _mc(spec, ins, aux, base):
    st.markdown(section("MONTE CARLO — what else could this return stream have looked like?", 3),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n = c1.slider("bootstrap paths", 100, caps().mc_paths, min(1000, caps().mc_paths), 100, key="bt_mn")
    block = c2.slider("block length (bars)", 5, 60, 20, key="bt_mb",
                      help="Blocks keep short-run autocorrelation (streaks, vol clusters) intact.")
    r = base.returns
    boot = bootstrap(r, n_sims=n, block=block, periods=base.periods)
    paths = bootstrap_paths(r, n_sims=min(n, 400), block=block)
    q = paths.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    x = r.index[: len(q)]
    fig = go.Figure()
    for lo, hi, a in ((0.05, 0.95, 0.12), (0.25, 0.75, 0.22)):
        fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(q[hi]) + list(q[lo][::-1]), fill="toself",
                                 fillcolor=up.fill(T.teal, a), line=dict(width=0),
                                 name=f"{int(lo * 100)}–{int(hi * 100)}%"))
    fig.add_trace(go.Scatter(x=x, y=q[0.5], name="median", line=dict(color=T.teal, width=1)))
    fig.add_trace(go.Scatter(x=x, y=(1 + r).cumprod().iloc[: len(q)].to_numpy(), name="actual",
                             line=dict(color=T.amber, width=1.4)))
    fig.update_yaxes(type="log", title_text="growth of 1")
    up.show(fig, 360, "bootstrapped equity fan", key="bt_mcfan")
    st.markdown(uk.kpis([
        {"label": "P(SHARPE < 0)", "value": f"{(boot['Sharpe'] < 0).mean() * 100:.0f}%",
         "color": T.coral if (boot['Sharpe'] < 0).mean() > 0.1 else T.teal},
        {"label": "SHARPE 5–95%", "value": f"{boot['Sharpe'].quantile(.05):.2f} … {boot['Sharpe'].quantile(.95):.2f}"},
        {"label": "MAX DD · WORST 5%", "value": f"{boot['MaxDD'].quantile(.05) * 100:.0f}%", "color": T.coral,
         "sub": f"actual {base.stats['MaxDD'] * 100:.0f}%"},
    ]), unsafe_allow_html=True)


def _others(spec, ins, aux, base):
    st.markdown(section("OTHER INSTRUMENTS — same rule, markets it was never tuned on", 3), unsafe_allow_html=True)
    if spec.src != "Market":
        st.caption("Switch the universe to Market data to use this test.")
        return
    pool = [t for t in C.QUICK if t != spec.name]
    pick = st.multiselect("instruments", pool, default=[t for t in ("QQQ", "IWM", "EFA", "TLT", "GLD") if t in pool],
                          max_selections=caps().tickers, key="bt_oi")
    key = f"oth|{spec.hash()}|{pick}"
    if st.button(f"RUN · {len(pick)} backtests", type="primary", key="bt_orun", disabled=not pick):
        rows, prog = [], C.progress_bar("loading")
        for i, t in enumerate(pick):
            try:
                df, _ = split_holdout(C.load(spec, t), spec.holdout)      # their OOS stays locked too
                res = run_spec(spec, df, aux)
                bh = C.bench(df["close"], res.returns)
                rows.append({"instrument": t, "Sharpe": res.stats["Sharpe"],
                             "b&h Sharpe": metrics.sharpe(bh, res.periods), "CAGR": res.stats["CAGR"],
                             "MaxDD": res.stats["MaxDD"]})
            except Exception as exc:  # noqa: BLE001
                rows.append({"instrument": t, "Sharpe": np.nan, "error": str(exc)[:60]})
            prog((i + 1) / len(pick))
        C.memo()[key] = pd.DataFrame(rows)
    df = _stale_or(key, "Each instrument is cut at the same holdout share, so their recent data stays unseen too.")
    if df is None or df.empty:
        return
    df = pd.concat([pd.DataFrame([{"instrument": f"{spec.name} (tuned)", "Sharpe": base.stats["Sharpe"]}]), df])
    fig = go.Figure(go.Bar(x=df["instrument"], y=df["Sharpe"],
                           marker_color=[T.violet] + [T.teal if v > 0 else T.coral for v in df["Sharpe"].iloc[1:]]))
    if "b&h Sharpe" in df:
        fig.add_trace(go.Scatter(x=df["instrument"], y=df["b&h Sharpe"], mode="markers", name="buy & hold",
                                 marker=dict(symbol="line-ew-open", size=22, color=T.amber, line=dict(width=2))))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_layout(hovermode="closest")
    up.show(fig, 340, "sharpe by instrument · amber tick = buy & hold", key="bt_obar")
    share = float((df["Sharpe"].iloc[1:] > 0).mean())
    C.banner(T.teal if share >= 0.7 else T.amber if share >= 0.4 else T.coral, f"{share * 100:.0f}% POSITIVE",
             "of the untuned instruments")


# =================================================================== FILTERS
def filters(spec: StrategySpec, ins: pd.DataFrame, aux: dict) -> None:
    st.markdown(section("FILTER ABLATION — does each filter earn its place?", 5), unsafe_allow_html=True)
    if not spec.filters:
        C.banner(T.amber, "NO FILTERS", "turn on one or more filters in 03 · FILTERS to test them")
        return
    key = f"abl|{spec.hash()}"
    n = 1 + len(spec.filters) + (1 + len(spec.filters) if len(spec.filters) > 1 else 0)
    if st.button(f"RUN · {n} variants", type="primary", key="bt_arun"):
        tbl, hashes = ablation(spec, ins, aux, progress=C.progress_bar("ablating"))
        lg.record(C.ledger(), spec.family(), hashes)
        C.memo()[key] = tbl
    tbl = _stale_or(key, "No filters, each alone, all together, and all minus each one — every variant counts as a trial.")
    if tbl is None:
        return
    t = tbl.iloc[1:]
    col = {"adds value": T.teal, "hurts": T.coral, "no clear value": T.amber}
    fig = go.Figure(go.Bar(
        x=t["ΔSharpe"], y=t["variant"], orientation="h", marker_color=[col.get(v, T.muted) for v in t["verdict"]],
        error_x=dict(type="data", symmetric=False, array=(t["CI hi"] - t["ΔSharpe"]).fillna(0),
                     arrayminus=(t["ΔSharpe"] - t["CI lo"]).fillna(0), color=T.muted, thickness=1, width=4)))
    fig.add_vline(x=0, line=dict(color=T.text, width=1))
    fig.update_yaxes(autorange="reversed", showspikes=False)
    fig.update_xaxes(title_text="Sharpe vs no filters (90% CI)")
    fig.update_layout(hovermode="closest")
    up.show(fig, 60 + 34 * len(t), "Sharpe change per filter set", key="bt_abar")
    show = tbl.copy()
    for k in ("Sharpe", "ΔSharpe", "CI lo", "CI hi"):
        show[k] = show[k].map(lambda v: C.fmt("x", v))
    for k in ("CAGR", "MaxDD", "exposure"):
        show[k] = show[k].map(lambda v: C.fmt(k, v))
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("A filter that raises Sharpe only by cutting exposure in a few lucky spots shows a wide interval "
               "spanning zero — keep it only if you have a reason it should work.")

"""ROBUSTNESS add-ons (Phase 3): OVERFITTING (PBO · CPCV · Reality Check),
PROP CHALLENGE (target/stop Monte Carlo), STRESS (crises + shocks), plus
extras for MONTE CARLO (auto block length, trade-order shuffle) and
SENSITIVITY (distribution of all combinations, top-N clusters)."""

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
from .. import ledger as lg
from ..config import caps
from ..engine.montecarlo import auto_block_length, challenge_sim, trade_returns, trade_shuffle_drawdowns
from ..robust.overfit import cpcv, n_cpcv_paths, pbo, reality_check, top_clusters, variant_returns
from ..robust.sensitivity import default_values, knob_choices, knob_label
from ..robust.stress import crisis_table, scenario_table
from . import common as C


# =============================================================== OVERFITTING
def overfit(spec, ins: pd.DataFrame, aux: dict, base) -> None:
    st.markdown(section("OVERFITTING — how likely is it that the best variant was luck?", 3,
                        help="Run a family of variants once (a parameter grid), then ask three questions of it: "
                             "PBO (does the in-sample winner hold up out of sample?), CPCV (what does the "
                             "out-of-sample record look like across many train/test splits?) and the Reality Check "
                             "(is the best variant's return real after searching over all of them?)."),
                unsafe_allow_html=True)
    knobs = knob_choices(spec)
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    px = c1.selectbox("vary", knobs, key="bt_of_x", format_func=knob_label)
    py = c2.selectbox("and", ["—"] + [k for k in knobs if k != px], key="bt_of_y",
                      format_func=lambda k: k if k == "—" else knob_label(k))
    py = None if py == "—" else py
    n = c3.slider("values each", 3, min(12, caps().grid_side), 6 if py else 10, key="bt_of_n")
    xs = default_values(spec, px, n)
    ys = default_values(spec, py, n) if py else None
    nv = len(xs) * (len(ys) if ys else 1)
    key = f"of|{spec.hash()}|{px}|{py}|{xs}|{ys}"
    if st.button(f"RUN · {nv} variants", type="primary", key="bt_of_run"):
        R, hashes = variant_returns(spec, ins, px, xs, py, ys, aux, progress=C.progress_bar("running variants"))
        lg.record(C.ledger(), spec.family(), hashes)
        R = R.loc[:, R.notna().any()]
        out = {"R": R, "pbo": pbo(R, periods=base.periods), "cpcv": cpcv(R, periods=base.periods),
               "rc": reality_check(R, n_boot=min(500, caps().permutations * 2))}
        C.memo()[key] = out
    res = C.memo().get(key)
    if res is None:
        st.caption(f"Each variant is a trial. {nv} variants · CPCV with 6 groups / 2 test → "
                   f"{n_cpcv_paths(6, 2)} out-of-sample paths.")
        return
    P, K, RC = res["pbo"], res["cpcv"], res["rc"]
    oos_med = float(np.nanmedian(K["sharpe"])) if len(K["sharpe"]) else np.nan
    st.markdown(uk.kpis([
        {"label": "PBO", "value": "—" if not np.isfinite(P["pbo"]) else f"{P['pbo'] * 100:.0f}%",
         "sub": "P(in-sample winner is below median OOS)",
         "color": T.teal if P["pbo"] < 0.2 else T.amber if P["pbo"] < 0.5 else T.coral},
        {"label": "CPCV MEDIAN OOS SHARPE", "value": f"{oos_med:.2f}", "sub": f"{K['n_paths']} paths",
         "color": T.teal if oos_med > 0.3 else T.amber if oos_med > 0 else T.coral},
        {"label": "REALITY CHECK p", "value": f"{RC['rc_p']:.3f}", "sub": "White (2000)",
         "color": T.teal if RC["rc_p"] < 0.05 else T.amber if RC["rc_p"] < 0.2 else T.coral},
        {"label": "SPA p", "value": f"{RC['spa_p']:.3f}", "sub": f"Hansen (2005) · block {RC['block']:.0f}",
         "color": T.teal if RC["spa_p"] < 0.05 else T.amber if RC["spa_p"] < 0.2 else T.coral},
    ]), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if len(P["logits"]):
            fig = go.Figure(go.Histogram(x=P["logits"], nbinsx=40, marker_color=up.fill(T.violet, 0.6),
                                         marker_line=dict(width=0)))
            fig.add_vline(x=0, line=dict(color=T.coral, width=2),
                          annotation_text="below median OOS ←", annotation_font=dict(size=9, color=T.coral))
            fig.update_layout(hovermode="closest", showlegend=False)
            fig.update_xaxes(title_text="logit of the winner's OOS rank", showspikes=False)
            up.show(fig, 300, "PBO · where the in-sample winner ranks out of sample", key="bt_of_pbo")
    with c2:
        if len(P["is_sr"]):
            fig = go.Figure(go.Scatter(x=P["is_sr"], y=P["oos_sr"], mode="markers",
                                       marker=dict(size=4, color=up.fill(T.teal, 0.45))))
            b = np.polyfit(P["is_sr"], P["oos_sr"], 1) if np.ptp(P["is_sr"]) > 0 else [0, 0]
            xx = np.linspace(P["is_sr"].min(), P["is_sr"].max(), 10)
            fig.add_trace(go.Scatter(x=xx, y=b[0] * xx + b[1], line=dict(color=T.amber, width=2), name="fit"))
            fig.add_hline(y=0, line=dict(color=T.line))
            fig.update_layout(hovermode="closest", showlegend=False)
            fig.update_xaxes(title_text="in-sample Sharpe of the winner", showspikes=False)
            fig.update_yaxes(title_text="its out-of-sample Sharpe")
            up.show(fig, 300, f"degradation · slope {b[0]:+.2f}", key="bt_of_deg")
    paths = K["paths"]
    if not paths.empty:
        eq = (1 + paths.fillna(0)).cumprod()
        fig = go.Figure()
        for i, col in enumerate(eq.columns):
            fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, line=dict(width=1.2, color=T.cat[i % len(T.cat)])))
        fig.update_yaxes(type="log", title_text="growth of 1")
        up.show(fig, 320, "CPCV · each line is a complete out-of-sample history built from held-out groups",
                key="bt_of_cpcv")
    st.caption("How to read it: PBO under ~20% and a positive CPCV median are good signs; PBO near 50% means "
               "picking the best variant was no better than picking at random. The Reality Check asks whether "
               "even the best of these variants made money once you account for trying all of them "
               "(Bailey et al. 2017; López de Prado 2018; White 2000; Hansen 2005).")


# ============================================================ PROP CHALLENGE
def challenge(spec, ins: pd.DataFrame, aux: dict, base) -> None:
    st.markdown(section("TARGET / STOP CHALLENGE — odds of hitting a goal before a loss limit", 3,
                        help="SharpeEdge's prop-firm Monte Carlo: resample your daily returns in blocks and "
                             "count how often the account reaches the profit target before breaching the loss "
                             "limits, within the time allowed."), unsafe_allow_html=True)
    c = st.columns(5)
    tgt = c[0].number_input("profit target %", 1.0, 100.0, 10.0, 1.0, key="bt_ch_t") / 100
    ml = c[1].number_input("max loss %", 1.0, 50.0, 10.0, 1.0, key="bt_ch_l") / 100
    dl = c[2].number_input("daily loss limit %", 0.5, 50.0, 5.0, 0.5, key="bt_ch_d") / 100
    days = int(c[3].number_input("trading days", 5, 750, 60, 5, key="bt_ch_n"))
    lev = c[4].number_input("leverage ×", 0.1, 10.0, 1.0, 0.1, key="bt_ch_x",
                            help="Scales every daily return. SharpeEdge's point: leverage, not 'risk per trade', "
                                 "is what moves the odds.")
    trailing = st.toggle("trailing max loss (from the high, not the start)", key="bt_ch_tr")
    n = min(5000, caps().mc_paths * 2)
    key = f"ch|{spec.hash()}|{tgt}|{ml}|{dl}|{days}|{lev}|{trailing}|{n}"
    if st.button(f"SIMULATE · {n} challenges", type="primary", key="bt_ch_run"):
        main = challenge_sim(base.returns, tgt, ml, dl, days, lev, n, trailing=trailing)
        sweep = []
        for L in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            s = challenge_sim(base.returns, tgt, ml, dl, days, L, max(400, n // 4), trailing=trailing)
            sweep.append({"leverage": L, "pass": s.get("pass", np.nan), "fail": s.get("fail", np.nan)})
        C.memo()[key] = {"main": main, "sweep": pd.DataFrame(sweep)}
    res = C.memo().get(key)
    if res is None or not res["main"]:
        return
    m = res["main"]
    st.markdown(uk.kpis([
        {"label": "PASS", "value": f"{m['pass'] * 100:.0f}%", "color": T.teal},
        {"label": "FAIL", "value": f"{m['fail'] * 100:.0f}%", "color": T.coral},
        {"label": "OUT OF TIME", "value": f"{m['timeout'] * 100:.0f}%", "color": T.amber},
        {"label": "MEDIAN DAYS TO PASS", "value": f"{np.median(m['days_to_pass']):.0f}" if len(m["days_to_pass"])
         else "—", "sub": f"block {m['block']:.0f} bars"},
    ]), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sw = res["sweep"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw["leverage"], y=sw["pass"] * 100, name="pass", mode="lines+markers",
                                 line=dict(color=T.teal)))
        fig.add_trace(go.Scatter(x=sw["leverage"], y=sw["fail"] * 100, name="fail", mode="lines+markers",
                                 line=dict(color=T.coral)))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(title_text="leverage ×", showspikes=False)
        fig.update_yaxes(title_text="%")
        up.show(fig, 300, "odds vs leverage", key="bt_ch_sw")
    with c2:
        fig = go.Figure(go.Histogram(x=m["final"] * 100, nbinsx=60, marker_color=up.fill(T.violet, 0.6),
                                     marker_line=dict(width=0)))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text="account change at the end (%)", showspikes=False)
        up.show(fig, 300, "where accounts ended", key="bt_ch_fin")
    st.caption("Uses the in-sample daily returns of the strategy as built, resampled in blocks (Politis & White "
               "automatic length) so streaks and volatility clusters survive. Past returns are the only input: if "
               "the edge decays, these odds are too high.")


# ==================================================================== STRESS
def stress(spec, ins: pd.DataFrame, aux: dict, base) -> None:
    st.markdown(section("STRESS — crises it lived through, and shocks it didn't", 3,
                        help="Crisis replay: the strategy's return through named sell-offs inside the in-sample "
                             "period, against buy & hold. Shocks: the same rule re-run on a rebuilt price path "
                             "with doubled volatility, sudden gaps, triple costs or the trend removed."),
                unsafe_allow_html=True)
    b = C.bench(ins["close"], base.returns)
    ct = crisis_table(base.returns, ins["close"].pct_change().reindex(base.returns.index).fillna(0))
    if ct.empty:
        st.caption("No named crisis falls inside the in-sample period (synthetic data has fake dates).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ct.index, y=ct["strategy"] * 100, name="strategy", marker_color=T.teal))
        fig.add_trace(go.Bar(x=ct.index, y=ct["buy & hold"] * 100, name="buy & hold", marker_color=T.amber))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(title_text="return over the window %")
        up.show(fig, 320, "crisis replay · peak-to-trough windows", key="bt_st_cr")
    key = f"st|{spec.hash()}"
    if st.button("RUN SHOCK SCENARIOS · 5 backtests", type="primary", key="bt_st_run"):
        C.memo()[key] = scenario_table(spec, ins, aux)
    t = C.memo().get(key)
    if t is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["scenario"], y=t["Sharpe"], name="Sharpe", marker_color=[T.violet] + [
        T.teal if v >= t["Sharpe"].iloc[0] * 0.5 else T.coral for v in t["Sharpe"].iloc[1:]]))
    fig.update_layout(hovermode="closest", showlegend=False)
    up.show(fig, 280, "sharpe under each shock (purple = as built)", key="bt_st_sh")
    show = t.copy()
    for k in ("CAGR", "MaxDD", "worst 5% bar", "best 5% bar"):
        show[k] = show[k].map(lambda v: f"{v * 100:.1f}%")
    show["Sharpe"] = show["Sharpe"].map(lambda v: f"{v:.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True, column_config=uc.colcfg(show.columns, {
        "worst 5% bar": "Average of the worst 5% of bars (expected tail loss).",
        "best 5% bar": "Average of the best 5% of bars.",
    }))
    st.caption("A robust rule degrades gracefully: some loss under 'vol x2' and 'costs x3' is normal; a collapse "
               "means the edge lived in a narrow set of conditions.")


# ======================================================== MONTE CARLO extras
def mc_extras(spec, ins: pd.DataFrame, base) -> None:
    ab = auto_block_length(base.returns.to_numpy())
    st.caption(f"Politis-White automatic block length for these returns: **{ab:.0f} bars** "
               "(set the slider near it).")
    tr = trade_returns(base.returns, base.position)
    if len(tr) < 10:
        return
    dds = trade_shuffle_drawdowns(tr, n_sims=min(2000, caps().mc_paths))
    eq = np.cumprod(1 + tr)
    actual = float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())
    rank = float((dds <= actual).mean())
    fig = go.Figure(go.Histogram(x=dds * 100, nbinsx=50, marker_color=up.fill(T.coral, 0.55),
                                 marker_line=dict(width=0)))
    fig.add_vline(x=actual * 100, line=dict(color=T.text, width=2), annotation_text="actual order",
                  annotation_font=dict(size=9))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title_text="max drawdown of the trade sequence %", showspikes=False)
    up.show(fig, 280, f"same {len(tr)} trades, shuffled order · {rank * 100:.0f}% of orderings were worse",
            key="bt_mc_tr")
    st.caption("SharpeEdge's trade-order permutation: the trades are fixed, only their order is luck. Plan for "
               f"the 5th-percentile drawdown ({np.percentile(dds, 5) * 100:.0f}%), not the one history happened "
               "to deal you.")


# ======================================================== SENSITIVITY extras
def grid_extras(df: pd.DataFrame, metric: str, px_label: str, py_label: str | None) -> None:
    v = df[metric].dropna()
    if len(v) < 5:
        return
    with st.expander("DISTRIBUTION OF ALL COMBINATIONS · TOP-N CLUSTERS"):
        fig = go.Figure(go.Histogram(x=v, nbinsx=30, marker_color=up.fill(T.violet, 0.6), marker_line=dict(width=0)))
        fig.add_vline(x=float(v.median()), line=dict(color=T.amber, width=2), annotation_text="median",
                      annotation_font=dict(size=9))
        fig.add_vline(x=float(v.max()), line=dict(color=T.teal, width=1, dash="dot"), annotation_text="best",
                      annotation_font=dict(size=9))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text=metric, showspikes=False)
        up.show(fig, 260, f"{metric} across every combination tried", key="bt_s_dist")
        good = (v > 0).mean() if metric in ("Sharpe", "CAGR", "Calmar", "Sortino") else np.nan
        st.caption("SharpeEdge: judge the rule by this whole distribution, not its best cell. "
                   + (f"{good * 100:.0f}% of combinations are positive." if np.isfinite(good) else ""))
        cols = ["x"] + (["y"] if py_label else [])
        tc = top_clusters(df, metric, cols, top=min(30, len(v)), k=3 if len(v) >= 12 else 2)
        if not tc.empty:
            tc = tc.rename(columns={"x": px_label, "y": py_label or "y"})
            st.dataframe(tc, use_container_width=True, hide_index=True)
            st.caption("K-means on the top combinations: a big cluster is a robust region; its representative "
                       "(nearest the centre) is a safer pick than the single best cell.")

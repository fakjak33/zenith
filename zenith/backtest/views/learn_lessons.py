"""LEARN › LESSONS — small interactive experiments, each tied to a tool.

Every lesson is a few sliders and one chart, built to make one idea stick for
someone with no statistics background. Ideas come from the MemLabs
Accelerator / SharpeEdge videos and the papers behind them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T


def log_vs_simple() -> None:
    st.caption("MemLabs · Accelerator Part 2. Up 20% then down 20% does not get you back to where you started.")
    c1, c2 = st.columns(2)
    up_ = c1.slider("first move %", 1, 90, 20, key="ln_up") / 100
    dn = c2.slider("second move % (down)", 1, 90, 20, key="ln_dn") / 100
    end = (1 + up_) * (1 - dn)
    need = 1 / (1 - dn) - 1
    st.markdown(uk.kpis([
        {"label": "SIMPLE: +UP −DOWN", "value": f"{(up_ - dn) * 100:+.1f}%", "sub": "what it looks like"},
        {"label": "ACTUAL RESULT", "value": f"{(end - 1) * 100:+.1f}%", "color": T.coral if end < 1 else T.teal},
        {"label": "LOG RETURNS ADD UP", "value": f"{(np.log(1 + up_) + np.log(1 - dn)) * 100:+.1f}",
         "sub": "log(1+r₁) + log(1−r₂), ×100"},
        {"label": f"GAIN NEEDED AFTER −{dn * 100:.0f}%", "value": f"{need * 100:.0f}%", "color": T.amber},
    ]), unsafe_allow_html=True)
    x = np.linspace(0.01, 0.9, 90)
    fig = go.Figure(go.Scatter(x=x * 100, y=(1 / (1 - x) - 1) * 100, line=dict(color=T.coral, width=2)))
    fig.add_vline(x=dn * 100, line=dict(color=T.text, dash="dot"))
    fig.update_xaxes(title_text="loss %", showspikes=False)
    fig.update_yaxes(title_text="gain needed to recover %", type="log")
    fig.update_layout(hovermode="x", showlegend=False)
    up.show(fig, 280, "why drawdowns hurt more than they look", key="ln_fig")
    st.caption("Log returns are symmetric and simply add over time, which is why quant models use them. "
               "Big losses need much bigger gains to recover — the reason MAX DD sits next to every Sharpe.")


def ev_vs_winrate() -> None:
    st.caption("MemLabs · Introduction to Quantitative Trading. Win rate alone tells you nothing — expected value "
               "(EV) does.")
    c1, c2, c3 = st.columns(3)
    p = c1.slider("win rate %", 5, 95, 45, key="ev_p") / 100
    w = c2.slider("average win", 0.1, 20.0, 1.25, 0.05, key="ev_w")
    l_ = c3.slider("average loss", 0.1, 20.0, 1.0, 0.05, key="ev_l")
    ev = p * w - (1 - p) * l_
    be = l_ / (w + l_)
    rng = np.random.default_rng(0)
    paths = np.cumsum(np.where(rng.random((40, 300)) < p, w, -l_), axis=1)
    fig = go.Figure()
    for i in range(40):
        fig.add_trace(go.Scatter(y=paths[i], line=dict(width=0.8, color=up.fill(T.teal if paths[i, -1] > 0
                                                                                 else T.coral, 0.5)),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=np.arange(1, 301) * ev, line=dict(color=T.text, width=2, dash="dash"), name="EV"))
    fig.update_xaxes(title_text="trade #", showspikes=False)
    fig.update_yaxes(title_text="P&L ($)")
    st.markdown(uk.kpis([
        {"label": "EV PER TRADE", "value": f"{ev:+.3f}", "color": T.teal if ev > 0 else T.coral},
        {"label": "BREAK-EVEN WIN RATE", "value": f"{be * 100:.1f}%"},
        {"label": "AFTER 300 TRADES", "value": f"{(paths[:, -1] > 0).mean() * 100:.0f}%", "sub": "of paths profitable"},
    ]), unsafe_allow_html=True)
    up.show(fig, 320, "40 traders with the same edge", key="ev_fig")
    st.caption("Try 25% wins with average win 35 and loss 1: a losing-most-of-the-time strategy that makes money. "
               "Trend following lives here; option selling lives at the opposite corner.")


def overfitting() -> None:
    st.caption("Bailey & López de Prado (2014). Try enough random strategies and the best one looks brilliant — "
               "then fails out of sample.")
    c1, c2 = st.columns(2)
    n = c1.select_slider("strategies tried", [1, 5, 10, 50, 100, 500, 1000], 100, key="of_n")
    yrs = c2.slider("years of data", 1, 20, 5, key="of_y")
    rng = np.random.default_rng(1)
    T_ = 252 * yrs
    R = rng.normal(0, 0.01, (T_ * 2, n))                  # zero true skill
    ins, oos = R[:T_], R[T_:]
    sr = ins.mean(0) / ins.std(0) * np.sqrt(252)
    best = int(np.argmax(sr))
    sr_oos = oos[:, best].mean() / oos[:, best].std() * np.sqrt(252)
    from scipy.stats import norm
    emc = 0.5772156649
    e_max = ((1 - emc) * norm.ppf(1 - 1 / max(n, 2)) + emc * norm.ppf(1 - 1 / (max(n, 2) * np.e))) if n > 1 else 0
    st.markdown(uk.kpis([
        {"label": "BEST IN-SAMPLE SHARPE", "value": f"{sr[best]:.2f}", "color": T.teal},
        {"label": "SAME STRATEGY OUT-OF-SAMPLE", "value": f"{sr_oos:.2f}", "color": T.coral},
        {"label": "EXPECTED BEST BY LUCK", "value": f"{e_max / np.sqrt(yrs):.2f}", "sub": "≈ E[max] / √years"},
    ]), unsafe_allow_html=True)
    fig = go.Figure(go.Histogram(x=sr, nbinsx=40, marker_color=up.fill(T.violet, 0.6), marker_line=dict(width=0)))
    fig.add_vline(x=sr[best], line=dict(color=T.teal, width=2), annotation_text="the one you'd publish")
    fig.update_xaxes(title_text="in-sample Sharpe of strategies with ZERO skill", showspikes=False)
    fig.update_layout(showlegend=False, hovermode="closest")
    up.show(fig, 280, "luck looks like skill when you search", key="of_fig")
    st.caption("This is what the TRIALS chip and the deflated Sharpe correct for, and what OVERFITTING (PBO) "
               "measures on your own parameter grid.")


def kernels() -> None:
    st.caption("SharpeEdge · Is the Kalman Filter Just a Fancy Moving Average? Every trend filter is a weighted sum "
               "of past returns; they differ mainly in the shape of those weights.")
    n = st.slider("lookback (bars)", 10, 200, 50, key="kn_n")
    k = np.arange(0, 3 * n)
    sma = np.where(k < n, (n - k) / n, 0)                            # SMA of price = triangle on returns
    a = 2 / (n + 1)
    ema = (1 - a) ** k
    mom = np.where(k < n, 1.0, 0)                                     # n-bar momentum = box on returns
    reg = np.where(k < n, 6 * (k + 1) * (n - k) / (n * (n + 1) * (n + 2)) * n / 1.5, 0)
    fig = go.Figure()
    for y, nm, c in ((mom, "momentum (n-bar return)", T.amber), (sma, "price − SMA", T.teal),
                     (ema, "EMA (= simplest Kalman)", T.violet), (reg, "regression slope", T.coral)):
        fig.add_trace(go.Scatter(x=k, y=y / y.sum(), name=nm, line=dict(color=c, width=2)))
    fig.update_xaxes(title_text="bars ago", showspikes=False)
    fig.update_yaxes(title_text="weight on that bar's return")
    up.show(fig, 320, "trend filters as kernels", key="kn_fig")
    st.caption("Same lookback, similar shapes → similar results. Picking 'the best' trend indicator is mostly "
               "picking noise (Levine & Pedersen 2016). Diversify across speeds instead — BLEND does that.")


def leakage() -> None:
    st.caption("López de Prado (2018). On a pure random walk, shuffled k-fold 'finds' an edge when the target "
               "overlaps in time; purged walk-forward does not.")
    h = st.slider("target horizon h (bars)", 1, 20, 10, key="lk_h")
    if st.button("RUN THE EXPERIMENT", key="lk_run"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, cross_val_score
        from ..ml.core import splits
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0, 0.01, 3000))
        y = (r.rolling(h).sum().shift(-h) > 0).astype(int)
        X = pd.concat({f"f{i}": r.rolling(h).sum().shift(i) for i in range(0, 3)}, axis=1)
        X["fwd_neighbour"] = y.shift(1)                 # yesterday's label: its window overlaps today's target
        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        Xv, yv = d.drop(columns="y").to_numpy(), d["y"].to_numpy()
        shuffled = cross_val_score(LogisticRegression(), Xv, yv, cv=KFold(5, shuffle=True, random_state=0)).mean()
        accs = []
        for tr, te in splits(len(d), "purged_kfold", 5, purge=h, embargo=2 * h):
            m = LogisticRegression().fit(Xv[tr][:, :-1], yv[tr])
            accs.append((m.predict(Xv[te][:, :-1]) == yv[te]).mean())
        st.session_state["lk_out"] = (shuffled, float(np.mean(accs)), float(max(yv.mean(), 1 - yv.mean())))
    out = st.session_state.get("lk_out")
    if out:
        st.markdown(uk.kpis([
            {"label": "SHUFFLED K-FOLD", "value": f"{out[0] * 100:.1f}%", "color": T.coral,
             "sub": "accuracy on a random walk"},
            {"label": "PURGED, NO LEAKY FEATURE", "value": f"{out[1] * 100:.1f}%", "color": T.teal,
             "sub": "honest"},
            {"label": "NAIVE: ALWAYS THE MAJORITY", "value": f"{out[2] * 100:.1f}%",
             "sub": "the bar to beat"},
        ]), unsafe_allow_html=True)
        st.caption("The shuffled score is fake: overlapping targets let the test fold peek at its neighbours. "
                   "Purging h bars before each test block is not enough on its own when features also look back "
                   "h bars: the embargo after the block must cover both (2h here). ML LAB never shuffles; set its "
                   "embargo to at least your longest feature window when you use purged k-fold.")


def persistence() -> None:
    st.caption("SharpeEdge · Market Regime Detection. A regime is only tradable if it lasts.")
    p = st.slider("chance a regime continues tomorrow", 0.50, 0.995, 0.95, 0.005, key="ps_p")
    rng = np.random.default_rng(3)
    s = np.zeros(500, dtype=int)
    for t in range(1, 500):
        s[t] = s[t - 1] if rng.random() < p else 1 - s[t - 1]
    fig = go.Figure(go.Scatter(y=s, line=dict(color=T.violet, width=1, shape="hv")))
    fig.update_yaxes(tickvals=[0, 1], ticktext=["calm", "stressed"])
    fig.update_layout(showlegend=False, hovermode="x")
    st.markdown(uk.kpis([{"label": "EXPECTED SPELL LENGTH", "value": f"{1 / (1 - p):.0f} bars",
                          "sub": "= 1 / (1 − p)"}]), unsafe_allow_html=True)
    up.show(fig, 220, "simulated regime path", key="ps_fig")
    st.caption("With spells of a few bars, a filter switches too late and pays costs both ways. BUILD › REGIMES "
               "shows the real transition matrix; the minimum-spell setting adds hysteresis.")


def fees() -> None:
    st.caption("MemLabs · Trend Following / Mean Reversion. A 52% hit rate can be a business with maker fees and "
               "a loss with taker fees.")
    c1, c2, c3 = st.columns(3)
    p = c1.slider("hit rate %", 45.0, 60.0, 52.0, 0.5, key="fe_p") / 100
    mv = c2.slider("average move per trade (bp)", 5, 200, 50, key="fe_m")
    fee = c3.slider("fee per side (bp)", 0.0, 10.0, 4.1, 0.1, key="fe_f")
    gross = (2 * p - 1) * mv
    net = gross - 2 * fee
    fees_x = np.linspace(0, 10, 101)
    fig = go.Figure(go.Scatter(x=fees_x, y=gross - 2 * fees_x, line=dict(color=T.teal, width=2)))
    fig.add_hline(y=0, line=dict(color=T.line))
    for f, nm in ((1.2, "maker 1.2"), (4.1, "taker 4.1")):
        fig.add_vline(x=f, line=dict(color=T.faint, dash="dot"), annotation_text=nm, annotation_font=dict(size=9))
    fig.update_xaxes(title_text="fee per side (bp)", showspikes=False)
    fig.update_yaxes(title_text="net edge per trade (bp)")
    fig.update_layout(showlegend=False, hovermode="x")
    st.markdown(uk.kpis([
        {"label": "GROSS EDGE", "value": f"{gross:+.1f} bp"},
        {"label": "NET EDGE", "value": f"{net:+.1f} bp", "color": T.teal if net > 0 else T.coral},
        {"label": "BREAK-EVEN FEE", "value": f"{gross / 2:.1f} bp / side"},
    ]), unsafe_allow_html=True)
    up.show(fig, 280, "how fees eat a small edge", key="fe_fig")


def leverage() -> None:
    st.caption("MemLabs · Let's Build a Strategy Part 2. Leverage multiplies returns — and moves the liquidation "
               "price closer.")
    c1, c2 = st.columns(2)
    L = c1.slider("leverage ×", 1.0, 50.0, 5.0, 0.5, key="lv_L")
    mm = c2.slider("maintenance margin %", 0.0, 5.0, 0.5, 0.1, key="lv_mm") / 100
    liq_long = L / (L + 1 - mm * L)
    move = (1 - liq_long) * 100
    st.markdown(uk.kpis([
        {"label": "LIQUIDATED AFTER A FALL OF", "value": f"{move:.1f}%", "color": T.coral},
        {"label": "1-DAY MOVES THAT BIG (SPY)", "value": "rare" if move > 5 else "a few a year" if move > 2.5
         else "common", "sub": "daily vol ≈ 1%"},
    ]), unsafe_allow_html=True)
    rng = np.random.default_rng(4)
    path = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 250)))
    liq = path[0] * liq_long
    hit = np.argmax(path <= liq) if (path <= liq).any() else None
    fig = go.Figure(go.Scatter(y=path, line=dict(color=T.text, width=1.2), name="price"))
    fig.add_hline(y=liq, line=dict(color=T.coral, dash="dash"), annotation_text="liquidation")
    if hit:
        fig.add_vline(x=hit, line=dict(color=T.coral))
    fig.update_layout(showlegend=False, hovermode="x")
    up.show(fig, 260, "one simulated year at this leverage", key="lv_fig")
    st.caption("Liquidation (long) = entry × L / (L + 1 − m·L). Kelly's rule caps sensible leverage at "
               "(expected excess return) ÷ variance — for most strategies far below what exchanges allow. The "
               "CHALLENGE test in ROBUSTNESS shows how leverage changes the odds of blowing up.")


LESSONS = {
    "Log vs simple returns": log_vs_simple,
    "EV beats win rate": ev_vs_winrate,
    "Overfitting: luck looks like skill": overfitting,
    "Trend filters are kernels": kernels,
    "Leakage and purging": leakage,
    "Regime persistence": persistence,
    "Fees vs a small edge": fees,
    "Leverage and liquidation": leverage,
}

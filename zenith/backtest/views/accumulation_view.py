"""BACKTEST / ACCUMULATION — when to put new savings to work (ported PARALLAX lab).

Compares a buy rule (dip-buying, VIX spikes, RSI, regression bands…) and an
optional trim rule against plain dollar-cost averaging and lump-sum buy & hold,
on the same contribution schedule — so 'waiting for a dip' pays for the cash it
sits on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import key_findings, section
from ..accumulation import strategies as AS
from ..accumulation.engine import AccumConfig, _accum_stats, benchmarks, run_accumulation
from ..accumulation.regression import regression_channel
from ..data.providers import TIMEFRAMES, resample_ohlcv
from ..data.synthetic import generate
from ..engine import metrics
from . import common as C

_FINDINGS = [
    {"stat": "Lump-sum investing beat dollar-cost averaging about two-thirds of the time across US, UK and "
             "Australian markets, because markets rise more often than they fall.", "cite": "Vanguard (2012)"},
    {"stat": "Waiting for a dip usually loses to buying now: the cash drag while waiting outweighs the "
             "discount captured.", "cite": "Ben Carlson / A Wealth of Common Sense (2019)"},
]


def _params(cls, prefix: str) -> dict:
    vals = {}
    cols = st.columns(2)
    for i, (n, (d, lo, hi, step)) in enumerate(cls.params.items()):
        with cols[i % 2]:
            if isinstance(d, int) and isinstance(step, int):
                vals[n] = int(st.number_input(n, int(lo), int(hi), int(d), int(step), key=f"ac_{prefix}_{cls.key}_{n}"))
            else:
                vals[n] = float(st.number_input(n, float(lo), float(hi), float(d), float(step),
                                                key=f"ac_{prefix}_{cls.key}_{n}"))
    return vals


def _builder():
    st.markdown(section("01 · ASSET", 0), unsafe_allow_html=True)
    src = st.radio("source", ["Market", "Synthetic"], horizontal=True, key="ac_src", label_visibility="collapsed")
    if src == "Market":
        pick = st.selectbox("asset", C.QUICK + ["custom…"], key="ac_quick")
        name = st.text_input("ticker", "AAPL", key="ac_custom").strip().upper() if pick == "custom…" else pick
    else:
        name = st.selectbox("synthetic market", C.SYNTH, key="ac_synth")
    c1, c2 = st.columns(2)
    start = c1.number_input("from", 1990, 2024, 2010, key="ac_start")
    tf = c2.selectbox("bars", list(TIMEFRAMES), index=1, key="ac_tf")

    st.markdown(section("02 · CONTRIBUTIONS", 1), unsafe_allow_html=True)
    cfg = AccumConfig()
    c1, c2 = st.columns(2)
    cfg.initial_cash = c1.number_input("initial $", 0.0, 1e9, 10_000.0, 1_000.0, key="ac_init")
    cfg.contribution = c2.number_input("each contribution $", 0.0, 1e8, 1_000.0, 100.0, key="ac_contrib")
    cfg.cadence = st.selectbox("cadence", ["Daily", "Weekly", "Monthly"], index=1, key="ac_cad")
    mode = st.radio("deploy per signal", ["% of cash", "fixed $"], horizontal=True, key="ac_dmode")
    if mode == "% of cash":
        cfg.deploy_mode, cfg.deploy_fraction = "pct_cash", st.slider("% of cash", 0.05, 1.0, 0.5, 0.05, key="ac_frac")
    else:
        cfg.deploy_mode = "fixed_dollar"
        cfg.deploy_dollar = st.number_input("$ per signal", 100.0, 1e8, 5_000.0, 500.0, key="ac_fixed")
    c1, c2 = st.columns(2)
    cfg.min_signal_gap = int(c1.number_input("min bars between buys", 0, 250, 0, 1, key="ac_gap"))
    cfg.cash_yield_annual = c2.number_input("idle cash yield", 0.0, 0.10, 0.03, 0.005, key="ac_yield",
                                            help="Waiting in T-bills earns something; 0 overstates the cost of waiting.")

    st.markdown(section("03 · BUY RULE", 2), unsafe_allow_html=True)
    buy_key = st.selectbox("trigger", list(AS.BUY_RULES), format_func=lambda k: AS.BUY_RULES[k].label, key="ac_buy")
    buy_cls = AS.BUY_RULES[buy_key]
    st.caption(buy_cls.desc)
    buy = buy_cls(**_params(buy_cls, "buy"))

    st.markdown(section("04 · TRIM RULE (OPTIONAL)", 4), unsafe_allow_html=True)
    sell_key = st.selectbox("trim", ["—"] + list(AS.SELL_RULES), key="ac_sell",
                            format_func=lambda k: k if k == "—" else AS.SELL_RULES[k].label)
    sell = None
    if sell_key != "—":
        sc = AS.SELL_RULES[sell_key]
        st.caption(sc.desc)
        cfg.sell_fraction = st.slider("sell fraction per signal", 0.05, 1.0, 0.25, 0.05, key="ac_sfrac")
        sell = sc(**_params(sc, "sell"))
    return src, name, int(start), tf, cfg, buy, sell, buy_cls


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        src, name, start, tf, cfg, buy, sell, buy_cls = _builder()
    with right:
        try:
            df = generate(name, n_days=4000, seed=42) if src == "Synthetic" else C._market(name)
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", f"{name}: {exc}")
            return
        df = resample_ohlcv(df[df.index >= pd.Timestamp(f"{start}-01-01")], TIMEFRAMES[tf])
        close = df["close"].dropna()
        ctx = {}
        if buy_cls.key == "vix":
            try:
                ctx["vix"] = C._market("^VIX")["close"]
            except Exception:  # noqa: BLE001
                C.banner(T.amber, "NO VIX", "couldn't load ^VIX — the VIX rule will not fire")
        res = run_accumulation(close, buy, sell, cfg, ctx)
        bench = benchmarks(close, cfg)
        dca = _accum_stats(bench["DCA"], res.invested, close)
        s = res.stats
        edge = s["FinalEquity"] - dca["FinalEquity"]
        st.markdown(uk.kpis([
            {"label": "FINAL EQUITY", "value": f"${s['FinalEquity']:,.0f}", "color": T.teal if edge >= 0 else T.coral,
             "sub": f"{'+' if edge >= 0 else '−'}${abs(edge):,.0f} vs plain DCA"},
            {"label": "CONTRIBUTED", "value": f"${s['Contributed']:,.0f}"},
            {"label": "RETURN ON SAVED $", "value": f"{s['ReturnOnContributed%']:.1f}%",
             "sub": f"DCA {dca['ReturnOnContributed%']:.1f}%"},
            {"label": "RETURN ON DEPLOYED $", "value": f"{s['ReturnOnDeployed%']:.1f}%", "sub": "excl. idle cash"},
            {"label": "MAX DD", "value": f"{s['MaxDD%']:.1f}%", "color": T.coral, "sub": f"DCA {dca['MaxDD%']:.1f}%"},
            {"label": "CASH NOW", "value": f"${res.cash.iloc[-1]:,.0f}", "sub": "dry powder"},
        ]), unsafe_allow_html=True)
        if edge < 0:
            C.banner(T.amber, "DCA WINS", "the rule ended behind simply investing every contribution on schedule — "
                     "the usual result for dip-buying; check whether it bought you a smaller drawdown instead")

        tabs = st.tabs(["EQUITY", "SIGNALS", "DRAWDOWN", "STATS", "REGRESSION"])
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=res.invested.index, y=res.invested, name="contributed",
                                     line=dict(color=T.faint, dash="dash", width=1)), 1, 1)
            for i, (nm, eq) in enumerate(bench.items()):
                fig.add_trace(go.Scatter(x=eq.index, y=eq, name=nm, line=dict(color=[T.amber, T.violet][i], width=1.2)),
                              1, 1)
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name=buy.label,
                                     line=dict(color=T.teal, width=1.8)), 1, 1)
            fig.add_trace(go.Scatter(x=res.profit.index, y=res.profit, name="P/L", fill="tozeroy",
                                     fillcolor=up.fill(T.teal, 0.15), line=dict(color=T.teal, width=1),
                                     showlegend=False), 2, 1)
            dpl = bench["DCA"] - res.invested.reindex(bench["DCA"].index)
            fig.add_trace(go.Scatter(x=dpl.index, y=dpl, name="DCA P/L", line=dict(color=T.amber, width=1),
                                     showlegend=False), 2, 1)
            fig.update_yaxes(title_text="$", row=1, col=1)
            fig.update_yaxes(title_text="P/L $", row=2, col=1)
            up.show(fig, 480, "equity vs DCA and lump sum · P/L below", key="ac_eq")
        with tabs[1]:
            bs = buy.signal(close, ctx).reindex(close.index).fillna(False).astype(bool).to_numpy()
            ss = (sell.signal(close, ctx).reindex(close.index).fillna(False).astype(bool).to_numpy()
                  if sell else np.zeros(len(close), bool))
            pan = None
            try:
                pan = buy.panel_indicator(close, ctx)
            except Exception:  # noqa: BLE001
                pass
            fig = make_subplots(rows=2 if pan else 1, cols=1, shared_xaxes=True,
                                row_heights=[0.65, 0.35] if pan else [1.0], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=close.index, y=close, name=name, line=dict(color=T.text, width=1)), 1, 1)
            fig.add_trace(go.Scatter(x=close.index[bs], y=close.to_numpy()[bs], mode="markers", name=f"buy ({bs.sum()})",
                                     marker=dict(symbol="triangle-up", size=8, color=T.teal)), 1, 1)
            if ss.any():
                fig.add_trace(go.Scatter(x=close.index[ss], y=close.to_numpy()[ss], mode="markers",
                                         name=f"trim ({ss.sum()})",
                                         marker=dict(symbol="triangle-down", size=8, color=T.coral)), 1, 1)
            if pan:
                ser, lbl, levels = pan
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=lbl, line=dict(color=T.amber, width=1)), 2, 1)
                for v, lab in levels:
                    fig.add_hline(y=v, line=dict(color=T.coral, dash="dot", width=1), row=2, col=1)
            up.show(fig, 460, "price, buy and trim signals", key="ac_sig")
        with tabs[2]:
            fig = go.Figure()
            for i, (nm, eq) in enumerate([(buy.label, res.equity)] + list(bench.items())):
                dd = metrics.drawdown_series(eq.pct_change().fillna(0)) * 100
                fig.add_trace(go.Scatter(x=dd.index, y=dd, name=nm, line=dict(color=[T.teal, T.amber, T.violet][i], width=1)))
            fig.update_yaxes(title_text="% below peak")
            up.show(fig, 340, "drawdown of the account (contributions included)", key="ac_dd")
        with tabs[3]:
            rows = {buy.label: s, **{nm: _accum_stats(eq, res.invested, close) for nm, eq in bench.items()}}
            cols = ["FinalEquity", "Contributed", "Profit", "ReturnOnContributed%", "ReturnOnDeployed%", "AnnVol%",
                    "MaxDD%", "Beta", "Corr"]
            st.dataframe(pd.DataFrame(rows).T[cols].round(2), use_container_width=True)
        with tabs[4]:
            c1, c2, c3 = st.columns(3)
            lb = int(c1.number_input("lookback bars", 60, 2000, 504, 10, key="ac_rlb"))
            k = float(c2.number_input("bands k·σ", 0.5, 4.0, 2.0, 0.25, key="ac_rk"))
            logfit = c3.toggle("log fit", True, key="ac_rlog")
            ch = regression_channel(close, lb, k, logfit)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close.index, y=close, name=name, line=dict(color=T.text, width=1)))
            fig.add_trace(go.Scatter(x=ch.index, y=ch["fit"], name="fit", line=dict(color=T.amber, width=1)))
            fig.add_trace(go.Scatter(x=ch.index, y=ch["upper"], name="+kσ", line=dict(color=T.coral, dash="dot", width=1)))
            fig.add_trace(go.Scatter(x=ch.index, y=ch["lower"], name="−kσ", line=dict(color=T.teal, dash="dot", width=1)))
            if logfit:
                fig.update_yaxes(type="log")
            up.show(fig, 400, "regression channel", key="ac_reg")

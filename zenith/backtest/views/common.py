"""Shared LAB plumbing: data loading, the block builder, the ledger handle,
result memo, guard-rail chips and the core result charts."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ... import ui_charts as uc
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import ledger as lg
from ..config import caps, is_local
from ..data.providers import TIMEFRAMES, get_prices, resample_ohlcv
from ..data.synthetic import generate
from ..engine import metrics
from ..rules import REGISTRY
from ..spec import FILTERS, StrategySpec

QUICK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "^GSPC",
         "ES=F", "NQ=F", "ZN=F", "GC=F", "CL=F", "6E=F", "BTC-USD"]
SYNTH = ["trending", "mean_reverting", "regime", "fat_tailed", "gbm"]
MIN_TRADES, MIN_YEARS = 30, 5.0


# ------------------------------------------------------------------ data ----
@st.cache_data(ttl=3600, show_spinner=False)
def _market(ticker: str) -> pd.DataFrame:
    return get_prices(ticker, start=None)


def load(spec: StrategySpec, name: str | None = None) -> pd.DataFrame:
    name = name or spec.name
    df = generate(name, n_days=5000, seed=42) if spec.src == "Synthetic" else _market(name)
    df = df[df.index >= pd.Timestamp(f"{spec.start_year}-01-01")]
    return resample_ohlcv(df, TIMEFRAMES[spec.tf]).dropna(subset=["close"])


def load_aux(spec: StrategySpec) -> dict:
    """Side series that filters need (VIX), loaded only when a filter uses them."""
    aux = {}
    if spec.src == "Market" and any(f["key"] == "vix" and f.get("on") for f in spec.filters):
        try:
            aux["vix"] = _market("^VIX")["close"]
        except Exception:  # noqa: BLE001
            aux["vix"] = None
    return aux


# ---------------------------------------------------------------- ledger ----
def ledger() -> dict:
    if "bt_ledger" not in st.session_state:
        st.session_state["bt_ledger"] = lg.load()
    return st.session_state["bt_ledger"]


def memo() -> dict:
    """Per-session store of heavy results, keyed by step + spec hash + settings."""
    return st.session_state.setdefault("bt_memo", {})


def apply_pending() -> None:
    """Widget values queued by a button (e.g. 'use plateau centre') before widgets exist."""
    for k, v in st.session_state.pop("bt_pending", {}).items():
        st.session_state[k] = v


def queue(**kw) -> None:
    st.session_state.setdefault("bt_pending", {}).update(kw)
    st.rerun()


def param_key(rule: str, name: str) -> str:
    return f"bt_p_{rule}_{name}"


# ------------------------------------------------------------- builder ----
def _num(label, spec_tuple, key, help=None):
    default, lo, hi, step = spec_tuple
    if isinstance(default, int) and isinstance(step, int):
        return int(st.number_input(label, int(lo), int(hi), int(default), int(step), key=key, help=help))
    return float(st.number_input(label, float(lo), float(hi), float(default), float(step), key=key,
                                 format="%.2f", help=help))


def builder() -> StrategySpec:
    """Left-hand block builder → StrategySpec."""
    st.markdown(section("01 · UNIVERSE", 0), unsafe_allow_html=True)
    src = st.radio("source", ["Market", "Synthetic"], horizontal=True, key="bt_src",
                   label_visibility="collapsed")
    if src == "Market":
        pick = st.selectbox("instrument", QUICK + ["custom…"], key="bt_quick")
        name = st.text_input("ticker", "AAPL", key="bt_custom").strip().upper() if pick == "custom…" else pick
    else:
        name = st.selectbox("synthetic market", SYNTH, key="bt_synth",
                            help="Known-answer worlds: a trend rule should win on 'trending' and fail on 'gbm'.")
    c1, c2 = st.columns(2)
    tf = c1.selectbox("bars", list(TIMEFRAMES), key="bt_tf")
    start_year = c2.number_input("from", 1990, 2024, 2005, key="bt_start")

    st.markdown(section("02 · SIGNAL", 1), unsafe_allow_html=True)
    keys = [k for k, c in REGISTRY.items() if not c.cross_sectional]
    rule = st.selectbox("rule", keys, index=keys.index("ema") if "ema" in keys else 0,
                        format_func=lambda k: REGISTRY[k].label, key="bt_rule")
    cls = REGISTRY[rule]
    if cls.plain:
        st.caption(cls.plain)
    params, cols = {}, st.columns(2)
    for i, (pname, spec_t) in enumerate(cls.params.items()):
        with cols[i % 2]:
            params[pname] = _num(pname, spec_t, param_key(rule, pname))

    st.markdown(section("03 · FILTERS", 5, help="Gates that can only remove exposure. Each one you add is "
                        "another trial; the FILTERS step tests whether it earns its place."),
                unsafe_allow_html=True)
    filters = []
    for fkey, f in FILTERS.items():
        on = st.toggle(f["label"], key=f"bt_f_{fkey}", help=f["help"])
        fp = {}
        if on:
            fc = st.columns(len(f["params"]))
            for j, (pn, pt) in enumerate(f["params"].items()):
                with fc[j]:
                    fp[pn] = _num(pn, pt, f"bt_fp_{fkey}_{pn}")
            filters.append({"key": fkey, "on": True, "params": fp})

    st.markdown(section("04 · SIZING & EXITS", 2), unsafe_allow_html=True)
    mode = st.radio("sizing", ["vol_target", "fixed_pct"], horizontal=True, key="bt_size",
                    format_func=lambda m: {"vol_target": "vol target", "fixed_pct": "fixed %"}[m])
    c1, c2 = st.columns(2)
    tvol = c1.slider("target vol %", 5, 40, 20, key="bt_tvol",
                     help="Carver's default is 20%. Ignored for fixed % sizing.") / 100
    direction = c2.selectbox("direction", ["both", "long", "short"], key="bt_dir")
    c1, c2 = st.columns(2)
    stop = c1.toggle("ATR stop", key="bt_stop")
    tp = c2.toggle("ATR target", key="bt_tp")
    stop_k = c1.number_input("stop × ATR", 0.5, 10.0, 3.0, 0.5, key="bt_stopk", disabled=not stop)
    tp_k = c2.number_input("target × ATR", 0.5, 20.0, 6.0, 0.5, key="bt_tpk", disabled=not tp)

    st.markdown(section("05 · COSTS & HOLDOUT", 4), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cost = c1.number_input("commission bps", 0.0, 50.0, 1.0, 0.5, key="bt_cost")
    slip = c2.number_input("slippage bps", 0.0, 50.0, 1.5, 0.5, key="bt_slip")
    hold = st.slider("locked out-of-sample %", 0, 50, 30, 5, key="bt_hold",
                     help="The most recent slice of history is hidden from every step until you reveal it "
                          "once, at the end. Setting 0 removes your only honest test.")
    return StrategySpec(src=src, name=name, tf=tf, start_year=int(start_year), rule=rule, params=params,
                        filters=filters, sizing_mode=mode, target_vol=tvol, direction=direction,
                        use_atr_stop=stop, use_atr_tp=tp, atr_stop_mult=stop_k, atr_tp_mult=tp_k,
                        cost_bps=cost, slippage_bps=slip, holdout=int(hold),
                        notes=st.session_state.get("bt_notes", ""))


# -------------------------------------------------------------- rails ----
def rails(spec: StrategySpec, n_trades: int | None, years: float, locked, n_trials: int) -> str:
    costs_on = spec.cost_bps + spec.slippage_bps > 0
    led = ledger()
    contaminated = lg.contaminated(led, spec.family(), spec.hash())
    seen = lg.revealed(led, spec.family())
    chips = [uc.chip("NEXT-BAR FILLS", T.teal), uc.chip("CAUSAL SCALING", T.teal),
             uc.chip("COSTS ON" if costs_on else "COSTS OFF", T.teal if costs_on else T.coral)]
    if locked is None:
        chips.append(uc.chip("NO HOLDOUT", T.coral))
    elif contaminated:
        chips.append(uc.chip("OOS CONTAMINATED", T.coral, "changed after reveal"))
    elif seen:
        chips.append(uc.chip("OOS REVEALED", T.amber, seen["at"]))
    else:
        chips.append(uc.chip(f"OOS LOCKED FROM {locked.date()}", T.teal))
    if n_trades is not None:
        chips.append(uc.chip(f"{n_trades} TRADES", T.teal if n_trades >= MIN_TRADES else T.amber))
    chips.append(uc.chip(f"{years:.1f} YRS IN-SAMPLE", T.teal if years >= MIN_YEARS else T.amber))
    chips.append(uc.chip(f"{n_trials} TRIALS", T.teal if n_trials <= 20 else T.amber if n_trials <= 100 else T.coral,
                         "feeds DSR"))
    chips.append(uc.chip("LOCAL · CAPS OFF" if is_local() else f"CLOUD · ≤{caps().permutations} PERMS", T.violet))
    return '<div style="margin:0.2rem 0 0.6rem 0;">' + "".join(chips) + "</div>"


def banner(color: str, label: str, msg: str) -> None:
    st.markdown(uc.state_banner(color, label, msg), unsafe_allow_html=True)


def fmt(k: str, v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if k in ("CAGR", "Vol", "MaxDD", "HitRate", "Ulcer", "exposure"):
        return f"{v * 100:.1f}%"
    if k in ("PSR", "DSR"):
        return f"{v * 100:.0f}%"
    if k == "MaxDD_bars":
        return f"{int(v)}"
    return f"{v:.2f}"


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)
    return lambda f: bar.progress(min(1.0, f), text=f"{label} · {f * 100:.0f}%")


# -------------------------------------------------------------- charts ----
def bench(close: pd.Series, strat: pd.Series) -> pd.Series:
    """Buy-and-hold of the same instrument, scaled to the strategy's realised vol."""
    bh = close.pct_change().reindex(strat.index).fillna(0.0)
    sv, bv = strat.std(), bh.std()
    return bh * (sv / bv) if bv > 1e-12 and sv > 1e-12 else bh


def equity_fig(r: pd.Series, b: pd.Series, gross: pd.Series | None = None,
               shade_from: pd.Timestamp | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    eq, beq = (1 + r).cumprod(), (1 + b).cumprod()
    if gross is not None:
        geq = (1 + gross).cumprod()
        fig.add_trace(go.Scatter(x=geq.index, y=geq, name="gross of costs", line=dict(color=T.faint, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=beq.index, y=beq, name="buy & hold (vol-matched)",
                             line=dict(color=T.amber, width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="strategy", line=dict(color=T.teal, width=1.8)), 1, 1)
    dd, bdd = metrics.drawdown_series(r), metrics.drawdown_series(b)
    fig.add_trace(go.Scatter(x=bdd.index, y=bdd, name="b&h drawdown", line=dict(color=T.amber, width=0.8),
                             showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="drawdown", fill="tozeroy", fillcolor=up.fill(T.coral, 0.25),
                             line=dict(color=T.coral, width=1), showlegend=False), 2, 1)
    if shade_from is not None:
        fig.add_vrect(x0=shade_from, x1=r.index[-1], fillcolor=up.fill(T.violet, 0.10), line_width=0,
                      annotation_text="OUT-OF-SAMPLE", annotation_position="top left",
                      annotation_font=dict(size=9, color=T.violet))
    fig.update_yaxes(type="log", title_text="growth of 1", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="drawdown", row=2, col=1)
    return fig


def price_fig(ohlcv: pd.DataFrame, res, strat, trades: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    close = ohlcv["close"]
    fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=T.text, width=1)), 1, 1)
    for i, (lbl, s) in enumerate(strat.indicator_lines(ohlcv).items()):
        fig.add_trace(go.Scatter(x=s.index, y=s, name=lbl, line=dict(color=T.cat[(i + 1) % len(T.cat)], width=1)),
                      1, 1)
    if not trades.empty:
        for side, sym, col in (("LONG", "triangle-up", T.teal), ("SHORT", "triangle-down", T.coral)):
            t = trades[trades["side"] == side]
            if not t.empty:
                fig.add_trace(go.Scatter(x=pd.to_datetime(t["entry_date"]), y=t["entry_price"], mode="markers",
                                         name=f"{side.lower()} entry",
                                         marker=dict(symbol=sym, size=8, color=col, line=dict(width=0))), 1, 1)
    f = res.forecast
    fig.add_trace(go.Scatter(x=f.index, y=f.clip(lower=0), fill="tozeroy", fillcolor=up.fill(T.teal, 0.3),
                             line=dict(width=0), name="forecast +", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=f.index, y=f.clip(upper=0), fill="tozeroy", fillcolor=up.fill(T.coral, 0.3),
                             line=dict(width=0), name="forecast −", showlegend=False), 2, 1)
    fig.update_yaxes(title_text="forecast", range=[-21, 21], row=2, col=1)
    return fig


def rolling_sharpe_fig(r: pd.Series, b: pd.Series, periods: int) -> go.Figure:
    w = max(periods, 12)
    rs = r.rolling(w).mean() / r.rolling(w).std() * np.sqrt(periods)
    bs = b.rolling(w).mean() / b.rolling(w).std() * np.sqrt(periods)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=T.line, width=1))
    fig.add_trace(go.Scatter(x=bs.index, y=bs, name="buy & hold", line=dict(color=T.amber, width=1)))
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name="strategy", line=dict(color=T.teal, width=1.6)))
    return fig


def monthly_fig(r: pd.Series) -> go.Figure:
    m = (1 + r).resample("ME").prod() - 1
    tbl = pd.DataFrame({"y": m.index.year, "m": m.index.month, "v": m.values}).pivot(index="y", columns="m",
                                                                                   values="v")
    tbl = tbl.reindex(columns=range(1, 13))
    yr = (1 + r).resample("YE").prod() - 1
    yr.index = yr.index.year
    cap = float(np.nanpercentile(np.abs(tbl.to_numpy()), 95)) if tbl.notna().any().any() else 0.05
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    vals = tbl.copy()
    vals[13] = yr.reindex(tbl.index)
    color = vals.copy()
    color[13] = color[13] / 3.5            # wider scale for the YEAR column; text shows the true value
    text = vals.map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}")
    fig = go.Figure(go.Heatmap(z=color.to_numpy(), customdata=vals.to_numpy(), x=months + ["YEAR"],
                               y=[str(y) for y in tbl.index], zmin=-cap, zmax=cap,
                               colorscale=up.DIVERGING, text=text.to_numpy(), texttemplate="%{text}",
                               textfont=dict(size=9), xgap=1, ygap=1, showscale=False,
                               hovertemplate="%{y} %{x}: %{customdata:.2%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showspikes=False, type="category", dtick=1)
    fig.update_xaxes(showspikes=False, side="top")
    fig.update_layout(hovermode="closest")
    return fig


def dist_fig(r: pd.Series, b: pd.Series) -> go.Figure:
    fig = go.Figure()
    for s, name, col in ((b, "buy & hold", T.amber), (r, "strategy", T.teal)):
        x = s[s != 0] * 100
        fig.add_trace(go.Histogram(x=x, name=name, nbinsx=80, marker_color=up.fill(col, 0.55),
                                   marker_line=dict(width=0)))
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title_text="bar return %", showspikes=False)
    return fig


def esc(x) -> str:
    return html.escape(str(x))

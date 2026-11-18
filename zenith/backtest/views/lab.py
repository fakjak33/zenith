"""BACKTEST / LAB — build a strategy by clicking, run it with the guard rails on.

Phase 1 ships the BUILD step: universe → signal → sizing → exits → costs →
holdout. Every run is next-bar, causally scaled, costed, annualized at the
data's own frequency, and shown beside a vol-matched buy-and-hold of the same
instrument. The last ``holdout`` % of history is locked away and never shown.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ... import ui_charts as uc
from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import key_findings, section
from ..config import BacktestConfig, RiskConfig, caps, is_local, scale_params_for_tf
from ..data.providers import TIMEFRAMES, get_prices, resample_ohlcv
from ..data.synthetic import generate
from ..engine import metrics
from ..engine.engine import run_single
from ..engine.trades import extract_trades
from ..rules import REGISTRY

STEPS = ["DEFINE", "BUILD", "SENSITIVITY", "ROBUSTNESS", "FILTERS", "ATTRIBUTION", "OUT-OF-SAMPLE"]
LIVE_STEPS = {"BUILD"}
QUICK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "^GSPC",
         "ES=F", "NQ=F", "ZN=F", "GC=F", "CL=F", "6E=F", "BTC-USD"]
SYNTH = ["trending", "mean_reverting", "regime", "fat_tailed", "gbm"]
MIN_TRADES, MIN_YEARS = 30, 5.0

_FINDINGS = [
    {"stat": "Fit as little as possible; judge a rule by how it behaves across a whole range of "
             "parameters, not at one tuned point.", "cite": "Carver, Systematic Trading (2015)"},
    {"stat": "Testing many variants inflates the best Sharpe you will find by luck alone — the deflated "
             "Sharpe ratio corrects for the number of trials.", "cite": "Bailey & López de Prado (2014)"},
    {"stat": "Most published anomalies shrink sharply out of sample and after publication.",
     "cite": "McLean & Pontiff (2016)"},
]


# ------------------------------------------------------------------ data ----
@st.cache_data(ttl=3600, show_spinner=False)
def _market(ticker: str) -> pd.DataFrame:
    return get_prices(ticker, start=None)


def load(src: str, name: str, tf: str, start_year: int) -> pd.DataFrame:
    if src == "Synthetic":
        df = generate(name, n_days=5000, seed=42)
    else:
        df = _market(name)
    df = df[df.index >= pd.Timestamp(f"{start_year}-01-01")]
    return resample_ohlcv(df, TIMEFRAMES[tf]).dropna(subset=["close"])


def split_holdout(df: pd.DataFrame, pct: float) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """In-sample frame and the first locked date (None when no holdout)."""
    if pct <= 0:
        return df, None
    cut = int(len(df) * (1 - pct / 100))
    return df.iloc[:cut], df.index[cut]


# ------------------------------------------------------------- builder ----
def _param_inputs(cls, key: str) -> dict:
    vals = {}
    cols = st.columns(2)
    for i, (name, (default, lo, hi, step)) in enumerate(cls.params.items()):
        with cols[i % 2]:
            if isinstance(default, int) and isinstance(step, int):
                vals[name] = int(st.number_input(name, int(lo), int(hi), int(default), int(step),
                                                 key=f"bt_p_{key}_{name}"))
            else:
                vals[name] = float(st.number_input(name, float(lo), float(hi), float(default), float(step),
                                                   key=f"bt_p_{key}_{name}", format="%.2f"))
    return vals


def builder() -> dict:
    """Left-hand block builder; returns everything a run needs."""
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
    raw = _param_inputs(cls, rule)

    st.markdown(section("03 · SIZING", 2), unsafe_allow_html=True)
    mode = st.radio("sizing", ["vol_target", "fixed_pct"], horizontal=True, key="bt_size",
                    format_func=lambda m: {"vol_target": "vol target", "fixed_pct": "fixed %"}[m])
    c1, c2 = st.columns(2)
    tvol = c1.slider("target vol %", 5, 40, 20, key="bt_tvol",
                     help="Carver's default is 20%. Ignored for fixed % sizing.") / 100
    direction = c2.selectbox("direction", ["both", "long", "short"], key="bt_dir")

    st.markdown(section("04 · EXITS", 3), unsafe_allow_html=True)
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
                     help="The most recent slice of history is hidden from every tab until you reveal it once, "
                          "at the end. Setting 0 removes your only honest test.")
    return dict(src=src, name=name, tf=tf, start_year=int(start_year), rule=rule, raw=raw,
                bt=BacktestConfig(sizing_mode=mode, target_vol=tvol, direction=direction,
                                  cost_bps=cost, slippage_bps=slip),
                risk=RiskConfig(use_atr_stop=stop, use_atr_tp=tp, atr_stop_mult=stop_k, atr_tp_mult=tp_k),
                hold=hold)


# -------------------------------------------------------------- charts ----
def _bench(close: pd.Series, strat: pd.Series) -> pd.Series:
    """Buy-and-hold of the same instrument, scaled to the strategy's realised vol."""
    bh = close.pct_change().reindex(strat.index).fillna(0.0)
    sv, bv = strat.std(), bh.std()
    return bh * (sv / bv) if bv > 1e-12 and sv > 1e-12 else bh


def equity_fig(r: pd.Series, b: pd.Series, gross: pd.Series) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    eq, beq, geq = (1 + r).cumprod(), (1 + b).cumprod(), (1 + gross).cumprod()
    fig.add_trace(go.Scatter(x=geq.index, y=geq, name="gross of costs", line=dict(color=T.faint, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=beq.index, y=beq, name="buy & hold (vol-matched)",
                             line=dict(color=T.amber, width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="strategy", line=dict(color=T.teal, width=1.8)), 1, 1)
    dd = metrics.drawdown_series(r)
    bdd = metrics.drawdown_series(b)
    fig.add_trace(go.Scatter(x=bdd.index, y=bdd, name="b&h drawdown", line=dict(color=T.amber, width=0.8),
                             showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="drawdown", fill="tozeroy", fillcolor=up.fill(T.coral, 0.25),
                             line=dict(color=T.coral, width=1), showlegend=False), 2, 1)
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
    # colour the YEAR column on a 3.5× wider scale so it doesn't saturate; text shows the true value
    color = vals.copy()
    color[13] = color[13] / 3.5
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


def cost_sensitivity(strat, ohlcv: pd.DataFrame, bt: BacktestConfig, risk: RiskConfig) -> pd.DataFrame:
    fc = strat.forecast(ohlcv)
    rows = []
    for k in (0, 1, 2, 3):
        cfg = BacktestConfig(**{**bt.__dict__, "cost_bps": bt.cost_bps * k, "slippage_bps": bt.slippage_bps * k})
        res = run_single(fc, ohlcv, bt=cfg, risk=risk)
        rows.append({"costs": f"{k}×", "Sharpe": res.stats["Sharpe"], "CAGR": res.stats["CAGR"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- page ----
def _rails(nb: int, years: float, cfg: dict, locked: pd.Timestamp | None) -> str:
    chips = [uc.chip("NEXT-BAR FILLS", T.teal), uc.chip("CAUSAL SCALING", T.teal),
             uc.chip("COSTS ON" if cfg["bt"].cost_bps + cfg["bt"].slippage_bps > 0 else "COSTS OFF",
                     T.teal if cfg["bt"].cost_bps + cfg["bt"].slippage_bps > 0 else T.coral),
             uc.chip(f"OOS LOCKED FROM {locked.date()}" if locked is not None else "NO HOLDOUT",
                     T.teal if locked is not None else T.coral),
             uc.chip(f"{nb} TRADES", T.teal if nb >= MIN_TRADES else T.amber),
             uc.chip(f"{years:.1f} YRS IN-SAMPLE", T.teal if years >= MIN_YEARS else T.amber),
             uc.chip("LOCAL · CAPS OFF" if is_local() else f"CLOUD · CAPS {caps().permutations} PERMS", T.violet)]
    return '<div style="margin:0.2rem 0 0.6rem 0;">' + "".join(chips) + "</div>"


def _step_rail(active: str) -> str:
    cells = []
    for i, s in enumerate(STEPS, 1):
        on = s == active
        live = s in LIVE_STEPS
        c = T.teal if on else (T.muted if live else T.faint)
        cells.append(f'<div style="flex:1 1 90px; border-top:2px solid {c if on else T.line}; padding:0.35rem 0.2rem;'
                     f' font-size:0.6rem; letter-spacing:0.16em; color:{c};">{i:02d} {html.escape(s)}'
                     f'{"" if live else " · soon"}</div>')
    return f'<div style="display:flex; flex-wrap:wrap; gap:0.3rem; margin:0 0 0.8rem 0;">{"".join(cells)}</div>'


def _fmt(k: str, v: float) -> str:
    if k in ("CAGR", "Vol", "MaxDD", "HitRate", "Ulcer"):
        return f"{v * 100:.1f}%"
    if k in ("PSR", "DSR"):
        return f"{v * 100:.0f}%"
    if k == "MaxDD_bars":
        return f"{int(v)}"
    return f"{v:.2f}"


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    st.markdown(_step_rail("BUILD"), unsafe_allow_html=True)

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        cfg = builder()

    with right:
        try:
            full = load(cfg["src"], cfg["name"], cfg["tf"], cfg["start_year"])
        except Exception as exc:  # noqa: BLE001
            st.markdown(uc.state_banner(T.coral, "NO DATA", f"{cfg['name']}: {exc}"), unsafe_allow_html=True)
            return
        ohlcv, locked = split_holdout(full, cfg["hold"])
        if len(ohlcv) < 120:
            st.markdown(uc.state_banner(T.amber, "TOO SHORT", f"only {len(ohlcv)} in-sample bars — widen the dates "
                                        "or lower the holdout"), unsafe_allow_html=True)
            return

        strat = REGISTRY[cfg["rule"]](**scale_params_for_tf(cfg["raw"], cfg["tf"]))
        res = run_single(strat.forecast(ohlcv), ohlcv, bt=cfg["bt"], risk=cfg["risk"], label=strat.describe())
        r, p = res.returns, res.periods
        b = _bench(ohlcv["close"], r)
        bstats = metrics.summary(b, periods=p)
        trades = extract_trades(res.position, ohlcv["close"])
        years = len(r) / p

        st.markdown(_rails(len(trades), years, cfg, locked), unsafe_allow_html=True)
        warn = []
        if len(trades) < MIN_TRADES:
            warn.append(f"{len(trades)} trades is thin evidence (want ≥{MIN_TRADES}).")
        if years < MIN_YEARS:
            warn.append(f"{years:.1f} years in-sample is short (want ≥{MIN_YEARS:.0f}).")
        if cfg["hold"] == 0:
            warn.append("No holdout: every number below has been seen, so none of it is out-of-sample.")
        if warn:
            st.markdown(uc.state_banner(T.amber, "CAUTION", " ".join(warn)), unsafe_allow_html=True)

        s = res.stats
        edge = s["Sharpe"] - bstats["Sharpe"]
        st.markdown(uk.kpis([
            {"label": "SHARPE", "value": f"{s['Sharpe']:.2f}", "sub": f"b&h {bstats['Sharpe']:.2f} · Δ {edge:+.2f}",
             "color": T.teal if edge > 0 else T.coral},
            {"label": "CAGR", "value": f"{s['CAGR'] * 100:.1f}%", "sub": f"b&h {bstats['CAGR'] * 100:.1f}% (vol-matched)"},
            {"label": "VOL", "value": f"{s['Vol'] * 100:.1f}%", "sub": f"target {cfg['bt'].target_vol * 100:.0f}%"},
            {"label": "MAX DD", "value": f"{s['MaxDD'] * 100:.1f}%", "sub": f"b&h {bstats['MaxDD'] * 100:.1f}%",
             "color": T.coral},
            {"label": "CALMAR", "value": f"{s['Calmar']:.2f}", "sub": f"b&h {bstats['Calmar']:.2f}"},
            {"label": "PSR", "value": f"{s['PSR'] * 100:.0f}%", "sub": "P(true Sharpe > 0)"},
            {"label": "TURNOVER", "value": f"{len(trades) / max(years, 1e-9):.1f}", "sub": "trades / yr"},
        ]), unsafe_allow_html=True)
        st.caption(f"{html.escape(res.label)} · {cfg['name']} · {cfg['tf'].lower()} · {p} bars/yr · "
                   f"{ohlcv.index[0].date()} → {ohlcv.index[-1].date()} in-sample")

        tabs = st.tabs(["EQUITY", "PRICE & TRADES", "ROLLING", "MONTHLY", "DISTRIBUTION", "COSTS", "STATS", "TRADES"])
        with tabs[0]:
            up.show(equity_fig(r, b, res.gross_returns), 460, "growth of 1 · log scale", key="bt_eq")
        with tabs[1]:
            up.show(price_fig(ohlcv, res, strat, trades), 480, "price, overlays, entries, forecast", key="bt_px")
        with tabs[2]:
            up.show(rolling_sharpe_fig(r, b, p), 320, "rolling 1-year sharpe", key="bt_roll")
        with tabs[3]:
            up.show(monthly_fig(r), max(260, 22 * len(set(r.index.year)) + 60), "monthly returns %", key="bt_mon")
        with tabs[4]:
            up.show(dist_fig(r, b), 320, "return distribution", key="bt_dist")
        with tabs[5]:
            cs = cost_sensitivity(strat, ohlcv, cfg["bt"], cfg["risk"])
            fig = go.Figure(go.Bar(x=cs["costs"], y=cs["Sharpe"], marker_color=[T.mint, T.teal, T.amber, T.coral],
                                   text=[f"{v:.2f}" for v in cs["Sharpe"]], textposition="outside"))
            fig.update_layout(hovermode="closest")
            up.show(fig, 300, "sharpe at 0× / 1× / 2× / 3× costs", key="bt_costs")
            drag = res.costs.sum() / years if years > 0 else 0.0
            st.caption(f"Cost drag ≈ {drag * 100:.2f}% a year. A rule whose Sharpe collapses at 2× costs has no "
                       "margin for real-world fills.")
        with tabs[6]:
            keys = list(s.keys())
            tbl = pd.DataFrame({"strategy": [_fmt(k, s[k]) for k in keys],
                                "buy & hold (vol-matched)": [_fmt(k, bstats.get(k, np.nan)) if k in bstats else "—"
                                                             for k in keys]}, index=keys)
            st.dataframe(tbl, use_container_width=True)
            st.caption("DSR deflates the Sharpe for the number of variants tried; phase 2 wires in your real "
                       "trial count. PSR/DSR are probabilities that the true Sharpe beats the bar.")
        with tabs[7]:
            st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)

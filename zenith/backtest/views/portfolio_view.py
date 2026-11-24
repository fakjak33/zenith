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
from ..portfolio_opt import METHODS, bucket_of, combine_with, sharpe_diff_ci, sharpe_vs_n
from ..spec import run_spec
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
    ins, _ = spec.split(C.load(spec))
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
    method = c2.selectbox("weights", list(METHODS), key="pf_w", format_func=lambda m: m.replace("_", " "),
                          help=" ".join(METHODS.values()))
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
    buckets = {nm: bucket_of(library.spec_of(store[nm])) for nm in df.columns}
    out = _combine(df, method, window, buckets)
    st.caption(METHODS[method])
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

    tabs = st.tabs(["EQUITY", "CORRELATION", "RISK", "WEIGHTS & IDM", "STATS", "SENSITIVITY", "METHODS vs 1/N",
                    "DIVERSIFICATION"])
    with tabs[5]:
        _sensitivity(df, method, window, periods)
    with tabs[6]:
        _methods_tab(df, window, periods, buckets)
    with tabs[7]:
        _diversification_tab(df, periods, buckets)
    with tabs[0]:
        wa, wb = C.window_pick(port.index, "pf_win", vm=False)
        if (wa, wb) != (port.index[0], port.index[-1]):
            C.window_kpis(C.clip(port, wa, wb), None, periods)
        fig = go.Figure()
        for i, (k, v) in enumerate(C.clip(df, wa, wb).items()):
            eq = (1 + v).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=k, line=dict(color=T.cat[(i + 1) % len(T.cat)], width=1),
                                     opacity=0.75))
        peq = (1 + C.clip(port, wa, wb)).cumprod()
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
        bt = st.text_input("compare against", "SPY", key="pf_divb").strip().upper() or "SPY"
        try:
            ub = C._market(bt)["close"].pct_change().reindex(port.index)
            C.diversify_panel(port, ub, periods, bt, "pf_div")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"couldn't load {bt}: {exc}")
    st.markdown(section("SAVED STRATEGIES", 0), unsafe_allow_html=True)
    for nm, rec in list(store.items()):
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{C.esc(nm)} · {rec['saved_at']} · hash {rec['hash']}")
        if c2.button("remove", key=f"pf_del_{rec['hash']}_{nm}"):
            library.delete(store, nm)
            st.rerun()


def vol_target_returns(r: pd.Series, target: float, lookback: int = 63, cap: float = 3.0) -> pd.Series:
    """Scale a return stream to ``target`` annual vol using trailing vol known the day before."""
    if not target:
        return r
    vol = r.rolling(lookback, min_periods=20).std().shift(1) * np.sqrt(252)
    lev = (target / vol).clip(upper=cap).fillna(1.0)
    return r * lev


def _sensitivity(df: pd.DataFrame, method: str, window: int, periods: int) -> None:
    from .sensitivity_panel import Knob, panel
    knobs = [Knob("window", "lookback (bars)", window, 21, 756, integer=True, log=True),
             Knob("method", "weights", method, choices=list(METHODS)),
             Knob("vol_target", "portfolio vol target (0 = off)", 0.0, 0.0, 0.30)]

    def run(vals):
        out = _combine(df, vals.get("method", method), int(vals.get("window", window)), {})
        r = vol_target_returns(out["returns"], float(vals.get("vol_target", 0.0)))
        return metrics.summary(r, periods=periods)

    panel(knobs, run, "pf_sens", ["Sharpe", "CAGR", "MaxDD", "Calmar", "Vol"],
          state=f"{list(df.columns)}|{len(df)}",
          note="Sweeps how the book is combined. The portfolio vol target scales the whole book to a target "
               "annual vol using the trailing 63-bar vol (capped at 3x).")


def _combine(df: pd.DataFrame, method: str, window: int, buckets: dict) -> dict:
    """``portfolio.combine`` for equal / inverse-vol, the Phase 5 optimisers otherwise (same output keys)."""
    if method in ("equal", "inverse_vol"):
        return combine(df, method, window)
    out = combine_with(df, method, window, buckets=buckets)
    rets = df.dropna(how="any")
    cov = rets.cov().to_numpy()
    wl = out["weights"].iloc[-1].to_numpy()
    tot = float(wl @ cov @ wl)
    rc = (wl * (cov @ wl)) / tot if tot > 0 else np.full(len(wl), np.nan)
    out.update({"corr": rets.corr(), "risk_contrib": pd.Series(rc, index=rets.columns),
                "div_ratio": float((wl * rets.std().to_numpy()).sum() / np.sqrt(tot)) if tot > 0 else np.nan,
                "components": rets})
    return out


def _methods_tab(df: pd.DataFrame, window: int, periods: int, buckets: dict) -> None:
    st.markdown(section("EVERY WEIGHTING METHOD vs PLAIN 1/N — walk-forward", 2,
                        help="Each method sets its weights from past data only, rebalancing monthly. The question "
                             "is not which looks best, but whether any beats equal weighting by more than "
                             "luck (the bootstrap interval)."), unsafe_allow_html=True)
    key = f"pfm|{list(df.columns)}|{len(df)}|{window}"
    if st.button(f"COMPARE · {len(METHODS)} methods", key="pf_mrun", type="primary"):
        res = {m: _combine(df, m, window, buckets)["returns"] for m in METHODS}
        rows = []
        for m, r in res.items():
            d, lo, hi = sharpe_diff_ci(r, res["equal"], periods) if m != "equal" else (0.0, 0.0, 0.0)
            s = metrics.summary(r, periods=periods)
            rows.append({"method": m, "Sharpe": s["Sharpe"], "vs 1/N": d, "lo": lo, "hi": hi, "MaxDD": s["MaxDD"],
                         "CAGR": s["CAGR"]})
        C.memo()[key] = {"rets": res, "table": pd.DataFrame(rows)}
    out = C.memo().get(key)
    if out is None:
        return
    t = out["table"]
    fig = go.Figure(go.Bar(x=t["method"], y=t["vs 1/N"],
                           marker_color=[T.faint if m == "equal" else T.teal if lo > 0 else T.coral if hi < 0
                                         else T.amber for m, lo, hi in zip(t["method"], t["lo"], t["hi"])],
                           error_y=dict(type="data", symmetric=False, array=(t["hi"] - t["vs 1/N"]).clip(lower=0),
                                        arrayminus=(t["vs 1/N"] - t["lo"]).clip(lower=0), color=T.muted)))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text="Sharpe minus 1/N's (95% CI)")
    up.show(fig, 320, "does any optimiser beat equal weight? · green = yes beyond luck", key="pf_m_bar")
    fig = go.Figure()
    for i, (m, r) in enumerate(out["rets"].items()):
        eq = (1 + r).cumprod()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=m, line=dict(width=2.2 if m == "equal" else 1.1,
                                                                    color=T.text if m == "equal" else
                                                                    T.cat[i % len(T.cat)])))
    fig.update_yaxes(type="log", title_text="growth of 1")
    up.show(fig, 360, "every method's book, walk-forward", key="pf_m_eq")
    show = t.copy()
    show["Sharpe"] = show["Sharpe"].map(lambda v: f"{v:.2f}")
    show["vs 1/N"] = [f"{d:+.2f} ({lo:+.2f} … {hi:+.2f})" if m != "equal" else "—"
                      for m, d, lo, hi in zip(t["method"], t["vs 1/N"], t["lo"], t["hi"])]
    show["MaxDD"] = show["MaxDD"].map(lambda v: f"{v * 100:.1f}%")
    show["CAGR"] = show["CAGR"].map(lambda v: f"{v * 100:.1f}%")
    st.dataframe(show.drop(columns=["lo", "hi"]), use_container_width=True, hide_index=True)
    st.caption("Usually nothing beats 1/N reliably once estimation error is counted (DeMiguel, Garlappi & Uppal "
               "2009) — risk-based methods (ERC, HRP) tend to cut drawdowns rather than raise Sharpe.")


def _diversification_tab(df: pd.DataFrame, periods: int, buckets: dict) -> None:
    sv = sharpe_vs_n(df, periods)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sv["N"], y=sv["theory"], name="theory: avg SR x sqrt(N / (1 + (N-1) avg corr))",
                             line=dict(color=T.faint, dash="dot")))
    fig.add_trace(go.Scatter(x=sv["N"], y=sv["actual"], name="your strategies (random subsets)",
                             mode="lines+markers", line=dict(color=T.teal, width=2)))
    fig.update_xaxes(title_text="number of strategies", dtick=1, showspikes=False)
    fig.update_yaxes(title_text="equal-weight Sharpe")
    up.show(fig, 320, "Sharpe vs number of strategies", key="pf_svn")
    st.caption("Diversification gains flatten fast when strategies are correlated: SharpeEdge notes most of the "
               "benefit comes from the first 10-15 genuinely different strategies.")
    bt = pd.DataFrame({"strategy": list(buckets), "bucket": list(buckets.values())})
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.caption("Buckets are set before looking at results (trend / reversion / other) — SharpeEdge's defence "
               "against overfitting the allocation. Pick 'buckets' in WEIGHTS to use them.")
    n = df.shape[1]
    if n >= 2:
        rc = df.rolling(126, min_periods=63).corr()
        avg = rc.groupby(level=0).apply(lambda m: (m.to_numpy().sum() - n) / (n * (n - 1)) if m.notna().all().all()
                                        else np.nan)
        book = df.mean(axis=1)
        vol = book.rolling(63, min_periods=21).std()
        stressed = vol > vol.expanding(min_periods=252).quantile(0.8)
        fig = go.Figure(go.Scatter(x=avg.index, y=avg, line=dict(color=T.violet, width=1.4), name="avg correlation"))
        from .regime_panel import ribbon
        lab = stressed.map({True: "stressed", False: "normal"}).where(vol.notna() & vol.expanding().count().ge(252))
        ribbon(fig, lab, ("normal", "stressed"), legend=False)
        fig.update_yaxes(title_text="avg pairwise correlation (126 bars)")
        up.show(fig, 300, "do your strategies move together when it matters? · shaded = book in its top-20% vol",
                key="pf_rcorr")

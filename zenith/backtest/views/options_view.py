"""BACKTEST / OPTIONS — systematic option strategies on synthetic chains (MODELED).

Calibrated against Cboe's published PUT and BXM indices; the calibration grade
is shown with every result. Same guard rails as LAB: costs on, locked holdout,
one logged reveal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import evidence_rating, key_findings, section
from .. import ledger as lg
from ..attrib import core as A
from ..engine import metrics
from ..options import data as D
from ..options import engine as E
from . import common as C

_FINDINGS = [
    {"stat": "Implied volatility has exceeded subsequently realised volatility most of the time — the variance risk "
             "premium that option sellers collect.", "cite": "Carr & Wu (2009)"},
    {"stat": "Cboe's PUT index matched the S&P 500's return with about two-thirds of its volatility over 1986–2018.",
     "cite": "Bondarenko (2019)"},
    {"stat": "Short-option returns are negatively skewed: steady gains, rare large losses. Sharpe flatters them.",
     "cite": "Israelov & Nielsen (2015)"},
]


def _strat_from_ui() -> tuple[str, int, E.OptionStrategy, int]:
    st.markdown(section("01 · UNDERLYING", 0), unsafe_allow_html=True)
    und = st.selectbox("underlying", list(D.UNDERLYINGS), key="op_und",
                       help="SPX uses the full VIX term structure and is the calibrated case. Others use their own Cboe "
                            "vol index where one exists, else VIX scaled by realised vol.")
    start = st.number_input("from", 1990 if und == "SPX" else 2005, 2022, 2007, key="op_start")
    st.markdown(section("02 · STRATEGY", 1), unsafe_allow_html=True)
    tpl = st.selectbox("template", list(E.TEMPLATES), key="op_tpl")
    base = E.TEMPLATES[tpl]
    legs = []
    for i, L in enumerate(base.legs):
        lbl = f"{'short' if L.side < 0 else 'long'} {L.kind}"
        if L.rule == "delta":
            v = st.slider(f"{lbl} · delta", 0.02, 0.50, float(L.value), 0.01, key=f"op_leg{tpl}{i}")
        elif L.rule == "pct":
            v = st.slider(f"{lbl} · strike % of spot", 0.70, 1.30, float(L.value), 0.01, key=f"op_leg{tpl}{i}")
        else:
            st.caption(f"{lbl} · at the money")
            v = L.value
        legs.append(replace(L, value=v))
    c1, c2 = st.columns(2)
    dte = c1.slider("days to expiry", 7, 120, base.dte, 1, key="op_dte")
    roll = c2.radio("cycle", ["monthly", "dte"], key="op_roll", horizontal=True,
                    format_func=lambda x: {"monthly": "3rd Friday", "dte": "rolling DTE"}[x])
    st.markdown(section("03 · MANAGEMENT", 2), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    tp = c1.slider("take profit at % of credit", 0, 90, 0, 5, key="op_tp") / 100
    sl = c2.slider("stop at × credit lost", 0.0, 5.0, 0.0, 0.5, key="op_sl")
    minv = st.slider("only open when VIX ≥", 0, 40, 0, 1, key="op_minv")
    st.markdown(section("04 · COSTS & HOLDOUT", 4), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    spread = c1.slider("half-spread % of premium", 0.0, 15.0, 3.0, 0.5, key="op_spr") / 100
    ratio = c2.slider("ATM / VIX ratio", 0.70, 1.00, E.ATM_RATIO, 0.01, key="op_ratio",
                      help=f"Calibrated {E.ATM_RATIO} on Cboe PUT. Changing it changes every price — keep the "
                           "calibrated value unless you are stress-testing the model.")
    hold = st.slider("locked out-of-sample %", 0, 50, 30, 5, key="op_hold")
    strat = replace(base, legs=legs, dte=dte, roll=roll, take_profit=tp, stop_mult=sl, min_vix=float(minv),
                    spread_pct=spread, atm_ratio=ratio)
    return und, int(start), strat, int(hold)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _inputs(und: str, start: int) -> dict:
    return D.load(und, start)


def _run(und: str, start: int, strat: E.OptionStrategy) -> E.OptionResult:
    return _run_cached(und, start, json.dumps(asdict(strat), sort_keys=True, default=str))


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=64)
def _run_cached(und: str, start: int, strat_json: str) -> E.OptionResult:
    d = json.loads(strat_json)
    d["legs"] = [E.Leg(**L) for L in d["legs"]]
    strat = E.OptionStrategy(**d)
    x = _inputs(und, start)
    return E.simulate(strat, x["S"], x["term"], x["skew"], x["rf"], x["stock_tr"], x["vol_scale"], x["q"])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _calibration(ratio: float) -> dict:
    out = {}
    for tpl, idx in (("Short put (PUT-style)", "PUT"), ("Covered call (BXM-style)", "BXM")):
        res = _run("SPX", 2007, replace(E.TEMPLATES[tpl], atm_ratio=ratio))
        cal = E.calibrate(res.returns, D.published(idx))
        cal.pop("months", None)
        out[idx] = cal
    return out


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        und, start, strat, hold = _strat_from_ui()
    with right:
        try:
            x = _inputs(und, start)
            with st.spinner("pricing synthetic chains"):
                res = _run(und, start, strat)
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", f"{und}: {exc}")
            return
        idx = res.returns.index
        cut = int(len(idx) * (1 - hold / 100)) if hold else len(idx)
        locked = idx[cut] if hold else None
        family = f"Options:{und}:{start}:{hold}"
        h = json.dumps(asdict(strat), sort_keys=True, default=str)
        import hashlib
        sh = hashlib.sha1(h.encode()).hexdigest()[:12]
        led = C.ledger()
        n_trials = lg.record(led, family, sh)
        rev = lg.revealed(led, family)
        show_all = bool(rev)
        end = len(idx) if show_all else cut
        r = res.returns.iloc[:end]
        p = 252
        tr = x["stock_tr"].reindex(r.index).ffill().pct_change().fillna(0.0)

        cal = _calibration(strat.atm_ratio)
        tiers = [E.calibration_tier(cal[k]) for k in ("PUT", "BXM")]
        tier = max(t[0] for t in tiers)
        note = (f"PUT: model {cal['PUT'].get('cagr_model', np.nan) * 100:.1f}% vs Cboe "
                f"{cal['PUT'].get('cagr_actual', np.nan) * 100:.1f}%/yr, TE {cal['PUT'].get('tracking_error', np.nan) * 100:.1f}% · "
                f"BXM: {cal['BXM'].get('cagr_model', np.nan) * 100:.1f}% vs {cal['BXM'].get('cagr_actual', np.nan) * 100:.1f}%, "
                f"TE {cal['BXM'].get('tracking_error', np.nan) * 100:.1f}% (2007→). "
                + ("" if und == "SPX" else f"{und} is priced off {x['vol_source']}; only SPX is calibrated directly."))
        st.markdown(evidence_rating(tier if und == "SPX" else "C", "MODELED — calibration vs Cboe", note),
                    unsafe_allow_html=True)
        chips = [C.uc.chip("MODELED CHAINS", T.amber), C.uc.chip("HALF-SPREAD ON", T.teal),
                 C.uc.chip(f"{n_trials} TRIALS", T.teal if n_trials <= 20 else T.amber),
                 C.uc.chip(("OOS REVEALED " + rev["at"]) if rev else
                           (f"OOS LOCKED FROM {locked.date()}" if locked is not None else "NO HOLDOUT"),
                           T.violet if rev else T.teal if locked is not None else T.coral)]
        if rev and rev["hash"] != sh:
            chips.append(C.uc.chip("OOS CONTAMINATED", T.coral))
        st.markdown("".join(chips), unsafe_allow_html=True)

        s = metrics.summary(r, periods=p, n_trials=n_trials)
        bs = metrics.summary(tr, periods=p)
        ab = A.alpha_beta(r, tr, p)
        st.markdown(uk.kpis([
            {"label": "SHARPE", "value": f"{s['Sharpe']:.2f}", "sub": f"{und} {bs['Sharpe']:.2f}"},
            {"label": "CAGR", "value": f"{s['CAGR'] * 100:.1f}%", "sub": f"{und} {bs['CAGR'] * 100:.1f}%"},
            {"label": "VOL", "value": f"{s['Vol'] * 100:.1f}%", "sub": f"{und} {bs['Vol'] * 100:.1f}%"},
            {"label": "MAX DD", "value": f"{s['MaxDD'] * 100:.1f}%", "color": T.coral, "sub": f"{und} {bs['MaxDD'] * 100:.1f}%"},
            {"label": "SKEW", "value": f"{s['Skew']:.2f}", "color": T.coral if s["Skew"] < -0.5 else T.text,
             "sub": "daily returns"},
            {"label": "BETA", "value": f"{ab['beta']:.2f}", "sub": f"alpha {ab['alpha_ann'] * 100:+.1f}%/yr t {ab['alpha_t']:.1f}"},
        ]), unsafe_allow_html=True)
        if s["Skew"] < -1:
            C.banner(T.amber, "NEGATIVE SKEW", "returns come in small steady gains and rare large losses; Sharpe "
                     "overstates how comfortable this is — look at the drawdown and the worst months")

        tabs = st.tabs(["EQUITY", "EXPOSURE", "MONTHLY", "DISTRIBUTION", "SENSITIVITY", "TRADES", "CALIBRATION",
                        "OUT-OF-SAMPLE"])
        with tabs[0]:
            b = tr * (r.std() / tr.std()) if tr.std() > 0 else tr
            up.show(C.equity_fig(r, b, shade_from=locked if show_all else None), 440,
                    f"strategy vs {und} (vol-matched) · log", key="op_eq")
        with tabs[1]:
            ex = res.exposure.iloc[:end]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=ex.index, y=ex["delta"], name="net delta", line=dict(color=T.teal, width=1)), 1, 1)
            fig.add_trace(go.Scatter(x=ex.index, y=ex["vega_per_volpt"] * 100, name="vega (% per vol pt)",
                                     line=dict(color=T.violet, width=1)), 2, 1)
            fig.update_yaxes(title_text="delta / $", row=1, col=1)
            fig.update_yaxes(title_text="% per vol pt", row=2, col=1)
            up.show(fig, 380, "greeks per $ of equity", key="op_ex")
        with tabs[2]:
            up.show(C.monthly_fig(r), max(260, 22 * len(set(r.index.year)) + 60), "monthly returns %", key="op_mon")
        with tabs[3]:
            up.show(C.dist_fig(r, tr), 320, "daily return distribution", key="op_dist")
            worst = ((1 + r).resample("ME").prod() - 1).nsmallest(5)
            st.caption("Worst months: " + ", ".join(f"{d:%Y-%m} {v * 100:.1f}%" for d, v in worst.items()))
        with tabs[4]:
            _sensitivity(und, start, strat, cut)
        with tabs[5]:
            st.dataframe(res.trades.iloc[: max(1, len(res.trades))][::-1] if show_all else
                         res.trades[pd.to_datetime(res.trades["open"]) < (locked or idx[-1] + pd.Timedelta(days=1))][::-1],
                         use_container_width=True, hide_index=True)
        with tabs[6]:
            rows = [{"index": k, **{m: v for m, v in cal[k].items() if m != "ok"}} for k in cal]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("ATM vol = ratio × VIX-term-structure vol, smile from Cboe SKEW, S&P dividend yield from "
                       "^SP500TR vs ^GSPC. The ratio was fitted on PUT only; BXM is the out-of-sample check.")
        with tabs[7]:
            if locked is None:
                C.banner(T.coral, "NO HOLDOUT", "set a locked out-of-sample % to keep an honest test")
            elif not rev:
                st.checkbox("I'm done tuning this options strategy", key="op_sure")
                if st.button("REVEAL OUT-OF-SAMPLE · ONCE", type="primary", key="op_reveal",
                             disabled=not st.session_state.get("op_sure")):
                    lg.reveal(led, family, sh, h)
                    st.rerun()
            else:
                ri, ro = res.returns.iloc[:cut], res.returns.iloc[cut:]
                si, so = metrics.sharpe(ri, p), metrics.sharpe(ro, p)
                st.markdown(uk.kpis([{"label": "IN-SAMPLE SHARPE", "value": f"{si:.2f}"},
                                     {"label": "OUT-OF-SAMPLE SHARPE", "value": f"{so:.2f}",
                                      "color": T.teal if so > 0 else T.coral},
                                     {"label": "OOS MAX DD", "value": f"{metrics.max_drawdown(ro) * 100:.1f}%",
                                      "color": T.coral}]), unsafe_allow_html=True)


def _sensitivity(und: str, start: int, strat: E.OptionStrategy, cut: int) -> None:
    di = [i for i, L in enumerate(strat.legs) if L.rule == "delta"]
    if not di:
        st.caption("Sensitivity sweeps the short-leg delta and DTE; pick a delta-based template.")
        return
    deltas = [0.10, 0.16, 0.20, 0.25, 0.30, 0.40]
    dtes = [14, 30, 45, 60, 90]
    key = f"opsens|{json.dumps(asdict(strat), sort_keys=True, default=str)}|{und}|{start}"
    if st.button(f"RUN DELTA × DTE GRID · {len(deltas) * len(dtes)} backtests", key="op_srun", type="primary"):
        grid = pd.DataFrame(index=dtes, columns=deltas, dtype=float)
        prog = C.progress_bar("pricing grid")
        n = 0
        for d in dtes:
            for de in deltas:
                legs = [replace(L, value=de) if i == di[0] else L for i, L in enumerate(strat.legs)]
                rr = _run(und, start, replace(strat, legs=legs, dte=d, roll="dte")).returns.iloc[:cut]
                grid.loc[d, de] = metrics.sharpe(rr, 252)
                n += 1
                prog(n / grid.size)
        C.memo()[key] = grid
    grid = C.memo().get(key)
    if grid is None:
        st.caption("Sweeps the first delta leg against days-to-expiry (rolling-DTE cycle), in-sample only.")
        return
    z = grid.to_numpy(float)
    fig = go.Figure(go.Heatmap(z=z, x=[f"{d:.2f}Δ" for d in grid.columns], y=[f"{d}d" for d in grid.index],
                               colorscale=up.DIVERGING, zmid=0, text=np.round(z, 2), texttemplate="%{text}",
                               xgap=1, ygap=1, colorbar=dict(thickness=10)))
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    fig.update_layout(hovermode="closest")
    up.show(fig, 360, "in-sample sharpe · delta × days to expiry", key="op_grid")

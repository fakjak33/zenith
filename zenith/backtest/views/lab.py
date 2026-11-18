"""BACKTEST / LAB — build a strategy by clicking, then walk it through the
research steps with the guard rails on.

DEFINE → BUILD → SENSITIVITY → ROBUSTNESS → FILTERS → ATTRIBUTION → OUT-OF-SAMPLE

Every step sees only the in-sample slice. The holdout is revealed once, at
the end, and the reveal is logged; changing the strategy afterwards marks the
out-of-sample result as contaminated. Every distinct variant you run is a
trial and feeds the deflated Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import key_findings, section
from .. import ledger as lg
from ..engine import metrics
from ..engine.trades import extract_trades
from ..spec import StrategySpec, run_spec, split_holdout
from . import common as C
from . import steps

STEPS = ["DEFINE", "BUILD", "SENSITIVITY", "ROBUSTNESS", "FILTERS", "ATTRIBUTION", "OUT-OF-SAMPLE"]
GOALS = ["Absolute return", "Beat buy-and-hold", "Crisis alpha / diversifier", "Income / carry"]

_FINDINGS = [
    {"stat": "Fit as little as possible; judge a rule by how it behaves across a whole range of "
             "parameters, not at one tuned point.", "cite": "Carver, Systematic Trading (2015)"},
    {"stat": "Testing many variants inflates the best Sharpe you will find by luck alone — the deflated "
             "Sharpe ratio corrects for the number of trials.", "cite": "Bailey & López de Prado (2014)"},
    {"stat": "Re-running the full optimisation on permuted data gives a p-value for the whole tuning "
             "process, not just the final rule.", "cite": "Masters, Permutation & Randomization Tests (2020)"},
    {"stat": "Most published anomalies shrink sharply out of sample and after publication.",
     "cite": "McLean & Pontiff (2016)"},
]


# ------------------------------------------------------------- steps ----
def step_define(spec: StrategySpec) -> None:
    st.markdown(section("DEFINE — write down why this should work before you look", 0), unsafe_allow_html=True)
    # widget keys are dropped when a step isn't rendered, so the text lives in its own key
    def _keep(src: str, dst: str):
        st.session_state[dst] = st.session_state.get(src, "")

    st.text_area("hypothesis", value=st.session_state.get("bt_notes", ""), key="bt_notes_w", height=110,
                 on_change=_keep, args=("bt_notes_w", "bt_notes"),
                 placeholder="e.g. Trends in equity indices persist for months because investors under-react to "
                             "news and then herd. A slow EWMAC should capture that, net of costs, without "
                             "relying on any one lookback.",
                 help="Saved with the strategy but not part of its hash: rewording isn't a new trial.")
    g = st.session_state.get("bt_goal", GOALS[0])
    st.selectbox("goal", GOALS, index=GOALS.index(g), key="bt_goal_w", on_change=_keep, args=("bt_goal_w", "bt_goal"))
    st.caption("Writing the economic reason first is the cheapest overfitting defence there is: if you can't say "
               "why the edge exists, a good backtest is more likely luck than signal. Benchmark auto-suggestion "
               "and the verdict card arrive with ATTRIBUTION.")


def step_build(spec: StrategySpec, ins: pd.DataFrame, res, b: pd.Series, trades: pd.DataFrame) -> None:
    r, p = res.returns, res.periods
    bstats = metrics.summary(b, periods=p)
    s = res.stats
    years = len(r) / p
    edge = s["Sharpe"] - bstats["Sharpe"]
    st.markdown(uk.kpis([
        {"label": "SHARPE", "value": f"{s['Sharpe']:.2f}", "sub": f"b&h {bstats['Sharpe']:.2f} · Δ {edge:+.2f}",
         "color": T.teal if edge > 0 else T.coral},
        {"label": "DEFLATED SR", "value": f"{s['DSR'] * 100:.0f}%", "sub": "P(real) after trials",
         "color": T.teal if s["DSR"] > 0.9 else T.amber if s["DSR"] > 0.5 else T.coral},
        {"label": "CAGR", "value": f"{s['CAGR'] * 100:.1f}%", "sub": f"b&h {bstats['CAGR'] * 100:.1f}% (vol-matched)"},
        {"label": "VOL", "value": f"{s['Vol'] * 100:.1f}%", "sub": f"target {spec.target_vol * 100:.0f}%"},
        {"label": "MAX DD", "value": f"{s['MaxDD'] * 100:.1f}%", "sub": f"b&h {bstats['MaxDD'] * 100:.1f}%",
         "color": T.coral},
        {"label": "CALMAR", "value": f"{s['Calmar']:.2f}", "sub": f"b&h {bstats['Calmar']:.2f}"},
        {"label": "TRADES / YR", "value": f"{len(trades) / max(years, 1e-9):.1f}", "sub": f"{len(trades)} total"},
    ]), unsafe_allow_html=True)
    st.caption(f"{C.esc(res.label)} · {spec.name} · {spec.tf.lower()} · {p} bars/yr · "
               f"{ins.index[0].date()} → {ins.index[-1].date()} in-sample")

    tabs = st.tabs(["EQUITY", "PRICE & TRADES", "ROLLING", "MONTHLY", "DISTRIBUTION", "COSTS", "STATS", "TRADES"])
    with tabs[0]:
        up.show(C.equity_fig(r, b, res.gross_returns), 460, "growth of 1 · log scale", key="bt_eq")
    with tabs[1]:
        up.show(C.price_fig(ins, res, spec.strategy(), trades), 480, "price, overlays, entries, forecast",
                key="bt_px")
    with tabs[2]:
        up.show(C.rolling_sharpe_fig(r, b, p), 320, "rolling 1-year sharpe", key="bt_roll")
    with tabs[3]:
        up.show(C.monthly_fig(r), max(260, 22 * len(set(r.index.year)) + 60), "monthly returns %", key="bt_mon")
    with tabs[4]:
        up.show(C.dist_fig(r, b), 320, "return distribution", key="bt_dist")
    with tabs[5]:
        rows = []
        for k in (0, 1, 2, 3):
            rk = res if k == 1 else run_spec(spec, ins, C.load_aux(spec), cost_mult=k)
            rows.append({"costs": f"{k}×", "Sharpe": rk.stats["Sharpe"]})
        cs = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=cs["costs"], y=cs["Sharpe"], marker_color=[T.mint, T.teal, T.amber, T.coral],
                               text=[f"{v:.2f}" for v in cs["Sharpe"]], textposition="outside"))
        fig.update_layout(hovermode="closest")
        up.show(fig, 300, "sharpe at 0× / 1× / 2× / 3× costs", key="bt_costs")
        drag = res.costs.sum() / years if years > 0 else 0.0
        st.caption(f"Cost drag ≈ {drag * 100:.2f}% a year. A rule whose Sharpe collapses at 2× costs has no "
                   "margin for real-world fills.")
    with tabs[6]:
        keys = list(s.keys())
        tbl = pd.DataFrame({"strategy": [C.fmt(k, s[k]) for k in keys],
                            "buy & hold (vol-matched)": [C.fmt(k, bstats[k]) if k in bstats else "—" for k in keys]},
                           index=keys)
        st.dataframe(tbl, use_container_width=True)
        st.caption("DSR deflates the Sharpe by the trial count shown in the rail above. PSR/DSR are the "
                   "probability that the true Sharpe beats the bar.")
    with tabs[7]:
        st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)


def _checklist(spec: StrategySpec) -> list[tuple[str, bool]]:
    h = spec.hash()
    m = C.memo()
    ran = lambda prefix: any(k.startswith(f"{prefix}|{h}") for k in m)  # noqa: E731
    return [("hypothesis written", bool(spec.notes.strip())), ("sensitivity sweep", ran("sens")),
            ("permutation test", ran("perm")), ("noise test", ran("noise")), ("walk-forward", ran("wf")),
            ("filter ablation", ran("abl") or not spec.filters)]


def step_oos(spec: StrategySpec, full: pd.DataFrame, locked, aux: dict, n_trials: int) -> None:
    st.markdown(section("OUT-OF-SAMPLE — one look, logged", 6), unsafe_allow_html=True)
    if locked is None:
        C.banner(T.coral, "NO HOLDOUT", "set a locked out-of-sample % in 05 · COSTS & HOLDOUT — with 0% there is "
                 "nothing unseen to test on")
        return
    led = C.ledger()
    fam = spec.family()
    rev = lg.revealed(led, fam)
    if not rev:
        items = _checklist(spec)
        done = sum(ok for _, ok in items)
        st.markdown("".join(
            f'<div style="font-size:0.8rem; color:{T.text if ok else T.muted}; margin:0.15rem 0;">'
            f'<span style="color:{T.teal if ok else T.faint};">{"■" if ok else "□"}</span>&nbsp; {C.esc(lbl)}</div>'
            for lbl, ok in items), unsafe_allow_html=True)
        C.banner(T.teal if done == len(items) else T.amber, f"{done}/{len(items)} DONE",
                 f"the holdout from {locked.date()} has never been shown. You get one look for this data + holdout; "
                 f"after that any change to the strategy is flagged.")
        sure = st.checkbox("I'm done tuning this strategy", key="bt_sure")
        if st.button("REVEAL OUT-OF-SAMPLE · ONCE", type="primary", disabled=not sure, key="bt_reveal"):
            lg.reveal(led, fam, spec.hash(), spec.to_json())
            st.rerun()
        return

    contaminated = rev["hash"] != spec.hash()
    shown = StrategySpec.from_json(rev["spec"]) if contaminated else spec
    if contaminated:
        C.banner(T.coral, "CONTAMINATED", f"you changed the strategy after the reveal on {rev['at']}. The numbers "
                 "below are for the strategy you revealed, not the current one — the current one has no clean "
                 "out-of-sample test left on this data.")
    else:
        C.banner(T.violet, "REVEALED", f"{rev['at']} · {rev['trials_at_reveal']} trials before the reveal")

    res = run_spec(shown, full, aux, n_trials=n_trials)          # causal: the IS part is identical to BUILD
    r, p = res.returns, res.periods
    ris, roos = r[r.index < locked], r[r.index >= locked]
    b = C.bench(full["close"], r)
    sis, soos = metrics.summary(ris, periods=p, n_trials=n_trials), metrics.summary(roos, periods=p)
    bo = metrics.summary(b[b.index >= locked], periods=p)
    deg = soos["Sharpe"] / sis["Sharpe"] if sis["Sharpe"] > 1e-9 else np.nan
    st.markdown(uk.kpis([
        {"label": "IN-SAMPLE SHARPE", "value": f"{sis['Sharpe']:.2f}"},
        {"label": "OUT-OF-SAMPLE SHARPE", "value": f"{soos['Sharpe']:.2f}",
         "color": T.teal if soos["Sharpe"] > 0 else T.coral, "sub": f"b&h {bo['Sharpe']:.2f}"},
        {"label": "KEPT", "value": "—" if not np.isfinite(deg) else f"{deg * 100:.0f}%",
         "color": T.teal if np.isfinite(deg) and deg > 0.5 else T.amber if np.isfinite(deg) and deg > 0 else T.coral,
         "sub": "OOS ÷ IS sharpe"},
        {"label": "OOS MAX DD", "value": f"{soos['MaxDD'] * 100:.1f}%", "color": T.coral,
         "sub": f"IS {sis['MaxDD'] * 100:.1f}%"},
        {"label": "OOS CAGR", "value": f"{soos['CAGR'] * 100:.1f}%", "sub": f"IS {sis['CAGR'] * 100:.1f}%"},
    ]), unsafe_allow_html=True)
    up.show(C.equity_fig(r, b, shade_from=locked), 460, "full history · shaded = out-of-sample", key="bt_oos_eq")
    keys = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "HitRate", "Skew", "PSR"]
    st.dataframe(pd.DataFrame({"in-sample": [C.fmt(k, sis[k]) for k in keys],
                               "out-of-sample": [C.fmt(k, soos[k]) for k in keys],
                               "b&h out-of-sample": [C.fmt(k, bo[k]) for k in keys]}, index=keys),
                 use_container_width=True)
    st.caption("Expect the out-of-sample Sharpe to be lower — roughly half is common for a fairly-tuned rule. "
               "A collapse to zero or below says the in-sample result was mostly the tuning.")


# --------------------------------------------------------------- page ----
def render() -> None:
    C.apply_pending()
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    step = st.segmented_control("step", STEPS, default="BUILD", key="bt_step",
                                format_func=lambda s: f"{STEPS.index(s) + 1:02d} {s}",
                                label_visibility="collapsed") or "BUILD"

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        spec = C.builder()

    with right:
        if step == "DEFINE":
            step_define(spec)
            return
        try:
            full = C.load(spec)
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", f"{spec.name}: {exc}")
            return
        ins, locked = split_holdout(full, spec.holdout)
        if len(ins) < 120:
            C.banner(T.amber, "TOO SHORT", f"only {len(ins)} in-sample bars — widen the dates or lower the holdout")
            return

        led = C.ledger()
        n_trials = lg.record(led, spec.family(), spec.hash())
        aux = C.load_aux(spec)
        res = run_spec(spec, ins, aux, n_trials=n_trials)
        trades = extract_trades(res.position, ins["close"])
        b = C.bench(ins["close"], res.returns)
        years = len(res.returns) / res.periods

        st.markdown(C.rails(spec, len(trades), years, locked, n_trials), unsafe_allow_html=True)
        warn = []
        if len(trades) < C.MIN_TRADES:
            warn.append(f"{len(trades)} trades is thin evidence (want ≥{C.MIN_TRADES}).")
        if years < C.MIN_YEARS:
            warn.append(f"{years:.1f} years in-sample is short (want ≥{C.MIN_YEARS:.0f}).")
        if spec.holdout == 0:
            warn.append("No holdout: every number below has been seen, so none of it is out-of-sample.")
        if any(f["key"] == "vix" for f in spec.filters) and aux.get("vix") is None:
            warn.append("VIX filter has no data here (synthetic or download failed) and is passing everything.")
        if warn and step in ("BUILD", "SENSITIVITY"):
            C.banner(T.amber, "CAUTION", " ".join(warn))

        if step == "BUILD":
            step_build(spec, ins, res, b, trades)
        elif step == "SENSITIVITY":
            steps.sensitivity(spec, ins, aux)
        elif step == "ROBUSTNESS":
            steps.robustness(spec, ins, aux, res)
        elif step == "FILTERS":
            steps.filters(spec, ins, aux)
        elif step == "ATTRIBUTION":
            C.banner(T.violet, "PHASE 3", "beta / alpha split, up- vs down-market capture, returns by regime, "
                     "factor regression and the verdict card land in the next release")
        elif step == "OUT-OF-SAMPLE":
            step_oos(spec, full, locked, aux, n_trials)

"""BACKTEST / ML LAB — machine learning for trading, without code and without
fooling yourself.

Pick features from the indicator library, a target, a model and a
time-ordered validation scheme; every prediction shown is out-of-sample (made
by a model that only saw the past). The scorecard leads with the trading
result and a naive baseline, not accuracy alone. Extras: CV-scheme comparison
(MemLabs), online learning across regimes (MemLabs), meta-labeling
(SharpeEdge), decomposed returns (MemLabs). A model can be saved as a normal
strategy for LIBRARY / PORTFOLIO / LAB.
"""

from __future__ import annotations

import datetime as _dt

import hashlib
import json

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
from .. import ledger as lg
from ..data.providers import TIMEFRAMES
from ..indicators import CATALOG, FAMILIES
from ..ml.core import (CV_SCHEMES, MAPPINGS, MODELS, TARGETS, build_features, calibration, coefficients,
                       fwd_return, hac_table, make_target, perm_importance, scorecard, to_position, walk_forward)
from ..ml.strategy import DEFAULT
from ..spec import StrategySpec, run_spec, with_aux
from . import common as C

_FINDINGS = [
    {"stat": "Out of sample, almost every return predictor that looked good in-sample loses to the plain "
             "historical average.", "cite": "Welch & Goyal (2008)"},
    {"stat": "Machine learning helps most through non-linear interactions — and only with careful, "
             "time-ordered validation.", "cite": "Gu, Kelly & Xiu (2020)"},
    {"stat": "Shuffled k-fold leaks the future in time series; purge and embargo the training set around "
             "every test block.", "cite": "López de Prado (2018)"},
    {"stat": "Judge a model by the money its predictions make after costs, not by accuracy or MSE.",
     "cite": "MemLabs · Quant Trading Accelerator"},
]
IND_DEFAULT = DEFAULT["features"]["indicators"]
BARS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}


def _controls() -> tuple[StrategySpec, dict, dict]:
    st.markdown(section("01 · DATA", 0), unsafe_allow_html=True)
    src = st.radio("source", ["Market", "Synthetic"], horizontal=True, key="ml_src", label_visibility="collapsed")
    if src == "Market":
        name = st.selectbox("instrument", C.QUICK, key="ml_name")
    else:
        name = st.selectbox("synthetic market", ["mean_reverting", "trending", "regime", "gbm", "fat_tailed"],
                            key="ml_synth", help="Known answers: models should find something on "
                                                 "'mean_reverting' and nothing on 'gbm' (a random walk).")
    tf = st.selectbox("bars", list(TIMEFRAMES), key="ml_tf")
    start_d, end_d = C.date_range("ml", _dt.date(2005, 1, 1))
    hold = C.lock_input("ml")
    st.caption("The locked years stay hidden, like LAB: everything here uses the in-sample part only. Save the "
               "model and reveal it once in LAB › OUT-OF-SAMPLE.")

    st.markdown(section("02 · FEATURES", 1, help="What the model sees at each bar. MemLabs: predict returns, not "
                        "prices; use few lags; another asset's lags can beat the asset's own."),
                unsafe_allow_html=True)
    fam = st.selectbox("indicator family", ["all", *FAMILIES], key="ml_fam",
                       format_func=lambda f: "ALL FAMILIES" if f == "all" else FAMILIES[f])
    opts = [k for k, i in CATALOG.items() if fam == "all" or i.family == fam]
    cur = [k for k in st.session_state.get("ml_ind", IND_DEFAULT)]
    opts = list(dict.fromkeys(cur + opts))
    if "ml_ind" not in st.session_state:
        st.session_state["ml_ind"] = IND_DEFAULT
    inds = st.multiselect("indicators", opts, key="ml_ind", format_func=lambda k: CATALOG[k].label,
                          help="Each becomes one input column (default settings). Price-level lines enter as % "
                               "distance from price.")
    c1, c2 = st.columns(2)
    lags = int(c1.slider("own return lags", 0, 10, 3, key="ml_lags", help="Last k bar returns (AR terms)."))
    memory = int(c2.slider("memory window", 0, 120, 0, 10, key="ml_mem",
                           help="MemLabs' regime 'memory': the rolling mean of recent returns, so a linear model "
                                "can switch between momentum and reversion. 0 = off."))
    cal = st.toggle("weekday dummies", key="ml_cal", help="Seasonality features (beware data-snooping).")
    ref_lags, ref = 0, "SPY"
    if src == "Market":
        c1, c2 = st.columns(2)
        ref_lags = int(c1.slider("reference lags", 0, 5, 0, key="ml_rl",
                                 help="Lagged returns of another ticker, plus the spread — MemLabs' cross returns."))
        ref = c2.text_input("reference ticker", "SPY", key="ml_ref").strip().upper() or "SPY"

    st.markdown(section("03 · TARGET & MODEL", 2), unsafe_allow_html=True)
    target = st.selectbox("predict", list(TARGETS), format_func=lambda t: TARGETS[t], key="ml_tgt")
    c1, c2 = st.columns(2)
    h = int(c1.selectbox("horizon h (bars)", [1, 2, 5, 10, 21], key="ml_h"))
    k = float(c2.number_input("barrier k·σ", 0.25, 5.0, 1.0, 0.25, key="ml_k", disabled=target != "triple_barrier",
                              help="Triple barrier: up/down barriers at ±k × recent volatility × √h."))
    task = "reg" if target == "return" else "clf"
    mopts = [m for m, s in MODELS.items() if s.task in (task, "both")]
    model = st.selectbox("model", mopts, format_func=lambda m: MODELS[m].label, key=f"ml_model_{task}",
                         help="Hover each choice's description below.")
    st.caption(MODELS[model].help)
    Cc = float(st.select_slider("regularisation", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], 1.0, key="ml_C",
                                help="Smaller = simpler model (stronger shrinkage). Linear models only."))
    scheme = st.selectbox("validation", list(CV_SCHEMES), format_func=lambda s: s.replace("_", " "), key="ml_cv",
                          help=" ".join(CV_SCHEMES.values()))
    st.caption(CV_SCHEMES[scheme])
    c1, c2, c3 = st.columns(3)
    folds = int(c1.number_input("folds", 2, 20, 6, key="ml_folds"))
    min_train = int(c2.number_input("min train bars", 100, 5000, 750, 50, key="ml_mt"))
    train_len = int(c3.number_input("rolling train", 100, 5000, 1000, 50, key="ml_tl",
                                    disabled=scheme != "rolling"))
    embargo = int(st.slider("embargo (bars)", 0, 20, 0, key="ml_emb",
                            help="Extra bars dropped from training after each test block (purged k-fold)."))

    st.markdown(section("04 · TRADING", 4), unsafe_allow_html=True)
    mapping = st.selectbox("prediction → position", list(MAPPINGS), key="ml_map", help=" ".join(MAPPINGS.values()))
    st.caption(MAPPINGS[mapping])
    c1, c2 = st.columns(2)
    hi = float(c1.slider("go long above", 0.50, 0.80, 0.55, 0.01, key="ml_hi", disabled=mapping != "threshold"))
    lo = float(c2.slider("go short below", 0.20, 0.50, 0.45, 0.01, key="ml_lo", disabled=mapping != "threshold"))
    long_only = st.toggle("long only", key="ml_lo_only")
    cost = float(st.number_input("cost bps per side", 0.0, 50.0, 2.5, 0.5, key="ml_cost"))

    spec = StrategySpec(src=src, name=name, tf=tf, start_year=start_d.year, holdout=0, lock_years=float(hold),
                        start_date=C._iso_start(start_d), end_date=C._iso_end(end_d), ref_ticker=ref,
                        cost_bps=cost, slippage_bps=0.0)
    feats = {"indicators": inds, "lags": lags, "memory": memory, "calendar": bool(cal), "ref_lags": ref_lags}
    cfg = {"features": feats, "target": target, "h": h, "k": k, "model": model, "C": Cc, "retrain": 126,
           "min_train": min_train, "window": "rolling" if scheme == "rolling" else "expanding",
           "train_len": train_len, "mapping": mapping, "hi": hi, "lo": lo, "long_only": bool(long_only)}
    run = {"scheme": scheme, "folds": folds, "embargo": embargo}
    spec = StrategySpec.from_dict({**spec.to_dict(), "mode": "ml", "ml": cfg})
    return spec, cfg, run


def _frame(spec: StrategySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = C.load(spec)
    ins, _ = spec.split(full)
    aux = C.load_aux(spec)
    return ins, with_aux(ins, aux, spec.needs())


def _compute(spec, cfg, run, ins_aux) -> dict:
    task = "reg" if cfg["target"] == "return" else "clf"
    X = build_features(ins_aux, cfg["features"], spec.tf)
    y = make_target(ins_aux, cfg["target"], cfg["h"], cfg["k"])
    fr = fwd_return(ins_aux, cfg["h"])
    if X.shape[1] == 0:
        return {"error": "pick at least one feature"}
    wf = walk_forward(X, y, cfg["model"], task, run["scheme"], run["folds"], cfg["h"], run["embargo"],
                      cfg["min_train"], cfg["train_len"], C=cfg["C"])
    if wf["pred"].notna().sum() < 50:
        return {"error": "not enough bars for this validation scheme — lower 'min train bars' or use more history"}
    pos = to_position(wf["pred"], task, cfg["mapping"], cfg["hi"], cfg["lo"], long_only=cfg["long_only"])
    # an h-bar target traded every bar overlaps; scale so the P&L is per bar
    sc = scorecard(wf["pred"], wf["y"], fr / cfg["h"], pos, task, wf["folds"], spec.cost_bps,
                   periods=BARS_PER_YEAR.get(spec.tf, 252))
    Xd = wf["X"]
    last = wf["folds"][-1] if wf["folds"] else None
    coefs = coefficients(wf["model"], list(Xd.columns)) if wf["model"] is not None else None
    imp = None
    if last is not None and wf["model"] is not None:
        te = Xd.index[last["te"]]
        imp = perm_importance(wf["model"], Xd.loc[te], wf["y"].loc[te], task)
    hac = hac_table(Xd, wf["y"], task)
    return {"task": task, "wf": wf, "pos": pos, "sc": sc, "coefs": coefs, "imp": imp, "hac": hac, "fr": fr,
            "X": X, "cal": calibration(wf["pred"], wf["y"]) if task == "clf" else pd.DataFrame()}


def render() -> None:
    C.apply_pending()
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        spec, cfg, run = _controls()
    with right:
        try:
            ins, ins_aux = _frame(spec)
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", f"{spec.name}: {exc}")
            return
        if len(ins) < 400:
            C.banner(T.amber, "TOO SHORT", f"only {len(ins)} in-sample bars")
            return
        miss = C.missing_aux(spec, C.load_aux(spec))
        if miss:
            C.banner(T.amber, "MISSING DATA", f"features need {', '.join(miss)} (market data only); they are empty.")
        tag = "ml:" + hashlib.sha1(json.dumps({**cfg, **run}, sort_keys=True).encode()).hexdigest()[:10]
        key = f"ml|{spec.family()}|{tag}"
        c1, c2 = st.columns([1, 2])
        if c1.button("TRAIN & TEST · walk-forward", type="primary", key="ml_run", use_container_width=True):
            with st.spinner("training on the past, predicting the next block, repeat…"):
                C.memo()[key] = _compute(spec, cfg, run, ins_aux)
            lg.record(C.ledger(), spec.family(), tag)
        n_trials = lg.trials(C.ledger(), spec.family())
        c2.caption(f"{len(ins)} in-sample bars · {ins.index[0].date()} → {ins.index[-1].date()} · "
                   f"{n_trials} trials on this data (each model configuration you test is one).")
        res = C.memo().get(key)
        if res is None:
            st.caption("Set things up on the left, then TRAIN & TEST. Every prediction you will see was made by a "
                       "model trained only on earlier bars.")
            _explainer()
            return
        if "error" in res:
            C.banner(T.amber, "CAN'T RUN", res["error"])
            return
        _results(spec, cfg, run, ins, ins_aux, res, n_trials)


def _explainer() -> None:
    with st.expander("HOW THIS PAGE WORKS — in five steps", expanded=True):
        st.markdown(
            "1. **Features**: numbers known at today's close (indicator readings, recent returns).\n"
            "2. **Target**: what you want to know about the future — will the next bar be up? by how much?\n"
            "3. **Model**: learns a rule mapping features → target on past data only.\n"
            "4. **Walk-forward**: train on the past, predict the next block, slide forward, repeat. Bars whose "
            "target overlaps the test block are removed from training (purging).\n"
            "5. **Trade it**: turn predictions into positions, subtract costs, and compare with doing nothing "
            "clever (the naive baseline). Accuracy above 50% is not enough — expected value after costs is what "
            "counts.")


def _results(spec, cfg, run, ins, ins_aux, res, n_trials) -> None:
    sc, wf, task = res["sc"], res["wf"], res["task"]
    lift = sc["accuracy"] - sc["naive"]
    st.markdown(uk.kpis([
        {"label": "HIT RATE", "value": f"{sc['accuracy'] * 100:.1f}%", "sub": f"naive {sc['naive'] * 100:.1f}%",
         "color": T.teal if lift > 0.01 else T.amber if lift > 0 else T.coral},
        {"label": "EDGE PER TRADE-BAR", "value": f"{sc['ev_bp']:+.1f} bp", "sub": "after costs",
         "color": T.teal if sc["ev_bp"] > 0 else T.coral},
        {"label": "NET SHARPE", "value": f"{sc['sharpe']:.2f}", "sub": f"exposure {sc['exposure'] * 100:.0f}%",
         "color": T.teal if sc["sharpe"] > 0.3 else T.amber if sc["sharpe"] > 0 else T.coral},
        {"label": "OOS BARS", "value": f"{sc['n']}", "sub": f"{len(wf['folds'])} folds"},
    ] + ([{"label": "OOS R²", "value": f"{sc['oos_r2'] * 100:.2f}%", "sub": "vs predicting zero mean",
           "color": T.teal if sc['oos_r2'] > 0 else T.coral}] if task == "reg" else [])), unsafe_allow_html=True)
    tabs = st.tabs(["EQUITY", "CALIBRATION", "WEIGHTS", "IMPORTANCE", "FEATURES", "CV SCHEMES", "ONLINE",
                    "META-LABEL", "DECOMPOSE", "SAVE"])
    with tabs[0]:
        net = sc["net"]
        C.vol_match_toggle("ml_eq")
        bh = C.match_vol(np.log(ins_aux["close"]).diff(), net)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=net.index, y=np.exp(bh.cumsum()), name=C.bench_label(),
                                 line=dict(color=T.amber, width=1.2)))
        fig.add_trace(go.Scatter(x=net.index, y=np.exp(net.cumsum()), name="model, net of costs",
                                 line=dict(color=T.teal, width=1.8)))
        for f in wf["folds"]:
            fig.add_vline(x=f["test_from"], line=dict(color=T.faint, width=1, dash="dot"))
        fig.update_yaxes(type="log", title_text="growth of 1")
        up.show(fig, 400, "out-of-sample equity · dotted = retrain points", key="ml_eq")
        ft = pd.DataFrame([{k: v for k, v in f.items() if k != "te"} for f in wf["folds"]])
        if not ft.empty:
            ft["fold Sharpe"] = [float(net.reindex(pd.date_range(a, b)).dropna().mean() /
                                       max(net.reindex(pd.date_range(a, b)).dropna().std(), 1e-12) * np.sqrt(252))
                                 for a, b in zip(ft["test_from"], ft["test_to"])]
            for c in ("train_from", "train_to", "test_from", "test_to"):
                ft[c] = pd.to_datetime(ft[c]).dt.date
            st.dataframe(ft.drop(columns=["base"]).round(2), use_container_width=True, hide_index=True)
            st.caption("A model that only works in one or two folds found a regime, not a rule.")
    with tabs[1]:
        if task != "clf" or res["cal"].empty:
            st.caption("Calibration applies to classification targets.")
        else:
            cal = res["cal"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.faint, dash="dot"),
                                     name="perfect"))
            fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers",
                                     marker=dict(size=7 + 10 * cal["n"] / cal["n"].max(), color=T.teal),
                                     line=dict(color=T.teal), name="model"))
            lo_, hi_ = float(cal[["predicted", "actual"]].min().min()) - 0.02, float(cal[["predicted", "actual"]].max().max()) + 0.02
            fig.update_xaxes(range=[lo_, hi_], title_text="predicted P(up)", showspikes=False)
            fig.update_yaxes(range=[lo_, hi_], title_text="actual share up")
            fig.update_layout(hovermode="closest")
            up.show(fig, 340, "calibration · do 60% calls come true 60% of the time?", key="ml_cal")
            p, y = wf["pred"].dropna(), wf["y"].reindex(wf["pred"].dropna().index)
            cm = pd.crosstab(np.where(p >= 0.5, "pred up", "pred down"), np.where(y >= 0.5, "was up", "was down"))
            st.dataframe(cm, use_container_width=True)
            st.caption("Real trading signals are usually close to 50/50: a calibrated model whose calls stay in "
                       "0.47-0.55 can still be profitable, an over-confident one is dangerous to size from.")
    with tabs[2]:
        cf = res["coefs"]
        if cf is None:
            st.caption("This model has no single set of weights (see IMPORTANCE).")
        else:
            cf = cf.sort_values()
            is_imp = MODELS[cfg["model"]].key in ("random_forest", "gbm")
            fig = go.Figure(go.Bar(x=cf.values, y=cf.index, orientation="h",
                                   marker_color=[T.teal if v > 0 else T.coral for v in cf.values]))
            fig.update_layout(hovermode="closest")
            up.show(fig, max(260, 22 * len(cf)), "feature importance (trees)" if is_imp
                    else "weights of the last fold's model (standardised features)", key="ml_coef")
            lag = [c for c in cf.index if c.startswith("ret_lag1")]
            if lag and not is_imp:
                w = cf[lag[0]]
                st.caption(f"Lag-1 weight {w:+.3f}: " + ("positive → the model bets on **momentum** (recent "
                           "moves continue)." if w > 0 else "negative → the model bets on **mean reversion** "
                           "(recent moves reverse)." ) + " (MemLabs)")
        if res["hac"] is not None:
            st.markdown(section("REGRESSION TABLE · Newey-West (HAC) errors", 2,
                                help="statsmodels fit on all in-sample bars. HAC standard errors allow for "
                                     "autocorrelated, heteroskedastic residuals — ordinary errors overstate "
                                     "significance on market data."), unsafe_allow_html=True)
            st.dataframe(res["hac"].round(4), use_container_width=True, column_config=uc.colcfg(
                ["t", "p"], {"t": "|t| > 2 is the usual line; Harvey, Liu & Zhu (2016) argue for 3 when many "
                                  "features were tried.", "p": "Chance of a weight this large if the true weight "
                                                               "were zero."}))
            st.caption("In-sample statistics: a significant weight here is a hypothesis, the walk-forward result "
                       "is the test.")
    with tabs[3]:
        imp = res["imp"]
        if imp is None or imp.empty:
            st.caption("Not available for this model.")
        else:
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                   marker_color=[T.teal if v > 0 else T.faint for v in imp.values]))
            fig.update_layout(hovermode="closest")
            up.show(fig, max(260, 22 * len(imp)), "permutation importance on the last test fold", key="ml_imp")
            st.caption("How much the out-of-sample score drops when one feature is scrambled. ≤ 0 = the model "
                       "would do as well without it.")
    with tabs[4]:
        X = res["X"].dropna()
        if X.shape[1] > 1:
            cm = X.corr()
            fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=list(cm.columns), y=list(cm.index), zmin=-1, zmax=1,
                                       colorscale=up.DIVERGING, showscale=False,
                                       hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showspikes=False)
            fig.update_xaxes(showspikes=False)
            fig.update_layout(hovermode="closest")
            up.show(fig, 380, "feature correlation · |ρ| > 0.8 = near duplicates", key="ml_fc")
            dup = [(a, b) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > 0.8]
            if dup:
                st.caption("Near-duplicates: " + ", ".join(f"{a} ~ {b}" for a, b in dup[:6]))
    with tabs[5]:
        _cv_tab(spec, cfg, ins_aux, res)
    with tabs[6]:
        _online_tab(ins_aux)
    with tabs[7]:
        _meta_tab(spec, ins, ins_aux, res)
    with tabs[8]:
        _decomp_tab(res)
    with tabs[9]:
        _save_tab(spec, cfg)


def _cv_tab(spec, cfg, ins_aux, res) -> None:
    st.caption("MemLabs: the same model can look great or useless depending on how you split time. Compare "
               "train fractions and window types; trust the configuration whose result is stable, not the best.")
    key = f"mlcv|{spec.family()}|{json.dumps(cfg, sort_keys=True)}"
    if st.button("COMPARE SCHEMES · 9 walk-forwards", key="ml_cvrun"):
        task = res["task"]
        X, y, fr = res["X"], make_target(ins_aux, cfg["target"], cfg["h"], cfg["k"]), fwd_return(ins_aux, cfg["h"])
        rows = []
        for frac in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            wf = walk_forward(X, y, cfg["model"], task, "split", 1, cfg["h"], frac=frac, C=cfg["C"])
            pos = to_position(wf["pred"], task, cfg["mapping"], cfg["hi"], cfg["lo"], long_only=cfg["long_only"])
            s = scorecard(wf["pred"], wf["y"], fr / cfg["h"], pos, task, wf["folds"], spec.cost_bps)
            rows.append({"scheme": f"split {int(frac * 100)}/{100 - int(frac * 100)}", "EV bp": s["ev_bp"],
                         "Sharpe": s["sharpe"], "hit": s["accuracy"]})
        for sch in ("expanding", "rolling", "purged_kfold"):
            wf = walk_forward(X, y, cfg["model"], task, sch, 6, cfg["h"], min_train=cfg["min_train"],
                              train_len=cfg["train_len"], C=cfg["C"])
            pos = to_position(wf["pred"], task, cfg["mapping"], cfg["hi"], cfg["lo"], long_only=cfg["long_only"])
            s = scorecard(wf["pred"], wf["y"], fr / cfg["h"], pos, task, wf["folds"], spec.cost_bps)
            rows.append({"scheme": sch.replace("_", " "), "EV bp": s["ev_bp"], "Sharpe": s["sharpe"],
                         "hit": s["accuracy"]})
        C.memo()[key] = pd.DataFrame(rows)
    t = C.memo().get(key)
    if t is None:
        return
    fig = go.Figure(go.Bar(x=t["scheme"], y=t["EV bp"], marker_color=[T.teal if v > 0 else T.coral for v in t["EV bp"]],
                           text=[f"{v:.2f}" for v in t["Sharpe"]], textposition="outside"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text="edge per trade-bar (bp, net)")
    up.show(fig, 320, "same model, different time splits · labels = net Sharpe", key="ml_cvbar")
    st.caption("Purged k-fold tests on blocks from the past too (with training data from both sides): it uses "
               "more data, but a fold trained on the future is only valid for checking stability, never for the "
               "trading record.")


def _online_tab(ins_aux) -> None:
    st.caption("MemLabs' fix for regime change: a linear model that updates its weights after every bar "
               "(stochastic gradient descent, partial_fit). Watch the lag-1 weight cross zero: above = momentum, "
               "below = mean reversion.")
    c1, c2 = st.columns(2)
    mem = int(c1.slider("memory feature", 0, 120, 40, 10, key="ml_on_mem"))
    lr_ = float(c2.select_slider("learning rate", [0.001, 0.003, 0.01, 0.03], 0.01, key="ml_on_lr",
                                 help="Higher = adapts faster, noisier."))
    key = f"mlon|{len(ins_aux)}|{mem}|{lr_}|{ins_aux.index[-1]}"
    if st.button("RUN ONLINE LEARNER", key="ml_on_run"):
        from ..ml.extras import online_weights
        C.memo()[key] = online_weights(ins_aux, lags=3, memory=mem, lr=lr_)
    w = C.memo().get(key)
    if w is None:
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
    for i, col in enumerate([c for c in w.columns if c != "prediction"]):
        fig.add_trace(go.Scatter(x=w.index, y=w[col].rolling(5).mean(), name=col,
                                 line=dict(width=1.4 if col == "lag1" else 1, color=T.cat[i % len(T.cat)])), 1, 1)
    fig.add_hline(y=0, line=dict(color=T.line), row=1, col=1)
    fig.add_trace(go.Scatter(x=ins_aux.index, y=ins_aux["close"], name="close", line=dict(color=T.text, width=1)),
                  2, 1)
    fig.update_yaxes(title_text="weight", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    up.show(fig, 440, "online weights through time", key="ml_on_w")
    pred = w["prediction"]
    r = np.log(ins_aux["close"]).diff().reindex(pred.index)
    ev = float((np.sign(pred) * r).mean() * 1e4)
    st.caption(f"Acting on its one-bar-ahead prediction (made before each bar): {ev:+.2f} bp per bar before costs. "
               f"Time with positive lag-1 weight: {(w['lag1'] > 0).mean() * 100:.0f}%.")


def _meta_tab(spec, ins, ins_aux, res) -> None:
    st.caption("SharpeEdge's meta-labeling: keep a simple base strategy for the *direction*; train a model only "
               "to decide which of its trades to *take*. Labels are each trade's outcome; features are the ones "
               "on the left, read at entry.")
    keys = C.classic_keys() + [f"i_{k}" for k in CATALOG]
    c1, c2, c3 = st.columns([1.6, 1, 1])
    base = c1.selectbox("base strategy", keys, index=keys.index("i_rsi2") if "i_rsi2" in keys else 0,
                        key="ml_meta_base", format_func=C.rule_label)
    model = c2.selectbox("filter model", ["logistic", "linear_svm", "random_forest", "gbm"], key="ml_meta_m",
                         format_func=lambda m: MODELS[m].label)
    thr = float(c3.slider("take if P(win) ≥", 0.40, 0.75, 0.55, 0.01, key="ml_meta_t"))
    key = f"mlmeta|{spec.family()}|{base}|{model}|{thr}|{tuple(res['X'].columns)}"
    if st.button("RUN META-LABELING", key="ml_meta_run"):
        from ..ml.extras import meta_label, trades_table
        bspec = StrategySpec.from_dict({**spec.to_dict(), "mode": "single", "rule": base, "params": {}, "ml": {}})
        bres = run_spec(bspec, ins, C.load_aux(spec))
        tt = trades_table(bres.position, bres.returns)
        out = meta_label(tt, res["X"], model, thr)
        C.memo()[key] = out
    out = C.memo().get(key)
    if out is None:
        return
    if not out or out["n_all"] < 10:
        st.caption("Too few base trades with features to learn from.")
        return
    st.markdown(uk.kpis([
        {"label": "TRADES TESTED", "value": f"{out['n_all']}", "sub": f"taken {out['n_taken']}"},
        {"label": "WIN RATE · ALL", "value": f"{out['win_all'] * 100:.1f}%",
         "sub": f"95% {out['ci_all'][0] * 100:.0f}–{out['ci_all'][1] * 100:.0f}% (Wilson)"},
        {"label": "WIN RATE · TAKEN", "value": f"{out['win_taken'] * 100:.1f}%",
         "sub": f"95% {out['ci_taken'][0] * 100:.0f}–{out['ci_taken'][1] * 100:.0f}%",
         "color": T.teal if out["win_taken"] > out["win_all"] else T.coral},
        {"label": "TAKEN vs SKIPPED p", "value": "—" if not np.isfinite(out["p"]) else f"{out['p']:.3f}",
         "sub": "two-proportion z-test", "color": T.teal if np.isfinite(out["p"]) and out["p"] < 0.05 else T.amber},
    ]), unsafe_allow_html=True)
    t = out["tested"]
    fig = go.Figure()
    for lab, m, col in (("taken", t["p_win"] >= thr, T.teal), ("skipped", t["p_win"] < thr, T.faint)):
        d = t[m]
        fig.add_trace(go.Scatter(x=d["entry"], y=d["ret"] * 100, mode="markers", name=lab,
                                 marker=dict(color=col, size=6)))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_yaxes(title_text="trade return %")
    fig.update_layout(hovermode="closest")
    up.show(fig, 320, "every out-of-sample base trade · taken vs skipped", key="ml_meta_sc")
    taken, skipped = t[t["p_win"] >= thr]["ret"], t[t["p_win"] < thr]["ret"]
    st.caption(f"Average trade: taken {taken.mean() * 100:+.2f}% · skipped {skipped.mean() * 100:+.2f}%. "
               "Layered models overfit in layers: trust this only if the taken trades beat the skipped ones and "
               "the CI's barely overlap.")


def _decomp_tab(res) -> None:
    st.caption("MemLabs: a return = its sign × its size. Learn the chance of an up move and the typical up and "
               "down sizes separately, then combine: E[r] = P(up)·size_up − (1 − P(up))·size_down.")
    key = f"mldec|{tuple(res['X'].columns)}|{len(res['X'])}|{res['X'].index[-1]}"
    if st.button("RUN DECOMPOSITION", key="ml_dec_run"):
        from ..ml.extras import decompose
        C.memo()[key] = decompose(res["X"], res["fr"])
    d = C.memo().get(key)
    if d is None:
        return
    st.markdown(uk.kpis([
        {"label": "UP SIZE vs DOWN SIZE", "value": f"{d['up_size'] * 1e4:.1f} / {d['down_size'] * 1e4:.1f} bp",
         "sub": f"gap p = {d['size_gap_p']:.2f} (permutation)"},
        {"label": "RMSE · DECOMPOSED", "value": f"{d['rmse_dec'] * 1e4:.2f} bp", "sub": f"direct {d['rmse_dir'] * 1e4:.2f}"
                                                                                    f" · zero {d['rmse_zero'] * 1e4:.2f}"},
        {"label": "EDGE · DECOMPOSED", "value": f"{d['ev_dec_bp']:+.2f} bp", "sub": f"direct {d['ev_dir_bp']:+.2f} bp"},
    ]), unsafe_allow_html=True)
    p = pd.Series(d["p_up"], index=d["index"]).dropna()
    e = pd.Series(d["e_dec"], index=d["index"]).dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=p.index, y=p, name="P(up)", line=dict(color=T.violet, width=1)), 1, 1)
    fig.add_hline(y=0.5, line=dict(color=T.line), row=1, col=1)
    fig.add_trace(go.Scatter(x=e.index, y=e * 1e4, name="E[r] bp", line=dict(color=T.teal, width=1)), 2, 1)
    fig.add_hline(y=0, line=dict(color=T.line), row=2, col=1)
    up.show(fig, 380, "the two halves of the forecast", key="ml_dec_ts")
    st.caption(f"Average drift {d['drift'] * 1e4:+.2f} bp/bar. If up and down sizes differ (low p), a 50% hit rate "
               "can still have a positive expectation — and a model of only the sign misses it.")


def _save_tab(spec, cfg) -> None:
    from .. import library
    st.caption("Saving makes this model a normal strategy: it re-trains on a schedule (every 126 bars, on past data "
               "only) inside LAB, PORTFOLIO and LIBRARY, and gets one locked out-of-sample reveal in LAB.")
    name = st.text_input("name", f"ML {MODELS[cfg['model']].label.split(' (')[0]} · {spec.name}", key="ml_save_name")
    if st.button("SAVE TO LIBRARY", key="ml_save"):
        library.save(C.saved(), name, spec, "ML LAB model")
        st.toast(f"saved '{name}'")
    st.caption("Saved models appear in LIBRARY (compare, export) and PORTFOLIO (combine with rules).")

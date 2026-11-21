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
from .. import glossary as G
from .. import ledger as lg
from ..config import FUTURES, caps, is_local
from ..data.providers import TIMEFRAMES, get_prices, resample_ohlcv
from ..data.synthetic import KIND_HELP, TICKER_KINDS, from_ticker, generate
from ..engine import diversify as dv
from ..engine import metrics
from ..indicators import CATALOG, FAMILIES
from ..rules import REGISTRY
from ..spec import FILTERS, SIZING_MODES, StrategySpec

QUICK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "^GSPC",
         "ES=F", "NQ=F", "ZN=F", "GC=F", "CL=F", "6E=F", "BTC-USD"]
SYNTH = ["trending", "mean_reverting", "regime", "fat_tailed", "gbm", *TICKER_KINDS]
MIN_TRADES, MIN_YEARS = 30, 5.0


# ------------------------------------------------------------------ data ----
@st.cache_data(ttl=3600, show_spinner=False)
def _market(ticker: str) -> pd.DataFrame:
    return get_prices(ticker, start=None)


@st.cache_data(ttl=3600, show_spinner=False)
def _synthetic(kind: str, seed: int, base: str) -> pd.DataFrame:
    if kind in TICKER_KINDS:
        return from_ticker(kind, _market(base), seed=seed)
    return generate(kind, n_days=5000, seed=seed)


def load(spec: StrategySpec, name: str | None = None, seed: int | None = None) -> pd.DataFrame:
    name = name or spec.name
    if spec.src == "Synthetic":
        df = _synthetic(name, int(spec.seed if seed is None else seed), spec.synth_base)
    else:
        df = _market(name)
    df = df[df.index >= pd.Timestamp(f"{spec.start_year}-01-01")]
    return resample_ohlcv(df, TIMEFRAMES[spec.tf]).dropna(subset=["close"])


def load_aux(spec: StrategySpec) -> dict:
    """Side series that filters need (VIX), loaded only when a filter uses them."""
    aux = {}
    needs = spec.needs()
    if spec.src != "Market":
        return aux
    if any(f["key"] == "vix" and f.get("on") for f in spec.filters) or "vix" in needs:
        try:
            aux["vix"] = _market("^VIX")["close"]
        except Exception:  # noqa: BLE001
            aux["vix"] = None
    if "vix3m" in needs:
        try:
            aux["vix3m"] = _market("^VIX3M")["close"]
        except Exception:  # noqa: BLE001
            aux["vix3m"] = None
    if "ref_close" in needs:
        try:
            aux["ref_close"] = _market(spec.ref_ticker)["close"]
        except Exception:  # noqa: BLE001
            aux["ref_close"] = None
    return aux


def missing_aux(spec: StrategySpec, aux: dict) -> list[str]:
    """Aux columns the rule needs but could not get (synthetic data or a failed download)."""
    return [c for c in spec.needs() if aux.get(c) is None]


# ---------------------------------------------------------------- saved ----
def saved() -> dict:
    if "bt_saved" not in st.session_state:
        from .. import library
        st.session_state["bt_saved"] = library.load_disk()
    return st.session_state["bt_saved"]


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


LIBS = ["CLASSIC RULES", "INDICATOR LIBRARY"]


def classic_keys() -> list[str]:
    return [k for k, c in REGISTRY.items() if not c.cross_sectional and not k.startswith("i_")]


def rule_label(k: str) -> str:
    c = REGISTRY[k]
    ind = getattr(c, "indicator", None)
    return f"{c.label}  ·  {FAMILIES[ind.family].split(' ')[0]}" if ind else c.label


def signal_picker(prefix: str = "bt") -> tuple[str, str]:
    """Library / family / rule pickers. Returns (rule key, reference ticker)."""
    cur = st.session_state.get(f"{prefix}_rule", "ema")
    if f"{prefix}_lib" not in st.session_state:
        st.session_state[f"{prefix}_lib"] = LIBS[1] if str(cur).startswith("i_") else LIBS[0]
    lib = st.radio("library", LIBS, horizontal=True, key=f"{prefix}_lib", label_visibility="collapsed")
    if lib == LIBS[1]:
        fams = ["all", *FAMILIES]
        fam = st.selectbox("family", fams, key=f"{prefix}_fam", format_func=lambda f: "ALL FAMILIES" if f == "all"
                           else FAMILIES[f], help="Indicator families. The rule box below is searchable: type to "
                                                  "filter.")
        keys = [f"i_{i.key}" for i in CATALOG.values() if fam == "all" or i.family == fam]
    else:
        keys = classic_keys()
    if st.session_state.get(f"{prefix}_rule") not in keys:
        st.session_state[f"{prefix}_rule"] = "ema" if "ema" in keys else keys[0]
    rule = st.selectbox("rule", keys, format_func=rule_label, key=f"{prefix}_rule",
                        help="Type to search. The GUIDE below explains the selected rule in plain words.")
    cls = REGISTRY[rule]
    if cls.plain:
        st.caption(cls.plain)
    ref = "SPY"
    if "ref_close" in (cls.needs or ()):
        ref = st.text_input("reference ticker", "SPY", key=f"{prefix}_ref",
                            help="The other asset this cross-asset indicator compares against or lags "
                                 "(market data only).").strip().upper() or "SPY"
    if "vix" in (cls.needs or ()):
        st.caption("Reads Cboe VIX (and VIX3M): market data only — flat on synthetic data.")
    return rule, ref


def guide(cls) -> None:
    """Plain-English guide for the selected rule (all explainer fields)."""
    e = cls.explain()
    with st.expander(f"GUIDE · {e['label']}"):
        if e.get("plain"):
            st.markdown(f"**What it is.** {e['plain']}")
        if e.get("analogy"):
            st.markdown(f"**{'Signal' if e.get('signal') else 'Analogy'}.** {e['analogy']}")
        steps = [x for x in e.get("how_it_works") or [] if x]
        if steps:
            st.markdown("**How it works.**\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1)))
        if e.get("works_when") and not e.get("signal"):
            st.markdown(f"**Works when.** {e['works_when']}")
        if e.get("fails_when"):
            st.markdown(f"**Fails when.** {e['fails_when']}")
        if e.get("evidence"):
            st.markdown(f"**Evidence.** {e['evidence']}")
        if e.get("video"):
            st.caption(f"Related video: {e['video']}")


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
                            help="Known-answer worlds: a trend rule should win on 'trending' and fail on 'gbm'. "
                                 "The last four learn from a real ticker.")
        st.caption(KIND_HELP.get(name, ""))
    seed, base = 42, "SPY"
    if src == "Synthetic":
        c1, c2 = st.columns(2)
        seed = int(c1.number_input("seed", 1, 1000, 42, 1, key="bt_seed",
                                   help="Same seed = same path. Try several, or run SEEDS in BUILD to test "
                                        "hundreds at once."))
        if name in TICKER_KINDS:
            base = c2.text_input("base ticker", "SPY", key="bt_base").strip().upper() or "SPY"
    fut = FUTURES.get(name.upper()) if src == "Market" else None
    if fut:
        st.caption(f"{fut[0]} · ${fut[1]:,.0f} per point · tick {fut[2]:g} · margin ≈ ${fut[3]:,.0f}/contract. "
                   "Sized in whole contracts.")
    c1, c2 = st.columns(2)
    tf = c1.selectbox("bars", list(TIMEFRAMES), key="bt_tf")
    start_year = c2.number_input("from", 1990, 2024, 2005, key="bt_start")

    st.markdown(section("02 · SIGNAL", 1, help="Pick what decides long / short. CLASSIC RULES are the curated "
                        "Carver-style strategies; the INDICATOR LIBRARY turns any of ~75 indicators into a rule. "
                        "Hover the ? on every control for what it does."), unsafe_allow_html=True)
    rule, ref = signal_picker()
    cls = REGISTRY[rule]
    params, cols = {}, st.columns(2)
    for i, (pname, spec_t) in enumerate(cls.params.items()):
        with cols[i % 2]:
            params[pname] = _num(pname, spec_t, param_key(rule, pname), help=G.param_help(cls, pname) or None)
    guide(cls)

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

    st.markdown(section("04 · SIZING & EXITS", 2, help="How big each position is. Pick ONE mode: only its "
                        "own controls are used, and the sentence under it says exactly what you are building."),
                unsafe_allow_html=True)
    mode = st.radio("sizing", list(SIZING_MODES), horizontal=True, key="bt_size",
                    format_func=lambda m: SIZING_MODES[m])
    fpct, risk, tvol, scale = 1.0, 0.01, 0.20, True
    stop_k_now = float(st.session_state.get("bt_stopk", 3.0))
    if mode == "fixed_pct":
        c1, c2 = st.columns([1.4, 1])
        fpct = c1.slider("notional % of capital", 1, 300, 100, 1, key="bt_fpct",
                         help="Market value of the position as a share of capital. 100% = fully invested, "
                              "no leverage; above 100% is leverage.") / 100
        scale = c2.toggle("scale by signal", False, key="bt_scale",
                          help="Off: every trade is the same size. On: size is proportional to forecast/10, so "
                               "a weak signal holds less and the strongest (forecast 20) holds double.")
        tail = (f" at a normal signal (forecast 10), up to {fpct * 200:.0f}% at the cap (20)." if scale
                else ", whatever the signal strength.")
        st.caption(f"Each trade holds **{fpct * 100:.0f}% of capital**{tail} No volatility adjustment: a quiet "
                   "and a wild market get the same dollars.")
    elif mode == "fixed_risk":
        c1, c2 = st.columns([1.4, 1])
        risk = c1.slider("risk % of capital per trade", 0.1, 20.0, 1.0, 0.1, key="bt_risk",
                         help="What you lose if the ATR stop is hit. 0.5-2% is the classic range; 10% "
                              "means a handful of straight stop-outs roughly halves the account.") / 100
        scale = c2.toggle("scale by signal", False, key="bt_scale",
                          help="Off: every trade risks the same amount. On: risk is proportional to forecast/10.")
        st.caption(f"Position = {risk * 100:.1f}% x capital / ({stop_k_now:g} x ATR). If price moves "
                   f"{stop_k_now:g} ATRs against you the stop closes the trade for a **{risk * 100:.1f}% loss** "
                   "(more on gaps). The ATR stop is switched on for you. Tight stops mean big positions: "
                   "the SIZING tab in BUILD shows the leverage this implies.")
        if risk >= 0.05:
            st.caption(f":orange[{risk * 100:.0f}% per trade is very aggressive: losing streaks of 5-10 trades "
                       "are normal for trend rules.]")
    else:
        tvol = st.slider("target vol %", 5, 60, 20, key="bt_tvol",
                         help="The annualised volatility this one position should run at. Carver: 10-25% for "
                              "a single strategy; above about half your Sharpe (as a %) is beyond half-Kelly.") / 100
        zone = ("conservative" if tvol < 0.10 else "typical (Carver uses ~20%)" if tvol <= 0.25
                else "aggressive" if tvol <= 0.40 else "very risky: deep drawdowns are near-certain")
        st.caption(f"Units = (forecast/10) x capital x {tvol * 100:.0f}% / (blended vol x price). At a normal "
                   f"signal the position swings about **{tvol * 100:.0f}% of capital a year**: {zone}. "
                   "Calmer market, bigger position. The SIZING tab in BUILD works today's numbers by hand.")
    c1, c2 = st.columns(2)
    direction = c1.selectbox("direction", ["both", "long", "short"], key="bt_dir")
    lev = float(c2.number_input("max leverage x", 0.0, 20.0, 0.0, 0.5, key="bt_lev",
                                help="Caps notional / capital. 0 = no cap. 1 = never borrow."))
    c1, c2 = st.columns(2)
    capital = c1.number_input("capital $", 10_000, 100_000_000, 1_000_000, 50_000, key="bt_cap",
                              help="Matters for futures: small accounts can't hold fractional contracts.")
    buffer = c2.slider("no-trade buffer %", 0, 30, 10 if fut else 0, 5, key="bt_buf",
                       help="Carver's buffer: only trade when the ideal position moves more than this share of an "
                            "average position away from what you hold. Cuts turnover and costs.") / 100
    c1, c2 = st.columns(2)
    if mode == "fixed_risk":
        st.session_state["bt_stop"] = True
    stop = c1.toggle("ATR stop", key="bt_stop", disabled=mode == "fixed_risk")
    tp = c2.toggle("ATR target", key="bt_tp")
    stop_k = c1.number_input("stop x ATR", 0.5, 10.0, 3.0, 0.5, key="bt_stopk", disabled=not stop)
    tp_k = c2.number_input("target x ATR", 0.5, 20.0, 6.0, 0.5, key="bt_tpk", disabled=not tp)

    st.markdown(section("05 · COSTS & HOLDOUT", 4), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cost = c1.number_input("commission bps", 0.0, 50.0, 1.0, 0.5, key="bt_cost")
    slip = c2.number_input("slippage bps", 0.0, 50.0, 1.5, 0.5, key="bt_slip")
    fee = st.number_input("$ per contract per side", 0.0, 20.0, 2.5, 0.25, key="bt_fee") if fut else 2.5
    hold = st.slider("locked out-of-sample %", 0, 50, 30, 5, key="bt_hold",
                     help="The most recent slice of history is hidden from every step until you reveal it "
                          "once, at the end. Setting 0 removes your only honest test.")
    return StrategySpec(src=src, name=name, tf=tf, start_year=int(start_year), rule=rule, params=params,
                        filters=filters, sizing_mode=mode, target_vol=tvol, direction=direction,
                        fixed_pct=float(fpct), risk_pct=float(risk), scale_by_forecast=bool(scale),
                        max_leverage=lev, seed=seed, synth_base=base,
                        use_atr_stop=stop, use_atr_tp=tp, atr_stop_mult=stop_k, atr_tp_mult=tp_k,
                        cost_bps=cost, slippage_bps=slip, holdout=int(hold), capital=float(capital),
                        buffer=float(buffer), contract_fee=float(fee), ref_ticker=ref,
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


def spec_to_widgets(spec: StrategySpec) -> dict:
    """LAB builder widget values that reproduce ``spec`` (used by LIBRARY › open in LAB)."""
    w = {"bt_src": spec.src, "bt_tf": spec.tf, "bt_start": int(spec.start_year), "bt_rule": spec.rule,
         "bt_size": spec.sizing_mode, "bt_tvol": int(round(spec.target_vol * 100)), "bt_dir": spec.direction,
         "bt_fpct": int(round(spec.fixed_pct * 100)), "bt_risk": float(round(spec.risk_pct * 100, 1)),
         "bt_scale": bool(spec.scale_by_forecast), "bt_lev": float(spec.max_leverage),
         "bt_seed": int(spec.seed), "bt_base": spec.synth_base,
         "bt_cap": int(spec.capital), "bt_buf": int(round(spec.buffer * 100)), "bt_stop": spec.use_atr_stop,
         "bt_tp": spec.use_atr_tp, "bt_stopk": float(spec.atr_stop_mult), "bt_tpk": float(spec.atr_tp_mult),
         "bt_cost": float(spec.cost_bps), "bt_slip": float(spec.slippage_bps), "bt_fee": float(spec.contract_fee),
         "bt_hold": int(spec.holdout), "bt_notes": spec.notes,
         "bt_lib": LIBS[1] if spec.rule.startswith("i_") else LIBS[0], "bt_fam": "all", "bt_ref": spec.ref_ticker}
    if spec.src == "Market":
        if spec.name in QUICK:
            w["bt_quick"] = spec.name
        else:
            w["bt_quick"], w["bt_custom"] = "custom…", spec.name
    else:
        w["bt_synth"] = spec.name
    for k, v in spec.params.items():
        w[param_key(spec.rule, k)] = v
    on = {f["key"]: f for f in spec.filters if f.get("on", True)}
    for fkey in FILTERS:
        w[f"bt_f_{fkey}"] = fkey in on
        for pn, pv in (on.get(fkey, {}).get("params") or {}).items():
            w[f"bt_fp_{fkey}_{pn}"] = pv
    return w


# ------------------------------------------------------- diversification ----
def diversify_panel(s_ret: pd.Series, u_ret: pd.Series, periods: int, u_label: str, key: str) -> None:
    """Correlation to the underlying, crisis behaviour, the add-to-portfolio test and the blend frontier."""
    from ... import ui_kit as uk
    d = dv.analyse(s_ret, u_ret, periods)
    st.markdown(section(f"DIVERSIFICATION vs {u_label}", 3,
                        help="Would this strategy improve a portfolio that already holds the underlying? "
                             "Low or negative correlation, especially in the worst months, is what you want."),
                unsafe_allow_html=True)
    if not d.get("ok"):
        st.caption("Not enough overlapping history to compare.")
        return
    cr = d["crisis"]
    st.markdown(uk.kpis([
        {"label": "CORRELATION", "value": f"{d['corr']:+.2f}", "sub": f"beta {d['beta']:+.2f}",
         "color": T.teal if abs(d["corr"]) < 0.3 else T.amber if abs(d["corr"]) < 0.7 else T.coral},
        {"label": "UP / DOWN MKT", "value": f"{d['corr_up']:+.2f} / {d['corr_down']:+.2f}",
         "sub": "corr when it rises / falls"},
        {"label": "CRISIS CORR", "value": "—" if not np.isfinite(cr["corr"]) else f"{cr['corr']:+.2f}",
         "sub": f"worst {cr['months']} months · strat {cr['s_mean'] * 100:+.1f}%/mo vs {cr['u_mean'] * 100:+.1f}%",
         "color": T.teal if cr["s_mean"] > 0 else T.coral},
        {"label": "ADDS TO SHARPE?", "value": "YES" if d["adds"] else "NO",
         "sub": f"SR {d['sharpe_s']:.2f} vs hurdle ρ·SR {d['hurdle']:.2f}",
         "color": T.teal if d["adds"] else T.coral},
        {"label": "ALPHA vs UNDERLYING", "value": f"{d['alpha'] * 100:+.1f}%/yr", "sub": "μ − β·μ_u"},
        {"label": "BEST BLEND", "value": f"{d['best_w'] * 100:.0f}% strat", "sub": f"of risk · Sharpe {d['best_sharpe']:.2f}"},
    ]), unsafe_allow_html=True)
    st.caption("**Add-to-portfolio rule:** a small allocation to the strategy raises the portfolio Sharpe exactly "
               f"when its Sharpe beats correlation x the portfolio's Sharpe ({d['corr']:+.2f} x "
               f"{d['sharpe_u']:.2f} = {d['hurdle']:.2f}). Equivalently its return beats beta x the underlying's "
               "return, i.e. it has positive alpha against what you already hold.")
    c1, c2 = st.columns(2)
    with c1:
        fr = d["frontier"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=fr["w"] * 100, y=fr["Sharpe"], name="Sharpe", line=dict(color=T.teal, width=2)))
        fig.add_trace(go.Scatter(x=fr["w"] * 100, y=fr["MaxDD"], name="max drawdown",
                                 line=dict(color=T.coral, width=1.4, dash="dot")), secondary_y=True)
        fig.add_vline(x=d["best_w"] * 100, line=dict(color=T.faint, width=1, dash="dash"))
        fig.update_xaxes(title_text=f"% of risk in strategy (rest in {u_label}), vol-matched")
        fig.update_yaxes(title_text="Sharpe", secondary_y=False)
        fig.update_yaxes(title_text="max DD", tickformat=".0%", secondary_y=True, showgrid=False)
        fig.update_layout(hovermode="x unified")
        up.show(fig, 300, "blend frontier · rebalanced each bar", key=f"{key}_fr")
    with c2:
        rc = dv.rolling_corr(s_ret, u_ret, max(periods, 12))
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color=T.line, width=1))
        fig.add_trace(go.Scatter(x=rc.index, y=rc, name="rolling 1y corr", line=dict(color=T.violet, width=1.4),
                                 fill="tozeroy", fillcolor=up.fill(T.violet, 0.12)))
        fig.update_yaxes(range=[-1, 1], title_text="correlation")
        up.show(fig, 300, "rolling 1-year correlation", key=f"{key}_rc")


# --------------------------------------------------------------- sizing ----
def sizing_panel(spec: StrategySpec, ins: pd.DataFrame, res) -> None:
    """Worked example of today's position and the leverage the sizing actually produced."""
    from ... import ui_kit as uk
    from ..core.volatility import blended_vol
    from ..risk.atr import atr
    close = ins["close"]
    p = res.periods
    pos = res.position.reindex(close.index).fillna(0.0)
    lev = (pos * close).abs() / spec.capital
    in_mkt = float((pos != 0).mean())
    f = float(res.forecast.dropna().iloc[-1]) if res.forecast.notna().any() else 0.0
    px = float(close.iloc[-1])
    st.markdown(section(f"SIZING · {SIZING_MODES.get(spec.sizing_mode, spec.sizing_mode).upper()}", 2),
                unsafe_allow_html=True)
    st.markdown(uk.kpis([
        {"label": "AVG EXPOSURE", "value": f"{lev[pos != 0].mean() * 100 if in_mkt else 0:.0f}%",
         "sub": "notional / capital while in a trade"},
        {"label": "MAX EXPOSURE", "value": f"{lev.max() * 100:.0f}%",
         "color": T.coral if lev.max() > 3 else T.amber if lev.max() > 1 else T.teal},
        {"label": "TIME IN MARKET", "value": f"{in_mkt * 100:.0f}%"},
        {"label": "REALISED VOL", "value": f"{res.stats['Vol'] * 100:.1f}%",
         "sub": f"target {spec.target_vol * 100:.0f}%" if spec.sizing_mode == "vol_target" else "not targeted"},
    ]), unsafe_allow_html=True)
    cap = spec.capital
    s_txt = f"forecast/10 = {f / 10:+.2f}" if (spec.scale_by_forecast or spec.sizing_mode == "vol_target") \
        else f"sign(forecast) = {np.sign(f):+.0f}"
    sc = f / 10 if (spec.scale_by_forecast or spec.sizing_mode == "vol_target") else float(np.sign(f))
    rows = [("capital", f"${cap:,.0f}"), ("price today", f"{px:,.2f}"), ("forecast today", f"{f:+.1f}"),
            ("signal scale", s_txt)]
    if spec.sizing_mode == "vol_target":
        bv = float(blended_vol(close, periods=p).iloc[-1])
        units = sc * cap * spec.target_vol / (bv * px)
        rows += [("blended vol (70% EWMA-32 + 30% 10y mean)", f"{bv * 100:.1f}%"),
                 ("risk budget = capital x target vol", f"${cap * spec.target_vol:,.0f}"),
                 ("units = scale x budget / (vol x price)", f"{units:,.1f}")]
    elif spec.sizing_mode == "fixed_risk":
        a = float(atr(ins).iloc[-1])
        dist = spec.atr_stop_mult * a
        units = sc * spec.risk_pct * cap / dist if dist > 0 else 0.0
        rows += [("ATR(14)", f"{a:,.2f}"), (f"stop distance = {spec.atr_stop_mult:g} x ATR", f"{dist:,.2f}"),
                 ("risk $ = risk % x capital", f"${spec.risk_pct * cap:,.0f}"),
                 ("units = scale x risk $ / stop distance", f"{units:,.1f}")]
    else:
        units = sc * spec.fixed_pct * cap / px
        rows += [("notional = fixed % x capital", f"${spec.fixed_pct * cap:,.0f}"),
                 ("units = scale x notional / price", f"{units:,.1f}")]
    if spec.max_leverage:
        cap_u = spec.max_leverage * cap / px
        units = float(np.clip(units, -cap_u, cap_u))
        rows.append((f"after {spec.max_leverage:g}x leverage cap", f"{units:,.1f}"))
    rows += [("notional", f"${units * px:,.0f}  ({abs(units * px) / cap * 100:.0f}% of capital)")]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Today's position, by hand**")
        st.dataframe(pd.DataFrame(rows, columns=["step", "value"]), hide_index=True, use_container_width=True)
        st.caption("Positions are sized on the account capital above, and the engine trades the next bar. "
                   "Buffer, whole-contract rounding and stops can move the held position away from this target.")
    with c2:
        fig = go.Figure(go.Scatter(x=lev.index, y=lev, fill="tozeroy", fillcolor=up.fill(T.cyan, 0.18),
                                   line=dict(color=T.cyan, width=1), name="exposure"))
        fig.add_hline(y=1, line=dict(color=T.amber, width=1, dash="dot"), annotation_text="100% (no leverage)",
                      annotation_font=dict(size=9, color=T.amber))
        fig.update_yaxes(tickformat=".0%", title_text="notional / capital")
        up.show(fig, 300, "exposure over time", key="bt_lev_fig")
    with st.expander("How the three sizing modes compare"):
        st.markdown(
            "| mode | you choose | position | best for |\n|---|---|---|---|\n"
            "| **fixed notional %** | how much money is in the trade | % x capital / price | simple, "
            "intuitive; ignores how volatile the market is |\n"
            "| **fixed risk % / trade** | how much you lose if the stop hits | risk $ / (k x ATR) | discretionary-"
            "style 'risk 1R per trade'; size shrinks when markets get wild |\n"
            "| **vol target** | how much the position should swing per year | (f/10) x capital x target / "
            "(vol x price) | Carver-style systematic trading; comparable risk across instruments and time |\n\n"
            "**Vol target ranges** (single strategy): under 10% conservative · 10-25% typical · 25-40% aggressive · "
            "over 40% very risky. The Kelly-optimal vol equals the Sharpe ratio (SR 0.5 means 50%); Carver "
            "recommends at most half of that because the backtest Sharpe overstates the future one.\n\n"
            "**To replicate a vol target by hand:** estimate annual vol (e.g. 20-day stdev x sqrt(252)), then "
            "dollars in the trade = capital x target / vol. With a 20% target and the market at 16% vol you hold "
            "125% of capital; at 40% vol you hold 50%.")


# ---------------------------------------------------------------- seeds ----
def seeds_panel(spec: StrategySpec, key: str = "bt_seeds") -> None:
    """Run the same strategy across many synthetic seeds: is the result robust or one lucky path?"""
    from ..robust.runner import pmap
    from ..spec import run_spec, split_holdout
    hi = 1000 if is_local() else 200
    c1, c2, c3 = st.columns([1, 1, 1])
    n = int(c1.number_input("seeds", 2, hi, min(50, hi), 1, key=f"{key}_n",
                            help=f"Up to {hi} here ({'local' if is_local() else 'cloud'} cap)."))
    first = int(c2.number_input("first seed", 1, 1000, 1, 1, key=f"{key}_0"))
    go_ = c3.button("RUN SEEDS", key=f"{key}_go", use_container_width=True)
    mk = f"seeds|{spec.hash()}|{first}|{n}"
    if go_:
        bar = progress_bar("running seeds")

        def one(sd):
            df, _ = split_holdout(load(spec, seed=sd), spec.holdout)
            r = run_spec(spec, df)
            return sd, r.stats, (1 + r.returns).cumprod()

        memo()[mk] = pmap(one, range(first, first + n), bar)
    out = memo().get(mk)
    if not out:
        st.caption("Each seed is a different path from the same synthetic world. A real edge should be positive "
                   "on most seeds; if only a few seeds work, the rule found noise.")
        return
    df = pd.DataFrame([{"seed": sd, **{k: stt[k] for k in ("Sharpe", "CAGR", "MaxDD", "Vol")}}
                       for sd, stt, _ in out])
    pos = float((df["Sharpe"] > 0).mean())
    from ... import ui_kit as uk
    st.markdown(uk.kpis([
        {"label": "SEEDS", "value": f"{len(df)}"},
        {"label": "MEDIAN SHARPE", "value": f"{df['Sharpe'].median():.2f}",
         "sub": f"5-95%: {df['Sharpe'].quantile(0.05):.2f} to {df['Sharpe'].quantile(0.95):.2f}"},
        {"label": "SHARPE > 0", "value": f"{pos * 100:.0f}%", "color": T.teal if pos > 0.8 else T.amber if pos > 0.5
         else T.coral},
        {"label": "MEDIAN CAGR", "value": f"{df['CAGR'].median() * 100:.1f}%"},
        {"label": "WORST MAX DD", "value": f"{df['MaxDD'].min() * 100:.1f}%", "color": T.coral},
    ]), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=df["Sharpe"], nbinsx=40, marker_color=up.fill(T.teal, 0.6),
                                     marker_line=dict(width=0)))
        fig.add_vline(x=0, line=dict(color=T.coral, width=1))
        fig.update_xaxes(title_text="Sharpe")
        fig.update_layout(hovermode="closest")
        up.show(fig, 300, "sharpe across seeds", key=f"{key}_h")
    with c2:
        eqs = pd.concat([e.reset_index(drop=True) for _, _, e in out], axis=1)
        q = eqs.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
        x = list(range(len(q)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=q[0.95], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=q[0.05], fill="tonexty", fillcolor=up.fill(T.teal, 0.12),
                                 line=dict(width=0), name="5-95%"))
        fig.add_trace(go.Scatter(x=x, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=q[0.25], fill="tonexty", fillcolor=up.fill(T.teal, 0.25),
                                 line=dict(width=0), name="25-75%"))
        fig.add_trace(go.Scatter(x=x, y=q[0.5], line=dict(color=T.teal, width=2), name="median"))
        fig.update_yaxes(type="log", title_text="growth of 1")
        fig.update_xaxes(title_text="bar")
        up.show(fig, 300, "equity fan across seeds", key=f"{key}_f")
    with st.expander("per-seed table"):
        st.dataframe(df.set_index("seed").style.format({"Sharpe": "{:.2f}", "CAGR": "{:.1%}", "MaxDD": "{:.1%}",
                                                        "Vol": "{:.1%}"}), use_container_width=True)


# --------------------------------------------------------------- window ----
WINDOWS = ["ALL", "1ST HALF", "2ND HALF", "LAST 10Y", "LAST 5Y", "LAST 3Y", "LAST 1Y", "CUSTOM"]


def window_pick(index: pd.DatetimeIndex, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Zoom window for every time-series chart below it. Charts re-slice and rebase to the window
    start, so the y axis always fits the period you picked (Plotly's own box-zoom can't refit y)."""
    lo, hi = pd.Timestamp(index[0]), pd.Timestamp(index[-1])
    c1, c2 = st.columns([3, 2])
    pick = c1.segmented_control("window", WINDOWS, default="ALL", key=f"{key}_w",
                                label_visibility="collapsed") or "ALL"
    if pick == "1ST HALF":
        a, b = lo, lo + (hi - lo) / 2
    elif pick == "2ND HALF":
        a, b = lo + (hi - lo) / 2, hi
    elif pick.startswith("LAST"):
        yrs = int(pick.split()[1].rstrip("Y"))
        a, b = max(lo, hi - pd.DateOffset(years=yrs)), hi
    elif pick == "CUSTOM":
        a, b = c2.slider("dates", lo.to_pydatetime(), hi.to_pydatetime(),
                         (lo.to_pydatetime(), hi.to_pydatetime()), format="YYYY-MM", key=f"{key}_rng",
                         label_visibility="collapsed")
        a, b = pd.Timestamp(a), pd.Timestamp(b)
    else:
        a, b = lo, hi
    if pick not in ("ALL", "CUSTOM"):
        c2.caption(f"{a.date()} → {b.date()} · charts rebased to the window start")
    return a, b


def clip(x, a: pd.Timestamp, b: pd.Timestamp):
    """Slice a Series/DataFrame to [a, b]."""
    return x[(x.index >= a) & (x.index <= b)]


def window_kpis(r: pd.Series, b: pd.Series | None, periods: int, label_b: str = "bench") -> None:
    """One-line stats for the zoom window."""
    if len(r) < 3:
        return
    s = metrics.summary(r, periods=periods)
    txt = (f"window · CAGR {s['CAGR'] * 100:.1f}% · Sharpe {s['Sharpe']:.2f} · max DD {s['MaxDD'] * 100:.1f}% "
           f"· vol {s['Vol'] * 100:.1f}%")
    if b is not None and len(b.dropna()) > 3:
        bs = metrics.summary(b.fillna(0), periods=periods)
        txt += f"  |  {label_b} CAGR {bs['CAGR'] * 100:.1f}% · Sharpe {bs['Sharpe']:.2f} · max DD {bs['MaxDD'] * 100:.1f}%"
    st.caption(txt)

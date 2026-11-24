"""Shared LAB plumbing: data loading, the block builder, the ledger handle,
result memo, guard-rail chips and the core result charts."""

from __future__ import annotations

import datetime as _dt
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
from ..config import FUTURES, caps, is_local
from ..data.providers import TIMEFRAMES, get_prices, resample_ohlcv
from ..data.synthetic import KIND_HELP, TICKER_KINDS, from_ticker, generate
from ..engine import diversify as dv
from ..engine import metrics
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
    df = df[df.index >= spec.start_ts()]
    end = spec.end_ts()
    if end is not None:
        df = df[df.index <= end]
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


def date_range(prefix: str, default_start: _dt.date, min_start: _dt.date = _dt.date(1970, 1, 1)
               ) -> tuple[_dt.date, _dt.date]:
    """Start / end date pickers shared by every lab. End defaults to today (= latest bar)."""
    today = _dt.date.today()
    c1, c2 = st.columns(2)
    a = c1.date_input("from", default_start, min_value=min_start, max_value=today, key=f"{prefix}_start",
                      format="YYYY-MM-DD",
                      help="First day of data used. Earlier data simply isn't loaded. Rules with long lookbacks "
                           "need a warm-up, so their first trades come later than this date.")
    b = c2.date_input("to", today, min_value=min_start, max_value=today, key=f"{prefix}_end",
                      format="YYYY-MM-DD",
                      help="Last day of data used. Leave at today to run to the latest available bar.")
    if b <= a:
        st.caption(":orange[end date is before the start date — using the start date + 1 year]")
        b = min(today, a + _dt.timedelta(days=365))
    return a, b


def _iso_start(d: _dt.date) -> str:
    """'' when the start is 1 January (then ``start_year`` alone describes it — keeps old hashes)."""
    return "" if (d.month, d.day) == (1, 1) else d.isoformat()


def _iso_end(d: _dt.date) -> str:
    """'' when the end is today, i.e. run to the latest bar."""
    return "" if d >= _dt.date.today() else d.isoformat()


def lock_input(prefix: str, default: float = 2.0) -> float:
    return float(st.number_input(
        "lock last N years (out-of-sample)", 0.0, 30.0, default, 0.5, key=f"{prefix}_lock",
        help="The most recent N years before the end date are hidden from every statistic and tuning step until "
             "you reveal them once, at the end. Charts still run to the end date, with the locked span shaded. "
             "Because it is counted back from the END date, moving the start date doesn't move the lock. "
             "0 = no holdout (nothing left to test on honestly)."))


def builder() -> StrategySpec:
    """Left-hand block builder → StrategySpec."""
    st.markdown(section("01 · UNIVERSE", 0, help="What the rule trades. Market = real Yahoo prices; Synthetic = "
                        "generated price paths with a known answer, to check a rule does what you think."),
                unsafe_allow_html=True)
    src = st.radio("source", ["Market", "Synthetic"], horizontal=True, key="bt_src",
                   label_visibility="collapsed")
    if src == "Market":
        pick = st.selectbox("instrument", QUICK + ["custom…"], key="bt_quick",
                            help="ETFs, indices (^GSPC), futures (=F, sized in whole contracts) or crypto. "
                                 "'custom…' takes any Yahoo ticker.")
        name = st.text_input("ticker", "AAPL", key="bt_custom",
                             help="Any Yahoo Finance symbol, e.g. AAPL, VTI, ^NDX, CL=F.").strip().upper()             if pick == "custom…" else pick
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
    tf = st.selectbox("bars", list(TIMEFRAMES), key="bt_tf",
                      help="Bar size the rule runs on. Weekly/monthly bars are built from daily data.")
    start_date, end_date = date_range("bt", _dt.date(2005, 1, 1))

    st.markdown(section("02 · SIGNAL", 1, help="The rule turns prices into a forecast between −20 and +20 "
                        "(+10 = a normal long). Sizing below turns the forecast into a position."),
                unsafe_allow_html=True)
    keys = [k for k, c in REGISTRY.items() if not c.cross_sectional]
    rule = st.selectbox("rule", keys, index=keys.index("ema") if "ema" in keys else 0,
                        format_func=lambda k: REGISTRY[k].label, key="bt_rule",
                        help="The trading rule. Its plain-English description appears below.")
    cls = REGISTRY[rule]
    if cls.plain:
        st.caption(cls.plain)
    params, cols = {}, st.columns(2)
    for i, (pname, spec_t) in enumerate(cls.params.items()):
        with cols[i % 2]:
            params[pname] = _num(pname, spec_t, param_key(rule, pname),
                                 help=f"Allowed {spec_t[1]:g}–{spec_t[2]:g}, default {spec_t[0]:g}. Lengths are in "
                                      "trading days and are rescaled automatically for weekly / monthly bars.")

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
                    format_func=lambda m: SIZING_MODES[m],
                    help="fixed notional % = dollars in the trade · fixed risk = dollars lost if the ATR stop hits · "
                         "vol target = how much the position swings per year. The SIZING tab in BUILD works an "
                         "example by hand.")
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
    direction = c1.selectbox("direction", ["both", "long", "short"], key="bt_dir",
                             help="both = follow the forecast either way · long = ignore short signals (go flat) · "
                                  "short = ignore long signals.")
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
    stop = c1.toggle("ATR stop", key="bt_stop", disabled=mode == "fixed_risk",
                     help="Exit when price moves k × ATR(14) against the position from its best level (a trailing "
                          "stop). Always on under fixed-risk sizing, which is sized from it.")
    tp = c2.toggle("ATR target", key="bt_tp", help="Exit when price moves k × ATR(14) in your favour from entry.")
    stop_k = c1.number_input("stop x ATR", 0.5, 10.0, 3.0, 0.5, key="bt_stopk", disabled=not stop)
    tp_k = c2.number_input("target x ATR", 0.5, 20.0, 6.0, 0.5, key="bt_tpk", disabled=not tp)

    st.markdown(section("05 · COSTS & HOLDOUT", 4, help="Trading costs are charged on every change in position. "
                        "The holdout keeps the most recent years unseen until the final, logged reveal."),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cost = c1.number_input("commission bps", 0.0, 50.0, 1.0, 0.5, key="bt_cost",
                           help="Charged on traded notional. 1 bp = 0.01%: $1 per $10,000 traded.")
    slip = c2.number_input("slippage bps", 0.0, 50.0, 1.5, 0.5, key="bt_slip",
                           help="Half the bid-ask spread plus market impact, per side. Liquid ETFs ≈ 1–2 bps.")
    fee = st.number_input("$ per contract per side", 0.0, 20.0, 2.5, 0.25, key="bt_fee",
                          help="Broker + exchange fee per futures contract, each way.") if fut else 2.5
    lock = lock_input("bt")
    return StrategySpec(src=src, name=name, tf=tf, start_year=start_date.year, rule=rule, params=params,
                        filters=filters, sizing_mode=mode, target_vol=tvol, direction=direction,
                        fixed_pct=float(fpct), risk_pct=float(risk), scale_by_forecast=bool(scale),
                        max_leverage=lev, seed=seed, synth_base=base,
                        use_atr_stop=stop, use_atr_tp=tp, atr_stop_mult=stop_k, atr_tp_mult=tp_k,
                        cost_bps=cost, slippage_bps=slip, holdout=0, lock_years=float(lock),
                        start_date=_iso_start(start_date), end_date=_iso_end(end_date),
                        capital=float(capital), buffer=float(buffer), contract_fee=float(fee),
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


# ------------------------------------------------------------ vol match ----
VM_KEY = "zen_volmatch"
VM_HELP = ("ON: the benchmark's daily returns are scaled so its volatility equals the strategy's over the period "
           "shown — a like-for-like test of skill at the same risk (Sharpe is unchanged; CAGR, drawdown and the "
           "equity line change). OFF: the plain, unscaled underlying, i.e. what simply holding it returned. "
           "The setting is shared by every lab.")


def vol_match_on() -> bool:
    """Shared across labs (a plain session key, not part of any spec, so flipping it is never a new trial)."""
    return bool(st.session_state.get(VM_KEY, True))


def vol_match_toggle(key: str, where=None) -> bool:
    k = f"{key}_vm"
    if k not in st.session_state:
        st.session_state[k] = vol_match_on()

    def _keep():
        st.session_state[VM_KEY] = bool(st.session_state[k])

    (where or st).toggle("vol-matched", key=k, on_change=_keep, help=VM_HELP)
    return vol_match_on()


def match_vol(b: pd.Series, r: pd.Series, on: bool | None = None) -> pd.Series:
    """``b`` scaled to ``r``'s realised vol when vol-matching is on (else returned as-is)."""
    on = vol_match_on() if on is None else on
    b = b.reindex(r.index).fillna(0.0)
    sv, bv = r.std(), b.std()
    return b * (sv / bv) if on and bv > 1e-12 and sv > 1e-12 else b


def bench_label(base: str = "buy & hold", on: bool | None = None) -> str:
    on = vol_match_on() if on is None else on
    return f"{base} (vol-matched)" if on else base


# -------------------------------------------------------------- charts ----
def bench(close: pd.Series, strat: pd.Series, on: bool | None = None) -> pd.Series:
    """Buy-and-hold of the same instrument, scaled to the strategy's realised vol when vol-match is on."""
    return match_vol(close.pct_change(), strat, on)


def lock_band(fig: go.Figure, a, b, rows: int = 1) -> None:
    """Grey band over the locked out-of-sample span (no data drawn there until it is revealed)."""
    if a is None or b is None or pd.Timestamp(b) <= pd.Timestamp(a):
        return
    fig.add_vrect(x0=a, x1=b, fillcolor=up.fill(T.faint, 0.16), line_width=0,
                  annotation_text="LOCKED · not in stats · reveal in OUT-OF-SAMPLE", annotation_position="top left",
                  annotation_font=dict(size=9, color=T.faint))
    fig.update_xaxes(range=[None, pd.Timestamp(b)])


def equity_fig(r: pd.Series, b: pd.Series, gross: pd.Series | None = None,
               shade_from: pd.Timestamp | None = None, locked_to: pd.Timestamp | None = None,
               b_label: str | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    eq, beq = (1 + r).cumprod(), (1 + b).cumprod()
    if gross is not None:
        geq = (1 + gross).cumprod()
        fig.add_trace(go.Scatter(x=geq.index, y=geq, name="gross of costs", line=dict(color=T.faint, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=beq.index, y=beq, name=b_label or bench_label(),
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
    if locked_to is not None and len(r):
        lock_band(fig, r.index[-1], locked_to)
    fig.update_yaxes(type="log", title_text="growth of 1", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="drawdown", row=2, col=1)
    return fig


def price_fig(ohlcv: pd.DataFrame, res, strat, trades: pd.DataFrame,
              locked_px: pd.Series | None = None) -> go.Figure:
    """``locked_px``: price over the locked span, drawn faint (price only — no signal or trades there)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    close = ohlcv["close"]
    fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=T.text, width=1)), 1, 1)
    if locked_px is not None and len(locked_px):
        fig.add_trace(go.Scatter(x=locked_px.index, y=locked_px, name="close (locked span)",
                                 line=dict(color=T.faint, width=1)), 1, 1)
        lock_band(fig, locked_px.index[0], locked_px.index[-1])
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
    fig.add_trace(go.Scatter(x=bs.index, y=bs, name=bench_label(), line=dict(color=T.amber, width=1)))
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
    for s, name, col in ((b, bench_label(), T.amber), (r, "strategy", T.teal)):
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
    end = spec.end_ts()
    w = {"bt_src": spec.src, "bt_tf": spec.tf, "bt_start": spec.start_ts().date(),
         "bt_end": end.date() if end is not None else _dt.date.today(), "bt_rule": spec.rule,
         "bt_size": spec.sizing_mode, "bt_tvol": int(round(spec.target_vol * 100)), "bt_dir": spec.direction,
         "bt_fpct": int(round(spec.fixed_pct * 100)), "bt_risk": float(round(spec.risk_pct * 100, 1)),
         "bt_scale": bool(spec.scale_by_forecast), "bt_lev": float(spec.max_leverage),
         "bt_seed": int(spec.seed), "bt_base": spec.synth_base,
         "bt_cap": int(spec.capital), "bt_buf": int(round(spec.buffer * 100)), "bt_stop": spec.use_atr_stop,
         "bt_tp": spec.use_atr_tp, "bt_stopk": float(spec.atr_stop_mult), "bt_tpk": float(spec.atr_tp_mult),
         "bt_cost": float(spec.cost_bps), "bt_slip": float(spec.slippage_bps), "bt_fee": float(spec.contract_fee),
         "bt_lock": float(spec.lock_years if spec.lock_years > 0 else (2.0 if spec.holdout else 0.0)),
         "bt_notes": spec.notes}
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
    vm = vol_match_on()
    d = dv.analyse(s_ret, u_ret, periods, vol_match=vm)
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
        {"label": "BEST BLEND", "value": f"{d['best_w'] * 100:.0f}% strat",
         "sub": f"of {'risk' if vm else 'capital'} · Sharpe {d['best_sharpe']:.2f}"},
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
        fig.update_xaxes(title_text=(f"% of risk in strategy (rest in {u_label}), vol-matched" if vm
                                     else f"% of capital in strategy (rest in {u_label}), not vol-matched"))
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
    from ..spec import run_spec
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
            df, _ = spec.split(load(spec, seed=sd))
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


def window_pick(index: pd.DatetimeIndex, key: str, vm: bool = True) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Zoom window for every time-series chart below it. Charts re-slice and rebase to the window
    start, so the y axis always fits the period you picked (Plotly's own box-zoom can't refit y).
    ``vm`` also shows the shared vol-matched toggle in the same row."""
    lo, hi = pd.Timestamp(index[0]), pd.Timestamp(index[-1])
    c1, c2, c3 = st.columns([3, 2, 1])
    if vm:
        vol_match_toggle(key, c3)
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

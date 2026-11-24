"""BACKTEST / OPTIONS — systematic option strategies on synthetic chains (MODELED).

Long-vol and short-vol: ~30 templates (filterable by credit/debit, long/short vol,
bias and risk), an editable leg table, premium / max-loss / notional sizing,
debit and credit exits, VIX-regime and trade-level entry filters, and 0DTE from
VIX1D. Calibrated against Cboe PUT, BXM and PPUT; the grade is shown with every
result. Same guard rails as LAB: costs on, locked holdout, one logged reveal.
"""

from __future__ import annotations

import datetime as _dt
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
from ..spec import lock_date
from . import common as C

_FINDINGS = [
    {"stat": "Implied volatility has exceeded subsequently realised volatility most of the time — the variance risk "
             "premium that option sellers collect.", "cite": "Carr & Wu (2009)"},
    {"stat": "Cboe's PUT index matched the S&P 500's return with about two-thirds of its volatility over 1986–2018.",
     "cite": "Bondarenko (2019)"},
    {"stat": "Short-option returns are negatively skewed: steady gains, rare large losses. Sharpe flatters them.",
     "cite": "Israelov & Nielsen (2015)"},
]


_ANY = "any"
_LEG_COLS = ["side", "qty", "kind", "rule", "value", "ref", "dte"]
_RULE_HELP = ("delta: 0.30 = 30Δ · pct: 0.95 = 95% of spot · atm · wing: width as a share of spot from leg `ref` "
              "(0-based) · premium: strike whose price ≈ value × spot (0.005 = 0.5%) · dte: this leg's own expiry "
              "in days (0 = the strategy's DTE; set it for calendars / diagonals)")


def _legs_df(strat: E.OptionStrategy) -> pd.DataFrame:
    return pd.DataFrame([{"side": "long" if L.side > 0 else "short", "qty": float(L.qty), "kind": L.kind,
                          "rule": L.rule, "value": float(L.value), "ref": int(L.ref), "dte": int(L.dte)}
                         for L in strat.legs], columns=_LEG_COLS)


def _legs_from_df(df: pd.DataFrame) -> list:
    legs = []
    for _, r in df.dropna(subset=["kind", "rule"]).iterrows():
        if r["kind"] not in ("put", "call") or r["rule"] not in ("delta", "pct", "atm", "wing", "premium"):
            continue
        legs.append(E.Leg(r["kind"], 1 if r["side"] == "long" else -1, r["rule"], float(r.get("value") or 0.0),
                          float(r.get("qty") or 1.0), int(r.get("ref") if pd.notna(r.get("ref")) else -1),
                          int(r.get("dte") if pd.notna(r.get("dte")) else 0)))
    return legs[:6]


def _band(label: str, key: str, lo: float, hi: float, default: tuple, step: float, fmt: str = "%.2f",
          help: str | None = None):
    on = st.toggle(label, key=f"{key}_on", help=help)
    if not on:
        return None
    return st.slider(label, lo, hi, default, step, key=f"{key}_v", format=fmt, label_visibility="collapsed")


def _filters_ui() -> dict:
    f: dict = {}
    v = _band("VIX range", "op_fvix", 5.0, 80.0, (12.0, 30.0), 0.5, "%.1f",
              help="Only open when VIX (or the underlying's own vol index) closes inside this range.")
    if v:
        f["vix_lo"], f["vix_hi"] = v
    z = _band("VIX z-score band", "op_fz", -4.0, 4.0, (-1.0, 4.0), 0.1,
              help="(VIX − its N-day mean) ÷ its N-day stdev. +2 = an unusually high VIX for the recent past.")
    if z:
        f["z_lo"], f["z_hi"] = z
        f["z_len"] = int(st.number_input("z-score lookback (days)", 10, 504, 63, 1, key="op_fz_len"))
    c = _band("VIX change (σ moves)", "op_fc", -6.0, 6.0, (-6.0, 1.0), 0.1,
              help="VIX's N-day change in units of its typical N-day change. +2 = a 2σ spike.")
    if c:
        f["chg_lo"], f["chg_hi"] = c
        f["chg_len"] = int(st.number_input("change length (days)", 1, 63, 5, 1, key="op_fc_len"))
    t = _band("term structure VIX / VIX3M", "op_fts", 0.5, 1.5, (0.5, 1.0), 0.01,
              help="Below 1 = contango (calm, normal); above 1 = backwardation (stress).")
    if t:
        f["ts_lo"], f["ts_hi"] = t
    r9 = _band("VIX9D / VIX", "op_fr9", 0.5, 1.6, (0.5, 1.0), 0.01,
               help="Above 1 = the market fears the next week more than the next month.")
    if r9:
        f["r9_lo"], f["r9_hi"] = r9
    pc = _band("VIX 1-year percentile", "op_fpct", 0.0, 100.0, (0.0, 80.0), 1.0, "%.0f")
    if pc:
        f["pct_lo"], f["pct_hi"] = pc
    if st.toggle("trend filter", key="op_fsma_on", help="Only open when the underlying is above / below its average."):
        c1, c2 = st.columns(2)
        f["sma_len"] = int(c1.number_input("average length", 5, 400, 200, 5, key="op_fsma_len"))
        f["sma_side"] = c2.radio("price is", ["above", "below"], key="op_fsma_side", horizontal=True)
    days = st.multiselect("weekdays", ["Mon", "Tue", "Wed", "Thu", "Fri"], default=[], key="op_fdow",
                          help="Empty = every day.")
    if days:
        f["dow"] = [["Mon", "Tue", "Wed", "Thu", "Fri"].index(d) for d in days]
    nc = _band("net credit (% of spot; debit < 0)", "op_fnet", -10.0, 10.0, (0.5, 10.0), 0.05,
               help="Checked on the trade itself at entry. Credit is positive, debit negative.")
    if nc:
        f["net_lo"], f["net_hi"] = nc
    nd = _band("net delta at entry", "op_fdel", -1.5, 1.5, (-0.2, 0.2), 0.01,
               help="Per unit of underlying: +0.30 behaves like 30% long the index.")
    if nd:
        f["delta_lo"], f["delta_hi"] = nd
    if st.toggle("min credit / width", key="op_fcw_on", help="Spreads only: credit ÷ widest strike gap."):
        f["cw_min"] = st.slider("credit / width at least", 0.05, 0.6, 0.2, 0.01, key="op_fcw")
    return f


def _strat_from_ui() -> tuple:
    st.markdown(section("01 · UNDERLYING", 0), unsafe_allow_html=True)
    und = st.selectbox("underlying", list(D.UNDERLYINGS), key="op_und",
                       help="SPX uses the full VIX term structure and is the calibrated case. Others use their own Cboe "
                            "vol index where one exists, else VIX scaled by realised vol.")
    a, b = C.date_range("op", _dt.date(2007, 1, 1), _dt.date(1990, 1, 1) if und == "SPX" else _dt.date(2005, 1, 1))
    start, end = a.isoformat(), C._iso_end(b)

    st.markdown(section("02 · STRATEGY", 1, help="Filter the catalogue, pick a starting template, then edit any "
                        "leg. Long-vol (debit) and short-vol (credit) strategies are both here."),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fl = c1.selectbox("flow", [_ANY, "credit", "debit"], key="op_bflow")
    vo = c2.selectbox("vol", [_ANY, "long", "short", "neutral"], key="op_bvol")
    c1, c2 = st.columns(2)
    bi = c1.selectbox("bias", [_ANY, "bullish", "bearish", "neutral"], key="op_bbias")
    rk = c2.selectbox("risk", [_ANY, "defined", "undefined", "stock"], key="op_brisk")
    names = [n for n, t in E.TAGS.items()
             if (fl == _ANY or t["flow"] == fl) and (vo == _ANY or t["vol"] == vo)
             and (bi == _ANY or t["bias"] == bi) and (rk == _ANY or t["risk"] == rk)]
    if not names:
        st.caption("No template matches — showing all.")
        names = list(E.TEMPLATES)
    tpl = st.selectbox(f"template · {len(names)} match", names, key="op_tpl")
    base = E.TEMPLATES[tpl]
    tg = E.TAGS[tpl]
    st.markdown("".join(C.uc.chip(f"{k.upper()} {v.upper()}", T.teal if v in ("credit", "short") else
                                  T.violet if v in ("debit", "long") else T.muted) for k, v in tg.items()),
                unsafe_allow_html=True)
    legs_df = st.data_editor(_legs_df(base), key=f"op_legs_{tpl}", num_rows="dynamic", hide_index=True,
                             use_container_width=True,
                             column_config={
                                 "side": st.column_config.SelectboxColumn(options=["long", "short"], required=True),
                                 "kind": st.column_config.SelectboxColumn(options=["put", "call"], required=True),
                                 "rule": st.column_config.SelectboxColumn(
                                     options=["delta", "pct", "atm", "wing", "premium"], required=True),
                                 "qty": st.column_config.NumberColumn(min_value=0.1, max_value=10.0, step=0.1),
                                 "value": st.column_config.NumberColumn(format="%.3f", step=0.005),
                                 "ref": st.column_config.NumberColumn(min_value=-1, max_value=5, step=1),
                                 "dte": st.column_config.NumberColumn(min_value=0, max_value=365, step=1)})
    st.caption(_RULE_HELP)
    legs = _legs_from_df(legs_df) or list(base.legs)
    stock = float(st.number_input("stock units held", 0.0, 2.0, float(base.stock), 0.25, key=f"op_stock_{tpl}",
                                  help="1 = own one unit of the underlying per contract-unit (covered call, collar)."))

    c1, c2 = st.columns(2)
    cycles = ["monthly", "dte", "0dte"]
    roll = c1.radio("cycle", cycles, index=cycles.index(base.roll), key=f"op_roll_{tpl}", horizontal=False,
                    format_func=lambda x: {"monthly": "3rd Friday", "dte": "rolling DTE", "0dte": "0DTE (same day)"}[x])
    extra = {"actual": True, "iv_scale": 1.0}
    if roll == "0dte":
        if und not in ("SPX", "SPY"):
            c2.caption(":orange[0DTE is modeled from VIX1D: SPX / SPY only. Using SPX.]")
        extra["actual"] = c2.toggle("only real SPX expiry days", True, key="op_0act",
                                    help="Fridays throughout; Wed from 2016, Mon from Aug 2016, every weekday from "
                                         "May 2022. Off = pretend every day had an expiry.")
        extra["iv_scale"] = c2.slider("IV scale", 0.6, 1.4, 1.0, 0.05, key="op_0ivs",
                                      help="Multiplies VIX1D. The CALIBRATION tab shows how VIX1D compares with the "
                                           "realised open-to-close move.")
        dte = 0
    else:
        dte = c2.slider("days to expiry", 7, 120, max(7, base.dte), 1, key=f"op_dte_{tpl}")

    st.markdown(section("03 · SIZING", 2, help="How much of the account each trade uses."), unsafe_allow_html=True)
    modes = list(E.SIZING)
    sizing = st.radio("sizing", modes, index=modes.index(base.sizing), key=f"op_size_{tpl}", horizontal=True,
                      label_visibility="collapsed")
    st.caption(E.SIZING[sizing])
    size_pct = 0.0
    if sizing == "premium":
        size_pct = st.slider("premium per trade, % of equity", 0.1, 10.0, max(0.1, base.size_pct * 100 or 2.0), 0.1,
                             key=f"op_sz_{tpl}") / 100
    elif sizing == "risk":
        size_pct = st.slider("max loss per trade, % of equity", 0.1, 20.0, max(0.1, base.size_pct * 100 or 2.0), 0.1,
                             key=f"op_sz_{tpl}") / 100
    elif sizing == "notional":
        size_pct = st.slider("notional per trade, % of equity", 5.0, 300.0, max(5.0, base.size_pct * 100 or 100.0),
                             5.0, key=f"op_sz_{tpl}") / 100

    st.markdown(section("04 · MANAGEMENT", 3), unsafe_allow_html=True)
    debit = tg["flow"] == "debit"
    c1, c2 = st.columns(2)
    if debit:
        tp = c1.slider("take profit at % gain", 0, 500, 0, 10, key=f"op_tp_{tpl}",
                       help="Close when the position is worth this much more than you paid (100 = doubled).") / 100
        sl = c2.slider("stop at % of debit lost", 0, 100, 0, 5, key=f"op_sl_{tpl}") / 100
    else:
        tp = c1.slider("take profit at % of credit", 0, 100, 0, 5, key=f"op_tp_{tpl}") / 100
        sl = c2.slider("stop at x credit lost", 0.0, 5.0, 0.0, 0.25, key=f"op_sl_{tpl}")
    c1, c2, c3 = st.columns(3)
    exit_dte = int(c1.number_input("exit at DTE", 0, 60, 0, 1, key="op_xdte", help="0 = hold to expiry"))
    max_hold = int(c2.number_input("max hold days", 0, 365, 0, 1, key="op_xhold", help="0 = no limit"))
    dexit = float(c3.number_input("delta exit", 0.0, 2.0, 0.0, 0.05, key="op_xdelta",
                                  help="Close if |net delta| per unit exceeds this. 0 = off."))

    st.markdown(section("05 · ENTRY FILTERS", 4, help="Every filter only removes trades. The FILTERS tab shows "
                        "which days passed."), unsafe_allow_html=True)
    with st.expander("VIX, trend, weekday and trade-level filters", expanded=False):
        filters = _filters_ui()
    if filters:
        st.caption(f"{len(filters)} filter setting(s) active.")

    st.markdown(section("06 · COSTS & HOLDOUT", 5), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    spread = c1.slider("half-spread % of premium", 0.0, 15.0, 3.0, 0.5, key="op_spr") / 100
    ratio = c2.slider("ATM / VIX ratio", 0.70, 1.00, E.ATM_RATIO, 0.01, key="op_ratio",
                      help=f"Calibrated {E.ATM_RATIO} on Cboe PUT. Changing it changes every price — keep the "
                           "calibrated value unless you are stress-testing the model.")
    hold = C.lock_input("op")
    strat = replace(base, legs=legs, stock=stock, dte=dte, roll=roll, take_profit=tp, stop_mult=sl, min_vix=0.0,
                    spread_pct=spread, atm_ratio=ratio, sizing=sizing, size_pct=size_pct, exit_dte=exit_dte,
                    max_hold=max_hold, delta_exit=dexit, filters=filters)
    return und, (start, end), strat, float(hold), extra


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _inputs(und: str, start) -> dict:
    """``start``: (start ISO, end ISO or '') — or a bare year (calibration)."""
    if isinstance(start, (tuple, list)):
        return D.load(und, start[0], start[1] or None)
    return D.load(und, start)


def _run(und: str, start, strat: E.OptionStrategy, extra: dict | None = None) -> E.OptionResult:
    ex = extra or {}
    return _run_cached(und, start, json.dumps(asdict(strat), sort_keys=True, default=str),
                       bool(ex.get("actual", True)), float(ex.get("iv_scale", 1.0)))


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=64)
def _run_cached(und: str, start, strat_json: str, actual: bool = True, iv_scale: float = 1.0) -> E.OptionResult:
    d = json.loads(strat_json)
    d["legs"] = [E.Leg(**L) for L in d["legs"]]
    strat = E.OptionStrategy(**d)
    if strat.roll == "0dte":
        x = _inputs("SPX" if und not in ("SPX", "SPY") else und, start)
        return E.simulate_0dte(strat, x["ohlc"], x["iv1d"], x["skew"], x["rf"], x["term"], iv_scale=iv_scale,
                               actual_expiries=actual)
    x = _inputs(und, start)
    return E.simulate(strat, x["S"], x["term"], x["skew"], x["rf"], x["stock_tr"], x["vol_scale"], x["q"])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _calibration(ratio: float) -> dict:
    out = {}
    for tpl, idx in (("Short put (PUT-style)", "PUT"), ("Covered call (BXM-style)", "BXM"),
                     ("Protective put (95%, PPUT-style)", "PPUT")):
        res = _run("SPX", 2007, replace(E.TEMPLATES[tpl], atm_ratio=ratio))
        cal = E.calibrate(res.returns, D.published(idx))
        cal.pop("months", None)
        out[idx] = cal
    return out


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        und, start, strat, hold, extra = _strat_from_ui()
    with right:
        try:
            x = _inputs(und, start)
            with st.spinner("pricing synthetic chains"):
                res = _run(und, start, strat, extra)
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", f"{und}: {exc}")
            return
        idx = res.returns.index
        locked = lock_date(idx, 0, hold)
        cut = int(idx.searchsorted(locked)) if locked is not None else len(idx)
        family = f"Options:{und}:{start[0]}:{start[1]}:L{hold:g}"
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
        tiers = [E.calibration_tier(cal[k]) for k in ("PUT", "BXM", "PPUT") if k in cal]
        tier = max(t[0] for t in tiers)
        note = (f"PUT: model {cal['PUT'].get('cagr_model', np.nan) * 100:.1f}% vs Cboe "
                f"{cal['PUT'].get('cagr_actual', np.nan) * 100:.1f}%/yr, TE {cal['PUT'].get('tracking_error', np.nan) * 100:.1f}% · "
                f"BXM: {cal['BXM'].get('cagr_model', np.nan) * 100:.1f}% vs {cal['BXM'].get('cagr_actual', np.nan) * 100:.1f}%, "
                f"TE {cal['BXM'].get('tracking_error', np.nan) * 100:.1f}% · long side PPUT: "
                f"{cal['PPUT'].get('cagr_model', np.nan) * 100:.1f}% vs {cal['PPUT'].get('cagr_actual', np.nan) * 100:.1f}% "
                f"(2007→). " + ("0DTE is priced from VIX1D (VIX9D-based before May 2022) — not calibrated to any "
                                "published index. " if strat.roll == "0dte" else "")
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
        trv = C.match_vol(tr, r)                       # shared vol-matched toggle (EQUITY tab)
        ul = und + (" vm" if C.vol_match_on() else "")
        bs = metrics.summary(trv, periods=p)
        ab = A.alpha_beta(r, tr, p)
        st.markdown(uk.kpis([
            {"label": "SHARPE", "value": f"{s['Sharpe']:.2f}", "sub": f"{ul} {bs['Sharpe']:.2f}"},
            {"label": "CAGR", "value": f"{s['CAGR'] * 100:.1f}%", "sub": f"{ul} {bs['CAGR'] * 100:.1f}%"},
            {"label": "VOL", "value": f"{s['Vol'] * 100:.1f}%", "sub": f"{ul} {bs['Vol'] * 100:.1f}%"},
            {"label": "MAX DD", "value": f"{s['MaxDD'] * 100:.1f}%", "color": T.coral, "sub": f"{ul} {bs['MaxDD'] * 100:.1f}%"},
            {"label": "SKEW", "value": f"{s['Skew']:.2f}", "color": T.coral if s["Skew"] < -0.5 else T.text,
             "sub": "daily returns"},
            {"label": "BETA", "value": f"{ab['beta']:.2f}", "sub": f"alpha {ab['alpha_ann'] * 100:+.1f}%/yr t {ab['alpha_t']:.1f}"},
        ]), unsafe_allow_html=True)
        if s["Skew"] < -1:
            C.banner(T.amber, "NEGATIVE SKEW", "returns come in small steady gains and rare large losses; Sharpe "
                     "overstates how comfortable this is — look at the drawdown and the worst months")

        _preview(und, start, strat, x)
        tabs = st.tabs(["EQUITY", "EXPOSURE", "MONTHLY", "DISTRIBUTION", "SENSITIVITY", "TRADES", "CALIBRATION",
                        "OUT-OF-SAMPLE", "STATS", "FILTERS"])
        with tabs[9]:
            _filters_tab(res, x, strat, end)
        with tabs[8]:
            keys = [k for k in s if k in bs]
            st.dataframe(pd.DataFrame({"strategy": [C.fmt(k, s[k]) for k in keys],
                                       C.bench_label(f"{und} total return"): [C.fmt(k, bs[k]) for k in keys]},
                                      index=keys),
                         use_container_width=True)
            _trade_stats(res.trades)
            C.diversify_panel(r, tr, p, und, "op_div")
        with tabs[0]:
            b = trv
            hidden = locked is not None and not show_all
            wa, wb = C.window_pick(idx if hidden else r.index, "op_win")
            rw, bw = C.clip(r, wa, wb), C.clip(b, wa, wb)
            if (wa, wb) != (idx[0] if hidden else r.index[0], idx[-1] if hidden else r.index[-1]):
                C.window_kpis(rw, C.clip(b, wa, wb), p, ul)
            up.show(C.equity_fig(rw, bw, shade_from=locked if show_all and locked is not None and locked <= wb
                                 else None, locked_to=wb if hidden and len(rw) and wb > rw.index[-1] else None),
                    440, f"strategy vs {C.bench_label(und)}" + (" · grey = locked out-of-sample" if hidden else ""),
                    key="op_eq")
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
            up.show(C.dist_fig(r, trv), 320, "daily return distribution", key="op_dist")
            worst = ((1 + r).resample("ME").prod() - 1).nsmallest(5)
            st.caption("Worst months: " + ", ".join(f"{d:%Y-%m} {v * 100:.1f}%" for d, v in worst.items()))
        with tabs[4]:
            _sensitivity(und, start, strat, cut, extra)
        with tabs[5]:
            tcol = "open" if "open" in res.trades else "date"
            st.dataframe(res.trades.iloc[: max(1, len(res.trades))][::-1] if (show_all or res.trades.empty) else
                         res.trades[pd.to_datetime(res.trades[tcol]) < (locked or idx[-1] + pd.Timedelta(days=1))][::-1],
                         use_container_width=True, hide_index=True)
        with tabs[6]:
            rows = [{"index": k, **{m: v for m, v in cal[k].items() if m != "ok"}} for k in cal]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("ATM vol = ratio × VIX-term-structure vol, smile from Cboe SKEW, S&P dividend yield from "
                       "^SP500TR vs ^GSPC. The ratio was fitted on PUT only; BXM (short calls) and PPUT (long puts) "
                       "are out-of-sample checks.")
            ivr = E.implied_vs_realised_0dte(x["ohlc"][x["ohlc"].index >= "2011-01-01"], x["iv1d"])
            if ivr.get("ok"):
                st.markdown(uk.kpis([
                    {"label": "0DTE · RMS OPEN→CLOSE MOVE", "value": f"{ivr['rms_realised'] * 100:.2f}%",
                     "sub": f"{ivr['n']} sessions"},
                    {"label": "IMPLIED BY VIX1D", "value": f"{ivr['rms_implied'] * 100:.2f}%",
                     "sub": "σ × √(1/252)"},
                    {"label": "REALISED ÷ IMPLIED", "value": f"{ivr['ratio']:.2f}",
                     "sub": "below 1 = same-day options were rich (sellers' edge)"},
                ]), unsafe_allow_html=True)
        with tabs[7]:
            if locked is None:
                C.banner(T.coral, "NO HOLDOUT", "set 'lock last N years' in 06 · COSTS & HOLDOUT to keep an honest test")
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


def option_knobs(strat: E.OptionStrategy) -> list:
    """Every sweepable setting of an options strategy."""
    from .sensitivity_panel import Knob
    ks = []
    for i, L in enumerate(strat.legs):
        side = "short" if L.side < 0 else "long"
        if L.rule == "delta":
            ks.append(Knob(f"leg{i}", f"{side} {L.kind} Δ (leg {i + 1})", L.value, 0.02, 0.50))
        elif L.rule == "pct":
            ks.append(Knob(f"leg{i}", f"{side} {L.kind} strike % (leg {i + 1})", L.value, 0.70, 1.30))
    if strat.roll != "0dte":
        ks.append(Knob("dte", "DTE", strat.dte, 5, 120, integer=True))
    ks += [Knob("take_profit", "take profit", strat.take_profit, 0.0, 3.0),
           Knob("stop_mult", "stop", strat.stop_mult, 0.0, 5.0),
           Knob("spread_pct", "half-spread %", strat.spread_pct, 0.0, 0.15)]
    if strat.sizing != "collateral":
        ks.append(Knob("size_pct", f"{strat.sizing} budget", strat.size_pct, 0.001, 0.2 if strat.sizing != "notional"
                       else 3.0))
    if strat.roll != "0dte":
        ks += [Knob("exit_dte", "exit at DTE", strat.exit_dte, 0, 30, integer=True),
               Knob("max_hold", "max hold days", strat.max_hold, 0, 90, integer=True)]
    rng = {"vix": (5.0, 80.0), "z": (-4.0, 4.0), "chg": (-6.0, 6.0), "ts": (0.5, 1.5), "r9": (0.5, 1.6),
           "pct": (0.0, 100.0), "net": (-10.0, 10.0), "delta": (-1.5, 1.5)}
    for k, v in (strat.filters or {}).items():
        stem = k.rsplit("_", 1)[0]
        if k.endswith(("_lo", "_hi")) and stem in rng:
            lo, hi = rng[stem]
            ks.append(Knob(f"f:{k}", E.FILTER_KEYS.get(k, k), v, lo, hi))
        elif k in ("z_len", "chg_len", "sma_len"):
            ks.append(Knob(f"f:{k}", E.FILTER_KEYS.get(k, k), v, 2, 400, integer=True, log=True))
        elif k == "cw_min":
            ks.append(Knob("f:cw_min", "min credit / width", v, 0.05, 0.6))
    if not (strat.filters or {}).get("vix_lo"):
        ks.append(Knob("f:vix_lo", "VIX at or above", 0.0, 0.0, 40.0))
    return ks


def apply_option_knobs(strat: E.OptionStrategy, vals: dict) -> E.OptionStrategy:
    legs = list(strat.legs)
    kw = {}
    for k, v in vals.items():
        if k.startswith("leg"):
            i = int(k[3:])
            legs[i] = replace(legs[i], value=float(v))
        elif k.startswith("f:"):
            kw.setdefault("filters", dict(strat.filters or {}))[k[2:]] = (
                int(v) if k[2:] in ("z_len", "chg_len", "sma_len") else float(v))
        elif k in ("dte", "exit_dte", "max_hold"):
            kw[k] = int(v)
        else:
            kw[k] = float(v)
    return replace(strat, legs=legs, **kw)


def _sensitivity(und: str, start: int, strat: E.OptionStrategy, cut: int, extra: dict | None = None) -> None:
    from .sensitivity_panel import panel

    def run(vals, cost_mult=1.0):
        s2 = apply_option_knobs(strat, vals)
        if cost_mult != 1.0:
            s2 = replace(s2, spread_pct=s2.spread_pct * cost_mult)
        r = _run(und, start, s2, extra).returns.iloc[:cut]
        return metrics.summary(r, periods=252)

    panel(option_knobs(strat), run, "op_sens", ["Sharpe", "CAGR", "MaxDD", "Calmar", "Sortino", "Vol", "Skew"],
          state=json.dumps(asdict(strat), sort_keys=True, default=str) + f"|{und}|{start}|{cut}|{extra}",
          supports_costs=True,
          note="In-sample only (the locked slice is excluded). Every cell re-prices the whole history; with "
               "'rolling DTE' the DTE knob sets the exact tenor, with '3rd Friday' it picks the nearest cycle.")


def _preview(und: str, start: int, strat: E.OptionStrategy, x: dict) -> None:
    """Today's version of the trade: legs, greeks and the expiry payoff."""
    try:
        term_row = x["term"].ffill().iloc[-1][E.TERM_COLS].to_numpy(float)
        sk = float(x["skew"].ffill().iloc[-1])
        r = float(x["rf"].ffill().iloc[-1]) if len(x["rf"]) else 0.0
        s0 = float(x["S"].iloc[-1])
        sc = float(x["vol_scale"].ffill().iloc[-1]) if x.get("vol_scale") is not None and len(x["vol_scale"]) else 1.0
        iv1 = float(x["iv1d"].ffill().iloc[-1]) if len(x["iv1d"]) else None
        pv = E.entry_preview(strat, s0, term_row, sk, r, sc, iv1)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"preview unavailable: {exc}")
        return
    with st.expander(f"TRADE PREVIEW · if opened today at {s0:,.2f}", expanded=False):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = go.Figure()
            pay = pv["payoff"] / s0 * 100
            fig.add_trace(go.Scatter(x=pv["grid"], y=np.where(pay >= 0, pay, np.nan), fill="tozeroy",
                                     fillcolor=up.fill(T.teal, 0.25), line=dict(color=T.teal, width=1.6),
                                     name="profit"))
            fig.add_trace(go.Scatter(x=pv["grid"], y=np.where(pay < 0, pay, np.nan), fill="tozeroy",
                                     fillcolor=up.fill(T.coral, 0.25), line=dict(color=T.coral, width=1.6),
                                     name="loss"))
            fig.add_vline(x=s0, line=dict(color=T.faint, dash="dash", width=1), annotation_text="spot",
                          annotation_font=dict(size=9, color=T.muted))
            for L in pv["legs"]:
                fig.add_vline(x=L["K"], line=dict(color=T.violet if L["leg"].side > 0 else T.amber, width=0.8,
                                                  dash="dot"))
            fig.add_hline(y=0, line=dict(color=T.line))
            fig.update_yaxes(title_text="P/L at expiry, % of spot")
            fig.update_xaxes(title_text="underlying at expiry")
            fig.update_layout(hovermode="x")
            up.show(fig, 300, "payoff at expiry (per unit) · violet = long strike, amber = short", key="op_pay",
                    scale_toggle=False)
        with c2:
            g = pv["greeks"]
            st.markdown(uk.kpis([
                {"label": "NET", "value": f"{abs(pv['net_credit_%']):.2f}%",
                 "sub": "credit" if pv["net_credit_%"] > 0 else "debit", "color": T.teal if pv["net_credit_%"] > 0
                 else T.violet},
                {"label": "MAX LOSS", "value": f"{pv['max_loss'] / s0 * 100:.2f}%", "color": T.coral,
                 "sub": "of spot · within ±25%"},
                {"label": "DELTA", "value": f"{g['delta']:+.2f}"},
                {"label": "VEGA / VOL PT", "value": f"{g['vega'] / s0 * 100:+.3f}%",
                 "sub": "long vol" if g["vega"] > 0 else "short vol"},
            ]), unsafe_allow_html=True)
            rows = [{"leg": f"{'+' if L['leg'].side > 0 else '-'}{L['leg'].qty:g} {L['leg'].kind}",
                     "strike": round(L["K"], 2), "% spot": round(L["K"] / s0 * 100, 1),
                     "price %": round(L["price"] / s0 * 100, 3), "iv %": round(L["iv"] * 100, 1),
                     "days": round(L["T_days"], 1)} for L in pv["legs"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if pv["breakevens"]:
                st.caption("breakevens: " + ", ".join(f"{b:,.0f} ({b / s0 - 1:+.1%})" for b in pv["breakevens"]))


def _filters_tab(res: E.OptionResult, x: dict, strat: E.OptionStrategy, end: int) -> None:
    g = res.gate.iloc[:end] if res.gate is not None else None
    if g is None or not (strat.filters or strat.min_vix):
        st.caption("No entry filters set: every day is eligible. Add some under 05 · ENTRY FILTERS.")
        if g is None:
            return
    vix = x["term"]["VIX"].reindex(g.index).ffill()
    share = float(g.mean())
    n_tr = len(res.trades)
    st.markdown(uk.kpis([
        {"label": "DAYS ELIGIBLE", "value": f"{share * 100:.0f}%", "sub": f"{int(g.sum())} of {len(g)}"},
        {"label": "TRADES", "value": f"{n_tr}"},
        {"label": "ACTIVE SETTINGS", "value": f"{len(strat.filters)}"},
    ]), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vix.index, y=vix.where(g), mode="lines", line=dict(color=T.teal, width=1.2),
                             name="eligible", connectgaps=False))
    fig.add_trace(go.Scatter(x=vix.index, y=vix.where(~g), mode="lines", line=dict(color=T.faint, width=0.8),
                             name="filtered out", connectgaps=False))
    for k, nm in (("vix_lo", "VIX floor"), ("vix_hi", "VIX cap")):
        if strat.filters.get(k) is not None:
            fig.add_hline(y=strat.filters[k], line=dict(color=T.amber, dash="dot", width=1), annotation_text=nm,
                          annotation_font=dict(size=9, color=T.amber))
    fig.update_yaxes(title_text="VIX")
    up.show(fig, 320, "VIX — teal where a new trade was allowed", key="op_gate")
    st.caption("Trade-level filters (net credit, delta, credit/width) are checked when the trade is priced, so a "
               "day can be eligible here and still skip the trade.")


def _trade_stats(trades: pd.DataFrame) -> None:
    if trades.empty or "pnl_%_equity" not in trades:
        return
    p = trades["pnl_%_equity"].dropna()
    if p.empty:
        return
    w, l_ = p[p > 0], p[p <= 0]
    st.markdown(uk.kpis([
        {"label": "TRADES", "value": f"{len(p)}"},
        {"label": "WIN RATE", "value": f"{len(w) / len(p) * 100:.0f}%"},
        {"label": "AVG WIN", "value": f"{w.mean() if len(w) else 0:+.2f}%", "sub": "of equity", "color": T.teal},
        {"label": "AVG LOSS", "value": f"{l_.mean() if len(l_) else 0:+.2f}%", "sub": "of equity", "color": T.coral},
        {"label": "PAYOFF RATIO", "value": f"{abs(w.mean() / l_.mean()) if len(w) and len(l_) and l_.mean() else 0:.2f}",
         "sub": "avg win ÷ avg loss"},
        {"label": "WORST TRADE", "value": f"{p.min():+.2f}%", "color": T.coral},
    ]), unsafe_allow_html=True)
    if "reason" in trades:
        st.caption("exits: " + " · ".join(f"{k} {v}" for k, v in trades["reason"].value_counts().items()))

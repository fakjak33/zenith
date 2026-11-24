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
    return legs[:12]


def _band(label: str, key: str, lo: float, hi: float, default: tuple, step: float, fmt: str = "%.2f",
          help: str | None = None):
    on = st.toggle(label, key=f"{key}_on", help=help)
    if not on:
        return None
    return st.slider(label, lo, hi, default, step, key=f"{key}_v", format=fmt, label_visibility="collapsed")


def _filters_ui() -> dict:
    f: dict = {}
    v = _band("VIX range", "op_fvix", 5.0, 150.0, (12.0, 30.0), 0.5, "%.1f",
              help="Only open when VIX (or the underlying's own vol index) closes inside this range. "
                   "e.g. 12–30 skips both very calm and panicky days.")
    if v:
        f["vix_lo"], f["vix_hi"] = v
    z = _band("VIX z-score band", "op_fz", -6.0, 6.0, (-1.0, 4.0), 0.1,
              help="(VIX − its N-day mean) ÷ its N-day stdev. +2 = an unusually high VIX for the recent past.")
    if z:
        f["z_lo"], f["z_hi"] = z
        f["z_len"] = int(st.number_input("z-score lookback (days)", 5, 2520, 63, 1, key="op_fz_len",
                                         help="Trading days used for the mean and stdev (63 ≈ 3 months)."))
    c = _band("VIX change (σ moves)", "op_fc", -10.0, 10.0, (-6.0, 1.0), 0.1,
              help="VIX's N-day change in units of its typical N-day change. +2 = a 2σ spike.")
    if c:
        f["chg_lo"], f["chg_hi"] = c
        f["chg_len"] = int(st.number_input("change length (days)", 1, 252, 5, 1, key="op_fc_len",
                                           help="How many trading days the change is measured over."))
    t = _band("term structure VIX / VIX3M", "op_fts", 0.3, 2.0, (0.5, 1.0), 0.01,
              help="Below 1 = contango (calm, normal); above 1 = backwardation (stress).")
    if t:
        f["ts_lo"], f["ts_hi"] = t
    r9 = _band("VIX9D / VIX", "op_fr9", 0.3, 2.5, (0.5, 1.0), 0.01,
               help="Above 1 = the market fears the next week more than the next month.")
    if r9:
        f["r9_lo"], f["r9_hi"] = r9
    pc = _band("VIX 1-year percentile", "op_fpct", 0.0, 100.0, (0.0, 80.0), 1.0, "%.0f",
               help="Where today's VIX sits within its own past year: 90 = higher than 90% of days.")
    if pc:
        f["pct_lo"], f["pct_hi"] = pc
    if st.toggle("trend filter", key="op_fsma_on", help="Only open when the underlying is above / below its average."):
        c1, c2 = st.columns(2)
        f["sma_len"] = int(c1.number_input("average length", 2, 1000, 200, 5, key="op_fsma_len",
                                           help="Simple moving average length in trading days."))
        f["sma_side"] = c2.radio("price is", ["above", "below"], key="op_fsma_side", horizontal=True)
    days = st.multiselect("weekdays", ["Mon", "Tue", "Wed", "Thu", "Fri"], default=[], key="op_fdow",
                          help="Only open on these weekdays. Empty = every day.")
    if days:
        f["dow"] = [["Mon", "Tue", "Wed", "Thu", "Fri"].index(d) for d in days]
    nc = _band("net credit (% of spot; debit < 0)", "op_fnet", -25.0, 25.0, (0.5, 10.0), 0.05,
               help="Checked on the trade itself at entry. Credit is positive, debit negative.")
    if nc:
        f["net_lo"], f["net_hi"] = nc
    nd = _band("net delta at entry", "op_fdel", -3.0, 3.0, (-0.2, 0.2), 0.01,
               help="Per unit of underlying: +0.30 behaves like 30% long the index.")
    if nd:
        f["delta_lo"], f["delta_hi"] = nd
    if st.toggle("min credit / width", key="op_fcw_on", help="Spreads only: credit ÷ widest strike gap."):
        f["cw_min"] = st.slider("credit / width at least", 0.01, 0.95, 0.2, 0.01, key="op_fcw")
    return f


# ------------------------------------------------------------------ builder ----
_STRUCTS = {
    "put": "single put", "call": "single call",
    "put spread": "put vertical spread (2 puts)", "call spread": "call vertical spread (2 calls)",
    "straddle": "straddle (put + call, same strike)", "strangle": "strangle (OTM put + OTM call)",
    "iron condor": "iron condor (put spread + call spread)",
}
_STRIKE_BY = {"delta": "delta (Δ)", "pct": "% of spot", "atm": "at the money", "premium": "premium (% of spot)"}


def _strike_input(prefix: str, label: str, default_delta: float, default_pct: float, by: str | None = None):
    """One strike choice → (rule, value)."""
    by = by or st.selectbox(f"{label} strike by", list(_STRIKE_BY), format_func=_STRIKE_BY.get, key=f"{prefix}_by",
                            help="delta: 5 = 5-delta, far out of the money (≈5% chance of finishing in the money); "
                                 "50 ≈ at the money. % of spot: 90 = strike 10% below today's price. "
                                 "premium: the strike whose option costs about this share of spot.")
    if by == "delta":
        d = st.number_input(f"{label} delta (Δ)", 0.1, 99.9, default_delta, 0.5, key=f"{prefix}_d", format="%.1f",
                            help="Any delta from 0.1 to 99.9. Lower = further out of the money and cheaper.")
        return "delta", d / 100
    if by == "pct":
        p = st.number_input(f"{label} strike, % of spot", 10.0, 300.0, default_pct, 0.5, key=f"{prefix}_p",
                            format="%.1f", help="100 = at the money. Puts below 100 / calls above 100 are OTM.")
        return "pct", p / 100
    if by == "premium":
        p = st.number_input(f"{label} premium, % of spot", 0.01, 20.0, 0.5, 0.05, key=f"{prefix}_pr", format="%.2f",
                            help="e.g. 0.5 = pick the strike whose option costs about 0.5% of the underlying.")
        return "premium", p / 100
    return "atm", 0.0


def _simple_legs() -> tuple[list, str]:
    c1, c2 = st.columns(2)
    act = c1.radio("action", ["buy", "sell"], horizontal=True, key="op_s_act",
                   help="buy = pay premium (long options: limited loss, long volatility). "
                        "sell = collect premium (short options: limited gain, short volatility).")
    struct = c2.selectbox("structure", list(_STRUCTS), format_func=_STRUCTS.get, key="op_s_struct",
                          help="What to trade. Spreads and condors cap both the gain and the loss.")
    side = 1 if act == "buy" else -1
    if struct in ("put", "call"):
        rule, val = _strike_input("op_s1", "", 5.0 if struct == "put" else 25.0, 90.0 if struct == "put" else 110.0)
        q = st.number_input("contracts per unit", 1.0, 20.0, 1.0, 1.0, key="op_s_q",
                            help="Ratio inside one position; sizing below decides how many positions-worth to buy.")
        return [E.Leg(struct, side, rule, val, q)], f"{act} {struct}"
    if struct in ("put spread", "call spread"):
        kind = struct.split()[0]
        st.caption("Near leg = closer to the money; far leg = further out. Buying the spread = long the near leg, "
                   "short the far leg (a debit); selling = the reverse (a credit).")
        r1, v1 = _strike_input("op_s1", "near leg", 30.0, 95.0 if kind == "put" else 105.0)
        r2, v2 = _strike_input("op_s2", "far leg", 10.0, 90.0 if kind == "put" else 110.0)
        return [E.Leg(kind, side, r1, v1), E.Leg(kind, -side, r2, v2)], f"{act} {struct}"
    if struct == "straddle":
        return [E.Leg("put", side, "atm"), E.Leg("call", side, "atm")], f"{act} straddle"
    if struct == "strangle":
        r1, v1 = _strike_input("op_s1", "put", 25.0, 95.0)
        r2, v2 = _strike_input("op_s2", "call", 25.0, 105.0)
        return [E.Leg("put", side, r1, v1), E.Leg("call", side, r2, v2)], f"{act} strangle"
    st.caption("Short strikes are the inner pair, long wings the outer pair (for 'sell'; 'buy' flips every leg).")
    d1 = st.number_input("inner strikes delta (Δ)", 0.5, 49.0, 16.0, 0.5, key="op_s_ic1",
                         help="Delta of the put and call nearer the money.")
    d2 = st.number_input("outer wings delta (Δ)", 0.1, 48.0, 5.0, 0.5, key="op_s_ic2",
                         help="Delta of the protective wings further out. Must be below the inner delta.")
    return ([E.Leg("put", side, "delta", d1 / 100), E.Leg("put", -side, "delta", d2 / 100),
             E.Leg("call", side, "delta", d1 / 100), E.Leg("call", -side, "delta", d2 / 100)], f"{act} iron condor")


_LEG_HELP = {
    "side": "long = buy, short = sell",
    "qty": "contracts of this leg per position unit (2 = a ratio leg)",
    "kind": "put or call",
    "rule": "how the strike is chosen: delta · pct (of spot) · atm · wing (width from leg `ref`) · premium",
    "value": "delta 0.05 = 5Δ · pct 0.90 = 90% of spot · wing 0.05 = 5% of spot · premium 0.005 = 0.5% of spot",
    "ref": "wing legs: which leg (0-based) the width is measured from; −1 = none",
    "dte": "this leg's own days to expiry (0 = the strategy's DTE; set it for calendars / diagonals)",
}


def _advanced_legs(seed: E.OptionStrategy, seed_name: str) -> list:
    df0 = _legs_df(seed) if seed.legs else pd.DataFrame([{"side": "long", "qty": 1.0, "kind": "put", "rule": "delta",
                                                         "value": 0.05, "ref": -1, "dte": 0}], columns=_LEG_COLS)
    legs_df = st.data_editor(df0, key=f"op_legs_adv_{seed_name}", num_rows="dynamic", hide_index=True,
                             use_container_width=True,
                             column_config={
                                 "side": st.column_config.SelectboxColumn(options=["long", "short"], required=True,
                                                                          help=_LEG_HELP["side"]),
                                 "kind": st.column_config.SelectboxColumn(options=["put", "call"], required=True,
                                                                          help=_LEG_HELP["kind"]),
                                 "rule": st.column_config.SelectboxColumn(
                                     options=["delta", "pct", "atm", "wing", "premium"], required=True,
                                     help=_LEG_HELP["rule"]),
                                 "qty": st.column_config.NumberColumn(min_value=0.01, max_value=100.0, step=0.01,
                                                                      help=_LEG_HELP["qty"]),
                                 "value": st.column_config.NumberColumn(format="%.4f", step=0.0005,
                                                                        help=_LEG_HELP["value"]),
                                 "ref": st.column_config.NumberColumn(min_value=-1, max_value=11, step=1,
                                                                      help=_LEG_HELP["ref"]),
                                 "dte": st.column_config.NumberColumn(min_value=0, max_value=1095, step=1,
                                                                      help=_LEG_HELP["dte"])})
    st.caption(_RULE_HELP + " · up to 12 legs.")
    return _legs_from_df(legs_df)


def _browse_templates() -> str:
    with st.expander("filter templates", expanded=False):
        c1, c2 = st.columns(2)
        fl = c1.selectbox("flow", [_ANY, "credit", "debit"], key="op_bflow",
                          help="credit = you collect premium up front · debit = you pay it")
        vo = c2.selectbox("vol", [_ANY, "long", "short", "neutral"], key="op_bvol",
                          help="long vol profits when volatility rises / big moves; short vol when markets stay calm")
        c1, c2 = st.columns(2)
        bi = c1.selectbox("bias", [_ANY, "bullish", "bearish", "neutral"], key="op_bbias",
                          help="the market direction the strategy wants")
        rk = c2.selectbox("risk", [_ANY, "defined", "undefined", "stock"], key="op_brisk",
                          help="defined = max loss known at entry · undefined = can lose far more than the credit · "
                               "stock = includes a share position")
    names = [n for n, t in E.TAGS.items()
             if (fl == _ANY or t["flow"] == fl) and (vo == _ANY or t["vol"] == vo)
             and (bi == _ANY or t["bias"] == bi) and (rk == _ANY or t["risk"] == rk)]
    if not names:
        st.caption("No template matches — showing all.")
        names = list(E.TEMPLATES)
    return st.selectbox(f"template · {len(names)} match", names, key="op_tpl",
                        help="Ready-made strategies. The lines below spell out exactly what each one does; "
                             "switch to 'advanced legs' to edit it.")


def _account_ui(und: str) -> dict:
    """$ account: total value, holdings, options sleeve, the rest in cash."""
    mode = st.radio("account", ["$ account", "% of equity"], horizontal=True, key="op_acct",
                    help="$ account: enter real dollar amounts — total value, what you hold, and the cash set aside "
                         "for options; trades are sized in dollars / whole contracts. % of equity: the original "
                         "returns-only mode where every size is a share of a notional $1 account.")
    if mode == "% of equity":
        return {"mode": "pct"}
    c1, c2 = st.columns(2)
    total = float(c1.number_input("total account $", 1_000.0, 1e10, 30_000.0, 1_000.0, key="op_total",
                                  help="Net value of the whole account: holdings + options cash + idle cash."))
    sleeve = float(c2.number_input("options cash $", 100.0, 1e10, 5_000.0, 500.0, key="op_sleeve",
                                   help="Cash set aside for the options strategy. Premiums are paid from (and "
                                        "credits / profits land in) this sleeve; it earns T-bills while idle."))
    proxy = {"SPX": "SPY"}.get(und, und)
    h0 = pd.DataFrame([{"ticker": proxy, "$": 25_000.0}])
    hdf = st.data_editor(h0, key="op_hold_tbl", num_rows="dynamic", hide_index=True, use_container_width=True,
                         column_config={"ticker": st.column_config.TextColumn(help="Any Yahoo ticker, bought and "
                                                                                    "held from the start date"),
                                        "$": st.column_config.NumberColumn(min_value=0.0, step=500.0, format="%.0f",
                                                                           help="$ invested on day one")})
    holds = {str(r["ticker"]).strip().upper(): float(r["$"] or 0) for _, r in hdf.dropna(subset=["ticker"]).iterrows()
             if str(r["ticker"]).strip()}
    cash = total - sleeve - sum(holds.values())
    if cash < -1e-6:
        st.caption(f":orange[holdings + options cash exceed the account by \\${-cash:,.0f} — the excess is treated as "
                   "borrowed (negative cash).]")
    else:
        st.caption(f"Holdings \\${sum(holds.values()):,.0f} · options \\${sleeve:,.0f} · idle cash \\${cash:,.0f} "
                   "(earns T-bills).")
    return {"mode": "usd", "total": total, "sleeve": sleeve, "holdings": holds, "cash": cash}


def _size_ui(acct: dict, base: E.OptionStrategy, debit: bool) -> dict:
    """Sizing controls → OptionStrategy fields."""
    if acct["mode"] == "pct":
        modes = ["collateral", "premium", "risk", "notional"]
        sizing = st.radio("sizing", modes, index=modes.index(base.sizing) if base.sizing in modes else 0,
                          key="op_size", horizontal=True,
                          help="collateral = one unit of underlying notional per option unit (Cboe index style) · "
                               "premium = spend X% of equity per entry · risk = worst case loses X% · "
                               "notional = control X% of equity in underlying")
        st.caption(E.SIZING[sizing])
        pct = 0.0
        if sizing == "premium":
            pct = st.number_input("premium per entry, % of equity", 0.01, 100.0, max(0.01, base.size_pct * 100 or 2.0),
                                  0.1, key="op_sz_p", format="%.2f") / 100
        elif sizing == "risk":
            pct = st.number_input("max loss per entry, % of equity", 0.01, 100.0, max(0.01, base.size_pct * 100 or 2.0),
                                  0.1, key="op_sz_r", format="%.2f") / 100
        elif sizing == "notional":
            pct = st.number_input("notional per entry, % of equity", 1.0, 1000.0,
                                  max(1.0, base.size_pct * 100 or 100.0), 5.0, key="op_sz_n") / 100
        return {"sizing": sizing, "size_pct": pct, "capital": 0.0, "contracts": "fractional"}
    labels = {"contracts": "fixed # contracts", "premium_usd": "$ premium per entry",
              "premium": "% of options cash (premium)", "risk_usd": "$ max loss per entry",
              "risk": "% of options cash (max loss)", "notional_usd": "$ notional per entry"}
    order = list(labels)
    default = "premium_usd" if debit else "risk_usd"
    sizing = st.selectbox("size each entry by", order, index=order.index(default), format_func=labels.get,
                          key="op_size_usd",
                          help="fixed # contracts: the same count every time · $ premium: spend about this much per "
                               "entry · % of options cash: a share of the sleeve as it grows or shrinks · $ max loss: "
                               "size so the worst case at expiry loses this much (best for credit trades) · "
                               "$ notional: control this much underlying.")
    out = {"sizing": sizing, "size_pct": 0.0, "size_usd": 0.0, "size_contracts": 0, "capital": acct["sleeve"]}
    if sizing == "contracts":
        out["size_contracts"] = int(st.number_input("contracts per entry", 1, 100_000, 1, 1, key="op_sz_c"))
    elif sizing in ("premium", "risk"):
        out["size_pct"] = st.number_input("% of options cash per entry", 0.01, 100.0, 10.0, 0.5, key="op_sz_sp",
                                          format="%.2f") / 100
    else:
        out["size_usd"] = float(st.number_input("$ per entry", 1.0, 1e10, 500.0 if debit else 1_000.0, 50.0,
                                                key="op_sz_usd"))
    c1, c2 = st.columns(2)
    whole = c1.toggle("whole contracts", True, key="op_whole",
                      help="Round every entry DOWN to whole contracts. An entry that can't afford one contract is "
                           "skipped (counted in the POSITIONS tab). Off = idealised fractional contracts.")
    out["cash_check"] = c2.toggle("cash limit", True, key="op_cashchk",
                                  help="Never pay more premium than the options cash has free; an entry is scaled "
                                       "down (or skipped) instead of going into debt.")
    out["topup_usd"] = float(st.number_input(
        "top up options cash each month $", 0.0, 1e9, 0.0, 50.0, key="op_topup",
        help="Moved from idle cash into the options sleeve on the first trading day of every month — a standing "
             "hedge budget. Counted as a transfer, not a return. 0 = the sleeve lives on what it starts with."))
    out["multiplier"] = float(st.number_input("contract multiplier", 1.0, 1000.0, 100.0, 1.0, key="op_mult",
                                              help="Underlying units per contract. US equity and index options = 100. "
                                                   "Use 10 to model XSP (mini-SPX, 1/10 the size of SPX)."))
    out["contracts"] = "whole" if whole else "fractional"
    return out


def _strat_from_ui() -> tuple:
    st.markdown(section("01 · UNDERLYING & DATES", 0, help="What the options are on, and the backtest window."),
                unsafe_allow_html=True)
    und = st.selectbox("underlying", list(D.UNDERLYINGS), key="op_und",
                       help="SPX uses the full VIX term structure and is the calibrated case. Others use their own Cboe "
                            "vol index where one exists, else VIX scaled by realised vol.")
    a, b = C.date_range("op", _dt.date(2007, 1, 1), _dt.date(1990, 1, 1) if und == "SPX" else _dt.date(2005, 1, 1))
    start, end = a.isoformat(), C._iso_end(b)
    try:
        xin = _inputs(und, (start, end))
    except Exception:  # noqa: BLE001
        xin = None

    st.markdown(section("02 · STRATEGY", 1, help="Simple: pick buy/sell, the structure, strikes and expiry. "
                        "Template: a ready-made strategy with a plain-English breakdown. Advanced legs: edit every "
                        "leg in a table."), unsafe_allow_html=True)
    how = st.radio("build", ["simple", "template", "advanced legs"], horizontal=True, key="op_how",
                   help="All three build the same thing underneath: a list of option legs.")
    if how == "simple":
        legs, name = _simple_legs()
        base = E.OptionStrategy(name, legs, dte=60 if legs and legs[0].side > 0 else 30)
        stock = 0.0
        tpl = None
    else:
        tpl = _browse_templates()
        base = E.TEMPLATES[tpl]
        tg = E.TAGS[tpl]
        st.markdown("".join(C.uc.chip(f"{k.upper()} {v.upper()}", T.teal if v in ("credit", "short") else
                                      T.violet if v in ("debit", "long") else T.muted) for k, v in tg.items()),
                    unsafe_allow_html=True)
        legs = _advanced_legs(base, tpl) if how == "advanced legs" else list(base.legs)
        legs = legs or list(base.legs)
        name = base.name
        stock = float(st.number_input("stock units held", 0.0, 10.0, float(base.stock), 0.25, key=f"op_stock_{tpl}",
                                      help="Shares of the underlying held inside the strategy per option unit "
                                           "(1 = covered call / collar / protective put). In a $ account you can "
                                           "set this to 0 and list the shares under holdings instead."))

    c1, c2 = st.columns(2)
    cycles = ["monthly", "dte", "0dte"]
    roll = c1.radio("expiry", cycles, index=cycles.index(base.roll), key=f"op_roll_{tpl or 'simple'}",
                    format_func=lambda x: {"monthly": "monthly (3rd Friday)", "dte": "exact DTE",
                                           "0dte": "0DTE (same day)"}[x],
                    help="monthly: the standard 3rd-Friday contract nearest your DTE · exact DTE: an expiry exactly "
                         "N calendar days out (weeklys/dailies) · 0DTE: opened at the open, settled at the close.")
    extra = {"actual": True, "iv_scale": 1.0}
    if roll == "0dte":
        if und not in ("SPX", "SPY"):
            c2.caption(":orange[0DTE is modeled from VIX1D: SPX / SPY only. Using SPX.]")
        extra["actual"] = c2.toggle("only real SPX expiry days", True, key="op_0act",
                                    help="Fridays throughout; Wed from 2016, Mon from Aug 2016, every weekday from "
                                         "May 2022. Off = pretend every day had an expiry.")
        extra["iv_scale"] = c2.number_input("IV scale", 0.3, 3.0, 1.0, 0.05, key="op_0ivs",
                                            help="Multiplies VIX1D. The CALIBRATION tab shows how VIX1D compares with "
                                                 "the realised open-to-close move.")
        dte = 0
    else:
        dte = int(c2.number_input("days to expiry", 1, 1095, max(1, base.dte), 1, key=f"op_dte_{tpl or 'simple'}",
                                  help="Calendar days from entry to expiry, 1–1095 (3 years). With 'monthly' the "
                                       "nearest 3rd-Friday contract at least ~DTE−25 days out is used."))

    draft = replace(base, legs=legs, stock=stock, dte=dte, roll=roll)
    pv, s0 = None, None
    if xin is not None:
        try:
            pv, s0 = _today_preview(draft, xin), float(xin["S"].dropna().iloc[-1])
        except Exception:  # noqa: BLE001
            pv = None
    if pv is not None and not np.isfinite(pv["book"]):
        pv = None
    debit = (pv["book"] > 0) if pv is not None else (E.TAGS[tpl]["flow"] == "debit" if tpl else legs[0].side > 0)
    with st.expander("WHAT THIS STRATEGY DOES", expanded=how != "advanced legs"):
        st.markdown("\n".join(f"- {_md(x)}" for x in E.describe(draft)[: len(legs) + 2 + (1 if stock else 0)]))
        if pv is not None:
            st.caption(f"Today ({und} {s0:,.2f}): net {'debit' if debit else 'credit'} "
                       f"{abs(pv['book']):,.2f} per unit = \\${abs(pv['book']) * 100:,.0f} per contract of 100.")

    st.markdown(section("03 · ACCOUNT & SIZING", 2, help="How much money there is and how big each entry is."),
                unsafe_allow_html=True)
    acct = _account_ui(und)
    sz = _size_ui(acct, base, debit)
    if acct["mode"] == "usd" and pv is not None:
        probe = replace(draft, **{k: v for k, v in sz.items()})
        legs_now = [{"leg": L["leg"], "K": L["K"]} for L in pv["legs"]]
        u = E._units(probe, acct["sleeve"], s0, legs_now, pv["book"])
        n_c = u / probe.multiplier
        per_c = abs(pv["book"]) * probe.multiplier
        if probe.contracts == "whole" and n_c < 1:
            C.banner(T.amber, "UNDER ONE CONTRACT",
                     f"one contract costs ≈ ${per_c:,.0f} today ({und} {s0:,.0f} × {probe.multiplier:g}), so this "
                     "size buys none and entries will be skipped. Raise the size, pick a smaller underlying "
                     "(SPY ≈ 1/10 of SPX), set the multiplier to 10 for mini-SPX, or turn off whole contracts.")
        st.caption(f"≈ **{n_c:,.2f} contracts** per entry at today's prices"
                   + (f" → {int(np.floor(n_c + 1e-9))} whole" if probe.contracts == "whole" else "")
                   + f" · premium ≈ \\${max(pv['book'], 0) * u:,.0f}" if debit else
                   f"≈ **{n_c:,.2f} contracts** per entry at today's prices · credit ≈ \\${-pv['book'] * u:,.0f}")

    burn_box = st.empty()
    st.markdown(section("04 · ENTRIES & POSITIONS", 3, help="When new positions open, how many can be open at "
                        "once, and rolling."), unsafe_allow_html=True)
    entry = st.selectbox("open a new position", list(E.ENTRY), format_func=E.ENTRY.get, key="op_entry",
                         help="every day a slot is free = re-enter as soon as a position closes (the classic "
                              "index-style roll) · weekly / monthly / every N days = a fixed calendar, which together "
                              "with 'max open' builds a ladder of overlapping positions.")
    ent = {"entry": entry}
    c1, c2 = st.columns(2)
    if entry == "weekly":
        ent["entry_weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri"].index(
            c1.selectbox("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri"], index=4, key="op_ewd"))
    elif entry == "monthly":
        ent["entry_day"] = int(c1.number_input("trading day of month", 1, 23, 1, 1, key="op_eday",
                                               help="1 = first trading day of each month."))
    elif entry == "every_n":
        ent["entry_every"] = int(c1.number_input("every N trading days", 1, 1000, 5, 1, key="op_eevery",
                                                 help="1 = a new position every trading day."))
    ent["max_open"] = int(c2.number_input("max open at once", 1, 1000, 1 if entry == "when_flat" else 12, 1,
                                          key=f"op_maxopen_{entry}",
                                          help="Positions allowed open together. Monthly 60-DTE entries need ≥ 3 "
                                               "to stay fully laddered; a daily ladder of 60-DTE options needs ~42."))
    c1, c2 = st.columns(2)
    ent["roll_dte"] = int(c1.number_input("roll at DTE", 0, 1095, 0, 1, key="op_rolldte",
                                          help="When a position reaches this many days to expiry, close it and open "
                                               "a fresh one the same day. 0 = off."))
    ent["roll_ignores_filters"] = c2.toggle("rolls ignore filters", True, key="op_rollf",
                                            help="On: a roll always reopens. Off: the entry filters must pass too.")

    if acct["mode"] == "usd" and debit and pv is not None:
        per_yr = {"monthly": 12, "weekly": 52, "every_n": 252 / max(ent.get("entry_every", 5), 1)}.get(
            entry, 365 / max(dte, 1) * ent["max_open"])
        spend = {"premium_usd": sz.get("size_usd", 0.0),
                 "contracts": sz.get("size_contracts", 0) * max(pv["book"], 0) * sz.get("multiplier", 100.0)}.get(
            sz["sizing"])
        if spend:
            yearly = spend * per_yr - 12 * sz.get("topup_usd", 0.0)
            if yearly > 0 and acct["sleeve"] / yearly < 3:
                burn_box.caption(
                    f":orange[Spending ≈ \\${spend:,.0f} × {per_yr:.0f} entries/yr ≈ \\${spend * per_yr:,.0f}/yr of premium "
                    f"from \\${acct['sleeve']:,.0f} of options cash: if the options mostly expire worthless (as far-OTM "
                    f"ones usually do) the cash lasts about {acct['sleeve'] / yearly * 12:.0f} months, after which "
                    "entries are skipped. Add a monthly top-up, size by % of options cash, or spend less per entry.]")

    st.markdown(section("05 · EXITS", 4, help="When an open position is closed early. With none set, every "
                        "position is held to expiry."), unsafe_allow_html=True)
    st.caption(f"This is a **{'debit (you pay)' if debit else 'credit (you collect)'}** trade, so profit and loss are "
               f"measured against the {'premium paid' if debit else 'credit received'}.")
    c1, c2 = st.columns(2)
    ride = c1.toggle("ride to expiry (no take-profit)", key="op_ride",
                     help="Hold every position until expiry (or another exit below), however far it runs.")
    if debit:
        tp = 0.0 if ride else c1.number_input("take profit at % gain", 1.0, 1_000_000.0, 1000.0, 50.0, key="op_tp_d",
                                              help="Close when the position is worth this much MORE than you paid: "
                                                   "100 = doubled, 1000 = 11× the premium.") / 100
        sl = c2.number_input("stop at % of debit lost", 0.0, 100.0, 0.0, 5.0, key="op_sl_d",
                             help="Close when this share of the premium is lost. 0 = off; 100 = only at zero.") / 100
    else:
        tp = 0.0 if ride else c1.number_input("take profit at % of credit", 1.0, 100.0, 50.0, 5.0, key="op_tp_c",
                                              help="Close when this share of the credit has been earned (50 = "
                                                   "buy back at half the price you sold).") / 100
        sl = c2.number_input("stop at × credit lost", 0.0, 100.0, 0.0, 0.25, key="op_sl_c",
                             help="Close when the loss reaches this multiple of the credit (2 = lose twice what you "
                                  "collected). 0 = off.")
    c1, c2, c3 = st.columns(3)
    exit_dte = int(c1.number_input("exit at DTE", 0, 1095, 0, 1, key="op_xdte",
                                   help="Close (without reopening) at this many days to expiry. 0 = hold to expiry."))
    max_hold = int(c2.number_input("max hold days", 0, 3650, 0, 1, key="op_xhold",
                                   help="Close after this many calendar days. 0 = no limit."))
    dexit = float(c3.number_input("delta exit", 0.0, 10.0, 0.0, 0.05, key="op_xdelta",
                                  help="Close if |net delta| per unit exceeds this. 0 = off."))

    st.markdown(section("06 · ENTRY FILTERS", 5, help="Every filter only removes entries. The FILTERS tab shows "
                        "which days passed."), unsafe_allow_html=True)
    with st.expander("VIX, trend, weekday and trade-level filters", expanded=False):
        filters = _filters_ui()
    if filters:
        st.caption(f"{len(filters)} filter setting(s) active.")

    st.markdown(section("07 · COSTS & HOLDOUT", 6, help="Bid-ask cost on every open and close, the volatility "
                        "model knob, and the locked out-of-sample years."), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    spread = c1.number_input("half-spread % of premium", 0.0, 50.0, 3.0, 0.5, key="op_spr",
                             help="Paid on every leg at every open and close: half the bid-ask spread as a share of "
                                  "the option's price. Far-OTM options often trade wider (5–15%).") / 100
    ratio = c2.number_input("ATM / VIX ratio", 0.5, 1.2, E.ATM_RATIO, 0.01, key="op_ratio",
                            help=f"Calibrated {E.ATM_RATIO} on Cboe PUT. Changing it changes every price — keep the "
                                 "calibrated value unless you are stress-testing the model.")
    hold = C.lock_input("op")
    strat = replace(draft, take_profit=float(tp), stop_mult=float(sl), min_vix=0.0, spread_pct=spread,
                    atm_ratio=ratio, exit_dte=exit_dte, max_hold=max_hold, delta_exit=dexit, filters=filters,
                    **ent, **sz)
    return und, (start, end), strat, float(hold), extra, acct


def _md(x: str) -> str:
    """Escape for st.markdown / st.caption: HTML, and '$' (pairs of '$' render as LaTeX)."""
    return C.esc(x).replace("$", "\\$")


def _today_preview(strat: E.OptionStrategy, x: dict) -> dict:
    term_row = x["term"].ffill().iloc[-1][E.TERM_COLS].to_numpy(float)
    sk = float(x["skew"].ffill().iloc[-1])
    r = float(x["rf"].ffill().iloc[-1]) if len(x["rf"]) else 0.0
    s0 = float(x["S"].dropna().iloc[-1])
    sc = float(x["vol_scale"].ffill().iloc[-1]) if x.get("vol_scale") is not None and len(x["vol_scale"]) else 1.0
    iv1 = float(x["iv1d"].ffill().iloc[-1]) if len(x["iv1d"]) else None
    return E.entry_preview(strat, s0, term_row, sk, r, sc, iv1)


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
        und, start, strat, hold, extra, acct = _strat_from_ui()
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
                + ("" if und == "SPX" else f"{und} is priced off {x['vol_source']}; only SPX is calibrated directly. "
                   f"Strikes and premiums use {x.get('price_basis', 'adjusted')} prices"
                   + (" with the dividend yield priced in." if x.get("q") is not None else ".")))
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
        pre = "SLEEVE " if acct["mode"] == "usd" else ""
        if pre:
            st.caption(f"Top row: the **options sleeve on its own** (the \\${acct['sleeve']:,.0f} of options cash). "
                       "The ACCOUNT tab shows the whole account — holdings + options + idle cash — in dollars.")
        st.markdown(uk.kpis([
            {"label": f"{pre}SHARPE", "value": f"{s['Sharpe']:.2f}", "sub": f"{ul} {bs['Sharpe']:.2f}"},
            {"label": f"{pre}CAGR", "value": f"{s['CAGR'] * 100:.1f}%", "sub": f"{ul} {bs['CAGR'] * 100:.1f}%"},
            {"label": f"{pre}VOL", "value": f"{s['Vol'] * 100:.1f}%", "sub": f"{ul} {bs['Vol'] * 100:.1f}%"},
            {"label": f"{pre}MAX DD", "value": f"{s['MaxDD'] * 100:.1f}%", "color": T.coral,
             "sub": f"{ul} {bs['MaxDD'] * 100:.1f}%"},
            {"label": "SKEW", "value": f"{s['Skew']:.2f}", "color": T.coral if s["Skew"] < -0.5 else T.text,
             "sub": "daily returns"},
            {"label": "BETA", "value": f"{ab['beta']:.2f}", "sub": f"alpha {ab['alpha_ann'] * 100:+.1f}%/yr t {ab['alpha_t']:.1f}"},
        ]), unsafe_allow_html=True)
        if s["Skew"] < -1:
            C.banner(T.amber, "NEGATIVE SKEW", "returns come in small steady gains and rare large losses; Sharpe "
                     "overstates how comfortable this is — look at the drawdown and the worst months")

        _preview(und, start, strat, x)
        usd = acct["mode"] == "usd"
        tnames = (["ACCOUNT"] if usd else []) + ["EQUITY", "POSITIONS", "TRADES", "EXPOSURE", "MONTHLY",
                                                 "DISTRIBUTION", "SENSITIVITY", "CALIBRATION", "OUT-OF-SAMPLE",
                                                 "STATS", "FILTERS"]
        tabs = dict(zip(tnames, st.tabs(tnames)))
        if usd:
            with tabs["ACCOUNT"]:
                _account_tab(res, x, acct, und, end, locked if show_all else None)
        with tabs["POSITIONS"]:
            _positions_tab(res, strat, end)
        with tabs["FILTERS"]:
            _filters_tab(res, x, strat, end)
        with tabs["STATS"]:
            keys = [k for k in s if k in bs]
            st.dataframe(pd.DataFrame({"strategy": [C.fmt(k, s[k]) for k in keys],
                                       C.bench_label(f"{und} total return"): [C.fmt(k, bs[k]) for k in keys]},
                                      index=keys),
                         use_container_width=True)
            _trade_stats(res.trades)
            C.diversify_panel(r, tr, p, und, "op_div")
        with tabs["EQUITY"]:
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
        with tabs["EXPOSURE"]:
            ex = res.exposure.iloc[:end]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=ex.index, y=ex["delta"], name="net delta", line=dict(color=T.teal, width=1)), 1, 1)
            fig.add_trace(go.Scatter(x=ex.index, y=ex["vega_per_volpt"] * 100, name="vega (% per vol pt)",
                                     line=dict(color=T.violet, width=1)), 2, 1)
            fig.update_yaxes(title_text="delta / $", row=1, col=1)
            fig.update_yaxes(title_text="% per vol pt", row=2, col=1)
            up.show(fig, 380, "greeks per $ of equity", key="op_ex")
        with tabs["MONTHLY"]:
            up.show(C.monthly_fig(r), max(260, 22 * len(set(r.index.year)) + 60), "monthly returns %", key="op_mon")
        with tabs["DISTRIBUTION"]:
            up.show(C.dist_fig(r, trv), 320, "daily return distribution", key="op_dist")
            worst = ((1 + r).resample("ME").prod() - 1).nsmallest(5)
            st.caption("Worst months: " + ", ".join(f"{d:%Y-%m} {v * 100:.1f}%" for d, v in worst.items()))
        with tabs["SENSITIVITY"]:
            _sensitivity(und, start, strat, cut, extra)
        with tabs["TRADES"]:
            tcol = "open" if "open" in res.trades else "date"
            st.dataframe(res.trades.iloc[: max(1, len(res.trades))][::-1] if (show_all or res.trades.empty) else
                         res.trades[pd.to_datetime(res.trades[tcol]) < (locked or idx[-1] + pd.Timedelta(days=1))][::-1],
                         use_container_width=True, hide_index=True)
        with tabs["CALIBRATION"]:
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
        with tabs["OUT-OF-SAMPLE"]:
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
            ks.append(Knob(f"leg{i}", f"{side} {L.kind} Δ (leg {i + 1})", L.value, max(0.005, L.value / 4),
                           min(0.95, max(0.5, L.value * 2))))
        elif L.rule == "pct":
            ks.append(Knob(f"leg{i}", f"{side} {L.kind} strike % (leg {i + 1})", L.value, min(0.7, L.value - 0.1),
                           max(1.3, L.value + 0.1)))
    if strat.roll != "0dte":
        ks.append(Knob("dte", "DTE", strat.dte, 5, max(120, strat.dte * 2), integer=True))
    ks += [Knob("take_profit", "take profit", strat.take_profit, 0.0, max(3.0, strat.take_profit * 2)),
           Knob("stop_mult", "stop", strat.stop_mult, 0.0, 5.0),
           Knob("spread_pct", "half-spread %", strat.spread_pct, 0.0, 0.15)]
    if strat.sizing in ("premium", "risk", "notional"):
        ks.append(Knob("size_pct", f"{strat.sizing} budget", strat.size_pct, 0.001, 0.2 if strat.sizing != "notional"
                       else 3.0))
    elif strat.sizing in ("premium_usd", "risk_usd", "notional_usd"):
        ks.append(Knob("size_usd", "$ per entry", strat.size_usd, max(1.0, strat.size_usd / 5), strat.size_usd * 3))
    if strat.max_open > 1:
        ks.append(Knob("max_open", "max open", strat.max_open, 1, max(4, strat.max_open * 2), integer=True))
    if strat.roll != "0dte":
        ks += [Knob("exit_dte", "exit at DTE", strat.exit_dte, 0, max(30, strat.dte // 2), integer=True),
               Knob("max_hold", "max hold days", strat.max_hold, 0, max(90, strat.dte), integer=True)]
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
        elif k in ("dte", "exit_dte", "max_hold", "max_open", "roll_dte"):
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
        s0 = float(x["S"].dropna().iloc[-1])
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


# ---------------------------------------------------------------- account ----
@st.cache_data(ttl=3600, show_spinner=False)
def _holding_px(ticker: str) -> pd.Series:
    return C._market(ticker)["close"]


def _account_tab(res: E.OptionResult, x: dict, acct: dict, und: str, end: int, shade) -> None:
    """The whole account in dollars: holdings + options sleeve + idle cash, vs the same account without options."""
    sleeve = res.equity.iloc[:end]
    if sleeve.empty:
        return
    holds, px, bad = acct["holdings"], {}, []
    for t in holds:
        try:
            px[t] = _holding_px(t)
        except Exception:  # noqa: BLE001
            bad.append(t)
    if bad:
        st.caption(f":orange[no prices for {', '.join(bad)} — left out of the account]")
    rf = x["rf"]
    a = E.account_equity(sleeve, px, holds, acct["cash"], rf,
                         res.contrib.iloc[:end] if res.contrib is not None else None)
    # the same account with the options cash simply left in T-bills
    sl0 = acct["sleeve"] * (1 + (rf.reindex(sleeve.index).ffill().fillna(0.0) / 252).shift(1).fillna(0.0)).cumprod()
    b = E.account_equity(sl0, px, holds, acct["cash"], rf)
    yrs = max(len(a) / 252, 1e-9)
    cagr = lambda v: (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1 if v.iloc[0] > 0 and v.iloc[-1] > 0 else np.nan  # noqa: E731
    dd = lambda v: float((v / v.cummax() - 1).min())  # noqa: E731
    added = float(res.contrib.iloc[:end].sum()) if res.contrib is not None else 0.0
    opt_pnl = float(sleeve.iloc[-1] - acct["sleeve"] - added)
    st.markdown(uk.kpis([
        {"label": "ACCOUNT NOW", "value": f"${a['total'].iloc[-1]:,.0f}", "sub": f"from ${a['total'].iloc[0]:,.0f}"},
        {"label": "ACCOUNT CAGR", "value": f"{cagr(a['total']) * 100:.1f}%",
         "sub": f"without options {cagr(b['total']) * 100:.1f}%",
         "color": T.teal if cagr(a["total"]) >= cagr(b["total"]) else T.coral},
        {"label": "ACCOUNT MAX DD", "value": f"{dd(a['total']) * 100:.1f}%", "color": T.coral,
         "sub": f"without options {dd(b['total']) * 100:.1f}%"},
        {"label": "OPTIONS P&L", "value": f"{'+' if opt_pnl >= 0 else '−'}${abs(opt_pnl):,.0f}",
         "sub": f"sleeve now ${sleeve.iloc[-1]:,.0f}" + (f" · ${added:,.0f} topped up" if added else ""),
         "color": T.teal if opt_pnl >= 0 else T.coral},
        {"label": "ENTRIES", "value": f"{len(res.trades)}",
         "sub": (f"{sum(res.skipped.values())} skipped" if res.skipped else "none skipped")},
    ]), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b.index, y=b["total"], name="account without options", line=dict(color=T.amber,
                                                                                                width=1.2)))
    fig.add_trace(go.Scatter(x=a.index, y=a["total"], name="account with options", line=dict(color=T.teal, width=2)))
    for i, c in enumerate([c for c in a.columns if c.startswith("h:")]):
        fig.add_trace(go.Scatter(x=a.index, y=a[c], name=c[2:], line=dict(color=T.cat[(i + 2) % len(T.cat)],
                                                                           width=1), visible="legendonly"))
    fig.add_trace(go.Scatter(x=a.index, y=a["options"], name="options sleeve", line=dict(color=T.violet, width=1.2)))
    fig.add_trace(go.Scatter(x=a.index, y=a["cash"], name="idle cash", line=dict(color=T.faint, width=1)))
    if shade is not None:
        fig.add_vrect(x0=shade, x1=a.index[-1], fillcolor=up.fill(T.violet, 0.10), line_width=0)
    fig.update_yaxes(title_text="$", tickprefix="$")
    up.show(fig, 440, "account value · click legend items to show holdings", key="op_acct_fig")
    if (a["cash"] < 0).any():
        st.caption(":orange[Idle cash goes negative: top-ups / holdings exceed the cash available, so the "
                   "shortfall is treated as borrowed at the T-bill rate.]")
    st.caption("Holdings are bought on the start date and held (Yahoo adjusted closes, so dividends are included). "
               "Idle cash and unused options cash earn the T-bill rate. 'Without options' keeps the options cash in "
               "T-bills instead — the difference is what the options strategy added or cost.")


def _positions_tab(res: E.OptionResult, strat: E.OptionStrategy, end: int) -> None:
    if res.positions is None:
        st.caption("0DTE positions open and close the same day.")
        return
    pos = res.positions.iloc[:end]
    ctr = res.contracts.iloc[:end] if res.contracts is not None else None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=pos.index, y=pos, name="positions open", line=dict(color=T.teal, width=1, shape="hv")),
                  1, 1)
    if ctr is not None and strat.capital > 0:
        fig.add_trace(go.Scatter(x=ctr.index, y=ctr, name="contracts open", line=dict(color=T.violet, width=1,
                                                                                         shape="hv")), 2, 1)
    fig.update_yaxes(title_text="positions", row=1, col=1)
    fig.update_yaxes(title_text="contracts", row=2, col=1)
    up.show(fig, 360, "open positions and contracts", key="op_pos")
    st.markdown(uk.kpis([
        {"label": "AVG OPEN", "value": f"{pos.mean():.1f}", "sub": f"max {int(pos.max())} of {strat.max_open} allowed"},
        {"label": "TIME INVESTED", "value": f"{(pos > 0).mean() * 100:.0f}%", "sub": "days with ≥1 position"},
        *([{"label": f"SKIPPED · {k.upper()}", "value": f"{v}", "color": T.amber} for k, v in res.skipped.items()]),
    ]), unsafe_allow_html=True)
    if res.skipped.get("not enough cash"):
        st.caption("Entries skipped for cash: the options sleeve ran low. Raise the options cash, shrink each entry, "
                   "or take profits sooner.")

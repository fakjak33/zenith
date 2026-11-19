"""BACKTEST / ETF LAB — design a fund, backtest it with costs and fees (ported PARALLAX lab).

Explicit baskets or screen-and-rank rules over a universe, several weighting
schemes, rebalancing, expense ratio and borrow costs, benchmark comparison,
holdings overlap and issuer economics. Guard rails: survivorship and
look-ahead are flagged loudly — fundamental factors are a *current* snapshot.
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
from ..data.providers import clean_ticker, get_panel
from ..engine import metrics
from ..etflab import economics as econ
from ..etflab import overlap as ov
from ..etflab import portfolio as pf
from ..etflab import presets, screens
from . import common as C

UNIVERSES = ["Sector ETFs", "Broad ETFs", "Industry ETFs", "Futures-like ETFs", "IWB (large-cap proxy)", "Stocks (500)"]
RANK = {"Momentum (12-1)": "momentum", "Trailing return": "trailing_return",
        "Prior calendar-year return": "calendar_return", "Low volatility": "low_volatility",
        "High volatility": "volatility", "High beta": "beta", "Low beta": "low_beta",
        "Shallowest drawdown": "max_drawdown", "5-yr dividend growth": "dividend_growth",
        "Low P/E · snapshot": "low_pe", "Low forward P/E · snapshot": "low_forward_pe",
        "High FCF/share · snapshot": "fcf_per_share", "High profit margin · snapshot": "profit_margin",
        "Low price/book · snapshot": "low_price_to_book", "High dividend yield · snapshot": "dividend_yield"}
WEIGHTS = {"Equal": "equal", "Inverse volatility": "inverse_vol", "Market cap · snapshot": "market_cap",
           "FCF-weighted · snapshot": "fcf_weighted", "Revenue-weighted · snapshot": "revenue_weighted",
           "Manual": "manual"}
BENCH = ["SPY", "QQQ", "IWM", "EFA", "AGG", "TLT", "GLD"]
_FINDINGS = [
    {"stat": "Fees are the most reliable predictor of future fund returns: lower-cost funds beat higher-cost "
             "ones across almost every category.", "cite": "Morningstar (Kinnel, 2016)"},
    {"stat": "Backtests built on today's index members overstate returns because the losers that dropped out "
             "are missing.", "cite": "Elton, Gruber & Blake (1996)"},
]


@st.cache_data(ttl=3600, show_spinner=False)
def _panel(tickers: tuple, start: str) -> pd.DataFrame:
    return get_panel(list(tickers), start=start, field="close")


@st.cache_data(ttl=3600, show_spinner=False, max_entries=24)
def _design(tickers: tuple, start: str, spec, cfg):
    """Rebalance schedule + backtest, cached on the design (every widget change used to redo both)."""
    panel = _panel(tickers, start)
    mkt = panel["SPY"] if "SPY" in panel else panel.iloc[:, 0]
    dts = pf.rebalance_dates(panel.index, spec.rebalance)
    ws = screens.build_weight_schedule(spec, panel, dts, mkt)
    res = pf.run_portfolio(panel, ws, cfg) if not (ws.abs().sum(axis=1) == 0).all() else None
    return dts, ws, res


def _builder():
    st.markdown(section("01 · FUND DESIGN", 0), unsafe_allow_html=True)
    how = st.radio("start from", ["Preset", "Build your own"], horizontal=True, key="ef_src")
    if how == "Preset":
        names = [p.name for p in presets.PRESETS if p.enabled]
        spec = presets.by_name(st.selectbox("example fund", names, key="ef_preset"))
        st.caption(spec.notes)
    else:
        sel_mode = st.radio("holdings", ["Basket", "Screen & rank"], horizontal=True, key="ef_mode")
        weighting = WEIGHTS[st.selectbox("weighting", list(WEIGHTS), key="ef_w")]
        c1, c2 = st.columns(2)
        max_w = c1.slider("max weight", 0.05, 1.0, 1.0, 0.05, key="ef_maxw")
        rebal = c2.selectbox("rebalance", ["Weekly", "Monthly", "Quarterly", "Biannual", "Annual"], index=1,
                             key="ef_rebal")
        manual, direction = None, "long"
        if sel_mode == "Basket":
            grp = st.selectbox("pick from", UNIVERSES[:4], key="ef_grp")
            opts = screens.resolve_universe(grp)
            picks = st.multiselect("holdings", opts, default=opts[:4], key="ef_picks")
            for raw in st.text_input("add tickers (comma-separated)", "", key="ef_add").split(","):
                try:
                    if raw.strip():
                        picks.append(clean_ticker(raw))
                except ValueError:
                    pass
            picks = list(dict.fromkeys(picks))
            if weighting == "manual" and picks:
                manual = {t: st.number_input(f"{t} weight", 0.0, 100.0, 1.0, 0.5, key=f"ef_mw_{t}") for t in picks}
            sel = screens.SelectionSpec(explicit=picks)
        else:
            uni = st.selectbox("universe", UNIVERSES, key="ef_uni")
            rf = RANK[st.selectbox("rank by", list(RANK), key="ef_rf")]
            c1, c2 = st.columns(2)
            lb = int(c1.number_input("lookback (days)", 21, 1000, 252, 21, key="ef_lb"))
            top = int(c2.number_input("# longs", 1, 100, 10, 1, key="ef_top"))
            direction = st.radio("direction", ["long", "short", "long_short"], horizontal=True, key="ef_dir")
            bottom = int(st.number_input("# shorts", 1, 100, 10, 1, key="ef_bot")) if direction != "long" else None
            sel = screens.SelectionSpec(universe=uni, rank_factor=rf, rank_lookback=lb, top_n=top, bottom_n=bottom)
        spec = screens.ETFSpec(name="Custom fund", selection=sel, weighting=weighting, rebalance=rebal,
                               direction=direction, manual_weights=manual, max_weight=max_w)

    st.markdown(section("02 · COSTS & FEES", 1), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cost = c1.number_input("commission bps", 0.0, 50.0, 1.0, 0.5, key="ef_cost")
    slip = c2.number_input("slippage bps", 0.0, 50.0, 1.0, 0.5, key="ef_slip")
    er = st.number_input("expense ratio %/yr", 0.0, 3.0, 0.40, 0.05, key="ef_er") / 100
    borrow = st.number_input("short borrow bps/yr", 0.0, 1000.0, 50.0, 10.0, key="ef_borrow") \
        if spec.direction != "long" else 0.0
    cfg = pf.PortfolioConfig(capital=1_000_000.0, rebalance=spec.rebalance, cost_bps=cost, slippage_bps=slip,
                             expense_ratio_annual=er, borrow_bps=borrow, max_weight=spec.max_weight,
                             direction=spec.direction)
    st.markdown(section("03 · WINDOW & BENCHMARKS", 2), unsafe_allow_html=True)
    start = st.number_input("from", 2000, 2024, 2012, key="ef_start")
    bench = st.multiselect("benchmarks", BENCH, default=["SPY"], key="ef_bench")
    return spec, cfg, f"{int(start)}-01-01", bench


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        spec, cfg, start, bench = _builder()
    with right:
        C.banner(T.amber, "SURVIVORSHIP", "universes are today's members — names that failed or were dropped are "
                 "missing, which flatters historical results")
        if screens.spec_is_lookahead(spec):
            C.banner(T.coral, "LOOK-AHEAD", "this design screens or weights on today's fundamental snapshot applied "
                     "to the whole history — illustrative only, not a historical result")
        need = set(spec.selection.explicit) if spec.selection.explicit else set(
            screens.resolve_universe(spec.selection.universe))
        if not need:
            st.caption("Pick some holdings.")
            return
        need |= set(bench) | {"SPY"}
        key = tuple(sorted(need))
        import time as _t
        t0 = _t.perf_counter()
        try:
            with st.status(f"loading {len(key)} price histories…", expanded=False) as stt:
                panel = _panel(key, start)
                t1 = _t.perf_counter()
                stt.update(label=f"building {spec.rebalance.lower()} rebalances and running the backtest…")
                dts, ws, res = _design(key, start, spec, cfg)
                t2 = _t.perf_counter()
                src = panel.attrs.get("from_store", 0)
                stt.update(label=f"{panel.shape[1]} prices in {t1 - t0:.1f}s ({src} from the EPHEMERIS store) · "
                                 f"design in {t2 - t1:.1f}s", state="complete")
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "NO DATA", str(exc))
            return
        if panel.attrs.get("errors"):
            st.caption(f"skipped {len(panel.attrs['errors'])} tickers with no data: "
                       + ", ".join(list(panel.attrs["errors"])[:12]))
        mkt = panel["SPY"] if "SPY" in panel else panel.iloc[:, 0]
        if res is None:
            C.banner(T.amber, "EMPTY", "this design selects nothing in the window — widen it or relax the screen")
            return
        r = res.returns
        bret = {b: panel[b].pct_change() for b in bench if b in panel}
        b0 = next(iter(bret.values()), mkt.pct_change())
        bs = metrics.summary(b0.reindex(r.index).fillna(0), periods=252)
        st.markdown(uk.kpis([
            {"label": "CAGR", "value": f"{res.stats['CAGR'] * 100:.1f}%", "sub": f"{bench[0] if bench else 'SPY'} {bs['CAGR'] * 100:.1f}%"},
            {"label": "SHARPE", "value": f"{res.stats['Sharpe']:.2f}", "sub": f"bench {bs['Sharpe']:.2f}",
             "color": T.teal if res.stats["Sharpe"] > bs["Sharpe"] else T.amber},
            {"label": "VOL", "value": f"{res.stats['AnnVol%']:.1f}%"},
            {"label": "MAX DD", "value": f"{res.stats['MaxDD'] * 100:.1f}%", "color": T.coral,
             "sub": f"bench {bs['MaxDD'] * 100:.1f}%"},
            {"label": "TURNOVER / REBAL", "value": f"{res.stats['AvgTurnover'] * 100:.0f}%"},
            {"label": "COSTS PAID", "value": f"{res.stats['TotalCost%']:.2f}%", "sub": "trading + fees"},
        ]), unsafe_allow_html=True)

        tabs = st.tabs(["BACKTEST", "HOLDINGS", "COMPARE", "OVERLAP", "FUND ECONOMICS", "STATS", "SENSITIVITY"])
        with tabs[6]:
            _sensitivity(spec, cfg, panel, mkt)
        with tabs[5]:
            ks = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "Skew", "HitRate"]
            full = metrics.summary(r, periods=252)
            st.dataframe(pd.DataFrame({spec.name: [C.fmt(k, full.get(k)) for k in ks],
                                       (bench[0] if bench else "SPY"): [C.fmt(k, bs.get(k)) for k in ks]}, index=ks),
                         use_container_width=True)
            C.diversify_panel(r, b0.reindex(r.index), 252, bench[0] if bench else "SPY", "ef_div")
        with tabs[0]:
            b = b0.reindex(r.index).fillna(0)
            wa, wb = C.window_pick(r.index, "ef_win")
            rw, bw = C.clip(r, wa, wb), C.clip(b, wa, wb)
            if (wa, wb) != (r.index[0], r.index[-1]):
                C.window_kpis(rw, bw, 252, bench[0] if bench else "SPY")
            up.show(C.equity_fig(rw, bw * (rw.std() / bw.std()) if bw.std() > 0 else bw), 440,
                    f"{spec.name} vs {bench[0] if bench else 'SPY'} (vol-matched)", key="ef_eq")
            up.show(C.monthly_fig(rw), max(260, 22 * len(set(rw.index.year)) + 60), "monthly returns %", key="ef_mon")
        with tabs[1]:
            latest = ws.iloc[-1]
            latest = latest[latest != 0].sort_values(key=abs, ascending=False)
            fig = go.Figure(go.Bar(x=[str(t) for t in latest.index], y=latest.values * 100,
                                   marker_color=[T.teal if v > 0 else T.coral for v in latest.values]))
            fig.update_layout(hovermode="closest")
            fig.update_yaxes(title_text="weight %")
            up.show(fig, 320, "holdings at the latest rebalance", key="ef_hold")
            nz = ws.columns[(ws != 0).any()]
            fig = go.Figure()
            for i, t in enumerate(nz[:40]):
                w = res.weights[t].reindex(res.weights.index).fillna(0) * 100
                fig.add_trace(go.Scatter(x=w.index, y=w, name=str(t), stackgroup="w",
                                         line=dict(width=0.5, color=T.cat[i % len(T.cat)])))
            up.show(fig, 340, "weights through time (top 40 names)", key="ef_wt")
            st.caption(f"{panel.shape[1]} tickers loaded · {len(dts)} {spec.rebalance.lower()} rebalances · "
                       f"weighting {spec.weighting}")
        with tabs[2]:
            series = {spec.name: r, **bret}
            rows = []
            for nm, s in series.items():
                s = s.dropna()
                rows.append({"fund": nm, "CAGR": C.fmt("CAGR", metrics.cagr(s, 252)),
                             "Sharpe": C.fmt("x", metrics.sharpe(s, 252)), "Vol": C.fmt("Vol", metrics.annual_vol(s, 252)),
                             "MaxDD": C.fmt("MaxDD", metrics.max_drawdown(s)),
                             "up capture": f"{metrics.upside_capture(s, b0):.0f}%",
                             "down capture": f"{metrics.downside_capture(s, b0):.0f}%"})
            st.dataframe(pd.DataFrame(rows).set_index("fund"), use_container_width=True)
            corr = pd.DataFrame(series).dropna(how="all").corr()
            if len(corr) > 1:
                fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
                                           colorscale=up.DIVERGING, text=np.round(corr.values, 2),
                                           texttemplate="%{text}", xgap=1, ygap=1))
                fig.update_layout(hovermode="closest")
                up.show(fig, 120 + 40 * len(corr), "return correlation", key="ef_corr")
        with tabs[3]:
            mine = {t: float(v) for t, v in ws.iloc[-1].items() if v != 0}
            if st.button("COMPARE OVERLAP WITH THE PRESET FUNDS", key="ef_ov"):
                baskets = {spec.name: mine}
                for p in presets.PRESETS:
                    if not p.enabled or p.name == spec.name:
                        continue
                    try:
                        pneed = set(p.selection.explicit) if p.selection.explicit else set(
                            screens.resolve_universe(p.selection.universe))
                        pp = _panel(tuple(sorted(pneed | {"SPY"})), start)
                        pws = screens.build_weight_schedule(p, pp, pf.rebalance_dates(pp.index, p.rebalance)[-1:],
                                                            pp["SPY"] if "SPY" in pp else pp.iloc[:, 0])
                        baskets[p.name] = {t: float(v) for t, v in pws.iloc[-1].items() if v != 0}
                    except Exception:  # noqa: BLE001
                        continue
                C.memo()["ef_overlap"] = ov.overlap_matrix(baskets)
            m = C.memo().get("ef_overlap")
            if m is None:
                st.caption("Weighted overlap (Σ min weight) between this fund and the preset designs. Third-party ETF "
                           "holdings aren't free, so it compares funds built here.")
            else:
                z = m.values.astype(float)
                fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in m.columns], y=[str(i) for i in m.index], zmin=0,
                                           zmax=1, colorscale=up.SEQUENTIAL, text=np.round(z, 2),
                                           texttemplate="%{text}", xgap=1, ygap=1))
                fig.update_layout(hovermode="closest")
                up.show(fig, 420, "weighted holdings overlap", key="ef_ovm")
        with tabs[4]:
            c1, c2, c3 = st.columns(3)
            aum = c1.number_input("AUM $", 1e6, 1e12, 100e6, 1e6, key="ef_aum")
            erp = c2.number_input("expense ratio %", 0.0, 3.0, cfg.expense_ratio_annual * 100, 0.05, key="ef_eer") / 100
            growth = c3.number_input("AUM growth %/yr", -50.0, 200.0, 10.0, 5.0, key="ef_g") / 100
            c4, c5, c6 = st.columns(3)
            start_cost = c4.number_input("startup $", 0.0, 1e8, 400_000.0, 50_000.0, key="ef_sc")
            maint = c5.number_input("maintenance $/yr", 0.0, 1e8, 150_000.0, 25_000.0, key="ef_mt")
            lic = c6.number_input("index licensing bps", 0.0, 50.0, 3.0, 0.5, key="ef_lic")
            years = int(st.slider("years", 1, 15, 5, key="ef_yrs"))
            ecfg = econ.FundEconConfig(aum=aum, expense_ratio=erp, expected_aum_growth=growth, startup_cost=start_cost,
                                       annual_maintenance=maint, index_licensing_bps=lic,
                                       num_holdings=int((ws.iloc[-1] != 0).sum()))
            edf = econ.economics(ecfg, years)
            be = econ.breakeven_aum(ecfg)
            drag = econ.investor_fee_drag(max(res.stats["CAGR"], 0.0), erp, years)
            st.markdown(uk.kpis([
                {"label": "YEAR-1 FEE REVENUE", "value": f"${edf.loc[1, 'Revenue']:,.0f}"},
                {"label": f"NET AFTER {years}Y", "value": f"${edf.loc[years, 'CumNetProfit']:,.0f}",
                 "color": T.teal if edf.loc[years, "CumNetProfit"] > 0 else T.coral},
                {"label": "BREAKEVEN AUM", "value": "∞" if be == float("inf") else f"${be:,.0f}"},
                {"label": f"FEE COST / $10K ({years}Y)", "value": f"${drag['fee_cost']:,.0f}",
                 "sub": f"net {drag['net_return_annual'] * 100:.1f}%/yr"},
            ]), unsafe_allow_html=True)
            st.dataframe(edf.map(lambda v: f"${v:,.0f}"), use_container_width=True)


def etf_knobs(spec) -> list:
    from ..etflab import weighting
    from .sensitivity_panel import Knob
    sel = spec.selection
    ks = []
    if sel.rank_factor or sel.short_rank_factor:
        ks += [Knob("top_n", "top N", sel.top_n or 10, 1, 50, integer=True),
               Knob("rank_lookback", "rank lookback (bars)", sel.rank_lookback, 21, 504, integer=True, log=True),
               Knob("rank_skip", "skip recent bars", sel.rank_skip, 0, 63, integer=True)]
        if spec.direction == "long_short":
            ks.append(Knob("bottom_n", "bottom N (shorts)", sel.bottom_n or sel.top_n or 5, 1, 50, integer=True))
    ks += [Knob("max_weight", "max weight per name", spec.max_weight, 0.02, 1.0),
           Knob("rebalance", "rebalance", spec.rebalance,
                choices=["Weekly", "Monthly", "Quarterly", "Biannual", "Annual"]),
           Knob("weighting", "weighting", spec.weighting,
                choices=[m for m in weighting.METHODS if m not in weighting.SNAPSHOT_METHODS])]
    return ks


def _sensitivity(spec, cfg, panel, mkt) -> None:
    from dataclasses import replace as _rep

    from .sensitivity_panel import panel as sens_panel

    def run(vals, cost_mult=1.0):
        sel_kw = {k: int(v) for k, v in vals.items() if k in ("top_n", "rank_lookback", "rank_skip", "bottom_n")}
        spec_kw = {k: v for k, v in vals.items() if k in ("max_weight", "rebalance", "weighting")}
        sp = _rep(spec, selection=_rep(spec.selection, **sel_kw), **spec_kw)
        c = _rep(cfg, cost_bps=cfg.cost_bps * cost_mult, slippage_bps=cfg.slippage_bps * cost_mult,
                 max_weight=sp.max_weight)
        ws = screens.build_weight_schedule(sp, panel, pf.rebalance_dates(panel.index, sp.rebalance), mkt)
        return pf.run_portfolio(panel, ws, c).stats

    sens_panel(etf_knobs(spec), run, "ef_sens", ["Sharpe", "CAGR", "MaxDD", "Calmar", "AnnVol%", "AvgTurnover",
                                                  "TotalCost%"],
               state=f"{spec}|{cfg}|{panel.shape}", supports_costs=True,
               note="Each cell rebuilds the whole rebalance schedule — big universes take a few seconds per cell, "
                    "so start with a small grid.")

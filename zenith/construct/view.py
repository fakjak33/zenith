"""PORTFOLIO / CONSTRUCT — a guided, interactive course in portfolio construction.

GUIDE walks through eight research-backed ideas, each with the equation, sliders and a live
chart on real (free) data. SANDBOX lets you build a portfolio of ETFs, stocks or saved LAB
strategies and see every idea applied to it. SOURCES lists the research behind each chapter.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import ui_kit as uk
from .. import ui_plotly as up
from ..config import DATA_DIR
from ..config import THEME as T
from ..ui_theme import key_findings, section
from . import engine as K

DEFAULT = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC", "VNQ"]
CHAPTERS = ["01 DIVERSIFY", "02 VOL DRAG", "03 SHANNON'S DEMON", "04 SHOULD I ADD IT?", "05 WEIGHTING",
            "06 REBALANCING", "07 LEVERAGE", "08 CRISES"]

_FINDINGS = [
    {"stat": "Across 14 optimisation models and 7 datasets, none beat equal weighting consistently out of "
             "sample: estimation error swamps the gains from optimising.",
     "cite": "DeMiguel, Garlappi & Uppal (2009)"},
    {"stat": "A new return stream improves a portfolio's Sharpe ratio exactly when its own Sharpe beats its "
             "correlation to the portfolio times the portfolio's Sharpe.", "cite": "mean-variance identity"},
    {"stat": "A rebalanced portfolio's compound growth exceeds the weighted growth of its parts by "
             "½(Σwσ² − σ²ₚ): the diversification return.", "cite": "Booth & Fama (1992); Fernholz & Shay (1982)"},
    {"stat": "Correlations between equity markets rise in bear markets — diversification is weakest when it "
             "is needed most.", "cite": "Longin & Solnik (2001)"},
]


# ------------------------------------------------------------------ data ----
@st.cache_data(ttl=3600, show_spinner="loading prices…")
def _prices(tickers: tuple, start: str = "2005-01-01") -> pd.DataFrame:
    from ..backtest.data.providers import get_panel
    return get_panel(list(tickers), start=start, field="close")


def _returns(tickers, start="2005-01-01") -> pd.DataFrame:
    px = _prices(tuple(sorted(set(tickers))), start)
    return px[[t for t in tickers if t in px]].pct_change().dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def _weights_cached(scheme: str, R: pd.DataFrame) -> np.ndarray:
    return K.weights_for(scheme, R)


def _ticker_input(label: str, default: list[str], key: str) -> list[str]:
    raw = st.text_input(label, ", ".join(default), key=key, help="Comma-separated Yahoo tickers.")
    out = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return list(dict.fromkeys(out))[:15] or default


def _formula(tex: str, note: str = "") -> None:
    st.latex(tex)
    if note:
        st.caption(note)


def _pct(x, d=1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


# ============================================================ chapters ======
def ch_diversify() -> None:
    st.markdown(section("01 · DIVERSIFICATION — how much risk can mixing remove?", 0), unsafe_allow_html=True)
    st.markdown("Adding assets lowers risk only as far as their correlation allows. With *N* equal-weight assets "
                "of the same volatility σ and pairwise correlation ρ:")
    _formula(r"\sigma_p=\sigma\sqrt{\tfrac{1}{N}+\left(1-\tfrac{1}{N}\right)\rho}\;\xrightarrow{N\to\infty}\;\sigma\sqrt{\rho}",
             "Past a handful of assets the curve flattens: once you own 10 things that are 0.8 correlated, the "
             "11th barely helps. Lower correlation, not more line items, is what diversifies.")
    c1, c2, c3 = st.columns(3)
    sig = c1.slider("asset vol σ", 0.05, 0.50, 0.20, 0.01, key="cn_d_sig")
    rho = c2.slider("pairwise correlation ρ", -0.2, 1.0, 0.30, 0.05, key="cn_d_rho")
    nmax = c3.slider("up to N assets", 5, 100, 30, 5, key="cn_d_n")
    n = np.arange(1, nmax + 1)
    fig = go.Figure()
    for r_, col in ((0.0, T.teal), (0.2, T.cyan), (0.5, T.violet), (0.8, T.amber)):
        fig.add_trace(go.Scatter(x=n, y=K.ew_portfolio_vol(n, sig, max(r_, -1 / nmax)), name=f"ρ = {r_:.1f}",
                                 line=dict(color=col, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=n, y=K.ew_portfolio_vol(n, sig, rho), name=f"your ρ = {rho:.2f}",
                             line=dict(color=T.text, width=2.4)))
    fig.add_hline(y=sig * np.sqrt(max(rho, 0)), line=dict(color=T.coral, dash="dash", width=1),
                  annotation_text="floor σ√ρ", annotation_font=dict(size=9, color=T.coral))
    fig.update_yaxes(tickformat=".0%", title_text="portfolio vol", rangemode="tozero")
    fig.update_xaxes(title_text="number of assets")
    up.show(fig, 360, "portfolio vol vs number of assets", key="cn_d_fig", scale_toggle=False)
    idm = 1 / np.sqrt(1 / nmax + (1 - 1 / nmax) * max(rho, 0))
    st.markdown(uk.kpis([
        {"label": f"VOL WITH {nmax}", "value": _pct(float(K.ew_portfolio_vol(nmax, sig, rho)))},
        {"label": "FLOOR", "value": _pct(sig * np.sqrt(max(rho, 0)))},
        {"label": "RISK REMOVED", "value": _pct(1 - float(K.ew_portfolio_vol(nmax, sig, rho)) / sig, 0)},
        {"label": "DIVERSIFICATION MULTIPLIER", "value": f"{min(idm, 2.5):.2f}x",
         "sub": "Carver caps it at 2.5"},
    ]), unsafe_allow_html=True)
    st.markdown("**Carver's diversification multiplier.** Because the mix is calmer than its parts, a "
                "vol-targeted portfolio can scale every position up by")
    _formula(r"\text{IDM}=\frac{1}{\sqrt{w^\top H\,w}}",
             "H is the correlation matrix (negative correlations floored at 0 to be conservative) and w the "
             "weights. Two uncorrelated assets: √2 = 1.41x.")
    with st.expander("On real ETFs: correlation matrix and multiplier"):
        tk = _ticker_input("tickers", DEFAULT, "cn_d_tk")
        R = _returns(tk)
        if R.shape[1] >= 2:
            c = R.corr()
            fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=list(c.columns), y=list(c.index), zmin=-1, zmax=1,
                                       colorscale=up.DIVERGING, text=np.round(c.to_numpy(), 2),
                                       texttemplate="%{text}", xgap=1, ygap=1, colorbar=dict(thickness=10)))
            fig.update_yaxes(autorange="reversed", showspikes=False)
            fig.update_xaxes(showspikes=False)
            fig.update_layout(hovermode="closest")
            up.show(fig, 120 + 36 * len(c), "daily return correlation", key="cn_d_hm")
            w = np.full(len(c), 1 / len(c))
            st.caption(f"Equal weights → IDM {K.diversification_multiplier(w, c.to_numpy()):.2f}x · average pairwise "
                       f"ρ {c.to_numpy()[np.triu_indices(len(c), 1)].mean():.2f}")


def ch_drag() -> None:
    st.markdown(section("02 · VOLATILITY DRAG — why the average return isn't what you earn", 1),
                unsafe_allow_html=True)
    st.markdown("Returns compound. A −50% year needs a +100% year to break even, so volatility itself eats "
                "growth. For log-normal returns:")
    _formula(r"g\;\approx\;\mu-\tfrac{1}{2}\sigma^2",
             "g = compound (geometric) growth, μ = arithmetic average, σ = volatility. Two assets with the same "
             "average but different volatility do NOT end up with the same wealth.")
    c1, c2 = st.columns(2)
    mu = c1.slider("arithmetic average μ", 0.0, 0.20, 0.08, 0.005, key="cn_v_mu")
    sig = c2.slider("volatility σ", 0.05, 0.80, 0.20, 0.01, key="cn_v_sig")
    s = np.linspace(0.0, 0.8, 81)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=np.full_like(s, mu), name="arithmetic μ", line=dict(color=T.faint, dash="dot")))
    fig.add_trace(go.Scatter(x=s, y=K.geometric_from_arithmetic(mu, s), name="compound growth g",
                             line=dict(color=T.teal, width=2.2)))
    fig.add_trace(go.Scatter(x=[sig], y=[float(K.geometric_from_arithmetic(mu, sig))], mode="markers",
                             marker=dict(size=12, color=T.amber), name="you"))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_xaxes(tickformat=".0%", title_text="volatility")
    fig.update_yaxes(tickformat=".0%", title_text="annual growth")
    up.show(fig, 340, "same average, more volatility, less growth", key="cn_v_fig", scale_toggle=False)
    g = float(K.geometric_from_arithmetic(mu, sig))
    st.markdown(uk.kpis([
        {"label": "COMPOUND GROWTH", "value": _pct(g), "color": T.teal if g > 0 else T.coral},
        {"label": "DRAG", "value": _pct(0.5 * sig ** 2), "sub": "σ²/2", "color": T.coral},
        {"label": "$100 AFTER 20 YRS", "value": f"${100 * np.exp(g * 20):,.0f}", "sub": f"vs ${100 * np.exp(mu * 20):,.0f} at μ"},
        {"label": "ZERO-GROWTH VOL", "value": _pct(np.sqrt(2 * mu)) if mu > 0 else "—", "sub": "σ where g = 0"},
    ]), unsafe_allow_html=True)
    st.caption("This is why diversification and rebalancing raise *compound* returns even when they don't raise "
               "the average: they cut σ, and the drag falls with σ² (chapter 03).")


def ch_shannon() -> None:
    st.markdown(section("03 · SHANNON'S DEMON — profit from a coin flip by rebalancing", 2), unsafe_allow_html=True)
    st.markdown("Claude Shannon's thought experiment: an asset that doubles (+100%) or halves (−50%) on a fair "
                "coin. Its **hit rate is 50/50** and its expected *log* growth is exactly zero — most paths drift to "
                "nothing. Now hold 50% in it and 50% in cash, and **rebalance back to 50/50 after every flip**. "
                "Each period you sell some after a rise and buy after a fall; the mix grows.")
    _formula(r"g(w)=p\,\ln(1+w\,u)+(1-p)\,\ln(1-w\,d)\qquad w^*=\frac{p\,u-(1-p)\,d}{u\,d}",
             "u / d = up and down move, p = probability of up, w = fraction held in the asset. w* is the "
             "growth-optimal (Kelly) fraction.")
    c1, c2, c3, c4 = st.columns(4)
    upm = c1.slider("up move", 0.05, 2.0, 1.0, 0.05, key="cn_s_u")
    dnm = c2.slider("down move", 0.05, 0.9, 0.5, 0.05, key="cn_s_d")
    p = c3.slider("P(up)", 0.3, 0.7, 0.5, 0.01, key="cn_s_p")
    w = c4.slider("weight in the asset", 0.05, 1.0, 0.5, 0.05, key="cn_s_w")
    c1, c2, c3, c4 = st.columns(4)
    how = c1.radio("rebalance", ["every N flips", "drift band"], key="cn_s_how")
    every = int(c2.number_input("N flips", 1, 50, 1, key="cn_s_ev", disabled=how != "every N flips"))
    band = c3.slider("band", 0.01, 0.4, 0.1, 0.01, key="cn_s_b", disabled=how != "drift band")
    cost = c4.slider("cost per $ traded", 0.0, 0.05, 0.0, 0.001, key="cn_s_c", format="%.3f")
    c1, c2, c3 = st.columns(3)
    npaths = c1.slider("paths", 50, 1000, 300, 50, key="cn_s_np")
    nsteps = c2.slider("flips", 20, 500, 150, 10, key="cn_s_ns")
    seed = int(c3.number_input("seed", 1, 10_000, 7, key="cn_s_seed"))
    sim = K.shannon_paths(npaths, nsteps, upm, dnm, p, w, every, band if how == "drift band" else 0.0, cost, seed)
    g_a = float(K.constant_mix_growth(1.0, upm, dnm, p))
    g_m = float(K.constant_mix_growth(w, upm, dnm, p))
    kf = K.kelly_fraction(upm, dnm, p)
    fa, fm = sim["asset"][:, -1], sim["mix"][:, -1]
    st.markdown(uk.kpis([
        {"label": "ASSET ALONE · LOG GROWTH / FLIP", "value": f"{g_a * 100:+.2f}%", "color": T.teal if g_a > 0 else T.coral},
        {"label": "REBALANCED MIX", "value": f"{g_m * 100:+.2f}%", "color": T.teal if g_m > 0 else T.coral,
         "sub": "constant-mix, every flip, no cost"},
        {"label": "MEDIAN END WEALTH", "value": f"{np.median(fm):,.2f}x", "sub": f"asset {np.median(fa):,.2f}x"},
        {"label": "MIX BEATS ASSET", "value": f"{(fm > fa).mean() * 100:.0f}%", "sub": "of paths"},
        {"label": "KELLY FRACTION", "value": f"{kf * 100:.0f}%", "sub": "growth-optimal weight"},
        {"label": "TURNOVER / FLIP", "value": f"{sim['turnover'].mean() * 100:.1f}%"},
    ]), unsafe_allow_html=True)
    a, b = st.columns([1.4, 1])
    with a:
        x = np.arange(1, nsteps + 1)
        fig = go.Figure()
        for arr, col, nm in ((sim["asset"], T.amber, "asset alone"), (sim["mix"], T.teal, "rebalanced mix")):
            q = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
            fig.add_trace(go.Scatter(x=x, y=q[2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=q[0], fill="tonexty", fillcolor=up.fill(col, 0.15), line=dict(width=0),
                                     name=f"{nm} 10–90%"))
            fig.add_trace(go.Scatter(x=x, y=q[1], line=dict(color=col, width=2), name=f"{nm} median"))
        fig.add_hline(y=1, line=dict(color=T.line))
        fig.update_yaxes(type="log", title_text="wealth (start = 1)")
        fig.update_xaxes(title_text="flip")
        up.show(fig, 380, "simulated wealth paths", key="cn_s_fan", scale_toggle=True)
    with b:
        ws = np.linspace(0, 1, 101)
        gs = K.constant_mix_growth(ws, upm, dnm, p)
        fig = go.Figure(go.Scatter(x=ws, y=gs * 100, line=dict(color=T.violet, width=2), name="g(w)"))
        fig.add_vline(x=w, line=dict(color=T.text, dash="dash", width=1), annotation_text="you",
                      annotation_font=dict(size=9, color=T.muted))
        fig.add_vline(x=kf, line=dict(color=T.teal, dash="dot", width=1), annotation_text="Kelly",
                      annotation_font=dict(size=9, color=T.teal))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.update_xaxes(tickformat=".0%", title_text="weight in the asset")
        fig.update_yaxes(title_text="log growth per flip, %")
        up.show(fig, 380, "growth vs weight — the demon's hump", key="cn_s_g", scale_toggle=False)
    st.markdown("**In real portfolios** the same effect is the *diversification return*:")
    _formula(r"g_{\text{rebalanced}}-\sum_i w_i g_i\;\approx\;\tfrac12\Big(\sum_i w_i\sigma_i^2-\sigma_p^2\Big)",
             "It is largest for volatile, weakly correlated assets and is eaten by costs. Rebalancing does NOT "
             "guarantee beating the best single asset (Hallerbach 2014) — it is a risk-control habit with a growth "
             "bonus, not a free lunch. Shannon himself reportedly said the commissions would kill you.")
    with st.expander("Diversification return of a real mix"):
        tk = _ticker_input("tickers", ["SPY", "TLT", "GLD", "DBC"], "cn_s_tk")
        R = _returns(tk)
        if R.shape[1] >= 2:
            R = R.dropna()
            vols = R.std().to_numpy() * np.sqrt(252)
            prem = K.rebalancing_premium(np.full(R.shape[1], 1 / R.shape[1]), vols, R.corr().to_numpy())
            st.caption(f"Equal-weight {', '.join(R.columns)} since {R.index[0]:%Y}: theoretical diversification return "
                       f"≈ **{prem * 100:.2f}% a year** (annual vols {', '.join(f'{v:.0%}' for v in vols)}).")


def ch_add() -> None:
    st.markdown(section("04 · SHOULD I ADD IT? — the correlation hurdle", 3), unsafe_allow_html=True)
    st.markdown("You hold portfolio **XYZ** and are thinking of adding a little of **ABC**. The napkin test "
                "(popularised by Multi-Strat Mark and others; it falls straight out of mean-variance maths):")
    _formula(r"\text{add ABC if}\quad SR_{ABC}\;>\;\rho_{ABC,XYZ}\times SR_{XYZ}"
             r"\quad\Longleftrightarrow\quad \mu_{ABC}\;>\;\beta_{ABC|XYZ}\times\mu_{XYZ}",
             "Left: Sharpe form. Right: the same test in return terms — ABC's return must beat what its beta to "
             "your portfolio 'explains', i.e. ABC must have positive alpha against what you already own. "
             "(β = ρ·σ_ABC/σ_XYZ; divide both sides by σ_ABC to get the Sharpe form.)")
    st.caption("If you've seen it written with betas — 'β_ABC > ρ × β_XYZ' — that's a shorthand of the same idea; "
               "the precise version compares Sharpe ratios (or returns vs beta-implied returns) as above. "
               "A zero-correlation stream with any positive Sharpe always helps; a 0.8-correlated stream must "
               "reach 80% of your portfolio's Sharpe.")
    mode = st.radio("inputs", ["real tickers", "sliders"], horizontal=True, key="cn_a_mode")
    if mode == "sliders":
        c1, c2, c3 = st.columns(3)
        srp = c1.slider("portfolio Sharpe", 0.1, 2.0, 0.6, 0.05, key="cn_a_sp")
        srn = c2.slider("candidate Sharpe", -0.5, 2.0, 0.3, 0.05, key="cn_a_sn")
        rho = c3.slider("correlation", -1.0, 1.0, 0.3, 0.05, key="cn_a_r")
        d = {"sr_new": srn, "sr_port": srp, "rho": rho, **K.add_test(srn, srp, rho)}
        label_n, label_p = "candidate", "portfolio"
    else:
        c1, c2, c3 = st.columns(3)
        port = _ticker_input("your portfolio (equal weight)", ["SPY"], "cn_a_p")
        cand = c1.text_input("candidate", "TLT", key="cn_a_c").strip().upper() or "TLT"
        start = c2.number_input("from", 1995, 2024, 2007, key="cn_a_s")
        R = _returns(port + [cand], f"{start}-01-01")
        if cand not in R or not set(port) & set(R.columns):
            st.caption("couldn't load those tickers")
            return
        pr = R[[t for t in port if t in R]].mean(axis=1)
        d = K.add_test_returns(R[cand], pr)
        if not d.get("ok"):
            st.caption("not enough overlapping history")
            return
        label_n, label_p = cand, "+".join(port)
        c3.caption(f"β = {d['beta']:+.2f} · μ {label_n} {d['mu_new'] * 100:+.1f}% vs β·μ "
                   f"{d['beta'] * d['mu_port'] * 100:+.1f}% → alpha {d['alpha'] * 100:+.1f}%/yr")
    best = K.optimal_mix(d["sr_new"], d["sr_port"], d["rho"])
    st.markdown(uk.kpis([
        {"label": f"{label_n} SHARPE", "value": f"{d['sr_new']:.2f}"},
        {"label": f"{label_p} SHARPE", "value": f"{d['sr_port']:.2f}"},
        {"label": "CORRELATION", "value": f"{d['rho']:+.2f}"},
        {"label": "HURDLE ρ·SR", "value": f"{d['hurdle']:.2f}"},
        {"label": "VERDICT", "value": "ADD" if d["adds"] else "SKIP", "color": T.teal if d["adds"] else T.coral,
         "sub": f"margin {d['margin']:+.2f}"},
        {"label": "BEST RISK SHARE", "value": f"{best * 100:.0f}%", "sub": "of a two-stream mix (equal vol)"},
    ]), unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        rr = np.linspace(-1, 1, 101)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rr, y=rr * d["sr_port"], line=dict(color=T.amber, width=2), name="hurdle ρ·SR"))
        top = max(2.0, d["sr_new"] + 0.3)
        fig.add_trace(go.Scatter(x=rr, y=np.full_like(rr, top), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rr, y=rr * d["sr_port"], fill="tonexty", fillcolor=up.fill(T.teal, 0.12),
                                 line=dict(width=0), name="adding helps"))
        fig.add_trace(go.Scatter(x=[d["rho"]], y=[d["sr_new"]], mode="markers+text", text=[label_n],
                                 textposition="top center", marker=dict(size=13, color=T.teal if d["adds"] else T.coral),
                                 name="candidate"))
        fig.update_xaxes(title_text="correlation to your portfolio", range=[-1, 1])
        fig.update_yaxes(title_text="candidate Sharpe")
        fig.update_layout(hovermode="closest")
        up.show(fig, 360, "above the line = worth adding", key="cn_a_h", scale_toggle=False)
    with b:
        ws = np.linspace(0, 1, 41)
        s1, s2, r_ = d["sr_new"], d["sr_port"], d["rho"]
        port_sr = (ws * s1 + (1 - ws) * s2) / np.sqrt(ws ** 2 + (1 - ws) ** 2 + 2 * ws * (1 - ws) * r_)
        fig = go.Figure(go.Scatter(x=ws * 100, y=port_sr, line=dict(color=T.teal, width=2)))
        fig.add_vline(x=best * 100, line=dict(color=T.faint, dash="dash"))
        fig.add_hline(y=s2, line=dict(color=T.amber, dash="dot", width=1), annotation_text="today",
                      annotation_font=dict(size=9, color=T.amber))
        fig.update_xaxes(title_text="% of risk in the candidate")
        fig.update_yaxes(title_text="mix Sharpe")
        up.show(fig, 360, "Sharpe of the mix (equal-vol streams)", key="cn_a_m", scale_toggle=False)
    st.caption("Caveat: Sharpe and correlation are estimated with error, and correlations jump in crises "
               "(chapter 08). Prefer candidates that pass with margin and have an economic reason to diversify.")


def ch_weighting() -> None:
    st.markdown(section("05 · WEIGHTING — optimisers vs estimation error", 4), unsafe_allow_html=True)
    st.markdown("Mean-variance optimisation is only as good as its inputs, and expected returns are the noisiest "
                "input there is. Here each scheme is **fitted on the first half** of history and **judged on the "
                "second half** — the honest test.")
    tk = _ticker_input("tickers", DEFAULT, "cn_w_tk")
    R = _returns(tk).dropna()
    if R.shape[1] < 2 or len(R) < 500:
        st.caption("need at least two tickers with overlapping history")
        return
    cut = len(R) // 2
    ins, oos = R.iloc[:cut], R.iloc[cut:]
    st.caption(f"fit {ins.index[0]:%Y-%m} → {ins.index[-1]:%Y-%m} · test {oos.index[0]:%Y-%m} → {oos.index[-1]:%Y-%m}")
    rows, W = [], {}
    for s, desc in K.SCHEMES.items():
        w = _weights_cached(s, ins)
        W[s] = w
        a = K.stats(K.run_mix(ins, w, "quarterly")["returns"])
        b = K.stats(K.run_mix(oos, w, "quarterly")["returns"])
        rows.append({"scheme": s, "in-sample Sharpe": a.get("Sharpe"), "out-of-sample Sharpe": b.get("Sharpe"),
                     "decay": (b.get("Sharpe", np.nan) - a.get("Sharpe", np.nan)),
                     "OOS vol": b.get("Vol"), "OOS max DD": b.get("MaxDD"),
                     "max weight": float(np.max(w)), "what it is": desc})
    df = pd.DataFrame(rows).set_index("scheme")
    st.dataframe(df.style.format({"in-sample Sharpe": "{:.2f}", "out-of-sample Sharpe": "{:.2f}", "decay": "{:+.2f}",
                                  "OOS vol": "{:.1%}", "OOS max DD": "{:.1%}", "max weight": "{:.0%}"}),
                 use_container_width=True)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df["in-sample Sharpe"], name="fitted on", marker_color=up.fill(T.amber, 0.6)))
        fig.add_trace(go.Bar(x=df.index, y=df["out-of-sample Sharpe"], name="judged on", marker_color=T.teal))
        fig.update_layout(barmode="group", hovermode="x unified")
        up.show(fig, 340, "Sharpe in-sample vs out-of-sample", key="cn_w_bar")
    with b:
        pick = st.selectbox("show weights for", list(W), index=2, key="cn_w_pick")
        cov = np.cov(ins.to_numpy().T)
        rc = K.risk_contributions(W[pick], cov)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(R.columns), y=W[pick] * 100, name="capital weight", marker_color=up.fill(T.amber, 0.6)))
        fig.add_trace(go.Bar(x=list(R.columns), y=rc * 100, name="share of risk", marker_color=T.violet))
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(title_text="%")
        up.show(fig, 300, f"{pick}: where the money vs the risk is", key="cn_w_rc")
    st.caption("Typical pattern (DeMiguel et al. 2009; Michaud 1989): the sample max-Sharpe portfolio looks best "
               "in-sample and decays most; shrinkage, resampling, risk parity and plain 1/N hold up better. "
               "Notice how capital weight and risk share differ — a 60/40 stock/bond mix is ~90% equity risk.")


def ch_rebalance() -> None:
    st.markdown(section("06 · REBALANCING — calendar, bands, or never?", 5), unsafe_allow_html=True)
    st.markdown("Rebalancing is first a **risk-control** habit: without it the winners take over and the "
                "portfolio drifts to whatever went up. Its growth bonus is the chapter-03 effect, minus costs.")
    tk = _ticker_input("tickers", ["SPY", "TLT", "GLD", "DBC"], "cn_r_tk")
    c1, c2, c3 = st.columns(3)
    scheme = c1.selectbox("target weights", list(K.SCHEMES), index=0, key="cn_r_sch")
    band = c2.slider("band width", 0.01, 0.20, 0.05, 0.01, key="cn_r_band")
    cost = c3.slider("cost, bps per $ traded", 0.0, 50.0, 5.0, 1.0, key="cn_r_cost")
    R = _returns(tk).dropna()
    if R.shape[1] < 2:
        return
    w = _weights_cached(scheme, R)
    rows, eqs = [], {}
    for rule in ("none", "monthly", "quarterly", "annual", "band"):
        out = K.run_mix(R, w, rule, band, cost)
        s = K.stats(out["returns"])
        eqs[rule] = (1 + out["returns"]).cumprod()
        rows.append({"rule": rule if rule != "band" else f"band ±{band:.0%}", **s,
                     "turnover / yr": out["turnover_per_year"], "rebalances": out["rebalances"]})
    df = pd.DataFrame(rows).set_index("rule")
    st.dataframe(df[["CAGR", "Vol", "Sharpe", "MaxDD", "turnover / yr", "rebalances"]].style.format(
        {"CAGR": "{:.2%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}", "turnover / yr": "{:.0%}"}),
        use_container_width=True)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for (k, e), col in zip(eqs.items(), [T.amber, T.faint, T.teal, T.violet, T.cyan]):
            fig.add_trace(go.Scatter(x=e.index, y=e, name=k, line=dict(color=col, width=1.6 if k != "none" else 2.2)))
        fig.update_yaxes(type="log", title_text="growth of 1")
        up.show(fig, 340, "rebalanced vs drifting", key="cn_r_eq")
    with b:
        X = R.to_numpy()
        cur = np.zeros_like(X)
        wt = w.copy()
        for i in range(len(X)):
            rp = float(wt @ X[i])
            wt = wt * (1 + X[i]) / (1 + rp)
            cur[i] = wt
        fig = go.Figure()
        for j, c_ in enumerate(R.columns):
            fig.add_trace(go.Scatter(x=R.index, y=cur[:, j], stackgroup="w", name=c_,
                                     line=dict(width=0.5, color=T.cat[j % len(T.cat)])))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        up.show(fig, 340, "weights if you never rebalance (drift)", key="cn_r_drift")
    vols = R.std().to_numpy() * np.sqrt(252)
    prem = K.rebalancing_premium(w, vols, R.corr().to_numpy())
    st.caption(f"Theoretical diversification return of this mix ≈ {prem * 100:.2f}%/yr before costs. Vanguard's "
               "research (Jaconetti et al. 2010) finds annual checks with ~5% bands a sensible balance of risk "
               "control and cost; monthly rebalancing mostly adds turnover.")


def ch_leverage() -> None:
    st.markdown(section("07 · LEVERAGE & VOL TARGETING — how much is too much?", 6), unsafe_allow_html=True)
    st.markdown("Expected compound growth rises with leverage, peaks at the **Kelly** level, then falls — and "
                "is zero at twice Kelly:")
    _formula(r"g(L)=r_f+L(\mu-r_f)-\tfrac12L^2\sigma^2\qquad L^*_{\text{Kelly}}=\frac{\mu-r_f}{\sigma^2}"
             r"\qquad \sigma^*_{\text{Kelly}}=SR",
             "Carver: estimated Sharpe ratios are optimistic, so target at most HALF the Kelly vol — e.g. a "
             "backtested Sharpe of 0.5 → no more than ~25% annual vol, and less for a single strategy.")
    src = st.radio("asset", ["sliders", "real ticker / portfolio"], horizontal=True, key="cn_l_src")
    if src == "sliders":
        c1, c2, c3 = st.columns(3)
        mu = c1.slider("arithmetic return μ", 0.0, 0.25, 0.08, 0.005, key="cn_l_mu")
        sig = c2.slider("vol σ", 0.03, 0.60, 0.16, 0.01, key="cn_l_sig")
        rf = c3.slider("cash rate", 0.0, 0.06, 0.02, 0.0025, key="cn_l_rf")
    else:
        tk = _ticker_input("ticker(s), equal weight", ["SPY", "TLT"], "cn_l_tk")
        R = _returns(tk).dropna()
        r = R.mean(axis=1)
        mu, sig = float(r.mean() * 252), float(r.std() * np.sqrt(252))
        rf = st.slider("cash rate", 0.0, 0.06, 0.02, 0.0025, key="cn_l_rf2")
        st.caption(f"estimated μ {mu:.1%}, σ {sig:.1%} since {R.index[0]:%Y} — with error bars wider than you'd like")
    df = K.growth_vs_leverage(mu, sig, rf)
    Lk = max((mu - rf) / sig ** 2, 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["L"], y=df["growth"] * 100, line=dict(color=T.teal, width=2.2), name="g(L)"))
    for x_, nm, col in ((Lk, "Kelly", T.coral), (Lk / 2, "half-Kelly", T.amber), (1.0, "1x", T.faint)):
        fig.add_vline(x=x_, line=dict(color=col, dash="dot", width=1), annotation_text=nm,
                      annotation_font=dict(size=9, color=col))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_xaxes(title_text="leverage (x)")
    fig.update_yaxes(title_text="expected compound growth %/yr")
    up.show(fig, 340, "growth vs leverage", key="cn_l_fig", scale_toggle=False)
    rows = []
    for nm, L in (("1x", 1.0), ("half-Kelly", Lk / 2), ("Kelly", Lk), ("2x Kelly", 2 * Lk)):
        g = rf + L * (mu - rf) - 0.5 * L ** 2 * sig ** 2
        rows.append({"leverage": nm, "x": L, "vol": L * sig, "growth": g,
                     "P(ever −50% from start)": K.prob_ever_drawdown(g, L * sig, 0.5),
                     "P(ever −80% from start)": K.prob_ever_drawdown(g, L * sig, 0.8)})
    st.dataframe(pd.DataFrame(rows).set_index("leverage").style.format(
        {"x": "{:.2f}", "vol": "{:.1%}", "growth": "{:.2%}", "P(ever −50% from start)": "{:.0%}",
         "P(ever −80% from start)": "{:.0%}"}), use_container_width=True)
    st.caption("Drawdown odds are for a frictionless log-normal world and measured from the START (drawdowns from "
               "peaks are worse). Vol-managed exposure (Moreira & Muir 2017; Harvey et al. 2018) mainly trims tail "
               "risk; out-of-sample Sharpe gains are debated (Cederburg et al. 2020).")


def ch_crises() -> None:
    st.markdown(section("08 · CRISES — what still diversifies when it matters?", 7), unsafe_allow_html=True)
    st.markdown("Correlations measured over calm years mislead: in sell-offs many 'diversifiers' fall with "
                "stocks (Longin & Solnik 2001). This compares each asset's correlation to the anchor in **all "
                "months** vs the anchor's **worst 10% of months**, plus what it actually returned then. A simple "
                "**trend-following sleeve** (12-month time-series momentum on the same assets, vol-scaled) is "
                "included — trend has historically earned 'crisis alpha' (Hurst, Ooi & Pedersen 2017).")
    c1, c2 = st.columns([1, 2])
    anchor = c1.text_input("anchor", "SPY", key="cn_c_anchor").strip().upper() or "SPY"
    tk = _ticker_input("assets", ["TLT", "IEF", "GLD", "DBC", "EFA", "VNQ", "UUP"], "cn_c_tk")
    px = _prices(tuple(sorted(set(tk + [anchor]))), "2003-01-01")
    R = px.pct_change()
    trend_univ = [t for t in ("SPY", "TLT", "GLD", "DBC", "EFA", "IEF", "UUP") if t in px]
    if len(trend_univ) >= 3:
        R["TREND (built here)"] = K.trend_sleeve(px[trend_univ])
    t = K.crisis_table(R, anchor)
    if t.empty:
        return
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t.index, y=t["corr_all"], name="all months", marker_color=up.fill(T.cyan, 0.6)))
        fig.add_trace(go.Bar(x=t.index, y=t["corr_crisis"], name=f"{anchor}'s worst 10% of months",
                             marker_color=T.coral))
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_yaxes(range=[-1, 1], title_text=f"correlation to {anchor}")
        up.show(fig, 340, "correlation: calm vs crisis", key="cn_c_corr")
    with b:
        fig = go.Figure(go.Bar(x=t.index, y=t["avg_in_crisis_%"],
                               marker_color=[T.teal if v > 0 else T.coral for v in t["avg_in_crisis_%"]],
                               text=[f"{v:+.1f}%" for v in t["avg_in_crisis_%"]], textposition="outside"))
        fig.update_yaxes(title_text="avg monthly return in those months, %")
        fig.update_layout(hovermode="closest")
        up.show(fig, 340, f"what each paid when {anchor} had its worst months", key="cn_c_ret")
    st.dataframe(t.style.format({"corr_all": "{:+.2f}", "corr_crisis": "{:+.2f}", "avg_in_crisis_%": "{:+.2f}",
                                 "hit_in_crisis_%": "{:.0f}"}), use_container_width=True)
    if "TLT" in R:
        rc = R[anchor].rolling(252).corr(R["TLT"])
        fig = go.Figure(go.Scatter(x=rc.index, y=rc, line=dict(color=T.violet, width=1.4), fill="tozeroy",
                                   fillcolor=up.fill(T.violet, 0.12)))
        fig.add_hline(y=0, line=dict(color=T.line))
        fig.update_yaxes(range=[-1, 1], title_text="1-year correlation")
        up.show(fig, 260, f"{anchor} vs TLT — the stock/bond correlation is a regime, not a constant", key="cn_c_roll")
    st.caption("Long-volatility (put buying) is the other classic crisis diversifier but costs carry in calm "
               "years — test it in BACKTEST › OPTIONS LAB (long-vol templates, PPUT calibration).")


# ============================================================== sandbox ======
def _library_returns() -> dict:
    try:
        from ..backtest import library
        from ..backtest.views.common import saved
        from ..backtest.views.portfolio_view import _component
        out = {}
        for nm, rec in saved().items():
            try:
                r, _ = _component(library.spec_of(rec).to_json())
                out[f"LAB:{nm}"] = r
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception:  # noqa: BLE001
        return {}


def sandbox() -> None:
    st.markdown(section("SANDBOX — build a portfolio and apply everything", 0), unsafe_allow_html=True)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        tk = _ticker_input("ETFs / stocks", ["SPY", "EFA", "TLT", "GLD", "DBC"], "cn_x_tk")
        lib = _library_returns()
        picks = st.multiselect("saved LAB strategies", list(lib), key="cn_x_lib",
                               help="Save strategies in BACKTEST › LAB to mix them in here.") if lib else []
        start = st.number_input("from", 1995, 2024, 2007, key="cn_x_start")
        scheme = st.selectbox("weighting", list(K.SCHEMES), index=2, key="cn_x_sch",
                              format_func=lambda s: s.replace("_", " "))
        st.caption(K.SCHEMES[scheme])
        rule = st.selectbox("rebalance", list(K.REBALANCE), index=2, key="cn_x_rule")
        band = st.slider("band", 0.01, 0.2, 0.05, 0.01, key="cn_x_band", disabled=rule != "band")
        cost = st.slider("cost bps", 0.0, 50.0, 5.0, 1.0, key="cn_x_cost")
        vt = st.slider("portfolio vol target (0 = off)", 0.0, 0.3, 0.0, 0.01, key="cn_x_vt")
        cands = _ticker_input("candidates to test adding", ["IEF", "VNQ", "UUP", "DBMF", "QQQ"], "cn_x_cand")
    with right:
        R = _returns(tk, f"{start}-01-01")
        for p in picks:
            R[p] = lib[p]
        R = R.dropna()
        if R.shape[1] < 2 or len(R) < 250:
            st.caption("need at least two components with a year of overlapping history")
            return
        w = _weights_cached(scheme, R)
        out = K.run_mix(R, w, rule, band, cost, vt)
        r = out["returns"]
        s = K.stats(r)
        bh = K.stats(K.run_mix(R, w, "none", band, cost)["returns"])
        corr = R.corr().to_numpy()
        idm = K.diversification_multiplier(w, corr)
        vols = R.std().to_numpy() * np.sqrt(252)
        prem = K.rebalancing_premium(w, vols, corr)
        st.markdown(uk.kpis([
            {"label": "CAGR", "value": _pct(s["CAGR"])},
            {"label": "SHARPE", "value": f"{s['Sharpe']:.2f}"},
            {"label": "VOL", "value": _pct(s["Vol"])},
            {"label": "MAX DD", "value": _pct(s["MaxDD"]), "color": T.coral},
            {"label": "IDM", "value": f"{idm:.2f}x", "sub": "diversification multiplier"},
            {"label": "REBALANCE BONUS", "value": _pct(s["CAGR"] - bh["CAGR"], 2), "sub": f"theory {prem * 100:.2f}%/yr"},
            {"label": "TURNOVER", "value": f"{out['turnover_per_year'] * 100:.0f}%/yr"},
        ]), unsafe_allow_html=True)
        tabs = st.tabs(["EQUITY", "WEIGHTS & RISK", "FRONTIER", "CORRELATION", "ADD A CANDIDATE?"])
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            eq = (1 + r).cumprod()
            for j, c_ in enumerate(R.columns):
                e = (1 + R[c_]).cumprod()
                fig.add_trace(go.Scatter(x=e.index, y=e, name=c_, opacity=0.6,
                                         line=dict(color=T.cat[(j + 1) % len(T.cat)], width=0.9)), 1, 1)
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name="PORTFOLIO", line=dict(color=T.teal, width=2.4)), 1, 1)
            dd = eq / eq.cummax() - 1
            fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", fillcolor=up.fill(T.coral, 0.25),
                                     line=dict(color=T.coral, width=1), showlegend=False), 2, 1)
            fig.update_yaxes(type="log", title_text="growth of 1", row=1, col=1)
            fig.update_yaxes(tickformat=".0%", row=2, col=1)
            up.show(fig, 460, "portfolio vs components", key="cn_x_eq")
        with tabs[1]:
            cov = np.cov(R.to_numpy().T)
            rc = K.risk_contributions(w, cov)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(R.columns), y=w * 100, name="capital", marker_color=up.fill(T.amber, 0.6)))
            fig.add_trace(go.Bar(x=list(R.columns), y=rc * 100, name="risk", marker_color=T.violet))
            fig.update_layout(barmode="group", hovermode="x unified")
            fig.update_yaxes(title_text="%")
            up.show(fig, 340, "capital weight vs share of risk", key="cn_x_rc")
            st.caption("Weights are fitted on the whole window shown — see chapter 05 for the out-of-sample view.")
        with tabs[2]:
            mu_a = R.mean().to_numpy() * 252
            cov_a = np.cov(R.to_numpy().T) * 252
            cloud = K.random_portfolios(mu_a, cov_a)
            fig = go.Figure(go.Scatter(x=cloud["vol"], y=cloud["ret"], mode="markers",
                                       marker=dict(size=3, color=cloud["sharpe"], colorscale=up.SEQUENTIAL,
                                                   showscale=True, colorbar=dict(thickness=10, title="Sharpe")),
                                       name="random long-only mixes"))
            for j, c_ in enumerate(R.columns):
                fig.add_trace(go.Scatter(x=[np.sqrt(cov_a[j, j])], y=[mu_a[j]], mode="markers+text", text=[c_],
                                         textposition="top center", marker=dict(size=8, color=T.faint),
                                         showlegend=False))
            for sch, col in (("equal", T.amber), ("risk_parity", T.violet), ("min_variance", T.cyan),
                             ("max_sharpe", T.coral)):
                ww = _weights_cached(sch, R)
                fig.add_trace(go.Scatter(x=[np.sqrt(ww @ cov_a @ ww)], y=[ww @ mu_a], mode="markers",
                                         marker=dict(size=11, symbol="diamond", color=col), name=sch))
            fig.add_trace(go.Scatter(x=[np.sqrt(w @ cov_a @ w)], y=[w @ mu_a], mode="markers",
                                     marker=dict(size=16, symbol="star", color=T.teal, line=dict(color=T.text, width=1)),
                                     name="YOU"))
            fig.update_xaxes(tickformat=".0%", title_text="vol")
            fig.update_yaxes(tickformat=".0%", title_text="arithmetic return")
            fig.update_layout(hovermode="closest")
            up.show(fig, 460, "the frontier, in hindsight", key="cn_x_fr", scale_toggle=False)
            st.caption("Hindsight frontier: the top-left edge is only knowable after the fact.")
        with tabs[3]:
            c = R.corr()
            fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=list(c.columns), y=list(c.index), zmin=-1, zmax=1,
                                       colorscale=up.DIVERGING, text=np.round(c.to_numpy(), 2),
                                       texttemplate="%{text}", xgap=1, ygap=1, colorbar=dict(thickness=10)))
            fig.update_yaxes(autorange="reversed", showspikes=False)
            fig.update_xaxes(showspikes=False)
            fig.update_layout(hovermode="closest")
            up.show(fig, 140 + 36 * len(c), "correlation", key="cn_x_hm")
        with tabs[4]:
            CR = _returns(cands, f"{start}-01-01")
            rows = []
            for cnd in CR.columns:
                d = K.add_test_returns(CR[cnd], r)
                if d.get("ok"):
                    rows.append({"candidate": cnd, "Sharpe": d["sr_new"], "corr": d["rho"], "hurdle": d["hurdle"],
                                 "margin": d["margin"], "beta": d["beta"], "alpha %/yr": d["alpha"] * 100,
                                 "verdict": "ADD" if d["adds"] else "skip",
                                 "best risk share": K.optimal_mix(d["sr_new"], d["sr_port"], d["rho"])})
            if rows:
                df = pd.DataFrame(rows).set_index("candidate").sort_values("margin", ascending=False)
                st.dataframe(df.style.format({"Sharpe": "{:.2f}", "corr": "{:+.2f}", "hurdle": "{:.2f}",
                                              "margin": "{:+.2f}", "beta": "{:+.2f}", "alpha %/yr": "{:+.2f}",
                                              "best risk share": "{:.0%}"}), use_container_width=True)
                st.caption(f"Each candidate vs YOUR portfolio (Sharpe {s['Sharpe']:.2f}) over its overlapping history: "
                           "add it if its Sharpe beats correlation × your Sharpe (chapter 04).")


def sources() -> None:
    st.markdown(section("SOURCES — the research behind each chapter", 0), unsafe_allow_html=True)
    try:
        d = json.loads((DATA_DIR / "construct" / "sources.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"sources unavailable: {exc}")
        return
    st.caption(d.get("note", ""))
    st.dataframe(pd.DataFrame(d["sources"])[["chapter", "cite", "use", "strength"]], hide_index=True,
                 use_container_width=True, height=760)


# ================================================================ page ======
def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    mode = st.segmented_control("mode", ["GUIDE", "SANDBOX", "SOURCES"], default="GUIDE", key="cn_mode",
                                label_visibility="collapsed") or "GUIDE"
    if mode == "SANDBOX":
        sandbox()
        return
    if mode == "SOURCES":
        sources()
        return
    ch = st.segmented_control("chapter", CHAPTERS, default=CHAPTERS[0], key="cn_ch",
                              label_visibility="collapsed") or CHAPTERS[0]
    {CHAPTERS[0]: ch_diversify, CHAPTERS[1]: ch_drag, CHAPTERS[2]: ch_shannon, CHAPTERS[3]: ch_add,
     CHAPTERS[4]: ch_weighting, CHAPTERS[5]: ch_rebalance, CHAPTERS[6]: ch_leverage, CHAPTERS[7]: ch_crises}[ch]()
    i = CHAPTERS.index(ch)
    st.caption(f"chapter {i + 1} of {len(CHAPTERS)} · next: {CHAPTERS[i + 1] if i + 1 < len(CHAPTERS) else 'try the SANDBOX'}")

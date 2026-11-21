"""LAB › EXPLORE — look for an edge before building a strategy around it.

The MemLabs method, made no-code: bucket a reading (an indicator, the last
bar's direction, the weekday, the regime), then compare the *next* bar's
return across buckets — separately in a DISCOVERY half and a CONFIRM half of
the in-sample data. A pattern is only interesting if it shows up in both.
Bucket edges are learnt on the discovery half only.

Every distinct view you look at here is logged as a trial (it feeds the
deflated Sharpe): browsing twenty calendar buckets until one looks good is
exactly how luck gets mistaken for an edge.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sps

from ... import ui_charts as uc
from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import glossary as G
from .. import ledger as lg
from ..indicators import CATALOG
from ..regime import DETECTORS
from ..spec import with_aux
from . import common as C
from . import regime_panel as RP

BY = {
    "signal": "the selected signal's reading",
    "lag1": "last bar's direction (up / down)",
    "dow": "day of week",
    "month": "month of year",
    "regime": "market regime",
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def reading(spec, df: pd.DataFrame, aux: dict) -> tuple[pd.Series, str]:
    strat = spec.strategy()
    frame = with_aux(df, aux, spec.needs())
    if hasattr(strat, "reading"):
        v = strat.reading(frame)
        if strat.indicator.overlay:
            # a price-level line (moving average, stop) is only comparable over time as a distance from price
            return df["close"] / v - 1, f"% distance of price from {strat.describe()}"
        return v, strat.describe()
    return strat.raw_forecast(frame), f"{strat.describe()} raw forecast"


def bucketize(x: pd.Series, n: int, fit_mask: pd.Series) -> tuple[pd.Series, list]:
    """Quantile buckets with edges learnt on ``fit_mask`` only; discrete readings keep their values."""
    v = x.dropna()
    if v.nunique() <= 12:
        return x.map(lambda a: a if pd.notna(a) else np.nan), sorted(v.unique().tolist())
    edges = np.unique(np.nanquantile(x[fit_mask].dropna(), np.linspace(0, 1, n + 1)))
    if len(edges) < 3:
        return pd.Series(np.nan, index=x.index), []
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(x, edges, labels=False)
    names = [f"Q{i + 1}" for i in range(len(edges) - 1)]
    return b.map(lambda i: names[int(i)] if pd.notna(i) else np.nan), names


def table(fwd: pd.Series, bucket: pd.Series, mask: pd.Series, keys: list) -> pd.DataFrame:
    rows = []
    for k in keys:
        x = fwd[mask & (bucket == k)].dropna().to_numpy()
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"bucket": k, "n": n, "mean": x.mean() if n else np.nan, "se": se,
                     "t": x.mean() / se if n > 1 and se > 0 else np.nan,
                     "win": (x > 0).mean() if n else np.nan})
    return pd.DataFrame(rows)


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay VR(q) and its heteroskedasticity-robust z-statistic."""
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4 * q:
        return np.nan, np.nan
    mu = r.mean()
    s1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (n - q + 1) * (1 - q / n)
    sq = ((rq - q * mu) ** 2).sum() / m
    vr = sq / s1
    d = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = (d[j:] * d[:-j]).sum() / (d.sum() ** 2) * n
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    return float(vr), float(z)


def render(spec, ins: pd.DataFrame, aux: dict) -> None:
    st.markdown(section("EXPLORE — is there anything here before you build on it?", 1,
                        help="Bucket a reading and compare the NEXT bar's return across buckets, in two separate "
                             "halves of the in-sample data. Real effects show up in both halves."),
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    by = c1.selectbox("bucket by", list(BY), format_func=lambda k: BY[k], key="bt_x_by",
                      help="What to group bars by. 'signal' uses the rule picked in 02 · SIGNAL.")
    nb = int(c2.slider("buckets", 2, 10, 5, key="bt_x_n", help="Quantile buckets for continuous readings "
                                                                 "(edges set on the discovery half only)."))
    h = int(c3.selectbox("next … bars", [1, 2, 5, 10, 21], key="bt_x_h",
                         help="Forward horizon: the return from this bar's close to the close h bars later."))
    disc = int(c4.slider("discovery %", 40, 80, 60, 5, key="bt_x_disc",
                         help="First part of the in-sample data where you look for the pattern; the rest is where "
                              "you check it."))
    close = ins["close"]
    lr = np.log(close).diff()
    fwd = np.log(close.shift(-h) / close)
    cut = ins.index[int(len(ins) * disc / 100)]
    m_d = pd.Series(ins.index < cut, index=ins.index)
    m_c = ~m_d

    label = BY[by]
    if by == "signal":
        x, label = reading(spec, ins, aux)
        bucket, keys = bucketize(x, nb, m_d)
    elif by == "lag1":
        x = np.sign(lr)
        bucket, keys = x.map({1.0: "after UP bar", -1.0: "after DOWN bar"}), ["after DOWN bar", "after UP bar"]
    elif by == "dow":
        x = pd.Series(ins.index.dayofweek, index=ins.index)
        keys = sorted(x.unique().tolist())
        bucket = x.map(lambda d: DOW[d])
        keys = [DOW[k] for k in keys]
    elif by == "month":
        x = pd.Series(ins.index.month, index=ins.index)
        bucket = x.map(lambda m: f"{m:02d}")
        keys = sorted(bucket.unique().tolist())
    else:
        with st.popover("REGIME SETTINGS"):
            cfg = RP.controls("bt_xr", src=spec.src)
        x = RP.labels_for(spec, ins, *cfg)
        d = DETECTORS[cfg[0]]
        bucket = x
        keys = [k for k in d.order if k in set(x.dropna())] + sorted(set(x.dropna()) - set(d.order))
        label = d.label

    # every distinct view is a trial
    tag = f"x:{by}:{nb}:{h}:{disc}:{spec.rule if by == 'signal' else ''}:{hashlib.sha1(str(spec.params).encode()).hexdigest()[:6] if by == 'signal' else ''}"
    n_trials = lg.record(C.ledger(), spec.family(), tag)

    if not keys:
        C.banner(T.amber, "NO BUCKETS", "the reading is empty or constant here (missing data, or too short).")
        return
    td, tc = table(fwd, bucket, m_d, keys), table(fwd, bucket, m_c, keys)
    both = td.merge(tc, on="bucket", suffixes=("_d", "_c"))
    agree = (np.sign(both["mean_d"]) == np.sign(both["mean_c"])) & both["mean_d"].notna() & both["mean_c"].notna()
    ok = both["mean_d"].notna() & both["mean_c"].notna()
    rho = sps.spearmanr(both.loc[ok, "mean_d"], both.loc[ok, "mean_c"])[0] if ok.sum() > 2 else np.nan
    sig = ((both["t_d"].abs() > 2) & (both["t_c"].abs() > 2) & agree).sum()
    st.markdown(uk.kpis([
        {"label": "BUCKETS", "value": f"{len(keys)}", "sub": f"{label[:38]}"},
        {"label": "SAME SIGN IN BOTH HALVES", "value": f"{int(agree.sum())}/{len(both)}",
         "color": T.teal if agree.mean() > 0.7 else T.amber, "sub": "coin-flip ≈ half"},
        {"label": "RANK AGREEMENT", "value": "—" if not np.isfinite(rho) else f"{rho:+.2f}",
         "sub": "Spearman ρ of bucket means", "color": T.teal if np.isfinite(rho) and rho > 0.5 else T.amber},
        {"label": "|t| > 2 IN BOTH", "value": f"{int(sig)}", "sub": "buckets",
         "color": T.teal if sig else T.muted},
        {"label": "TRIALS", "value": f"{n_trials}", "sub": "this view counted", "color": T.violet},
    ]), unsafe_allow_html=True)

    fig = go.Figure()
    for t, nm, col in ((td, f"discovery (to {cut.date()})", T.violet), (tc, f"confirm (from {cut.date()})", T.teal)):
        fig.add_trace(go.Bar(x=[str(k) for k in t["bucket"]], y=t["mean"] * 1e4, name=nm, marker_color=col,
                             error_y=dict(type="data", array=1.96 * t["se"] * 1e4, color=T.muted, thickness=1),
                             customdata=np.c_[t["n"], t["win"] * 100],
                             hovertemplate="%{x}: %{y:.1f} bp · n=%{customdata[0]} · win %{customdata[1]:.0f}%"
                                           "<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.line))
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(title_text=f"mean next-{h}-bar return (bp)")
    fig.update_xaxes(type="category", showspikes=False)
    up.show(fig, 340, f"next {h}-bar return by {label[:48]} · whiskers = 95% interval", key="bt_x_bar")

    show = pd.DataFrame({
        "bucket": both["bucket"].astype(str),
        "n (disc)": both["n_d"], "mean bp (disc)": (both["mean_d"] * 1e4).round(1), "t (disc)": both["t_d"].round(2),
        "win (disc)": (both["win_d"] * 100).round(0),
        "n (conf)": both["n_c"], "mean bp (conf)": (both["mean_c"] * 1e4).round(1), "t (conf)": both["t_c"].round(2),
        "win (conf)": (both["win_c"] * 100).round(0), "same sign": np.where(agree, "✓", "✗"),
    })
    st.dataframe(show, use_container_width=True, hide_index=True, column_config=uc.colcfg(show.columns, {
        "t (disc)": "Mean ÷ its standard error. |t| > 2 is the usual 'probably not zero' line — but with many "
                    "buckets some will cross it by luck (Harvey, Liu & Zhu 2016 argue for |t| > 3).",
        "t (conf)": "Same, on the confirm half. The one that matters.",
        "win (disc)": "% of bars in this bucket followed by a positive return.",
        "same sign": "Did the bucket point the same way in both halves?",
    }))
    st.caption("How to read it: a real edge has buckets whose means keep their sign and ordering in BOTH halves. "
               "A pattern that only shows in discovery is the classic overfit. MemLabs' rule of thumb: want 100+ bars "
               "per bucket. " + (f"Calendar warning: {CATALOG['dow'].evidence.split('. ')[-1]}" if by in ("dow", "month")
                                 else ""))

    if by == "signal":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=close.index, y=close, name="close", line=dict(color=T.text, width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=x.index, y=x, name=label[:40], line=dict(color=T.violet, width=1)), 2, 1)
        if x.nunique() > 12:
            edges = np.nanquantile(x[m_d].dropna(), np.linspace(0, 1, nb + 1))[1:-1]
            for e in edges:
                fig.add_hline(y=float(e), line=dict(color=T.faint, width=1, dash="dot"), row=2, col=1)
        fig.add_vline(x=cut, line=dict(color=T.amber, width=1, dash="dash"))
        up.show(fig, 420, "price and reading · dotted = bucket edges · dashed = discovery / confirm split",
                key="bt_x_px")

    st.markdown(section("SERIAL DEPENDENCE — momentum or mean reversion?", 2,
                        help="Do returns tend to repeat (positive autocorrelation, variance ratio > 1 → trend "
                             "rules) or reverse (negative, VR < 1 → mean reversion)?"), unsafe_allow_html=True)
    rr = lr.dropna().to_numpy()
    n = len(rr)
    lags = list(range(1, 21))
    ac = [float(pd.Series(rr).autocorr(k)) for k in lags]
    qs = [2, 3, 5, 10, 15, 20]
    vr = [variance_ratio(rr, q) for q in qs]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=lags, y=ac, marker_color=[T.teal if a > 0 else T.coral for a in ac], name="ACF"))
        band = 1.96 / np.sqrt(max(n, 1))
        for y in (band, -band):
            fig.add_hline(y=y, line=dict(color=T.faint, width=1, dash="dot"))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text="lag (bars)", showspikes=False)
        up.show(fig, 280, "autocorrelation of returns · dotted = 95% noise band", key="bt_x_acf")
    with c2:
        fig = go.Figure(go.Scatter(x=qs, y=[v[0] for v in vr], mode="lines+markers", line=dict(color=T.violet),
                                   customdata=[v[1] for v in vr],
                                   hovertemplate="q=%{x}: VR %{y:.3f} · z %{customdata:.2f}<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=T.line))
        fig.update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(title_text="horizon q (bars)", showspikes=False)
        fig.update_yaxes(title_text="variance ratio")
        up.show(fig, 280, "variance ratio VR(q) · 1 = random walk", key="bt_x_vr")
    sig_vr = [q for q, v in zip(qs, vr) if np.isfinite(v[1]) and abs(v[1]) > 1.96]
    st.caption(f"Lag-1 autocorrelation {ac[0]:+.3f} "
               f"({'outside' if abs(ac[0]) > 1.96 / np.sqrt(max(n, 1)) else 'inside'} the noise band). "
               + (f"Variance ratio significant (robust |z| > 1.96) at q = {', '.join(map(str, sig_vr))}. "
                  if sig_vr else "No variance ratio is significant: consistent with a random walk. ")
               + "Lo & MacKinlay (1988).")
    st.caption(G.metric_help("DSR"))

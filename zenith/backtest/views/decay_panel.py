"""DECAY — is this strategy still working? (SharpeEdge: noise, bug or decay)

Used by LAB (step DECAY) and LIBRARY. Before the out-of-sample reveal it runs
in PRACTICE mode: pick a pretend go-live date inside the in-sample data, so the
locked holdout stays unseen. After the reveal it monitors the real
out-of-sample period against the in-sample record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_charts as uc
from ... import ui_kit as uk
from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import ledger as lg
from ..monitor.decay import cone, cusum, dd_breaker, month_ranks, rolling_alpha, verdict
from ..spec import StrategySpec, run_spec
from . import common as C

CHECKLIST = [
    ("Bug?", "Did anything change in the data or the code — a ticker switch, a split, a new cost setting, lot "
             "rounding? Rebuild the strategy from its saved spec and compare. SharpeEdge's case study found "
             "rounding and timing bugs before any real decay."),
    ("Noise?", "Is the live record inside the bootstrap cone and the drawdown above the breaker? Then a bad "
               "stretch is what this strategy normally does — stopping now locks in the loss."),
    ("Decay?", "CUSUM alarm, below the cone, several bottom-decile months, and alpha drifting to zero together "
               "are evidence the edge has gone. Published anomalies lose roughly a third to a half of their "
               "return after publication (McLean & Pontiff 2016)."),
]


def render(spec: StrategySpec, key: str = "dc") -> None:
    st.markdown(section("DECAY — is it still working?", 6,
                        help="Compare what happened after the research period with what the research period says "
                             "should happen. Five tests, one verdict."), unsafe_allow_html=True)
    try:
        full = C.load(spec)
    except Exception as exc:  # noqa: BLE001
        C.banner(T.coral, "NO DATA", str(exc))
        return
    ins, locked = spec.split(full)
    aux = C.load_aux(spec)
    rev = lg.revealed(C.ledger(), spec.family()) if locked is not None else None
    live_mode = bool(rev) and rev["hash"] == spec.hash()
    if live_mode:
        data, start = full, locked
        C.banner(T.violet, "LIVE MODE", f"monitoring the revealed out-of-sample period from {locked.date()}")
    else:
        data = ins
        idx = ins.index
        lo, hi = idx[int(len(idx) * 0.4)], idx[int(len(idx) * 0.9)]
        default = idx[int(len(idx) * 0.7)]
        start = pd.Timestamp(st.slider("pretend you went live on", lo.to_pydatetime(), hi.to_pydatetime(),
                                       default.to_pydatetime(), format="YYYY-MM-DD", key=f"{key}_golive",
                                       help="PRACTICE mode: everything stays inside the in-sample data. After "
                                            "the out-of-sample reveal in LAB this switches to the real thing."))
        C.banner(T.amber, "PRACTICE MODE", "the locked holdout stays unseen; going live is simulated inside the "
                 "in-sample period")
    res = run_spec(spec, data, aux)
    r = res.returns
    ref, live = r[r.index < start], r[r.index >= start]
    if len(ref) < 252 or len(live) < 21:
        C.banner(T.amber, "TOO SHORT", "need at least a year before and a month after the go-live date")
        return
    p = res.periods
    cs = cusum(ref, live)
    cn = cone(ref, live)
    brk = dd_breaker(ref, len(live))
    mr = month_ranks(ref, live)
    v = verdict(cs, cn, brk, live, mr)
    col = {"WORKING": T.teal, "WATCH": T.amber, "LIKELY BROKEN": T.coral}[v["state"]]
    sr_ref = ref.mean() / ref.std() * np.sqrt(p) if ref.std() > 0 else np.nan
    sr_live = live.mean() / live.std() * np.sqrt(p) if live.std() > 0 else np.nan
    st.markdown(uk.kpis([
        {"label": "VERDICT", "value": v["state"], "color": col, "sub": f"{sum(v['flags'].values())} of 4 flags"},
        {"label": "SHARPE · BEFORE / AFTER", "value": f"{sr_ref:.2f} / {sr_live:.2f}",
         "color": T.teal if sr_live > 0.5 * sr_ref else T.amber if sr_live > 0 else T.coral},
        {"label": "LIVE DRAWDOWN", "value": f"{v['live_dd'] * 100:.1f}%", "sub": f"breaker {brk * 100:.1f}%",
         "color": T.coral if v["live_dd"] < brk else T.teal},
        {"label": "CUSUM", "value": "ALARM" if cs["alarm"] is not None else "quiet",
         "sub": str(cs["alarm"].date()) if cs["alarm"] is not None else f"threshold {cs['h']:g}",
         "color": T.coral if cs["alarm"] is not None else T.teal},
    ]), unsafe_allow_html=True)
    st.markdown('<div style="margin:0.2rem 0 0.6rem 0;">' + "".join(
        uc.chip(k.upper(), T.coral if f else T.teal, "flagged" if f else "ok") for k, f in v["flags"].items())
        + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        x = list(cn.index)
        for a, b, op in (("p5", "p95", 0.12), ("p25", "p75", 0.22)):
            fig.add_trace(go.Scatter(x=x + x[::-1], y=list(cn[b]) + list(cn[a][::-1]), fill="toself",
                                     fillcolor=up.fill(T.teal, op), line=dict(width=0), name=f"{a}–{b}"))
        fig.add_trace(go.Scatter(x=x, y=cn["p50"], line=dict(color=T.teal, width=1), name="median"))
        fig.add_trace(go.Scatter(x=x, y=cn["actual"], line=dict(color=T.amber, width=2), name="actual"))
        fig.update_yaxes(title_text="growth since go-live")
        up.show(fig, 320, "live equity vs the bootstrap cone from the research period", key=f"{key}_cone")
    with c2:
        s = cs["s"]
        fig = go.Figure(go.Scatter(x=s.index, y=s, fill="tozeroy", fillcolor=up.fill(T.coral, 0.2),
                                   line=dict(color=T.coral, width=1.2), name="CUSUM"))
        fig.add_hline(y=cs["h"], line=dict(color=T.text, dash="dash"), annotation_text="alarm",
                      annotation_font=dict(size=9))
        fig.update_layout(hovermode="x unified", showlegend=False)
        up.show(fig, 320, "CUSUM · evidence the mean has dropped (Page 1954)", key=f"{key}_cusum")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=mr.index, y=mr["percentile"] * 100,
                               marker_color=[T.coral if q < 0.1 else T.amber if q < 0.3 else T.teal
                                             for q in mr["percentile"]],
                               customdata=mr["return"] * 100,
                               hovertemplate="%{x|%b %Y}: %{customdata:.1f}% · percentile %{y:.0f}<extra></extra>"))
        fig.add_hline(y=10, line=dict(color=T.coral, dash="dot"))
        fig.update_yaxes(range=[0, 100], title_text="percentile vs research months")
        fig.update_layout(hovermode="closest")
        up.show(fig, 280, "each live month ranked against the research months", key=f"{key}_mr")
    with c2:
        b = data["close"].pct_change().reindex(r.index)
        ra = rolling_alpha(r, b, window=min(252, len(r) // 3), periods=p)
        if not ra.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(ra.index) + list(ra.index[::-1]), y=list(ra["hi"] * 100) +
                                     list(ra["lo"][::-1] * 100), fill="toself", fillcolor=up.fill(T.violet, 0.18),
                                     line=dict(width=0), name="95% HAC band"))
            fig.add_trace(go.Scatter(x=ra.index, y=ra["alpha"] * 100, line=dict(color=T.violet, width=1.6),
                                     name="alpha"))
            fig.add_hline(y=0, line=dict(color=T.line))
            fig.add_vline(x=start, line=dict(color=T.amber, dash="dash"))
            fig.update_yaxes(title_text="annual alpha vs buy & hold %")
            up.show(fig, 280, "rolling alpha · dashed = go-live", key=f"{key}_alpha")
    with st.expander("NOISE, BUG OR DECAY? — what to check before you stop"):
        for t, d in CHECKLIST:
            st.markdown(f"**{t}** {d}")

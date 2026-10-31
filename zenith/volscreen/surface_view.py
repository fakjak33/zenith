"""VOL SURFACE page — the implied-volatility surface in 3D, its history, and
each tenor's IV against its own trailing-year distribution (sigma bands).

Modes (radio `vsf_mode`):
  SURFACE     delta x tenor surface; the scrubber morphs through stored sessions
  MONEYNESS   the same on a K/F grid (0.80 .. 1.20)
  TERRAIN     date x tenor: the ATM term structure through the past year
  CHANGE      today's surface minus N sessions ago (diverging)
  VRP LANDSCAPE  core basket x date: who carried a premium, and when
Below the 3D scene (and standing in for it when WebGL is unavailable): sigma
bands per tenor, smile and term slices against N sessions ago, the grid.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME as T
from ..config import VOLSCREEN_FILES, VOLSCREEN_SURF_ASSET, VOLSCREEN_TERM_ASSET
from ..ui_theme import evidence_rating, key_findings, section, stamp
from ..viz import surface3d, volviz
from . import DISCLAIMER, load

MODES = ["SURFACE", "MONEYNESS", "TERRAIN", "CHANGE", "VRP LANDSCAPE"]
_FINDINGS = [
    {"stat": "Implied variance is additive in time, so maturities are interpolated in total variance "
             "(σ²T) — the construction behind the VIX family.", "cite": "Cboe VIX methodology"},
    {"stat": "Equity smiles slope down (puts over calls): the market pays up for crash insurance, and the "
             "skew steepens when volatility rises.", "cite": "Bates (2000); Bakshi, Kapadia & Madan (2003)"},
    {"stat": "A surface reading is only meaningful against its own history: each tenor here is placed in "
             "standard deviations of its trailing-year distribution.", "cite": "Zenith σ-bands"},
]


@st.cache_data(ttl=900, show_spinner=False)
def _data() -> dict:
    return {"surfaces": load("surfaces", {}) or {}, "screen": load("screen", {}) or {}}


def _asset(key: str, asset: str):
    p = VOLSCREEN_FILES[key]
    if p.exists():
        return p
    from ..ephemeris.store_px import ensure_local
    return ensure_local(asset)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _surf_hist(ticker: str) -> pd.DataFrame:
    from .history import read_ticker
    return read_ticker(_asset("surf_local", VOLSCREEN_SURF_ASSET), ticker)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _term_hist(ticker: str) -> pd.DataFrame:
    from .history import read_ticker
    return read_ticker(_asset("term_local", VOLSCREEN_TERM_ASSET), ticker)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def _vrp_panel(tickers: tuple[str, ...]) -> pd.DataFrame:
    p = _asset("term_local", VOLSCREEN_TERM_ASSET)
    if p is None or not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p, columns=["ticker", "date", "vrp_30"], filters=[("ticker", "in", list(tickers))])
    except Exception:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df


def today_badge() -> str | None:
    return None


def _tenor_label(t: int) -> str:
    return f"{t}D" if t < 365 else "1Y"


def _frames_delta(sf: dict, hist: pd.DataFrame, tenors: list[int], as_of: str, n: int = 60) -> list:
    frames = []
    if len(hist):
        h = hist.sort_values("date")
        dates = sorted(h["date"].unique())[-n:]
        for d in dates:
            g = h[h["date"] == d].set_index("tenor")
            if pd.Timestamp(d).date().isoformat() == as_of:
                continue
            grid = [[g.loc[t, f"d{j}"] if t in g.index else None for j in range(9)] for t in tenors]
            frames.append((pd.Timestamp(d).date().isoformat(), grid))
    frames.append((as_of, sf["delta"]))
    return frames


def _cell_z(hist: pd.DataFrame, tenors: list[int], today: list[list]) -> list[list] | None:
    if hist is None or len(hist) < 20 * len(tenors):
        return None
    out = []
    for i, t in enumerate(tenors):
        g = hist[hist["tenor"] == t].sort_values("date").tail(252)
        row = []
        for j in range(9):
            s = g[f"d{j}"].dropna()
            v = today[i][j]
            sd = float(s.std(ddof=1)) if len(s) >= 20 else float("nan")
            row.append((v - float(s.mean())) / sd if v is not None and sd and math.isfinite(sd) and sd > 0 else None)
        out.append(row)
    return out


def _lag_grid(hist: pd.DataFrame, tenors: list[int], lag: int) -> tuple[str | None, list | None]:
    if not len(hist):
        return None, None
    dates = sorted(hist["date"].unique())
    if len(dates) <= lag:
        return None, None
    d = dates[-1 - lag]
    g = hist[hist["date"] == d].set_index("tenor")
    return pd.Timestamp(d).date().isoformat(), [[g.loc[t, f"d{j}"] if t in g.index else None for j in range(9)]
                                               for t in tenors]


def _scene(mode: str, tk: str, sf: dict, sd: dict, row: dict, lag: int):
    tenors = sd.get("tenors", [7, 14, 30, 60, 90, 180, 365])
    yt = [_tenor_label(t) for t in tenors]
    as_of = sd.get("as_of", "")
    hist = _surf_hist(tk)
    if mode == "SURFACE":
        frames = _frames_delta(sf, hist, tenors, as_of)
        extras = []
        z = _cell_z(hist, tenors, sf["delta"])
        if z:
            extras.append({"name": "1Y σ-Z", "fmt": "sigma", "grid": z})
        ld, lg = _lag_grid(hist, tenors, lag)
        if lg:
            extras.append({"name": f"Δ vs {ld}", "fmt": "pts",
                           "grid": [[(a - b) if a is not None and b is not None else None for a, b in zip(ra, rb)]
                                    for ra, rb in zip(sf["delta"], lg)]})
        return surface3d.payload(f"{tk} · implied volatility surface", sd.get("deltas", []), yt, frames,
                                 subtitle=f"delta × tenor · {len(frames)} session(s) on the scrubber",
                                 extras=extras)
    if mode == "MONEYNESS":
        xt = [f"{m:.2f}" if i % 2 == 0 else f"{m:.3f}" for i, m in enumerate(sd.get("mny", []))]
        return surface3d.payload(f"{tk} · surface by moneyness", xt, yt, [(as_of, sf["mny"])],
                                 x_label="K / F", subtitle="strike over forward · today", x_every=2)
    if mode == "TERRAIN":
        th = _term_hist(tk)
        if th.empty:
            return None
        th = th.sort_values("date").tail(252)
        step = max(1, len(th) // 70)
        th = th.iloc[::-1].iloc[::step].iloc[::-1]
        grid = [[r.get(f"iv_{t}") for r in th.to_dict("records")] for t in tenors]
        grid = [list(x) for x in zip(*grid)]                         # dates x tenors
        yd = [pd.Timestamp(d).strftime("%y-%m-%d") for d in th["date"]]
        return surface3d.payload(f"{tk} · term structure through time", yt, yd, [(as_of, grid)],
                                 x_label="TENOR", y_label="DATE",
                                 subtitle="ATM implied vol by tenor, past year (oldest at the back)",
                                 y_every=max(1, len(yd) // 8))
    if mode == "CHANGE":
        ld, lg = _lag_grid(hist, tenors, lag)
        if lg is None:
            return None
        diff = [[(a - b) if a is not None and b is not None else None for a, b in zip(ra, rb)]
                for ra, rb in zip(sf["delta"], lg)]
        return surface3d.payload(f"{tk} · surface change", sd.get("deltas", []), yt, [(f"vs {ld}", diff)],
                                 z_label="Δ IV", z_fmt="pts", diverging=True,
                                 subtitle=f"today minus {ld}: coral = richer, teal = cheaper")
    if mode == "VRP LANDSCAPE":
        core = tuple((_data()["screen"].get("core") or [])[:30])
        if not core:
            return None
        df = _vrp_panel(core)
        if df.empty:
            return None
        piv = df.pivot_table(index="date", columns="ticker", values="vrp_30").sort_index().tail(90)
        piv = piv[[c for c in core if c in piv.columns]]
        if piv.empty:
            return None
        step = max(1, len(piv) // 60)
        piv = piv.iloc[::-1].iloc[::step].iloc[::-1]
        grid = [[None if not np.isfinite(v) else float(v) for v in row] for row in piv.to_numpy()]
        return surface3d.payload("core basket · VRP landscape", list(piv.columns),
                                 [d.strftime("%y-%m-%d") for d in piv.index], [(as_of, grid)],
                                 x_label="UNDERLYING", y_label="DATE", z_label="VRP30", z_fmt="pts",
                                 diverging=True, subtitle="ex-earnings IV30 minus GARCH forecast: coral = rich",
                                 x_every=max(1, len(piv.columns) // 15), y_every=max(1, len(piv) // 8))
    return None


def _sigma_bands(row: dict, tenors: list[int]) -> None:
    st.markdown(section("σ-bands — each tenor against its own year", 1,
                        help="Shaded = ±1σ and ±2σ of the tenor's trailing-year IV; the hairline is the year's "
                             "range; the dot is today, coloured by its z."), unsafe_allow_html=True)
    bands = row.get("iv_bands") or {}
    if not bands:
        st.caption("Needs 20+ stored sessions for this name.")
        return
    html = "".join(volviz.sigma_strip(_tenor_label(t), (row.get("iv") or {}).get(str(t)),
                                      (bands.get(str(t)) or {}).get("mean"), (bands.get(str(t)) or {}).get("sd"),
                                      (bands.get(str(t)) or {}).get("lo"), (bands.get(str(t)) or {}).get("hi"),
                                      w=560) for t in tenors)
    st.markdown(f'<div style="max-width:620px">{html}</div>', unsafe_allow_html=True)
    if row.get("ivr_src") != "observed":
        st.caption(f"History source: {row.get('ivr_src')} — bands become fully observed after 252 sessions.")


def _slices(tk: str, sf: dict, sd: dict, lag: int) -> None:
    tenors = sd.get("tenors", [7, 14, 30, 60, 90, 180, 365])
    deltas = sd.get("deltas", [])
    hist = _surf_hist(tk)
    ld, lg = _lag_grid(hist, tenors, lag)
    c1, c2 = st.columns(2)
    with c1:
        ten = st.select_slider("Smile at tenor", options=tenors, value=30, key="vsf_ten",
                               format_func=_tenor_label)
        i = tenors.index(ten)
        rows = [{"delta": d, "k": k, "series": "today", "iv": sf["delta"][i][k]} for k, d in enumerate(deltas)]
        if lg:
            rows += [{"delta": d, "k": k, "series": ld, "iv": lg[i][k]} for k, d in enumerate(deltas)]
        d = pd.DataFrame(rows).dropna()

        def build(alt):
            return alt.Chart(d).mark_line(point=True).encode(
                x=alt.X("delta:N", sort=deltas, title="DELTA"),
                y=alt.Y("iv:Q", title="IV", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", scale=alt.Scale(range=[T.teal, T.faint]), legend=alt.Legend(orient="bottom")),
                tooltip=["delta", "series", alt.Tooltip("iv:Q", format=".2%")]).properties(height=240)
        st.markdown(section(f"Smile · {_tenor_label(ten)}", 2), unsafe_allow_html=True)
        uc.render_chart(build, d, width="stretch")
    with c2:
        dl = st.select_slider("Term at delta", options=deltas, value="50" if "50" in deltas else deltas[0],
                              key="vsf_delta")
        j = deltas.index(dl)
        rows = [{"tenor": t, "series": "today", "iv": sf["delta"][i][j]} for i, t in enumerate(tenors)]
        if lg:
            rows += [{"tenor": t, "series": ld, "iv": lg[i][j]} for i, t in enumerate(tenors)]
        d = pd.DataFrame(rows).dropna()

        def build(alt):
            return alt.Chart(d).mark_line(point=True).encode(
                x=alt.X("tenor:Q", scale=alt.Scale(type="log"), title="DAYS"),
                y=alt.Y("iv:Q", title="IV", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", scale=alt.Scale(range=[T.teal, T.faint]), legend=alt.Legend(orient="bottom")),
                tooltip=["tenor", "series", alt.Tooltip("iv:Q", format=".2%")]).properties(height=240)
        st.markdown(section(f"Term · {dl}Δ", 3), unsafe_allow_html=True)
        uc.render_chart(build, d, width="stretch")
    grid = pd.DataFrame([[None if v is None else round(v * 100, 1) for v in r] for r in sf["delta"]],
                        index=[_tenor_label(t) for t in tenors], columns=deltas)
    st.markdown(section("Surface grid (IV %, delta × tenor)", 4), unsafe_allow_html=True)
    st.dataframe(grid, width="stretch")


def render() -> None:
    d = _data()
    sd, sc = d["surfaces"], d["screen"]
    surf = sd.get("surf") or {}
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    st.markdown(evidence_rating("B", "measurement, not a signal",
                                "The surface is a faithful read of end-of-day quotes; what it predicts is the "
                                "screener's job. Wings are interpolated where a delta was not quoted (hover "
                                "says so)."), unsafe_allow_html=True)
    if not surf:
        st.info("No surfaces yet — they arrive with the nightly VOL SCREENER build.")
        st.caption(DISCLAIMER)
        return
    st.markdown(stamp(sd.get("as_of", ""), "VOL SURFACE"), unsafe_allow_html=True)
    rows = {r["ticker"]: r for r in sc.get("rows", [])}
    order = (sc.get("core") or []) + [t for t in surf if t not in set(sc.get("core") or [])]
    order = [t for t in order if t in surf]
    c1, c2, c3 = st.columns([1.2, 3, 1])
    tk = c1.selectbox("Underlying", order, index=order.index("SPY") if "SPY" in order else 0, key="vsf_ticker")
    mode = c2.radio("View", MODES, horizontal=True, key="vsf_mode")
    lag = c3.select_slider("Compare", options=[1, 5, 21], value=5, key="vsf_lag",
                           format_func=lambda n: f"−{n} sess")
    sf = surf[tk]
    row = rows.get(tk, {})
    p = _scene(mode, tk, sf, sd, row, lag)
    if p is None:
        st.info("Not enough stored history for this view yet — it fills in as nightly sessions accrue. "
                "SURFACE and MONEYNESS work from day one.")
    else:
        surface3d.render(p, height=600)
    _sigma_bands(row, sd.get("tenors", [7, 14, 30, 60, 90, 180, 365]))
    _slices(tk, sf, sd, lag)
    st.caption(DISCLAIMER)

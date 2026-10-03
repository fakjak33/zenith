"""EPHEMERIS tab — blind-chart tape-reading trainer.

Everything below the profile bar runs inside one @st.fragment: a full ZENITH
rerun executes all tabs (~30 s), so the game must rerun only itself.
"""

from __future__ import annotations

import json
import random
import time

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import EPHEMERIS_DB_SECRET, EPHEMERIS_FILES, THEME
from ..ui_theme import evidence_rating, key_findings, section
from . import (CLASSES, CONVICTION, DEFAULT_CONVICTION, DEFAULT_HORIZON, DEFAULT_LOOKBACK, DISCLAIMER,
               HORIZONS, LOOKBACKS, START_BALANCE, SURVIVORSHIP_NOTE, stake_for)
from . import chart as board
from . import profiles
from . import universe as uni
from .repo import get_repo
from .sampler import NoChartError, draw
from .scoring import score_trade
from .store_px import PxStore

SUBVIEWS = ["Play", "History"]

_EVIDENCE_NOTE = ("Deliberate practice — many repetitions, immediate feedback, a measurable score — is the "
                  "best-supported route to perceptual expertise. Whether chart reading itself carries an edge "
                  "is contested: some technical patterns do carry information, but most discretionary traders "
                  "fail to beat simple benchmarks. That is why every result here is judged against the base "
                  "rate and a trend rule on the same charts, not against 50%.")

_FINDINGS = [
    {"stat": "Expert performance tracks accumulated deliberate practice with immediate, specific feedback.",
     "cite": "Ericsson, Krampe & Tesch-Römer (1993)"},
    {"stat": "Technical patterns, detected algorithmically, carry incremental information in US stocks.",
     "cite": "Lo, Mamaysky & Wang (2000)"},
    {"stat": "Fewer than 1% of day traders are predictably profitable after fees.",
     "cite": "Barber, Lee, Liu & Odean (2014)"},
]


# ================================================================ resources ====
@st.cache_resource(show_spinner=False)
def _store() -> PxStore:
    return PxStore()


@st.cache_data(ttl=3600, show_spinner=False)
def _universe() -> list[dict]:
    return uni.load()


def _db_url() -> str | None:
    try:
        return st.secrets.get(EPHEMERIS_DB_SECRET) or None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _repo_for(url: str | None):
    return get_repo(url)


def _repo():
    url = _db_url()
    try:
        return _repo_for(url), None
    except Exception as exc:              # hosted DB down/misconfigured -> keep playing locally
        return _repo_for(None), f"{type(exc).__name__}: {str(exc)[:140]}"


# ================================================================ badge ====
def today_badge() -> str | None:
    try:
        st_ = json.loads(EPHEMERIS_FILES["status"].read_text(encoding="utf-8"))
        n = (st_.get("daily") or {}).get("written")
        if not n:
            return None
        return uc.chip(f"EPHEMERIS — blind-chart practice · {n:,} instruments loaded",
                       color=THEME.mustard, sub="see EPHEMERIS tab")
    except Exception:
        return None


# ================================================================ helpers ====
def _account(repo, player_id: int, mode: str = "practice") -> dict:
    df = repo.guesses_df(player_id, mode)
    resets = repo.resets(player_id, mode)
    if resets and not df.empty:
        cut = pd.Timestamp(resets[-1]["ts"])
        df = df[df["ts"] > cut]
    pnl = float(df["pnl"].astype(float).sum()) if not df.empty else 0.0
    n = len(df)
    wins = int(df["win"].astype(int).sum()) if n else 0
    return {"balance": START_BALANCE + pnl, "n": n, "wins": wins}


def _settings() -> dict:
    with st.expander("SETTINGS — universe, window, horizon, blinding", expanded=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            classes = st.multiselect("Universe", CLASSES, default=list(CLASSES), key="eph_classes",
                                     help="Sampling is balanced across the classes you pick.")
        with c2:
            lookback = st.selectbox("Visible candles", LOOKBACKS, index=LOOKBACKS.index(DEFAULT_LOOKBACK),
                                    key="eph_lookback")
        with c3:
            hopts = list(HORIZONS) + ["Custom"]
            hsel = st.selectbox("Horizon (candles)", hopts, index=HORIZONS.index(DEFAULT_HORIZON),
                                key="eph_horizon")
            horizon = (int(st.number_input("Custom horizon", 1, 250, 15, key="eph_hcustom"))
                       if hsel == "Custom" else int(hsel))
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            hide_ticker = st.toggle("Hide ticker", True, key="eph_hide_ticker")
        with b2:
            hide_dates = st.toggle("Hide dates", True, key="eph_hide_dates")
        with b3:
            rebase = st.toggle("Rebase to 100", True, key="eph_rebase",
                               help="Off: real price levels, for practising round numbers.")
        with b4:
            st.caption("Timeframe: **Daily** (1H / 4H / Weekly / Monthly arrive in phase 3).")
    return {"classes": tuple(classes or CLASSES), "tf": "Daily", "lookback": int(lookback),
            "horizon": horizon, "hide_ticker": hide_ticker, "hide_dates": hide_dates, "rebase": rebase}


def _sig(s: dict) -> tuple:
    return (s["classes"], s["tf"], s["lookback"], s["horizon"])


def _new_round(repo, player: dict, s: dict) -> dict | None:
    try:
        ch = draw(_store(), _universe(), classes=s["classes"], tf=s["tf"], lookback=s["lookback"],
                  horizon=s["horizon"], seen=repo.seen_charts(player["id"]), rng=random.Random())
    except NoChartError as exc:
        st.warning(str(exc))
        return None
    return {"chart": ch, "sig": _sig(s), "state": "decide", "t0": time.time()}


# ================================================================ play ====
def _play(repo, player: dict) -> None:
    s = _settings()
    rnd = st.session_state.get("eph_round")
    if rnd is None or (rnd["state"] == "decide" and rnd["sig"] != _sig(s)):
        rnd = _new_round(repo, player, s)
        st.session_state["eph_round"] = rnd
    if rnd is None:
        return
    ch = rnd["chart"]

    acct = _account(repo, player["id"])
    if rnd["state"] == "revealed":       # don't spoil the reveal through the balance
        acct["balance"] -= rnd["result"]["pnl"]
        acct["n"] -= 1
        acct["wins"] -= int(rnd["result"]["win"])
    hit = f"{acct['wins'] / acct['n']:.0%}" if acct["n"] else "—"
    bal_c = THEME.teal if acct["balance"] >= START_BALANCE else THEME.coral
    st.markdown(uc.numeric_slab([
        {"label": "Paper account", "value": f"${acct['balance']:,.0f}", "color": bal_c,
         "sub": f"practice · started ${START_BALANCE:,.0f}"},
        {"label": "Calls", "value": f"{acct['n']:,}"},
        {"label": "Hit rate", "value": hit, "sub": "judge it vs base rate — phase 2"},
    ], min_width=140), unsafe_allow_html=True)

    if rnd["state"] == "decide":
        board.render(board.payload(ch, rebase=s["rebase"], hide_dates=s["hide_dates"],
                                   hide_ticker=s["hide_ticker"]))
        c1, c2 = st.columns([2, 3])
        with c1:
            conv = st.radio("Conviction", list(CONVICTION), index=list(CONVICTION).index(DEFAULT_CONVICTION),
                            horizontal=True, key="eph_conv",
                            help="Stake = $1,000 × 25% / 50% / 100%.")
        with c2:
            note = st.text_input("Why is the opposite move unlikely? (optional)", key="eph_note",
                                 max_chars=200, placeholder="one sentence, before you see the answer")
        u, d = st.columns(2)
        with u:
            up = st.button("▲  UP / LONG", type="primary", use_container_width=True, key="eph_up")
        with d:
            down = st.button("▼  DOWN / SHORT", use_container_width=True, key="eph_down")
        if up or down:
            _commit(repo, player, rnd, s, 1 if up else -1, conv, note)
            st.rerun(scope="fragment")
    else:
        p = board.payload(ch, rebase=s["rebase"], hide_dates=s["hide_dates"], hide_ticker=s["hide_ticker"],
                          direction=rnd["direction"], stake=rnd["stake"], result=rnd["result"])
        board.render(p)
        if st.button("NEXT CHART  ▸", type="primary", use_container_width=True, key="eph_next"):
            st.session_state["eph_round"] = None
            st.session_state.pop("eph_note", None)
            st.rerun(scope="fragment")

    st.caption(f"{SURVIVORSHIP_NOTE} Prices: yfinance, split- & dividend-adjusted. "
               f"Entry = close of the last visible candle.")


def _commit(repo, player, rnd, s, direction: int, conv: str, note: str) -> None:
    ch = rnd["chart"]
    stake = stake_for(conv)
    fut = ch.future()
    res = score_trade(direction=direction, entry=ch.entry, o=fut["o"], h=fut["h"], l=fut["l"],
                      c=fut["c"], stake=stake)
    repo.log_guess({
        "player_id": player["id"], "mode": "practice", "ticker": ch.ticker, "name": ch.name,
        "asset_class": ch.cls, "sector": ch.sector, "timeframe": ch.tf, "horizon": ch.horizon,
        "lookback": ch.lookback, "window_start": ch.window_start.isoformat(),
        "decision_date": ch.decision_date.isoformat(), "chart_key": ch.key,
        "indicators_json": [], "blinding_json": {k: s[k] for k in ("hide_ticker", "hide_dates", "rebase")},
        "direction": direction, "conviction": conv, "stake": stake, "entry": ch.entry,
        "note": (note or "").strip() or None, "decision_ms": int((time.time() - rnd["t0"]) * 1000),
        **{k: res[k] for k in ("outcome", "market_ret", "trade_ret", "pnl", "win", "r_mult", "atr_ret",
                               "mfe", "mae", "mfe_full", "mae_full", "candles_held", "exit_reason",
                               "ambiguous", "gap_fill")},
    })
    rnd.update(state="revealed", result=res, direction=direction, stake=stake)


def _history(repo, player: dict) -> None:
    df = repo.guesses_df(player["id"])
    if df.empty:
        st.info("No calls yet — play a few charts first.")
        return
    view = df.sort_values("id", ascending=False).head(200)
    out = pd.DataFrame({
        "When": view["ts"].dt.strftime("%Y-%m-%d %H:%M"), "Mode": view["mode"],
        "Ticker": view["ticker"], "Class": view["asset_class"], "TF": view["timeframe"],
        "H": view["horizon"], "Call": view["direction"].map({1: "UP", -1: "DOWN"}),
        "Conv": view["conviction"], "Move": view["market_ret"].astype(float),
        "Trade": view["trade_ret"].astype(float), "PnL": view["pnl"].astype(float),
        "Exit": view["exit_reason"], "Note": view["note"],
    })
    st.dataframe(out.style.format({"Move": "{:+.2%}", "Trade": "{:+.2%}", "PnL": lambda v: f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"})
                 .map(lambda v: uc.grad_diverging(v, 0.05), subset=["Trade"]),
                 use_container_width=True, height=460, hide_index=True)
    st.download_button("Download history (CSV)", df.to_csv(index=False).encode(),
                       file_name=f"ephemeris_{player['handle']}.csv", mime="text/csv")


# ================================================================= main ====
def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating("B", "practice method well supported; chart-reading edge is contested",
                                _EVIDENCE_NOTE), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    repo, db_err = _repo()
    if db_err:
        st.warning(f"Hosted database unavailable ({db_err}) — saving to local SQLite for now.")
    player = profiles.current_player(repo)
    profiles.profile_bar(repo, player)
    if player is None:
        return
    if not _store().available("Daily"):
        st.info("No price data yet — the nightly EPHEMERIS job publishes it. "
                "Locally: `python -m zenith.ephemeris.prefetch --limit 120`.")
        return

    _body(repo, player)
    st.caption(f"Storage: {repo.label}.")


@st.fragment
def _body(repo, player: dict) -> None:
    """Everything interactive reruns only this fragment -- a full ZENITH rerun
    executes all tabs and takes tens of seconds."""
    sub = st.radio("View", SUBVIEWS, horizontal=True, key="eph_sub", label_visibility="collapsed")
    st.markdown(section("PRACTICE — call the next candles" if sub == "Play" else "YOUR CALLS", 1),
                unsafe_allow_html=True)
    if sub == "Play":
        _play(repo, player)
    else:
        _history(repo, player)

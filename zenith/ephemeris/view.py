"""EPHEMERIS tab — blind-chart tape-reading trainer.

Everything below the profile bar runs inside one @st.fragment: a full ZENITH
rerun executes all tabs (~30 s), so the game must rerun only itself.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import EPHEMERIS_DB_SECRET, EPHEMERIS_FILES, THEME
from ..ui_theme import evidence_rating, key_findings, section
from . import (CLASSES, CONVICTION, TIMEFRAMES, DEFAULT_CONVICTION, DEFAULT_HORIZON, DEFAULT_LOOKBACK, DISCLAIMER,
               HORIZONS, LOOKBACKS, START_BALANCE, SURVIVORSHIP_NOTE, stake_for)
from . import chart as board
from . import indicators as ind
from . import profiles
from . import universe as uni
from .repo import db_url_problem, get_repo
from . import daily as daily_five
from .sampler import NoChartError, chart_at, draw
from .benchmarks import TREND_RULE_DEFAULT, base_rate, trend_rule_call, trend_rule_label
from .regime import summary as regime_summary
from .scoring import InvalidLevels, level_from_spec, score_trade
from .store_px import PxStore

SUBVIEWS = ["Play", "Daily Five", "History"]

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
        return _repo_for(None), _db_error_text(url, exc)


def _db_error_text(url: str | None, exc: Exception) -> str:
    """Actionable, password-free explanation of a hosted-DB failure."""
    raw = " ".join(str(exc).split())
    if url and url in raw:
        raw = raw.replace(url, "<url>")
    diag = db_url_problem(url) if url else None
    if diag:
        return f"{diag}. Fix the `ephemeris_db_url` secret."
    if "password authentication failed" in raw:
        return ("Supabase rejected the database password. In Supabase: Project Settings → Database → "
                "Reset database password, choose one with only letters and digits, then paste the Session "
                "pooler URI with that password (no [ ] brackets) into the `ephemeris_db_url` secret.")
    if "could not translate host name" in raw:
        return "the database host name does not resolve — copy the Session pooler URI again."
    return f"{type(exc).__name__}: {raw[:300]}"


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


ERA_YEARS = ["Any", 1990, 2000, 2010, 2015, 2020]

# One-click trading-style bundles. Keys are the settings widgets' session keys.
DRILLS = {
    "Classic · Daily · 10 ahead · SMA 20/50/200": {
        "tf": "Daily", "lookback": 120, "horizon": 10, "indicators": ind.DEFAULT_SET},
    "Swing · 4H · 20 ahead · EMA 21/50 + RSI": {
        "tf": "4H", "lookback": 250, "horizon": 20,
        "indicators": [{"id": "ema", "params": {"lengths": "21,50"}}, {"id": "rsi", "params": {"n": 14}}]},
    "Position · Weekly · 30 ahead · SMA 40 + slope": {
        "tf": "Weekly", "lookback": 250, "horizon": 30,
        "indicators": [{"id": "sma", "params": {"lengths": "40"}}, {"id": "slope", "params": {"n": 40, "m": 10}}]},
    "Intraday · 1H · 5 ahead · VWAP + volume": {
        "tf": "1H", "lookback": 120, "horizon": 5,
        "indicators": [{"id": "rvwap", "params": {"n": 20}}, {"id": "volume", "params": {"n": 20}}]},
    "Macro · Monthly · 10 ahead · SMA 10/20": {
        "tf": "Monthly", "lookback": 120, "horizon": 10,
        "indicators": [{"id": "sma", "params": {"lengths": "10,20"}}]},
    "Crisis tape · Daily · 5 ahead · BB + RSI": {
        "tf": "Daily", "lookback": 120, "horizon": 5, "crisis": True,
        "indicators": [{"id": "bb", "params": {"n": 20, "k": 2.0}}, {"id": "rsi", "params": {"n": 14}}]},
}

_DEFAULTS = {"eph_classes": list(CLASSES), "eph_tf": "Daily", "eph_lookback": DEFAULT_LOOKBACK,
             "eph_horizon": DEFAULT_HORIZON, "eph_hcustom": 15, "eph_hide_ticker": True,
             "eph_hide_dates": True, "eph_rebase": True, "eph_rule_n": TREND_RULE_DEFAULT["n"],
             "eph_rule_m": TREND_RULE_DEFAULT["m"], "eph_min_year": "Any", "eph_crisis": False}


def _init_defaults() -> None:
    """Seed widget state once, so presets can write it without Streamlit's
    'default value AND session state' warning."""
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if "eph_ind_sel" not in st.session_state:
        _write_indicators(ind.DEFAULT_SET)


def _write_indicators(specs: list[dict]) -> None:
    st.session_state["eph_ind_sel"] = [x["id"] for x in specs if x["id"] in ind.REGISTRY]
    for x in specs:
        defaults = ind.REGISTRY.get(x["id"], {}).get("params", {})
        for k, v in (x.get("params") or {}).items():
            if k in defaults:              # widget type must match (int vs float vs str)
                st.session_state[f"eph_ip_{x['id']}_{k}"] = type(defaults[k])(v)


def _apply_drill(repo, player_id: int) -> None:
    name = st.session_state.get("eph_drill")
    d = DRILLS.get(name) or repo.presets(player_id, "params").get(name)
    if not d:
        return
    st.session_state["eph_tf"] = d.get("tf", "Daily")
    st.session_state["eph_lookback"] = int(d.get("lookback", DEFAULT_LOOKBACK))
    h = int(d.get("horizon", DEFAULT_HORIZON))
    if h in HORIZONS:
        st.session_state["eph_horizon"] = h
    else:
        st.session_state["eph_horizon"], st.session_state["eph_hcustom"] = "Custom", h
    if d.get("classes"):
        st.session_state["eph_classes"] = [c for c in d["classes"] if c in CLASSES]
    st.session_state["eph_crisis"] = bool(d.get("crisis", False))
    st.session_state["eph_min_year"] = d.get("min_year", "Any")
    if "rule" in d:
        st.session_state["eph_rule_n"], st.session_state["eph_rule_m"] = d["rule"]["n"], d["rule"]["m"]
    if "indicators" in d:
        _write_indicators(d["indicators"])


def _depth_ok(tf: str, need: int) -> int:
    return sum(1 for n in _store().counts(tf).values() if n >= need)


def _settings(repo, player: dict) -> dict:
    _init_defaults()
    saved = repo.presets(player["id"], "params")
    st.selectbox("Drill preset", list(DRILLS) + sorted(saved), index=None, key="eph_drill",
                 placeholder="drill presets — one click switches timeframe, horizon and indicators…",
                 on_change=_apply_drill, args=(repo, player["id"]), label_visibility="collapsed")
    tfs = [tf for tf in TIMEFRAMES if _store().available(tf)]
    if st.session_state.get("eph_tf") not in tfs:
        st.session_state["eph_tf"] = "Daily"
    with st.expander("SETTINGS — timeframe, universe, window, horizon, era, blinding, benchmark",
                     expanded=False):
        c0, c1, c2, c3 = st.columns([1, 3, 1, 1])
        with c0:
            tf = st.selectbox("Candle size", tfs, key="eph_tf",
                              help="Weekly / Monthly are resampled from daily; 4H from 1H (US sessions "
                                   "09:30–13:30 and 13:30–16:00; 24h assets on 4-hour boundaries). "
                                   "Timeframes without stored history are hidden.")
        with c1:
            classes = st.multiselect("Universe", CLASSES, key="eph_classes",
                                     help="Sampling is balanced across the classes you pick.")
        with c2:
            lookback = st.selectbox("Visible candles", LOOKBACKS, key="eph_lookback")
        with c3:
            hsel = st.selectbox("Horizon (candles)", list(HORIZONS) + ["Custom"], key="eph_horizon")
            horizon = (int(st.number_input("Custom horizon", 1, 250, key="eph_hcustom"))
                       if hsel == "Custom" else int(hsel))
        e1, e2, e3 = st.columns([1, 1, 2])
        with e1:
            min_year = st.selectbox("Era: from", ERA_YEARS, key="eph_min_year")
        with e2:
            crisis = st.toggle("Crisis periods only", key="eph_crisis",
                               help="Decision date inside 2000–02, 2007–09, Feb–Jun 2020 or 2022.")
        with e3:
            n_ok = _depth_ok(tf, int(lookback) + horizon + 1)
            msg = f"{n_ok:,} instruments have {int(lookback) + horizon + 1}+ {tf} candles."
            (st.warning if n_ok < 20 else st.caption)(
                msg + (" Too few — shorten the window or horizon." if n_ok < 20 else ""))
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            hide_ticker = st.toggle("Hide ticker", key="eph_hide_ticker")
        with b2:
            hide_dates = st.toggle("Hide dates", key="eph_hide_dates")
        with b3:
            rebase = st.toggle("Rebase to 100", key="eph_rebase",
                               help="Off: real price levels, for practising round numbers.")
        with b4:
            rule_n = int(st.number_input("Trend rule SMA", 5, 300, key="eph_rule_n",
                                         help="Benchmark: long if close > SMA and SMA rising, short if "
                                              "both reversed, else no trade."))
        with b5:
            rule_m = int(st.number_input("…slope over (bars)", 1, 100, key="eph_rule_m"))
    specs = _indicator_settings(repo, player)
    s = {"classes": tuple(classes or CLASSES), "tf": tf, "lookback": int(lookback),
         "horizon": horizon, "hide_ticker": hide_ticker, "hide_dates": hide_dates, "rebase": rebase,
         "rule": {"n": rule_n, "m": rule_m}, "indicators": specs,
         "min_year": None if min_year == "Any" else int(min_year), "crisis": bool(crisis)}
    with st.popover("Save as drill preset"):
        nm = st.text_input("Preset name", key="eph_drill_name", placeholder="e.g. My 4H crypto swing")
        if st.button("SAVE PRESET", key="eph_drill_save") and nm.strip():
            repo.save_preset(player["id"], "params", nm.strip()[:48], {
                "tf": tf, "lookback": int(lookback), "horizon": horizon, "classes": list(s["classes"]),
                "rule": s["rule"], "indicators": specs, "crisis": s["crisis"],
                "min_year": min_year})
            st.toast(f"Saved drill “{nm.strip()[:48]}”.")
    return s


_BUILTIN_SETS = {"Default (SMA 20/50/200 + volume)": {"specs": ind.DEFAULT_SET}, "None": {"specs": []}}


def _apply_ind_preset(repo, player_id: int) -> None:
    name = st.session_state.get("eph_ind_preset")
    sets = dict(_BUILTIN_SETS)
    sets.update(repo.presets(player_id, "indicators"))
    specs = (sets.get(name) or {}).get("specs")
    if specs is not None:
        _write_indicators(specs)


def _indicator_settings(repo, player: dict) -> list[dict]:
    with st.expander("INDICATORS — overlays & lower panes", expanded=False):
        saved = repo.presets(player["id"], "indicators")
        opts = list(_BUILTIN_SETS) + sorted(saved)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.selectbox("Load a set", opts, index=None, placeholder="favourite sets…", key="eph_ind_preset",
                         on_change=_apply_ind_preset, args=(repo, player["id"]))
        if "eph_ind_sel" not in st.session_state:
            st.session_state["eph_ind_sel"] = [x["id"] for x in ind.DEFAULT_SET]
        sel = st.multiselect("Indicators", list(ind.REGISTRY), key="eph_ind_sel",
                             format_func=lambda k: ind.REGISTRY[k]["label"],
                             help="Up to 4 lower panes. Every value at or before NOW uses only data "
                                  "available at that candle.")
        specs = []
        for sid in sel:
            params = {}
            defaults = ind.REGISTRY[sid]["params"]
            if defaults:
                cols = st.columns(len(defaults) + 1)
                cols[0].caption(ind.REGISTRY[sid]["label"])
                for col, (k, dv) in zip(cols[1:], defaults.items()):
                    key = f"eph_ip_{sid}_{k}"
                    if key not in st.session_state:
                        st.session_state[key] = dv
                    if isinstance(dv, str):
                        params[k] = col.text_input(k, key=key)
                    elif isinstance(dv, float):
                        params[k] = float(col.number_input(k, 0.1, 10.0, step=0.1, key=key))
                    else:
                        params[k] = int(col.number_input(k, 1, 400, step=1, key=key))
            specs.append({"id": sid, "params": params})
        with c2:
            nm = st.text_input("Save current set as", key="eph_ind_name", placeholder="e.g. Swing: EMA 21 + RSI")
        with c3:
            st.write("")
            if st.button("SAVE", key="eph_ind_save", use_container_width=True) and nm.strip():
                repo.save_preset(player["id"], "indicators", nm.strip()[:40], {"specs": specs})
                st.toast(f"Saved indicator set “{nm.strip()[:40]}”.")
    return specs


def _sig(s: dict) -> tuple:
    return (s["classes"], s["tf"], s["lookback"], s["horizon"], s["min_year"], s["crisis"])


def _new_round(repo, player: dict, s: dict) -> dict | None:
    try:
        ch = draw(_store(), _universe(), classes=s["classes"], tf=s["tf"], lookback=s["lookback"],
                  horizon=s["horizon"], seen=repo.seen_charts(player["id"]), rng=random.Random(),
                  min_year=s["min_year"], crisis_only=s["crisis"])
    except NoChartError as exc:
        st.warning(str(exc))
        return None
    a = ind.atr(ch.h, ch.l, ch.c, 14)[ch.t]
    return {"chart": ch, "sig": _sig(s), "state": "decide", "t0": time.time(),
            "atr": float(a) if np.isfinite(a) else None}


def _stops_ui(rnd) -> dict:
    """Optional SL/TP as % or ATR multiples -> {"on", "mode", "sl", "tp"} (distances)."""
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        on = st.toggle("Stop / target", False, key="eph_stops_on",
                       help="Walk-forward on OHLC: first level touched exits; both in one candle = stop "
                            "(flagged ambiguous); gaps fill at the open.")
    if not on:
        return {"on": False}
    atr_ok = rnd.get("atr") is not None
    with c2:
        mode = st.radio("Units", ["%", "ATR"] if atr_ok else ["%"], horizontal=True, key="eph_stops_mode")
    dflt = (2.0, 4.0) if mode == "%" else (1.5, 3.0)
    with c3:
        sl = st.number_input(f"Stop ({mode}) · 0 = none", 0.0, 50.0, dflt[0], 0.25, key=f"eph_sl_{mode}")
    with c4:
        tp = st.number_input(f"Target ({mode}) · 0 = none", 0.0, 100.0, dflt[1], 0.25, key=f"eph_tp_{mode}")
    return {"on": True, "mode": mode, "sl": sl or None, "tp": tp or None}


def _levels(rnd, stops: dict, direction: int) -> tuple[float | None, float | None]:
    ch = rnd["chart"]
    if not stops.get("on"):
        return None, None
    sl = (level_from_spec(direction, ch.entry, "sl", stops["mode"], stops["sl"], rnd["atr"])
          if stops["sl"] else None)
    tp = (level_from_spec(direction, ch.entry, "tp", stops["mode"], stops["tp"], rnd["atr"])
          if stops["tp"] else None)
    return sl, tp


# ================================================================ play ====
def _play(repo, player: dict) -> None:
    s = _settings(repo, player)
    rnd = st.session_state.get("eph_round")
    if rnd is None or (rnd["state"] == "decide" and rnd["sig"] != _sig(s)):
        rnd = _new_round(repo, player, s)
        st.session_state["eph_round"] = rnd
    if rnd is None:
        return
    ch = rnd["chart"]
    ind_data = ind.compute(s["indicators"], ch.o, ch.h, ch.l, ch.c, ch.v, anchor=ch.vis0)

    acct = _account(repo, player["id"])
    if rnd["state"] == "revealed":       # don't spoil the reveal through the balance
        acct["balance"] -= rnd["result"]["pnl"]
        acct["n"] -= 1
        acct["wins"] -= int(rnd["result"]["win"])
    hit = f"{acct['wins'] / acct['n']:.0%}" if acct["n"] else "—"
    bal_c = THEME.teal if acct["balance"] >= START_BALANCE else THEME.coral
    st.markdown(uc.numeric_slab([
        {"label": "Practice account", "value": f"${acct['balance']:,.0f}", "color": bal_c,
         "sub": f"started ${START_BALANCE:,.0f}"},
        {"label": "Calls", "value": f"{acct['n']:,}"},
        {"label": "Hit rate", "value": hit, "sub": "full benchmark read in STATS (phase 4)"},
    ], min_width=140), unsafe_allow_html=True)

    blind = {k: s[k] for k in ("rebase", "hide_dates", "hide_ticker")}
    if rnd["state"] == "decide":
        stops = _stops_ui(rnd)
        direction = 0
        if stops["on"]:
            direction = 1 if st.session_state.get("eph_dir", "▲ LONG") == "▲ LONG" else -1
        sl, tp = _levels(rnd, stops, direction or 1)
        board.render(board.payload(ch, **blind, sl=sl, tp=tp, direction=0, ind=ind_data))
        c1, c2 = st.columns([2, 3])
        with c1:
            conv = st.radio("Conviction", list(CONVICTION), index=list(CONVICTION).index(DEFAULT_CONVICTION),
                            horizontal=True, key="eph_conv",
                            help="Stake = $1,000 × 25% / 50% / 100%.")
        with c2:
            note = st.text_input("Why is the opposite move unlikely? (optional)", key="eph_note",
                                 max_chars=200, placeholder="one sentence, before you see the answer")
        if stops["on"]:
            d1, d2 = st.columns([1, 2])
            with d1:
                st.radio("Direction", ["▲ LONG", "▼ SHORT"], horizontal=True, key="eph_dir",
                         label_visibility="collapsed")
            with d2:
                go = st.button("SUBMIT CALL  ▸", type="primary", use_container_width=True, key="eph_submit")
            if go and _commit(repo, player, rnd, s, direction, conv, note, stops, sl, tp):
                st.rerun(scope="fragment")
        else:
            u, d = st.columns(2)
            with u:
                up = st.button("▲  UP / LONG", type="primary", use_container_width=True, key="eph_up")
            with d:
                down = st.button("▼  DOWN / SHORT", use_container_width=True, key="eph_down")
            if (up or down) and _commit(repo, player, rnd, s, 1 if up else -1, conv, note, stops, None, None):
                st.rerun(scope="fragment")
    else:
        p = board.payload(ch, **blind, direction=rnd["direction"], stake=rnd["stake"], result=rnd["result"],
                          sl=rnd.get("sl"), tp=rnd.get("tp"), ind=ind_data)
        board.render(p)
        if st.button("NEXT CHART  ▸", type="primary", use_container_width=True, key="eph_next"):
            st.session_state["eph_round"] = None
            st.session_state.pop("eph_note", None)
            st.rerun(scope="fragment")

    st.caption(f"{SURVIVORSHIP_NOTE} Prices: yfinance, split- & dividend-adjusted (total-return style, "
               f"so long-run drift is upward). Entry = close of the last visible candle.")


def context_rows(ch, s: dict, res: dict, rule: int, br: dict | None) -> list[tuple[str, str]]:
    rule_ret = rule * res["market_ret"]
    return [
        ("BASE RATE", f"{br['p']:.0%} of {ch.cls} {ch.tf.lower()} {br['h']}-bar moves go up (n={br['n']:,})"
         if br else "—"),
        (f"TREND RULE (SMA{s['rule']['n']})", trend_rule_label(rule)
         + (f" → {rule_ret * 100:+.2f}%" if rule else "")),
        ("REGIME", regime_summary(ch.extra.get("regime", {})) or "—"),
    ]


def _commit(repo, player, rnd, s, direction: int, conv: str, note: str, stops: dict,
            sl: float | None, tp: float | None, mode: str = "practice") -> int | None:
    ch = rnd["chart"]
    stake = stake_for(conv)
    fut = ch.future()
    try:
        res = score_trade(direction=direction, entry=ch.entry, o=fut["o"], h=fut["h"], l=fut["l"],
                          c=fut["c"], stake=stake, sl=sl, tp=tp, atr=rnd.get("atr"))
    except InvalidLevels as exc:
        st.error(str(exc))
        return None
    br = base_rate(ch.cls, ch.tf, ch.horizon)
    rule = trend_rule_call(ch.c, ch.t, s["rule"]["n"], s["rule"]["m"])
    res["context_rows"] = context_rows(ch, s, res, rule, br)
    rg = ch.extra.get("regime", {})

    def spec(k):
        if not (stops.get("on") and stops.get(k)):
            return None
        return f"{stops[k]:g}{'%' if stops['mode'] == '%' else '×ATR'}"

    gid = repo.log_guess({
        "player_id": player["id"], "mode": mode, "ticker": ch.ticker, "name": ch.name,
        "asset_class": ch.cls, "sector": ch.sector, "timeframe": ch.tf, "horizon": ch.horizon,
        "lookback": ch.lookback, "window_start": ch.window_start.isoformat(),
        "decision_date": ch.decision_date.isoformat(), "chart_key": ch.key,
        "indicators_json": s["indicators"],
        "blinding_json": {k: s[k] for k in ("hide_ticker", "hide_dates", "rebase")},
        "direction": direction, "conviction": conv, "stake": stake, "entry": ch.entry,
        "sl": sl, "tp": tp, "sl_spec": spec("sl"), "tp_spec": spec("tp"),
        "note": (note or "").strip() or None, "decision_ms": int((time.time() - rnd["t0"]) * 1000),
        "base_rate": br["p"] if br else None, "rule_call": rule, "rule_ret": rule * res["market_ret"],
        "regime_json": rg | {"rule": s["rule"]}, "setup_tag": None,
        **{k: res[k] for k in ("outcome", "market_ret", "trade_ret", "pnl", "win", "r_mult", "atr_ret",
                               "mfe", "mae", "mfe_full", "mae_full", "candles_held", "exit_reason",
                               "ambiguous", "gap_fill")},
    })
    rnd.update(state="revealed", result=res, direction=direction, stake=stake, sl=sl, tp=tp)
    return gid


# ================================================================ daily five ====
_DAILY_S = {"indicators": daily_five.INDICATORS, "rule": dict(TREND_RULE_DEFAULT), "hide_ticker": True,
            "hide_dates": True, "rebase": True}


def _uni_row(ticker: str) -> dict | None:
    return next((r for r in _universe() if r["ticker"] == ticker), None)


def _daily_chart(slot: dict):
    row = _uni_row(slot["ticker"])
    if row is None:
        return None
    s = daily_five.SETTINGS
    return chart_at(_store(), row, s["tf"], s["lookback"], s["horizon"], slot["decision"])


def _daily_played(repo, player_id: int) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in repo.daily_for(player_id):
        out.setdefault(r["game_date"], {})[int(r["slot"])] = r
    return out


def _daily_results_html(rows: list[dict]) -> str:
    cells = []
    for r in rows:
        c = THEME.teal if r["win"] else THEME.coral
        cells.append(f'<span style="display:inline-block;width:2.2rem;height:2.2rem;margin-right:.35rem;'
                     f'background:{c};color:#000;font-family:{THEME.font_display};font-size:1.5rem;'
                     f'text-align:center;line-height:2.2rem;">{"▲" if r["direction"] > 0 else "▼"}</span>')
    return '<div style="margin:.3rem 0 .8rem 0;">' + "".join(cells) + "</div>"


def _daily(repo, player: dict) -> None:
    day = daily_five.today_utc()
    sched = daily_five.load_schedule()
    slots = daily_five.slots_for(day, sched)
    played = _daily_played(repo, player["id"])
    completed = {pd.Timestamp(d).date() for d, s in played.items() if len(s) >= daily_five.SLOTS}
    streak_n = daily_five.streak(completed, day)
    today_rows = played.get(day.isoformat(), {})
    rnd = st.session_state.get("eph_daily_round")
    if rnd and rnd.get("day") != day.isoformat():
        rnd = st.session_state["eph_daily_round"] = None

    # don't let the slab spoil an in-progress reveal
    shown = dict(today_rows)
    if rnd and rnd["state"] == "revealed":
        shown.pop(rnd["slot"], None)
    acct = _account(repo, player["id"], "daily")
    if rnd and rnd["state"] == "revealed":
        acct["balance"] -= rnd["result"]["pnl"]
    st.markdown(uc.numeric_slab([
        {"label": f"Daily Five #{daily_five.game_number(day)}", "value": f"{len(shown)}/{daily_five.SLOTS}",
         "sub": day.isoformat() + " UTC"},
        {"label": "Streak", "value": f"{streak_n}", "sub": f"best {daily_five.best_streak(completed)}",
         "color": THEME.mustard},
        {"label": "Daily account", "value": f"${acct['balance']:,.0f}",
         "color": THEME.teal if acct["balance"] >= START_BALANCE else THEME.coral,
         "sub": "separate from practice"},
    ], min_width=140), unsafe_allow_html=True)
    if len(slots) < daily_five.SLOTS:
        st.info("Today's Daily Five is being prepared by the nightly job — check back soon, or practise meanwhile.")
        return
    st.caption("Same five charts for every player today. Fixed settings so days compare: Daily candles, "
               "120 visible, call 10 ahead, SMA 20/50/200 + volume, no stops. One attempt per chart.")

    if rnd and rnd["state"] == "revealed":
        ch = rnd["chart"]
        ind_data = ind.compute(_DAILY_S["indicators"], ch.o, ch.h, ch.l, ch.c, ch.v, anchor=ch.vis0)
        board.render(board.payload(ch, rebase=True, hide_dates=True, hide_ticker=True,
                                   direction=rnd["direction"], stake=rnd["stake"], result=rnd["result"],
                                   ind=ind_data))
        last = len(today_rows) >= daily_five.SLOTS
        if st.button("SEE TODAY'S RESULT  ▸" if last else "NEXT CHART  ▸", type="primary",
                     use_container_width=True, key="eph_d_next"):
            st.session_state["eph_daily_round"] = None
            st.session_state.pop("eph_d_note", None)
            st.rerun(scope="fragment")
        return

    if len(today_rows) >= daily_five.SLOTS:
        _daily_done(day, [today_rows[i] for i in range(daily_five.SLOTS)], streak_n)
    else:
        nxt = next(s for s in slots if s["slot"] not in today_rows)
        if not rnd or rnd["slot"] != nxt["slot"]:
            ch = _daily_chart(nxt)
            if ch is None:
                st.error("This chart's price history is unavailable right now — try again after the next data refresh.")
                return
            a = ind.atr(ch.h, ch.l, ch.c, 14)[ch.t]
            rnd = {"chart": ch, "state": "decide", "t0": time.time(), "slot": nxt["slot"],
                   "day": day.isoformat(), "atr": float(a) if np.isfinite(a) else None}
            st.session_state["eph_daily_round"] = rnd
        ch = rnd["chart"]
        st.markdown(f"**CHART {nxt['slot'] + 1} OF {daily_five.SLOTS}** · difficulty: {nxt['difficulty']}")
        ind_data = ind.compute(_DAILY_S["indicators"], ch.o, ch.h, ch.l, ch.c, ch.v, anchor=ch.vis0)
        board.render(board.payload(ch, rebase=True, hide_dates=True, hide_ticker=True, ind=ind_data))
        c1, c2 = st.columns([2, 3])
        with c1:
            conv = st.radio("Conviction", list(CONVICTION), index=list(CONVICTION).index(DEFAULT_CONVICTION),
                            horizontal=True, key="eph_d_conv", help="Stake = $1,000 × 25% / 50% / 100%.")
        with c2:
            note = st.text_input("Why is the opposite move unlikely? (optional)", key="eph_d_note",
                                 max_chars=200, placeholder="one sentence, before you see the answer")
        u, d = st.columns(2)
        with u:
            up = st.button("▲  UP / LONG", type="primary", use_container_width=True, key="eph_d_up")
        with d:
            down = st.button("▼  DOWN / SHORT", use_container_width=True, key="eph_d_down")
        if up or down:
            gid = _commit(repo, player, rnd, _DAILY_S, 1 if up else -1, conv, note, {"on": False},
                          None, None, mode="daily")
            if gid and not repo.save_daily(player["id"], day.isoformat(), nxt["slot"], gid):
                st.warning("This chart was already played (another tab?) — the first call stands.")
                st.session_state["eph_daily_round"] = None
            st.rerun(scope="fragment")

    _yesterday(repo, played, sched, day)


def _daily_done(day, rows: list[dict], streak_n: int) -> None:
    wins = sum(int(r["win"]) for r in rows)
    pnl = sum(float(r["pnl"]) for r in rows)
    st.markdown(uc.state_banner(THEME.teal if pnl >= 0 else THEME.coral, "Daily Five complete",
                                f"{wins}/5 · P&L {'+' if pnl >= 0 else '−'}${abs(pnl):,.0f} · streak {streak_n}"),
                unsafe_allow_html=True)
    st.markdown(_daily_results_html(rows), unsafe_allow_html=True)
    st.caption("Share (copy button on the right) — no answers revealed:")
    st.code(daily_five.share_text(day, rows, streak_n), language=None)
    st.dataframe(pd.DataFrame([{
        "#": int(r["slot"]) + 1, "Ticker": r["ticker"], "Name": r["name"], "Class": r["asset_class"],
        "Call": "UP" if r["direction"] > 0 else "DOWN", "Conv": r["conviction"],
        "Move": float(r["market_ret"]), "Trade": float(r["trade_ret"]), "Base rate": r["base_rate"],
        "Rule": {1: "LONG", -1: "SHORT", 0: "—"}.get(r["rule_call"], "—"), "Your note": r["note"],
    } for r in rows]).style.format({"Move": "{:+.2%}", "Trade": "{:+.2%}", "Base rate": "{:.0%}"}),
        hide_index=True, use_container_width=True)
    st.caption("Come back tomorrow for five new charts. Meanwhile, PLAY has unlimited practice.")


def _yesterday(repo, played: dict, sched: dict, day) -> None:
    yd = day - pd.Timedelta(days=1).to_pytimedelta()
    ys = daily_five.slots_for(yd, sched)
    if not ys:
        return
    mine = played.get(yd.isoformat(), {})
    with st.expander(f"YESTERDAY'S CHARTS — Daily Five #{daily_five.game_number(yd)}", expanded=False):
        recs = []
        for s in ys:
            ch = _daily_chart(s)
            mv = (ch.c[ch.t + ch.horizon] / ch.entry - 1) if ch is not None else None
            g = mine.get(s["slot"])
            recs.append({"#": s["slot"] + 1, "Ticker": s["ticker"], "Class": s["cls"], "Decision": s["decision"],
                         "Difficulty": s["difficulty"], "Move (10 bars)": mv,
                         "Your call": ("UP" if g["direction"] > 0 else "DOWN") if g else "not played",
                         "Result": ("WIN" if g["win"] else "LOSS") if g else ""})
        st.dataframe(pd.DataFrame(recs).style.format({"Move (10 bars)": "{:+.2%}"}, na_rep="—"),
                     hide_index=True, use_container_width=True)


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
    title = {"Play": "PRACTICE — call the next candles", "Daily Five": "DAILY FIVE — the same five charts "
             "for everyone today", "History": "YOUR CALLS"}[sub]
    st.markdown(section(title, 1), unsafe_allow_html=True)
    if sub == "Play":
        _play(repo, player)
    elif sub == "Daily Five":
        _daily(repo, player)
    else:
        _history(repo, player)

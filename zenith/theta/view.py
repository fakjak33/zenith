"""THETA RUN tab: lobby -> run (screener / ticket / book / charts / log) -> debrief.

Everything interactive lives in one @st.fragment so a click reruns this tab only
(a full ZENITH rerun executes every tab). The run state is a `clock.Run` object
in session_state, mirrored to the database on week/event advances and at the end.
"""

from __future__ import annotations

import html
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section
from . import DISCLAIMER, load
from .coach import Vault, adherence, on_events, pretrade
from .engine import analytics, bot, clock, scenario, scoring
from .engine.book import Book
from .engine.strategies import TEMPLATES, Leg, build, default_width
from .market import Market, round_strike

SEV_COLOR = {"alert": THEME.coral, "warn": THEME.mustard, "info": THEME.navy, "good": THEME.teal}
AUTOPLAY_DAYS = 25

_FINDINGS = [
    {"stat": "VIX beat next-month realized vol on 85% of days (1990-2026): the premium sellers harvest",
     "cite": "OPTIONS VAULT repro:vrp_spx"},
    {"stat": "50% / 21 DTE management lifts the win rate to ~87% and cuts the tail, but adds no edge",
     "cite": "OPTIONS VAULT repro:strangle_mgmt"},
    {"stat": "The bot here reproduces the vault's strangle backtest on the same tape (tested)",
     "cite": "tests/test_theta.py"},
]

_HOW = """
**The loop.** Screen the universe at the sim date, open trades from tastytrade-style templates,
then advance time: **+1 day**, **+1 week**, **next event** (21 DTE, expiry, earnings tomorrow,
50% target, 2x-credit loss, buying-power breach) or **autoplay** (streams days until something
happens). Close, roll out for a credit, roll the untested side, or sell calls against assigned stock.

**Scoring.** You're measured against **buy-and-hold SPY** and **THE BOT**: a mechanical tastytrade
trader (SPY 16-delta strangle, ~45 DTE, closed at 50% or 21 DTE, ~30% of net liq in buying power)
playing the same tape. XP rewards return, beating the bot and the tape, small drawdowns and
following the management calls. Difficulty multiplies XP.

**Prices.** Underlying prices are **real** daily closes (2005-today). Option prices are **modeled**:
a realized-vol forecast times the index volatility premium (SPY uses VIX itself), an earnings
bump that is crushed after the report, and a put skew that steepens with VIX. Fills are mid minus
a liquidity-tier half-spread (Normal/Hard). $1 per contract to open, $0 to close. Equity options
settle into stock at expiry when in the money; early assignment is not simulated.

**The coach** quotes OPTIONS VAULT: the card for your strategy, the management trees, the
allocation rule -- and the evidence check where our tests disagree with the rule of thumb.
"""


# ================================================================ data ======
@st.cache_resource(show_spinner="Loading THETA RUN market…")
def _market() -> Market | None:
    try:
        return Market.load()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _vault() -> Vault:
    return Vault()


@st.cache_data(ttl=3600, show_spinner=False)
def _challenges() -> list[dict]:
    return load("challenges", {}).get("challenges", [])


def _repo_and_store():
    try:
        from ..ephemeris.view import _repo
        repo, err = _repo()
        return repo, _store_for(repo.label, repo), err
    except Exception as exc:                       # never let saves take the game down
        return None, None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def _store_for(label: str, _repo):
    from .repo import ThetaStore
    return ThetaStore(_repo)


def today_badge() -> str | None:
    try:
        ch = load("challenges", {}).get("challenges", [])
        if not ch:
            return None
        pick = ch[date.today().toordinal() % len(ch)]
        return uc.chip(f"THETA RUN — today's challenge: {pick['title']}", color=THEME.teal,
                       sub="see THETA RUN tab")
    except Exception:
        return None


# ================================================================ helpers ===
def _ss():
    return st.session_state


def _rerun() -> None:
    """Rerun just this tab's fragment; a full rerun when not inside a fragment rerun
    (the first click after a full app run, and headless tests)."""
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


def _label(t: str) -> str:
    sc = _ss().get("th_sc") or {}
    return scenario.label(sc, t)


def _day(m: Market, i: int) -> str:
    sc = _ss().get("th_sc") or {}
    if sc.get("hide"):
        return f"DAY {i - sc['start_i']}"
    return str(m.dates[i].date())


def _money(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def _player(repo) -> dict | None:
    """Shares the EPHEMERIS handle (?player=). Kept separate from its profile bar so
    both tabs can render in one run without widget-key clashes."""
    if repo is None:
        return None
    h = _ss().get("eph_player") or st.query_params.get("player")
    p = repo.player(h) if h else None
    if p is not None:
        if p.get("pin_hash") and p["handle"] not in _ss().get("eph_pin_ok", set()):
            st.info("This handle has a PIN: unlock it once in the EPHEMERIS tab.")
            return None
        return p
    with st.form("th_handle_form", border=False):
        c1, c2 = st.columns([3, 1])
        handle = c1.text_input("Handle", placeholder="pick a handle to save runs and XP",
                               max_chars=24, label_visibility="collapsed")
        go = c2.form_submit_button("ENTER ▸", type="primary")
    if go:
        from ..ephemeris.repo import valid_handle
        err = valid_handle(handle)
        if err:
            st.error(f"Handle: {err}.")
        else:
            p = repo.get_or_create_player(handle)
            _ss()["eph_player"] = p["handle"]
            try:
                st.query_params["player"] = p["display"]
            except Exception:
                pass
            return p
    st.caption("Same handle as EPHEMERIS. Your handle lives in the page URL: bookmark it.")
    return None


def _nudge_html(n: dict) -> str:
    c = SEV_COLOR.get(n["severity"], THEME.navy)
    src = " ".join(n.get("sources", [])[:3])
    return (f'<div style="border-left:4px solid {c}; background:rgba(0,0,0,0.35); '
            f'padding:0.45rem 0.65rem; margin:0 0 0.45rem 0;">'
            f'<div style="font-family:{THEME.font_display}; letter-spacing:0.12em; color:{c}; '
            f'font-size:0.95rem; text-transform:uppercase;">{html.escape(n["title"])}</div>'
            f'<div style="font-size:0.8rem; color:{THEME.text}; line-height:1.4;">{html.escape(n["text"])}</div>'
            + (f'<div style="font-size:0.66rem; color:{THEME.muted}; margin-top:0.2rem;">'
               f'{html.escape(src)}</div>' if src else "") + '</div>')


def _bp_gauge(share: float) -> str:
    share = max(0.0, float(share))
    c = THEME.teal if share <= 0.35 else (THEME.mustard if share <= 0.5 else THEME.coral)
    w = min(share, 1.0) * 100
    return (f'<div style="font-family:{THEME.font_display}; font-size:0.72rem; letter-spacing:0.16em; '
            f'color:{THEME.muted};">BUYING POWER IN USE · guideline 25–50%</div>'
            f'<div style="position:relative; height:16px; border:1px solid {THEME.grid}; '
            f'background:repeating-linear-gradient(90deg, {THEME.panel} 0 9px, {THEME.bg} 9px 10px); '
            f'margin:0.2rem 0 0.8rem 0;">'
            f'<div style="position:absolute; left:25%; width:25%; top:0; bottom:0; '
            f'background:rgba(46,196,182,0.10); border-left:1px dashed {THEME.teal}; '
            f'border-right:1px dashed {THEME.teal};"></div>'
            f'<div style="position:absolute; left:0; top:3px; bottom:3px; width:{w:.1f}%; '
            f'background:{c}; box-shadow:0 0 8px {c};"></div>'
            f'<div style="position:absolute; right:6px; top:-2px; font-family:{THEME.font_display}; '
            f'color:{c}; font-size:1rem;">{share:.0%}</div></div>')


def _xp_bar(total: int) -> str:
    lvl, into, need = scoring.level(total)
    w = into / need * 100
    return (f'<div style="display:flex; align-items:center; gap:0.8rem; margin:0.2rem 0 0.8rem 0;">'
            f'<div style="font-family:{THEME.font_display}; font-size:2.2rem; color:{THEME.mustard};">'
            f'LV {lvl}</div><div style="flex:1;"><div style="font-size:0.7rem; color:{THEME.muted}; '
            f'letter-spacing:0.14em;">XP {total:,} · {into:,}/{need:,} TO NEXT</div>'
            f'<div style="height:10px; border:1px solid {THEME.mustard}; margin-top:0.2rem;">'
            f'<div style="height:100%; width:{w:.1f}%; background:{THEME.mustard}; '
            f'box-shadow:0 0 10px {THEME.mustard};"></div></div></div></div>')


def _badges(have: set[str]) -> str:
    cells = []
    for k, (name, desc) in scoring.ACHIEVEMENTS.items():
        on = k in have
        c = THEME.teal if on else THEME.grid
        cells.append(f'<div title="{html.escape(desc)}" style="border:1px solid {c}; '
                     f'padding:0.3rem 0.55rem; color:{THEME.text if on else THEME.muted}; '
                     f'font-family:{THEME.font_display}; letter-spacing:0.1em; font-size:0.85rem; '
                     f'{"box-shadow:0 0 8px " + c + ";" if on else "opacity:0.55;"}">'
                     f'{"◆" if on else "◇"} {html.escape(name)}</div>')
    return '<div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.2rem 0 1rem 0;">' + \
        "".join(cells) + "</div>"


# ================================================================ render ====
def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating("B", "real prices, modeled options; mechanics follow OPTIONS VAULT",
                                "Underlying paths are real; IV, skew and earnings moves are a "
                                "calibrated model (SPY = VIX; stocks ~1.16x later realized vol)."),
                unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    with st.expander("How to play — loop, scoring, prices, coach"):
        st.markdown(_HOW)
    m = _market()
    if m is None:
        st.info("THETA RUN market data isn't published yet. Locally: `python -m zenith.theta.build`.")
        return
    repo, store, err = _repo_and_store()
    if err:
        st.warning(f"Saves unavailable ({err}); runs still work in this session.")
    player = _player(repo)
    _body(m, store, player)


@st.fragment
def _body(m: Market, store, player: dict | None) -> None:
    if _ss().get("th_debrief"):
        _debrief(m, store, player)
    elif _ss().get("th_run") is not None:
        _game(m, store, player)
    else:
        _lobby(m, store, player)


# ================================================================ lobby =====
def _start(m: Market, store, player, sc: dict, mode: str, run_dict: dict | None = None,
           run_id: int | None = None) -> None:
    diff = scenario.DIFFICULTY[sc["difficulty"]]
    if run_dict:
        run = clock.Run.from_dict(run_dict)
    else:
        book = Book(sc["cash"], sc["cash"], slip=diff["slip"])
        run = clock.Run(book, sc["start_i"], sc["end_i"], sc["start_i"])
        clock.snapshot(m, run)
    b = bot.play(m, sc["start_i"], sc["end_i"], sc["cash"], slip=diff["slip"])
    spy = m.data["SPY"]["close"][sc["start_i"]: sc["end_i"] + 1]
    _ss().update({"th_sc": sc, "th_run": run, "th_mode": mode, "th_bot": b["equity"],
                  "th_spy": list(spy / spy[0] * sc["cash"]), "th_nudges": [],
                  "th_ticket": None, "th_debrief": None})
    if store is not None and player is not None and run_id is None:
        try:
            run_id = store.create_run(player["id"], mode, sc, run.to_dict())
        except Exception:
            run_id = None
    _ss()["th_run_id"] = run_id


def _lobby(m: Market, store, player) -> None:
    st.markdown(section("THETA RUN — manage an options book against the tape and the bot", 0),
                unsafe_allow_html=True)
    total, have, active = 0, set(), None
    if store is not None and player is not None:
        try:
            total = store.total_xp(player["id"])
            have = store.achievements(player["id"])
            active = store.active_run(player["id"])
        except Exception:
            pass
    if player is not None:
        st.markdown(f'<div style="font-family:{THEME.font_display}; font-size:1.1rem; '
                    f'letter-spacing:0.14em; color:{THEME.teal};">OPERATOR · '
                    f'{html.escape(player["display"])}</div>', unsafe_allow_html=True)
        st.markdown(_xp_bar(total), unsafe_allow_html=True)
        st.markdown(_badges(have), unsafe_allow_html=True)
    if active:
        sc = active["scenario"]
        c1, c2 = st.columns([3, 1])
        c1.markdown(uc.chip(f"Saved run · {active['mode']} · {sc['difficulty']}",
                            THEME.mustard, sub=f"day {active['state']['i'] - sc['start_i']} of "
                                               f"{sc['end_i'] - sc['start_i']}"), unsafe_allow_html=True)
        if c2.button("CONTINUE ▸", key="th_continue", type="primary"):
            _start(m, store, player, sc, active["mode"], active["state"], active["id"])
            _rerun()

    q, d, c = st.columns(3)
    with q:
        st.markdown(section("QUICK RUN", 1), unsafe_allow_html=True)
        st.caption("A random window from 2006 on. Normal/Hard hide tickers and dates.")
        length = st.selectbox("Length", list(scenario.LENGTHS), index=1, key="th_len")
        diff = st.radio("Difficulty", list(scenario.DIFFICULTY), index=1, horizontal=True, key="th_diff")
        if st.button("START RUN ▸", key="th_go_quick", type="primary", use_container_width=True):
            seed = int(time.time() * 1000) % 2_147_483_647
            _start(m, store, player, scenario.random_scenario(m, seed, scenario.LENGTHS[length], diff),
                   "quick")
            _rerun()
    with d:
        st.markdown(section("DAILY RUN", 2), unsafe_allow_html=True)
        today = date.today().isoformat()
        st.caption(f"Same blind half-year for everyone on {today}. Normal difficulty.")
        if st.button("PLAY TODAY'S RUN ▸", key="th_go_daily", use_container_width=True):
            sc = scenario.random_scenario(m, scenario.daily_seed(today), 126, "normal")
            _start(m, store, player, sc, f"daily:{today}")
            _rerun()
    with c:
        st.markdown(section("CHALLENGES", 3), unsafe_allow_html=True)
        chs = _challenges()
        names = {f"{ch['title']}{' ☠' if ch.get('crisis') else ''}": ch for ch in chs}
        pick = st.selectbox("Challenge", list(names), key="th_ch") if names else None
        if pick:
            ch = names[pick]
            st.caption(ch["brief"])
            cd = st.radio("Difficulty", list(scenario.DIFFICULTY), index=1, horizontal=True,
                          key="th_ch_diff")
            if st.button("ENTER CHALLENGE ▸", key="th_go_ch", use_container_width=True):
                _start(m, store, player, scenario.challenge_scenario(m, ch, cd), f"challenge:{ch['id']}")
                _rerun()

    if store is not None:
        try:
            lb = store.leaderboard(limit=10)
        except Exception:
            lb = []
        if lb:
            st.markdown(section("LEADERBOARD — top runs by XP", 4), unsafe_allow_html=True)
            df = pd.DataFrame(lb)
            df["ret"] = df["ret"].map(lambda v: uc.fmt_pct(v))
            df["max_dd"] = df["max_dd"].map(lambda v: uc.fmt_pct(v, signed=False))
            st.dataframe(df.rename(columns={"display": "operator", "ret": "return", "max_dd": "max DD",
                                            "updated_at": "when"}),
                         hide_index=True, use_container_width=True)
    if player is None:
        st.caption("Playing as a guest: runs work, but XP and saves need a handle.")


# ================================================================ game ======
def _save(store, run: clock.Run) -> None:
    rid = _ss().get("th_run_id")
    if store is not None and rid:
        try:
            store.save_state(rid, run.to_dict())
        except Exception:
            pass


def _after_advance(m: Market, store, run: clock.Run, events: list, save: bool) -> None:
    diff = scenario.DIFFICULTY[_ss()["th_sc"]["difficulty"]]
    if diff["coach"]:
        _ss()["th_nudges"] = [n.to_dict() for n in on_events(_vault(), [e.to_dict() for e in events])]
    if save:
        _save(store, run)
    if run.done:
        _finish(m, store)


def _hud(m: Market, run: clock.Run) -> str:
    b, i = run.book, run.i
    sc = _ss()["th_sc"]
    nlv = b.nlv(m, i)
    ret = nlv / b.start_cash - 1.0
    k = i - sc["start_i"]
    bot_eq = _ss()["th_bot"]
    bot_nlv = bot_eq[min(k, len(bot_eq) - 1)][1] if bot_eq else b.start_cash
    spy_nlv = _ss()["th_spy"][min(k, len(_ss()["th_spy"]) - 1)]
    g = b.portfolio_greeks(m, i)
    up = THEME.teal if ret >= 0 else THEME.coral
    return uc.numeric_slab([
        {"label": "net liq", "value": uc.fmt_money(nlv, 1), "sub": f"{_money(nlv)} · {uc.fmt_pct(ret)}",
         "color": up},
        {"label": "vs the bot", "value": uc.fmt_pct(nlv / bot_nlv - 1.0),
         "sub": f"bot {_money(bot_nlv)}", "color": THEME.teal if nlv >= bot_nlv else THEME.coral},
        {"label": "vs SPY", "value": uc.fmt_pct(nlv / spy_nlv - 1.0), "sub": f"tape {_money(spy_nlv)}",
         "color": THEME.teal if nlv >= spy_nlv else THEME.coral},
        {"label": "theta / day", "value": _money(g["theta"]),
         "sub": f"{g['theta'] / max(nlv, 1):.2%} of NLV", "color": THEME.mustard},
        {"label": "β-wtd delta", "value": f"{g['bw_delta']:+,.0f}", "sub": "SPY shares",
         "color": THEME.navy},
        {"label": "vega", "value": _money(g["vega"]), "sub": "per vol pt", "color": THEME.mauve},
    ], min_width=130)


def _game(m: Market, store, player) -> None:
    run: clock.Run = _ss()["th_run"]
    sc = _ss()["th_sc"]
    i = run.i
    total = sc["end_i"] - sc["start_i"]
    ch = next((c for c in _challenges() if c["id"] == sc.get("challenge")), None)
    title = ch["title"] if ch else ("DAILY RUN" if _ss().get("th_mode", "").startswith("daily") else "QUICK RUN")
    vix = m.vix[i]
    st.markdown(uc.state_banner(THEME.teal if vix < 0.2 else (THEME.mustard if vix < 0.3 else THEME.coral),
                                f"{title} · {_day(m, i)}",
                                f"day {i - sc['start_i']} of {total} · VIX {vix * 100:.1f} · "
                                f"{sc['difficulty']}"), unsafe_allow_html=True)
    hud = st.empty()
    hud.markdown(_hud(m, run), unsafe_allow_html=True)
    nlv = run.book.nlv(m, i)
    gauge = st.empty()
    gauge.markdown(_bp_gauge(run.book.bp_used(m, i) / max(nlv, 1)), unsafe_allow_html=True)

    c = st.columns([1, 1, 1.3, 1.3, 1])
    if c[0].button("+1 DAY", key="th_day", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "day"), save=False)
        _rerun()
    if c[1].button("+1 WEEK", key="th_week", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "week"), save=True)
        _rerun()
    if c[2].button("▶▶ NEXT EVENT", key="th_event", type="primary", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "event"), save=True)
        _rerun()
    if c[3].button(f"⏩ AUTOPLAY {AUTOPLAY_DAYS}D", key="th_auto", use_container_width=True):
        events = []
        for _ in range(AUTOPLAY_DAYS):
            ev = clock.step(m, run)
            events += ev
            hud.markdown(_hud(m, run), unsafe_allow_html=True)
            gauge.markdown(_bp_gauge(run.book.bp_used(m, run.i) / max(run.book.nlv(m, run.i), 1)),
                           unsafe_allow_html=True)
            if ev or run.done:
                break
            time.sleep(0.08)
        _after_advance(m, store, run, events, save=True)
        _rerun()
    if c[4].button("END RUN", key="th_end", use_container_width=True):
        _finish(m, store)
        _rerun()

    main, side = st.columns([3, 1.15])
    with main:
        tabs = st.tabs(["SCREENER", "TICKET", f"BOOK ({len(run.book.open_positions())})", "CHARTS", "LOG"])
        with tabs[0]:
            _screener(m, run)
        with tabs[1]:
            _ticket(m, store, run)
        with tabs[2]:
            _book(m, store, run)
        with tabs[3]:
            _charts(m, run)
        with tabs[4]:
            _log(m, run)
    with side:                      # filled last, so it shows this run's ticket nudges
        st.markdown(section("COACH", 3), unsafe_allow_html=True)
        if not scenario.DIFFICULTY[sc["difficulty"]]["coach"]:
            st.caption("Hard mode: the coach is silent until the debrief.")
        else:
            ns = _ss().get("th_nudges") or []
            pre = _ss().get("th_pretrade") or []
            if ns:
                st.caption("WHAT JUST HAPPENED")
                for n in ns[:4]:
                    st.markdown(_nudge_html(n), unsafe_allow_html=True)
            if pre:
                st.caption("ON YOUR TICKET")
                for n in pre[:4]:
                    st.markdown(_nudge_html(n), unsafe_allow_html=True)
            if not ns and not pre:
                st.caption("Nudges appear when something happens, and on the trade ticket.")


def _screener(m: Market, run: clock.Run) -> None:
    rows = m.snapshot(run.i)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No tradeable names on this date.")
        return
    f1, f2, f3, f4 = st.columns(4)
    kinds = f1.multiselect("Type", ["index", "etf", "stock"], default=["index", "etf", "stock"],
                           key="th_f_kind")
    ivr_min = f2.slider("Min IV rank", 0, 100, 0, 5, key="th_f_ivr")
    earn = f3.selectbox("Earnings", ["any", "report within 30d", "no report within 30d"], key="th_f_earn")
    sort = f4.selectbox("Sort by", ["IV rank", "IV", "IV / RV", "expected move %"], key="th_f_sort")
    df = df[df["kind"].isin(kinds) & (df["ivr"].fillna(0) * 100 >= ivr_min)]
    if earn == "report within 30d":
        df = df[(df["earn_days"] >= 0) & (df["earn_days"] <= 30)]
    elif earn == "no report within 30d":
        df = df[(df["earn_days"] < 0) | (df["earn_days"] > 30)]
    df = df.assign(em_pct=df["em_30d"] / df["price"])
    key = {"IV rank": "ivr", "IV": "iv", "IV / RV": "iv_rv", "expected move %": "em_pct"}[sort]
    df = df.sort_values(key, ascending=False)
    show = pd.DataFrame({
        "symbol": df["ticker"].map(_label), "type": df["kind"], "liq": df["tier"],
        "price": df["price"].round(2), "IV": (df["iv"] * 100).round(1),
        "IV rank": (df["ivr"] * 100).round(0), "IV pctl": (df["ivp"] * 100).round(0),
        "IV/RV": df["iv_rv"].round(2), "30d move ±$": df["em_30d"].round(2),
        "report in": df["earn_days"].map(lambda d: f"{int(d)}d" if d >= 0 else "—"),
        "beta": df["beta"].round(2)})
    st.caption(f"{len(show)} names · IV rank over the trailing year · modeled IV · liq A/B/C = spread tier")
    st.dataframe(show, hide_index=True, use_container_width=True, height=420, column_config={
        "IV rank": st.column_config.ProgressColumn("IV rank", min_value=0, max_value=100, format="%d"),
        "IV pctl": st.column_config.ProgressColumn("IV pctl", min_value=0, max_value=100, format="%d")})
    order = list(df["ticker"]) + [t for t in m.tickers if t not in set(df["ticker"])]
    pick = st.selectbox("Open a ticket for", m.tickers, index=m.tickers.index(order[0]),
                        format_func=_label, key="th_pick")
    if st.button("TO TICKET ▸", key="th_to_ticket"):
        _ss()["th_ticket"] = pick
        st.toast(f"{_label(pick)} loaded in the TICKET tab")


def _payoff_chart(a: dict, S0: float, cost_note: str):
    df = pd.DataFrame({"price": a["grid"], "expiry": a["expiry"], "today": a["now"],
                       f"T+{a['half_days']}": a["half"]})
    long = df.melt("price", var_name="when", value_name="pnl")

    def build_(alt):
        base = alt.Chart(df)
        band = alt.Chart(pd.DataFrame({"lo": [S0 - a["em"]], "hi": [S0 + a["em"]]})).mark_rect(
            opacity=0.10, color=THEME.navy).encode(x="lo:Q", x2="hi:Q")
        pos = base.mark_area(opacity=0.35, color=THEME.teal).encode(
            x=alt.X("price:Q", title="underlying at expiry", scale=alt.Scale(zero=False)),
            y=alt.Y("pos:Q", title="P&L $"), ).transform_calculate(pos="max(datum.expiry, 0)")
        neg = base.mark_area(opacity=0.35, color=THEME.coral).encode(
            x="price:Q", y="neg:Q").transform_calculate(neg="min(datum.expiry, 0)")
        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x="price:Q", y="pnl:Q",
            color=alt.Color("when:N", scale=alt.Scale(domain=["expiry", "today", f"T+{a['half_days']}"],
                                                       range=[THEME.text, THEME.mustard, THEME.mauve]),
                            legend=alt.Legend(orient="top", title=None, labelLimit=0)),
            tooltip=[alt.Tooltip("price:Q", format=".2f"), "when:N", alt.Tooltip("pnl:Q", format=",.0f")])
        spot = alt.Chart(pd.DataFrame({"x": [S0]})).mark_rule(color=THEME.teal, strokeDash=[4, 3]).encode(x="x:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.grid).encode(y="y:Q")
        return (band + pos + neg + zero + lines + spot).properties(height=300)
    st.caption(cost_note)
    uc.render_chart(build_, fallback=df)


def _ticket(m: Market, store, run: clock.Run) -> None:
    i = run.i
    default = _ss().get("th_ticket") or "SPY"
    c1, c2 = st.columns([1, 1.4])
    t = c1.selectbox("Underlying", m.tickers,
                     index=m.tickers.index(default) if default in m.tickers else 0,
                     format_func=_label, key="th_t")
    if not m.live(t, i):
        st.info(f"{_label(t)} isn't trading on this date.")
        return
    tpl = c2.selectbox("Strategy", list(TEMPLATES), format_func=lambda k: f"{TEMPLATES[k]['name']} · "
                       f"{TEMPLATES[k]['risk']} · {TEMPLATES[k]['outlook']}", key="th_tpl")
    spec = TEMPLATES[tpl]
    exps = m.expiry_choices(t, i, 120)
    if not exps:
        st.info("No listed expirations in range.")
        return
    dflt = min(range(len(exps)), key=lambda k: abs(m.days_between(i, exps[k]) - 45))
    c3, c4, c5, c6 = st.columns(4)
    e = c3.selectbox("Expiration", exps, index=dflt, key="th_exp",
                     format_func=lambda x: f"{m.days_between(i, x)} DTE"
                     + ("" if (_ss().get("th_sc") or {}).get("hide") else f" · {m.dates[x]:%b %d}"))
    dlt = c4.slider("Short delta", 0.05, 0.50, float(spec["delta"]), 0.01, key=f"th_dlt_{tpl}",
                    disabled=spec["delta"] >= 0.5)
    S = m.spot(t, i)
    w = c5.number_input("Wing width $", min_value=0.5, value=float(default_width(S)), step=0.5,
                        key=f"th_w_{tpl}_{t}", disabled=not spec["wing"])
    n = int(c6.number_input("Contracts", 1, 100, 1, key="th_n"))
    flat = run.book.flat_iv
    legs = build(tpl, m, t, i, e, n, dlt, w if spec["wing"] else None, flat)
    pv = run.book.preview(m, t, i, legs)
    a = analytics.analyse(m, t, i, legs, -pv["credit"], flat)
    nlv = run.book.nlv(m, i)
    st.markdown(uc.numeric_slab([
        {"label": "credit" if pv["credit"] >= 0 else "debit", "value": _money(abs(pv["credit"])),
         "color": THEME.teal if pv["credit"] >= 0 else THEME.coral, "sub": f"fees {_money(pv['fees'])}"},
        {"label": "buying power", "value": _money(pv["bpr"]), "sub": f"{pv['bpr'] / max(nlv, 1):.1%} of NLV",
         "color": THEME.mustard},
        {"label": "POP (model)", "value": f"{a['pop']:.0%}", "sub": f"ATM IV {a['atm_iv']:.0%}",
         "color": THEME.navy},
        {"label": "max profit", "value": _money(a["max_profit"]), "color": THEME.teal,
         "sub": f"return on BP {a['max_profit'] / max(pv['bpr'], 1):.0%}"},
        {"label": "max loss", "value": "∞" if a["unlimited_up"] else _money(a["max_loss"]),
         "color": THEME.coral, "sub": "undefined risk (naked call)" if a["unlimited_up"]
         else f"at expiry · worst case {_money(a['max_loss'])}"},
    ], min_width=120), unsafe_allow_html=True)
    be = " / ".join(f"{x:,.2f}" for x in a["breakevens"]) or "—"
    _payoff_chart(a, S, f"{_label(t)} {S:,.2f} · breakevens {be} · shaded = ±1σ by expiry")
    legs_df = pd.DataFrame([{"leg": f"{'+' if lg.qty > 0 else ''}{lg.qty:g} {lg.kind}",
                             "strike": lg.strike if lg.kind != "stock" else "",
                             "fill": round(lg.open_px, 2)} for lg in pv["legs"]])
    st.dataframe(legs_df, hide_index=True, use_container_width=True)

    sc = _ss()["th_sc"]
    if scenario.DIFFICULTY[sc["difficulty"]]["coach"]:
        row = next((r for r in m.snapshot(i) if r["ticker"] == t), {"ticker": t})
        ns = pretrade(_vault(), row, tpl, pv, m.days_between(i, e), nlv, run.book.bp_used(m, i),
                      [p.ticker for p in run.book.open_positions()], m.earnings_between(t, i, e))
        _ss()["th_pretrade"] = [x.to_dict() for x in ns][:4]
    if st.button("SEND ORDER ▸", key="th_send", type="primary"):
        run.book.open(m, t, i, tpl, legs, dlt, w if spec["wing"] else 0.0)
        clock.snapshot(m, run)
        _ss()["th_pretrade"] = []
        _save(store, run)
        _rerun()


def _book(m: Market, store, run: clock.Run) -> None:
    i = run.i
    b = run.book
    opens = b.open_positions()
    if not opens:
        st.caption("No open positions. Screen, then open a ticket.")
    for p in opens:
        pnl = b.position_pnl(m, p, i)
        dte = m.days_between(i, p.expiry_i()) if p.option_legs() else None
        g = b.position_greeks(m, p, i)
        pct = pnl / p.credit if p.credit > 0 else None
        c = THEME.teal if pnl >= 0 else THEME.coral
        legs = ", ".join(f"{lg.qty:+g} {lg.kind}{'' if lg.kind == 'stock' else ' ' + format(lg.strike, 'g')}"
                         for lg in p.legs)
        st.markdown(
            f'<div style="border:1px solid {THEME.grid}; border-left:5px solid {c}; padding:0.5rem 0.8rem; '
            f'margin:0.3rem 0; background:rgba(0,0,0,0.35);">'
            f'<div style="font-family:{THEME.font_display}; font-size:1.2rem; letter-spacing:0.1em;">'
            f'#{p.id} {html.escape(_label(p.ticker))} · {html.escape(TEMPLATES.get(p.template, {}).get("name", p.template))}'
            f'<span style="float:right; color:{c};">{_money(pnl)}'
            f'{"" if pct is None else f" · {pct:+.0%} of credit"}</span></div>'
            f'<div style="font-size:0.78rem; color:{THEME.muted};">{html.escape(legs)} · '
            f'{"" if dte is None else f"{dte} DTE · "}Δ {g["delta"]:+.0f} · Θ {_money(g["theta"])}/day · '
            f'BP {_money(b.position_bpr(m, p, i))}{" · rolled " + str(p.rolls) + "x" if p.rolls else ""}</div></div>',
            unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        if a1.button("CLOSE", key=f"th_close_{p.id}", use_container_width=True):
            b.close(m, p, i)
            clock.snapshot(m, run)
            _save(store, run)
            _rerun()
        if p.option_legs():
            later = [e for e in m.expiry_choices(p.ticker, i, 120) if e > p.expiry_i()]
            if later:
                tgt = min(later, key=lambda e: abs(m.days_between(i, e) - 45))
                if a2.button(f"ROLL OUT · {m.days_between(i, tgt)}D", key=f"th_roll_{p.id}",
                             use_container_width=True):
                    if b.roll(m, p, i, tgt, "out") is None:
                        st.toast("No roll for a credit available here (tastylive: don't roll for a debit).")
                    else:
                        clock.snapshot(m, run)
                        _save(store, run)
                        _rerun()
            shorts = {lg.kind for lg in p.option_legs() if lg.qty < 0}
            if len(shorts) == 2 and a3.button("ROLL UNTESTED", key=f"th_unt_{p.id}",
                                              use_container_width=True):
                if b.roll(m, p, i, p.expiry_i(), "untested") is None:
                    st.toast("Rolling the untested side wouldn't collect a credit right now.")
                else:
                    clock.snapshot(m, run)
                    _save(store, run)
                    _rerun()
        stock = [lg for lg in p.legs if lg.kind == "stock" and lg.qty > 0]
        has_call = any(lg.kind == "call" and lg.qty < 0 for lg in p.legs)
        if stock and not has_call:
            ex = m.nearest_expiry(p.ticker, i, 35)
            if ex and a4.button("SELL CALL (wheel)", key=f"th_cc_{p.id}", use_container_width=True):
                k = m.strike_for_delta(p.ticker, i, ex, 0.30, True, b.flat_iv)
                b.add_legs(m, p, i, [Leg("call", -stock[0].qty / 100.0, k, ex)], "Sold covered call")
                p.flags.append("wheel")
                clock.snapshot(m, run)
                _save(store, run)
                _rerun()
    closed = [p for p in b.positions if not p.is_open]
    if closed:
        st.markdown(section(f"CLOSED ({len(closed)})", 5), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"#": p.id, "symbol": _label(p.ticker),
                                    "strategy": TEMPLATES.get(p.template, {}).get("name", p.template),
                                    "opened": _day(m, p.opened_i), "closed": _day(m, p.closed_i),
                                    "status": p.status, "credit": round(p.credit),
                                    "P&L": round(p.pnl), "rolls": p.rolls} for p in closed]),
                     hide_index=True, use_container_width=True)


def _charts(m: Market, run: clock.Run) -> None:
    sc = _ss()["th_sc"]
    eq = pd.DataFrame([e[:3] for e in run.equity], columns=["i", "nlv", "bp"])
    k = eq["i"] - sc["start_i"]
    bot_eq = pd.DataFrame([e[:3] for e in _ss()["th_bot"]], columns=["i", "nlv", "bp"])
    bot_eq = bot_eq[bot_eq["i"] <= run.i]
    spy = _ss()["th_spy"][: run.i - sc["start_i"] + 1]
    cash = run.book.start_cash
    df = pd.concat([
        pd.DataFrame({"day": k, "who": "YOU", "ret": eq["nlv"] / cash - 1}),
        pd.DataFrame({"day": bot_eq["i"] - sc["start_i"], "who": "THE BOT", "ret": bot_eq["nlv"] / cash - 1}),
        pd.DataFrame({"day": range(len(spy)), "who": "SPY (buy & hold)", "ret": np.array(spy) / cash - 1}),
    ])

    def eq_chart(alt):
        return alt.Chart(df).mark_line(strokeWidth=2.2, interpolate="monotone").encode(
            x=alt.X("day:Q", title="trading day"),
            y=alt.Y("ret:Q", title="return", axis=alt.Axis(format="+%")),
            color=alt.Color("who:N", scale=alt.Scale(domain=["YOU", "THE BOT", "SPY (buy & hold)"],
                                                     range=[THEME.teal, THEME.mustard, THEME.muted]),
                            legend=alt.Legend(orient="top", title=None, labelLimit=0)),
            tooltip=["who:N", "day:Q", alt.Tooltip("ret:Q", format="+.2%")],
        ).properties(height=260, title="RACE — you vs the bot vs the tape")
    uc.render_chart(eq_chart, fallback=df)

    bp = pd.DataFrame({"day": k, "bp": eq["bp"] / eq["nlv"].clip(lower=1)})

    def bp_chart(alt):
        area = alt.Chart(bp).mark_area(color=THEME.mustard, opacity=0.45, interpolate="step-after").encode(
            x=alt.X("day:Q", title=None), y=alt.Y("bp:Q", title="BP / NLV", axis=alt.Axis(format="%")))
        guide = alt.Chart(pd.DataFrame({"y": [0.25, 0.5]})).mark_rule(color=THEME.teal, strokeDash=[4, 4]).encode(y="y:Q")
        return (area + guide).properties(height=120)
    uc.render_chart(bp_chart, fallback=bp)

    tick = sorted({p.ticker for p in run.book.positions} | {"SPY"})
    t = st.selectbox("Underlying", tick, format_func=_label, key="th_chart_t")
    lo = sc["start_i"] - 60
    seg = slice(max(lo, 0), run.i + 1)
    d = m.data[t]
    px = pd.DataFrame({"day": np.arange(seg.start, seg.stop) - sc["start_i"], "close": d["close"][seg],
                       "IV": d["iv30"][seg], "RV": d["rv21"][seg]})
    strikes = [lg.strike for p in run.book.open_positions() if p.ticker == t for lg in p.option_legs()]

    def px_chart(alt):
        line = alt.Chart(px).mark_line(color=THEME.text, strokeWidth=1.6).encode(
            x=alt.X("day:Q", title="trading day (0 = start)"),
            y=alt.Y("close:Q", title=_label(t), scale=alt.Scale(zero=False)))
        start = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=THEME.navy, strokeDash=[2, 2]).encode(x="x:Q")
        layers = line + start
        if strikes:
            sk = pd.DataFrame({"K": strikes})
            layers += alt.Chart(sk).mark_rule(color=THEME.coral, strokeDash=[5, 3]).encode(y="K:Q")
        return layers.properties(height=240)
    uc.render_chart(px_chart, fallback=px)
    vol = px.melt("day", ["IV", "RV"], "series", "vol")

    def vol_chart(alt):
        return alt.Chart(vol).mark_line(strokeWidth=1.6).encode(
            x=alt.X("day:Q", title=None), y=alt.Y("vol:Q", title="vol", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", scale=alt.Scale(domain=["IV", "RV"], range=[THEME.mauve, THEME.mint]),
                            legend=alt.Legend(orient="top", title=None))).properties(height=140)
    uc.render_chart(vol_chart, fallback=vol)


def _log(m: Market, run: clock.Run) -> None:
    rows = list(reversed(run.book.log))[:120]
    if not rows:
        st.caption("Nothing yet.")
        return
    color = {"open": THEME.teal, "close": THEME.text, "roll": THEME.mauve, "expire": THEME.muted,
             "assign": THEME.coral, "target": THEME.teal, "stop": THEME.coral, "dte21": THEME.mustard,
             "earnings": THEME.orange, "bp": THEME.coral, "adjust": THEME.mauve, "end": THEME.navy}
    out = []
    for r in rows:
        c = color.get(r["kind"], THEME.muted)
        out.append(f'<div style="font-size:0.8rem; border-left:3px solid {c}; padding:0.1rem 0.5rem; '
                   f'margin-bottom:0.2rem;"><span style="color:{c}; font-family:{THEME.font_display}; '
                   f'letter-spacing:0.1em;">{html.escape(_day(m, r["i"]))} · {r["kind"].upper()}</span> '
                   f'<span style="color:{THEME.text};">{html.escape(_alias_text(r["msg"]))}</span></div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def _alias_text(msg: str) -> str:
    sc = _ss().get("th_sc") or {}
    if not sc.get("hide"):
        return msg
    for t, a in sc.get("aliases", {}).items():
        msg = msg.replace(f" {t} ", f" {a} ").replace(f" {t}:", f" {a}:").replace(f" {t} (", f" {a} (")
    return msg


# ================================================================ debrief ===
def _finish(m: Market, store) -> None:
    run: clock.Run = _ss()["th_run"]
    sc = _ss()["th_sc"]
    clock.snapshot(m, run)
    me = scoring.curve_stats(run.equity)
    bot_eq = [e for e in _ss()["th_bot"] if e[0] <= run.i]
    bo = scoring.curve_stats(bot_eq)
    spy = _ss()["th_spy"][: run.i - sc["start_i"] + 1]
    spy_ret = spy[-1] / spy[0] - 1.0 if len(spy) > 1 else 0.0
    adh, rows = adherence(run.book.log)
    eq = pd.DataFrame([e[:3] for e in run.equity], columns=["i", "nlv", "bp"])
    max_bp = float((eq["bp"] / eq["nlv"].clip(lower=1)).max()) if len(eq) else 0.0
    ch = next((c for c in _challenges() if c["id"] == sc.get("challenge")), None)
    thetas = [e[3] / e[1] for e in run.equity if len(e) > 3 and e[1] > 0 and e[3] > 0]
    flags = {"trades": len(run.book.positions),
             "closed": sum(1 for p in run.book.positions if not p.is_open),
             "max_bp": max_bp, "wheel": any("wheel" in p.flags for p in run.book.positions),
             "theta_nlv": float(np.mean(thetas)) if thetas else 0.0}
    sc_ = scoring.score(me, bo, spy_ret, adh, sc["difficulty"], bool(ch and ch.get("crisis")), flags)
    deb = {"me": me, "bot": bo, "spy_ret": spy_ret, "adherence": adh, "adh_rows": rows,
           "score": sc_, "challenge": ch, "max_bp": max_bp}
    _ss()["th_debrief"] = deb
    rid = _ss().get("th_run_id")
    if store is not None and rid:
        try:
            store.finish(rid, run.to_dict(), me["ret"], me["max_dd"], sc_["xp"], sc_["achievements"])
        except Exception:
            pass


def _debrief(m: Market, store, player) -> None:
    deb = _ss()["th_debrief"]
    run: clock.Run = _ss()["th_run"]
    sc = _ss()["th_sc"]
    me, bo, s = deb["me"], deb["bot"], deb["score"]
    won = me["ret"] > bo["ret"]
    st.markdown(uc.state_banner(THEME.teal if won else THEME.coral, "RUN COMPLETE",
                                f"+{s['xp']:,} XP · {'you beat the bot' if won else 'the bot wins this one'}"),
                unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "your return", "value": uc.fmt_pct(me["ret"]), "color": THEME.teal if me["ret"] >= 0 else THEME.coral,
         "sub": f"max DD {me['max_dd']:.1%} · Sharpe {me['sharpe']:.2f}"},
        {"label": "the bot", "value": uc.fmt_pct(bo["ret"]), "color": THEME.mustard,
         "sub": f"max DD {bo['max_dd']:.1%}"},
        {"label": "SPY buy & hold", "value": uc.fmt_pct(deb["spy_ret"]), "color": THEME.muted},
        {"label": "management calls followed", "value": "—" if deb["adherence"] is None else f"{deb['adherence']:.0%}",
         "color": THEME.navy, "sub": f"{len(deb['adh_rows'])} calls (21 DTE / 50%)"},
        {"label": "XP", "value": f"{s['xp']:,}", "color": THEME.mustard, "sub": f"x{s['mult']:.1f} difficulty"},
    ]), unsafe_allow_html=True)
    if s["achievements"]:
        st.markdown(section("UNLOCKED", 1), unsafe_allow_html=True)
        st.markdown(_badges(set(s["achievements"])), unsafe_allow_html=True)
    if deb.get("challenge"):
        st.markdown(uc.note_strip("THE LESSON", [deb["challenge"]["lesson"]]), unsafe_allow_html=True)
    _charts(m, run)
    if sc.get("hide"):
        traded = sorted({p.ticker for p in run.book.positions})
        st.markdown(section("REVEAL", 2), unsafe_allow_html=True)
        st.markdown(" ".join(uc.chip(f"{sc['aliases'].get(t, t)} = {t}", THEME.mauve) for t in traded)
                    + uc.chip(f"window {m.dates[sc['start_i']].date()} → {m.dates[run.i].date()}", THEME.navy),
                    unsafe_allow_html=True)
    nudges = on_events(_vault(), [{"kind": r["kind"], "msg": f"{'Followed' if r['followed'] else 'Missed'} "
                                   f"the {r['kind']} call on #{r['pid']}", "pid": r["pid"], "i": r["i"]}
                                  for r in deb["adh_rows"][:4]])
    for n in nudges[:4]:
        st.markdown(_nudge_html(n.to_dict()), unsafe_allow_html=True)
    if st.button("BACK TO LOBBY ▸", key="th_lobby", type="primary"):
        for k in ("th_run", "th_sc", "th_debrief", "th_bot", "th_spy", "th_run_id", "th_nudges", "th_pretrade"):
            _ss().pop(k, None)
        _rerun()

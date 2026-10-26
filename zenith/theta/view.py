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

from .. import art
from .. import ui_charts as uc
from .. import ui_kit as uk
from ..config import THEME
from ..ui_theme import evidence_rating, key_findings, section
from . import DISCLAIMER, load
from .coach import Vault, adherence, on_events, pretrade
from .engine import analytics, bot, clock, scenario, scoring
from .engine.book import Book
from .engine.strategies import TEMPLATES, Leg, build, default_width
from .market import Market, round_strike

SEV_COLOR = {"alert": THEME.coral, "warn": THEME.amber, "info": THEME.cyan, "good": THEME.teal}
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
    c = SEV_COLOR.get(n["severity"], THEME.cyan)
    src = " ".join(n.get("sources", [])[:2])
    return (f'<div style="border-left:2px solid {c}; padding:0.3rem 0 0.3rem 0.65rem; margin:0 0 0.5rem 0;">'
            f'<div style="letter-spacing:0.16em; color:{c}; font-size:0.62rem; text-transform:uppercase;">'
            f'{html.escape(n["title"])}</div>'
            f'<div style="font-size:0.74rem; color:{THEME.text}; line-height:1.45;">{html.escape(n["text"])}</div>'
            + (f'<div style="font-size:0.6rem; color:{THEME.faint}; margin-top:0.15rem;">'
               f'{html.escape(src)}</div>' if src else "") + '</div>')


def _xp_bar(total: int) -> str:
    lvl, into, need = scoring.level(total)
    w = into / need * 100
    return (f'<div style="display:flex; align-items:center; gap:1rem; margin:0.2rem 0 0.8rem 0;">'
            f'<div style="font-size:1.8rem; font-weight:300; color:{THEME.amber};">LV {lvl}</div>'
            f'<div style="flex:1;"><div class="z-lbl">XP {total:,} · {into:,}/{need:,} to next</div>'
            f'<div style="height:3px; background:{THEME.grid}; margin-top:0.35rem;">'
            f'<div style="height:100%; width:{w:.1f}%; background:{THEME.amber};"></div></div></div></div>')


def _badges(have: set[str]) -> str:
    cells = []
    for k, (name, desc) in scoring.ACHIEVEMENTS.items():
        on = k in have
        c = THEME.amber if on else THEME.line
        cells.append(f'<div title="{html.escape(desc)}" style="border:1px solid {c}; padding:0.25rem 0.55rem; '
                     f'color:{THEME.text if on else THEME.faint}; letter-spacing:0.12em; font-size:0.6rem;">'
                     f'{"◆" if on else "◇"} {html.escape(name)}</div>')
    return '<div style="display:flex; flex-wrap:wrap; gap:0.35rem; margin:0.2rem 0 0.6rem 0;">' + \
        "".join(cells) + "</div>"


# ================================================================ render ====
def render() -> None:
    m = _market()
    if m is None:
        st.caption(DISCLAIMER)
        st.info("THETA RUN market data isn't published yet. Locally: `python -m zenith.theta.build`.")
        return
    repo, store, err = _repo_and_store()
    if err:
        st.warning(f"Saves unavailable ({err}); runs still work in this session.")
    in_run = _ss().get("th_run") is not None or _ss().get("th_debrief")
    player = None if in_run and _ss().get("th_player_cached") is None else _ss().get("th_player_cached")
    if not in_run:
        player = _player(repo)
        _ss()["th_player_cached"] = player
    _body(m, store, player)
    st.caption(DISCLAIMER)


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
    total, have, active = 0, set(), None
    if store is not None and player is not None:
        try:
            total = store.total_xp(player["id"])
            have = store.achievements(player["id"])
            active = store.active_run(player["id"])
        except Exception:
            pass
    hero, side = st.columns([1.5, 1], gap="large")
    with hero:
        st.markdown(
            f'<div style="display:flex; gap:1.4rem; align-items:center;">{art.orbits(190, seed=7)}<div>'
            f'<div class="z-lbl">options-portfolio speedrun</div>'
            f'<div style="font-size:1.6rem; font-weight:300; letter-spacing:0.14em; line-height:1.25; margin:0.4rem 0;">'
            f'SELL PREMIUM.<br>MANAGE THE BOOK.<br>BEAT THE BOT.</div>'
            f'<div style="font-size:0.76rem; color:{THEME.muted}; line-height:1.55; max-width:30rem;">'
            f'Real price history, modeled options, tastytrade mechanics. Click a name, press SELL, '
            f'advance to the next event — and race a mechanical 45-DTE / 16-delta bot on the same tape.</div>'
            f'</div></div>', unsafe_allow_html=True)
    with side:
        with uk.panel("QUICK RUN", THEME.teal, "one click"):
            _ss().setdefault("th_len", list(scenario.LENGTHS)[1])
            _ss().setdefault("th_diff", "normal")
            length = st.radio("Length", list(scenario.LENGTHS), horizontal=True, key="th_len",
                              label_visibility="collapsed")
            diff = st.radio("Difficulty", list(scenario.DIFFICULTY), horizontal=True, key="th_diff",
                            label_visibility="collapsed")
            if st.button("▶ START RUN", key="th_go_quick", type="primary", use_container_width=True):
                seed = int(time.time() * 1000) % 2_147_483_647
                _start(m, store, player, scenario.random_scenario(m, seed, scenario.LENGTHS[length], diff),
                       "quick")
                _rerun()
            today = date.today().isoformat()
            if st.button(f"DAILY RUN · {today}", key="th_go_daily", use_container_width=True):
                _start(m, store, player, scenario.random_scenario(m, scenario.daily_seed(today), 126, "normal"),
                       f"daily:{today}")
                _rerun()
            if active:
                sc = active["scenario"]
                if st.button(f"CONTINUE SAVED RUN · DAY {active['state']['i'] - sc['start_i']}",
                             key="th_continue", use_container_width=True):
                    _start(m, store, player, sc, active["mode"], active["state"], active["id"])
                    _rerun()
    if player is not None:
        a, b = st.columns([1, 2])
        with a:
            st.markdown(f'<div class="z-lbl">operator</div><div style="font-size:1rem; letter-spacing:0.12em;">'
                        f'{html.escape(player["display"])}</div>' + _xp_bar(total), unsafe_allow_html=True)
        with b:
            st.markdown(_badges(have), unsafe_allow_html=True)
    else:
        st.caption("Playing as a guest: runs work; XP and saves need a handle (top of this page).")

    st.markdown(section("CHALLENGES — live through the episodes that made the rules", 1), unsafe_allow_html=True)
    _ss().setdefault("th_ch_diff", "normal")
    cd = st.radio("Challenge difficulty", list(scenario.DIFFICULTY), horizontal=True, key="th_ch_diff")
    chs = _challenges()
    cols = st.columns(4)
    accents = [THEME.coral, THEME.amber, THEME.violet, THEME.coral, THEME.cyan, THEME.teal, THEME.mint]
    for k, ch in enumerate(chs):
        with cols[k % 4]:
            c = accents[k % len(accents)]
            st.markdown(
                f'<div style="border:1px solid {THEME.line}; border-top:2px solid {c}; padding:0.7rem 0.8rem; '
                f'background:{THEME.panel}; min-height:9.5rem;">'
                f'<div style="display:flex; justify-content:space-between;"><span style="font-size:0.58rem; '
                f'letter-spacing:0.2em; color:{c};">{ch["start"][:4]}{" · CRISIS" if ch.get("crisis") else ""}'
                f'{" · BLIND" if ch.get("blind") else ""}</span></div>'
                f'<div style="font-size:0.95rem; font-weight:300; letter-spacing:0.12em; margin:0.3rem 0;">'
                f'{html.escape(ch["title"])}</div>'
                f'<div style="font-size:0.7rem; color:{THEME.muted}; line-height:1.45;">{html.escape(ch["brief"][:150])}</div>'
                f'</div>', unsafe_allow_html=True)
            if st.button("ENTER ▸", key=f"th_go_ch_{ch['id']}", use_container_width=True):
                _start(m, store, player, scenario.challenge_scenario(m, ch, cd), f"challenge:{ch['id']}")
                _rerun()

    l1, l2 = st.columns([1.2, 1], gap="large")
    with l1:
        st.markdown(evidence_rating("B", "real prices, modeled options; mechanics follow OPTIONS VAULT",
                                    "Underlying paths are real; IV, skew and earnings moves are a calibrated "
                                    "model (SPY = VIX; stocks ~1.16x later realized vol)."), unsafe_allow_html=True)
        st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
        with st.expander("HOW TO PLAY — loop, scoring, prices, coach"):
            st.markdown(_HOW)
    with l2:
        if store is not None:
            try:
                lb = store.leaderboard(limit=10)
            except Exception:
                lb = []
            if lb:
                with uk.panel("LEADERBOARD", THEME.amber, "top runs by xp"):
                    df = pd.DataFrame(lb)
                    df["ret"] = df["ret"].map(lambda v: uc.fmt_pct(v))
                    df["max_dd"] = df["max_dd"].map(lambda v: uc.fmt_pct(v, signed=False))
                    st.dataframe(df.rename(columns={"display": "operator", "ret": "return", "max_dd": "max DD",
                                                    "updated_at": "when"}),
                                 hide_index=True, use_container_width=True)


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


SHORT = {"short_strangle": "STRANGLE", "short_straddle": "STRADDLE", "iron_condor": "IRON CONDOR",
         "iron_fly": "IRON FLY", "jade_lizard": "JADE LIZARD", "short_put": "SHORT PUT",
         "put_credit_spread": "PUT SPREAD", "call_credit_spread": "CALL SPREAD",
         "covered_call": "COVERED CALL", "put_bwb": "PUT BWB"}
_BY_SHORT = {v: k for k, v in SHORT.items()}
FILTERS = ["ETF", "STOCK", "IVR ≥ 30", "IVR ≥ 50", "EARNINGS ≤ 30D", "NO EARNINGS"]


def _race_svg(run: clock.Run, w: int = 220, h: int = 44) -> str:
    """You / bot / SPY as one tiny race chart for the HUD."""
    sc = _ss()["th_sc"]
    cash = run.book.start_cash
    k = run.i - sc["start_i"]
    you = [e[1] / cash for e in run.equity]
    bot_eq = [e[1] / cash for e in _ss()["th_bot"][: k + 1]]
    spy = [v / cash for v in _ss()["th_spy"][: k + 1]]
    series = [(spy, THEME.faint), (bot_eq, THEME.amber), (you, THEME.teal)]
    vals = [v for s, _ in series for v in s] or [1.0]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = max(len(s) for s, _ in series) or 1
    lines = []
    for s, col in series:
        if len(s) < 2:
            continue
        pts = " ".join(f"{j * w / max(n - 1, 1):.1f},{h - 2 - (v - lo) / span * (h - 4):.1f}" for j, v in enumerate(s))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><line x1="0" x2="{w}" '
            f'y1="{h - 2 - (1 - lo) / span * (h - 4):.1f}" y2="{h - 2 - (1 - lo) / span * (h - 4):.1f}" '
            f'stroke="{THEME.line}" stroke-dasharray="2 3"/>{"".join(lines)}</svg>')


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
    bp = b.bp_used(m, i) / max(nlv, 1)
    bpc = THEME.teal if bp <= 0.35 else (THEME.amber if bp <= 0.5 else THEME.coral)
    kp = uk.kpis([
        {"label": "net liq", "value": uc.fmt_money(nlv, 1), "sub": uc.fmt_pct(ret),
         "color": THEME.teal if ret >= 0 else THEME.coral},
        {"label": "vs the bot", "value": uc.fmt_pct(nlv / bot_nlv - 1.0), "sub": f"bot {uc.fmt_money(bot_nlv, 1)}",
         "color": THEME.teal if nlv >= bot_nlv else THEME.coral},
        {"label": "vs spy", "value": uc.fmt_pct(nlv / spy_nlv - 1.0), "sub": f"tape {uc.fmt_money(spy_nlv, 1)}",
         "color": THEME.teal if nlv >= spy_nlv else THEME.coral},
        {"label": "θ / day", "value": _money(g["theta"]), "sub": f"{g['theta'] / max(nlv, 1):.2%} of nlv",
         "color": THEME.amber},
        {"label": "β-delta", "value": f"{g['bw_delta']:+,.0f}", "sub": "spy shares", "color": THEME.violet},
        {"label": "bp in use", "value": f"{bp:.0%}", "sub": "guide 25–50%", "color": bpc},
    ])
    return (f'<div style="display:grid; grid-template-columns: 1fr auto; gap:1rem; align-items:center;">{kp}'
            f'<div><div class="z-lbl">race · you / bot / spy</div>{_race_svg(run)}</div></div>')


def _controls(m: Market, store, run: clock.Run, hud) -> None:
    c = st.columns([1, 1, 1.4, 1.3, 1])
    if c[0].button("+1 DAY", key="th_day", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "day"), save=False)
        _rerun()
    if c[1].button("+1 WEEK", key="th_week", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "week"), save=True)
        _rerun()
    if c[2].button("▶▶ NEXT EVENT", key="th_event", type="primary", use_container_width=True):
        _after_advance(m, store, run, clock.advance(m, run, "event"), save=True)
        _rerun()
    if c[3].button(f"⏩ AUTO {AUTOPLAY_DAYS}D", key="th_auto", use_container_width=True):
        events = []
        for _ in range(AUTOPLAY_DAYS):
            ev = clock.step(m, run)
            events += ev
            hud.markdown(_hud(m, run), unsafe_allow_html=True)
            if ev or run.done:
                break
            time.sleep(0.06)
        _after_advance(m, store, run, events, save=True)
        _rerun()
    if c[4].button("END RUN", key="th_end", use_container_width=True):
        _finish(m, store)
        _rerun()


def _game(m: Market, store, player) -> None:
    run: clock.Run = _ss()["th_run"]
    sc = _ss()["th_sc"]
    i = run.i
    total = sc["end_i"] - sc["start_i"]
    ch = next((c for c in _challenges() if c["id"] == sc.get("challenge")), None)
    title = ch["title"] if ch else ("DAILY RUN" if _ss().get("th_mode", "").startswith("daily") else "QUICK RUN")
    vix = m.vix[i] * 100
    vc = THEME.teal if vix < 20 else (THEME.amber if vix < 30 else THEME.coral)
    prog = (i - sc["start_i"]) / max(total, 1)
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:1.2rem; margin-bottom:0.6rem;">'
        f'<div style="font-size:1.25rem; font-weight:300; letter-spacing:0.16em;">{html.escape(title)}</div>'
        f'<div style="font-size:0.7rem; letter-spacing:0.16em; color:{THEME.muted};">{html.escape(_day(m, i))} · '
        f'DAY {i - sc["start_i"]}/{total} · <span style="color:{vc};">VIX {vix:.1f}</span> · '
        f'{sc["difficulty"].upper()}</div>'
        f'<div style="flex:1; height:2px; background:{THEME.grid}; position:relative;">'
        f'<div style="position:absolute; left:0; top:0; bottom:0; width:{prog * 100:.1f}%; background:{THEME.teal};">'
        f'</div></div></div>', unsafe_allow_html=True)
    hud = st.empty()
    hud.markdown(_hud(m, run), unsafe_allow_html=True)
    _controls(m, store, run, hud)

    left, mid, right = st.columns([1.12, 1.5, 1.08], gap="medium")
    with left:
        _screener(m, run)
    with mid:
        _ticket(m, store, run)
    with right:
        _book(m, store, run)
        _coach_feed(m, run, sc)
    b1, b2 = st.columns([1.6, 1], gap="medium")
    with b1:
        with uk.panel("RACE · YOU vs THE BOT vs THE TAPE", THEME.teal, "net liq & buying power"):
            _race_chart(m, run)
    with b2:
        with uk.panel("UNDERLYING", THEME.violet, "price · iv vs rv"):
            _underlying_chart(m, run)


def _screener(m: Market, run: clock.Run) -> None:
    with uk.panel("SCREENER", THEME.cyan, "click a row to load it"):
        df = pd.DataFrame(m.snapshot(run.i))
        if df.empty:
            st.info("No tradeable names on this date.")
            return
        f = st.pills("Filters", FILTERS, selection_mode="multi", key="th_flt", label_visibility="collapsed") or []
        kinds = [k for k, lab in (("etf", "ETF"), ("stock", "STOCK")) if lab in f]
        if "ETF" in f:
            kinds.append("index")
        if kinds:
            df = df[df["kind"].isin(kinds)]
        if "IVR ≥ 50" in f:
            df = df[df["ivr"] >= 0.5]
        elif "IVR ≥ 30" in f:
            df = df[df["ivr"] >= 0.3]
        if "EARNINGS ≤ 30D" in f:
            df = df[(df["earn_days"] >= 0) & (df["earn_days"] <= 30)]
        if "NO EARNINGS" in f:
            df = df[(df["earn_days"] < 0) | (df["earn_days"] > 30)]
        df = df.sort_values("ivr", ascending=False).reset_index(drop=True)
        show = pd.DataFrame({
            "SYM": df["ticker"].map(_label), "IV": (df["iv"] * 100).round(0),
            "IVR": (df["ivr"] * 100).round(0), "IV/RV": df["iv_rv"].round(2),
            "±30D": (df["em_30d"] / df["price"] * 100).round(1),
            "ERN": df["earn_days"].map(lambda d: f"{int(d)}d" if d >= 0 else "")})
        ev = st.dataframe(show, hide_index=True, use_container_width=True, height=470,
                          on_select="rerun", selection_mode="single-row", key="th_scr",
                          column_config={
                              "IVR": st.column_config.ProgressColumn("IVR", min_value=0, max_value=100, format="%d"),
                              "IV": st.column_config.NumberColumn("IV", format="%d"),
                              "±30D": st.column_config.NumberColumn("±30D", format="%.1f%%",
                                                                    help="30-day expected move, % of price")})
        rows = (ev.selection.rows if ev is not None and hasattr(ev, "selection") else []) or []
        if rows and rows[0] < len(df):
            _ss()["th_ticket"] = df["ticker"].iloc[rows[0]]
        st.caption(f"{len(df)} names · modeled IV · IVR = rank in trailing year")


def _payoff_chart(a: dict, S0: float, height: int = 230):
    df = pd.DataFrame({"price": a["grid"], "expiry": a["expiry"], "today": a["now"],
                       f"T+{a['half_days']}": a["half"]})
    long = df.melt("price", var_name="when", value_name="pnl")

    def build_(alt):
        base = alt.Chart(df)
        band = alt.Chart(pd.DataFrame({"lo": [S0 - a["em"]], "hi": [S0 + a["em"]]})).mark_rect(
            opacity=0.07, color=THEME.ice).encode(x="lo:Q", x2="hi:Q")
        pos = base.mark_area(opacity=0.28, color=THEME.teal).encode(
            x=alt.X("price:Q", title=None, scale=alt.Scale(zero=False)),
            y=alt.Y("pos:Q", title=None)).transform_calculate(pos="max(datum.expiry, 0)")
        neg = base.mark_area(opacity=0.28, color=THEME.coral).encode(
            x="price:Q", y="neg:Q").transform_calculate(neg="min(datum.expiry, 0)")
        lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
            x="price:Q", y="pnl:Q",
            color=alt.Color("when:N", scale=alt.Scale(domain=["expiry", "today", f"T+{a['half_days']}"],
                                                       range=[THEME.text, THEME.amber, THEME.violet]),
                            legend=alt.Legend(orient="top-left", title=None, direction="horizontal")),
            tooltip=[alt.Tooltip("price:Q", format=".2f"), "when:N", alt.Tooltip("pnl:Q", format=",.0f")])
        spot = alt.Chart(pd.DataFrame({"x": [S0]})).mark_rule(color=THEME.cyan, strokeDash=[3, 3]).encode(x="x:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.line).encode(y="y:Q")
        return (band + pos + neg + zero + lines + spot).properties(height=height)
    uc.render_chart(build_, fallback=df)


def _ticket(m: Market, store, run: clock.Run) -> None:
    i = run.i
    t = _ss().get("th_ticket") or "SPY"
    if t not in m.tickers or not m.live(t, i):
        t = "SPY"
    row = next((r for r in m.snapshot(i) if r["ticker"] == t), {"ticker": t})
    S = m.spot(t, i)
    ivr = row.get("ivr", float("nan"))
    with uk.panel(f"TICKET · {_label(t)}", THEME.amber,
                  f"{S:,.2f} · IV {row.get('iv', 0) * 100:.0f} · IVR {ivr * 100:.0f}" if ivr == ivr else f"{S:,.2f}"):
        pick = st.pills("Strategy", list(SHORT.values()), default="STRANGLE", key="th_tpl_pill",
                        label_visibility="collapsed") or "STRANGLE"
        tpl = _BY_SHORT[pick]
        spec = TEMPLATES[tpl]
        exps = m.expiry_choices(t, i, 120)
        if not exps:
            st.info("No listed expirations in range.")
            return
        c1, c2 = st.columns([1.3, 1])
        dte_pick = c1.segmented_control("DTE", ["30", "45", "60"], default="45", key="th_dte",
                                        label_visibility="collapsed") or "45"
        n = int(c2.number_input("Contracts", 1, 100, 1, key="th_n", label_visibility="collapsed"))
        e = min(exps, key=lambda x: abs(m.days_between(i, x) - int(dte_pick)))
        with st.expander("TWEAK STRIKES", expanded=False):
            t1, t2, t3 = st.columns(3)
            e = t1.selectbox("Expiration", exps, index=exps.index(e), key=f"th_exp_{dte_pick}",
                             format_func=lambda x: f"{m.days_between(i, x)} DTE"
                             + ("" if (_ss().get("th_sc") or {}).get("hide") else f" · {m.dates[x]:%b %d}"))
            dlt = t2.slider("Short delta", 0.05, 0.50, float(spec["delta"]), 0.01, key=f"th_dlt_{tpl}",
                            disabled=spec["delta"] >= 0.5)
            w = t3.number_input("Wing $", min_value=0.5, value=float(default_width(S)), step=0.5,
                                key=f"th_w_{tpl}_{t}", disabled=not spec["wing"])
        flat = run.book.flat_iv
        legs = build(tpl, m, t, i, e, n, dlt, w if spec["wing"] else None, flat)
        pv = run.book.preview(m, t, i, legs)
        a = analytics.analyse(m, t, i, legs, -pv["credit"], flat)
        nlv = run.book.nlv(m, i)
        credit = pv["credit"] >= 0
        st.markdown(uk.kpis([
            {"label": "credit" if credit else "debit", "value": _money(abs(pv["credit"])),
             "color": THEME.teal if credit else THEME.coral, "sub": f"fees {_money(pv['fees'])}"},
            {"label": "buying power", "value": _money(pv["bpr"]), "color": THEME.amber,
             "sub": f"{pv['bpr'] / max(nlv, 1):.1%} of nlv"},
            {"label": "pop", "value": f"{a['pop']:.0%}", "color": THEME.cyan, "sub": f"{a['dte']} dte"},
            {"label": "max loss", "value": "∞" if a["unlimited_up"] else _money(a["max_loss"]),
             "color": THEME.coral, "sub": " / ".join(f"{x:,.1f}" for x in a["breakevens"]) or "—"},
        ]), unsafe_allow_html=True)
        _payoff_chart(a, S)
        legs_txt = "  ".join(f"{'+' if lg.qty > 0 else ''}{lg.qty:g} {lg.kind.upper()}"
                             f"{'' if lg.kind == 'stock' else ' ' + format(lg.strike, 'g')} @ {lg.open_px:.2f}"
                             for lg in pv["legs"])
        st.markdown(f'<div style="font-size:0.68rem; color:{THEME.muted}; margin:-0.6rem 0 0.6rem 0;">'
                    f'{html.escape(legs_txt)}</div>', unsafe_allow_html=True)
        sc = _ss()["th_sc"]
        if scenario.DIFFICULTY[sc["difficulty"]]["coach"]:
            ns = pretrade(_vault(), row, tpl, pv, m.days_between(i, e), nlv, run.book.bp_used(m, i),
                          [p.ticker for p in run.book.open_positions()], m.earnings_between(t, i, e))
            for nd in [x.to_dict() for x in ns][:2]:
                st.markdown(_nudge_html(nd), unsafe_allow_html=True)
        verb = "SELL" if credit else "BUY"
        if st.button(f"{verb} {pick} · {_money(abs(pv['credit']))} {'CREDIT' if credit else 'DEBIT'}",
                     key="th_send", type="primary", use_container_width=True):
            run.book.open(m, t, i, tpl, legs, dlt, w if spec["wing"] else 0.0)
            clock.snapshot(m, run)
            _save(store, run)
            _rerun()


def _book(m: Market, store, run: clock.Run) -> None:
    i = run.i
    b = run.book
    opens = b.open_positions()
    with uk.panel(f"BOOK · {len(opens)} OPEN", THEME.teal, f"{len(b.positions) - len(opens)} closed"):
        if not opens:
            st.markdown(f'<div style="font-size:0.74rem; color:{THEME.muted};">No positions. Click a screener row, '
                        f'pick a strategy, press SELL.</div>', unsafe_allow_html=True)
        for p in opens:
            pnl = b.position_pnl(m, p, i)
            dte = m.days_between(i, p.expiry_i()) if p.option_legs() else None
            pct = pnl / p.credit if p.credit > 0 else None
            c = THEME.teal if pnl >= 0 else THEME.coral
            ring = ""
            if dte is not None:
                frac = max(0.0, min(1.0, dte / 60))
                ring = uk.radial_gauge(frac, 0, 1, f"{dte}d", THEME.amber if dte <= 21 else THEME.cyan, 54,
                                       fmt=" ")
            st.markdown(
                f'<div style="display:grid; grid-template-columns: 1fr auto; gap:0.4rem; align-items:center; '
                f'border-left:2px solid {c}; padding:0.2rem 0 0.2rem 0.6rem; margin:0.25rem 0;">'
                f'<div><div style="font-size:0.8rem;"><b style="font-weight:600;">{html.escape(_label(p.ticker))}</b> '
                f'<span style="color:{THEME.muted}; font-size:0.66rem; letter-spacing:0.1em;">'
                f'{html.escape(SHORT.get(p.template, p.template))}</span></div>'
                f'<div style="font-size:1.05rem; font-weight:300; color:{c};">{_money(pnl)}'
                f'<span style="font-size:0.66rem; color:{THEME.muted};">'
                f'{"" if pct is None else f"  {pct:+.0%} of credit"}</span></div></div>'
                f'<div>{ring}</div></div>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            if a1.button("CLOSE", key=f"th_close_{p.id}", use_container_width=True):
                b.close(m, p, i)
                clock.snapshot(m, run)
                _save(store, run)
                _rerun()
            if p.option_legs():
                later = [e for e in m.expiry_choices(p.ticker, i, 120) if e > p.expiry_i()]
                if later:
                    tgt = min(later, key=lambda e: abs(m.days_between(i, e) - 45))
                    if a2.button("ROLL", key=f"th_roll_{p.id}", use_container_width=True,
                                 help=f"Roll out to {m.days_between(i, tgt)} DTE for a credit"):
                        if b.roll(m, p, i, tgt, "out") is None:
                            st.toast("No roll for a credit here (tastylive: don't roll for a debit).")
                        else:
                            clock.snapshot(m, run)
                            _save(store, run)
                            _rerun()
                shorts = {lg.kind for lg in p.option_legs() if lg.qty < 0}
                if len(shorts) == 2 and a3.button("UNTESTED", key=f"th_unt_{p.id}", use_container_width=True,
                                                  help="Roll the untested short back to the target delta"):
                    if b.roll(m, p, i, p.expiry_i(), "untested") is None:
                        st.toast("Rolling the untested side wouldn't collect a credit right now.")
                    else:
                        clock.snapshot(m, run)
                        _save(store, run)
                        _rerun()
            stock = [lg for lg in p.legs if lg.kind == "stock" and lg.qty > 0]
            if stock and not any(lg.kind == "call" and lg.qty < 0 for lg in p.legs):
                ex = m.nearest_expiry(p.ticker, i, 35)
                if ex and a3.button("WHEEL", key=f"th_cc_{p.id}", use_container_width=True,
                                    help="Sell a 30-delta call against the assigned stock"):
                    k = m.strike_for_delta(p.ticker, i, ex, 0.30, True, b.flat_iv)
                    b.add_legs(m, p, i, [Leg("call", -stock[0].qty / 100.0, k, ex)], "Sold covered call")
                    p.flags.append("wheel")
                    clock.snapshot(m, run)
                    _save(store, run)
                    _rerun()


def _coach_feed(m: Market, run: clock.Run, sc: dict) -> None:
    with uk.panel("COACH & EVENTS", THEME.violet, "live"):
        if scenario.DIFFICULTY[sc["difficulty"]]["coach"]:
            for n in (_ss().get("th_nudges") or [])[:3]:
                st.markdown(_nudge_html(n), unsafe_allow_html=True)
        else:
            st.caption("Hard mode: the coach is silent until the debrief.")
        color = {"open": THEME.teal, "close": THEME.text, "roll": THEME.violet, "expire": THEME.muted,
                 "assign": THEME.coral, "target": THEME.teal, "stop": THEME.coral, "dte21": THEME.amber,
                 "earnings": THEME.orange, "bp": THEME.coral, "adjust": THEME.violet, "end": THEME.cyan}
        rows = [(_day(m, r["i"]), r["kind"], _alias_text(r["msg"]), color.get(r["kind"], THEME.muted))
                for r in list(reversed(run.book.log))[:14]]
        if rows:
            st.markdown(uk.log_feed(rows), unsafe_allow_html=True)


def _race_chart(m: Market, run: clock.Run) -> None:
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
        pd.DataFrame({"day": range(len(spy)), "who": "SPY", "ret": np.array(spy) / cash - 1}),
    ])
    bp = pd.DataFrame({"day": k, "bp": eq["bp"] / eq["nlv"].clip(lower=1)})

    def build_(alt):
        lines = alt.Chart(df).mark_line(strokeWidth=1.8, interpolate="monotone").encode(
            x=alt.X("day:Q", title=None, scale=alt.Scale(domain=[0, sc["end_i"] - sc["start_i"]])),
            y=alt.Y("ret:Q", title=None, axis=alt.Axis(format="+%")),
            color=alt.Color("who:N", scale=alt.Scale(domain=["YOU", "THE BOT", "SPY"],
                                                     range=[THEME.teal, THEME.amber, THEME.faint]),
                            legend=alt.Legend(orient="top-left", direction="horizontal", title=None)),
            tooltip=["who:N", "day:Q", alt.Tooltip("ret:Q", format="+.2%")]).properties(height=170)
        area = alt.Chart(bp).mark_area(color=THEME.amber, opacity=0.35, interpolate="step-after").encode(
            x=alt.X("day:Q", title=None, scale=alt.Scale(domain=[0, sc["end_i"] - sc["start_i"]])),
            y=alt.Y("bp:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8]))
        ).properties(height=60)
        return alt.vconcat(lines, area, spacing=4)
    uc.render_chart(build_, fallback=df)


def _underlying_chart(m: Market, run: clock.Run) -> None:
    sc = _ss()["th_sc"]
    t = _ss().get("th_ticket") or "SPY"
    if t not in m.data:
        t = "SPY"
    seg = slice(max(sc["start_i"] - 60, 0), run.i + 1)
    d = m.data[t]
    px = pd.DataFrame({"day": np.arange(seg.start, seg.stop) - sc["start_i"], "close": d["close"][seg],
                       "IV": d["iv30"][seg], "RV": d["rv21"][seg]})
    strikes = [lg.strike for p in run.book.open_positions() if p.ticker == t for lg in p.option_legs()]

    def build_(alt):
        line = alt.Chart(px).mark_line(color=THEME.text, strokeWidth=1.3).encode(
            x=alt.X("day:Q", title=None), y=alt.Y("close:Q", title=None, scale=alt.Scale(zero=False)))
        layers = line + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=THEME.faint,
                                                                      strokeDash=[2, 2]).encode(x="x:Q")
        if strikes:
            layers += alt.Chart(pd.DataFrame({"K": strikes})).mark_rule(color=THEME.coral,
                                                                         strokeDash=[4, 3]).encode(y="K:Q")
        vol = alt.Chart(px.melt("day", ["IV", "RV"], "s", "v")).mark_line(strokeWidth=1.2).encode(
            x=alt.X("day:Q", title=None), y=alt.Y("v:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("s:N", scale=alt.Scale(domain=["IV", "RV"], range=[THEME.violet, THEME.mint]),
                            legend=alt.Legend(orient="top-left", direction="horizontal", title=None)))
        return alt.vconcat(layers.properties(height=130, title=_label(t)), vol.properties(height=80), spacing=4)
    uc.render_chart(build_, fallback=px)


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
    c1, c2 = st.columns([1.6, 1])
    with c1:
        _race_chart(m, run)
    with c2:
        _underlying_chart(m, run)
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

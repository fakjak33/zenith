"""Streamlit rendering for the OPTIONS VAULT tab.

Reads only committed JSON in data/vault/ (VAULT_FILES) through @st.cache_data: no network, no
research dependencies, nothing from data/vault/cache/. Source links come from build.json, which
`python -m zenith.vault.build` packs from the manifest, pages inventory and literature.

Seven sections behind one radio (key ``vault_sub``, pre-seedable in tests like the other tabs).
The interactive parts (rolodex filters + card, literature search, sizing calculator, glossary
search) run inside @st.fragment: app.py executes every tab on each rerun, and a filter click here
should not pay for the other tabs.

Honesty rules carried onto the page: every card shows DRAFT and, where set, its thin-evidence
reason; CONVENTIONs say what they do to the P&L distribution; every number links to its source.
"""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from . import DISCLAIMER, load
from . import payoff as po
from . import sizing as sz
from .help_text import help_for
from .repro import charts as rc
from .. import ui_charts as uc
from ..config import THEME, VAULT_FILES
from ..ui_theme import evidence_rating, help_badge, key_findings, section, stamp

SUBVIEWS = ["START HERE", "STRATEGY ROLODEX", "EDGES & RESEARCH", "MECHANICS",
            "PORTFOLIO & SIZING", "GLOSSARY", "SOURCES & METHOD"]
TIER_COLORS = {"A": THEME.teal, "B": THEME.mustard, "C": THEME.coral}
KIND_COLORS = {"EDGE": THEME.teal, "CONVENTION": THEME.mauve, "MECHANIC": THEME.navy}
VERDICT_COLORS = {"agrees": THEME.teal, "mixed": THEME.mustard, "contradicts": THEME.coral,
                  "untested": THEME.muted}
REPRO_ORDER = ["vrp_spx", "strangle_mgmt", "expected_move", "vix_spikes", "cboe_short_vol",
               "cboe_tail_hedge", "earnings_drift"]
FIXED_WIDTH_REPRO = {"strangle_mgmt", "cboe_tail_hedge", "vix_spikes"}
REF_IN_BRACKETS =re.compile(r"\[((?:ep|lit|repro|page):[A-Za-z0-9._-]*[A-Za-z0-9](?:@\d+:\d{2})?)\]")

_FINDINGS = [
    {"stat": "VIX was above the next 21 days' realized SPX volatility on 85% of days since 1990, "
             "by +4.1 vol points on average: the variance risk premium is the edge.",
     "cite": "repro:vrp_spx · Bondarenko 2019"},
    {"stat": "Taking 50% or exiting at 21 DTE raises the win rate and cuts the tail but adds no "
             "expected return: 5.84% vs 5.83% a year in our strangle model.",
     "cite": "repro:strangle_mgmt · Williams 1991"},
    {"stat": "Since 2009 every Cboe short-volatility index trailed the S&P 500 total return; the "
             "premium is real, the crash exposure is the price.",
     "cite": "repro:cboe_short_vol"},
    {"stat": "After outlier earnings moves 51% kept drifting at +21 days: a coin flip, not the "
             "'more than half' seed hypothesis.",
     "cite": "repro:earnings_drift · Martineau 2022"},
]
_EVIDENCE_NOTE = ("Graded card by card. The index variance risk premium is replicated (A); most "
                  "tastylive management rules are conventions that reshape the distribution; 21 of "
                  "30 cards rest on thin evidence. Every number links to an episode, page, paper or "
                  "our own reproduction.")


# ============================================================== data ====
@st.cache_data(ttl=600, show_spinner=False)
def _art(cache_bust: str = "") -> dict:
    names = ["strategies", "edges", "mechanics", "portfolio", "glossary", "pop", "ranking",
             "coverage", "literature", "build"]
    a = {n: load(n, {}) for n in names}
    a["repro"] = {}
    d = VAULT_FILES["repro_dir"]
    for n in REPRO_ORDER:
        from .repro import load as rload
        obj = rload(n) if d.exists() else None
        if obj:
            a["repro"][n] = obj
    return a


def _bust() -> str:
    p = VAULT_FILES["strategies"]
    try:
        return str(p.stat().st_mtime) if p.exists() else ""
    except OSError:
        return ""


def _md(text) -> str:
    """Markdown-safe text: escapes $ (LaTeX) and ~ (strikethrough)."""
    return str(text or "").replace("$", "\\$").replace("~", "\\~")


def _ref_link(ref: str, refs: dict) -> str:
    kind, _, rest = ref.partition(":")
    key, _, ts = rest.partition("@")
    r = refs.get(f"{kind}:{key}", {})
    url = r.get("url", "")
    if kind == "ep":
        label = f"▶ {ts}" if ts else "episode"
        if r.get("yt"):
            secs = 0
            if ts:
                m, s = ts.split(":")
                secs = int(m) * 60 + int(s)
            url = f"https://www.youtube.com/watch?v={r['yt']}&t={secs}s"
    elif kind == "page":
        label = "tasty"
    elif kind == "lit":
        lab = r.get("label", key)
        m = re.match(r"([^,(&]+).*?\((\d{4})\)", lab)
        label = f"{m.group(1).strip().split()[-1]} {m.group(2)}" if m else key
    else:
        label = f"repro:{key}"
    if not url:
        return f"`{label}`"
    tip = r.get("label", ref).replace('"', "'")
    return f'[{label}]({url} "{_md(tip)}")'


def _linked(text, refs: dict) -> str:
    """Card prose with [kind:key@ts] markers turned into small source links."""
    s = _md(text)
    return REF_IN_BRACKETS.sub(lambda m: "⟨" + _ref_link(m.group(1), refs) + "⟩", s)


def _links(sources: list[str], refs: dict) -> str:
    return " ".join(_ref_link(s, refs) for s in sources or [])


def _tier_chip(tier: str, label: str = "") -> str:
    return uc.chip(f"{tier}{(' · ' + label) if label else ''}", TIER_COLORS.get(tier[:1], THEME.muted))


# ============================================================= badge ====
def today_badge() -> str | None:
    """TODAY chip: 'OPTIONS VAULT · 30 strategies · 15 edges'. Never raises."""
    try:
        b = load("build", {})
        c = b.get("counts") or {}
        if not c.get("strategies"):
            return None
        return uc.chip(f"OPTIONS VAULT · {c['strategies']} strategies · {c.get('edges', 0)} edges",
                       color=THEME.mauve,
                       sub=f"{b.get('status', 'draft')} · see OPTIONS VAULT tab")
    except Exception:
        return None


# ============================================================== main ====
def _signature(a: dict) -> None:
    """THE EVIDENCE MAP: every strategy card and edge placed by tier (A/B/C) and kind
    (edge / convention / mechanic) -- where the vault's confidence actually sits."""
    import altair as alt

    from .. import ui_kit as uk
    rows = []
    for c in a["strategies"].get("cards", []):
        ev = c.get("evidence") or {}
        rows.append({"what": "card", "name": c.get("name", c.get("slug")), "tier": str(ev.get("tier", "C"))[:1],
                     "kind": ev.get("kind", "EDGE")})
    for e in a.get("edges", {}).get("edges", []) if isinstance(a.get("edges"), dict) else []:
        rows.append({"what": "edge", "name": e.get("title", e.get("slug")), "tier": str(e.get("tier", "C"))[:1],
                     "kind": e.get("kind", "EDGE")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["j"] = df.groupby(["tier", "kind"]).cumcount()
    with uk.panel("THE EVIDENCE MAP", THEME.violet, f"{len(df)} cards & edges · tier × kind"):
        ch = alt.Chart(df).mark_point(filled=True, size=120).encode(
            x=alt.X("tier:N", sort=["A", "B", "C"], title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("kind:N", sort=["EDGE", "CONVENTION", "MECHANIC"], title=None),
            xOffset=alt.XOffset("j:O"),
            shape=alt.Shape("what:N", scale=alt.Scale(domain=["card", "edge"], range=["square", "diamond"]),
                            legend=alt.Legend(orient="top", title=None)),
            color=alt.Color("tier:N", scale=alt.Scale(domain=["A", "B", "C"],
                                                      range=[THEME.teal, THEME.amber, THEME.coral]), legend=None),
            tooltip=["name", "what", "tier", "kind"]).properties(height=220)
        st.altair_chart(ch, use_container_width=True)
        st.caption("Squares = strategy cards, diamonds = edges. Hover for the name; most of the vault is "
                   "honest tier C until a study or reproduction backs it.")


def render() -> None:
    st.caption(DISCLAIMER)
    st.markdown(evidence_rating("B", "mixed by card — A for the variance premium, C for most rules",
                                _EVIDENCE_NOTE), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    a = _art(_bust())
    cards = a["strategies"].get("cards", [])
    if not cards:
        st.markdown(stamp("—", "OPTIONS VAULT"), unsafe_allow_html=True)
        st.info("No data yet. The vault artefacts in data/vault/ are built by the research phases "
                "(prompts/vault/); then run `python -m zenith.vault.build` to pack the source links.")
        return
    st.markdown(stamp(a["build"].get("generated") or a["strategies"].get("generated", "—"),
                      "OPTIONS VAULT"), unsafe_allow_html=True)
    if a["strategies"].get("status", "draft") == "draft":
        st.markdown(uc.state_banner(THEME.mustard, "Draft",
                                    "cards, edges and trees are Phase 05 drafts; check each linked "
                                    "source before acting on a number"), unsafe_allow_html=True)
    _signature(a)
    sub = st.radio("Section", SUBVIEWS, horizontal=True, key="vault_sub",
                   label_visibility="collapsed")
    refs = a["build"].get("refs", {})
    {"START HERE": _start_here, "STRATEGY ROLODEX": _rolodex, "EDGES & RESEARCH": _edges,
     "MECHANICS": _mechanics, "PORTFOLIO & SIZING": _portfolio, "GLOSSARY": _glossary,
     "SOURCES & METHOD": _sources}[sub](a, refs)


def _open_card(slug: str) -> None:
    """on_click target for 'act with' / 'related' buttons: jump to a card."""
    st.session_state["vault_sub"] = "STRATEGY ROLODEX"
    st.session_state["vault_card"] = slug
    for k in ("vault_f_outlook", "vault_f_iv", "vault_f_risk", "vault_f_acct", "vault_f_cx"):
        st.session_state.pop(k, None)


# ======================================================== START HERE ====
def _start_here(a: dict, refs: dict) -> None:
    edges = {e["slug"]: e for e in a["edges"].get("edges", [])}
    st.markdown(section("Where the edge comes from", 0, help=help_for("kind")), unsafe_allow_html=True)
    vrp = edges.get("vrp")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**1 · The variance risk premium**")
        st.markdown(_linked(vrp["finding"], refs) if vrp else "—")
    with c2, st.container(border=True):
        st.markdown("**2 · Why probabilities and mechanics matter**")
        st.markdown(_md(
            "A short-premium trade wins often and loses rarely but large. The win rate is a "
            "property of the strikes, not of skill: 16-delta strikes sit near one standard "
            "deviation, and SPX stayed inside the VIX 1-sigma band 83.9% of the time, with "
            "downside breaks rarer but twice as far.") + " " + _ref_link("repro:expected_move", refs)
            + "\n\n" + _md("Management rules (50% profit, 21 DTE, 2x stop) change the shape of the "
                           "P&L distribution, not its expected value.") + " "
            + _ref_link("repro:strangle_mgmt", refs))
    with c3, st.container(border=True):
        st.markdown("**3 · Trade small, trade often**")
        occ = edges.get("number-of-occurrences")
        st.markdown(_linked(occ["finding"], refs) if occ else
                    _md("Keep each trade small so results come from many occurrences."))
    st.markdown(section("How to read a card", 2), unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("".join(_tier_chip(t, d) for t, d in (
            ("A", "replicated across periods & markets"),
            ("B", "documented, period/cost sensitive"),
            ("C", "tastylive-only, small sample or contradicted"))), unsafe_allow_html=True)
    with l2:
        st.markdown("".join(uc.chip(k, c, sub) for k, c, sub in (
            ("EDGE", KIND_COLORS["EDGE"], "a source of expected return"),
            ("CONVENTION", KIND_COLORS["CONVENTION"], "reshapes the distribution, rule of thumb"),
            ("MECHANIC", KIND_COLORS["MECHANIC"], "how it works"))), unsafe_allow_html=True)
    st.markdown(uc.chip("DRAFT", THEME.mustard, help_for("draft"))
                + uc.chip("THIN EVIDENCE", THEME.coral, help_for("thin")), unsafe_allow_html=True)

    st.markdown(section("Evidence ladder", 4, help=help_for("tier")), unsafe_allow_html=True)
    rows = [{"set": "strategy cards", "tier": c["evidence"]["tier"], "kind": c["evidence"]["kind"]}
            for c in a["strategies"].get("cards", [])]
    rows += [{"set": "edges", "tier": e["tier"], "kind": e["kind"]} for e in a["edges"].get("edges", [])]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["set", "tier", "kind"]).size().reset_index(name="n")
        df["row"] = df["tier"] + " · " + df["set"]
        order = [f"{t} · {s}" for t in "ABC" for s in ("strategy cards", "edges")]

        def build(alt):
            return alt.Chart(df).mark_bar().encode(
                y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False)),
                x=alt.X("n:Q", title="count"),
                color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=list(KIND_COLORS), range=list(KIND_COLORS.values()))),
                tooltip=["set", "tier", "kind", "n"]).properties(height=260, width="container")
        uc.render_chart(build, fallback=df)
        st.caption("Most cards are tier C: the Learn Center pages give rules without study numbers, "
                   "and only the variance premium and the cost of long premium are tier A.")

    rk = a["ranking"]
    if rk.get("most_reliable_defined") or rk.get("most_reliable_undefined"):
        st.markdown(section("Most reliable strategies (evidence first)", 1), unsafe_allow_html=True)
        st.caption(_linked(rk.get("note", ""), refs))
        cols = st.columns(2)
        for col, key, title in ((cols[0], "most_reliable_defined", "defined risk"),
                                (cols[1], "most_reliable_undefined", "undefined risk")):
            rows = [{"strategy": r["name"], "tier": r["tier"], "kind": r["kind"],
                     "POP (model, realized vol)": (f"{r['pop_realized']:.0%}"
                                                   if r.get("pop_realized") is not None else "—"),
                     "thin evidence": "yes" if r.get("thin_evidence") else ""} for r in rk.get(key, [])]
            with col:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             column_config={"POP (model, realized vol)": st.column_config.TextColumn(
                                 help=help_for("pop") + " '—' = no POP model for this card.")})


# ==================================================== STRATEGY ROLODEX ====
def _outlook_tags(c: dict) -> set[str]:
    t = (c.get("when", {}).get("outlook") or "").lower()
    first = t.split(".")[0]
    tags = set()
    if "neutral" in first:
        tags.add("neutral")
    if "bull" in first:
        tags.add("bullish")
    if "bear" in first:
        tags.add("bearish")
    if "volatility" in first and ("up" in first or "move" in first or "big" in first):
        tags.add("big move")
    if "defensive" in first or "protect" in first or "hedge" in first:
        tags.add("hedge")
    return tags or {"other"}


def _iv_tag(c: dict) -> str:
    t = (c.get("when", {}).get("iv") or "").lower()
    if re.search(r"low iv|iv is low|lowest-iv|low implied|iv rank is low|low iv rank", t):
        return "low IV"
    if re.search(r"high iv|iv rank above|iv is high|iv rank and high|high implied|elevated", t):
        return "high IV"
    return "no IV rule"


def _complexity(c: dict) -> str:
    n = len(c.get("legs") or [])
    return {0: "varies", 1: "1 leg", 2: "2 legs"}.get(n, "3+ legs")


def _greek_glyphs(g: dict) -> str:
    out = []
    for sym, k in (("Δ", "delta"), ("Γ", "gamma"), ("Θ", "theta"), ("ν", "vega")):
        v = str(g.get(k) or "")
        s = v.strip()[:1] if v.strip()[:1] in "+-0~" else "·"
        color = {"+": THEME.teal, "-": THEME.coral}.get(s, THEME.muted)
        out.append(f'<span title="{html.escape(k + ": " + v)}" style="display:inline-block; '
                   f'min-width:2.6rem; text-align:center; border:1px solid {color}; padding:0.1rem 0.3rem; '
                   f'margin-right:0.3rem; color:{color}; font-family:{THEME.font_display};">'
                   f'{sym} {html.escape(s)}</span>')
    return "".join(out)


def _tile(c: dict, selected: bool) -> str:
    ev = c["evidence"]
    tc = TIER_COLORS.get(ev["tier"], THEME.muted)
    thin = (f'<span style="color:{THEME.coral}; font-size:0.68rem;"> · thin</span>'
            if c.get("thin_evidence") else "")
    border = THEME.text if selected else tc
    return (f'<div style="flex:1 1 170px; min-width:160px; max-width:240px; border:1px solid {border}; '
            f'border-left:5px solid {tc}; padding:0.35rem 0.5rem; background:rgba(0,0,0,0.3);">'
            f'<div style="font-size:0.86rem; color:{THEME.text};">{html.escape(c["name"])}</div>'
            f'<div style="font-size:0.68rem; color:{THEME.muted};">'
            f'<span style="color:{tc};">{ev["tier"]}</span> · {html.escape(ev["kind"])} · '
            f'{html.escape(c["risk"])}{thin}</div></div>')


def _payoff_chart(slug: str):
    s = po.setup_for(slug)
    if not s:
        return None, None, None
    legs = po.price_legs(s["legs"])
    rows = po.curves(legs)
    summ = po.summary(legs)
    df = pd.DataFrame(rows)
    horizons = list(dict.fromkeys(df["horizon"]))
    colors = {"T+0": THEME.teal, f"T+{po.HORIZON}": THEME.mustard, "expiry": THEME.text}

    def build(alt):
        exp = df[df["horizon"] == "expiry"]
        # profit / loss fills between the expiry curve and zero, as explicit y / y2 bands
        # profit / loss fill between the expiry curve and zero, one thin rect per grid step
        step = float(exp["S"].iloc[1] - exp["S"].iloc[0])
        fill = pd.DataFrame({"S": exp["S"] - step / 2, "S2": exp["S"] + step / 2,
                             "a": exp["pnl"].clip(upper=0), "b": exp["pnl"].clip(lower=0)})
        fill["zone"] = ["profit" if v > 0 else "loss" for v in exp["pnl"]]
        x = alt.X("S:Q", title="underlying price (entry = 100)",
                  scale=alt.Scale(domain=[70, 130], zero=False, nice=False))
        y_title = "P&L per share (x100 per contract)"
        area = alt.Chart(fill).mark_rect(opacity=0.22).encode(
            x=x, x2="S2:Q", y=alt.Y("a:Q", title=y_title), y2="b:Q",
            fill=alt.Fill("zone:N", legend=None, scale=alt.Scale(domain=["profit", "loss"],
                                                                 range=[THEME.teal, THEME.coral])))
        lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
            x=x, y=alt.Y("pnl:Q", title=y_title),
            color=alt.Color("horizon:N", title=None, sort=horizons, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=horizons, range=[colors[h] for h in horizons])),
            tooltip=["S", "horizon", alt.Tooltip("pnl:Q", format="+.2f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=THEME.muted).encode(y="y:Q")
        ks = [k for k in summ["strikes"] if 70 <= k <= 130]
        marks = pd.DataFrame({"S": [100.0] + ks, "what": ["entry"] + ["strike"] * len(ks)})
        rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3]).encode(
            x=x, color=alt.Color("what:N", legend=None, scale=alt.Scale(
                domain=["entry", "strike"], range=[THEME.muted, THEME.mauve])),
            tooltip=["what", alt.Tooltip("S:Q", format=".2f")])
        return (area + zero + rules + lines).resolve_scale(color="independent").properties(
            height=300, width="container")
    return build, s, summ


@st.fragment
def _rolodex(a: dict, refs: dict) -> None:
    cards = a["strategies"].get("cards", [])
    st.markdown(section("Strategy rolodex", 1, help=help_for("tier")), unsafe_allow_html=True)
    f = st.columns(5)
    outlooks = sorted({t for c in cards for t in _outlook_tags(c)})
    o = f[0].multiselect("Outlook", outlooks, key="vault_f_outlook")
    iv = f[1].multiselect("IV environment", ["high IV", "low IV", "no IV rule"], key="vault_f_iv",
                          help=help_for("iv rank"))
    risk = f[2].multiselect("Risk", sorted({c["risk"] for c in cards}), key="vault_f_risk",
                            help=help_for("undefined risk"))
    acct = f[3].selectbox("Account", ["any"] + list(sz.ACCOUNTS), key="vault_f_acct",
                          format_func=lambda k: "any account" if k == "any" else sz.ACCOUNTS[k])
    cx = f[4].multiselect("Complexity", ["1 leg", "2 legs", "3+ legs", "varies"], key="vault_f_cx")
    shown = [c for c in cards
             if (not o or _outlook_tags(c) & set(o)) and (not iv or _iv_tag(c) in iv)
             and (not risk or c["risk"] in risk) and (acct == "any" or c["accounts"].get(acct))
             and (not cx or _complexity(c) in cx)]
    st.caption(f"{len(shown)} of {len(cards)} cards · {a['strategies'].get('status', 'draft').upper()} · "
               f"{sum(1 for c in shown if c.get('thin_evidence'))} with thin evidence")
    if not shown:
        st.info("No card matches these filters.")
        return
    slugs = [c["slug"] for c in shown]
    if st.session_state.get("vault_card") not in slugs:
        st.session_state["vault_card"] = slugs[0]
    sel = st.session_state["vault_card"]
    st.markdown('<div style="display:flex; gap:0.45rem; flex-wrap:wrap; margin:0.2rem 0 0.8rem 0;">'
                + "".join(_tile(c, c["slug"] == sel) for c in shown) + "</div>", unsafe_allow_html=True)
    names = {c["slug"]: c["name"] for c in cards}
    slug = st.selectbox("Open card", slugs, key="vault_card", format_func=lambda s: names.get(s, s))
    _card(next(c for c in cards if c["slug"] == slug), a, refs, names)


def _box(title: str, rows: list[tuple[str, str]], refs: dict, help_key: str = "") -> None:
    with st.container(border=True):
        badge = help_badge(help_for(help_key)) if help_key else ""
        st.markdown(f'<div style="font-family:{THEME.font_display}; letter-spacing:0.12em; '
                    f'color:{THEME.mustard}; text-transform:uppercase;">{html.escape(title)}{badge}</div>',
                    unsafe_allow_html=True)
        for label, text in rows:
            if text:
                st.markdown(f"**{_md(label)}** · {_linked(text, refs)}")


def _strangle_stop_callout(a: dict) -> str | None:
    r = a["repro"].get("strangle_mgmt") or {}
    try:
        ep = r["episodes"]
        base, stop = ep["p50_d21"]["covid_2020"], ep["p50_d21_stop2x"]["covid_2020"]
        rules = {x["rule"]: x for x in r["chart_data"]["rules"]}
        return (f"Our model test of the 2x-credit stop: the worst March 2020 trade went from "
                f"{base['worst_pnl_credits']:+.1f} to {stop['worst_pnl_credits']:+.1f} credits, while the "
                f"mean per trade fell slightly ({rules['p50_d21']['mean_pnl_credits']:+.2f} → "
                f"{rules['p50_d21_stop2x']['mean_pnl_credits']:+.2f} credits). Closes only: real stop "
                f"fills in a gap would be worse.")
    except (KeyError, TypeError):
        return None


def _card(c: dict, a: dict, refs: dict, names: dict) -> None:
    ev = c["evidence"]
    st.markdown(f"### {_md(c['name'])}")
    chips = (_tier_chip(ev["tier"], "evidence tier")
             + uc.chip(ev["kind"], KIND_COLORS.get(ev["kind"], THEME.muted))
             + uc.chip(f"{c['risk']} risk", THEME.coral if c["risk"] == "undefined" else THEME.teal)
             + uc.chip("DRAFT", THEME.mustard, "not yet reviewed line by line"))
    if c.get("thin_evidence"):
        chips += uc.chip("THIN EVIDENCE", THEME.coral)
    st.markdown(chips + "&nbsp;" + _greek_glyphs(c.get("greeks", {})), unsafe_allow_html=True)
    st.markdown(_linked(c["one_liner"], refs))
    if c.get("thin_evidence"):
        st.warning(f"Thin evidence: {c['thin_evidence']}", icon="⚠️")
    st.markdown(evidence_rating(ev["tier"], ev["kind"].lower(), re.sub(r"\s*\[[^\]]+\]", "", ev["note"])),
                unsafe_allow_html=True)
    if c["slug"] == "short-strangle":
        call = _strangle_stop_callout(a)
        if call:
            st.markdown(uc.note_strip("Phase 05 finding · 2x-credit stop", [call]), unsafe_allow_html=True)
            st.caption("Source: " + _ref_link("repro:strangle_mgmt", refs))

    w, s, m, size = c.get("when", {}), c.get("setup", {}), c.get("manage", {}), c.get("size", {})
    r1 = st.columns(2)
    with r1[0]:
        _box("When", [("IV", w.get("iv")), ("Outlook", w.get("outlook")), ("Events", w.get("events")),
                      ("Avoid", " · ".join(c.get("avoid", [])[:2]))], refs, "iv rank")
    with r1[1]:
        _box("Setup", [("DTE", s.get("dte")), ("Strikes", s.get("strikes")),
                       ("Credit target", s.get("credit_target")), ("Underlying", s.get("underlying"))],
             refs, "dte")
    r2 = st.columns(2)
    with r2[0]:
        _box("Manage", [("Profit target", m.get("profit_target")), ("Time exit", m.get("time_exit")),
                        ("Roll", m.get("roll")), ("Defend", m.get("defend"))], refs)
    with r2[1]:
        _box("Exit & size", [("Loss rule", m.get("loss_rule")), ("Per trade", size.get("per_trade")),
                             ("BPR", size.get("bpr_formula")), ("Notes", size.get("notes"))], refs, "bpr")

    build, setup, summ = _payoff_chart(c["slug"])
    if build:
        st.markdown(section("Payoff", 3, help="Black-Scholes on an underlying of 100, flat 20% IV, no skew "
                            "or costs (the pop.py model). T+0 = right after entry; T+21 = 21 days later at "
                            "unchanged IV; expiry = first expiration."), unsafe_allow_html=True)
        uc.render_chart(build)
        be = ", ".join(f"{b:.1f}" for b in summ["breakevens"]) or "—"
        mp = "open-ended" if summ["max_profit"] is None else f"{summ['max_profit']:+.2f}"
        ml = "open-ended" if summ["max_loss"] is None else f"{summ['max_loss']:+.2f}"
        st.markdown(uc.numeric_slab([
            {"label": f"net {summ['net']}", "value": f"{summ['premium']:.2f}",
             "color": THEME.teal if summ["net"] == "credit" else THEME.coral},
            {"label": "breakevens", "value": be},
            {"label": "max profit", "value": mp, "color": THEME.teal},
            {"label": "max loss", "value": ml, "color": THEME.coral}], min_width=130),
            unsafe_allow_html=True)
        pop = next((r for r in a["pop"].get("rows", []) if r["slug"] == c["slug"]), None)
        cap = f"Model setup: {setup['setup']}."
        if setup["ours"]:
            cap += " Illustrative strikes chosen by us; the card gives no numbers for them."
        if pop:
            cap += (f" Held to expiry, POP {pop['pop_theoretical']:.0%} at the pricing vol and "
                    f"{pop['pop_realized']:.0%} at realized = 0.78 × implied")
        st.caption(_md(cap) + (" " + _links(a["pop"].get("sources", []), refs) if pop else ""))
    else:
        st.caption("No payoff diagram: the card has no single default structure.")
    st.caption("Build it with live prices in [OptionStrat](https://optionstrat.com) or check IV and "
               "earnings dates on [Market Chameleon](https://marketchameleon.com).")

    with st.expander("Card back · sourced stats, conventions, history, mistakes, example"):
        if c.get("stats"):
            st.markdown("**Sourced stats**")
            for x in c["stats"]:
                st.markdown(f"- **{_md(x['label'])}**: {_md(x['value'])}  \n"
                            f"  <small>{_md(x.get('conditions', ''))}</small> {_links(x.get('sources'), refs)}",
                            unsafe_allow_html=True)
        if c.get("conventions"):
            st.markdown("**Conventions: what each rule does to the distribution**")
            for x in c["conventions"]:
                st.markdown(f"- **{_md(x['rule'])}**: {_linked(x['effect'], refs)}")
        if c.get("history"):
            st.markdown("**Rule history**")
            for x in sorted(c["history"], key=lambda h: h.get("year") or 0):
                st.markdown(f"- **{x.get('year')}**: {_md(x['change'])} {_links(x.get('sources'), refs)}")
        if c.get("mistakes"):
            st.markdown("**Common mistakes**")
            for x in c["mistakes"]:
                st.markdown(f"- {_linked(x, refs)}")
        if c.get("avoid"):
            st.markdown("**Avoid**")
            for x in c["avoid"]:
                st.markdown(f"- {_linked(x, refs)}")
        if c.get("example"):
            st.markdown("**Worked example**")
            for k, v in c["example"].items():
                st.markdown(f"- *{_md(k.replace('_', ' '))}*: {_linked(v, refs)}")
        if c.get("variants"):
            st.markdown("**Variants folded in**: " + ", ".join(
                _md(v if isinstance(v, str) else v.get("name", v.get("slug", ""))) for v in c["variants"]))
        if c.get("accounts_note"):
            st.markdown(f"**Accounts**: {_linked(c['accounts_note'], refs)}")
        st.markdown("**Evidence**: " + _links(ev.get("literature", []) + ev.get("repro", []), refs))
        if c.get("popularity"):
            p = c["popularity"]
            st.caption(f"Archive popularity: {p.get('episodes')} episode mentions "
                       f"{p.get('by_period', '')}. {p.get('note', '')}")
        rel = [r for r in c.get("related", []) if r in names]
        if rel:
            st.markdown("**Related cards**")
            cols = st.columns(min(len(rel), 4))
            for i, r in enumerate(rel):
                cols[i % len(cols)].button(names[r], key=f"vault_rel_{c['slug']}_{r}",
                                           on_click=_set_card, args=(r,))


def _set_card(slug: str) -> None:
    st.session_state["vault_card"] = slug


# ================================================== EDGES & RESEARCH ====
def _edges(a: dict, refs: dict) -> None:
    edges = a["edges"].get("edges", [])
    names = {c["slug"]: c["name"] for c in a["strategies"].get("cards", [])}
    st.markdown(section("Edges and conventions by tier", 2, help=help_for("kind")), unsafe_allow_html=True)
    for tier in "ABC":
        group = [e for e in edges if e["tier"] == tier]
        if not group:
            continue
        st.markdown(_tier_chip(tier, f"{len(group)} edge cards"), unsafe_allow_html=True)
        for e in group:
            with st.container(border=True):
                st.markdown(f"**{_md(e['title'])}**")
                st.markdown(_tier_chip(e["tier"]) + uc.chip(e["kind"], KIND_COLORS.get(e["kind"], THEME.muted))
                            + uc.chip(f"tastylive vs evidence: {e['agreement']}",
                                      VERDICT_COLORS.get(e["agreement"], THEME.muted)),
                            unsafe_allow_html=True)
                st.markdown(_linked(e["finding"], refs))
                with st.expander("Numbers, conditions, caveats, act with"):
                    for n in e.get("numbers", []):
                        st.markdown(f"- **{_md(n['label'])}**: {_md(n['value'])} {_links(n.get('sources'), refs)}")
                    if e.get("tasty_position"):
                        st.markdown(f"**tastylive's position**: {_linked(e['tasty_position'], refs)}")
                    if e.get("conditions"):
                        st.markdown(f"**Conditions**: {_linked(e['conditions'], refs)}")
                    for cv in e.get("caveats", []):
                        st.markdown(f"- ⚠ {_linked(cv, refs)}")
                    act = [s for s in e.get("act_with", []) if s in names]
                    if act:
                        st.markdown("**Act with**")
                        cols = st.columns(4)
                        for i, s in enumerate(act):
                            cols[i % 4].button(names[s], key=f"vault_act_{e['slug']}_{s}",
                                               on_click=_open_card, args=(s,))

    st.markdown(section("Our reproductions: tastylive vs our free-data test", 4), unsafe_allow_html=True)
    rows = []
    for n, r in a["repro"].items():
        for cl in r.get("claims", []):
            rows.append({"repro": n, "claim": cl["text"], "tastylive": cl.get("tasty_value") or "—",
                         "ours": cl.get("our_value"), "verdict": cl.get("verdict")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=320)
        st.caption("Verdicts are on the claim, not on tastylive. 'pending'/'—' = no tastylive figure found.")
    for n, r in a["repro"].items():
        with st.expander(f"repro:{n} · {r.get('verdict', '')} · {r.get('claim', '')}"):
            st.markdown(f"**Ours**: {_md(r.get('our_value'))}")
            if r.get("tasty_value"):
                st.markdown(f"**tastylive**: {_md(r['tasty_value'])} {_links(r.get('tasty_sources'), refs)}")
            st.caption(_md(f"Data: {r.get('data', '')} · period {r.get('period', '')}"))
            if n in FIXED_WIDTH_REPRO:
                # concat / faceted charts with fixed panel widths: stretching clips their labels
                try:
                    st.altair_chart(rc.chart(n, r), use_container_width=False)
                except Exception:
                    st.caption("Chart unavailable.")
            elif n in rc.BUILDERS:
                uc.render_chart(lambda alt, n=n, r=r: rc.BUILDERS[n](r, alt))
            if r.get("caveats"):
                st.markdown(uc.note_strip("Caveats", r["caveats"][:4]), unsafe_allow_html=True)

    st.markdown(section("Literature", 5), unsafe_allow_html=True)
    _literature(a, refs)
    st.info("A searchable per-episode study catalog is not built: the full transcript extraction "
            "(Phase 02 steps 3-4) was replaced by the Trader Resources pivot, so episode evidence "
            "here is limited to the 20 checked pilot extracts cited on the cards.")


@st.fragment
def _literature(a: dict, refs: dict) -> None:
    topics = a["literature"].get("topics", [])
    if not topics:
        st.caption("No literature artefact.")
        return
    st.dataframe(pd.DataFrame([{"topic": t["title"], "kind": t.get("kind"), "tier": t.get("tier"),
                                "agreement with tastylive": t.get("agreement_with_tasty"),
                                "papers": len(t.get("papers", []))} for t in topics]),
                 hide_index=True, use_container_width=True)
    q = st.text_input("Search papers (author, title, finding)", key="vault_lit_q")
    rows = [{"topic": t["id"], "authors": p.get("authors"), "year": p.get("year"), "title": p.get("title"),
             "venue": p.get("venue"), "kind": p.get("kind"), "verified": p.get("verified"),
             "finding": p.get("finding"), "link": p.get("url") or ""}
            for t in topics for p in t.get("papers", [])]
    df = pd.DataFrame(rows)
    if q.strip():
        ql = q.lower()
        mask = df[["authors", "title", "finding", "topic"]].astype(str).apply(
            lambda col: col.str.lower().str.contains(ql, regex=False)).any(axis=1)
        df = df[mask]
    n_unique = len({p.get("id") for t in topics for p in t.get("papers", [])})
    st.caption(f"{len(df)} of {len(rows)} rows ({n_unique} unique papers; a paper can sit under several "
               f"topics) · 'partial' = metadata checked, some numbers from secondary summaries.")
    st.dataframe(df, hide_index=True, use_container_width=True, height=360,
                 column_config={"link": st.column_config.LinkColumn("link", display_text="open")})


# ========================================================= MECHANICS ====
def _wrap(text: str, width: int = 34) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\n".join(lines)


def tree_dot(t: dict) -> str:
    """DOT for one decision tree (decision = mustard diamond-ish box, action = teal, info = muted note)."""
    style = {"decision": f'shape=box, style="rounded,bold", color="{THEME.mustard}"',
             "action": f'shape=box, color="{THEME.teal}"',
             "info": f'shape=note, color="{THEME.muted}", fontcolor=THEME.muted'}
    out = ['digraph G {', 'bgcolor="transparent"; rankdir=TB; nodesep=0.3; ranksep=0.35;',
           'node [fontname="Courier", fontsize=10, margin="0.18,0.08", fontcolor=THEME.text, penwidth=1.4];',
           'edge [color=THEME.muted, fontcolor=THEME.muted, fontsize=9, fontname="Courier"];']
    for n in t.get("nodes", []):
        txt = re.sub(r"\s*\[[^\]]+\]", "", n["text"]).replace('"', "'")
        out.append(f'"{n["id"]}" [label="{_wrap(txt)}", {style.get(n["type"], style["action"])}];')
    for e in t.get("edges", []):
        lab = (e.get("label") or "").replace('"', "'")
        out.append(f'"{e["from"]}" -> "{e["to"]}" [label="{lab}"];')
    out.append("}")
    return "\n".join(out)


def dte_timeline_rows() -> list[dict]:
    """Share of a 16-delta 45-DTE strangle's credit captured by day, price and IV unchanged."""
    legs = po.price_legs(po.setup_for("short-strangle")["legs"])
    credit = -po.entry_cost(legs)
    # value_at is signed (a short position is worth minus its buy-back cost)
    return [{"dte": 45 - d, "captured": round(1 + po.value_at(legs, 100.0, d) / credit, 4)}
            for d in range(0, 46)]


def _mechanics(a: dict, refs: dict) -> None:
    m = a["mechanics"]
    st.markdown(section("Trade lifecycle: ~45 DTE → 50% → 21 DTE", 3, help=help_for("dte")),
                unsafe_allow_html=True)
    df = pd.DataFrame(dte_timeline_rows())

    def build(alt):
        x = alt.X("dte:Q", title="days to expiration (time runs left to right)",
                  scale=alt.Scale(domain=[0, 60], reverse=True))
        band = alt.Chart(pd.DataFrame({"a": [60], "b": [30]})).mark_rect(color=THEME.teal, opacity=0.12
                                                                      ).encode(x="a:Q", x2="b:Q")
        line = alt.Chart(df).mark_line(color=THEME.teal, strokeWidth=2.4).encode(
            x=x, y=alt.Y("captured:Q", title="share of credit captured", axis=alt.Axis(format="%"),
                         scale=alt.Scale(domain=[0, 1])),
            tooltip=["dte", alt.Tooltip("captured:Q", format=".0%")])
        marks = pd.DataFrame({"dte": [45, 21, 0], "label": ["enter ~45 DTE", "21-DTE exit", "expiry"]})
        rules = alt.Chart(marks).mark_rule(color=THEME.mustard, strokeDash=[5, 3]).encode(x="dte:Q")
        text = alt.Chart(marks).mark_text(color=THEME.mustard, align="left", dx=4).encode(
            x="dte:Q", text="label:N", y=alt.value(10))
        half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=THEME.coral, strokeDash=[3, 3]).encode(y="y:Q")
        return (band + line + rules + text + half).properties(height=280, width="container")
    uc.render_chart(build, fallback=df)
    hit = next((r["dte"] for r in dte_timeline_rows() if r["captured"] >= 0.5), None)
    st.caption(_md(f"Model only (flat 20% IV, price unchanged): the 50% target is reached at about "
                   f"{hit} DTE if nothing moves; real trades hit it earlier or later with price and IV. "
                   f"Shaded = the 30-60 DTE entry window. Rules: ") + " ".join(
        _ref_link(s, refs) for s in ("page:concepts-strangle", "ep:market-measures-2024-06-17@08:59")))

    st.markdown(section("Decision trees", 0), unsafe_allow_html=True)
    if m.get("note"):
        st.caption(_linked(m["note"], refs))
    trees = m.get("trees", [])
    if not trees:
        st.caption("No decision trees.")
        return
    st.markdown(uc.chip("decision", THEME.mustard) + uc.chip("action", THEME.teal)
                + uc.chip("info", THEME.muted), unsafe_allow_html=True)
    tabs = st.tabs([t["title"] for t in trees])
    for tab, t in zip(tabs, trees):
        with tab:
            st.markdown(uc.chip(t.get("kind", "MECHANIC"), KIND_COLORS.get(t.get("kind"), THEME.muted)),
                        unsafe_allow_html=True)
            try:
                st.graphviz_chart(tree_dot(t), use_container_width=False)
            except Exception:
                st.caption("Tree could not be drawn; the steps are listed below.")
            with st.expander("Steps with sources"):
                for n in t.get("nodes", []):
                    st.markdown(f"- *{n['type']}* · {_linked(n['text'], refs)} {_links(n.get('sources'), refs)}")


# ================================================ PORTFOLIO & SIZING ====
def _latest_vix(a: dict) -> tuple[float | None, str]:
    try:
        from ..config import BETA_FILES
        import json
        d = json.loads(BETA_FILES["hedge"].read_text(encoding="utf-8"))
        v = (d.get("vrp") or {}).get("vix")
        if v:
            return float(v), f"CLEAN BETA hedge monitor, {d.get('as_of', '')}"
    except Exception:
        pass
    try:
        rows = a["repro"]["vrp_spx"]["chart_data"]["by_year"]
        return float(rows[-1]["vix"]), f"repro:vrp_spx {rows[-1]['year']} mean"
    except Exception:
        return None, ""


def _portfolio(a: dict, refs: dict) -> None:
    p = a["portfolio"]
    st.markdown(section("Account types", 5), unsafe_allow_html=True)
    am = p.get("account_matrix", [])
    if am:
        tabs = st.tabs([r["account"] for r in am])
        for tab, r in zip(tabs, am):
            with tab:
                for k in ("strategies_allowed", "not_allowed", "bpr_basis", "minimums", "notes"):
                    if r.get(k):
                        st.markdown(f"**{k.replace('_', ' ').capitalize()}** · {_linked(r[k], refs)}")

    st.markdown(section("Buying power in use vs VIX", 1, help=help_for("bpr")), unsafe_allow_html=True)
    vix, vix_src = _latest_vix(a)
    bp = (p.get("allocation") or {}).get("bp_vs_vix", {})

    def build(alt):
        xs = list(range(10, 51))
        df = pd.DataFrame({"vix": xs, "lo": [0.25] * len(xs), "hi": [0.5] * len(xs)})
        band = alt.Chart(df).mark_area(color=THEME.teal, opacity=0.22).encode(
            x=alt.X("vix:Q", title="VIX"), y=alt.Y("lo:Q", title="share of net liq in short premium",
                                                  axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
            y2="hi:Q", tooltip=[alt.Tooltip("lo:Q", format=".0%"), alt.Tooltip("hi:Q", format=".0%")])
        step = alt.Chart(pd.DataFrame({"vix": [15, 50], "v": [0.30, 0.30]})).mark_line(
            color=THEME.mustard, strokeWidth=2, interpolate="step-after").encode(x="vix:Q", y="v:Q")
        cap = alt.Chart(pd.DataFrame({"y": [0.75]})).mark_rule(color=THEME.coral, strokeDash=[5, 3]).encode(y="y:Q")
        layers = band + step + cap
        if vix:
            layers += alt.Chart(pd.DataFrame({"x": [vix]})).mark_rule(color=THEME.text).encode(x="x:Q")
        return layers.properties(height=260, width="container")
    uc.render_chart(build)
    st.caption(_md("Teal band = tastylive's 25-50%; coral = 75% maximum; mustard = about 30% with VIX "
                   "above 15; white = latest VIX"
                   + (f" ({vix:.1f}, {vix_src})" if vix else "") + ". No VIX-tiered ladder is drawn "
                   "because none could be verified.") + " " + _links(
        ["ep:options-jive-2023-12-27@06:56", "lit:tastylive-allocation-2022"], refs))
    if bp.get("notes"):
        st.markdown(_linked(bp["notes"], refs))

    _calculator(a, refs, vix)

    st.markdown(section("Sleeve playbook", 2), unsafe_allow_html=True)
    for q in p.get("sleeve_playbook", []):
        with st.expander(q["question"]):
            st.markdown(_linked(q["answer"], refs))
            bs = q.get("by_strategy")
            if isinstance(bs, dict):
                for k, v in bs.items():
                    st.markdown(f"- **{_md(k)}**: {_linked(v, refs)}")
            elif isinstance(bs, list):
                for v in bs:
                    st.markdown(f"- {_linked(v if isinstance(v, str) else ' · '.join(map(str, v.values())), refs)}")

    st.markdown(section("Account-size tiers", 3), unsafe_allow_html=True)
    tiers = p.get("account_tiers", [])
    if tiers:
        cols = st.columns(len(tiers))
        for col, t in zip(cols, tiers):
            with col, st.container(border=True):
                st.markdown(f"**{_md(t['tier'])}**")
                for k in ("practical", "constraints", "contracts"):
                    if t.get(k):
                        st.markdown(f"*{k}* · {_linked(t[k], refs)}")

    alloc = p.get("allocation") or {}
    with st.expander("Allocation guidance: per trade, positions, theta, beta-weighted delta, correlation, stress"):
        for k in ("per_trade", "positions", "theta_nlv", "beta_weighted_delta", "correlation", "stress"):
            if alloc.get(k):
                st.markdown(f"**{k.replace('_', ' ')}** · {_linked(alloc[k].get('guidance', ''), refs)}")
    with st.expander("Taxes"):
        for t in p.get("taxes", []):
            st.markdown(f"- **{_md(t.get('topic'))}**: {_linked(t.get('point'), refs)}")
    with st.expander(f"Unverified items ({len(p.get('unverified', []))})"):
        for u in p.get("unverified", []):
            st.markdown(f"- **{_md(u.get('item'))}**: {_md(u.get('why'))}")


@st.fragment
def _calculator(a: dict, refs: dict, vix: float | None) -> None:
    st.markdown(section("Sizing calculator", 4), unsafe_allow_html=True)
    c = st.columns(4)
    nlv = c[0].number_input("Net liquidation value ($)", min_value=0.0, value=100_000.0, step=5_000.0,
                            key="vault_calc_nlv")
    sleeve = c[1].slider("Options sleeve (% of net liq)", 0, 100, 35, key="vault_calc_sleeve")
    acct = c[2].selectbox("Account type", list(sz.ACCOUNTS), index=2, key="vault_calc_acct",
                          format_func=lambda k: sz.ACCOUNTS[k])
    v = c[3].number_input("VIX", min_value=0.0, value=float(round(vix, 1)) if vix else 20.0, step=0.5,
                          key="vault_calc_vix")
    r = sz.calculate(nlv, sleeve, acct, v)
    if not r.get("nlv"):
        for w in r["warnings"]:
            st.warning(w)
        return
    lo, hi = r["bp_range"]
    pl, ph = r["per_trade"]
    items = [
        {"label": "BP in use", "value": f"{uc.fmt_money(lo)}–{uc.fmt_money(hi)}",
         "sub": f"{r['bp_pct'][0]:.0%}–{r['bp_pct'][1]:.0%} of net liq · max {uc.fmt_money(r['bp_max'])}",
         "color": THEME.teal},
        {"label": "max BPR per trade", "value": f"{uc.fmt_money(pl)}–{uc.fmt_money(ph)}",
         "sub": "1-5% of net liq"},
        {"label": "positions", "value": f"{r['positions'][0]}–{r['positions'][1]}",
         "sub": "band ÷ per-trade size"},
        {"label": "theta / day", "value": f"~{uc.fmt_money(r['theta_day'])}",
         "sub": "0.1% of net liq yardstick (one episode)"},
    ]
    if r["stress_cap"]:
        items.append({"label": "undefined-risk index BPR", "value":
                      f"{uc.fmt_money(r['stress_cap'][0])}–{uc.fmt_money(r['stress_cap'][1])}",
                      "sub": "8-17% stress cap (our arithmetic)", "color": THEME.coral})
    st.markdown(uc.numeric_slab(items, min_width=150), unsafe_allow_html=True)
    if r.get("lean"):
        st.caption(_md(f"With VIX {v:.1f}: {r['lean']}."))
    for w in r["warnings"]:
        st.warning(_linked(w, refs), icon="⚠️")
    st.caption("Sources: " + _links(r["sources"], refs))


# ========================================================== GLOSSARY ====
@st.fragment
def _glossary(a: dict, refs: dict) -> None:
    terms = a["glossary"].get("terms", [])
    st.markdown(section(f"Glossary · {len(terms)} terms", 0), unsafe_allow_html=True)
    q = st.text_input("Search terms and definitions", key="vault_gloss_q")
    if q.strip():
        ql = q.lower()
        terms = [t for t in terms if ql in t["term"].lower() or ql in t["definition"].lower()]
    st.caption(f"{len(terms)} shown. The same definitions appear as ? tooltips across this tab.")
    names = {c["slug"]: c["name"] for c in a["strategies"].get("cards", [])}
    for t in terms[:120]:
        rel = [names[r] for r in t.get("related", []) if r in names]
        st.markdown(f"**{_md(t['term'])}** · {_linked(t['definition'], refs)} "
                    + (f"<small>(cards: {_md(', '.join(rel))})</small> " if rel else "")
                    + _links(t.get("sources"), refs), unsafe_allow_html=True)
    if len(terms) > 120:
        st.caption(f"{len(terms) - 120} more: narrow the search.")


# =================================================== SOURCES & METHOD ====
def _sources(a: dict, refs: dict) -> None:
    cov = a["coverage"]
    st.markdown(section("Corpus coverage by show and year", 1), unsafe_allow_html=True)
    rows = [{"show": s.get("name", s["show"]), "year": y["year"], "episodes": y["episodes"],
             "transcripts": y.get("transcript", 0), "transcript %": y.get("transcript_pct", 0.0)}
            for s in cov.get("shows", []) for y in s.get("years", [])]
    if rows:
        df = pd.DataFrame(rows)

        def build(alt):
            base = alt.Chart(df).encode(x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
                                        y=alt.Y("show:N", title=None, axis=alt.Axis(labelLimit=0)))
            rect = base.mark_rect().encode(
                color=alt.Color("transcript %:Q", title="transcript %",
                                scale=alt.Scale(range=[THEME.panel, THEME.teal], domain=[0, 100])),
                tooltip=["show", "year", "episodes", "transcripts", "transcript %"])
            text = base.mark_text(fontSize=9, color=THEME.text).encode(text="episodes:Q")
            return (rect + text).properties(height=34 * df["show"].nunique() + 30, width="container")
        uc.render_chart(build, fallback=df)
        tr = cov.get("transcripts", {})
        st.caption(f"Numbers = episodes; colour = share with a transcript. {tr.get('n', 0)} transcripts "
                   f"(all YouTube auto captions). tastylive.com is metadata-only for episodes older than "
                   f"about three months, and Market Measures has no YouTube copy for 2012-2022.")

    st.markdown(section("Method", 2), unsafe_allow_html=True)
    st.markdown(_md(
        "1. **Harvest**: the full tastylive episode inventory (5,515 rows) matched to the @tastyliveshow "
        "YouTube channel; 695 auto-caption transcripts; 89 Learn Center pages.\n"
        "2. **Extract**: a frozen taxonomy (41 strategies, 48 topics); a 20-episode pilot read by "
        "subagents with every number grounded against the transcript within ±60 s (96% grounded, 0 "
        "absent). The full episode run was replaced by tastylive's written Trader Resources.\n"
        "3. **Literature**: 12 topics, 209 papers, each DOI checked against Crossref; 'partial' marks "
        "numbers taken from secondary summaries.\n"
        "4. **Reproduce**: 7 tests on free data (Cboe, Yahoo, FRED) with the verdict coded in each build.\n"
        "5. **Synthesize**: cards, edges, trees and the portfolio playbook, graded A/B/C; integrity "
        "check: every reference resolves."))
    st.markdown(section("Evidence rubric", 3, help=help_for("tier")), unsafe_allow_html=True)
    st.markdown(_md(
        "- **A**: replicated in peer-reviewed or major practitioner research across periods and markets, "
        "and consistent with tastylive data and our reproduction where one exists.\n"
        "- **B**: documented but sensitive to period, costs or implementation; or tastylive data plus one "
        "independent source.\n"
        "- **C**: tastylive-only, small sample, period-specific, or contradicted elsewhere.\n"
        "- **CONVENTIONs** are rules of thumb, labelled with their effect on the P&L distribution."))
    st.markdown(section("Limitations", 4), unsafe_allow_html=True)
    st.markdown(uc.note_strip("Read the numbers with these in mind", [
        "tastylive studies are mostly SPY/SPX with mid-price fills and unpublished methodology, often "
        "over short windows.",
        "Our reproductions are model-priced (Black-Scholes off VIX, no skew, no costs) or index-based; "
        "they show shapes, not fills.",
        "Short premium has fat left tails: 2008, Feb 2018, Mar 2020. A high win rate is not safety.",
        "Episode evidence is limited to 20 checked pilot extracts; most cards rest on written pages.",
        "All artefacts are drafts. Rules changed over 2011-2026; the card back records when."]),
        unsafe_allow_html=True)
    st.markdown(section("Copyright", 5), unsafe_allow_html=True)
    st.caption("Transcripts and page text are third-party material: they stay in a local, gitignored "
               "cache and are never displayed. This tab shows paraphrased takeaways and numbers only, "
               "each linked to its source, with at most one short attributed phrase per card.")
    st.markdown(section("Tools to act elsewhere", 0), unsafe_allow_html=True)
    st.markdown("[OptionStrat](https://optionstrat.com) — build and price the structure on a live chain · "
                "[Market Chameleon](https://marketchameleon.com) — IV rank, earnings dates and "
                "expected moves. OPTIONS VAULT has no live quotes by design.")
    st.caption(DISCLAIMER)
    b = a["build"]
    if b:
        st.caption(f"Build {b.get('generated', '—')}: {b.get('counts', {}).get('refs', 0)} source "
                   f"references resolved, {len(b.get('problems', []))} problems; rolodex check "
                   f"{b.get('rolodex_check', '—')}.")

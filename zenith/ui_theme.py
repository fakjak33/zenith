"""ZENITH v2 theme: institutional monospace on pure black.

Page chrome only (global CSS, section headers, stamps, evidence badges, research
cards, help badges). Colours and fonts come from THEME; the component kit lives
in ui_kit, the brand mark and illustrations in art, the Altair theme in ui_altair.
"""

from __future__ import annotations

import html
from pathlib import Path

from .config import THEME

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _ROOT / "assets"
T = THEME

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, .stApp, [class*="css"], input, button, select, textarea, .stMarkdown, p, label, span,
div, td, th, li {{ font-family: {T.font_body} !important; }}
[data-testid="stIconMaterial"], span.material-icons, [class*="material-symbols"], [class*="material-icons"] {{
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important; }}
.stApp {{ background: {T.bg}; color: {T.text}; }}
.stApp, .stMarkdown, p, label, span, div {{ color: {T.text}; }}
p, li {{ font-size: 0.86rem; line-height: 1.6; }}
h1, h2, h3, h4 {{ font-family: {T.font_display} !important; font-weight: 300 !important; color: {T.text};
    letter-spacing: 0.04em; }}
::selection {{ background: {T.teal}; color: #000; }}
.block-container {{ padding: 3.4rem 2rem 3rem 2rem; max-width: 1640px; }}
header[data-testid="stHeader"] {{ background: {T.bg}; border-bottom: none; }}

/* --- sidebar: the navigation rail ------------------------------------------------ */
section[data-testid="stSidebar"] {{ background: {T.bg}; border-right: 1px solid {T.line}; }}
section[data-testid="stSidebar"] > div {{ background: {T.bg}; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 0 !important; padding: 0.3rem 0.8rem !important; }}
[data-testid="stSidebarNavLink"] span {{ font-size: 0.72rem !important; letter-spacing: 0.12em; text-transform: uppercase;
    color: {T.muted} !important; }}
[data-testid="stSidebarNavLink"]:hover {{ background: {T.raised} !important; }}
[data-testid="stSidebarNavLink"]:hover span {{ color: {T.text} !important; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: linear-gradient(90deg, {T.teal}26, transparent) !important;
    box-shadow: inset 2px 0 0 {T.teal}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: {T.text} !important; }}
[data-testid="stNavSectionHeader"] {{ font-size: 0.6rem !important; letter-spacing: 0.24em !important;
    color: {T.faint} !important; text-transform: uppercase; }}
[data-testid="stNavSectionHeader"] * {{ color: {T.faint} !important; }}
[data-testid="stSidebarHeader"] {{ height: auto !important; padding: 1.1rem 1rem 0.6rem 1rem !important; }}
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] {{ height: 3.6rem !important; width: auto !important;
    max-width: 100% !important; }}

/* --- brand --------------------------------------------------------------------- */
.z-lockup {{ display: flex; align-items: center; gap: 0.9rem; }}
.z-word {{ font-weight: 300; letter-spacing: 0.55em; color: {T.text}; white-space: nowrap; }}
.z-tag {{ font-size: 0.6rem; letter-spacing: 0.32em; color: {T.muted}; margin-top: 0.3rem; }}
.z-art {{ display: block; }}
.z-hero {{ position: relative; height: 220px; border: 1px solid {T.line}; overflow: hidden; margin-bottom: 1rem; }}
.z-hero .z-art {{ position: absolute; inset: 0; }}
.z-hero-copy {{ position: absolute; left: 1.6rem; top: 1.3rem; max-width: 46%; }}
.z-hero-copy .k {{ font-size: 0.62rem; letter-spacing: 0.3em; color: {T.muted}; }}
.z-hero-copy .h {{ font-size: 1.9rem; font-weight: 300; line-height: 1.15; letter-spacing: 0.04em; margin: 0.5rem 0; }}
.z-hero-copy .p {{ font-size: 0.8rem; color: {T.muted}; line-height: 1.55; }}
@media (prefers-reduced-motion: reduce) {{ .z-mark animateTransform {{ display: none; }}
    .z-tape > div {{ animation: none !important; }} }}
@media (max-width: 900px) {{ .z-hero-copy {{ max-width: 90%; }} }}

/* --- top bar --------------------------------------------------------------------- */
.z-topbar {{ display: grid; grid-template-columns: minmax(9rem, 15rem) 1fr auto; gap: 1.2rem; align-items: center;
    border-bottom: 1px solid {T.line}; padding: 0 0 0.75rem 0; margin: -0.6rem 0 1.1rem 0; }}
.z-crumb {{ font-size: 0.66rem; letter-spacing: 0.24em; color: {T.muted}; text-transform: uppercase; }}
.z-crumb b {{ color: {T.text}; font-weight: 400; }}
.z-asof {{ font-size: 0.58rem; letter-spacing: 0.18em; color: {T.faint}; text-align: right; }}
.z-asof b {{ color: {T.muted}; font-weight: 400; }}
.z-tkstrip {{ display: flex; gap: 1.3rem; flex-wrap: wrap; }}
.z-tk {{ display: flex; align-items: center; gap: 0.45rem; font-size: 0.72rem; white-space: nowrap; }}
.z-tk b {{ color: {T.text}; font-weight: 600; letter-spacing: 0.06em; }}
.z-tk span {{ color: {T.muted}; }}
@media (max-width: 900px) {{ .z-topbar {{ grid-template-columns: 1fr; }} .z-asof {{ text-align: left; }} }}

/* --- panels: every bordered container -------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ border: 1px solid {T.line} !important; border-radius: 0 !important;
    background: {T.panel} !important; transition: border-color 0.2s ease; }}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{ border-color: {T.faint} !important; }}
.z-ptitle {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.68rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: {T.text}; margin: -0.15rem 0 0.75rem 0; padding-bottom: 0.55rem;
    border-bottom: 1px solid {T.line}; }}
.z-tick {{ width: 2px; height: 0.8rem; background: var(--a); }}
.z-ptag {{ margin-left: auto; color: {T.faint}; font-size: 0.58rem; letter-spacing: 0.16em; }}
.z-pright {{ margin-left: 0.6rem; }}

/* --- section headers ------------------------------------------------------------- */
.parallax-sec {{ display: flex; align-items: center; gap: 0.6rem; margin: 1.4rem 0 0.6rem 0;
    font-size: 0.76rem; letter-spacing: 0.22em; text-transform: uppercase; color: {T.text}; }}
.parallax-sec::before {{ content: ""; width: 2px; height: 0.9rem; background: var(--sec, {T.teal}); flex: none; }}
.parallax-sec::after {{ content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, {T.line}, transparent); }}
.parallax-tag {{ color: {T.muted}; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; }}

/* --- figures ----------------------------------------------------------------------- */
.z-kpis {{ display: flex; flex-wrap: wrap; border: 1px solid {T.line}; background: {T.panel}; }}
.z-kpi {{ flex: 1 1 120px; padding: 0.55rem 0.8rem; border-right: 1px solid {T.line}; position: relative; }}
.z-kpi:last-child {{ border-right: none; }}
.z-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 1px; background: var(--c); opacity: 0.7; }}
.z-lbl {{ font-size: 0.58rem; letter-spacing: 0.2em; color: {T.muted}; text-transform: uppercase; }}
.z-val {{ font-size: 1.55rem; font-weight: 300; line-height: 1.2; margin-top: 0.15rem; white-space: nowrap; }}
.z-sub {{ font-size: 0.64rem; color: {T.faint}; }}
.z-cells {{ display: grid; gap: 2px; }}
.z-cell {{ padding: 0.4rem 0.5rem; min-height: 3.1rem; display: flex; flex-direction: column; justify-content: space-between;
    font-size: 0.64rem; letter-spacing: 0.06em; text-transform: uppercase; }}
.z-cell * {{ color: inherit !important; }}
.z-cell .n {{ font-size: 1.1rem; font-weight: 300; letter-spacing: 0; }}
.z-cell .s {{ font-size: 0.56rem; opacity: 0.85; }}
.z-led {{ display: grid; grid-template-columns: 8.5rem 1fr 3.2rem; gap: 0.6rem; align-items: center;
    font-size: 0.7rem; margin: 0.22rem 0; }}
.z-segs {{ display: grid; grid-template-columns: repeat(24, 1fr); gap: 2px; }}
.z-segs i {{ height: 0.6rem; display: block; }}
.z-dm {{ font-size: 0.62rem; overflow-x: auto; }}
.z-dmr {{ display: flex; align-items: center; gap: 3px; margin: 2px 0; }}
.z-dmr > span {{ width: 5.5rem; flex: none; color: {T.muted}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.z-dmr i {{ display: block; flex: none; }}
.z-dmh .z-dmc {{ width: 11px; writing-mode: vertical-rl; transform: rotate(180deg); color: {T.faint};
    font-size: 0.55rem; height: 4.2rem; overflow: hidden; flex: none; }}
.z-donut {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }}
.z-leg {{ display: grid; grid-template-columns: 0.6rem 1fr auto; gap: 0.5rem; align-items: center; font-size: 0.7rem; margin: 0.22rem 0; min-width: 9rem; }}
.z-leg i {{ width: 0.55rem; height: 0.55rem; display: block; }}
.z-leg b {{ font-weight: 400; color: {T.muted}; }}
.z-dl {{ fill: {T.text}; font-size: 15px; font-weight: 300; }}
.z-ds {{ fill: {T.muted}; font-size: 8.5px; letter-spacing: 0.16em; }}
.z-gv {{ font-size: 19px; font-weight: 300; }}
.z-logs {{ font-size: 0.7rem; max-height: 22rem; overflow-y: auto; }}
.z-log {{ display: grid; grid-template-columns: 5.6rem 5rem 1fr; gap: 0.6rem; padding: 0.22rem 0;
    border-bottom: 1px dashed {T.line}; }}
.z-log .t {{ color: {T.faint}; }}
.z-log .m {{ color: {T.text}; }}
.z-empty {{ display: flex; align-items: center; gap: 1.2rem; padding: 1rem 0; flex-wrap: wrap; }}
.z-empty-msg {{ color: {T.muted}; font-size: 0.8rem; max-width: 34rem; }}
.z-tape {{ overflow: hidden; white-space: nowrap; border-top: 1px solid {T.line}; border-bottom: 1px solid {T.line};
    padding: 0.3rem 0; margin-bottom: 1rem; }}
.z-tape > div {{ display: inline-block; padding-left: 100%; animation: ztape 90s linear infinite; }}
.z-tape span {{ margin-right: 2rem; font-size: 0.7rem; color: {T.muted}; }}
.z-tape b {{ color: {T.text}; font-weight: 400; }}
.z-tape i {{ font-style: normal; }}
@keyframes ztape {{ from {{ transform: translate(0, 0); }} to {{ transform: translate(-100%, 0); }} }}

/* --- research cards ---------------------------------------------------------------- */
.z-card {{ border: 1px solid {T.line}; background: {T.panel}; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
    transition: border-color 0.2s ease; }}
.z-card:hover {{ border-color: {T.teal}; }}
.z-src {{ font-size: 0.58rem; letter-spacing: 0.16em; text-transform: uppercase; }}
.z-title a {{ color: {T.text} !important; text-decoration: none; font-size: 0.88rem; }}
.z-title a:hover {{ color: {T.teal} !important; }}
.z-sum {{ color: {T.muted}; font-size: 0.76rem; margin-top: 0.25rem; }}

/* --- help badge ---------------------------------------------------------------------- */
.z-help {{ display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px;
    border:1px solid {T.faint}; color:{T.muted}; font-size:0.58rem; line-height:1; cursor:help; position:relative;
    margin-left:0.4rem; vertical-align:middle; text-transform:none; letter-spacing:0; }}
.z-help:hover {{ color:{T.teal}; border-color:{T.teal}; }}
.z-help .z-tip {{ visibility:hidden; opacity:0; position:absolute; z-index:60; bottom:150%; left:50%;
    transform:translateX(-50%); width:max-content; max-width:min(340px,72vw); background:{T.raised};
    color:{T.text}; border:1px solid {T.line}; padding:0.5rem 0.65rem; font-size:0.72rem; line-height:1.4;
    letter-spacing:0.01em; text-transform:none; text-align:left; box-shadow:0 8px 28px rgba(0,0,0,0.7);
    transition:opacity 0.12s ease; pointer-events:none; white-space:normal; }}
.z-help:hover .z-tip {{ visibility:visible; opacity:1; }}

/* --- widgets ----------------------------------------------------------------------------- */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{ border-radius: 0 !important;
    border: 1px solid {T.line} !important; background: {T.bg} !important; color: {T.text} !important;
    letter-spacing: 0.14em; text-transform: uppercase; min-height: 2.3rem; }}
.stButton>button p, .stDownloadButton>button p {{ font-size: 0.7rem !important; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color: {T.teal} !important; color: {T.teal} !important; }}
.stButton>button:hover p {{ color: {T.teal} !important; }}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {{ background: {T.teal} !important;
    color: #000 !important; border-color: {T.teal} !important; }}
.stButton>button[kind="primary"] p {{ color: #000 !important; font-weight: 600; }}
.stButton>button[kind="primary"]:hover {{ background: {T.cyan} !important; border-color: {T.cyan} !important; }}
[data-testid="stPageLink"] a {{ border-radius: 0 !important; border: 1px solid {T.line}; padding: 0.15rem 0.6rem !important; }}
[data-testid="stPageLink"] a:hover {{ border-color: {T.teal}; background: {T.raised}; }}
[data-testid="stPageLink"] p {{ font-size: 0.66rem !important; letter-spacing: 0.16em; text-transform: uppercase; }}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stDateInput input {{
    border-radius: 0 !important; border: 1px solid {T.line} !important; background: {T.bg} !important; color: {T.text} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {T.line}; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; border: none; padding: 0.4rem 0; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: {T.muted}; }}
.stTabs [aria-selected="true"] p {{ color: {T.text} !important; }}
.stTabs [aria-selected="true"] {{ box-shadow: inset 0 -2px 0 {T.teal}; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button {{ border-radius: 0 !important; }}
[data-testid="stRadio"] label p {{ font-size: 0.74rem; }}
[data-testid="stMetric"] {{ padding: 0.45rem 0.7rem; border: 1px solid {T.line}; background: {T.panel}; }}
[data-testid="stMetricValue"] {{ font-weight: 300; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.62rem !important; letter-spacing: 0.16em; text-transform: uppercase; color: {T.muted} !important; }}
[data-testid="stExpander"] details {{ border: 1px solid {T.line} !important; border-radius: 0 !important; background: {T.panel}; }}
[data-testid="stExpander"] summary p {{ font-size: 0.72rem; letter-spacing: 0.12em; }}
[data-testid="stAlert"] {{ border-radius: 0 !important; border: 1px solid {T.line}; background: {T.panel}; }}
[data-testid="stCaptionContainer"] p {{ font-size: 0.72rem; color: {T.muted}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {T.line}; margin: 0.4rem 0 1.2rem 0 !important; }}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], [data-testid="stAltairChart"] {{
    margin: 0.4rem 0 1.2rem 0 !important; overflow: visible !important; }}
[data-testid="stVegaLiteChart"] svg, [data-testid="stArrowVegaLiteChart"] svg {{ overflow: visible !important; }}
.vega-embed .mark-line path {{ filter: drop-shadow(0 0 2px rgba(46,196,182,0.22)); }}
</style>
"""

BANNER = ""          # the brand lives in the sidebar (st.logo) and the HOME hero


def _logo_vector(size: int = 76) -> str:
    """The classic Zenith sun mark (kept for the archive and its test): an
    off-centre cream sun radiating open contour rings, clipped to a circle."""
    uid = f"zsun{size}"
    sx, sy = 50, 33
    cream = "#fdf6e3"
    rings = [(18, THEME.navy, 0.95), (26, THEME.navy, 0.85),
             (34, THEME.teal, 0.72), (43, THEME.mint, 0.60),
             (53, THEME.mustard, 0.48), (63, THEME.orange, 0.38),
             (74, THEME.coral, 0.30)]
    circles = "".join(
        f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="none" stroke="{c}" '
        f'stroke-width="3.2" opacity="{o}"/>' for r, c, o in rings)
    halo = (f'<circle cx="{sx}" cy="{sy}" r="11" fill="none" stroke="{cream}" '
            f'stroke-width="1.4" opacity="0.35">'
            f'<animate attributeName="r" values="11;15;11" dur="4.5s" repeatCount="indefinite"/>'
            f'</circle>')
    sun = f'<circle cx="{sx}" cy="{sy}" r="7.5" fill="{cream}" opacity="0.9"/>'
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
            f'<defs><clipPath id="{uid}"><circle cx="50" cy="52" r="46"/></clipPath></defs>'
            f'<g clip-path="url(#{uid})">{circles}{halo}{sun}</g></svg>')


def stamp(as_of: str, page: str = "") -> str:
    """The page's 'data as of' line."""
    pre = f"{html.escape(page)} · " if page else ""
    return (f'<div style="font-size:0.6rem; letter-spacing:0.2em; color:{T.amber}; text-transform:uppercase; '
            f'display:inline-flex; align-items:center; gap:0.5rem; margin:0.1rem 0 0.9rem 0;">'
            f'<span style="width:6px; height:6px; background:{T.amber}; display:inline-block;"></span>'
            f'{pre}data as of {html.escape(str(as_of))}</div>')


def help_badge(text: str) -> str:
    """An inline '?' badge whose hover tooltip shows ``text``."""
    if not text:
        return ""
    safe = html.escape(str(text))
    return f'<span class="z-help">?<span class="z-tip">{safe}</span></span>'


def key_findings(items: list[dict]) -> str:
    """The research a feature is built on: hairline cells with an accent rule.
    items: [{"stat": headline finding, "cite": "Author & Author (Year)"}]."""
    cols = THEME.section_colors
    cards = []
    for i, it in enumerate(items):
        c = cols[i % len(cols)]
        stat = html.escape(str(it.get("stat", "")))
        cite = html.escape(str(it.get("cite", "")))
        cards.append(
            f'<div style="flex:1 1 220px; min-width:190px; border-top:1px solid {c}; '
            f'padding:0.55rem 0.8rem 0.6rem 0; margin-right:1rem;">'
            f'<div style="font-size:0.8rem; line-height:1.5; color:{T.text};">{stat}</div>'
            f'<div style="font-size:0.58rem; letter-spacing:0.16em; color:{c}; margin-top:0.4rem; '
            f'text-transform:uppercase;">{cite}</div></div>')
    return (f'<div style="margin:0.4rem 0 1rem 0;">'
            f'<div style="font-size:0.58rem; letter-spacing:0.24em; color:{T.faint}; text-transform:uppercase; '
            f'margin-bottom:0.3rem;">key findings — the research behind this page</div>'
            f'<div style="display:flex; flex-wrap:wrap;">' + "".join(cards) + '</div></div>')


_TIER_COLORS = {"A": THEME.teal, "B": THEME.amber, "C": THEME.coral}


def evidence_rating(tier: str, label: str, note: str) -> str:
    """Evidence-strength badge: A = replicated live edge, B = documented but
    attenuated, C = faded or exhibit-only. The note is the one-line justification."""
    color = _TIER_COLORS.get(tier[:1].upper(), THEME.muted)
    return (
        f'<div style="display:flex; align-items:center; gap:0.9rem; margin:0.3rem 0 0.8rem 0;">'
        f'<div style="font-size:1.5rem; font-weight:300; line-height:1; color:{color}; '
        f'border:1px solid {color}; padding:0.35rem 0.6rem; min-width:2.6rem; text-align:center;">'
        f'{html.escape(tier)}</div>'
        f'<div><div style="font-size:0.58rem; letter-spacing:0.2em; color:{color}; text-transform:uppercase;">'
        f'evidence strength — {html.escape(str(label))}</div>'
        f'<div style="font-size:0.78rem; color:{T.muted}; line-height:1.5;">{html.escape(str(note))}</div>'
        f'</div></div>')


def section(label: str, idx: int = 0, help: str | None = None) -> str:
    """Section header: accent tick, tracked uppercase label, a hairline to the edge."""
    cols = THEME.section_colors
    c1 = cols[idx % len(cols)]
    badge = help_badge(help) if help else ""
    return (f'<div class="parallax-sec" style="--sec:{c1};">'
            f'<span>{label}</span>{badge}</div>')


def card_html(item: dict) -> str:
    cols = THEME.section_colors
    color = cols[hash(item["source"]) % len(cols)]
    pub = (item.get("published") or "")[:16]
    sm = item.get("summary") or ""
    return (
        f'<div class="z-card">'
        f'<div class="z-src" style="color:{color}">{item["source"]}'
        + (f' · {pub}' if pub else '') + '</div>'
        f'<div class="z-title"><a href="{item["link"]}" target="_blank" rel="noopener">'
        f'{item["title"]}</a></div>'
        + (f'<div class="z-sum">{sm}</div>' if sm else '')
        + '</div>'
    )


# One registered Altair theme for every chart (fonts, axes, grid, colour ranges).
from . import ui_altair as _ui_altair  # noqa: E402

_ui_altair.enable()

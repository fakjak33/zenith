"""Zenith theme: brutalist, vintage-mono, parallax gradient background + logo."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from .config import THEME

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _ROOT / "assets"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&family=Share+Tech+Mono&display=swap');

html, body, .stApp, [class*="css"],
input, button, select, textarea, .stMarkdown, p, label, span, div, td, th {{
    font-family: {THEME.font_body} !important;
}}
[data-testid="stIconMaterial"], span.material-icons, [class*="material-symbols"], [class*="material-icons"] {{
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}}
.stApp {{ background: {THEME.bg}; color: {THEME.text}; }}
.stApp, .stMarkdown, p, label, span, div {{ color: {THEME.text}; }}

/* parallax-depth background: layered fixed palette gradients + slow drift */
.stApp::before {{
    content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none; opacity: 0.45;
    background:
      radial-gradient(40rem 40rem at 12% 8%, {THEME.navy}22, transparent 60%),
      radial-gradient(34rem 34rem at 88% 18%, {THEME.mauve}22, transparent 60%),
      radial-gradient(46rem 46rem at 70% 88%, {THEME.teal}1c, transparent 62%),
      radial-gradient(30rem 30rem at 20% 80%, {THEME.coral}1a, transparent 60%);
    animation: zdrift 40s ease-in-out infinite alternate;
}}
@keyframes zdrift {{ from {{ transform: translate3d(0,0,0) scale(1); }}
                     to {{ transform: translate3d(-2%, -2%, 0) scale(1.06); }} }}
/* faint scanlines */
.stApp::after {{
    content: ""; position: fixed; inset: 0; z-index: 1; pointer-events: none; opacity: 0.18;
    background: repeating-linear-gradient(0deg, rgba(0,0,0,0) 0 2px, rgba(0,0,0,0.22) 3px, rgba(0,0,0,0) 4px);
}}
.block-container {{ position: relative; z-index: 2; }}
section[data-testid="stSidebar"] {{ background: {THEME.panel}; border-right: 2px solid {THEME.border}; z-index: 3; }}

h1, h2, h3, h4 {{ font-family: {THEME.font_display}; color: #fff; }}

.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input, .stDateInput input, div[data-baseweb="select"] {{
    border-radius: 0 !important; border: 2px solid {THEME.border} !important;
    background: {THEME.bg} !important; color: #fff !important;
}}
.stButton>button, .stDownloadButton>button {{
    border-radius: 0 !important; border: 2px solid {THEME.border} !important; background: {THEME.bg};
    color: #fff; font-family: {THEME.font_display};
}}
.stButton>button:hover {{ background: linear-gradient(90deg, {THEME.navy}, {THEME.mauve}, {THEME.coral}); color:#fff; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 16px; border-bottom: 2px solid {THEME.border}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: {THEME.font_display}; letter-spacing: 0.06em; color: {THEME.muted};
    background: {THEME.panel}; border: 2px solid {THEME.border}; border-bottom: none;
    border-radius: 0 !important; padding: 10px 22px;
}}
.stTabs [aria-selected="true"] {{ color: #fff !important;
    background: linear-gradient(120deg, {THEME.navy}, {THEME.mauve}, {THEME.coral}); }}
[data-testid="stTooltipIcon"] svg, label svg {{ border: none !important; background: transparent !important;
    color: {THEME.muted} !important; fill: {THEME.muted} !important; }}

/* item cards */
.z-card {{ border: 2px solid {THEME.border}; background: {THEME.panel}; padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem; }}
.z-src {{ font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; }}
.z-title a {{ color: #fff !important; font-weight: 700; text-decoration: none; }}
.z-title a:hover {{ text-decoration: underline; }}
.z-sum {{ color: {THEME.muted}; font-size: 0.85rem; margin-top: 0.25rem; }}
.parallax-sec {{ font-family: {THEME.font_display}; font-size: 0.9rem; text-transform: uppercase;
    letter-spacing: 0.16em; margin: 0.5rem 0 0.3rem 0; padding: 0.2rem 0.5rem;
    border-left: 4px solid var(--sec, {THEME.teal}); }}
.parallax-tag {{ color: {THEME.muted}; font-size: 0.85rem; letter-spacing: 0.2em; text-transform: uppercase; }}

/* "?" help badge with a hover tooltip (for custom HTML where Streamlit's native
   help= doesn't reach: section headers, cards, metrics) */
.z-help {{ display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px;
    border:1px solid {THEME.muted}; border-radius:50%; color:{THEME.muted}; font-size:0.62rem;
    line-height:1; cursor:help; position:relative; margin-left:0.4rem; vertical-align:middle;
    font-family:{THEME.font_body}; text-transform:none; letter-spacing:0; }}
.z-help:hover {{ color:{THEME.teal}; border-color:{THEME.teal}; }}
.z-help .z-tip {{ visibility:hidden; opacity:0; position:absolute; z-index:60; bottom:150%; left:50%;
    transform:translateX(-50%); width:max-content; max-width:min(340px,72vw); background:{THEME.panel};
    color:{THEME.text}; border:2px solid {THEME.border}; padding:0.5rem 0.65rem; font-size:0.74rem;
    line-height:1.3; letter-spacing:0.01em; text-transform:none; text-align:left; font-weight:400;
    box-shadow:0 6px 24px rgba(0,0,0,0.55); transition:opacity 0.12s ease; pointer-events:none;
    white-space:normal; }}
.z-help:hover .z-tip {{ visibility:visible; opacity:1; }}
</style>
"""


def _logo_svg(size: int = 84) -> str:
    """Modernist 3D mark — nested rotating squares (a gradient vortex).

    Concentric square frames shrink and rotate around the centre, forming a
    spiralling tunnel that reads three-dimensional. A single diagonal palette
    gradient runs through every frame so the whole mark recedes from a light
    mint edge to a deep navy core. Bold, minimalist, basic shapes + rotation +
    depth + gradient. Transparent background.
    """
    gid = f"zv{size}"                               # unique gradient id per render
    deep = "#163f54"
    frames = []
    n = 6
    for i in range(n):
        half = 44 - i * 7                          # 44,37,30,23,16,9 — shrinking
        angle = i * 13                             # progressive rotation = spiral
        w = 4.2 - i * 0.35                         # outer frames a touch bolder
        frames.append(
            f'<rect x="{50-half}" y="{50-half}" width="{2*half}" height="{2*half}" '
            f'fill="none" stroke="url(#{gid})" stroke-width="{w:.2f}" '
            f'transform="rotate({angle} 50 50)"/>'
        )
    inner = "".join(frames)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" shape-rendering="geometricPrecision">'
        f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
        f'x1="6" y1="6" x2="94" y2="94">'
        f'<stop offset="0" stop-color="{THEME.mint}"/>'
        f'<stop offset="0.45" stop-color="{THEME.teal}"/>'
        f'<stop offset="0.8" stop-color="{THEME.navy}"/>'
        f'<stop offset="1" stop-color="{deep}"/></linearGradient></defs>'
        f'{inner}</svg>'
    )


def _logo_anim(size: int = 72) -> str:
    """Animated Zenith mark — a wireframe sphere (longitude/latitude ellipses) that
    rotates, with two semi-transparent trailing copies (phase-offset) for a motion
    trail. Palette gradient stroke. Pure inline SVG + SMIL, so it animates in the
    browser with no JS."""
    gid = f"zsph{size}"
    deep = "#163f54"
    ells = "".join(
        f'<ellipse cx="50" cy="50" rx="{rx}" ry="{ry}" fill="none" '
        f'stroke="url(#{gid})" stroke-width="1.7"/>'
        for rx, ry in [(15, 43), (30, 43), (43, 15), (43, 30)])

    def spin(opacity: float, begin: str) -> str:
        return (f'<g opacity="{opacity}">{ells}'
                f'<animateTransform attributeName="transform" attributeType="XML" '
                f'type="rotate" from="0 50 50" to="360 50 50" dur="11s" '
                f'begin="{begin}s" repeatCount="indefinite"/></g>')

    trails = spin(0.16, "-0.7") + spin(0.40, "-0.35") + spin(1.0, "0")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
        f'x1="6" y1="6" x2="94" y2="94">'
        f'<stop offset="0" stop-color="{THEME.mint}"/>'
        f'<stop offset="0.45" stop-color="{THEME.teal}"/>'
        f'<stop offset="0.8" stop-color="{THEME.navy}"/>'
        f'<stop offset="1" stop-color="{deep}"/></linearGradient></defs>'
        f'<circle cx="50" cy="50" r="43.5" fill="none" stroke="url(#{gid})" '
        f'stroke-width="1" opacity="0.3"/>{trails}</svg>')


def _logo_mark(size: int = 64) -> str:
    """The Zenith disc logo as an <img>.

    Prefers Streamlit's static file server (``static/logo.png`` -> ``app/static``):
    the browser fetches the PNG once and caches it, instead of the ~100 KB base64
    blob being re-sent inline in the banner HTML on every rerun. Falls back to a
    self-contained base64 embed, then to the legacy SVG vortex, if the file is
    missing or static serving is disabled.

    The mark is a smooth gradient outline (no fill), so use normal rendering."""
    sty = "display:block;"
    served = _ROOT / "static" / "logo.png"
    if served.exists():
        return (f'<img src="app/static/logo.png" width="{size}" height="{size}" '
                f'alt="Zenith logo" style="{sty}" />')
    png = _ASSETS / "logo_256.png"
    if png.exists():
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        return (f'<img src="data:image/png;base64,{b64}" width="{size}" '
                f'height="{size}" alt="Zenith logo" style="{sty}" />')
    return _logo_svg(size)


BANNER = f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.3em; position:relative; z-index:2;">
  <div style="line-height:0;">{_logo_anim(76)}</div>
  <div>
    <div style="font-family:{THEME.font_display}; font-size:4.4rem; letter-spacing:0.16em;
                color:#fff; line-height:0.9; -webkit-text-stroke:1px #fff;">ZENITH</div>
    <div class="parallax-tag">daily insights &amp; research aggregator</div>
  </div>
</div>
"""


def stamp(as_of: str, page: str = "") -> str:
    """A prominent 'data as of' banner so every page shows its anchor date."""
    pre = f"{page} · " if page else ""
    return (f'<div style="font-family:{THEME.font_display}; font-size:0.95rem; '
            f'letter-spacing:0.12em; color:{THEME.mustard}; border-left:4px solid {THEME.mustard}; '
            f'padding:0.2rem 0.7rem; margin:0.1rem 0 0.7rem 0; text-transform:uppercase;">'
            f'{pre}data as of {as_of}</div>')


def help_badge(text: str) -> str:
    """An inline '?' badge whose hover tooltip shows ``text``. Safe to embed in any
    HTML rendered with unsafe_allow_html (section headers, cards, metric labels)."""
    if not text:
        return ""
    safe = html.escape(str(text))
    return f'<span class="z-help">?<span class="z-tip">{safe}</span></span>'


def section(label: str, idx: int = 0, help: str | None = None) -> str:
    cols = THEME.section_colors
    c1, c2 = cols[idx % len(cols)], cols[(idx + 1) % len(cols)]
    grad = f"linear-gradient(90deg, {c1}, {c2})"
    badge = help_badge(help) if help else ""
    # gradient clip applies to the label text only; the help badge stays muted
    return (f'<div class="parallax-sec" style="--sec:{c1};">'
            f'<span style="background:{grad}; -webkit-background-clip:text; '
            f'background-clip:text; -webkit-text-fill-color:transparent;">{label}</span>'
            f'{badge}</div>')


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

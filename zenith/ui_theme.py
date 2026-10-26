"""Zenith theme: brutalist, vintage-mono, parallax gradient background + logo."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from .config import THEME
from .ui_term import CSS as _TERM_CSS

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _ROOT / "assets"

CSS = _TERM_CSS.replace("</style>", f"""
/* --- zenith components on the terminal glass --------------------------------- */
.z-card {{ border: 1px solid {THEME.line}; border-left: 3px solid {THEME.phosphor_dim};
    background: linear-gradient(180deg, {THEME.panel}ee, {THEME.glass}ee); padding: 0.65rem 0.9rem;
    margin-bottom: 0.55rem; }}
.z-card:hover {{ border-left-color: {THEME.phosphor}; box-shadow: 0 0 14px {THEME.phosphor}22; }}
.z-src {{ font-family: {THEME.font_label} !important; font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase; }}
.z-title a {{ color: {THEME.text} !important; font-weight: 400; text-decoration: none; font-size: 0.98rem; }}
.z-title a:hover {{ color: {THEME.phosphor} !important; text-shadow: 0 0 8px {THEME.phosphor}88; }}
.z-sum {{ color: {THEME.muted}; font-size: 0.84rem; margin-top: 0.25rem; }}
.parallax-sec {{ font-family: {THEME.font_display} !important; font-size: 1.25rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 0.9rem 0 0.45rem 0; padding: 0.05rem 0.55rem 0.1rem 0;
    display: flex; align-items: center; gap: 0.55rem; color: var(--sec, {THEME.phosphor});
    text-shadow: 0 0 8px color-mix(in srgb, var(--sec, {THEME.phosphor}) 60%, transparent); }}
.parallax-sec::before {{ content: ""; width: 0.6rem; height: 0.95rem; background: var(--sec, {THEME.phosphor});
    box-shadow: 0 0 10px var(--sec, {THEME.phosphor}); flex: none; }}
.parallax-sec::after {{ content: ""; flex: 1; border-bottom: 1px dashed {THEME.line}; margin-left: 0.4rem; }}
.parallax-sec * {{ color: inherit; }}
.parallax-tag {{ color: {THEME.muted}; font-size: 0.8rem; letter-spacing: 0.18em; text-transform: uppercase; }}

/* "?" help badge with a hover tooltip (for custom HTML where Streamlit's native
   help= doesn't reach: section headers, cards, metrics) */
.z-help {{ display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px;
    border:1px solid {THEME.muted}; color:{THEME.muted}; font-size:0.62rem; text-shadow:none;
    line-height:1; cursor:help; position:relative; margin-left:0.4rem; vertical-align:middle;
    font-family:{THEME.font_body}; text-transform:none; letter-spacing:0; }}
.z-help:hover {{ color:{THEME.phosphor}; border-color:{THEME.phosphor}; }}
.z-help .z-tip {{ visibility:hidden; opacity:0; position:absolute; z-index:60; bottom:150%; left:50%;
    transform:translateX(-50%); width:max-content; max-width:min(340px,72vw); background:{THEME.panel};
    color:{THEME.text}; border:1px solid {THEME.phosphor_dim}; padding:0.5rem 0.65rem; font-size:0.74rem;
    line-height:1.3; letter-spacing:0.01em; text-transform:none; text-align:left; font-weight:400;
    box-shadow:0 0 18px {THEME.phosphor}22, 0 6px 24px rgba(0,0,0,0.6); transition:opacity 0.12s ease;
    pointer-events:none; white-space:normal; }}
.z-help:hover .z-tip {{ visibility:visible; opacity:1; }}

/* breathing room around charts & tables so axis labels never crowd/clip */
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"],
.stVegaLiteChart, [data-testid="stAltairChart"] {{
    margin: 0.6rem 0 1.6rem 0 !important; overflow: visible !important;
}}
[data-testid="stVegaLiteChart"] svg, [data-testid="stArrowVegaLiteChart"] svg {{ overflow: visible !important; }}
[data-testid="stDataFrame"] {{ margin: 0.5rem 0 1.4rem 0 !important; }}
[data-testid="stMetric"] {{ padding: 0.35rem 0.6rem; border: 1px solid {THEME.line}; background: {THEME.glass}cc; }}
[data-testid="stMetricValue"] {{ font-family: {THEME.font_display} !important; color: {THEME.phosphor};
    text-shadow: 0 0 8px {THEME.phosphor}66; }}
[data-testid="stExpander"] details {{ border: 1px solid {THEME.line} !important; border-radius: 0 !important;
    background: {THEME.glass}cc; }}
[data-testid="stExpander"] summary:hover {{ color: {THEME.phosphor}; }}
[data-testid="stAlert"] {{ border-radius: 0 !important; border: 1px solid {THEME.line}; }}
.block-container {{ padding-left: 2.2rem; padding-right: 2.2rem; }}
/* top navigation (st.navigation position=top) */
[data-testid="stTopNavLinkContainer"] a, [data-testid="stNavSectionHeader"], header [data-testid="stNavLink"] {{
    font-family: {THEME.font_display} !important; letter-spacing: 0.08em; }}
header[data-testid="stHeader"] {{ background: {THEME.bg}e6; border-bottom: 1px solid {THEME.line};
    backdrop-filter: none; }}
</style>""")


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


def _logo_vector(size: int = 76) -> str:
    """The Zenith sun mark, minimalist cut — an off-centre cream sun radiating
    open contour rings (cool near the sun, warming toward the rim), clipped to
    a circle. Stroked outlines only, no fills, so the background shows through
    the gaps (Parallax-style transparency), and each ring fades with distance
    from the sun. Subtle SMIL sun-glow pulse. Transparent background."""
    uid = f"zsun{size}"
    sx, sy = 50, 33                       # sun sits in the upper third
    cream = "#fdf6e3"
    # (radius, color, opacity) — contour rings radiating outward from the sun
    rings = [(18, "#163f54", 0.95), (26, THEME.navy, 0.85),
             (34, THEME.teal, 0.72), (43, THEME.mint, 0.60),
             (53, THEME.mustard, 0.48), (63, THEME.orange, 0.38),
             (74, THEME.coral, 0.30)]
    circles = "".join(
        f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="none" stroke="{c}" '
        f'stroke-width="3.2" opacity="{o}"/>' for r, c, o in rings)
    halo = (f'<circle cx="{sx}" cy="{sy}" r="11" fill="none" stroke="{cream}" '
            f'stroke-width="1.4" opacity="0.35">'
            f'<animate attributeName="r" values="11;15;11" dur="4.5s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.35;0.08;0.35" dur="4.5s" repeatCount="indefinite"/>'
            f'</circle>')
    sun = (f'<circle cx="{sx}" cy="{sy}" r="7.5" fill="{cream}" opacity="0.9">'
           f'<animate attributeName="r" values="7.5;8.4;7.5" dur="4.5s" repeatCount="indefinite"/>'
           f'</circle>')
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<defs><clipPath id="{uid}"><circle cx="50" cy="52" r="46"/></clipPath></defs>'
        f'<g clip-path="url(#{uid})">{circles}{halo}{sun}</g>'
        f'<circle cx="50" cy="52" r="46" fill="none" stroke="{cream}" '
        f'stroke-width="0.8" opacity="0.25"/></svg>')


def _rot(dur: float, frm: int = 0, to: int = 360, begin: str = "0") -> str:
    return (f'<animateTransform attributeName="transform" attributeType="XML" '
            f'type="rotate" from="{frm} 50 50" to="{to} 50 50" dur="{dur}s" '
            f'begin="{begin}s" repeatCount="indefinite"/>')


def _logo_anim(size: int = 76) -> str:
    """Animated Zenith mark — a multi-layer orrery: a counter-rotating wireframe
    sphere (with a motion trail), an outer dashed ring spinning the other way, three
    orbiting nodes, and a pulsing gradient core. Warm palette gradients (mauve →
    mustard → orange, and navy → mint → mustard). Pure inline SVG + SMIL (no JS)."""
    g1, g2, gc = f"zg1{size}", f"zg2{size}", f"zgc{size}"
    defs = (
        f'<linearGradient id="{g1}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{THEME.mauve}"/>'
        f'<stop offset="0.5" stop-color="{THEME.mustard}"/>'
        f'<stop offset="1" stop-color="{THEME.orange}"/></linearGradient>'
        f'<linearGradient id="{g2}" x1="0" y1="1" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{THEME.navy}"/>'
        f'<stop offset="0.5" stop-color="{THEME.mint}"/>'
        f'<stop offset="1" stop-color="{THEME.mustard}"/></linearGradient>'
        f'<radialGradient id="{gc}">'
        f'<stop offset="0" stop-color="{THEME.mustard}"/>'
        f'<stop offset="0.6" stop-color="{THEME.orange}"/>'
        f'<stop offset="1" stop-color="{THEME.mauve}"/></radialGradient>')

    # wireframe sphere (rotates one way) + 2 phase-offset trail copies
    ells = "".join(f'<ellipse cx="50" cy="50" rx="{rx}" ry="{ry}" fill="none" '
                   f'stroke="url(#{g2})" stroke-width="1.7"/>'
                   for rx, ry in [(16, 40), (31, 40), (40, 16), (40, 31)])
    sphere = "".join(f'<g opacity="{op}">{ells}{_rot(9, begin=bg)}</g>'
                     for op, bg in [(0.18, "-0.7"), (0.45, "-0.35"), (1.0, "0")])

    # outer dashed ring, spinning the OTHER way
    outer = (f'<g><circle cx="50" cy="50" r="46" fill="none" stroke="url(#{g1})" '
             f'stroke-width="2" stroke-dasharray="5 7" opacity="0.9"/>{_rot(18, 360, 0)}</g>')

    # three orbiting nodes (a little planetary system)
    orbit = ("<g>"
             + "".join(f'<g transform="rotate({a} 50 50)">'
                       f'<circle cx="50" cy="8" r="2.6" fill="url(#{g1})"/></g>'
                       for a in (0, 120, 240))
             + _rot(7) + "</g>")

    # pulsing gradient core
    core = (f'<circle cx="50" cy="50" r="7" fill="url(#{gc})">'
            f'<animate attributeName="r" values="6;9.5;6" dur="3.2s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.85;1;0.85" dur="3.2s" repeatCount="indefinite"/>'
            f'</circle>')

    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
            f'<defs>{defs}</defs>{outer}{sphere}{orbit}{core}</svg>')


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
<div class="zt-boot" style="display:flex; align-items:center; gap:1.1rem; position:relative; z-index:2;">
  <div style="line-height:0; filter: drop-shadow(0 0 10px {THEME.phosphor}55);">{_logo_vector(76)}</div>
  <div>
    <div class="zt-logo">ZENITH<b>//</b>TERM</div>
    <div class="zt-status zt-cursor">&gt; research terminal · <b>ONLINE</b> · free data only ·
      evidence-graded · not investment advice</div>
  </div>
</div>
"""


def stamp(as_of: str, page: str = "") -> str:
    """A prominent 'data as of' banner so every page shows its anchor date."""
    pre = f"{html.escape(page)} · " if page else ""
    return (f'<div style="font-family:{THEME.font_label}; font-size:0.66rem; letter-spacing:0.1em; '
            f'color:{THEME.amber}; border:1px solid {THEME.amber}66; border-left:4px solid {THEME.amber}; '
            f'display:inline-block; padding:0.25rem 0.7rem; margin:0.1rem 0 0.8rem 0; text-transform:uppercase; '
            f'background:{THEME.glass}cc; box-shadow:0 0 12px {THEME.amber}22;">'
            f'◉ {pre}data as of {html.escape(str(as_of))}</div>')


def help_badge(text: str) -> str:
    """An inline '?' badge whose hover tooltip shows ``text``. Safe to embed in any
    HTML rendered with unsafe_allow_html (section headers, cards, metric labels)."""
    if not text:
        return ""
    safe = html.escape(str(text))
    return f'<span class="z-help">?<span class="z-tip">{safe}</span></span>'


def key_findings(items: list[dict]) -> str:
    """Always-visible strip of the research findings a feature is built on.

    items: [{"stat": headline finding, "cite": "Author & Author (Year)"}] —
    2-4 short entries; full references belong in the tab's methodology
    expander. Rendered as bordered chips that sit on the parallax background.
    """
    cols = THEME.section_colors
    cards = []
    for i, it in enumerate(items):
        c = cols[i % len(cols)]
        stat = html.escape(str(it.get("stat", "")))
        cite = html.escape(str(it.get("cite", "")))
        cards.append(
            f'<div style="flex:1 1 200px; min-width:180px; border:1px solid {c}55; '
            f'border-top:3px solid {c}; padding:0.45rem 0.6rem; '
            f'background:linear-gradient(180deg, {c}14, {THEME.glass}dd); box-shadow:0 0 14px {c}14;">'
            f'<div style="font-size:0.85rem; line-height:1.38; color:{THEME.text};">{stat}</div>'
            f'<div style="font-family:{THEME.font_label}; font-size:0.6rem; '
            f'letter-spacing:0.08em; color:{c}; margin-top:0.35rem; '
            f'text-transform:uppercase;">» {cite}</div></div>')
    return (f'<div style="margin:0.2rem 0 0.8rem 0;">'
            f'<div style="font-family:{THEME.font_label}; font-size:0.62rem; '
            f'letter-spacing:0.1em; color:{THEME.muted}; text-transform:uppercase; '
            f'margin-bottom:0.35rem;">key findings — the research behind this tab</div>'
            f'<div style="display:flex; gap:0.6rem; flex-wrap:wrap;">'
            + "".join(cards) + '</div></div>')


_TIER_COLORS = {"A": THEME.phosphor, "B": THEME.amber, "C": THEME.coral}


def evidence_rating(tier: str, label: str, note: str) -> str:
    """Evidence-strength badge for a feature or screen segment.

    tier: 'A'/'A-'/'B+'/'B'/'B-'/'C+'/'C' — A = strong live edge (replicated
    recently, significant), B = documented but attenuated/short own-sample,
    C = faded or exhibit-only. The note is the one-line justification with
    the anchor citation; full references live in the methodology expander.
    """
    color = _TIER_COLORS.get(tier[:1].upper(), THEME.muted)
    safe_note = html.escape(str(note))
    safe_label = html.escape(str(label))
    return (
        f'<div style="display:flex; align-items:center; gap:0.6rem; '
        f'margin:0.2rem 0 0.6rem 0;">'
        f'<div style="font-family:{THEME.font_display}; font-size:1.9rem; '
        f'line-height:1; color:{color}; border:2px solid {color}; '
        f'padding:0.2rem 0.55rem; min-width:2.6rem; text-shadow:0 0 10px {color}; '
        f'box-shadow:0 0 14px {color}33, inset 0 0 12px rgba(0,0,0,0.6); background:rgba(0,0,0,0.45); '
        f'text-align:center;">{html.escape(tier)}</div>'
        f'<div><div style="font-family:{THEME.font_label}; font-size:0.62rem; '
        f'letter-spacing:0.08em; color:{color}; text-transform:uppercase;">'
        f'evidence strength — {safe_label}</div>'
        f'<div style="font-size:0.82rem; color:{THEME.text}; opacity:0.85; line-height:1.38;">'
        f'{safe_note}</div></div></div>')


def section(label: str, idx: int = 0, help: str | None = None) -> str:
    """Terminal section header: a glowing block, the label, a dashed rule to the edge."""
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

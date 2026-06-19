"""Zenith theme: brutalist, vintage-mono, parallax gradient background + logo."""

from __future__ import annotations

from .config import THEME

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
</style>
"""


def _logo_svg(size: int = 84) -> str:
    """Zenith mark — modernist / bold / abstract / vintage-tech.

    A solid gradient peak (the 'zenith') sliced by retro synthwave sun-slats,
    crowned by an apex orb at the high point, anchored on a bold horizon bar.
    Hard geometry, no thin lines. Transparent background.
    """
    gid = f"zg{size}"          # keep gradient/clip ids unique per render
    cid = f"zc{size}"
    peak = "50,16 88,84 12,84"
    # retro "sun" slats: bg-colored bars across the peak, thicker toward the base
    slats = "".join(
        f'<rect x="0" y="{y}" width="100" height="{h}" fill="{THEME.bg}"/>'
        for y, h in ((50, 2.6), (57, 3.4), (65, 4.4), (74, 5.6))
    )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs>'
        f'<linearGradient id="{gid}" x1="0" y1="1" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{THEME.coral}"/>'
        f'<stop offset="0.38" stop-color="{THEME.mustard}"/>'
        f'<stop offset="0.7" stop-color="{THEME.mint}"/>'
        f'<stop offset="1" stop-color="{THEME.navy}"/>'
        f'</linearGradient>'
        f'<clipPath id="{cid}"><polygon points="{peak}"/></clipPath>'
        f'</defs>'
        f'<polygon points="{peak}" fill="url(#{gid})"/>'
        f'<g clip-path="url(#{cid})">{slats}</g>'
        f'<circle cx="50" cy="13" r="7" fill="{THEME.mustard}" '
        f'stroke="#ffffff" stroke-width="2.5"/>'
        f'<rect x="6" y="86" width="88" height="5" fill="{THEME.teal}"/>'
        f'</svg>'
    )


BANNER = f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.3em; position:relative; z-index:2;">
  <div style="line-height:0;">{_logo_svg(88)}</div>
  <div>
    <div style="font-family:{THEME.font_display}; font-size:4.4rem; letter-spacing:0.16em;
                color:#fff; line-height:0.9; -webkit-text-stroke:1px #fff;">ZENITH</div>
    <div class="parallax-tag">daily insights &amp; research aggregator</div>
  </div>
</div>
"""


def section(label: str, idx: int = 0) -> str:
    cols = THEME.section_colors
    c1, c2 = cols[idx % len(cols)], cols[(idx + 1) % len(cols)]
    grad = f"linear-gradient(90deg, {c1}, {c2})"
    return (f'<div class="parallax-sec" style="--sec:{c1}; background:{grad}; '
            f'-webkit-background-clip:text; background-clip:text; '
            f'-webkit-text-fill-color:transparent;">{label}</div>')


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

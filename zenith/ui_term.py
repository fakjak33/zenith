"""ZENITH//TERM -- the cyberpunk-terminal look (prototype, branch `redesign`).

A CRT phosphor console: near-black glass with a faint green grid, scanlines and
vignette; every bordered Streamlit container becomes a terminal WINDOW with an
inverse title bar and corner brackets; numbers sit in boxed counters; charts get
a phosphor glow. Zenith's own palette (teal, coral, mustard, mauve, navy) stays
for meaning; phosphor green, amber, cyan and magenta are added for chrome.

Everything here is CSS + small HTML builders (no JS), respects
`prefers-reduced-motion`, and reads colours only from TERM so a later token swap
restyles the lot.
"""

from __future__ import annotations

import html

from .config import THEME


TERM = THEME          # one token set: zenith.config.Theme
T = TERM


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&family=Silkscreen&family=Space+Mono:wght@400;700&display=swap');

:root {{ --ph: {T.phosphor}; --amber: {T.amber}; --cyan: {T.cyan}; --mag: {T.magenta}; }}
html, body, .stApp, [class*="css"], input, button, select, textarea, .stMarkdown, p, label, span,
div, td, th {{ font-family: {T.font_body} !important; }}
[data-testid="stIconMaterial"], span.material-icons, [class*="material-symbols"], [class*="material-icons"] {{
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important; }}

/* --- CRT glass: grid + vignette + scanlines + slow flicker ------------------ */
.stApp {{ background:
    radial-gradient(120% 90% at 50% 0%, {_rgba(T.phosphor, 0.06)}, transparent 55%),
    radial-gradient(140% 120% at 50% 50%, transparent 55%, rgba(0,0,0,0.85) 100%),
    linear-gradient({_rgba(T.grid, 0.55)} 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(90deg, {_rgba(T.grid, 0.55)} 1px, transparent 1px) 0 0 / 34px 34px,
    {T.bg};
    color: {T.text}; }}
.stApp::after {{ content: ""; position: fixed; inset: 0; z-index: 1; pointer-events: none;
    background: repeating-linear-gradient(0deg, rgba(0,0,0,0) 0 2px, rgba(0,0,0,0.28) 3px, rgba(0,0,0,0) 4px);
    animation: ztflicker 6s steps(60) infinite; }}
@keyframes ztflicker {{ 0%, 100% {{ opacity: 0.55; }} 48% {{ opacity: 0.62; }} 50% {{ opacity: 0.45; }} 52% {{ opacity: 0.6; }} }}
.block-container {{ position: relative; z-index: 2; padding-top: 4.4rem; max-width: 1500px; }}
.stApp, .stMarkdown, p, label, span, div {{ color: {T.text}; }}
h1, h2, h3, h4 {{ font-family: {T.font_display}; color: {T.phosphor};
    text-shadow: 0 0 6px {_rgba(T.phosphor, 0.65)}, 0 0 18px {_rgba(T.phosphor, 0.25)}; }}
::selection {{ background: {T.phosphor}; color: #000; }}

/* --- every bordered container is a terminal WINDOW -------------------------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zt-titlebar),
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid {T.line} !important; border-radius: 0 !important;
    background: linear-gradient(180deg, {_rgba(T.panel, 0.92)}, {_rgba(T.glass, 0.92)}) !important;
    box-shadow: inset 0 0 0 1px {_rgba(T.phosphor, 0.06)}, 0 0 24px {_rgba(T.phosphor, 0.06)},
                0 12px 30px rgba(0,0,0,0.6);
    position: relative; }}
.zt-titlebar {{ display: flex; align-items: center; gap: 0.6rem; margin: -1rem -1rem 0.8rem -1rem;
    padding: 0.18rem 0.6rem; background: var(--c, {T.phosphor}); color: #000;
    font-family: {T.font_label} !important; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }}
.zt-titlebar * {{ color: #000 !important; font-family: {T.font_label} !important; }}
.zt-titlebar .zt-tag {{ margin-left: auto; opacity: 0.75; }}
.zt-titlebar .zt-ctl {{ letter-spacing: 0.3em; }}
.zt-corner {{ position: relative; }}

/* --- type ------------------------------------------------------------------- */
.zt-label {{ font-family: {T.font_label} !important; font-size: 0.62rem; letter-spacing: 0.12em;
    color: {T.muted}; text-transform: uppercase; }}
.zt-big {{ font-family: {T.font_display} !important; font-size: 3.4rem; line-height: 0.95; }}
.zt-glow {{ text-shadow: 0 0 6px currentColor, 0 0 20px currentColor; }}
.zt-cursor::after {{ content: "█"; margin-left: 0.15rem; animation: ztblink 1.05s steps(1) infinite; color: {T.phosphor}; }}
@keyframes ztblink {{ 50% {{ opacity: 0; }} }}

/* --- boot header with a hover glitch ---------------------------------------- */
.zt-boot {{ border-bottom: 1px solid {T.line}; padding: 0.2rem 0 0.6rem 0; margin-bottom: 0.6rem; }}
.zt-logo {{ font-family: {T.font_display} !important; font-size: 4.2rem; line-height: 0.9; color: {T.phosphor};
    letter-spacing: 0.08em; position: relative; display: inline-block;
    text-shadow: 0 0 8px {_rgba(T.phosphor, 0.8)}, 0 0 28px {_rgba(T.phosphor, 0.35)}; }}
.zt-logo b {{ color: {T.amber}; text-shadow: 0 0 8px {_rgba(T.amber, 0.8)}; font-weight: 400; }}
.zt-logo:hover {{ animation: ztglitch 0.35s steps(2) 2; }}
@keyframes ztglitch {{
  0% {{ text-shadow: 2px 0 {T.magenta}, -2px 0 {T.cyan}; transform: translate(0); }}
  50% {{ text-shadow: -3px 0 {T.magenta}, 3px 0 {T.cyan}; transform: translate(1px, -1px); clip-path: inset(10% 0 40% 0); }}
  100% {{ text-shadow: 0 0 8px {T.phosphor}; transform: translate(0); clip-path: none; }} }}
.zt-status {{ font-size: 0.78rem; color: {T.muted}; letter-spacing: 0.06em; }}
.zt-status b {{ color: {T.phosphor}; font-weight: 400; }}

/* --- nav bar ---------------------------------------------------------------- */
.zt-nav {{ display: flex; flex-wrap: wrap; gap: 0; border: 1px solid {T.line}; margin: 0.4rem 0 0.9rem 0;
    background: {_rgba(T.panel, 0.9)}; }}
.zt-nav a {{ padding: 0.45rem 1rem; font-family: {T.font_display} !important; font-size: 1.15rem;
    color: {T.muted} !important; text-decoration: none; letter-spacing: 0.08em; border-right: 1px solid {T.line}; }}
.zt-nav a:hover {{ color: #000 !important; background: {T.phosphor}; }}
.zt-nav a.on {{ color: {T.phosphor} !important; box-shadow: inset 0 -3px 0 {T.phosphor};
    text-shadow: 0 0 8px {_rgba(T.phosphor, 0.7)}; }}
.zt-nav .sp {{ margin-left: auto; padding: 0.45rem 1rem; color: {T.amber}; font-size: 0.8rem; align-self: center; }}

/* --- data-stream tape -------------------------------------------------------- */
.zt-tape {{ overflow: hidden; white-space: nowrap; border-top: 1px solid {T.line}; border-bottom: 1px solid {T.line};
    padding: 0.25rem 0; margin-bottom: 0.9rem; background: {_rgba(T.bg, 0.8)}; }}
.zt-tape > div {{ display: inline-block; padding-left: 100%; animation: zttape 70s linear infinite; }}
.zt-tape span {{ margin-right: 2.2rem; font-size: 0.85rem; }}
@keyframes zttape {{ from {{ transform: translate(0, 0); }} to {{ transform: translate(-100%, 0); }} }}

/* --- boxed counters ---------------------------------------------------------- */
.zt-counters {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
.zt-counter {{ flex: 1 1 120px; border: 2px solid var(--c); padding: 0.35rem 0.6rem 0.45rem;
    box-shadow: inset 0 0 18px {_rgba('#000000', 0.6)}, 0 0 14px var(--g); background: rgba(0,0,0,0.45); }}
.zt-counter .v {{ font-family: {T.font_display} !important; font-size: 2.9rem; line-height: 0.95; color: var(--c);
    text-shadow: 0 0 10px var(--c); }}
.zt-counter .s {{ font-size: 0.7rem; color: {T.muted}; }}

/* --- cell heatmap ------------------------------------------------------------ */
.zt-cells {{ display: grid; gap: 3px; }}
.zt-cell {{ padding: 0.3rem 0.35rem; font-size: 0.72rem; border: 1px solid rgba(0,0,0,0.6); color: #000;
    font-family: {T.font_label} !important; text-transform: uppercase; letter-spacing: 0.04em; min-height: 2.6rem;
    display: flex; flex-direction: column; justify-content: space-between; }}
.zt-cell * {{ color: inherit !important; font-family: inherit !important; }}
.zt-cell .n {{ font-family: {T.font_display} !important; font-size: 1.35rem; line-height: 1; }}
.zt-cell:hover {{ outline: 1px solid {T.phosphor}; box-shadow: 0 0 12px {T.phosphor}; }}

/* --- meters ------------------------------------------------------------------ */
.zt-meter {{ display: grid; grid-template-columns: 9rem 1fr 3.4rem; gap: 0.5rem; align-items: center;
    font-size: 0.78rem; margin: 0.18rem 0; }}
.zt-meter .bar {{ height: 0.75rem; background: repeating-linear-gradient(90deg, {_rgba(T.grid, 0.9)} 0 6px, transparent 6px 8px);
    position: relative; border: 1px solid {T.line}; }}
.zt-meter .fill {{ position: absolute; top: 1px; bottom: 1px; background:
    repeating-linear-gradient(90deg, var(--c) 0 6px, transparent 6px 8px); box-shadow: 0 0 8px var(--c); }}
.zt-meter .mid {{ position: absolute; left: 50%; top: -3px; bottom: -3px; border-left: 1px dashed {T.muted}; }}

/* --- widgets ------------------------------------------------------------------ */
.stButton>button, .stDownloadButton>button {{ border-radius: 0 !important; border: 1px solid {T.phosphor} !important;
    background: rgba(0,0,0,0.6) !important; color: {T.phosphor} !important; font-family: {T.font_display} !important;
    font-size: 1.1rem !important; letter-spacing: 0.08em; text-shadow: 0 0 6px {_rgba(T.phosphor, 0.6)}; }}
.stButton>button:hover {{ background: {T.phosphor} !important; color: #000 !important; text-shadow: none;
    box-shadow: 0 0 18px {_rgba(T.phosphor, 0.6)}; }}
.stButton>button[kind="primary"] {{ border-color: {T.amber} !important; color: {T.amber} !important;
    text-shadow: 0 0 6px {_rgba(T.amber, 0.6)}; }}
.stButton>button[kind="primary"]:hover {{ background: {T.amber} !important; color: #000 !important; }}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{ border-radius: 0 !important;
    border: 1px solid {T.line} !important; background: rgba(0,0,0,0.6) !important; color: {T.text} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {T.line}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {T.font_display} !important; font-size: 1.1rem; letter-spacing: 0.08em;
    color: {T.muted}; background: transparent; border: 1px solid {T.line}; border-bottom: none; border-radius: 0 !important;
    padding: 6px 18px; margin-right: -1px; }}
.stTabs [aria-selected="true"] {{ color: #000 !important; background: {T.phosphor} !important;
    box-shadow: 0 0 14px {_rgba(T.phosphor, 0.5)}; }}
.stTabs [aria-selected="true"] p {{ color: #000 !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {T.line}; }}

/* --- charts: phosphor glow on marks ------------------------------------------ */
.vega-embed .mark-line path, .vega-embed .mark-rule line {{ filter: drop-shadow(0 0 3px {_rgba(T.phosphor, 0.55)}); }}
.vega-embed .mark-area path {{ filter: drop-shadow(0 0 6px {_rgba(T.cyan, 0.25)}); }}
.vega-embed .mark-rect path {{ filter: drop-shadow(0 0 2px rgba(0,0,0,0.6)); }}

@media (prefers-reduced-motion: reduce) {{
  .stApp::after, .zt-tape > div, .zt-cursor::after, .zt-logo:hover {{ animation: none !important; }}
}}
</style>
"""


# ------------------------------------------------------------------ builders --

def titlebar(title: str, color: str = T.phosphor, tag: str = "") -> str:
    """Inverse title strip; put it first inside st.container(border=True)."""
    return (f'<div class="zt-titlebar" style="--c:{color};"><span>▣ {html.escape(title)}</span>'
            f'<span class="zt-tag">{html.escape(tag)}</span><span class="zt-ctl">_□✕</span></div>')


def boot(status: list[tuple[str, str]], subtitle: str = "") -> str:
    parts = " · ".join(f"{html.escape(k)} <b>{html.escape(v)}</b>" for k, v in status)
    return (f'<div class="zt-boot"><div class="zt-logo">ZENITH<b>//</b>TERM</div>'
            f'<div class="zt-status zt-cursor">{html.escape(subtitle)} {parts}</div></div>')


def nav(groups: list[str], active: str, right: str = "") -> str:
    links = "".join(f'<a class="{"on" if g == active else ""}">{html.escape(g)}</a>' for g in groups)
    return f'<div class="zt-nav">{links}<span class="sp">{html.escape(right)}</span></div>'


def tape(items: list[tuple[str, float]]) -> str:
    def one(t, v):
        c = T.pos if v >= 0 else T.neg
        return f'<span><b style="color:{T.text}">{html.escape(t)}</b> <span style="color:{c}">{v:+.2f}</span></span>'
    body = "".join(one(t, v) for t, v in items)
    return f'<div class="zt-tape"><div>{body}{body}</div></div>'


def counters(items: list[dict]) -> str:
    cells = []
    for it in items:
        c = it.get("color", T.phosphor)
        cells.append(f'<div class="zt-counter" style="--c:{c}; --g:{_rgba(c, 0.18)};">'
                     f'<div class="zt-label">{html.escape(it["label"])}</div>'
                     f'<div class="v">{html.escape(str(it["value"]))}</div>'
                     f'<div class="s">{html.escape(it.get("sub", ""))}</div></div>')
    return f'<div class="zt-counters">{"".join(cells)}</div>'


def ramp(v: float, cap: float) -> str:
    """Diverging cell colour: coral <- black -> phosphor."""
    x = max(-1.0, min(1.0, v / cap if cap else 0.0))
    if x >= 0:
        a, b = (8, 21, 16), (57, 255, 136)
    else:
        a, b, x = (8, 21, 16), (255, 77, 58), -x
    r, g, bl = (int(a[k] + (b[k] - a[k]) * (0.15 + 0.85 * x)) for k in range(3))
    return f"rgb({r},{g},{bl})"


def cells(rows: list[dict], cols: int = 4, cap: float = 4.0) -> str:
    """rows: [{"name", "value", "sub"}] -> glowing diverging cell grid."""
    out = []
    for r in rows:
        bg = ramp(r["value"], cap)
        fg = "#000" if abs(r["value"]) > cap * 0.25 else T.text
        out.append(f'<div class="zt-cell" style="background:{bg}; color:{fg};">'
                   f'<span>{html.escape(r["name"])}</span><span class="n">{r["value"]:+.2f}</span>'
                   f'<span>{html.escape(r.get("sub", ""))}</span></div>')
    return (f'<div class="zt-cells" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">'
            + "".join(out) + "</div>")


def meters(rows: list[tuple[str, float]], cap: float) -> str:
    """Centered bidirectional meters (segmented LED bars)."""
    out = []
    for name, v in rows:
        x = max(-1.0, min(1.0, v / cap if cap else 0.0))
        c = T.pos if v >= 0 else T.neg
        left, width = (50.0, x * 50.0) if x >= 0 else (50.0 + x * 50.0, -x * 50.0)
        out.append(f'<div class="zt-meter" style="--c:{c};"><span>{html.escape(name)}</span>'
                   f'<div class="bar"><div class="mid"></div><div class="fill" style="left:{left:.1f}%; '
                   f'width:{width:.1f}%;"></div></div><span style="color:{c}; text-align:right;">{v:+.2f}</span></div>')
    return "".join(out)

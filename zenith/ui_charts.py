"""Shared chart + table primitives in the Zenith visual language.

`ui_theme` owns page chrome (CSS, banner, section headers, badges). This module
owns the pieces that sit *inside* a tab: gradient table cells, state banners,
chips, big numeric displays, and the two Altair marks the app leans on most.

House conventions baked in here so callers cannot forget them:
  * every chart is `use_container_width=True` with an explicit pixel height,
    sized per row rather than squeezed;
  * every category axis gets `labelLimit=0` — truncated labels caused a real
    misreading of a Zenith chart once and the rule exists because of it;
  * every chart is wrapped so a failure degrades to a dataframe, never to a
    traceback in the middle of the page.

Every colour here derives from `THEME`, so the ZENITH//TERM token set restyles
all tabs that use these helpers at once.
"""

from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

from .config import THEME

# Panel black is the zero point of every ramp, so "no change" recedes into the
# page instead of competing with the changes that matter.
def rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    """Hex colour t of the way from a to b."""
    ra, rb = rgb(a), rgb(b)
    return "#" + "".join(f"{int(x + (y - x) * t):02x}" for x, y in zip(ra, rb))


# Colour ranges replacing Vega's named schemes (redyellowgreen, tableau20, reds,
# tealblues): coral -> a dim terminal midpoint -> teal, keeping the app's sign
# convention while the midpoint sinks into the glass instead of glowing yellow.
DIV3 = [THEME.coral, mix(THEME.coral, THEME.raised, 0.55), THEME.raised,
        mix(THEME.teal, THEME.raised, 0.55), THEME.teal]
SEQ_TEAL = [mix(THEME.teal, THEME.panel, 0.78), THEME.teal]
SEQ_CORAL = [mix(THEME.coral, THEME.panel, 0.72), THEME.coral]
CAT20 = list(THEME.cat) + [mix(c, THEME.panel, 0.4) for c in THEME.cat] + [THEME.mint, THEME.navy,
                                                                            THEME.orange, THEME.muted]

_PANEL = rgb(THEME.panel)
_TEAL = rgb(THEME.teal)
_CORAL = rgb(THEME.coral)


def grad(v, cap: float, color: tuple[int, int, int]) -> str:
    """Panel-black -> `color` CSS ramp, saturating at |v| = cap."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    a = max(0.0, min(1.0, abs(float(v)) / cap)) if cap else 0.0
    r, g, b = (int(p + (c - p) * a) for p, c in zip(_PANEL, color))
    return f"background-color: rgb({r},{g},{b}); color: {THEME.text};"


def grad_teal(v, cap: float = 1.0) -> str:
    return grad(v, cap, _TEAL)


def grad_coral(v, cap: float = 1.0) -> str:
    return grad(v, cap, _CORAL)


def grad_diverging(v, cap: float = 0.05) -> str:
    """Teal above zero, coral below — the app-wide sign convention."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return grad_teal(v, cap) if float(v) >= 0 else grad_coral(v, cap)


def diverging_scale(cap: float, alt):
    """Altair colour scale matching `grad_diverging`.

    Five stops rather than two so the midpoint really is panel-black: a
    two-stop diverging scheme interpolates through grey and small changes stay
    visible when they should disappear.
    """
    cap = max(float(cap or 0), 1e-9)
    return alt.Scale(
        domain=[-cap, -cap / 2, 0, cap / 2, cap],
        range=[THEME.coral, mix(THEME.coral, THEME.panel, 0.55), THEME.panel,
               mix(THEME.teal, THEME.panel, 0.55), THEME.teal],
        clamp=True,
    )


def fmt_pct(v, digits: int = 1, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


def fmt_money(v, digits: int = 0) -> str:
    """Compact signed dollars: -$3.99bn, +$644m, +$8.0m."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    v = float(v)
    sign = "-" if v < 0 else ""
    a = abs(v)
    for cut, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if a >= cut:
            return f"{sign}${a / cut:,.{digits if cut < 1e9 else 2}f}{suffix}"
    return f"{sign}${a:,.0f}"


def colcfg(cols, help_map: dict) -> dict:
    """`column_config` giving every known column its hover help."""
    return {c: st.column_config.Column(help=help_map[c])
            for c in cols if c in help_map}


def state_banner(color: str, label: str, message: str) -> str:
    """The page's status line: a colour block, the state, and its one-line reason."""
    return (
        f'<div style="display:flex; align-items:center; gap:0.8rem; border:1px solid {THEME.line}; '
        f'border-left:3px solid {color}; background:linear-gradient(90deg, {color}1f, transparent 60%); '
        f'padding:0.55rem 0.9rem; margin:0.2rem 0 0.9rem 0;">'
        f'<span style="font-size:0.95rem; font-weight:300; letter-spacing:0.18em; color:{color}; '
        f'text-transform:uppercase; white-space:nowrap;">{_html.escape(label)}</span>'
        f'<span style="font-size:0.74rem; color:{THEME.muted};">{_html.escape(message)}</span></div>')


def chip(text: str, color: str = THEME.teal, sub: str = "") -> str:
    """Small bordered inline-block tag."""
    tail = (f'<span style="color:{THEME.muted}; margin-left:0.4rem; '
            f'font-size:0.72rem;">{_html.escape(sub)}</span>') if sub else ""
    return (f'<span style="display:inline-block; border:1px solid {color}; '
            f'border-left:4px solid {color}; padding:0.18rem 0.5rem; '
            f'margin:0 0.35rem 0.35rem 0; font-size:0.8rem; color:{THEME.text}; '
            f'background:rgba(0,0,0,0.45); box-shadow:0 0 10px {color}22;">{_html.escape(text)}{tail}</span>')


def numeric_slab(items: list[dict], min_width: int = 150) -> str:
    """Large, quiet numerals — the headline read of a tab.

    `items`: [{"label", "value", "sub"?, "color"?}]. Deliberately not
    `st.metric`: this needs display-face numerals at 2.6rem with the label
    above in small caps, and no delta arrows. Flex-wraps, so on a phone the row
    becomes a column without a media query.
    """
    cells = []
    for it in items:
        c = it.get("color") or THEME.text
        sub = it.get("sub") or ""
        cells.append(
            f'<div style="flex:1 1 {min_width}px; min-width:{min_width}px; '
            f'border-top:2px solid {c}; padding:0.45rem 0.7rem 0.6rem 0;">'
            f'<div style="font-family:{THEME.font_display}; font-size:0.72rem; '
            f'letter-spacing:0.16em; color:{THEME.muted}; '
            f'text-transform:uppercase; margin-bottom:0.15rem;">'
            f'{_html.escape(str(it.get("label", "")))}</div>'
            f'<div style="font-family:{THEME.font_display}; font-size:1.9rem; font-weight:300; '
            f'line-height:1.05; color:{c};">'
            f'{_html.escape(str(it.get("value", "—")))}</div>'
            + (f'<div style="font-size:0.74rem; color:{THEME.muted}; '
               f'margin-top:0.1rem;">{_html.escape(str(sub))}</div>' if sub else "")
            + '</div>')
    return ('<div style="display:flex; gap:1.1rem; flex-wrap:wrap; '
            'margin:0.2rem 0 1.1rem 0;">' + "".join(cells) + '</div>')


def note_strip(title: str, items: list[str]) -> str:
    """Editorial caveat chips — how to read the numbers, not a disclaimer dump."""
    cols = THEME.section_colors
    cards = []
    for i, text in enumerate(items):
        c = cols[i % len(cols)]
        cards.append(
            f'<div style="flex:1 1 230px; min-width:200px; border-left:4px solid {c}; '
            f'padding:0.4rem 0.6rem; background:rgba(0,0,0,0.28);">'
            f'<div style="font-size:0.82rem; line-height:1.4; color:{THEME.text};">'
            f'{_html.escape(text)}</div></div>')
    return (f'<div style="margin:0.2rem 0 1rem 0;">'
            f'<div style="font-family:{THEME.font_display}; font-size:0.8rem; '
            f'letter-spacing:0.14em; color:{THEME.muted}; text-transform:uppercase; '
            f'margin-bottom:0.35rem;">{_html.escape(title)}</div>'
            f'<div style="display:flex; gap:0.6rem; flex-wrap:wrap;">'
            + "".join(cards) + '</div>')


def render_chart(build, fallback: pd.DataFrame | None = None, **kw):
    """Run an Altair builder, degrading to a table if anything goes wrong.

    `build(alt)` receives the altair module and returns a chart. Returns the
    value of `st.altair_chart` (a selection payload when `on_select` is set),
    or None on the fallback path.
    """
    try:
        import altair as alt
        chart = build(alt)
        if chart is None:
            raise ValueError("no chart")
        return st.altair_chart(chart, use_container_width=True, **kw)
    except Exception:
        if fallback is not None and not fallback.empty:
            st.dataframe(fallback, use_container_width=True, hide_index=True)
        return None


def hbar(df: pd.DataFrame, x: str, y: str, *, cap: float, title: str = "",
         tooltip: list | None = None, px_per_row: int = 26,
         min_height: int = 180, fmt: str = "+.2%"):
    """Horizontal diverging bar chart, one row per category."""
    def build(alt):
        return (alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X(f"{x}:Q", title=title or None,
                    axis=alt.Axis(format=fmt, labelLimit=0)),
            y=alt.Y(f"{y}:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=0)),
            color=alt.Color(f"{x}:Q", scale=diverging_scale(cap, alt),
                            legend=None),
            tooltip=tooltip or [y, alt.Tooltip(f"{x}:Q", format=fmt)],
        ).properties(height=max(min_height, px_per_row * max(len(df), 1))))
    return render_chart(build, fallback=df)

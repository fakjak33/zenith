"""ZENITH component kit: small, token-driven HTML/SVG builders for panels and
data displays. Chrome stays quiet; the vintage-terminal ideas live in the data
(segmented meters, dot matrices, block sparklines, log feeds).

Everything returns an HTML string for st.markdown(..., unsafe_allow_html=True)
except `panel`, which returns a bordered Streamlit container with a micro-title.
"""

from __future__ import annotations

import html
import math

import streamlit as st

from .config import THEME as T

TERM = T          # legacy name used by older call sites


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


_rgba = rgba


def _e(x) -> str:
    return html.escape(str(x))


# ------------------------------------------------------------------- panels --

def title(text: str, accent: str = T.teal, tag: str = "", right: str = "") -> str:
    """Micro-title row for a panel: accent tick, uppercase label, optional tag."""
    return (f'<div class="z-ptitle" style="--a:{accent};"><span class="z-tick"></span>'
            f'<span>{_e(text)}</span>'
            + (f'<span class="z-ptag">{_e(tag)}</span>' if tag else "")
            + (f'<span class="z-pright">{right}</span>' if right else "") + "</div>")


titlebar = title   # legacy name


def panel(text: str, accent: str = T.teal, tag: str = ""):
    """A hairline Streamlit container with a micro-title; use as a context manager."""
    box = st.container(border=True)
    box.markdown(title(text, accent, tag), unsafe_allow_html=True)
    return box


# --------------------------------------------------------------- top chrome --

def sparkline_svg(values, color: str = T.teal, w: int = 64, h: int = 18, fill: bool = False) -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = w / (len(vals) - 1)
    pts = " ".join(f"{k * step:.1f},{h - 1 - (v - lo) / span * (h - 2):.1f}" for k, v in enumerate(vals))
    area = (f'<polygon points="0,{h} {pts} {w},{h}" fill="{color}" opacity="0.12"/>' if fill else "")
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" class="z-spark">{area}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3"/></svg>')


def ticker_strip(items: list[dict]) -> str:
    """items: [{"sym", "last", "chg", "spark"}] -> one quiet row (top bar)."""
    cells = []
    for it in items:
        c = T.pos if it["chg"] >= 0 else T.neg
        arrow = "▲" if it["chg"] >= 0 else "▼"
        cells.append(f'<div class="z-tk"><b>{_e(it["sym"])}</b>'
                     f'<span>{it["last"]:,.2f}</span>'
                     f'<span style="color:{c}">{arrow} {it["chg"]:+.2%}</span>'
                     f'{sparkline_svg(it.get("spark", []), c, 54, 16)}</div>')
    return f'<div class="z-tkstrip">{"".join(cells)}</div>'


def topbar(crumb: str, as_of: str, strip_html: str = "") -> str:
    return (f'<div class="z-topbar"><div class="z-crumb">{_e(crumb)}</div>'
            f'<div class="z-tbstrip">{strip_html}</div>'
            f'<div class="z-asof">DATA AS OF<br><b>{_e(as_of)}</b></div></div>')


def tape(items: list[tuple[str, float]]) -> str:
    def one(t, v):
        c = T.pos if v >= 0 else T.neg
        return f'<span><b>{_e(t)}</b> <i style="color:{c}">{v:+.2f}</i></span>'
    body = "".join(one(t, v) for t, v in items)
    return f'<div class="z-tape"><div>{body}{body}</div></div>'


# ------------------------------------------------------------------ figures --

def kpis(items: list[dict]) -> str:
    """[{"label", "value", "sub"?, "color"?}] -> a row of quiet KPI cells."""
    cells = []
    for it in items:
        c = it.get("color", T.text)
        cells.append(f'<div class="z-kpi" style="--c:{c};"><div class="z-lbl">{_e(it["label"])}</div>'
                     f'<div class="z-val" style="color:{c}">{_e(it["value"])}</div>'
                     f'<div class="z-sub">{_e(it.get("sub", ""))}</div></div>')
    return f'<div class="z-kpis">{"".join(cells)}</div>'


counters = kpis   # legacy name


def ramp(v: float, cap: float) -> str:
    """Diverging fill: coral <- near-black -> teal."""
    x = max(-1.0, min(1.0, v / cap if cap else 0.0))
    a = (21, 26, 32)
    b = (46, 196, 182) if x >= 0 else (255, 90, 60)
    x = abs(x)
    r, g, bl = (int(a[k] + (b[k] - a[k]) * (0.12 + 0.88 * x)) for k in range(3))
    return f"rgb({r},{g},{bl})"


def cells(rows: list[dict], cols: int = 4, cap: float = 4.0) -> str:
    """rows: [{"name", "value", "sub"}] -> diverging tile grid."""
    out = []
    for r in rows:
        bg = ramp(r["value"], cap)
        fg = "#000" if abs(r["value"]) > cap * 0.45 else T.text
        out.append(f'<div class="z-cell" style="background:{bg}; color:{fg};">'
                   f'<span>{_e(r["name"])}</span><span class="n">{r["value"]:+.2f}</span>'
                   f'<span class="s">{_e(r.get("sub", ""))}</span></div>')
    return (f'<div class="z-cells" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">'
            + "".join(out) + "</div>")


def led_meter(rows: list[tuple[str, float]], cap: float, segments: int = 24) -> str:
    """Centered, segmented (LED) bidirectional meters."""
    out = []
    half = segments // 2
    for name, v in rows:
        x = max(-1.0, min(1.0, v / cap if cap else 0.0))
        n = round(abs(x) * half)
        c = T.pos if v >= 0 else T.neg
        segs = []
        for k in range(segments):
            on = (k >= half and k < half + n) if v >= 0 else (k < half and k >= half - n)
            segs.append(f'<i style="background:{c if on else T.grid};'
                        f'{"box-shadow:0 0 4px " + rgba(c, 0.5) + ";" if on else ""}"></i>')
        out.append(f'<div class="z-led"><span>{_e(name)}</span><div class="z-segs">{"".join(segs)}</div>'
                   f'<span style="color:{c}; text-align:right;">{v:+.2f}</span></div>')
    return "".join(out)


meters = led_meter   # legacy name


def dot_matrix(rows: list[str], cols: list[str], values: dict, cap: float, size: int = 11) -> str:
    """Heat grid of square dots: values[(row, col)] in [-cap, cap] (None = blank)."""
    head = "".join(f'<span class="z-dmc">{_e(c)}</span>' for c in cols)
    body = []
    for r in rows:
        dots = []
        for c in cols:
            v = values.get((r, c))
            if v is None:
                dots.append(f'<i style="width:{size}px;height:{size}px;background:{T.grid};"></i>')
            else:
                a = min(1.0, abs(v) / cap) if cap else 0
                col = T.pos if v >= 0 else T.neg
                dots.append(f'<i title="{_e(r)} · {_e(c)}: {v:+.2f}" style="width:{size}px;height:{size}px;'
                            f'background:{rgba(col, 0.15 + 0.85 * a)};"></i>')
        body.append(f'<div class="z-dmr"><span>{_e(r)}</span>{"".join(dots)}</div>')
    return f'<div class="z-dm"><div class="z-dmr z-dmh"><span></span>{head}</div>{"".join(body)}</div>'


_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_spark(values) -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in vals)


def ring_donut(parts: list[tuple[str, float, str]], size: int = 150, label: str = "", sub: str = "") -> str:
    """parts: [(name, value, colour)] -> SVG donut with centre label + legend."""
    tot = sum(max(v, 0) for _, v, _ in parts) or 1.0
    r, c, sw = size * 0.36, size / 2, size * 0.12
    circ = 2 * math.pi * r
    off = 0.0
    arcs = []
    for name, v, col in parts:
        frac = max(v, 0) / tot
        arcs.append(f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="{sw:.1f}" '
                    f'stroke-dasharray="{frac * circ:.2f} {circ:.2f}" stroke-dashoffset="{-off * circ:.2f}" '
                    f'transform="rotate(-90 {c} {c})"><title>{_e(name)} {frac:.0%}</title></circle>')
        off += frac
    legend = "".join(f'<div class="z-leg"><i style="background:{col}"></i><span>{_e(n)}</span>'
                     f'<b>{max(v, 0) / tot:.0%}</b></div>' for n, v, col in parts)
    return (f'<div class="z-donut"><svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="{T.grid}" stroke-width="{sw:.1f}"/>'
            f'{"".join(arcs)}<text x="{c}" y="{c - 2}" text-anchor="middle" class="z-dl">{_e(label)}</text>'
            f'<text x="{c}" y="{c + 14}" text-anchor="middle" class="z-ds">{_e(sub)}</text></svg>'
            f'<div>{legend}</div></div>')


def radial_gauge(value: float, lo: float, hi: float, label: str, color: str = T.teal,
                 size: int = 120, fmt: str = "{:.0%}") -> str:
    t = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    r, c = size * 0.38, size / 2
    a0, a1 = math.radians(150), math.radians(150 + 240 * t)
    end = math.radians(390)
    def pt(a):
        return c + r * math.cos(a), c + r * math.sin(a)
    x0, y0 = pt(a0)
    x1, y1 = pt(a1)
    xe, ye = pt(end)
    big = 1 if 240 * t > 180 else 0
    return (f'<svg width="{size}" height="{size * 0.85:.0f}" viewBox="0 0 {size} {size * 0.85:.0f}">'
            f'<path d="M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 1 1 {xe:.1f} {ye:.1f}" fill="none" '
            f'stroke="{T.grid}" stroke-width="7" stroke-linecap="round"/>'
            f'<path d="M {x0:.1f} {y0:.1f} A {r:.1f} {r:.1f} 0 {big} 1 {x1:.1f} {y1:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="7" stroke-linecap="round"/>'
            f'<text x="{c}" y="{c + 4}" text-anchor="middle" class="z-gv" fill="{color}">{_e(fmt.format(value))}</text>'
            f'<text x="{c}" y="{c + 20}" text-anchor="middle" class="z-ds">{_e(label)}</text></svg>')


def log_feed(rows: list[tuple[str, str, str, str]]) -> str:
    """rows: [(stamp, kind, message, colour)] -> terminal-style event list."""
    out = [f'<div class="z-log"><span class="t">{_e(s)}</span><span class="k" style="color:{c}">'
           f'{_e(k.upper())}</span><span class="m">{_e(m)}</span></div>' for s, k, m, c in rows]
    return f'<div class="z-logs">{"".join(out)}</div>'


def empty_state(art_svg: str, message: str) -> str:
    return (f'<div class="z-empty"><div class="z-empty-art">{art_svg}</div>'
            f'<div class="z-empty-msg">{_e(message)}</div></div>')

"""Radial and planar SVG signatures: regime quadrant clock, radar, 24-hour clock,
and a sunburst (partition) for hierarchies."""

from __future__ import annotations

import html
import math

from ..config import THEME as T


def _e(x) -> str:
    return html.escape(str(x))


def _f(x: float) -> str:
    return f"{x:.2f}" if math.isfinite(x) else "0"


# ----------------------------------------------------------- regime quadrant --

QUADS = (("GOLDILOCKS", "growth ↑ inflation ↓", T.teal, 1, -1),
         ("REFLATION", "growth ↑ inflation ↑", T.amber, 1, 1),
         ("STAGFLATION", "growth ↓ inflation ↑", T.coral, -1, 1),
         ("DEFLATION", "growth ↓ inflation ↓", T.violet, -1, -1))


def regime_quadrant(trail: list[tuple[float, float]], size: int = 300, cap: float | None = None,
                    labels: list[str] | None = None) -> str:
    """Growth z (x) vs inflation z (y) plane with a fading trajectory and a pulsing
    current point. trail: oldest -> newest [(growth_z, infl_z)]. The scale auto-fits
    the trail (at least +-0.5) unless `cap` is given."""
    if cap is None:
        span = [abs(v) for g, i in trail for v in (g, i) if v is not None and math.isfinite(v)]
        cap = max(0.5, 1.15 * max(span)) if span else 2.0
    pad = 18
    inner = size - 2 * pad
    c = size / 2

    def xy(g, i):
        gx = max(-cap, min(cap, g)) / cap
        iy = max(-cap, min(cap, i)) / cap
        return c + gx * inner / 2, c - iy * inner / 2

    tiles = []
    for name, sub, col, sx, sy in QUADS:
        x0 = c if sx > 0 else pad
        y0 = pad if sy > 0 else c
        tiles.append(f'<rect x="{x0}" y="{y0}" width="{inner / 2:.1f}" height="{inner / 2:.1f}" fill="{col}" '
                     f'fill-opacity="0.06" stroke="{col}" stroke-opacity="0.35"/>'
                     f'<text x="{x0 + 8:.1f}" y="{y0 + 16:.1f}" class="zv-q" fill="{col}">{name}</text>'
                     f'<text x="{x0 + 8:.1f}" y="{y0 + 28:.1f}" class="zv-qs">{sub}</text>')
    grid = "".join(f'<circle cx="{c}" cy="{c}" r="{inner / 2 * k / 4:.1f}" fill="none" stroke="{T.grid}"/>'
                   for k in (1, 2, 3, 4))
    pts = [xy(g, i) for g, i in trail if math.isfinite(g) and math.isfinite(i)]
    path = ""
    n = len(pts)
    for k in range(1, n):
        (x0, y0), (x1, y1) = pts[k - 1], pts[k]
        op = 0.08 + 0.85 * (k / max(n - 1, 1)) ** 2
        path += (f'<line x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}" stroke="{T.ice}" '
                 f'stroke-opacity="{op:.2f}" stroke-width="1.3"/>')
    dots = "".join(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="1.6" fill="{T.ice}" '
                   f'fill-opacity="{0.15 + 0.6 * (k / max(n - 1, 1)):.2f}"/>' for k, (x, y) in enumerate(pts[:-1]))
    cur = ""
    if pts:
        x, y = pts[-1]
        cur = (f'<circle cx="{_f(x)}" cy="{_f(y)}" r="5" fill="{T.text}"/>'
               f'<circle cx="{_f(x)}" cy="{_f(y)}" r="5" fill="none" stroke="{T.text}">'
               f'<animate attributeName="r" values="5;14;5" dur="2.6s" repeatCount="indefinite"/>'
               f'<animate attributeName="opacity" values="0.9;0;0.9" dur="2.6s" repeatCount="indefinite"/></circle>')
    axes = (f'<line x1="{pad}" y1="{c}" x2="{size - pad}" y2="{c}" stroke="{T.line}"/>'
            f'<line x1="{c}" y1="{pad}" x2="{c}" y2="{size - pad}" stroke="{T.line}"/>'
            f'<text x="{size - pad}" y="{c - 5}" text-anchor="end" class="zv-ax">GROWTH →</text>'
            f'<text x="{c + 5}" y="{pad + 8}" class="zv-ax">INFLATION ↑</text>')
    return (f'<svg class="zv" width="100%" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'{_STYLE}{grid}{"".join(tiles)}{axes}{path}{dots}{cur}</svg>')


# -------------------------------------------------------------------- radar --

def radar(axes: list[str], series: list[tuple[str, list[float], str]], size: int = 300,
          lo: float = -2.0, hi: float = 2.0) -> str:
    """Polar radar: series [(name, values per axis, colour)] on a shared [lo, hi] scale."""
    n = len(axes)
    c, r = size / 2, size / 2 - 42
    if n < 3:
        return ""

    def pt(k, v):
        t = (max(lo, min(hi, v)) - lo) / (hi - lo)
        a = -math.pi / 2 + 2 * math.pi * k / n
        return c + r * t * math.cos(a), c + r * t * math.sin(a)
    rings = "".join(f'<polygon points="{" ".join(f"{_f(c + r * f * math.cos(-math.pi / 2 + 2 * math.pi * k / n))},{_f(c + r * f * math.sin(-math.pi / 2 + 2 * math.pi * k / n))}" for k in range(n))}" '
                    f'fill="none" stroke="{T.grid}"/>' for f in (0.25, 0.5, 0.75, 1.0))
    zero_f = (0 - lo) / (hi - lo)
    zero = (f'<polygon points="{" ".join(f"{_f(c + r * zero_f * math.cos(-math.pi / 2 + 2 * math.pi * k / n))},{_f(c + r * zero_f * math.sin(-math.pi / 2 + 2 * math.pi * k / n))}" for k in range(n))}" '
            f'fill="none" stroke="{T.faint}" stroke-dasharray="2 3"/>')
    spokes, labels = "", ""
    for k, name in enumerate(axes):
        a = -math.pi / 2 + 2 * math.pi * k / n
        x, y = c + r * math.cos(a), c + r * math.sin(a)
        spokes += f'<line x1="{c}" y1="{c}" x2="{_f(x)}" y2="{_f(y)}" stroke="{T.grid}"/>'
        lx, ly = c + (r + 16) * math.cos(a), c + (r + 16) * math.sin(a)
        anchor = "middle" if abs(math.cos(a)) < 0.3 else ("start" if math.cos(a) > 0 else "end")
        labels += f'<text x="{_f(lx)}" y="{_f(ly + 3)}" text-anchor="{anchor}" class="zv-ax">{_e(name)}</text>'
    polys = ""
    for name, vals, col in series:
        p = " ".join(f"{_f(x)},{_f(y)}" for x, y in (pt(k, v) for k, v in enumerate(vals)))
        polys += (f'<polygon points="{p}" fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="1.5">'
                  f'<title>{_e(name)}</title></polygon>')
        polys += "".join(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="2.4" fill="{col}"/>'
                         for x, y in (pt(k, v) for k, v in enumerate(vals)))
    return (f'<svg class="zv" width="100%" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'{_STYLE}{rings}{zero}{spokes}{polys}{labels}</svg>')


# ------------------------------------------------------------- 24-hour clock --

def day_clock(night: float, day: float, label: str, size: int = 150, cap: float = 0.2) -> str:
    """A 24h dial: the overnight arc (16:00 -> 09:30) and the session arc (09:30 -> 16:00),
    each thickened and coloured by its cumulative return."""
    c, r = size / 2, size / 2 - 14

    def arc(h0, h1, val, width):
        a0 = -math.pi / 2 + 2 * math.pi * h0 / 24
        a1 = -math.pi / 2 + 2 * math.pi * h1 / 24
        if a1 < a0:
            a1 += 2 * math.pi
        x0, y0 = c + r * math.cos(a0), c + r * math.sin(a0)
        x1, y1 = c + r * math.cos(a1), c + r * math.sin(a1)
        big = 1 if a1 - a0 > math.pi else 0
        col = T.teal if val >= 0 else T.coral
        w = 2 + width * min(1.0, abs(val) / cap)
        return (f'<path d="M {_f(x0)} {_f(y0)} A {r} {r} 0 {big} 1 {_f(x1)} {_f(y1)}" fill="none" '
                f'stroke="{col}" stroke-width="{w:.1f}" stroke-linecap="butt"/>')
    ticks = "".join(f'<line x1="{_f(c + (r - 4) * math.cos(-math.pi / 2 + 2 * math.pi * h / 24))}" '
                    f'y1="{_f(c + (r - 4) * math.sin(-math.pi / 2 + 2 * math.pi * h / 24))}" '
                    f'x2="{_f(c + (r + 4) * math.cos(-math.pi / 2 + 2 * math.pi * h / 24))}" '
                    f'y2="{_f(c + (r + 4) * math.sin(-math.pi / 2 + 2 * math.pi * h / 24))}" stroke="{T.faint}"/>'
                    for h in range(0, 24, 3))
    return (f'<svg class="zv" width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'{_STYLE}<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{T.grid}" stroke-width="2"/>{ticks}'
            f'{arc(16, 9.5, night, 9)}{arc(9.5, 16, day, 9)}'
            f'<text x="{c}" y="{c - 4}" text-anchor="middle" class="zv-big">{_e(label)}</text>'
            f'<text x="{c}" y="{c + 10}" text-anchor="middle" class="zv-qs">night {night:+.1%}</text>'
            f'<text x="{c}" y="{c + 21}" text-anchor="middle" class="zv-qs">day {day:+.1%}</text></svg>')


# ----------------------------------------------------------------- sunburst --

def sunburst(tree: list[dict], size: int = 420, cap: float = 10.0, rings: int = 3) -> str:
    """Partition rings. tree: [{"name", "value", "color_value", "children": [...]}].
    Angle = value share, colour = colour_value on the diverging teal/coral ramp."""
    c = size / 2
    ring_w = (c - 30) / rings
    out = []

    def colour(v):
        x = max(-1.0, min(1.0, (v or 0) / cap))
        base, tgt = (21, 26, 32), ((46, 196, 182) if x >= 0 else (255, 90, 60))
        x = abs(x)
        return "rgb({},{},{})".format(*(int(base[k] + (tgt[k] - base[k]) * (0.15 + 0.85 * x)) for k in range(3)))

    def walk(nodes, depth, a0, a1):
        tot = sum(max(n.get("value", 0), 0) for n in nodes) or 1.0
        a = a0
        for nd in nodes:
            span = (a1 - a0) * max(nd.get("value", 0), 0) / tot
            if span <= 0.004:                  # hairline slivers add noise, not information
                a += max(span, 0)
                continue
            r0, r1 = 26 + depth * ring_w, 26 + (depth + 1) * ring_w - 1.5
            b = a + span
            big = 1 if span > math.pi else 0
            p = (f"M {_f(c + r0 * math.cos(a))} {_f(c + r0 * math.sin(a))} "
                 f"L {_f(c + r1 * math.cos(a))} {_f(c + r1 * math.sin(a))} "
                 f"A {r1:.1f} {r1:.1f} 0 {big} 1 {_f(c + r1 * math.cos(b))} {_f(c + r1 * math.sin(b))} "
                 f"L {_f(c + r0 * math.cos(b))} {_f(c + r0 * math.sin(b))} "
                 f"A {r0:.1f} {r0:.1f} 0 {big} 0 {_f(c + r0 * math.cos(a))} {_f(c + r0 * math.sin(a))} Z")
            out.append(f'<path d="{p}" fill="{colour(nd.get("color_value"))}" stroke="{T.bg}" stroke-width="1">'
                       f'<title>{_e(nd["name"])} · {nd.get("color_value", 0):+.2f}</title></path>')
            if span > 0.16 and depth < 2:
                mid = a + span / 2
                rr = (r0 + r1) / 2
                rot = math.degrees(mid)
                flip = 180 if 90 < (rot % 360) < 270 else 0
                out.append(f'<text x="{_f(c + rr * math.cos(mid))}" y="{_f(c + rr * math.sin(mid))}" '
                           f'transform="rotate({rot + flip:.1f} {_f(c + rr * math.cos(mid))} {_f(c + rr * math.sin(mid))})" '
                           f'text-anchor="middle" dominant-baseline="middle" class="zv-sb">{_e(nd["name"][:16])}</text>')
            if nd.get("children") and depth + 1 < rings:
                walk(nd["children"], depth + 1, a, b)
            a = b
    walk(tree, 0, -math.pi / 2, 3 * math.pi / 2)
    return (f'<svg class="zv" width="100%" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'{_STYLE}{"".join(out)}</svg>')


_STYLE = (f'<style>.zv-q{{font:600 9px "JetBrains Mono",monospace;letter-spacing:.14em}}'
          f'.zv-qs{{font:8px "JetBrains Mono",monospace;fill:{T.muted}}}'
          f'.zv-ax{{font:8.5px "JetBrains Mono",monospace;fill:{T.faint};letter-spacing:.1em}}'
          f'.zv-big{{font:300 13px "JetBrains Mono",monospace;fill:{T.text}}}'
          f'.zv-sb{{font:8px "JetBrains Mono",monospace;fill:#000;letter-spacing:.04em}}</style>')

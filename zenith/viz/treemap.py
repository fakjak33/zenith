"""Squarified treemap (Bruls, Huizing & van Wijk 2000) rendered as SVG.

Two levels: groups (e.g. sectors) tiled first, then their members inside each
group's rectangle. Area = size, fill = colour value on the teal/coral ramp."""

from __future__ import annotations

import html
import math

from ..config import THEME as T


def _worst(row: list[float], w: float) -> float:
    s = sum(row)
    if s <= 0 or w <= 0:
        return math.inf
    mx, mn = max(row), min(row)
    return max(w * w * mx / (s * s), (s * s) / (w * w * mn))


def squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Rectangles (x, y, w, h) for positive `values` (same order) filling the box."""
    vals = [max(float(v), 0.0) for v in values]
    total = sum(vals)
    if total <= 0 or w <= 0 or h <= 0:
        return [(x, y, 0.0, 0.0) for _ in vals]
    scale = w * h / total
    areas = [v * scale for v in vals]
    order = sorted(range(len(areas)), key=lambda k: -areas[k])
    rects: dict[int, tuple] = {}
    queue = [k for k in order if areas[k] > 0]
    for k in order:
        if areas[k] <= 0:
            rects[k] = (x, y, 0.0, 0.0)
    while queue:
        short = min(w, h)
        row = [queue.pop(0)]
        while queue and _worst([areas[i] for i in row + [queue[0]]], short) <= _worst([areas[i] for i in row], short):
            row.append(queue.pop(0))
        s = sum(areas[i] for i in row)
        if w >= h:                       # lay the row down the left side
            cw = s / h if h else 0
            cy = y
            for i in row:
                ch = areas[i] / cw if cw else 0
                rects[i] = (x, cy, cw, ch)
                cy += ch
            x, w = x + cw, w - cw
        else:                            # lay the row across the top
            rh = s / w if w else 0
            cx = x
            for i in row:
                cw = areas[i] / rh if rh else 0
                rects[i] = (cx, y, cw, rh)
                cx += cw
            y, h = y + rh, h - rh
    return [rects[k] for k in range(len(vals))]


def _fill(v: float, cap: float) -> str:
    x = max(-1.0, min(1.0, (v or 0) / cap if cap else 0))
    base, tgt = (18, 22, 28), ((46, 196, 182) if x >= 0 else (255, 90, 60))
    x = abs(x)
    return "rgb({},{},{})".format(*(int(base[k] + (tgt[k] - base[k]) * (0.1 + 0.9 * x)) for k in range(3)))


def treemap(groups: list[dict], width: int = 960, height: int = 460, cap: float = 0.05,
            fmt: str = "{:+.1%}", head: int = 16) -> str:
    """groups: [{"name", "children": [{"name", "size", "value", "sub"?}]}]."""
    gsizes = [sum(max(c.get("size", 0), 0) for c in g.get("children", [])) for g in groups]
    out = []
    for g, (gx, gy, gw, gh) in zip(groups, squarify(gsizes, 0, 0, width, height)):
        if gw < 2 or gh < 2:
            continue
        out.append(f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" fill="none" '
                   f'stroke="{T.bg}" stroke-width="3"/>')
        hh = head if gh > head * 2.5 else 0
        kids = g.get("children", [])
        for c, (x, y, w, h) in zip(kids, squarify([k.get("size", 0) for k in kids], gx + 1.5, gy + hh + 1.5,
                                                   gw - 3, gh - hh - 3)):
            if w < 1 or h < 1:
                continue
            v = c.get("value", 0) or 0
            fg = "#000" if abs(v) > cap * 0.55 else T.text
            label = ""
            if w > 34 and h > 18:
                fs = max(8, min(15, int(min(w / 5.5, h / 2.6))))
                label = (f'<text x="{x + 4:.1f}" y="{y + fs + 2:.1f}" fill="{fg}" font-size="{fs}" '
                         f'class="zv-tm">{html.escape(str(c["name"])[: max(3, int(w / (fs * 0.62)))])}</text>')
                if h > fs * 2 + 10:
                    label += (f'<text x="{x + 4:.1f}" y="{y + fs * 2 + 4:.1f}" fill="{fg}" font-size="{max(8, fs - 3)}" '
                              f'class="zv-tm" opacity="0.8">{html.escape(fmt.format(v))}</text>')
            tip = f'{c["name"]} · {fmt.format(v)}' + (f' · {c["sub"]}' if c.get("sub") else "")
            out.append(f'<g><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{_fill(v, cap)}" '
                       f'stroke="{T.bg}" stroke-width="1"><title>{html.escape(tip)}</title></rect>{label}</g>')
        if hh:
            out.append(f'<text x="{gx + 5:.1f}" y="{gy + 11.5:.1f}" class="zv-tmh">{html.escape(str(g["name"]).upper())}</text>')
    style = (f'<style>.zv-tm{{font-family:"JetBrains Mono",monospace;font-weight:400}}'
             f'.zv-tmh{{font:600 9px "JetBrains Mono",monospace;fill:{T.muted};letter-spacing:.14em}}</style>')
    return (f'<svg class="zv" width="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg">{style}{"".join(out)}</svg>')

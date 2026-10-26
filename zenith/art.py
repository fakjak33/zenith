"""ZENITH brand mark and vector illustrations: pure, deterministic SVG strings.

The mark is a spire (a tall cone, ice -> teal -> amber -> coral) in front of a
faint planet disc, circled by two tilted orbit rings. Each ring is drawn twice:
the full ellipse behind the spire, then only its lower arc in front, so the rings
genuinely pass behind and in front of it. The illustrations (horizon, orbits,
contours, globe) reuse the same palette for hero bands, the game lobby and empty
states. Every colour comes from THEME; animation is a slow ring precession that
CSS switches off under prefers-reduced-motion.
"""

from __future__ import annotations

import math

from .config import THEME as T

_uid = [0]


def _id(prefix: str) -> str:
    _uid[0] += 1
    return f"{prefix}{_uid[0]}"


def _ring(cx, cy, rx, ry, rot, color, width, front: bool, anim: str) -> str:
    if front:
        shape = (f'<path d="M {cx - rx:.2f} {cy:.2f} A {rx} {ry} 0 0 0 {cx + rx:.2f} {cy:.2f}" '
                 f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')
    else:
        shape = (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{color}" '
                 f'stroke-width="{width}" opacity="0.55"/>')
    return f'<g transform="rotate({rot} {cx} {cy})"><g class="z-orbit">{shape}{anim}</g></g>'


def _osc(cx, cy, amp: float, dur: float) -> str:
    return (f'<animateTransform attributeName="transform" type="rotate" '
            f'values="-{amp} {cx} {cy};{amp} {cx} {cy};-{amp} {cx} {cy}" dur="{dur}s" '
            f'repeatCount="indefinite"/>')


def mark(size: int = 48, animate: bool = True) -> str:
    """The spire-and-orbits logo mark (viewBox 100 x 120)."""
    g, p, h = _id("zsp"), _id("zpl"), _id("zhl")
    a1 = _osc(50, 66, 3, 22) if animate else ""
    a2 = _osc(50, 76, 4, 30) if animate else ""
    rings_back = (_ring(50, 66, 42, 9.5, -16, T.amber, 2.2, False, a1)
                  + _ring(50, 76, 34, 7, 13, T.coral, 1.8, False, a2))
    rings_front = (_ring(50, 66, 42, 9.5, -16, T.amber, 2.2, True, a1)
                   + _ring(50, 76, 34, 7, 13, T.coral, 1.8, True, a2))
    w = int(size * 100 / 120)
    return (
        f'<svg class="z-mark" width="{w}" height="{size}" viewBox="0 0 100 120" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Zenith">'
        f'<defs>'
        f'<linearGradient id="{g}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{T.ice}"/><stop offset="0.38" stop-color="{T.teal}"/>'
        f'<stop offset="0.72" stop-color="{T.amber}"/><stop offset="1" stop-color="{T.coral}"/></linearGradient>'
        f'<radialGradient id="{p}" cx="0.4" cy="0.35" r="0.7">'
        f'<stop offset="0" stop-color="{T.amber}" stop-opacity="0.55"/>'
        f'<stop offset="1" stop-color="{T.coral}" stop-opacity="0"/></radialGradient>'
        f'<linearGradient id="{h}" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="#ffffff" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
        f'</defs>'
        f'<circle cx="62" cy="44" r="26" fill="url(#{p})"/>'
        f'{rings_back}'
        f'<polygon points="50,4 31,108 69,108" fill="url(#{g})"/>'
        f'<polygon points="50,4 31,108 47,108" fill="url(#{h})"/>'
        f'<ellipse cx="50" cy="109" rx="22" ry="2.6" fill="{T.coral}" opacity="0.35"/>'
        f'{rings_front}'
        f'</svg>')


def wordmark(size: float = 1.35) -> str:
    return (f'<span class="z-word" style="font-size:{size}rem;">Z&#8202;E&#8202;N&#8202;I&#8202;T&#8202;H</span>')


def lockup(mark_px: int = 44, tagline: bool = True) -> str:
    tag = ('<div class="z-tag">HIGHER PERSPECTIVE.&nbsp;&nbsp;CLEARER SIGNALS.</div>' if tagline else "")
    return (f'<div class="z-lockup">{mark(mark_px)}<div>{wordmark()}{tag}</div></div>')


# ------------------------------------------------------------- illustrations --

def _noise(x: float, seed: int, octaves: int = 4) -> float:
    """Smooth deterministic 1-D noise from a sum of sines."""
    v, amp, fr = 0.0, 1.0, 1.0
    for k in range(octaves):
        ph = (seed * 12.9898 + k * 78.233) % 6.283
        v += amp * math.sin(x * fr + ph) * math.cos(x * fr * 0.37 + ph * 1.7)
        amp *= 0.5
        fr *= 2.13
    return v


def _ridge(w: int, base: float, height: float, seed: int, step: int = 6) -> str:
    pts = []
    for x in range(0, w + step, step):
        y = base - height * (0.55 + 0.45 * _noise(x / 90.0, seed))
        pts.append(f"{x},{y:.1f}")
    return " ".join(pts)


def horizon(w: int = 1200, h: int = 260, seed: int = 7) -> str:
    """Planet with rings rising over contour-line mountains (hero band art)."""
    pg, fade = _id("zhp"), _id("zhf")
    px, py, pr = int(w * 0.74), int(h * 0.44), int(h * 0.36)
    ridges = []
    cols = [T.violet, T.teal, T.cyan, T.teal, T.ice]
    for k in range(9):
        base = h * (0.62 + k * 0.05)
        col = cols[k % len(cols)]
        op = 0.18 + 0.08 * k
        ridges.append(f'<polyline points="{_ridge(w, base, h * (0.32 - k * 0.025), seed + k)} {w},{h} 0,{h}" '
                      f'fill="{T.bg}" fill-opacity="0.92" stroke="{col}" stroke-opacity="{min(op, 0.85):.2f}" '
                      f'stroke-width="1"/>')
    return (
        f'<svg class="z-art" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice" '
        f'xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
        f'<defs><radialGradient id="{pg}" cx="0.35" cy="0.3" r="0.75">'
        f'<stop offset="0" stop-color="{T.mustard}"/><stop offset="0.55" stop-color="{T.amber}"/>'
        f'<stop offset="1" stop-color="{T.coral}"/></radialGradient>'
        f'<linearGradient id="{fade}" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{T.bg}"/><stop offset="0.45" stop-color="{T.bg}" stop-opacity="0.2"/>'
        f'<stop offset="1" stop-color="{T.bg}" stop-opacity="0"/></linearGradient></defs>'
        f'<circle cx="{px}" cy="{py}" r="{pr}" fill="url(#{pg})" opacity="0.92"/>'
        f'<g transform="rotate(-12 {px} {py})" fill="none">'
        f'<ellipse cx="{px}" cy="{py}" rx="{pr * 2.1:.0f}" ry="{pr * 0.42:.0f}" stroke="{T.cyan}" stroke-width="2" opacity="0.7"/>'
        f'<ellipse cx="{px}" cy="{py}" rx="{pr * 2.5:.0f}" ry="{pr * 0.55:.0f}" stroke="{T.coral}" stroke-width="1.4" opacity="0.6"/>'
        f'<ellipse cx="{px}" cy="{py}" rx="{pr * 2.9:.0f}" ry="{pr * 0.66:.0f}" stroke="{T.amber}" stroke-width="1" opacity="0.45"/>'
        f'</g>{"".join(ridges)}'
        f'<rect width="{w}" height="{h}" fill="url(#{fade})"/></svg>')


def orbits(size: int = 220, seed: int = 3) -> str:
    """A ring system: tilted ellipses with satellites, for lobby / empty states."""
    c = size / 2
    rings = []
    cols = [T.teal, T.amber, T.violet, T.coral, T.cyan]
    for k in range(5):
        rx, ry = c * (0.35 + 0.13 * k), c * (0.12 + 0.045 * k)
        rot = -18 + 9 * k
        ang = (seed * 37 + k * 71) % 360
        sx = c + rx * math.cos(math.radians(ang))
        sy = c + ry * math.sin(math.radians(ang))
        rings.append(f'<g transform="rotate({rot} {c} {c})"><ellipse cx="{c}" cy="{c}" rx="{rx:.1f}" '
                     f'ry="{ry:.1f}" fill="none" stroke="{cols[k]}" stroke-width="1" opacity="0.6"/>'
                     f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.4" fill="{cols[k]}"/></g>')
    return (f'<svg class="z-art" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'xmlns="http://www.w3.org/2000/svg"><circle cx="{c}" cy="{c}" r="{c * 0.16:.1f}" '
            f'fill="{T.amber}" opacity="0.85"/>{"".join(rings)}</svg>')


def contours(w: int = 600, h: int = 160, seed: int = 11, lines: int = 14) -> str:
    """Topographic line field (section backgrounds, empty states)."""
    out = []
    for k in range(lines):
        base = h * (k + 1) / (lines + 1)
        pts = " ".join(f"{x},{base + 10 * _noise(x / 70.0 + k * 0.4, seed):.1f}" for x in range(0, w + 8, 8))
        col = T.teal if k % 3 else T.violet
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.8" '
                   f'opacity="{0.12 + 0.03 * (k % 5):.2f}"/>')
    return (f'<svg class="z-art" viewBox="0 0 {w} {h}" preserveAspectRatio="none" width="100%" '
            f'height="{h}" xmlns="http://www.w3.org/2000/svg">{"".join(out)}</svg>')


def grid_globe(size: int = 180, seed: int = 5) -> str:
    """Dot-matrix globe: dots inside a sphere, shaded, with 'land' clusters."""
    c, r, step = size / 2, size * 0.46, size / 34
    dots = []
    y = c - r
    while y <= c + r:
        x = c - r
        while x <= c + r:
            dx, dy = (x - c) / r, (y - c) / r
            d2 = dx * dx + dy * dy
            if d2 <= 1:
                z = math.sqrt(1 - d2)
                land = _noise(dx * 3.1 + 7, seed) + _noise(dy * 2.7 + 3, seed + 1) > 0.35
                col = T.teal if land else T.line
                op = (0.35 + 0.65 * z) if land else 0.5 * z + 0.2
                dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{step * 0.28:.2f}" fill="{col}" '
                            f'opacity="{op:.2f}"/>')
            x += step
        y += step
    return (f'<svg class="z-art" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(dots)}</svg>')


def favicon_png(path, px: int = 128) -> None:
    """Rasterise a simple version of the mark for the browser tab (Pillow)."""
    from PIL import Image, ImageDraw

    s = 4
    im = Image.new("RGBA", (px * s, px * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    W = px * s

    def hexrgb(hx, a=255):
        hx = hx.lstrip("#")
        return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4)) + (a,)
    stops = [hexrgb(T.ice), hexrgb(T.teal), hexrgb(T.amber), hexrgb(T.coral)]
    top, bot = int(W * 0.04), int(W * 0.9)
    for yy in range(top, bot):
        t = (yy - top) / (bot - top)
        seg = min(int(t * 3), 2)
        u = t * 3 - seg
        a, b = stops[seg], stops[seg + 1]
        col = tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(4))
        half = (yy - top) / (bot - top) * W * 0.17
        d.line([(W / 2 - half, yy), (W / 2 + half, yy)], fill=col)
    d.ellipse([W * 0.12, W * 0.48, W * 0.88, W * 0.64], outline=hexrgb(T.amber), width=int(W * 0.03))
    im = im.resize((px, px), Image.LANCZOS)
    im.save(path)

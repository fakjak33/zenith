"""Hand-drawn SVG for the volatility pages: term ribbon, VRP state diagram,
circular dendrogram, sigma-band strips. Token colours only; returns HTML."""

from __future__ import annotations

import html
import math

from ..config import THEME as T

STATE_COLORS = {"ALL-CLEAR": T.teal, "RECOVERING": T.mint, "NEUTRAL": T.muted, "ALARM": T.amber,
                "CONFIRMED": T.coral}
CLASS_COLORS = {"CORE": T.teal, "SURGE": T.amber, "ACTIVE": T.cyan, "FADING": T.violet, "THIN": T.faint}
_FONT = 'font-family="JetBrains Mono, monospace"'


def _e(x) -> str:
    return html.escape(str(x))


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def term_cell(tk: str, iv: dict, ivx: dict, tenors: list[int], state: str, contango: bool,
              kink: bool, w: int = 150, h: int = 64, lo: float | None = None, hi: float | None = None) -> str:
    pts = [(t, iv.get(str(t))) for t in tenors]
    pts = [(t, v) for t, v in pts if _ok(v)]
    if len(pts) < 2:
        return ""
    vals = [v for _, v in pts] + [v for v in (ivx.get(str(t)) for t in tenors) if _ok(v)]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = (hi - lo) or 0.01
    lx = [math.log(t) for t in tenors]
    x0, x1 = lx[0], lx[-1]
    def xy(t, v):
        x = 8 + (math.log(t) - x0) / (x1 - x0) * (w - 16)
        y = 22 + (1 - (v - lo) / span) * (h - 30)
        return f"{x:.1f},{y:.1f}"
    col = T.teal if contango else T.coral
    line = " ".join(xy(t, v) for t, v in pts)
    ex = ""
    if kink:
        xp = [(t, ivx.get(str(t))) for t in tenors if _ok(ivx.get(str(t)))]
        if len(xp) >= 2:
            ex = (f'<polyline points="{" ".join(xy(t, v) for t, v in xp)}" fill="none" stroke="{T.amber}" '
                  f'stroke-width="1" stroke-dasharray="3 2"/>')
    iv30 = iv.get("30")
    dots = "".join(f'<circle cx="{xy(t, v).split(",")[0]}" cy="{xy(t, v).split(",")[1]}" r="1.4" fill="{col}"/>'
                   for t, v in pts)
    sc = STATE_COLORS.get(state, T.muted)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background:{T.panel};'
            f'border:1px solid {T.line};margin:0 6px 6px 0">'
            f'<title>{_e(tk)} · {_e(state)} · IV30 {iv30 * 100:.1f}%</title>'
            f'<text x="8" y="13" fill="{T.text}" font-size="10" {_FONT} font-weight="600">{_e(tk)}</text>'
            f'<text x="{w - 18}" y="13" fill="{T.muted}" font-size="9" {_FONT} text-anchor="end">'
            f'{(iv30 or 0) * 100:.1f}</text>'
            f'<circle cx="{w - 9}" cy="10" r="3.2" fill="{sc}"/>'
            f'<line x1="8" y1="{h - 7}" x2="{w - 8}" y2="{h - 7}" stroke="{T.line}" stroke-width="1"/>'
            f'{ex}<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.5"/>{dots}</svg>')


def term_ribbon(rows: list[dict], tenors: list[int]) -> str:
    cells = [term_cell(r["ticker"], r.get("iv", {}), r.get("ivx", {}), tenors, r.get("state", "NEUTRAL"),
                       bool(r.get("contango")), bool(r.get("kink_explained"))) for r in rows]
    legend = (f'<div style="font-size:0.62rem;letter-spacing:0.14em;color:{T.muted};margin:0.2rem 0 0.5rem 0">'
              f'<span style="color:{T.teal}">━ CONTANGO</span> &nbsp; <span style="color:{T.coral}">━ BACKWARDATION'
              f'</span> &nbsp; <span style="color:{T.amber}">┅ EX-EARNINGS</span> &nbsp; ● VRP STATE &nbsp; '
              f'x-axis 7D → 365D (log)</div>')
    return legend + '<div style="display:flex;flex-wrap:wrap">' + "".join(c for c in cells if c) + "</div>"


def state_diagram(current: str, days: int = 0, size: int = 300) -> str:
    """The four states on a cycle with the transition rules on the arrows."""
    nodes = [("ALL-CLEAR", 0.50, 0.12), ("ALARM", 0.88, 0.50), ("CONFIRMED", 0.50, 0.88),
             ("RECOVERING", 0.12, 0.50)]
    rules = ["slope z ≤ −1.5", "VRP ≤ 0", "slope ↑ 2d", "slope ≥ 0 3d"]
    W = H = size
    out = [f'<svg width="100%" viewBox="0 0 {W} {H}" style="max-width:{size}px">',
           f'<defs><marker id="zarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
           f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{T.faint}"/></marker></defs>']
    for i, (a, b) in enumerate(zip(nodes, nodes[1:] + nodes[:1])):
        x1, y1 = a[1] * W, a[2] * H
        x2, y2 = b[1] * W, b[2] * H
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        cx, cy = mx + (mx - W / 2) * 0.35, my + (my - H / 2) * 0.35
        # shorten the ends so arrows stop at the node rims
        def pull(x, y, tx, ty, d=34):
            L = math.hypot(tx - x, ty - y) or 1
            return x + (tx - x) / L * d, y + (ty - y) / L * d
        sx, sy = pull(x1, y1, cx, cy)
        ex, ey = pull(x2, y2, cx, cy)
        out.append(f'<path d="M{sx:.1f},{sy:.1f} Q{cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" fill="none" '
                   f'stroke="{T.faint}" stroke-width="1" marker-end="url(#zarr)"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy:.1f}" fill="{T.muted}" font-size="8.5" {_FONT} '
                   f'text-anchor="middle">{_e(rules[i])}</text>')
    for name, fx, fy in nodes:
        x, y = fx * W, fy * H
        c = STATE_COLORS[name]
        on = name == current
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="30" fill="{T.panel if not on else c}" '
                   f'fill-opacity="{1 if not on else 0.16}" stroke="{c}" stroke-width="{2 if on else 1}"/>')
        out.append(f'<text x="{x:.1f}" y="{y + 3:.1f}" fill="{c if on else T.muted}" font-size="8.5" {_FONT} '
                   f'text-anchor="middle" font-weight="{600 if on else 400}">{_e(name)}</text>')
        if on and days:
            out.append(f'<text x="{x:.1f}" y="{y + 15:.1f}" fill="{c}" font-size="8" {_FONT} '
                       f'text-anchor="middle">{days}d</text>')
    out.append(f'<text x="{W / 2}" y="{H / 2 + 3}" fill="{T.faint}" font-size="8" {_FONT} '
               f'text-anchor="middle">VRP30</text></svg>')
    return "".join(out)


def circular_dendrogram(merges: list, tickers: list[str], order: list[int], labels: list[int],
                        leaf_colors: dict[str, str], basket: set[str], size: int = 620) -> str:
    n = len(tickers)
    if n < 3 or not merges:
        return ""
    R = size / 2 - 70
    cx = cy = size / 2
    dmax = max(m[2] for m in merges) or 1.0
    ang = {}
    for pos, leaf in enumerate(order):
        ang[leaf] = 2 * math.pi * pos / n - math.pi / 2
    rad = {i: R for i in range(n)}
    span = {i: (ang[i], ang[i]) for i in range(n)}
    paths = []
    for s, (a, b, d, _) in enumerate(merges):
        node = n + s
        rp = R * (1 - 0.92 * d / dmax)
        lo = min(span[a][0], span[b][0])
        hi = max(span[a][1], span[b][1])
        ang[node] = (ang[a] + ang[b]) / 2
        span[node] = (lo, hi)
        rad[node] = rp
        for c in (a, b):
            x1, y1 = cx + rad[c] * math.cos(ang[c]), cy + rad[c] * math.sin(ang[c])
            x2, y2 = cx + rp * math.cos(ang[c]), cy + rp * math.sin(ang[c])
            paths.append(f"M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}")
        a0, a1 = sorted((ang[a], ang[b]))
        xa, ya = cx + rp * math.cos(a0), cy + rp * math.sin(a0)
        xb, yb = cx + rp * math.cos(a1), cy + rp * math.sin(a1)
        large = 1 if a1 - a0 > math.pi else 0
        paths.append(f"M{xa:.1f},{ya:.1f}A{rp:.1f},{rp:.1f} 0 {large} 1 {xb:.1f},{yb:.1f}")
    cat = T.cat
    out = [f'<svg width="100%" viewBox="0 0 {size} {size}" style="max-width:{size}px">']
    # cluster arcs on the rim
    for pos, leaf in enumerate(order):
        a = ang[leaf]
        c = cat[labels[leaf] % len(cat)]
        r1, r2 = R + 4, R + 9
        da = math.pi / n
        p = [(cx + r * math.cos(a + s * da), cy + r * math.sin(a + s * da)) for r, s in
             ((r1, -1), (r2, -1), (r2, 1), (r1, 1))]
        out.append(f'<path d="M{p[0][0]:.1f},{p[0][1]:.1f}L{p[1][0]:.1f},{p[1][1]:.1f}'
                   f'L{p[2][0]:.1f},{p[2][1]:.1f}L{p[3][0]:.1f},{p[3][1]:.1f}z" fill="{c}" opacity="0.75"/>')
    out.append(f'<path d="{" ".join(paths)}" fill="none" stroke="{T.faint}" stroke-width="0.8"/>')
    fs = 8 if n <= 120 else 6.5 if n <= 220 else 5.5
    for leaf in range(n):
        a = ang[leaf]
        tk = tickers[leaf]
        r = R + 13
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        deg = math.degrees(a)
        flip = 90 < (deg % 360) < 270
        rot = deg + 180 if flip else deg
        col = leaf_colors.get(tk, T.muted)
        wt = "600" if tk in basket else "400"
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{col}" font-size="{fs}" {_FONT} font-weight="{wt}" '
                   f'text-anchor="{"end" if flip else "start"}" dominant-baseline="middle" '
                   f'transform="rotate({rot:.1f} {x:.1f} {y:.1f})"><title>{_e(tk)}</title>'
                   f'{"◆ " if tk in basket and not flip else ""}{_e(tk)}{" ◆" if tk in basket and flip else ""}</text>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{T.faint}"/>'
               f'<text x="{cx}" y="{cy + 16}" fill="{T.faint}" font-size="8" {_FONT} text-anchor="middle">'
               f'ρ distance</text></svg>')
    return "".join(out)


def sigma_strip(label: str, value: float | None, mean: float | None, sd: float | None, lo: float | None,
                hi: float | None, w: int = 340, h: int = 34) -> str:
    """One tenor's IV against its trailing-year distribution: ±1σ / ±2σ bands,
    the year's range, and today's mark coloured by its z."""
    if not all(_ok(x) for x in (value, mean, sd)) or sd <= 0:
        return ""
    lo = lo if _ok(lo) else mean - 3 * sd
    hi = hi if _ok(hi) else mean + 3 * sd
    a, b = min(lo, mean - 2.5 * sd, value), max(hi, mean + 2.5 * sd, value)
    def x(v):
        return 70 + (v - a) / (b - a) * (w - 120)
    z = (value - mean) / sd
    c = T.coral if z >= 2 else T.amber if z >= 1 else T.teal if z <= -1 else T.text
    return (f'<svg width="100%" viewBox="0 0 {w} {h}" style="max-width:{w}px;display:block">'
            f'<text x="0" y="{h / 2 + 3}" fill="{T.muted}" font-size="9" {_FONT}>{_e(label)}</text>'
            f'<line x1="{x(lo):.1f}" y1="{h / 2}" x2="{x(hi):.1f}" y2="{h / 2}" stroke="{T.line}" stroke-width="1"/>'
            f'<rect x="{x(mean - 2 * sd):.1f}" y="{h / 2 - 7}" width="{x(mean + 2 * sd) - x(mean - 2 * sd):.1f}" '
            f'height="14" fill="{T.navy}" opacity="0.14"/>'
            f'<rect x="{x(mean - sd):.1f}" y="{h / 2 - 7}" width="{x(mean + sd) - x(mean - sd):.1f}" height="14" '
            f'fill="{T.navy}" opacity="0.30"/>'
            f'<line x1="{x(mean):.1f}" y1="{h / 2 - 9}" x2="{x(mean):.1f}" y2="{h / 2 + 9}" stroke="{T.muted}" '
            f'stroke-width="1"/>'
            f'<circle cx="{x(value):.1f}" cy="{h / 2}" r="4.5" fill="{c}"/>'
            f'<text x="{w - 2}" y="{h / 2 + 3}" fill="{c}" font-size="9" {_FONT} text-anchor="end">'
            f'{value * 100:.1f} · {z:+.1f}σ</text></svg>')


def heat_matrix(rows: list[str], cols: list[str], values: dict, cap: float, sub: dict | None = None) -> str:
    """Labelled diverging heat tiles: values[(row, col)] (decimal returns), sub = small second line."""
    from ..ui_kit import ramp
    head = "".join(f'<div style="color:{T.faint};font-size:0.6rem;letter-spacing:0.16em;text-align:center">'
                   f'{_e(c)}</div>' for c in cols)
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = values.get((r, c))
            if not _ok(v):
                cells.append(f'<div style="background:{T.grid};height:46px"></div>')
                continue
            bg = ramp(v, cap)
            fg = "#000" if abs(v) > cap * 0.5 else T.text
            s2 = (sub or {}).get((r, c))
            cells.append(f'<div title="{_e(r)} · {_e(c)}" style="background:{bg};color:{fg};height:46px;'
                         f'display:flex;flex-direction:column;justify-content:center;align-items:center;'
                         f'font-size:0.82rem">{v * 100:+.1f}%'
                         + (f'<span style="font-size:0.6rem;opacity:0.75">{_e(s2)}</span>' if s2 else "")
                         + '</div>')
        body.append(f'<div style="color:{T.muted};font-size:0.66rem;letter-spacing:0.12em;align-self:center">'
                    f'{_e(r)}</div>' + "".join(cells))
    n = len(cols)
    return (f'<div style="display:grid;grid-template-columns:minmax(110px,1.3fr) repeat({n},minmax(56px,1fr));'
            f'gap:3px;margin:0.3rem 0 0.8rem 0"><div></div>{head}{"".join(body)}</div>')


def vix_curve(points: list[tuple[str, float, float]], w: int = 300, h: int = 120) -> str:
    """Mini VIX term structure: [(label, days, level)]."""
    pts = [(lab, d, v) for lab, d, v in points if _ok(v)]
    if len(pts) < 2:
        return ""
    lo, hi = min(v for *_, v in pts), max(v for *_, v in pts)
    span = (hi - lo) or 1.0
    lx0, lx1 = math.log(pts[0][1]), math.log(pts[-1][1])
    def xy(d, v):
        return 22 + (math.log(d) - lx0) / ((lx1 - lx0) or 1) * (w - 44), 18 + (1 - (v - lo) / span) * (h - 44)
    col = T.teal if pts[0][2] <= pts[-1][2] else T.coral
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(d, v) for _, d, v in pts))
    marks = "".join(
        f'<circle cx="{xy(d, v)[0]:.1f}" cy="{xy(d, v)[1]:.1f}" r="3" fill="{col}"/>'
        f'<text x="{xy(d, v)[0]:.1f}" y="{xy(d, v)[1] - 7:.1f}" fill="{T.text}" font-size="9" {_FONT} '
        f'text-anchor="middle">{v:.1f}</text>'
        f'<text x="{xy(d, v)[0]:.1f}" y="{h - 6}" fill="{T.muted}" font-size="8" {_FONT} '
        f'text-anchor="middle">{_e(lab)}</text>' for lab, d, v in pts)
    return (f'<svg width="100%" viewBox="0 0 {w} {h}" style="max-width:{w}px">'
            f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.6"/>{marks}</svg>')

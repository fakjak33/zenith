"""TREND FOLLOWING bespoke visuals — pure HTML/SVG string builders.

Altair covers everything with axes (see view.py). These are the few pieces
that are glyphs rather than charts, and they are what make the seven-speed
term structure readable at a glance:

  trend_ladder     seven diverging bars, fast (top) -> slow (bottom), with
                   ghost ticks where each speed stood 5 and 20 days ago
  speed_ribbon     a compact 7-cell strip (one cell per speed) for cards
  speed_pips       ●●●○○○○ -- which speeds took part in an event
  spectrum_strip   the universe's score distribution on a -20..+20 band
  universe_panel   the cross-universe header (Stocks vs ETFs)
  event_card       one row of the Triggers feed

Color is Zenith's app-wide sign convention (ui_charts: teal above zero,
coral below, panel-black at zero so "no trend" recedes). Text never wears
the series color; values sit in the text tokens beside a colored mark.
Pure functions: no Streamlit import, unit-testable, safe to cache.
"""

from __future__ import annotations

import html as _html

from ..config import THEME
from ..ui_charts import _CORAL, _PANEL, _TEAL
from . import SPEED_KEYS, SPEED_NAMES, SPEED_SHORT
from .structure import STRUCTURE_COLORS


def div_rgb(v, cap: float = 20.0) -> str:
    """Panel-black -> teal (v > 0) / coral (v < 0) ramp, saturating at |v| = cap."""
    if v is None:
        return "rgb(40,40,40)"
    a = max(0.0, min(1.0, abs(float(v)) / cap)) if cap else 0.0
    c = _TEAL if v >= 0 else _CORAL
    r, g, b = (int(p + (q - p) * a) for p, q in zip(_PANEL, c))
    return f"rgb({r},{g},{b})"


def _esc(x) -> str:
    return _html.escape(str(x))


def _fmt(v, nd: int = 1, signed: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


# ---------------------------------------------------------------- ladder --
def trend_ladder(forecasts: list, ghost5: list | None = None, ghost20: list | None = None,
                 last_cross: dict | None = None, width: int = 560) -> str:
    """The Trend Ladder: one diverging bar per speed around a zero spine."""
    row_h, top, label_w, val_w = 34, 26, 150, 132
    plot_w = width - label_w - val_w
    mid = label_w + plot_w / 2
    half = plot_w / 2 - 4
    h = top + row_h * len(SPEED_KEYS) + 10
    parts = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px;display:block;'
        f'font-family:{THEME.font_body};" role="img" aria-label="Trend ladder: seven EWMAC speeds">',
        '<style>.zl-bar{transform-origin:var(--o) 50%;animation:zlgrow .7s cubic-bezier(.2,.7,.2,1) both}'
        '@keyframes zlgrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}'
        '.zl-row:hover .zl-bg{fill:rgba(255,255,255,0.06)}</style>',
        # scale header
        f'<text x="{label_w}" y="14" fill="{THEME.muted}" font-size="10">-20 BEARISH</text>',
        f'<text x="{mid}" y="14" fill="{THEME.muted}" font-size="10" text-anchor="middle">0</text>',
        f'<text x="{label_w + plot_w}" y="14" fill="{THEME.muted}" font-size="10" '
        f'text-anchor="end">BULLISH +20</text>',
    ]
    for q in (-10, 10):
        x = mid + half * q / 20
        parts.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{h - 8}" '
                     f'stroke="{THEME.grid}" stroke-width="1"/>')
    for i, k in enumerate(SPEED_KEYS):
        y = top + i * row_h
        v = forecasts[i] if i < len(forecasts) else None
        cy = y + row_h / 2
        tip = f"{SPEED_SHORT[k]} EWMAC · {SPEED_NAMES[k]}: {_fmt(v)}"
        lc = (last_cross or {}).get(k)
        if lc:
            tip += (f" · turned {'bullish' if lc['dir'] > 0 else 'bearish'} {lc['date']}"
                    f" ({lc['bars_ago']} bars ago)")
        parts.append(f'<g class="zl-row"><title>{_esc(tip)}</title>'
                     f'<rect class="zl-bg" x="0" y="{y + 2}" width="{width}" height="{row_h - 4}" '
                     f'fill="transparent"/>')
        parts.append(f'<text x="0" y="{cy - 2}" fill="{THEME.text}" font-size="13" '
                     f'font-family="{THEME.font_display}" letter-spacing="1">{SPEED_SHORT[k]}</text>'
                     f'<text x="0" y="{cy + 11}" fill="{THEME.muted}" font-size="9" '
                     f'letter-spacing="1">{SPEED_NAMES[k].upper()}</text>')
        parts.append(f'<rect x="{label_w}" y="{cy - 9}" width="{plot_w}" height="18" '
                     f'fill="rgba(255,255,255,0.03)"/>')
        if v is None:
            parts.append(f'<text x="{mid}" y="{cy + 4}" fill="{THEME.muted}" font-size="10" '
                         f'text-anchor="middle">insufficient history</text></g>')
            continue
        w = half * min(1.0, abs(v) / 20.0)
        x0 = mid if v >= 0 else mid - w
        origin = f"{mid:.1f}px"
        parts.append(f'<rect class="zl-bar" style="--o:{origin};animation-delay:{i * 55}ms" '
                     f'x="{x0:.1f}" y="{cy - 8}" width="{max(w, 1.5):.1f}" height="16" rx="2" '
                     f'fill="{div_rgb(v)}"/>')
        for g, col, dash in ((ghost20, THEME.muted, "2 2"), (ghost5, THEME.mustard, "")):
            gv = g[i] if g and i < len(g) else None
            if gv is None:
                continue
            gx = mid + half * max(-1.0, min(1.0, gv / 20.0))
            parts.append(f'<line x1="{gx:.1f}" y1="{cy - 12}" x2="{gx:.1f}" y2="{cy + 12}" '
                         f'stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
        side = "▲" if v > 0 else ("▼" if v < 0 else "•")
        since = f"{lc['bars_ago']}d" if lc else "—"
        parts.append(f'<text x="{width - val_w + 10}" y="{cy + 5}" fill="{THEME.text}" font-size="15" '
                     f'font-family="{THEME.font_display}">{_fmt(v)}</text>'
                     f'<text x="{width - val_w + 62}" y="{cy + 4}" fill="{THEME.teal if v > 0 else THEME.coral}" '
                     f'font-size="11">{side}</text>'
                     f'<text x="{width - 4}" y="{cy + 4}" fill="{THEME.muted}" font-size="10" '
                     f'text-anchor="end">{since}</text></g>')
    parts.append(f'<line x1="{mid}" y1="{top - 4}" x2="{mid}" y2="{h - 8}" stroke="{THEME.text}" '
                 f'stroke-width="1.5" opacity="0.7"/></svg>')
    legend = (f'<div style="font-size:0.72rem;color:{THEME.muted};margin-top:0.25rem;">'
              f'bar = today’s forecast · <span style="color:{THEME.mustard}">▏</span> 5 days ago · '
              f'<span style="color:{THEME.muted}">┆</span> 20 days ago · right column = bars since that '
              f'speed last crossed. Hover a row for the crossover date.</div>')
    return "".join(parts) + legend


# ------------------------------------------------------------ small glyphs --
def speed_ribbon(forecasts: list, cell: int = 14, gap: int = 2) -> str:
    """Seven cells, fast -> slow, colored by forecast."""
    cells = []
    for i, k in enumerate(SPEED_KEYS):
        v = forecasts[i] if forecasts and i < len(forecasts) else None
        cells.append(f'<span title="{_esc(SPEED_SHORT[k])}: {_esc(_fmt(v))}" style="display:inline-block;'
                     f'width:{cell}px;height:{cell}px;margin-right:{gap}px;background:{div_rgb(v)};'
                     f'border-radius:2px;vertical-align:middle;"></span>')
    return f'<span style="white-space:nowrap;">{"".join(cells)}</span>'


def speed_pips(speeds: list[str], direction: int) -> str:
    """● for each participating speed (fast -> slow), ○ otherwise."""
    col = THEME.teal if direction > 0 else THEME.coral
    out = []
    for k in SPEED_KEYS:
        on = k in (speeds or [])
        out.append(f'<span title="{_esc(SPEED_SHORT[k])}" style="color:{col if on else "#444"};'
                   f'font-size:0.9rem;letter-spacing:1px;">{"●" if on else "○"}</span>')
    return "".join(out)


# ------------------------------------------------------------ spectrum ----
def spectrum_strip(scores: list[float], width: int = 900, height: int = 92) -> str:
    """The universe on one -20..+20 line: a 41-bin density profile over a
    diverging band, with the median marked and the bearish / neutral / bullish
    shares written at the edges."""
    vals = [s for s in scores if s is not None]
    if not vals:
        return ""
    bins = [0] * 41
    for s in vals:
        bins[int(round(max(-20.0, min(20.0, s)))) + 20] += 1
    peak = max(bins) or 1
    pad, band_h, top = 8, 10, 18
    plot_w = width - 2 * pad
    bw = plot_w / 41
    base = height - band_h - 16
    n = len(vals)
    med = sorted(vals)[n // 2] if n % 2 else (sorted(vals)[n // 2 - 1] + sorted(vals)[n // 2]) / 2
    bull = sum(1 for s in vals if s >= 5) / n
    bear = sum(1 for s in vals if s < -5) / n
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block;'
             f'font-family:{THEME.font_body};" role="img" aria-label="Trend score distribution">']
    for i, c in enumerate(bins):
        v = i - 20
        bh = (base - top) * c / peak
        x = pad + i * bw
        parts.append(f'<rect x="{x + 1:.1f}" y="{base - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" '
                     f'rx="2" fill="{div_rgb(v, 20)}" opacity="0.95"><title>score {v:+d}: {c} '
                     f'assets</title></rect>')
    for i in range(41):
        v = i - 20
        parts.append(f'<rect x="{pad + i * bw:.1f}" y="{base + 3}" width="{bw + 0.5:.1f}" '
                     f'height="{band_h}" fill="{div_rgb(v, 20)}"/>')
    for t in (-5, 5):
        x = pad + (t + 20.5) * bw
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{base + band_h + 3}" '
                     f'stroke="{THEME.grid}" stroke-width="1"/>')
    mx = pad + (med + 20.5) * bw
    parts.append(f'<line x1="{mx:.1f}" y1="{top - 6}" x2="{mx:.1f}" y2="{base + band_h + 3}" '
                 f'stroke="{THEME.text}" stroke-width="2"/>'
                 f'<text x="{mx:.1f}" y="{top - 8}" fill="{THEME.text}" font-size="11" '
                 f'text-anchor="middle">median {med:+.1f}</text>')
    parts.append(f'<text x="{pad}" y="{height - 1}" fill="{THEME.coral}" font-size="11">'
                 f'-20 · {bear:.0%} bearish</text>'
                 f'<text x="{width / 2}" y="{height - 1}" fill="{THEME.muted}" font-size="11" '
                 f'text-anchor="middle">{1 - bull - bear:.0%} neutral</text>'
                 f'<text x="{width - pad}" y="{height - 1}" fill="{THEME.teal}" font-size="11" '
                 f'text-anchor="end">{bull:.0%} bullish · +20</text></svg>')
    return "".join(parts)


# ------------------------------------------------------ cross-universe panel --
def universe_panel(summaries: list[dict]) -> str:
    """summaries: [{"label", "n", "pct_bull", "pct_bear", "median", "persist_up",
    "persist_down", "breadth": [7 x %bullish], "as_of"}]"""
    cards = []
    for s in summaries:
        if not s:
            continue
        bull, bear = s["pct_bull"], s["pct_bear"]
        neutral = max(0.0, 1 - bull - bear)
        bar = (f'<div style="display:flex;height:8px;margin:0.45rem 0 0.35rem 0;gap:2px;">'
               f'<div style="flex:{bear};background:{THEME.coral};border-radius:2px 0 0 2px;"></div>'
               f'<div style="flex:{neutral};background:#333;"></div>'
               f'<div style="flex:{bull};background:{THEME.teal};border-radius:0 2px 2px 0;"></div></div>')
        speeds = "".join(
            f'<div title="{_esc(SPEED_SHORT[k])}: {p:.0%} of assets bullish" style="flex:1;text-align:center;">'
            f'<div style="height:30px;display:flex;align-items:flex-end;justify-content:center;">'
            f'<div style="width:70%;height:{max(2, 30 * p):.0f}px;background:{div_rgb((p - 0.5) * 40, 20)};'
            f'border-radius:2px 2px 0 0;"></div></div>'
            f'<div style="font-size:0.62rem;color:{THEME.muted};">{_esc(SPEED_SHORT[k])}</div></div>'
            for k, p in zip(SPEED_KEYS, s.get("breadth") or [0.5] * 7) if p is not None)
        cards.append(
            f'<div style="flex:1 1 320px;min-width:280px;border:1px solid {THEME.grid};'
            f'border-top:3px solid {THEME.teal if bull >= bear else THEME.coral};'
            f'background:rgba(0,0,0,0.35);padding:0.6rem 0.8rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<div style="font-family:{THEME.font_display};font-size:1.3rem;letter-spacing:0.14em;'
            f'color:#fff;">{_esc(s["label"]).upper()}</div>'
            f'<div style="font-size:0.7rem;color:{THEME.muted};">{s["n"]} scored · {_esc(s.get("as_of", ""))}'
            f'</div></div>'
            f'<div style="display:flex;gap:1.2rem;margin-top:0.2rem;font-family:{THEME.font_display};">'
            f'<div><span style="font-size:2rem;color:{THEME.teal};">{bull:.0%}</span>'
            f'<span style="font-size:0.75rem;color:{THEME.muted};"> BULLISH</span></div>'
            f'<div><span style="font-size:2rem;color:{THEME.coral};">{bear:.0%}</span>'
            f'<span style="font-size:0.75rem;color:{THEME.muted};"> BEARISH</span></div>'
            f'<div><span style="font-size:2rem;color:#fff;">{s["median"]:+.1f}</span>'
            f'<span style="font-size:0.75rem;color:{THEME.muted};"> MEDIAN</span></div></div>'
            f'{bar}'
            f'<div style="font-size:0.7rem;color:{THEME.muted};letter-spacing:0.1em;">'
            f'% BULLISH AT EACH SPEED · FAST → SLOW</div>'
            f'<div style="display:flex;gap:4px;margin-top:0.2rem;">{speeds}</div>'
            f'<div style="font-size:0.74rem;color:#ccc;margin-top:0.35rem;">'
            f'{s["persist_up"]} persistent uptrends · {s["persist_down"]} persistent downtrends</div></div>')
    return ('<div style="display:flex;gap:0.9rem;flex-wrap:wrap;margin:0.3rem 0 1rem 0;">'
            + "".join(cards) + '</div>')


# --------------------------------------------------------------- event card --
TYPE_LABELS = {"cross": "Crossover", "trigger": "Trigger", "upgrade": "Signal Upgrade",
               "downgrade": "Signal Downgrade", "confirmation": "Multi-Speed Confirmation"}


def event_title(e: dict) -> str:
    d = e.get("dir", 0)
    t = e["type"]
    if t == "trigger":
        return "Bullish Trigger" if d > 0 else "Bearish Trigger"
    if t == "cross":
        return f"{SPEED_SHORT.get(e.get('speed'), '')} {'bullish' if d > 0 else 'bearish'} crossover"
    if t == "confirmation":
        return f"Multi-Speed Confirmation · {'bullish' if d > 0 else 'bearish'}"
    return TYPE_LABELS[t]


def event_card(e: dict, name: str = "", forecasts: list | None = None) -> str:
    d = e.get("dir", 0)
    col = THEME.teal if d > 0 else THEME.coral
    big = e["type"] == "confirmation"
    sb, sa = e.get("score_before"), e.get("score_after")
    move = (f'{_fmt(sb)} → {_fmt(sa)}' if sb is not None and sa is not None
            else (f'score {_fmt(sa)}' if sa is not None else ""))
    states = (f'{_esc(e.get("from") or "")} → {_esc(e.get("to") or "")}'
              if e.get("from") and e.get("to") else "")
    border = (f'border:2px solid transparent;background:linear-gradient({THEME.panel},{THEME.panel}) '
              f'padding-box, linear-gradient(90deg,{col},{THEME.mustard}) border-box;'
              if big else f'border:1px solid {THEME.grid};border-left:4px solid {col};'
                          f'background:rgba(0,0,0,0.35);')
    pad = "0.65rem 0.85rem" if big else "0.4rem 0.7rem"
    return (
        f'<div style="{border}padding:{pad};margin-bottom:0.45rem;display:flex;gap:0.9rem;'
        f'align-items:center;flex-wrap:wrap;">'
        f'<div style="min-width:5.6rem;font-size:0.74rem;color:{THEME.muted};">{_esc(e["date"])}</div>'
        f'<div style="min-width:4.5rem;font-family:{THEME.font_display};font-size:{1.45 if big else 1.2}rem;'
        f'color:#fff;letter-spacing:0.06em;">{_esc(e["ticker"])}</div>'
        f'<div style="flex:1 1 220px;min-width:180px;">'
        f'<div style="font-size:{0.9 if big else 0.82}rem;color:#eee;">'
        f'<span style="color:{col};">{"▲" if d > 0 else "▼"}</span> {_esc(event_title(e))}</div>'
        f'<div style="font-size:0.7rem;color:{THEME.muted};overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap;">{_esc(name)}</div></div>'
        f'<div title="speeds involved, fast → slow">{speed_pips(e.get("speeds") or [], d)}</div>'
        + (f'<div>{speed_ribbon(forecasts, cell=10, gap=1)}</div>' if forecasts else "")
        + f'<div style="min-width:9rem;text-align:right;font-size:0.8rem;color:#ddd;">{move}'
          f'<div style="font-size:0.66rem;color:{THEME.muted};">{states or _esc(e.get("horizon") or "")}'
          f'</div></div></div>')


def structure_chip(label: str) -> str:
    c = STRUCTURE_COLORS.get(label, THEME.muted)
    return (f'<span style="display:inline-block;border:1px solid {c};border-left:4px solid {c};'
            f'padding:0.15rem 0.5rem;font-size:0.78rem;color:#eee;background:rgba(0,0,0,0.3);">'
            f'{_esc(label)}</span>')

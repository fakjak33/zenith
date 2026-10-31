"""Interactive 3D surfaces (three.js in an iframe, same token/payload pattern as
the EPHEMERIS board).

A payload is a grid of values on two labelled axes, optionally a stack of
FRAMES (dates) the in-page scrubber morphs through. The surface is coloured
vertex by vertex from a token-built heat ramp, with a faint floor shadow,
hairline grid lines following the surface, a hover read-out, click-to-slice
insets (smile + term) and render-on-demand (no idle animation loop).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import THEME as T

_TEMPLATE = Path(__file__).with_name("surface3d") / "scene.html"
THREE_VERSION = "0.160.0"


def ramp(kind: str = "iv") -> list[list]:
    """Heat ramp stops from THEME tokens. 'iv': cool -> hot; 'div': teal <- dark -> coral."""
    if kind == "div":
        return [[0.0, T.teal], [0.35, T.phosphor_dim], [0.5, T.raised], [0.65, T.mauve], [1.0, T.coral]]
    return [[0.0, T.navy], [0.28, T.teal], [0.52, T.mint], [0.74, T.mustard], [0.88, T.orange], [1.0, T.coral]]


def fill(grid) -> tuple[list[list[float]], list[list[bool]]]:
    """Fill gaps by linear interpolation along rows, then columns, then the
    nearest value; returns (filled grid, was-missing mask)."""
    a = np.array([[np.nan if v is None else float(v) for v in row] for row in grid], float)
    miss = ~np.isfinite(a)
    if miss.all():
        return [], []
    def lin(v):
        ok = np.isfinite(v)
        if ok.sum() == 0:
            return v
        if ok.sum() == 1:
            return np.where(ok, v, v[ok][0])
        x = np.arange(len(v))
        return np.interp(x, x[ok], v[ok])
    b = a.copy()
    for i in range(b.shape[0]):
        if np.isfinite(b[i]).any():
            b[i] = lin(b[i])
    for j in range(b.shape[1]):
        b[:, j] = lin(b[:, j])
    b = np.where(np.isfinite(b), b, np.nanmean(b))
    return b.tolist(), miss.tolist()


def payload(title: str, x_ticks: list[str], y_ticks: list[str], frames: list[tuple[str, list]],
            x_label: str = "DELTA", y_label: str = "TENOR", z_label: str = "IMPLIED VOL",
            z_fmt: str = "pct", diverging: bool = False, subtitle: str = "",
            extras: list[dict] | None = None, x_every: int = 1, y_every: int = 1) -> dict | None:
    """frames: [(label, rows x cols grid)], oldest first. None when nothing to draw."""
    out_frames, masks = [], None
    for lab, g in frames:
        f, m = fill(g)
        if f:
            out_frames.append({"label": lab, "z": [[round(v, 5) for v in row] for row in f]})
            masks = m
    if not out_frames:
        return None
    allv = np.array([v for fr in out_frames for row in fr["z"] for v in row], float)
    lo, hi = float(np.nanpercentile(allv, 1)), float(np.nanpercentile(allv, 99))
    if diverging:
        m = max(abs(lo), abs(hi), 1e-6)
        lo, hi = -m, m
    elif hi - lo < 1e-6:
        hi = lo + 0.01
    ex = []
    for e in extras or []:
        ex.append({"name": e["name"], "fmt": e.get("fmt", ""),
                   "grid": [[None if (v is None or not math.isfinite(float(v))) else round(float(v), 5)
                             for v in row] for row in e["grid"]]})
    return {"title": title, "subtitle": subtitle, "x": {"label": x_label, "ticks": x_ticks, "every": x_every},
            "y": {"label": y_label, "ticks": y_ticks, "every": y_every}, "z": {"label": z_label, "fmt": z_fmt},
            "zmin": lo, "zmax": hi, "diverging": diverging, "frames": out_frames, "masks": masks,
            "extras": ex}


def scene_html(p: dict, height: int = 600) -> str:
    tpl = _TEMPLATE.read_text(encoding="utf-8")
    subs = {"__BG__": T.bg, "__PANEL__": T.panel, "__LINE__": T.line, "__TEXT__": T.text,
            "__MUTED__": T.muted, "__FAINT__": T.faint, "__ACCENT__": T.teal, "__AMBER__": T.amber,
            "__HEIGHT__": str(height),
            "__RAMP__": json.dumps(ramp("div" if p.get("diverging") else "iv")),
            "__PAYLOAD__": json.dumps(p, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")}
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    return tpl


def render(p: dict, height: int = 600) -> None:
    import streamlit.components.v1 as components
    components.html(scene_html(p, height), height=height + 6, scrolling=False)

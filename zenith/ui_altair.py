"""One registered Altair theme for ZENITH//TERM: every chart inherits fonts,
axes, grid and colour ranges from here instead of setting them per mark."""

from __future__ import annotations

import altair as alt

from .ui_term import TERM as T

NAME = "zenith_term"


def _config() -> dict:
    axis = {"domainColor": T.line, "gridColor": T.grid, "gridDash": [2, 3], "tickColor": T.line,
            "labelColor": T.muted, "titleColor": T.muted, "labelFont": "Share Tech Mono",
            "titleFont": "Silkscreen", "titleFontSize": 9, "titleFontWeight": "normal",
            "labelFontSize": 11, "labelLimit": 0}
    return {"config": {
        "background": "transparent",
        "view": {"stroke": T.line, "strokeWidth": 1},
        "font": "Share Tech Mono",
        "axis": axis,
        "legend": {"labelColor": T.text, "titleColor": T.muted, "labelFont": "Share Tech Mono",
                   "titleFont": "Silkscreen", "symbolType": "square", "labelLimit": 0},
        "title": {"color": T.phosphor, "font": "VT323", "fontSize": 20, "fontWeight": "normal",
                  "anchor": "start"},
        "range": {"category": list(T.cat),
                  "diverging": [T.coral, "#5a1d14", T.glass, "#0f5c33", T.phosphor],
                  "heatmap": [T.glass, T.phosphor_dim, T.phosphor],
                  "ramp": [T.glass, T.phosphor_dim, T.phosphor]},
        "line": {"strokeWidth": 2, "color": T.phosphor},
        "area": {"color": T.cyan, "opacity": 0.25},
        "bar": {"color": T.phosphor},
        "point": {"color": T.phosphor, "filled": True},
        "rule": {"color": T.amber},
    }}


def enable() -> None:
    try:
        alt.theme.register(NAME, enable=True)(_config)
    except AttributeError:                     # altair < 5.5
        alt.themes.register(NAME, _config)
        alt.themes.enable(NAME)

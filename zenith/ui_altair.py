"""One registered Altair theme for ZENITH v2: every chart inherits a monospace,
hairline, black-glass look and the categorical / diverging / sequential ranges
from THEME instead of setting them per mark."""

from __future__ import annotations

import altair as alt

from .config import THEME as T

NAME = "zenith_v2"
FONT = "JetBrains Mono"
DIVERGING = [T.coral, "#7a2e22", "#151a20", "#16615b", T.teal]


def _config() -> dict:
    axis = {"domain": False, "gridColor": T.grid, "gridDash": [1, 3], "gridOpacity": 1,
            "tickColor": T.line, "tickSize": 3, "labelColor": T.muted, "titleColor": T.faint,
            "labelFont": FONT, "titleFont": FONT, "labelFontSize": 10, "titleFontSize": 9,
            "titleFontWeight": "normal", "labelLimit": 0, "titlePadding": 8}
    return {"config": {
        "background": "transparent",
        "padding": {"left": 4, "right": 8, "top": 6, "bottom": 4},
        "view": {"stroke": None},
        "font": FONT,
        "axis": axis,
        "axisX": {"grid": False},
        "legend": {"labelColor": T.muted, "titleColor": T.faint, "labelFont": FONT, "titleFont": FONT,
                   "labelFontSize": 10, "titleFontSize": 9, "titleFontWeight": "normal",
                   "symbolType": "square", "symbolSize": 60, "labelLimit": 0},
        "header": {"labelColor": T.muted, "titleColor": T.faint, "labelFont": FONT, "titleFont": FONT},
        "title": {"color": T.text, "font": FONT, "fontSize": 11, "fontWeight": "normal",
                  "anchor": "start", "offset": 10},
        "range": {"category": list(T.cat), "diverging": DIVERGING,
                  "heatmap": ["#0b1a1a", "#145a55", T.teal, T.mint],
                  "ramp": ["#0b1a1a", "#145a55", T.teal, T.mint]},
        "line": {"strokeWidth": 1.6, "color": T.teal},
        "area": {"color": T.teal, "opacity": 0.22},
        "bar": {"color": T.teal},
        "rect": {"color": T.teal},
        "point": {"color": T.teal, "filled": True, "size": 36},
        "circle": {"color": T.teal},
        "rule": {"color": T.faint},
        "text": {"font": FONT, "color": T.text, "fontSize": 10},
        "tick": {"color": T.teal},
    }}


def enable() -> None:
    try:
        alt.theme.register(NAME, enable=True)(_config)
    except AttributeError:                     # altair < 5.5
        alt.themes.register(NAME, _config)
        alt.themes.enable(NAME)

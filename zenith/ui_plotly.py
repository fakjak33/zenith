"""ZENITH v2 for Plotly — used by the BACKTEST pages, where zoom, hover-linked
subplots and 3D surfaces earn their place over Altair.

Plotly is MIT-licensed and renders fully offline; nothing here touches Chart
Studio or any hosted service. Every colour comes from ``config.THEME``.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .config import THEME as T
from .ui_kit import rgba

AXIS = dict(gridcolor=T.grid, linecolor=T.line, zerolinecolor=T.line, tickcolor=T.line,
            showspikes=True, spikecolor=T.faint, spikethickness=1, spikedash="dot", spikemode="across",
            title_font=dict(size=10, color=T.muted), tickfont=dict(size=10, color=T.muted))

DIVERGING = [[0.0, T.coral], [0.5, T.raised], [1.0, T.teal]]
SEQUENTIAL = [[0.0, T.bg], [0.35, rgba(T.violet, 0.7)], [0.7, T.cyan], [1.0, T.teal]]

pio.templates["zenith_v2"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor=T.bg, plot_bgcolor=T.bg,
    font=dict(family=T.font_body, color=T.text, size=11),
    colorway=list(T.cat),
    xaxis=AXIS, yaxis=AXIS,
    hovermode="x unified",
    hoverlabel=dict(bgcolor=T.raised, bordercolor=T.line, font=dict(family=T.font_body, size=11, color=T.text)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=10, color=T.muted),
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=48, r=16, t=28, b=32),
    title=dict(font=dict(size=11, color=T.muted), x=0, xanchor="left"),
    scene=dict(xaxis=dict(backgroundcolor=T.bg, gridcolor=T.line, color=T.muted),
               yaxis=dict(backgroundcolor=T.bg, gridcolor=T.line, color=T.muted),
               zaxis=dict(backgroundcolor=T.bg, gridcolor=T.line, color=T.muted)),
    coloraxis=dict(colorscale=DIVERGING, colorbar=dict(outlinewidth=0, tickfont=dict(size=10, color=T.muted))),
))

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
          "toImageButtonOptions": {"format": "png", "scale": 2}}


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(template="zenith_v2", height=height)
    return fig


def label(title: str) -> str:
    """Chart title as quiet chrome above the plot (never colliding with the legend)."""
    return (f'<div style="font-size:0.6rem; letter-spacing:0.2em; color:{T.muted}; text-transform:uppercase; '
            f'margin:0.6rem 0 -0.2rem 0;">{html.escape(title)}</div>')


def _is_time_axis(fig: go.Figure) -> bool:
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None or len(x) == 0:
            continue
        v = x[0]
        if hasattr(v, "year") or (isinstance(v, str) and len(v) >= 8 and v[:4].isdigit() and v[4] == "-"):
            return True
        try:
            import numpy as np
            if np.issubdtype(np.asarray(x).dtype, np.datetime64):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _main_y_positive(fig: go.Figure) -> bool:
    """True when every trace on the primary y axis is strictly positive (log scale is meaningful)."""
    import numpy as np
    seen = False
    for tr in fig.data:
        if getattr(tr, "type", "") not in ("scatter", "scattergl", "bar", "candlestick", "ohlc"):
            continue
        if (getattr(tr, "yaxis", None) or "y") != "y":
            continue
        y = getattr(tr, "y", None)
        if y is None or len(y) == 0:
            continue
        try:
            arr = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            return False
        arr = arr[np.isfinite(arr)]
        if arr.size and arr.min() <= 0:
            return False
        seen = seen or arr.size > 0
    return seen


def add_scale_toggle(fig: go.Figure) -> go.Figure:
    """LIN / LOG buttons for the primary y axis — client-side, so switching never reruns the page."""
    cur = fig.layout.yaxis.type or "linear"
    btn = dict(bgcolor=T.raised, bordercolor=T.line, borderwidth=1, font=dict(size=9, color=T.muted),
               active=1 if cur == "log" else 0, pad=dict(l=2, r=2, t=0, b=0))
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", showactive=True, x=1.0, xanchor="right", y=1.02, yanchor="bottom",
        buttons=[dict(label="LIN", method="relayout", args=[{"yaxis.type": "linear", "yaxis.autorange": True}]),
                 dict(label="LOG", method="relayout", args=[{"yaxis.type": "log", "yaxis.autorange": True}])],
        **btn)])
    return fig


def show(fig: go.Figure, height: int = 360, title: str = "", key: str | None = None,
         scale_toggle: bool | None = None) -> None:
    """Render a themed figure; a failing chart degrades to a caption, never the page.

    Time-series charts whose main series is positive get a LIN/LOG toggle automatically
    (``scale_toggle`` forces it on or off).
    """
    try:
        if title:
            st.markdown(label(title), unsafe_allow_html=True)
        want = scale_toggle if scale_toggle is not None else (_is_time_axis(fig) and _main_y_positive(fig))
        if want and not fig.layout.updatemenus:
            add_scale_toggle(fig)
        st.plotly_chart(style(fig, height), use_container_width=True, theme=None,
                        config=CONFIG, key=key)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"chart unavailable: {type(exc).__name__}")


def fill(color: str, a: float = 0.12) -> str:
    return rgba(color, a)

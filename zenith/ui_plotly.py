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


def show(fig: go.Figure, height: int = 360, title: str = "", key: str | None = None) -> None:
    """Render a themed figure; a failing chart degrades to a caption, never the page."""
    try:
        if title:
            st.markdown(label(title), unsafe_allow_html=True)
        st.plotly_chart(style(fig, height), use_container_width=True, theme=None,
                        config=CONFIG, key=key)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"chart unavailable: {type(exc).__name__}")


def fill(color: str, a: float = 0.12) -> str:
    return rgba(color, a)

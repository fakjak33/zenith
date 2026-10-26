"""Altair signature charts: ridgeline, percentile fan, bump chart, calendar heat,
small multiples and a beeswarm. Each takes a tidy DataFrame and returns a chart
(the registered ZENITH theme supplies fonts, axes and ranges)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import THEME as T
from ..ui_altair import DIVERGING

MID = DIVERGING[2]


def ridgeline(df: pd.DataFrame, group: str, value: str, order: list | None = None,
              step: int = 34, overlap: float = 1.8, width_px: int | None = None,
              domain: tuple[float, float] | None = None, colors: list[str] | None = None, bw: float | None = None):
    """Stacked density curves per group (KDE computed by Vega-Lite)."""
    import altair as alt
    order = order or list(dict.fromkeys(df[group]))
    lo, hi = domain or (float(df[value].min()), float(df[value].max()))
    dens = alt.Chart(df).transform_density(
        value, groupby=[group], as_=[value, "density"], extent=[lo, hi],
        **({"bandwidth": bw} if bw else {})
    ).mark_area(interpolate="monotone", fillOpacity=0.55, stroke=T.text, strokeWidth=0.6,
                strokeOpacity=0.6).encode(
        x=alt.X(f"{value}:Q", title=None, scale=alt.Scale(domain=[lo, hi])),
        y=alt.Y("density:Q", stack=None, axis=None, scale=alt.Scale(range=[step, -step * overlap])),
        fill=alt.Fill(f"{group}:N", legend=None, sort=order,
                      scale=alt.Scale(domain=order, range=colors or list(T.cat))),
    ).properties(height=step, **({"width": width_px} if width_px else {}))
    return dens.facet(row=alt.Row(f"{group}:N", sort=order, title=None,
                                  header=alt.Header(labelAngle=0, labelAlign="left", labelColor=T.muted,
                                                    labelFontSize=10, labelPadding=4))
                      ).properties(bounds="flush").configure_facet(spacing=0).configure_view(stroke=None)


def fan(df: pd.DataFrame, x: str, value: str, group: str | None = None, height: int = 300,
        colors: dict | None = None, paths: pd.DataFrame | None = None, path_id: str | None = None):
    """Percentile fan: 10-90 and 25-75 bands plus the median, optionally per group,
    with faint sample paths underneath."""
    import altair as alt
    keys = [x] + ([group] if group else [])
    q = df.groupby(keys)[value].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = ["p10", "p25", "p50", "p75", "p90"]
    q = q.reset_index()
    col = alt.Color(f"{group}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                    legend=alt.Legend(orient="top", title=None)) if group and colors else alt.value(T.teal)
    base = alt.Chart(q).encode(x=alt.X(f"{x}:Q", title=x.replace("_", " ")))
    layers = []
    if paths is not None and path_id:
        layers.append(alt.Chart(paths).mark_line(strokeWidth=0.6, opacity=0.08).encode(
            x=f"{x}:Q", y=f"{value}:Q", detail=f"{path_id}:N", color=col))
    layers += [base.mark_area(opacity=0.14).encode(y=alt.Y("p10:Q", title=value.replace("_", " ")), y2="p90:Q", color=col),
               base.mark_area(opacity=0.26).encode(y="p25:Q", y2="p75:Q", color=col),
               base.mark_line(strokeWidth=2).encode(y="p50:Q", color=col),
               alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=T.faint, strokeDash=[2, 3]).encode(y="y:Q")]
    return alt.layer(*layers).properties(height=height)


def bump(df: pd.DataFrame, x: str, series: str, rank: str, height: int = 360, highlight: list | None = None):
    """Rank-over-time bump chart: rank 1 at top, smooth lines, end labels."""
    import altair as alt
    last = df[df[x] == df[x].max()]
    hl = set(highlight or [])
    op = alt.condition(alt.FieldOneOfPredicate(field=series, oneOf=list(hl)), alt.value(1.0),
                       alt.value(0.35)) if hl else alt.value(0.9)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{rank}:Q", scale=alt.Scale(reverse=True, zero=False), title="rank"),
        color=alt.Color(f"{series}:N", legend=None), opacity=op)
    lines = base.mark_line(interpolate="monotone", strokeWidth=2)
    pts = base.mark_circle(size=34)
    lbl = alt.Chart(last).mark_text(align="left", dx=8, fontSize=10).encode(
        x=f"{x}:O", y=f"{rank}:Q", text=f"{series}:N", color=alt.Color(f"{series}:N", legend=None))
    return (lines + pts + lbl).properties(height=height)


def calendar_heat(df: pd.DataFrame, x: str, y: str, value: str, cap: float, height_per: int = 16,
                  fmt: str = "+.2f", x_type: str = "O"):
    """Square-cell heat grid (date/period on x, category on y) on the diverging range."""
    import altair as alt
    ny = max(df[y].nunique(), 1)
    return alt.Chart(df).mark_rect(cornerRadius=1.5, stroke=T.bg, strokeWidth=1.5).encode(
        x=alt.X(f"{x}:{x_type}", title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y(f"{y}:N", title=None),
        color=alt.Color(f"{value}:Q", legend=None,
                        scale=alt.Scale(domain=[-cap, -cap / 2, 0, cap / 2, cap], clamp=True,
                                        range=DIVERGING)),
        tooltip=[f"{y}:N", f"{x}:{x_type}", alt.Tooltip(f"{value}:Q", format=fmt)],
    ).properties(height=max(120, ny * height_per))


def beeswarm(df: pd.DataFrame, x: str, group: str, color: str | None = None, size: int = 28,
             height_per: int = 46, jitter: float = 0.36, seed: int = 1, tooltip: list | None = None):
    """Dot strips per group with deterministic vertical jitter."""
    import altair as alt
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_j"] = rng.uniform(-jitter, jitter, len(d))
    ng = max(d[group].nunique(), 1)
    return alt.Chart(d).mark_circle(size=size, opacity=0.8).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{group}:N", title=None),
        yOffset=alt.YOffset("_j:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
        color=(alt.Color(f"{color}:Q", legend=None,
                         scale=alt.Scale(range=[T.coral, MID, T.teal], domainMid=0)) if color
               else alt.value(T.teal)),
        tooltip=tooltip or [group, x],
    ).properties(height=ng * height_per)

"""Interval-return distribution + skew — a strategy-evaluation view.

Given a strategy's per-bar return series, aggregate to a chosen interval (the
native candle, or daily/weekly/monthly) and describe the distribution's shape,
especially skew (positive / negative / ~none). Used on the backtest pages to
judge whether a strategy's return profile is favourably skewed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# interval label -> pandas resample rule (None = keep native bars)
INTERVALS = {"Candle (native)": None, "Daily": "D", "Weekly": "W-FRI", "Monthly": "ME"}
# coarseness order so we never try to go finer than the native timeframe
_ORDER = {"Daily": 0, "Weekly": 1, "Monthly": 2}
_NATIVE_RANK = {"Daily": 0, "Weekly": 1, "Monthly": 2}


def interval_returns(returns: pd.Series, native_tf: str,
                     interval: str) -> tuple[pd.Series, str]:
    """Aggregate per-bar returns to ``interval`` by compounding within periods.

    Returns (series, effective_label). If the requested interval is finer than
    the native timeframe (e.g. weekly bars asked for daily), falls back to the
    native candle and says so in the label.
    """
    r = returns.dropna()
    rule = INTERVALS.get(interval)
    if rule is None or rule == "D" and native_tf == "Daily":
        return r, "Candle (native)" if rule is None else interval
    # guard: can't synthesize finer bars than we have
    if interval in _ORDER and _ORDER[interval] < _NATIVE_RANK.get(native_tf, 0):
        return r, f"{native_tf} candle (native — finer not available)"
    if not isinstance(r.index, pd.DatetimeIndex):
        return r, "Candle (native)"
    agg = (1.0 + r).resample(rule).prod() - 1.0
    return agg.dropna(), interval


def skew_summary(r: pd.Series) -> dict:
    """Distribution shape stats, with a categorical skew label."""
    r = r.dropna()
    n = int(r.shape[0])
    if n < 5:
        return {"n": n, "skew": 0.0, "label": "n/a", "mean": 0.0,
                "median": 0.0, "std": 0.0}
    sk = float(stats.skew(r))
    label = "positive" if sk > 0.1 else ("negative" if sk < -0.1 else "~none")
    return {"n": n, "skew": sk, "label": label,
            "mean": float(r.mean()), "median": float(r.median()),
            "std": float(r.std())}


def distribution_figure(r: pd.Series, summary: dict, theme,
                        title: str = "INTERVAL RETURN DISTRIBUTION") -> go.Figure:
    """Histogram of interval returns coloured by skew sign, with mean/median lines."""
    color = {"positive": theme.long_color, "negative": theme.short_color}.get(
        summary.get("label"), theme.mustard)
    r = r.dropna() * 100.0          # show in percent
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=max(12, min(60, summary["n"] // 3)),
                               marker=dict(color=color, line=dict(color="#000", width=0.5)),
                               name="returns", opacity=0.85))
    if summary["n"] >= 5:
        fig.add_vline(x=summary["mean"] * 100, line=dict(color="#fff", width=1.5, dash="dash"),
                      annotation_text="mean", annotation_position="top")
        fig.add_vline(x=summary["median"] * 100, line=dict(color=theme.muted, width=1, dash="dot"),
                      annotation_text="median", annotation_position="bottom")
    badge = (f"skew {summary['skew']:+.2f} ({summary['label']})  ·  "
             f"mean {summary['mean']*100:+.2f}%  ·  n={summary['n']}")
    fig.update_layout(title=f"{title} — {badge}",
                      xaxis_title="return %", yaxis_title="count", bargap=0.04)
    return fig

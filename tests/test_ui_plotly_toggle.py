"""LIN/LOG toggle is added only where log scale is meaningful."""

import pandas as pd
import plotly.graph_objects as go

from zenith import ui_plotly as up


def _ts(y):
    return go.Figure(go.Scatter(x=pd.bdate_range("2020-01-01", periods=len(y)), y=y))


def test_positive_time_series_gets_toggle():
    f = _ts([1.0, 1.1, 1.3])
    assert up._is_time_axis(f) and up._main_y_positive(f)
    up.add_scale_toggle(f)
    assert [b.label for b in f.layout.updatemenus[0].buttons] == ["LIN", "LOG"]


def test_negative_or_categorical_skipped():
    assert not up._main_y_positive(_ts([0.1, -0.2, 0.3]))
    assert not up._is_time_axis(go.Figure(go.Bar(x=["a", "b"], y=[1, 2])))


def test_secondary_axis_ignored():
    f = _ts([1.0, 2.0])
    f.add_trace(go.Scatter(x=f.data[0].x, y=[-1, -2], yaxis="y2"))
    assert up._main_y_positive(f)

"""Signature-visual builders and brand art -- offline, synthetic data only (no network)."""

from __future__ import annotations

import re
import xml.dom.minidom as md

import numpy as np
import pandas as pd
import pytest

from zenith import art, ui_kit as uk
from zenith.viz import charts, radial, treemap


def _well_formed(svg: str) -> None:
    md.parseString(svg)
    assert not re.search(r"(?<![a-z])(nan|-?inf)(?![a-z])", svg.lower())


# ------------------------------------------------------------------ treemap --

def test_squarify_fills_the_box_exactly():
    vals = [6, 6, 4, 3, 2, 2, 1]
    rects = treemap.squarify(vals, 0, 0, 600, 400)
    area = sum(w * h for _, _, w, h in rects)
    assert area == pytest.approx(600 * 400, rel=1e-9)
    for (x, y, w, h), v in zip(rects, vals):
        assert w * h == pytest.approx(v / sum(vals) * 600 * 400, rel=1e-9)
        assert 0 <= x <= 600 and 0 <= y <= 400


def test_squarify_handles_zeros():
    rects = treemap.squarify([0, 3, 0], 0, 0, 100, 100)
    assert rects[0][2] * rects[0][3] == 0 and rects[1][2] * rects[1][3] == pytest.approx(10000)


def test_treemap_svg_well_formed():
    groups = [{"name": "Tech", "children": [{"name": "AAPL", "size": 3, "value": 0.02},
                                            {"name": "MSFT", "size": 2, "value": -0.01}]},
              {"name": "Energy", "children": [{"name": "XOM", "size": 1, "value": 0.05}]}]
    svg = treemap.treemap(groups, 400, 200, cap=0.05)
    _well_formed(svg)
    assert svg.count("<rect") >= 5


# ------------------------------------------------------------------- radial --

def test_regime_quadrant_and_radar():
    _well_formed(radial.regime_quadrant([(0.1, -0.2), (0.4, 0.3), (float("nan"), 1.0), (-0.5, 0.1)]))
    _well_formed(radial.radar(["A", "B", "C", "D"], [("x", [0.5, -1, 2, 0], "#2ec4b6")]))
    assert radial.radar(["A", "B"], []) == ""


def test_day_clock_and_sunburst():
    _well_formed(radial.day_clock(0.12, -0.03, "SPY"))
    tree = [{"name": "Equity", "value": 10, "color_value": 3,
             "children": [{"name": "US", "value": 6, "color_value": 5, "children": []},
                          {"name": "Intl", "value": 4, "color_value": -2, "children": []}]},
            {"name": "Bonds", "value": 5, "color_value": -4, "children": []}]
    svg = radial.sunburst(tree, 300, cap=5)
    _well_formed(svg)
    assert svg.count("<path") == 4


# ------------------------------------------------------------------- charts --

def test_altair_signatures_build():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(list("abc"), 50), "v": rng.normal(size=150)})
    charts.ridgeline(df, "g", "v", domain=(-3, 3)).to_dict()
    fan = pd.DataFrame({"day": np.tile([0, 5, 20], 40), "x": rng.normal(size=120),
                        "side": np.repeat(["L", "S"], 60), "e": np.repeat(range(40), 3)})
    charts.fan(fan, "day", "x", "side", colors={"L": "#2ec4b6", "S": "#ff5a3c"}, paths=fan, path_id="e").to_dict()
    b = pd.DataFrame({"m": np.repeat(["2026-01", "2026-02"], 3), "f": ["x", "y", "z"] * 2, "r": [1, 2, 3, 3, 1, 2]})
    charts.bump(b, "m", "f", "r", highlight=["x"]).to_dict()
    charts.calendar_heat(b, "m", "f", "r", cap=3).to_dict()
    charts.beeswarm(pd.DataFrame({"s": [0.1, -0.4, 0.9], "g": ["a", "a", "b"]}), "s", "g", color="s").to_dict()


# ---------------------------------------------------------------- kit + art --

def test_kit_builders():
    assert "<svg" in uk.sparkline_svg([1, 2, 3, 2])
    assert uk.sparkline_svg([1]) == ""
    assert uk.ascii_spark([1, 5, 3]) and len(uk.ascii_spark([1, 5, 3])) == 3
    assert "z-segs" in uk.led_meter([("x", 0.5), ("y", -2)], cap=1)
    assert "z-dm" in uk.dot_matrix(["a"], ["c1", "c2"], {("a", "c1"): 0.5}, cap=1)
    _well_formed(uk.ring_donut([("a", 1, "#2ec4b6"), ("b", 3, "#f2a93b")]).split("<div>")[0].split('class="z-donut">')[1])
    _well_formed(uk.radial_gauge(0.4, 0, 1, "BP"))


def test_brand_art_well_formed():
    for svg in (art.mark(48), art.mark(48, animate=False), art.horizon(600, 160), art.orbits(120),
                art.contours(300, 80), art.grid_globe(90)):
        _well_formed(svg)
    assert "<animateTransform" in art.mark(48) and "<animateTransform" not in art.mark(48, animate=False)


# ------------------------------------------------------------- volatility --

def _surface_payload():
    from zenith.viz import surface3d
    g = [[0.2 + 0.01 * i + 0.005 * j for j in range(9)] for i in range(7)]
    g[0][0] = None
    return surface3d.payload("SPY", [str(j) for j in range(9)], [str(i) for i in range(7)],
                             [("2026-10-29", g), ("2026-10-30", g)],
                             extras=[{"name": "z", "fmt": "sigma", "grid": g}])


def test_surface3d_scene_fills_every_placeholder():
    from zenith.viz import surface3d
    p = _surface_payload()
    assert p["masks"][0][0] is True and p["frames"][-1]["z"][0][0] is not None   # gap filled, flagged
    p["title"] = "</script><b>x"
    out = surface3d.scene_html(p, 500)
    assert not re.search(r"__[A-Z]+__", out)
    assert "</script><b>" not in out                              # payload cannot close the script
    assert re.search(r"three@\d+\.\d+\.\d+/", out)                # exact version pinned


def test_surface3d_template_has_no_hex_literals():
    from pathlib import Path
    tpl = (Path(__file__).resolve().parents[1] / "zenith" / "viz" / "surface3d" / "scene.html").read_text(encoding="utf-8")
    assert not re.search(r"#[0-9a-fA-F]{6}\b", tpl)
    for needle in ("text-shadow", "scanline", "VT323"):
        assert needle not in tpl


def test_volviz_svgs_well_formed():
    from zenith.viz import volviz
    _well_formed(volviz.state_diagram("ALARM", 3))
    cell = volviz.term_cell("SPY", {"7": 0.2, "30": 0.18, "90": 0.19}, {"7": 0.2, "30": 0.17, "90": 0.19},
                            [7, 30, 90], "ALL-CLEAR", True, True)
    _well_formed(cell)
    _well_formed(volviz.sigma_strip("30D", 0.25, 0.2, 0.03, 0.15, 0.3))
    merges = [[0, 1, 0.2, 2], [2, 3, 0.3, 2], [4, 5, 0.9, 4]]
    _well_formed(volviz.circular_dendrogram(merges, ["A", "B", "C", "D"], [0, 1, 2, 3], [0, 0, 1, 1],
                                            {"A": "#000000"}, {"A"}))
    _well_formed(volviz.vix_curve([("9D", 9, 19.0), ("30D", 30, 17.2), ("3M", 93, 20.8)]))

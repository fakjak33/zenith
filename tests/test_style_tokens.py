"""ZENITH//TERM style guard -- offline. Colours live in zenith.config.THEME only;
views use tokens and the registered Altair theme, never raw hex or Vega schemes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"config.py", "ui_theme.py", "ui_charts.py", "ui_altair.py", "ui_kit.py", "art.py"}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
# black text on bright fills is deliberate; '#588' etc. in data strings are not colours
OK_SHORT = {"#000"}


def _view_sources():
    for p in (ROOT / "zenith").rglob("*.py"):
        if p.name not in ALLOWED:
            yield p


def test_no_six_digit_hex_outside_theme_modules():
    bad = []
    for p in _view_sources():
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
            if HEX.search(code) and not line.lstrip().startswith("#"):
                bad.append(f"{p.relative_to(ROOT)}:{n}")
    assert not bad, bad


def test_no_named_vega_schemes():
    bad = [str(p.relative_to(ROOT)) for p in _view_sources() if "scheme=" in p.read_text(encoding="utf-8")]
    assert not bad, bad


def test_streamlit_config_matches_theme():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "gen_streamlit_config.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8") == gen.render()


def test_altair_theme_registered():
    import altair as alt
    import zenith.ui_theme  # noqa: F401  (registers on import)
    assert alt.theme.active == "zenith_v2"


def test_no_arcade_chrome_left():
    """v2 is institutional monospace: no pixel fonts, scanlines or CRT flicker."""
    bad = []
    for p in (ROOT / "zenith").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for needle in ("VT323", "Silkscreen", "Share+Tech+Mono", "ztflicker", "repeating-linear-gradient(0deg"):
            if needle in text:
                bad.append(f"{p.relative_to(ROOT)}: {needle}")
    assert not bad, bad

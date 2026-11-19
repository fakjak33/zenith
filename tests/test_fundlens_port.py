"""FundLens port: licensed data must never be committed, and the page runs on demo data."""

import subprocess
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_no_spreadsheets_committed():
    """Morningstar exports and firm approved lists are licensed / confidential: none in git."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    bad = [f for f in out.splitlines() if f.lower().endswith((".xlsx", ".xls", ".xlsm"))
           or "morningstar" in f.lower() or "local_fixtures" in f.lower()]
    assert not bad, bad


def test_demo_export_scores():
    from zenith.fundlens.demo import demo_files
    from zenith.fundlens.engine import run
    import tempfile
    (dn, db), (an, ab) = demo_files()
    with tempfile.TemporaryDirectory() as tmp:
        d, a = Path(tmp) / dn, Path(tmp) / an
        d.write_bytes(db)
        a.write_bytes(ab)
        comps = run([d], approved_path=a)
    full = next(c for c in comps if c.kind == "full")
    assert full.scores.rank.notna().sum() >= 10 and {c.kind for c in comps} == {"full", "approved"}


def test_page_renders_on_demo_data():
    at = AppTest.from_string("from zenith.fundlens.view import render\nrender()", default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert any("Funds ranked" in m.label for m in at.metric)

"""FundLens (RESEARCH / FUNDLENS) — ported from the standalone repo's synthetic test suites.

Every fixture is generated (zenith/fundlens/demo.py); no licensed Morningstar data is used or
committed. The standalone repo's local-fixture tests (real exports) are intentionally not ported.
"""

import pytest

from zenith.fundlens.demo import write_approved, write_transposed


@pytest.fixture
def synth(tmp_path):
    return write_transposed(tmp_path / "synthetic.xlsx")


@pytest.fixture
def synth_approved(tmp_path):
    return write_approved(tmp_path / "approved.xlsx", ["AAA", "CCC", "EEE", "GGG", "III", "KKK", "ZZZ"])

import hashlib
import math

import openpyxl
import pandas as pd
import pytest

from zenith.fundlens.dictionary import load_dictionary
from zenith.fundlens.engine import run
from zenith.fundlens.ingest import load_approved, load_dataset
from zenith.fundlens.profile import load_profile
from zenith.fundlens.quality import QualityLog
from zenith.fundlens.render_xlsx import write_workbook
from zenith.fundlens.scoring import composite, pct_rank


# --------------------------------------------------------------------------- dictionary
@pytest.mark.parametrize("label,col", [
    ("Name ", "name"),
    ("Sortino  Ratio  3 Yr", "sortino_3y"),
    ("Bear Market %(mo-end)", "bear_market_pct"),
    ("NAV %  Rank Cat 12 Mo (Qtr-end)", "nav_rank_cat_12m"),
    ("NAV Tot Return 10 Yr (Qtr-end) ", "nav_ret_10y"),
    ("Mkt Tot Return 10 Yr (Qtr-end)", "mkt_ret_10y"),
    ("Sharpe Ratio 3 Yr (% Rank Category)", "sharpe_rank_cat_3y"),
    ("Tracking  Error", "index_te"),
    ("Tracking  Error Ratio 15 Yr", "te_ratio_15y"),
    ("Downside Capture   Ratio  15 Yr", "down_capture_15y"),
    ("NAV Annl Return 2016", "nav_annual_2016"),
    ("Net Exp Ratio (%)", "net_er"),
    ("Total Assets $MM", "aum"),
    ("Strategic Beta", "strategic_beta"),
    ("Strategic Beta Group", "sb_group"),
])
def test_label_matching(label, col):
    m = load_dictionary().match(label)
    assert m is not None and m.column == col


def test_unknown_label_is_unmapped():
    assert load_dictionary().match("Some Unknown Field") is None


# --------------------------------------------------------------------------- percentiles
def test_pct_rank_directions_and_ties():
    s = pd.Series([1.0, 2.0, 2.0, 4.0, None], index=list("abcde"))
    hi = pct_rank(s, "higher")
    assert hi["d"] == 100 and hi["a"] == 0 and hi["b"] == hi["c"] == pytest.approx(50.0)
    assert math.isnan(hi["e"])
    lo = pct_rank(s, "lower")
    assert lo["a"] == 100 and lo["d"] == 0
    tg = pct_rank(pd.Series([0.6, 1.02, 1.3, 0.95]), "target", 1.0)
    assert tg.idxmax() == 1 and tg.idxmin() == 0          # 1.02 closest to 1.0; 0.6 furthest


def test_single_value_scores_50():
    assert pct_rank(pd.Series([3.0]), "higher").iloc[0] == 50.0


def test_shrinkage_formula():
    ps = pd.DataFrame({"a": [90.0, 90.0]}, index=["full", "thin"])
    pc = pd.DataFrame({"a": [1.0, 0.55]}, index=["full", "thin"])
    raw, cov, adj = composite(ps, pc, {"a": 1.0}, 1.0)
    assert adj["full"] == pytest.approx(90.0)
    assert adj["thin"] == pytest.approx(50 + 40 * 0.55)   # 72, as documented


# --------------------------------------------------------------------------- ingest
def test_ingest_blocks_dedupe_and_unmapped(synth):
    log = QualityLog()
    ds = load_dataset([synth], log)
    assert len(ds.df) == 12 and ds.n_blocks == 2
    assert ds.duplicates == ["DDD"]
    assert ds.df.at["DDD", "nav_ret_3y"] is not None       # kept the record with more data
    assert "some unknown field" in ds.unmapped
    assert ds.df.at["AAA", "ret_date"].isoformat() == "2026-09-30"


def test_approved_ignores_disclaimer_rows(synth_approved):
    ap = load_approved(synth_approved, QualityLog())
    assert ap.tickers == ["AAA", "CCC", "EEE", "GGG", "III", "KKK", "ZZZ"]
    assert ap.ignored_rows >= 2


# --------------------------------------------------------------------------- pipeline
def test_pipeline_full_and_approved(synth, synth_approved, tmp_path):
    full, appr = run([synth], approved_path=synth_approved)
    assert full.scores.rank.notna().sum() == 12
    assert sorted(appr.df.index) == ["AAA", "CCC", "EEE", "GGG", "III", "KKK"]
    st = appr.audit.set_index("ticker")["status"]
    assert st["ZZZ"] == "Approved - not in data" and st["BBB"] == "Not approved"
    # horizons renormalised: young funds score, but coverage < 1 shrinks them toward 50
    assert full.scores.coverage["KKK"] < full.scores.coverage["AAA"]
    s = full.scores
    assert all(abs(s.adjusted[t] - 50) <= abs(s.raw[t] - 50) + 1e-9 for t in s.order)
    # ranks are a permutation, ties broken deterministically
    assert sorted(s.rank.astype(int)) == list(range(1, 13))
    # unmapped numeric code is never given a made-up label
    L = next(L for L in full.lenses if L.field == "sb_group")
    assert "Value" in L.groups and not L.unmapped_codes   # code 12 is verified


def test_metric_directions_in_profile():
    from zenith.fundlens.scoring import effective_direction
    d, p = load_dictionary(), load_profile("us_equity_style")
    assert effective_direction("std_dev", d, p) == ("lower", None)
    assert effective_direction("down_capture", d, p) == ("lower", None)
    assert effective_direction("net_er", d, p) == ("lower", None)
    assert effective_direction("sharpe", d, p) == ("higher", None)
    assert effective_direction("beta", d, p) == ("target", 1.0)
    alts = load_profile("alts_diversifier")
    assert effective_direction("r_squared", d, alts)[0] == "lower"
    assert effective_direction("beta", d, alts) == ("target", 0.0)


def test_profile_inheritance():
    core = load_profile("us_equity_core")
    assert core["pillars"]["cost"]["weight"] == 25
    assert "te_ratio" in core["pillars"]["fit"]["metrics"]
    assert core["pillars"]["returns"]["metrics"] == {"nav_ret": 1.0}     # inherited


def _book_digest(path) -> str:
    wb = openpyxl.load_workbook(path)
    h = hashlib.sha256()
    for ws in wb.worksheets:
        h.update(ws.title.encode())
        for row in ws.iter_rows():
            for c in row:
                fill = c.fill.start_color.rgb if c.fill and c.fill.fill_type == "solid" else ""
                h.update(f"{c.coordinate}|{c.value}|{fill}|{c.number_format};".encode())
    return h.hexdigest()


def test_workbook_is_deterministic_and_colours_follow_direction(synth, synth_approved, tmp_path):
    a = write_workbook(run([synth], approved_path=synth_approved)[0], tmp_path / "a")
    b = write_workbook(run([synth], approved_path=synth_approved)[0], tmp_path / "b")
    assert _book_digest(a) == _book_digest(b)
    wb = openpyxl.load_workbook(a)
    assert {"Summary", "Master Ranking", "Pillar Scorecard", "Group Leaders", "Calendar Years", "Risk & Fit",
            "Sensitivity", "Approval Audit", "Original Data", "Metric Guide", "Methodology",
            "Data Quality"} <= set(wb.sheetnames)
    ws = wb["Master Ranking"]
    hdr = {ws.cell(2, k).value: k for k in range(1, ws.max_column + 1)}
    col = hdr["Std Dev 5Y"]
    vals = [(ws.cell(r, col).value, ws.cell(r, col).fill.start_color.rgb[-6:]) for r in range(3, ws.max_row + 1)
            if isinstance(ws.cell(r, col).value, (int, float))]
    lowest, highest = min(vals), max(vals)
    assert lowest[1] == "63BE7B"        # lowest volatility -> green
    assert highest[1] == "F8696B"       # highest volatility -> red


# --------------------------------------------------------------------------- recency
def test_ex_latest_year_return_is_exact():
    from zenith.fundlens.derive import add_recency
    df = pd.DataFrame({"nav_ret_12m": [10.8, 30.0], "nav_ret_3y": [16.35, 12.0], "nav_ret_5y": [9.68, 8.0],
                       "name": ["A", "B Momentum ETF"]}, index=["A", "B"])
    add_recency(df, load_profile("us_equity_style"))
    expect = ((1.1635 ** 3 / 1.108) ** 0.5 - 1) * 100          # 2 years before the latest 12 months
    assert df.at["A", "nav_ret_xr_3y"] == pytest.approx(expect)
    assert df.at["B", "momentum_tilt"] == "Yes" and pd.isna(df.at["A", "momentum_tilt"])


def test_recency_modes_expand_profile():
    from zenith.fundlens.derive import apply_recency
    base = load_profile("us_equity_style")
    off = apply_recency(dict(base, recency={"mode": "off", "surge_flag": 35}))
    bal = apply_recency(dict(base, recency={"mode": "balanced", "surge_flag": 35}))
    strict = apply_recency(dict(base, recency={"mode": "strict", "surge_flag": 35}))
    assert off["pillars"]["returns"]["metrics"] == {"nav_ret": 1.0}
    assert bal["pillars"]["returns"]["metrics"] == {"nav_ret": 1.0, "nav_ret_xr": 1.0}
    assert bal["pillars"]["consistency"]["metrics"]["recency_surge"] == 1.0
    assert strict["pillars"]["returns"]["metrics"] == {"nav_ret_xr": 1.0}
    assert strict["pillars"]["consistency"]["metrics"]["recency_surge"] == 2.0


def test_verified_strategic_beta_codes():
    from zenith.fundlens.dictionary import load_codes
    from zenith.fundlens.lenses import group_values
    codes = load_codes()
    assert codes["sb_group"]["12"] == "Value" and codes["sb_group"]["8"] == "Multifactor"
    assert group_values("7", "sb_group", False, codes, {}) == ["Momentum"]
    unm = {}
    assert group_values("3", "sb_group", False, codes, unm) == ["Code 3 (unmapped)"] and unm == {"3": 1}


def test_theme_scan_flags_hot_and_persistent_groups():
    from types import SimpleNamespace
    from zenith.fundlens.themes import theme_scan
    prof = load_profile("us_equity_style")
    idx = [f"F{i}" for i in range(12)]
    grp = ["Momentum"] * 3 + ["Quality"] * 3 + ["Value"] * 6
    r1 = [40, 38, 42] + [24, 25, 23] + [10, 11, 9, 10, 12, 8]          # momentum & quality beat the field
    xr = [8, 9, 7] + [16, 17, 15] + [10, 9, 11, 10, 9, 11]             # ...but only quality did so before
    sk = [60, 55, 58] + [5, 0, 8] + [-5, 0, 3, -2, 1, 0]
    df = pd.DataFrame({"sb_group": grp, "nav_ret_12m": r1, "nav_ret_xr_3y": xr, "recency_skew": sk}, index=idx)
    s = SimpleNamespace(rank=pd.Series(range(1, 13), index=idx, dtype=float))
    prof["recency"]["theme_dimensions"] = [{"field": "sb_group", "label": "Strategic Beta Group"}]
    tab, hot = theme_scan(df, s, s.rank, prof, {"sb_group": {}}, QualityLog())
    flags = dict(zip(tab["group"], tab["flag"]))
    assert flags["Momentum"] == "Hot year"
    assert flags["Quality"] == "Persistent leader"
    assert hot["F0"] == ["Momentum"] and "F3" not in hot

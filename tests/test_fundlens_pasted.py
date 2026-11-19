"""Copy-pasted Morningstar Advisor Workstation grids: stacked pages, wrapped headers, separate name pane."""
import openpyxl
import pandas as pd
import pytest

from zenith.fundlens.engine import run
from zenith.fundlens.ingest import load_dataset
from zenith.fundlens.pasted import Page, resolve_names
from zenith.fundlens.quality import QualityLog

TICKERS = ["AAAX", "BBBX", "CCCX", "DDDX", "EEEX", "FFFX", "GGGX", "HHHX"]
NAMES = [f"Fund {t[0]} Inst" for t in TICKERS]


def _page(ws, r, header_rows, rows, names_present, title=None, bench=None):
    """Write one pasted page starting at row r; returns next free row."""
    if title:
        ws.cell(r, 1, title)
        r += 1
    for i in names_present:
        ws.cell(r, 2, i + 1)
        ws.cell(r, 3, NAMES[i])
        r += 1
    for hr in header_rows:
        for c, v in enumerate(hr, start=1):
            if v is not None:
                ws.cell(r, c, v)
        r += 1
    if bench:
        for c, v in enumerate(["-"] + bench, start=1):
            ws.cell(r, c, v)
        r += 1
    for row in rows:
        for c, v in enumerate(row, start=1):
            ws.cell(r, c, v)
        r += 1
    return r + 2


def write_paste(path, drop_basics=(), drop_risk=(3, 4, 5), names_risk=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    hdr1 = [["Ticker", "Tot Ret", "Tot Ret", "Annual Report", "Fund"],
            [None, "12 Mo", "3 Yr Annlzd", "Net Expense", "Objective"],
            [None, "(mo-end)", "(mo-end)", "Ratio", None]]
    rows1 = [[t, 10 + i, 8 + i * 0.5, 0.5 + i * 0.05, "Growth"] for i, t in enumerate(TICKERS) if i not in drop_basics]
    r = _page(ws, 1, hdr1, rows1, range(8), title="S&P 500 (TR) (1970)", bench=[15.7, 22.9, "-", "-"])
    hdr2 = [["Ticker", "Sharpe", "Standard", "Cyclical"], [None, "Ratio", "Deviation", None], [None, "3 Yr", "3 Yr", None]]
    rows2 = [[t, 0.5 + i * 0.05, 15 - i * 0.3, 30 + i] for i, t in enumerate(TICKERS) if i not in drop_risk]
    keep = names_risk if names_risk is not None else [i for i in range(8) if i not in drop_risk]
    _page(ws, r, hdr2, rows2, keep, bench=[1.37, 12.4, 23.2])
    wb.save(path)
    return path


def test_pages_headers_benchmark_and_coverage(tmp_path):
    log = QualityLog()
    ds = load_dataset([write_paste(tmp_path / "paste.xlsx")], log)
    assert len(ds.pages) == 2 and [p["listed"] for p in ds.pages] == [8, 8]
    assert {"nav_ret_12m", "nav_ret_3y", "annual_net_er", "sharpe_3y", "std_dev_3y", "sec_cyclical"} <= set(ds.df.columns)
    assert ds.df.at["BBBX", "nav_ret_3y"] == 8.5
    assert ds.benchmark["nav_ret_3y"] == 22.9 and ds.benchmark["sharpe_3y"] == 1.37
    assert ds.benchmark_name == "S&P 500 (TR) (1970)"
    assert any(e.area == "Coverage" for e in log.entries)                 # risk page 5/8
    # basics page has every fund and every name -> all names provably matched
    assert ds.df.at["HHHX", "name"] == "Fund H Inst" and ds.name_info["resolved"] == 8


def test_names_left_blank_when_ambiguous(tmp_path):
    # Basics page misses fund C; no page has both its data row and a complete name list, so
    # funds A/B are certain (before the gap?) only if constraints prove it - here names stay blank.
    ds = load_dataset([write_paste(tmp_path / "p.xlsx", drop_basics=(2,), drop_risk=(2,), names_risk=[])], QualityLog())
    assert ds.name_info["n_tickers"] == 7 and ds.name_info["ambiguous"] > 0
    unresolved = ds.df["name"].isna().sum()
    assert unresolved == ds.name_info["ambiguous"]
    # every name that was assigned is the true one
    for t in ds.df.index:
        n = ds.df.at[t, "name"]
        assert pd.isna(n) or n == NAMES[TICKERS.index(t)]


def test_resolve_names_uses_page_constraints():
    p1 = Page(0, [], [], [], ["A", "B", "D"], {1: "a", 2: "b", 3: "c", 4: "d"}, None)
    p2 = Page(0, [], [], [], ["A", "B"], {1: "a", 2: "b"}, None)
    names, info = resolve_names([p1, p2])
    assert names["A"] == "a" and names["B"] == "b" and "D" not in names    # D could be position 3 or 4


def test_second_file_fills_names_and_engine_runs(tmp_path):
    paste = write_paste(tmp_path / "p.xlsx", drop_basics=(2,), drop_risk=(2,), names_risk=[])
    names = tmp_path / "names.csv"
    names.write_text("Name,Ticker\n" + "\n".join(f"{n},{t}" for n, t in zip(NAMES, TICKERS)), encoding="utf-8")
    comps = run([paste, names])
    df = comps[0].df
    assert df["name"].notna().all() and df.at["AAAX", "name"] == "Fund A Inst"
    assert comps[0].scores.rank.notna().sum() == 7



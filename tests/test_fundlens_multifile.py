"""One Advisor Workstation export per data set, merged by ticker."""
import hashlib
import json
from pathlib import Path

import openpyxl
import pytest

from zenith.fundlens.engine import run
from zenith.fundlens.ingest import load_dataset
from zenith.fundlens.quality import QualityLog
from zenith.fundlens.results import results_dict

TICK = ["AAAX", "BBBX", "CCCX", "DDDX", "EEEX", "FFFX", "GGGX"]


def write_export(path, data_set, header, rows, bench):
    """Export layout: title block, Name/Ticker header, data, 'Average' dash row, Benchmark row, (c) footer."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MSEXP123.xls"
    ws.append(["TEST LIST", "US Mutual Fund Universe", data_set, "  For advisor use only.  ", "Print Date 10-22-2026"])
    ws.append(["", "", "Ranked by:", "ascending", "Name"])
    ws.append([])
    ws.append(["Name", "Ticker"] + header)
    for r in rows:
        ws.append(r)
    ws.append(["Average:"])
    ws.append(["-"] * (len(header) + 2))
    ws.append(["Benchmark:"])
    ws.append(["S&P 500 (TR) (1970)", "-"] + bench)
    ws.append(["©2026 Morningstar. All Rights Reserved."])
    wb.save(path)
    return path


def make_files(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    ret = write_export(d / "returns.xlsx", "Annualized Returns Monthly",
                       ["Tot Ret 12 Mo (mo-end)", "Tot Ret 3 Yr Annlzd (mo-end)", "Tot Ret 5 Yr Annlzd (mo-end)", "Price"],
                       [[f"Fund {t}", t, 10 + i, 8 + i, 6 + i * 0.5, 20 + i] for i, t in enumerate(TICK)], [15.7, 22.9, 13.8, "-"])
    risk = write_export(d / "risk.xlsx", "Risk Analysis",
                        ["Sharpe Ratio 3 Yr", "Standard Deviation 3 Yr", "Price (mo-end)"],
                        [[f"Fund {t}", t, 0.5 + i * 0.05, 15 - i * 0.2, 19 + i] for i, t in enumerate(TICK)], [1.37, 12.4, "-"])
    sec = write_export(d / "sectors.xlsx", "Sector Exposure", ["Technology", "Financial Services", "Healthcare"],
                       [[f"Fund {t}", t, 20 + i * 4, 18 - i, 12] for i, t in enumerate(TICK)], [40.45, 11.2, 9.26])
    med = write_export(d / "medalist.xlsx", "Medalist Rating Input Data",
                       ["Average Brand Longest Manager Tenure - Ranked - Active People Pillar",
                        "Average Brand Longest Manager Tenure - Ranked - Parent Pillar"],
                       [[f"Fund {t}", t, 0.1 + i * 0.05, 0.2 + i * 0.05] for i, t in enumerate(TICK)], ["-", "-"])
    return [ret, risk, sec, med]


def test_bottom_benchmark_metadata_and_no_false_conflicts(tmp_path):
    log = QualityLog()
    ds = load_dataset(make_files(tmp_path / "in"), log)
    assert len(ds.df) == 7 and ds.df["name"].notna().all()
    assert ds.benchmark_name == "S&P 500 (TR) (1970)"
    assert ds.benchmark["sharpe_3y"] == 1.37 and ds.benchmark["sec_technology"] == 40.45
    assert {p["kind"] for p in ds.pages} == {"Annualized Returns Monthly", "Risk Analysis", "Sector Exposure",
                                             "Medalist Rating Input Data"}
    assert all(p["print_date"] == "10-22-2026" for p in ds.pages) and ds.list_name == "TEST LIST"
    # people vs parent tenure ranks and current vs month-end price are distinct data points
    assert ds.df.at["BBBX", "med_brand_tenure_rk"] == pytest.approx(0.15)
    assert ds.df.at["BBBX", "med_parent_tenure_rk"] == pytest.approx(0.25)
    assert ds.df.at["BBBX", "price_now"] == 21 and ds.df.at["BBBX", "price_mo"] == 20
    assert not [e for e in log.entries if "differs between input files" in e.message]


def test_conflicts_are_summarised_once_per_column(tmp_path):
    files = make_files(tmp_path / "in")
    extra = write_export(tmp_path / "in" / "zz_returns_v2.xlsx", "Annualized Returns Monthly",
                         ["Tot Ret 12 Mo (mo-end)"], [[f"Fund {t}", t, 99.0] for t in TICK], [15.7])
    log = QualityLog()
    ds = load_dataset(files + [extra], log)
    msgs = [e for e in log.entries if "differs between input files" in e.message]
    assert len(msgs) == 1 and "for 7 fund(s)" in msgs[0].message
    assert ds.df.at["AAAX", "nav_ret_12m"] == 10              # first file in name order wins


def test_result_does_not_depend_on_file_order(tmp_path):
    files = make_files(tmp_path / "in")

    def digest(paths):
        res = results_dict(run(paths)[0])
        return hashlib.sha256(json.dumps(res["funds"], sort_keys=True, default=str).encode()).hexdigest()
    assert digest(files) == digest(list(reversed(files)))


def test_sector_active_weights_and_themes(tmp_path):
    c = run(make_files(tmp_path / "in"))[0]
    assert c.df.at["AAAX", "sec_technology_active"] == pytest.approx(20 - 40.45)
    assert c.df.at["AAAX", "sector_under"] == "Technology"
    assert c.df.at["AAAX", "sector_tilt"] == "Financial Services"


def test_unlabelled_name_column_in_paste(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([None, None, "Ticker", "Tot Ret"])
    ws.append([None, None, None, "12 Mo (mo-end)"])
    ws.append([None, None, "-", 15.7])
    for i, t in enumerate(TICK):
        ws.append([i + 1, f"Fund {t} Inst", t, 10 + i])
    wb.save(tmp_path / "basics.xlsx")
    ds = load_dataset([tmp_path / "basics.xlsx"], QualityLog())
    assert ds.df.at["CCCX", "name"] == "Fund CCCX Inst"
    assert ds.benchmark["nav_ret_12m"] == 15.7

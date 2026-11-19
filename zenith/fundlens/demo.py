"""Synthetic, Morningstar-shaped demo exports (seeded; no licensed data).

Real Advisor Workstation exports are licensed and never ship with Zenith: you
upload your own and they are parsed in a temp folder that is deleted at once.
These generators make a realistic stand-in — two stacked blocks (basics + risk),
funds in columns, a label column — so the page, the tests and new users can
exercise every feature without any real fund data.
"""

from __future__ import annotations

import io
import random

import openpyxl

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL",
           "MMM", "NNN", "OOO", "PPP", "QQQ1", "RRR", "SSS", "TTT"]


def _fund_values(rng: random.Random, i: int, young_from: int) -> dict:
    young = i >= young_from
    def h(base, spread, horizons=("3 Yr", "5 Yr", "10 Yr", "15 Yr")):
        return {hz: (round(base + rng.uniform(-spread, spread), 2) if not (young and hz != "3 Yr") else "-")
                for hz in horizons}
    return {"young": young, "ret": h(10, 4), "sharpe": h(0.6, 0.3), "std": h(16, 3),
            "down": h(95, 15), "up": h(95, 15), "beta": h(1.0, 0.25), "r2": h(70, 20)}


def transposed_workbook(n: int = 12, seed: int = 7, dup: bool = True) -> openpyxl.Workbook:
    """Advisor-Workstation-style export: basics block then risk block, funds in columns."""
    rng = random.Random(seed)
    tick = TICKERS[:n]
    young_from = max(1, n - 2)
    funds = [_fund_values(rng, i, young_from) for i in range(n)]
    cols = list(range(n)) + ([3] if dup else [])     # duplicate of fund #3 (thin record) at the end
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MSEXP.xls"
    r = 2

    def row(label, vals):
        nonlocal r
        ws.cell(r, 2, label)
        for k, v in enumerate(vals):
            ws.cell(r, 3 + k, v)
        r += 1

    row("Name ", [f"Fund {tick[i]}" + (" Dup Class" if dup and k == n else "") for k, i in enumerate(cols)])
    row("Ticker", [tick[i] for i in cols])
    row("Type", ["ETF"] * len(cols))
    row("Morningstar Category", ["US Fund Mid-Cap Value"] * len(cols))
    row("Fund Objective", ["Growth" if i % 3 else "Equity-Income" for i in cols])
    row("Strategic Beta Group", ["Dividend" if i % 2 else "12" for i in cols])
    row("Index Weighting", ["Equal,Value" if i % 4 == 0 else "-" for i in cols])
    row("Total Assets $MM", [round(100 + 900 * rng.random()) for _ in cols])
    row("Prospectus Net Expense Ratio", [round(0.05 + 0.05 * i, 2) for i in cols])
    row("Leveraged Fund", ["No"] * len(cols))
    r += 1
    row("Name ", [f"Fund {tick[i]}" for i in cols])
    row("Ticker", [tick[i] for i in cols])
    row("Ret Date (Qtr-end)", [46295] * len(cols))
    for hz in ("3 Yr", "5 Yr", "10 Yr", "15 Yr"):
        row(f"NAV Tot Return {hz} (Qtr-end) ", [funds[i]["ret"][hz] if not (dup and k == n) else "-"
                                                for k, i in enumerate(cols)])
        row(f"Sharpe Ratio {hz}", [funds[i]["sharpe"][hz] for i in cols])
        row(f"Standard Deviation {hz}", [funds[i]["std"][hz] for i in cols])
        row(f"Downside Capture Ratio {hz}", [funds[i]["down"][hz] for i in cols])
        row(f"Upside Capture Ratio {hz}", [funds[i]["up"][hz] for i in cols])
        row(f"Beta {hz}", [funds[i]["beta"][hz] for i in cols])
        row(f"R-Squared {hz}", [funds[i]["r2"][hz] for i in cols])
    for y in range(2025, 2015, -1):
        row(f"NAV Annl Return {y}", [round(rng.uniform(-15, 30), 2) if not (funds[i]["young"] and y < 2024) else "-"
                                     for i in cols])
    row("Some Unknown Field", ["x"] * len(cols))
    return wb


def approved_workbook(tickers: list[str]) -> openpyxl.Workbook:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Ticker", "Morningstar Category", "Available for Solicitation"])
    for t in tickers:
        ws.append([f"Fund {t}", t, "Mid-Cap Value", "Yes"])
    ws.append([])
    ws.append(["INVESTORS SHOULD CONSIDER THE INVESTMENT OBJECTIVES... (disclaimer)"])
    ws.append(["Securities offered by Example LLC. Member FINRA and SIPC"])
    return wb


def _bytes(wb: openpyxl.Workbook) -> bytes:
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def demo_files(n: int = 16, seed: int = 7) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
    """(export, approved list) as (filename, bytes) pairs — what the page's uploader would give."""
    data = ("synthetic demo - mid cap value - ETFs.xlsx", _bytes(transposed_workbook(n, seed)))
    appr = ("synthetic demo - approved funds.xlsx",
            _bytes(approved_workbook([t for i, t in enumerate(TICKERS[:n]) if i % 2 == 0] + ["ZZZ"])))
    return data, appr


def write_transposed(path, n: int = 12, seed: int = 7, dup: bool = True):
    transposed_workbook(n, seed, dup).save(path)
    return path


def write_approved(path, tickers: list[str]):
    approved_workbook(tickers).save(path)
    return path

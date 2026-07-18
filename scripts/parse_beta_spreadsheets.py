"""One-off parser: Morningstar strategic-beta ETF exports -> data/fmom/etf_catalog.json.

  python scripts/parse_beta_spreadsheets.py [source_dir]

The "BETA - *.xls" files are Excel-2003 SpreadsheetML (XML, not real xls).
Quirks handled here (verified against the 2026-06-29 export):
  * sparse cells carry ss:Index attributes, so cell position must be tracked;
  * data rows insert an extra numeric group-code column after "Strategic Beta",
    shifting "Strategic Beta Group" one to the right (except sheets like
    DIVIDEND where the name sits at the header index directly);
  * an ETF can appear on several sheets (e.g. MULTIFACTOR 1 and QUALITY) —
    dedupe on ticker, union groups/sheets.

The committed catalog feeds the FACTOR MOMENTUM feature's ETF-proxy picks and
the Man-style composite membership rules. Keep this script around for future
refreshes; the JSON is also safe to hand-edit.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
IDX = "{urn:schemas-microsoft-com:office:spreadsheet}Index"

DEFAULT_SRC = Path(r"C:\Users\User\Downloads\stylefactormomentumreiteration")
OUT = Path(__file__).resolve().parent.parent / "data" / "fmom" / "etf_catalog.json"


def _row_values(row) -> list[str]:
    """Cell texts with ss:Index-aware positioning (sparse rows)."""
    vals: list[str] = []
    pos = 0
    for cell in row.findall("ss:Cell", NS):
        if IDX in cell.attrib:
            pos = int(cell.attrib[IDX]) - 1
        data = cell.find("ss:Data", NS)
        while len(vals) < pos:
            vals.append("")
        vals.append(data.text if data is not None and data.text else "")
        pos += 1
    return vals


def _num(v: str) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_file(path: Path) -> list[dict]:
    sheet = path.stem.replace("BETA - ", "").strip()
    rows = ET.parse(path).findall(".//ss:Worksheet/ss:Table/ss:Row", NS)
    out: list[dict] = []
    hdr: list[str] | None = None
    gi = ci = ai = ei = 0
    for row in rows:
        vals = _row_values(row)
        if vals and vals[0] == "Name ":
            hdr = vals
            gi = hdr.index("Strategic Beta Group")
            ci = hdr.index("Morningstar Category")
            ai = hdr.index("Total Assets $MM")
            ei = hdr.index("Prospectus Net Expense Ratio")
            continue
        if not hdr or len(vals) <= gi:
            continue
        ticker = (vals[1] or "").strip()
        if not ticker or not ticker.isupper() or len(ticker) > 6:
            continue                                   # not a data row
        group_raw = (vals[gi] or "").strip()
        if group_raw.replace(".", "").isdigit():       # numeric code column quirk
            group_raw = (vals[gi + 1] or "").strip() if len(vals) > gi + 1 else ""
        groups = sorted({g.strip() for g in group_raw.split(",")
                         if g.strip() and g.strip() != "-"})
        out.append({
            "ticker": ticker,
            "name": (vals[0] or "").strip(),
            "category": (vals[ci] or "").strip(),
            "groups": groups,
            "sheet": sheet,
            "aum_m": _num(vals[ai]) if len(vals) > ai else None,
            "er": _num(vals[ei]) if len(vals) > ei else None,
        })
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    files = sorted(src.glob("BETA*.xls"))
    if not files:
        raise SystemExit(f"no 'BETA - *.xls' files found in {src}")

    merged: dict[str, dict] = {}
    for f in files:
        for rec in parse_file(f):
            t = rec["ticker"]
            if t in merged:
                cur = merged[t]
                cur["groups"] = sorted(set(cur["groups"]) | set(rec["groups"]))
                cur["sheets"] = sorted(set(cur["sheets"]) | {rec["sheet"]})
                cur["aum_m"] = cur["aum_m"] if cur["aum_m"] is not None else rec["aum_m"]
                cur["er"] = cur["er"] if cur["er"] is not None else rec["er"]
            else:
                rec["sheets"] = [rec.pop("sheet")]
                merged[t] = rec

    etfs = sorted(merged.values(), key=lambda r: r["ticker"])
    obj = {
        "generated": date.today().isoformat(),
        "source": "Morningstar US ETF Universe strategic-beta exports (BETA - *.xls)",
        "n": len(etfs),
        "etfs": etfs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    n_grouped = sum(1 for e in etfs if e["groups"])
    print(f"[etf_catalog] wrote {len(etfs)} ETFs ({n_grouped} with group tags) -> {OUT}")


if __name__ == "__main__":
    main()

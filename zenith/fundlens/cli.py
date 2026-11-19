"""Command line: `python -m fundlens run --data export.xlsx --approved approved.xlsx --profile us_equity_style --out out/`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dictionary import load_dictionary
from .engine import run
from .ingest import load_dataset
from .profile import list_profiles
from .quality import QualityLog
from .render_xlsx import write_workbook
from .results import results_dict


def cmd_run(a) -> int:
    shortlist = [t.strip().upper() for t in (a.shortlist or "").split(",") if t.strip()]
    comps = run(a.data, profile=a.profile, approved_path=a.approved, shortlist=shortlist,
                approved_require_solicitation=a.require_solicitation)
    out = Path(a.out)
    summary = []
    for c in comps:
        path = write_workbook(c, out, a.palette)
        res = results_dict(c)
        if a.json:
            (out / (path.stem + ".json")).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        summary.append({"workbook": str(path), "kind": c.kind, "funds_ranked": res["n_ranked"],
                        "top5": [r["ticker"] for r in res["funds"][:5]],
                        "warnings": [e["message"] for e in res["quality"] if e["severity"] in ("warning", "error")]})
    print(json.dumps(summary, indent=1))
    return 0


def cmd_inspect(a) -> int:
    log = QualityLog()
    ds = load_dataset(a.data, log, load_dictionary())
    info = {"funds": len(ds.df), "layouts": ds.layouts, "blocks": ds.n_blocks,
            "returns_basis": sorted(ds.freqs), "duplicates": ds.duplicates,
            "mapped_columns": sorted(c for c in ds.df.columns if not c.startswith("_")),
            "unmapped_labels": ds.unmapped,
            "log": [f"{e.severity}: {e.area}: {e.ticker} {e.message}" for e in log.sorted()]}
    print(json.dumps(info, indent=1, default=str))
    return 0


def cmd_profiles(a) -> int:
    print(json.dumps(list_profiles(), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fundlens", description="Deterministic fund comparison workbooks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate workbook(s)")
    r.add_argument("--data", nargs="+", required=True, help="Morningstar export file(s)")
    r.add_argument("--approved", help="approved-funds list (optional)")
    r.add_argument("--profile", default="us_equity_style", help="profile id or path to a .yaml")
    r.add_argument("--out", default="output", help="output folder")
    r.add_argument("--palette", default="standard", choices=["standard", "colorblind"])
    r.add_argument("--shortlist", help="comma-separated tickers for a Shortlist tab")
    r.add_argument("--require-solicitation", action="store_true", help="approved funds must be 'Available for Solicitation = Yes'")
    r.add_argument("--json", action="store_true", help="also write a results .json next to each workbook")
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("inspect", help="parse files and report mapping / unmapped labels (no output)")
    i.add_argument("--data", nargs="+", required=True)
    i.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("profiles", help="list available profiles")
    p.set_defaults(fn=cmd_profiles)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

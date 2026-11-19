"""Read Morningstar exports (any orientation, stacked blocks, several files) into one table.

Supported containers: .xlsx/.xlsm, Excel-2003 SpreadsheetML saved as .xls (a common
Morningstar quirk), genuine .xls (xlrd), .csv.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .dictionary import Dictionary, Match, load_dictionary, norm_label
from .pasted import find_pages, resolve_names
from .quality import QualityLog

MISSING = {"", "-", "--", "—", "–", "n/a", "na", "nan", "none", "null", "not available"}
TICKER_RX = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
TICKER_LABELS = {"ticker", "symbol", "ticker symbol"}
EXCEL_EPOCH = dt.date(1899, 12, 30)


# --------------------------------------------------------------------------- reading
@dataclass
class Grid:
    file: str
    sheet: str
    rows: list[list]          # raw cell values, row-major, ragged rows allowed


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _trim(rows: list[list]) -> list[list]:
    out = []
    for r in rows:
        r = list(r)
        while r and (r[-1] is None or (isinstance(r[-1], str) and not r[-1].strip())):
            r.pop()
        out.append(r)
    while out and not out[-1]:
        out.pop()
    return out


def _read_xlsx(path: Path) -> list[Grid]:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    grids = [Grid(path.name, ws.title, _trim(ws.iter_rows(values_only=True))) for ws in wb.worksheets]
    wb.close()
    return grids


def _read_spreadsheetml(data: bytes, name: str) -> list[Grid]:
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = ET.fromstring(data)
    grids = []
    for ws in root.findall("ss:Worksheet", ns):
        rows: list[list] = []
        table = ws.find("ss:Table", ns)
        if table is None:
            continue
        for row in table.findall("ss:Row", ns):
            idx = row.get(f"{{{ns['ss']}}}Index")
            if idx:                                   # sparse rows
                while len(rows) < int(idx) - 1:
                    rows.append([])
            vals: list = []
            for cell in row.findall("ss:Cell", ns):
                ci = cell.get(f"{{{ns['ss']}}}Index")
                if ci:                                # sparse cells
                    while len(vals) < int(ci) - 1:
                        vals.append(None)
                d = cell.find("ss:Data", ns)
                v = None
                if d is not None:
                    v = "".join(d.itertext())
                    t = d.get(f"{{{ns['ss']}}}Type")
                    if t == "Number":
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                vals.append(v)
            rows.append(vals)
        grids.append(Grid(name, ws.get(f"{{{ns['ss']}}}Name") or "Sheet", _trim(rows)))
    return grids


def _read_xls(path: Path) -> list[Grid]:
    data = path.read_bytes()
    head = data[:2048].lstrip().lower()
    if head.startswith(b"<?xml") or b"urn:schemas-microsoft-com:office:spreadsheet" in data[:4096]:
        return _read_spreadsheetml(data, path.name)
    if head.startswith(b"<html") or b"<table" in head:
        tables = pd.read_html(io.BytesIO(data), header=None)
        return [Grid(path.name, f"Table{i+1}", _trim(t.astype(object).where(t.notna(), None).values.tolist()))
                for i, t in enumerate(tables)]
    import xlrd
    book = xlrd.open_workbook(file_contents=data)
    return [Grid(path.name, sh.name, _trim([sh.row_values(r) for r in range(sh.nrows)])) for sh in book.sheets()]


def _read_csv(path: Path) -> list[Grid]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rows = list(csv.reader(io.StringIO(text)))
    return [Grid(path.name, path.stem, _trim([[c if c != "" else None for c in r] for r in rows]))]


def read_grids(path: str | Path) -> list[Grid]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        return _read_xlsx(path)
    if ext == ".xls":
        return _read_xls(path)
    if ext in (".csv", ".txt"):
        return _read_csv(path)
    raise ValueError(f"Unsupported file type: {path.name}")


# --------------------------------------------------------------------------- values
def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and v.strip().lower() in MISSING


def clean_text(v):
    if is_missing(v):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return re.sub(r"\s+", " ", str(v)).strip()


def to_number(v):
    if is_missing(v):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return None
    s = str(v).strip().replace(",", "").replace("%", "").replace("$", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def to_date(v):
    if is_missing(v):
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, (int, float)) and 10000 < v < 80000:
        return EXCEL_EPOCH + dt.timedelta(days=int(v))
    s = str(v).strip()
    n = to_number(s)
    if n is not None and 10000 < n < 80000:
        return EXCEL_EPOCH + dt.timedelta(days=int(n))
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def coerce(match: Match, v):
    kind = match.metric.kind
    if kind in ("id", "text"):
        t = clean_text(v)
        if match.metric.id == "ticker" and t:
            t = t.upper()
        return t
    if kind == "date":
        return to_date(v)
    return to_number(v)


def is_ticker(v) -> bool:
    return isinstance(v, str) and bool(TICKER_RX.match(v.strip()))


# --------------------------------------------------------------------------- layout
@dataclass
class RowMap:
    """For 'Original Data': which canonical column each source row (transposed) feeds."""
    grid_index: int
    row: int
    label: str
    match: Match | None
    block: int


@dataclass
class Dataset:
    df: pd.DataFrame                                   # index = ticker, canonical columns
    grids: list[Grid] = field(default_factory=list)
    rowmaps: list[RowMap] = field(default_factory=list)
    layouts: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)   # name -> sha256[:12]
    freqs: set = field(default_factory=set)
    unmapped: list[str] = field(default_factory=list)
    n_blocks: int = 0
    duplicates: list[str] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)        # funds-in-rows tables: kind, header row, tickers, listed
    benchmark: dict = field(default_factory=dict)          # canonical column -> benchmark value (pasted '-' rows)
    benchmark_name: str | None = None
    list_name: str | None = None                           # export title row, e.g. 'LARGE VALUE BLEND LIST'
    name_info: dict = field(default_factory=dict)
    colmaps: list[dict] = field(default_factory=list)      # for Original Data colouring of funds-in-rows tables

    # convenience
    def mode(self, col: str):
        if col not in self.df:
            return None
        s = self.df[col].dropna()
        if s.empty:
            return None
        c = Counter(s.tolist())
        return sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _find_ticker_anchor(rows: list[list]):
    """Return ('transposed', label_col) or ('rows', header_row, ticker_col) or None."""
    best = None
    for r, row in enumerate(rows[:400]):
        for c, v in enumerate(row[:60]):
            if norm_label(v) not in TICKER_LABELS:
                continue
            right = sum(is_ticker(x) for x in row[c + 1:])
            below = sum(is_ticker(rows[rr][c]) if c < len(rows[rr]) else 0
                        for rr in range(r + 1, min(len(rows), r + 2000)))
            cand = ("transposed", c) if right >= below else ("rows", r, c)
            score = max(right, below)
            if score >= 1 and (best is None or score > best[0]):
                best = (score, cand)
        if best and best[0] >= 2:
            break
    return best[1] if best else None


def _parse_transposed(grid: Grid, gi: int, label_col: int, d: Dictionary, log: QualityLog,
                      recs: dict, order: list, rowmaps: list, block_base: int, unmapped: Counter, freqs: set) -> int:
    rows = grid.rows
    ticker_rows = [r for r, row in enumerate(rows)
                   if len(row) > label_col and norm_label(row[label_col]) in TICKER_LABELS]
    # Each block starts at its ticker row, or the 'Name' row immediately above it.
    starts = []
    for tr in ticker_rows:
        s = tr
        if tr > 0 and len(rows[tr - 1]) > label_col and norm_label(rows[tr - 1][label_col]) in ("name", "fund name"):
            s = tr - 1
        starts.append(s)
    starts.append(len(rows))
    for b, tr in enumerate(ticker_rows):
        # column -> fund key, with occurrence suffix for repeated tickers (e.g. two share classes)
        seen: Counter = Counter()
        colkey: dict[int, str] = {}
        for c in range(label_col + 1, len(rows[tr])):
            t = rows[tr][c]
            if is_ticker(t):
                t = t.strip().upper()
                seen[t] += 1
                key = t if seen[t] == 1 else f"{t}#{seen[t]}"
                colkey[c] = key
                if key not in recs:
                    recs[key] = {"ticker": t}
                    order.append(key)
        for r in range(starts[b], starts[b + 1]):
            row = rows[r]
            if len(row) <= label_col or is_missing(row[label_col]):
                continue
            label = row[label_col]
            m = d.match(label)
            rowmaps.append(RowMap(gi, r, str(label), m, block_base + b))
            if m is None:
                unmapped[norm_label(label)] += 1
                continue
            if m.freq and m.metric.horizons:
                freqs.add(m.freq)
            for c, key in colkey.items():
                v = coerce(m, row[c]) if c < len(row) else None
                _put(recs[key], m.column, v, key, log)
    return len(ticker_rows)


def _put(rec: dict, col: str, v, key: str, log: QualityLog):
    if v is None:
        rec.setdefault(col, None)
        return
    old = rec.get(col)
    if old is None:
        rec[col] = v
    elif old != v and col not in ("name",):
        c = log.conflicts.setdefault(col, [0, None])
        c[0] += 1
        if c[1] is None:
            c[1] = (key.split("#")[0], old, v)


def _parse_rows(grid: Grid, gi: int, tcol: int, d: Dictionary, log: QualityLog, recs: dict, order: list,
                unmapped: Counter, freqs: set, ds: "Dataset") -> int:
    """One or more funds-in-rows tables (incl. Morningstar grids pasted page by page)."""
    pages = find_pages(grid.rows, tcol, d)
    for pg in pages:
        for lab, m in zip(pg.labels, pg.matches):
            if m is None and lab and norm_label(lab) not in TICKER_LABELS:
                unmapped[norm_label(lab)] += 1
            elif m is not None and m.freq and m.metric.horizons:
                freqs.add(m.freq)
        seen: Counter = Counter()
        for r, t in zip(pg.data_rows, pg.tickers):
            row = grid.rows[r]
            seen[t] += 1
            key = t if seen[t] == 1 else f"{t}#{seen[t]}"
            if key not in recs:
                recs[key] = {"ticker": t}
                order.append(key)
            for i, m in enumerate(pg.matches):
                if m is not None and i < len(row) and m.metric.id != "ticker":
                    _put(recs[key], m.column, coerce(m, row[i]), key, log)
        for i, v in pg.benchmark.items():
            m = pg.matches[i] if i < len(pg.matches) else None
            if m is not None and m.metric.kind in ("directional", "characteristic"):
                bv = coerce(m, v)
                if bv is not None:
                    ds.benchmark.setdefault(m.column, bv)
        if pg.title and not ds.benchmark_name:
            ds.benchmark_name = pg.title
        ds.pages.append({"file": grid.file, "kind": pg.kind, "row": pg.header_row + 1, "tickers": list(pg.tickers),
                         "listed": pg.n_listed, "print_date": pg.print_date, "list_name": pg.list_name,
                         "bench": len([v for v in pg.benchmark.values() if isinstance(v, (int, float))])})
        if pg.list_name and not ds.list_name:
            ds.list_name = pg.list_name
        ds.colmaps.append({"grid": gi, "rows": list(pg.data_rows),
                           "cols": {i: m for i, m in enumerate(pg.matches) if m is not None}})
    # names live in a separate pane of the pasted grid: attach only provably-unique matches
    if any(pg.names for pg in pages):
        names, info = resolve_names(pages)
        for t, nm in names.items():
            if t in recs and not recs[t].get("name"):
                recs[t]["name"] = nm
        ds.name_info = info
    return len(pages)


def load_dataset(paths: list[str | Path], log: QualityLog, d: Dictionary | None = None) -> Dataset:
    d = d or load_dictionary()
    recs: dict[str, dict] = {}
    order: list[str] = []
    ds = Dataset(df=pd.DataFrame())
    unmapped: Counter = Counter()
    # Deterministic: the same set of files gives the same result whatever order they were uploaded in.
    paths = sorted((Path(p) for p in paths), key=lambda p: (p.name.lower(), file_hash(p)))
    log.conflicts.clear()
    for p in paths:
        ds.files[p.name] = file_hash(p)
        for grid in read_grids(p):
            if not grid.rows:
                continue
            anchor = _find_ticker_anchor(grid.rows)
            if anchor is None:
                log.add("warning", "Ingest", f"No ticker row/column found in {p.name} / {grid.sheet}; sheet skipped")
                continue
            gi = len(ds.grids)
            ds.grids.append(grid)
            if anchor[0] == "transposed":
                nb = _parse_transposed(grid, gi, anchor[1], d, log, recs, order, ds.rowmaps,
                                       ds.n_blocks, unmapped, ds.freqs)
                ds.n_blocks += nb
                ds.layouts.append(f"{p.name} / {grid.sheet}: funds in columns, {nb} stacked block(s)")
            else:
                nb = _parse_rows(grid, gi, anchor[2], d, log, recs, order, unmapped, ds.freqs, ds)
                ds.n_blocks += nb
                ds.layouts.append(f"{p.name} / {grid.sheet}: funds in rows, {nb} table(s)"
                                  + (" (pasted Morningstar grid)" if ds.name_info else ""))
    ds.unmapped = sorted(unmapped)
    for u in ds.unmapped:
        log.add("info", "Dictionary", f"Unmapped label kept in Original Data only: '{u}'")
    _report_pages(ds, log)
    for col, (n, (t, kept, other)) in sorted(log.conflicts.items()):
        log.add("info", "Ingest", f"'{d.column_label(col)}' differs between input files for {n} fund(s); kept the value "
                                  f"from the first file in name order (e.g. {t}: kept {kept!r}, other {other!r})")
    log.conflicts.clear()
    if len(ds.freqs) > 1:
        log.add("warning", "Ingest", "Both quarter-end and month-end returns found; as-of dates differ across fields")

    # ---- de-duplicate repeated tickers: keep the record with the most data
    keep: dict[str, str] = {}
    for key in order:
        t = recs[key]["ticker"]
        if t not in keep:
            keep[t] = key
            continue
        a, b = recs[keep[t]], recs[key]
        na = sum(v is not None for v in a.values())
        nb = sum(v is not None for v in b.values())
        winner = keep[t] if na >= nb else key
        loser = key if winner == keep[t] else keep[t]
        keep[t] = winner
        ds.duplicates.append(t)
        log.add("warning", "Duplicates",
                f"Ticker {t} appears more than once ('{recs[loser].get('name')}' vs '{recs[winner].get('name')}'); "
                f"kept the record with more data ({max(na, nb)} vs {min(na, nb)} fields)", t)
    rows = []
    for t, key in keep.items():
        r = dict(recs[key])
        r["ticker"] = t
        r["_source_key"] = key
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ticker", drop=False)
    if "name" not in df:
        df["name"] = None                               # e.g. pasted grids where no name could be matched
    df.index.name = None
    ds.df = df
    return ds


# --------------------------------------------------------------------------- approved list
@dataclass
class ApprovedList:
    tickers: list[str]
    rows: dict[str, dict]           # ticker -> {normalised header: value}
    file: str
    hash: str
    ignored_rows: int


def load_approved(path: str | Path, log: QualityLog) -> ApprovedList:
    path = Path(path)
    tickers: list[str] = []
    rows: dict[str, dict] = {}
    ignored = 0
    for grid in read_grids(path):
        anchor = _find_ticker_anchor(grid.rows)
        if anchor is None:
            continue
        if anchor[0] == "rows":
            hdr = [norm_label(h) for h in grid.rows[anchor[1]]]
            tcol = anchor[2]
            for row in grid.rows[anchor[1] + 1:]:
                if tcol < len(row) and is_ticker(row[tcol]):
                    t = row[tcol].strip().upper()
                    if t not in rows:
                        tickers.append(t)
                        rows[t] = {h: clean_text(v) for h, v in zip(hdr, row) if h}
                elif any(not is_missing(v) for v in row):
                    ignored += 1
        else:                                          # funds in columns
            r = next(i for i, row in enumerate(grid.rows)
                     if len(row) > anchor[1] and norm_label(row[anchor[1]]) in TICKER_LABELS)
            for v in grid.rows[r][anchor[1] + 1:]:
                if is_ticker(v) and v.strip().upper() not in rows:
                    tickers.append(v.strip().upper())
                    rows[v.strip().upper()] = {}
    if not tickers:
        log.add("error", "Approved list", f"No tickers found in {path.name}")
    else:
        log.add("info", "Approved list",
                f"{len(tickers)} approved tickers read from {path.name}; {ignored} non-fund rows (headers/disclaimers) ignored")
    return ApprovedList(tickers, rows, path.name, file_hash(path), ignored)


def _report_pages(ds: "Dataset", log: QualityLog) -> None:
    if len(ds.pages) > 1:
        union = set().union(*[set(pg["tickers"]) for pg in ds.pages])
        for pg in ds.pages:
            miss = sorted(union - set(pg["tickers"]))
            if miss and not pg["listed"]:
                log.add("warning", "Coverage", f"{pg['kind']} ({pg['file']}) is missing {len(miss)} of the {len(union)} funds "
                                               f"found in the other files: {', '.join(miss[:15])}{' ...' if len(miss) > 15 else ''}")
    pasted = [pg for pg in ds.pages if pg["listed"]]
    if not pasted:
        return
    listed = max(pg["listed"] for pg in pasted)
    short = [pg for pg in pasted if len(pg["tickers"]) < pg["listed"]]
    if short:
        allt = set().union(*[set(pg["tickers"]) for pg in ds.pages])
        complete = set.intersection(*[set(pg["tickers"]) for pg in ds.pages])
        txt = "; ".join(f"{pg['kind']} {len(pg['tickers'])}/{pg['listed']}" for pg in ds.pages)
        log.add("warning", "Coverage",
                f"The pasted Morningstar grid captured only part of the {listed}-fund list on some pages ({txt}). "
                f"{len(allt)} funds have some data, {len(complete)} have every page. The web grid only copies rows "
                "rendered on screen - zoom the browser out or page through fewer rows before copying. "
                "Funds missing pages are scored on less evidence (see Data Coverage tab).")
        if len(allt) < listed:
            log.add("warning", "Coverage", f"{listed - len(allt)} fund(s) in the name list have no data rows on any page.")
    ni = ds.name_info
    if ni:
        if ni.get("order_conflict"):
            log.add("warning", "Names", "Fund order differs between pasted pages, so names could not be matched to tickers.")
        elif ni.get("ambiguous"):
            log.add("warning", "Names",
                    f"Fund names were matched to {ni['resolved']} of {ni['n_tickers']} tickers. The pasted name list and data rows "
                    f"were captured with different gaps, so {ni['ambiguous']} names cannot be matched with certainty and are "
                    "shown as '(name not captured)'. Add any file with Name + Ticker (e.g. an export or approved list) "
                    "as a second data file to fill them.")

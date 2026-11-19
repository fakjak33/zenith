"""Write a Comparison to a committee-ready .xlsx.

Colours are computed by the engine and written as static fills, so the workbook looks
identical in Excel, Google Sheets and LibreOffice, supports target-direction metrics
(beta closest to 1.0) and group-relative colouring on the lens tabs.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .derive import annual_years
from .dictionary import HORIZON_LABEL, SCORING_HORIZONS
from .engine import Comparison
from .lenses import NOT_CLASSIFIED, group_values
from .dictionary import load_codes, norm_label
from .ingest import to_date
from .scoring import effective_direction, pct_rank

# --------------------------------------------------------------------------- palette
PALETTES = {
    "standard":   ((0xF8, 0x69, 0x6B), (0xFF, 0xEB, 0x84), (0x63, 0xBE, 0x7B)),   # red-yellow-green
    "colorblind": ((0xE6, 0x61, 0x01), (0xF7, 0xF7, 0xF7), (0x5E, 0x9C, 0xD3)),   # orange-white-blue
}
NEUTRAL = ((0xFF, 0xFF, 0xFF), (0xDD, 0xEB, 0xF7), (0x9D, 0xC3, 0xE6))
NAVY, INK, MUTED, GRID = "1F2A44", "1F2937", "6B7280", "D9DDE3"
MISSING_FILL = "F2F2F2"
BAND_COLORS = {"identity": "1F2A44", "composite": "0F4C5C", "returns": "2E5E8C", "risk_adjusted": "2F7D6B",
               "downside": "8C4A3A", "consistency": "6B5B95", "cost": "8A6D2F", "fit": "4A6274",
               "display": "595959", "pillars": "0F4C5C", "group": "1F2A44"}
FORMATS = {"pct": '0.00"%"', "ratio": "0.00", "int": "0", "money_mm": "#,##0", "score": "0.0",
           "cov": "0%", "num": "#,##0.00", "date": "yyyy-mm-dd", "text": "@", "rank": "0"}
FONT = "Calibri"
NAME_MISSING = "(name not captured)"
DATA_NOTICE = ("Source data: Morningstar, Inc. All rights reserved. Data reformatted and analysed by FundLens for "
               "internal research use only; not for redistribution.")


def _hex(rgb) -> str:
    return "".join(f"{int(round(c)):02X}" for c in rgb)


def _interp(stops, s: float) -> str:
    s = min(100.0, max(0.0, s)) / 100.0
    lo, mid, hi = stops
    if s <= 0.5:
        a, b, t = lo, mid, s / 0.5
    else:
        a, b, t = mid, hi, (s - 0.5) / 0.5
    return _hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


class Styles:
    def __init__(self, palette: str):
        self.stops = PALETTES.get(palette, PALETTES["standard"])
        self._fills: dict[str, PatternFill] = {}
        self._fonts: dict[tuple, Font] = {}

    def fill(self, hex_: str) -> PatternFill:
        f = self._fills.get(hex_)
        if f is None:
            f = self._fills[hex_] = PatternFill("solid", start_color=hex_, end_color=hex_)
        return f

    def heat(self, score: float, neutral: bool = False) -> PatternFill:
        return self.fill(_interp(NEUTRAL if neutral else self.stops, round(score)))

    def font(self, bold=False, color=INK, size=10, italic=False) -> Font:
        k = (bold, color, size, italic)
        f = self._fonts.get(k)
        if f is None:
            f = self._fonts[k] = Font(name=FONT, bold=bold, color=color, size=size, italic=italic)
        return f


THIN = Side(style="thin", color=GRID)
MED = Side(style="medium", color="8A93A3")
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")
WRAP = Alignment(horizontal="left", vertical="top", wrap_text=True)
HDR_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)


# --------------------------------------------------------------------------- column spec
@dataclass
class Col:
    header: str
    get: Callable[[str], Any]
    fmt: str = "text"
    width: float = 10.0
    color: str | None = None          # 'score' | 'dir' | 'rank' | None
    direction: str = "higher"         # for 'dir': higher | lower | target | none(neutral)
    target: float | None = None
    band: str = ""                    # band label (merged header above)
    band_key: str = "display"
    key: str | None = None            # canonical column (used for the benchmark reference row)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _scope_scores(col: Col, values: list, groups: list | None, min_group: int = 3) -> list[float | None]:
    """0-100 colour position for each value, or None (no colour)."""
    if col.color is None:
        return [None] * len(values)
    if col.color == "score":
        return [None if v is None or isinstance(v, str) else float(v) for v in values]
    direction = "lower" if col.color == "rank" else (col.direction if col.direction != "none" else "higher")
    num = pd.Series([v if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan for v in values],
                    dtype=float)
    out = pd.Series(np.nan, index=num.index)
    if groups is None:
        scopes = [list(num.index)]
    else:
        gser = pd.Series(groups)
        sizes = gser.value_counts()
        small = [i for i in num.index if sizes[groups[i]] < min_group]
        scopes = [list(gser[gser == g].index) for g in sizes.index if sizes[g] >= min_group]
        if small:
            scopes.append(("ALL", small))
    for sc in scopes:
        if isinstance(sc, tuple):                      # small groups -> coloured vs whole table
            full = pct_rank(num, direction, col.target)
            for i in sc[1]:
                out[i] = full[i]
        else:
            sub = pct_rank(num.loc[sc], direction, col.target)
            out.loc[sc] = sub
    return [None if pd.isna(x) else float(x) for x in out]


# --------------------------------------------------------------------------- writer
class Writer:
    def __init__(self, c: Comparison, palette: str = "standard"):
        self.c = c
        self.st = Styles(palette)
        self.palette = palette
        self.codes = load_codes()
        self.wb = Workbook()
        self.wb.remove(self.wb.active)

    # ---- value helpers -----------------------------------------------------------
    def v(self, t: str, col: str):
        df = self.c.df
        if col in df.columns and t in df.index:
            val = _clean(df.at[t, col])
            if col == "name" and val is None:
                return NAME_MISSING
            return val
        if col == "name" and t in df.index:
            return NAME_MISSING
        return None

    def label(self, col: str) -> str:
        return self.c.d.column_label(col)

    def fmt_of(self, col: str) -> str:
        m, _, _ = self.c.d.column_meta(col)
        return m.format if m else "num"

    def dir_of(self, col: str) -> tuple[str, float | None]:
        m, _, _ = self.c.d.column_meta(col)
        if m is None:
            return "none", None
        if m.kind in ("text", "id", "date"):
            return "text", None
        return effective_direction(m.id, self.c.d, self.c.prof)

    def metric_col(self, col: str, band: str = "", band_key: str = "display", width: float = 9.0) -> Col:
        direction, target = self.dir_of(col)
        if direction == "text":
            return Col(self.label(col), lambda t, c=col: self.v(t, c), "text", 22, None, band=band, band_key=band_key)
        color = "dir"
        return Col(self.label(col), lambda t, c=col: self.v(t, c), self.fmt_of(col), width, color,
                   direction, target, band, band_key, key=col)

    def group_label(self, t: str, field: str) -> str:
        lz = next((x for x in self.c.prof["lenses"] if x["field"] == field), {})
        vals = group_values(self.v(t, field), field, bool(lz.get("explode")), self.codes, {})
        return ", ".join(vals)

    # ---- generic table -------------------------------------------------------------
    def table(self, ws: Worksheet, cols: list[Col], tickers: list[str], top: int = 1,
              groups: list | None = None, band_row: bool = True, freeze_cols: int = 0, bench: bool = False,
              group_col: int | None = None) -> int:
        st = self.st
        hdr_row = top + (1 if band_row else 0)
        # band row
        if band_row:
            i = 0
            while i < len(cols):
                j = i
                while j + 1 < len(cols) and cols[j + 1].band == cols[i].band:
                    j += 1
                c1 = ws.cell(top, i + 1, cols[i].band or None)
                fill = st.fill(BAND_COLORS.get(cols[i].band_key, NAVY)) if cols[i].band else st.fill("FFFFFF")
                for k in range(i, j + 1):
                    ws.cell(top, k + 1).fill = fill
                c1.font = st.font(bold=True, color="FFFFFF", size=9)
                c1.alignment = CENTER
                if j > i and cols[i].band:
                    ws.merge_cells(start_row=top, start_column=i + 1, end_row=top, end_column=j + 1)
                i = j + 1
        # header
        for k, col in enumerate(cols, start=1):
            c = ws.cell(hdr_row, k, col.header)
            c.font = st.font(bold=True, color="FFFFFF", size=9)
            c.fill = st.fill(NAVY)
            c.alignment = HDR_ALIGN
            c.border = Border(bottom=MED)
            ws.column_dimensions[get_column_letter(k)].width = col.width
        ws.row_dimensions[hdr_row].height = 42
        # values + colours
        data = [[_clean(col.get(t)) for t in tickers] for col in cols]
        heat = [_scope_scores(col, vals, groups) for col, vals in zip(cols, data)]
        r0 = hdr_row + 1
        prev_group = None
        band_toggle = False
        for i, t in enumerate(tickers):
            r = r0 + i
            new_group = groups is not None and groups[i] != prev_group
            if new_group:
                band_toggle = not band_toggle
                prev_group = groups[i]
            for k, col in enumerate(cols):
                v = data[k][i]
                cell = ws.cell(r, k + 1)
                if v is None:
                    if col.color:
                        cell.value = "—"
                        cell.fill = st.fill(MISSING_FILL)
                        cell.font = st.font(color="A0A6B0")
                        cell.alignment = CENTER
                    continue
                if isinstance(v, (dt.date, dt.datetime)):
                    cell.value = v
                    cell.number_format = FORMATS["date"]
                elif isinstance(v, bool):
                    cell.value = "Yes" if v else ""
                else:
                    cell.value = v
                    if not isinstance(v, str):
                        cell.number_format = FORMATS.get(col.fmt, "General")
                cell.font = st.font()
                cell.alignment = LEFT if isinstance(v, str) and col.width >= 14 else CENTER
                h = heat[k][i]
                if h is not None:
                    cell.fill = st.heat(h, neutral=(col.color == "dir" and col.direction == "none"))
                elif groups is not None and band_toggle and (group_col is None or k <= group_col):
                    cell.fill = st.fill("EEF2F8")
            if new_group and i > 0:
                for k in range(len(cols)):
                    cell = ws.cell(r, k + 1)
                    cell.border = Border(top=MED)
        last = r0 + len(tickers) - 1
        if bench:
            self.bench_row(ws, cols, last + 2)
        if tickers:
            ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(len(cols))}{last}"
        ws.freeze_panes = ws.cell(hdr_row + 1, freeze_cols + 1)
        ws.print_title_rows = f"{top}:{hdr_row}"
        self._print_setup(ws)
        return last

    def bench_row(self, ws: Worksheet, cols: list[Col], r: int):
        b = self.c.dataset.benchmark
        if not b or not any(c.key in b for c in cols):
            return
        st = self.st
        name_col = next((k for k, c in enumerate(cols) if c.header in ("Fund", "Group / Theme")), 0)
        for k, col in enumerate(cols):
            cell = ws.cell(r, k + 1)
            cell.fill = st.fill("E7EBF3")
            cell.font = st.font(italic=True, bold=(k == name_col))
            if k == name_col:
                cell.value = f"Benchmark: {self.c.dataset.benchmark_name or 'reference row from input'}"
                cell.alignment = LEFT
            elif col.key in b:
                cell.value = b[col.key]
                cell.number_format = FORMATS.get(col.fmt, "General")
                cell.alignment = CENTER

    def _print_setup(self, ws: Worksheet):
        ws.oddFooter.left.text = DATA_NOTICE
        ws.oddFooter.left.size = 7
        ws.oddFooter.right.text = "&P / &N"
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.sheet_view.zoomScale = 90
        ws.sheet_view.showGridLines = False

    def note(self, ws: Worksheet, row: int, text: str, ncols: int = 12):
        c = ws.cell(row, 1, text)
        c.font = self.st.font(italic=True, color=MUTED, size=9)
        c.alignment = Alignment(horizontal="left", vertical="center")

    # ---- shared column groups --------------------------------------------------------
    def id_cols(self, approved: bool = True) -> list[Col]:
        s = self.c.scores
        cols = [
            Col("Rank", lambda t: s.rank[t], "rank", 6, "rank", band="Fund", band_key="identity"),
            Col("Tier", lambda t: s.tier[t], "text", 7, None, band="Fund", band_key="identity"),
            Col("Ticker", lambda t: t, "text", 8, None, band="Fund", band_key="identity"),
            Col("Fund", lambda t: self.v(t, "name"), "text", 36, None, band="Fund", band_key="identity"),
        ]
        if approved and self.c.approved is not None and self.c.kind == "full":
            cols.append(Col("Approved", lambda t: bool(self.c.approved_flag.get(t, False)), "text", 9, None,
                            band="Fund", band_key="identity"))
        return cols

    def composite_cols(self) -> list[Col]:
        s, dg = self.c.scores, self.c.diag
        n = dg.n_pillars
        return [
            Col("Composite\n(Adjusted)", lambda t: s.adjusted[t], "score", 10, "score", band="Composite", band_key="composite"),
            Col("Raw Score", lambda t: s.raw[t], "score", 8, "score", band="Composite", band_key="composite"),
            Col("Coverage", lambda t: s.coverage[t], "cov", 8, "dir", band="Composite", band_key="composite"),
            Col(f"Breadth\n(of {n})", lambda t: dg.breadth[t], "int", 8, "dir", band="Composite", band_key="composite"),
            Col("Rank Range", lambda t: (f"{int(dg.rank_best[t])}-{int(dg.rank_worst[t])}"
                                         if pd.notna(dg.rank_best[t]) else None), "text", 9, None,
                band="Composite", band_key="composite"),
            Col("Robust\nLeader", lambda t: bool(dg.robust[t]), "text", 8, None, band="Composite", band_key="composite"),
            Col("Weakest Pillar", lambda t: dg.weakest[t] or None, "text", 16, None, band="Composite", band_key="composite"),
            Col("Recency Flag", lambda t: self.recency_flag(t), "text", 13, None, band="Composite", band_key="composite"),
        ]

    def recency_flag(self, t: str):
        sk = self.v(t, "recency_skew")
        thr = float(self.c.prof["recency"]["surge_flag"])
        if sk is None:
            return None
        if sk >= thr:
            return "Recent surge"
        if sk <= -thr:
            return "Recent slump"
        return None

    def pillar_cols(self, band="Pillar Scores (0-100)") -> list[Col]:
        s = self.c.scores
        return [Col(s.pillar_labels[p], lambda t, p=p: s.pillar_scores.at[t, p], "score", 10, "score",
                    band=band, band_key="pillars") for p in s.pillar_weights]

    def scored_metric_cols(self) -> list[Col]:
        cols = []
        for st in self.c.scores.statuses:
            if st.status != "scored":
                continue
            label = self.c.scores.pillar_labels[st.pillar]
            for col in st.columns:
                cols.append(self.metric_col(col, band=label, band_key=st.pillar))
        return cols

    # ---- sheets --------------------------------------------------------------------------
    def build(self) -> Workbook:
        self.summary()
        self.master()
        self.scorecard()
        for L in self.c.lenses:
            self.lens(L)
        self.leaders()
        self.calendar()
        self.portfolio()
        self.medalist()
        self.risk_fit()
        self.correlation()
        self.sensitivity()
        self.recency()
        self.theme_tab()
        if self.c.shortlist:
            self.shortlist()
        if self.c.audit is not None:
            self.audit()
        self.coverage()
        self.original()
        self.metric_guide()
        self.methodology()
        self.quality()
        return self.wb

    def _ws(self, title: str, tab: str | None = None) -> Worksheet:
        title = re.sub(r"[\[\]:*?/\\]", "-", title)[:31]
        ws = self.wb.create_sheet(title)
        if tab:
            ws.sheet_properties.tabColor = tab
        return ws

    # 1. Summary ------------------------------------------------------------------------------
    def summary(self):
        c, s, dg, st = self.c, self.c.scores, self.c.diag, self.st
        ws = self._ws("Summary", NAVY)
        ws.sheet_view.showGridLines = False
        for k, w in enumerate([7, 9, 38, 11, 9] + [11] * 6 + [9, 9, 18], start=1):
            ws.column_dimensions[get_column_letter(k)].width = w
        scope = "Approved funds only" if c.kind == "approved" else "Full universe"
        ws["A1"] = f"FundLens  |  {c.title}"
        ws["A1"].font = st.font(bold=True, color=NAVY, size=18)
        asof = c.asof.isoformat() if c.asof else "n/a"
        src = c.dataset.list_name
        sets_ = sorted({p["kind"] for p in c.dataset.pages}) if len(c.dataset.pages) > 1 else []
        if src or sets_:
            ws["F3"] = ((f"Source list: {src}" if src else "Inputs") +
                        (f"   ·   {len(sets_)} data sets merged: " + ", ".join(sets_) if sets_ else ""))
            ws["F3"].font = st.font(color=MUTED, size=8)
        ws["A2"] = (f"{scope}   ·   Profile: {c.prof['name']}   ·   Returns as of {asof} "
                    f"({c.meta['returns_basis']})   ·   Recency control: {c.prof['recency']['mode']}   ·   Engine v{c.meta['engine_version']}")
        ws["A2"].font = st.font(color=MUTED, size=10)
        ws["A3"] = DATA_NOTICE
        ws["A3"].font = st.font(color=MUTED, size=8, italic=True)

        n_ranked = int(s.rank.notna().sum())
        tiles = [("Funds in file", c.meta["n_input"]), ("Compared", n_ranked),
                 ("Screened out", len(c.excluded)),
                 ("Approved matched", int((c.audit["status"] == "Included").sum()) if c.audit is not None else "—"),
                 ("Tier 1 funds", int((s.tier == "Tier 1").sum())), ("Robust leaders", int(dg.robust.sum()))]
        col = 1
        for lab, val in tiles:
            ws.merge_cells(start_row=4, start_column=col, end_row=4, end_column=col + 1)
            ws.merge_cells(start_row=5, start_column=col, end_row=5, end_column=col + 1)
            a, b = ws.cell(4, col, lab), ws.cell(5, col, val)
            a.font, b.font = st.font(color=MUTED, size=9), st.font(bold=True, color=NAVY, size=18)
            a.alignment = b.alignment = CENTER
            for r in (4, 5):
                for k in (col, col + 1):
                    ws.cell(r, k).fill = st.fill("F3F5F9")
            col += 2
        ws.row_dimensions[5].height = 30

        # Top 10
        r = 7
        ws.cell(r, 1, "Top 10 - composite ranking").font = st.font(bold=True, color=NAVY, size=12)
        top = [t for t in s.order if pd.notna(s.rank[t])][:10]
        cols = [Col("Rank", lambda t: s.rank[t], "rank", 7),
                Col("Ticker", lambda t: t, "text", 9),
                Col("Fund", lambda t: self.v(t, "name"), "text", 38),
                Col("Composite", lambda t: s.adjusted[t], "score", 11, "score"),
                Col("Tier", lambda t: s.tier[t], "text", 9)] + \
               [Col(s.pillar_labels[p], lambda t, p=p: s.pillar_scores.at[t, p], "score", 11, "score")
                for p in s.pillar_weights] + \
               [Col(f"Breadth (of {dg.n_pillars})", lambda t: dg.breadth[t], "int", 9),
                Col("Robust", lambda t: bool(dg.robust[t]), "text", 9),
                Col("Weakest Pillar", lambda t: dg.weakest[t], "text", 18)]
        self._mini_table(ws, r + 1, cols, top)

        # Standouts
        r = r + 3 + len(top)
        ws.cell(r, 1, "Pillar standouts").font = st.font(bold=True, color=NAVY, size=12)
        r += 1
        for k, h in enumerate(["Pillar", "", "Leader", "Score", "Runner-up", "", "Score"], start=1):
            if h:
                cc = ws.cell(r, k, h)
                cc.font, cc.fill, cc.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), CENTER
        for p in s.pillar_weights:
            r += 1
            ser = s.pillar_scores[p].dropna()
            ranked = sorted(ser.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
            ws.cell(r, 1, s.pillar_labels[p]).font = st.font(bold=True)
            if ranked:
                t1, v1 = ranked[0]
                ws.cell(r, 3, f"{t1} - {self.v(t1, 'name')}").font = st.font()
                cc = ws.cell(r, 4, v1); cc.number_format = "0.0"; cc.fill = st.heat(v1); cc.alignment = CENTER
            if len(ranked) > 1:
                t2, v2 = ranked[1]
                ws.cell(r, 5, f"{t2} - {self.v(t2, 'name')}").font = st.font()
                cc = ws.cell(r, 7, v2); cc.number_format = "0.0"; cc.fill = st.heat(v2); cc.alignment = CENTER

        # Robust leaders
        r += 2
        ws.cell(r, 1, "Robust leaders (top quintile under every weighting scheme)").font = st.font(bold=True, color=NAVY, size=12)
        rl = [t for t in s.order if dg.robust.get(t, False)]
        ws.cell(r + 1, 1, ", ".join(rl) if rl else "None - rankings are sensitive to weighting; review the Sensitivity tab.").font = st.font()

        # How to read
        r += 3
        ws.cell(r, 1, "How to read this workbook").font = st.font(bold=True, color=NAVY, size=12)
        lines = [
            "Green = better, red = worse, in the direction that is good for that metric (e.g. lower downside capture, "
            "volatility and cost are green; for this profile beta closest to "
            f"{(c.prof['overrides'].get('beta') or {}).get('target', 1.0)} is green). Grey '—' = no data.",
            "Composite (Adjusted) blends six pillar scores; funds with short histories are pulled toward 50 in proportion "
            "to missing evidence (Coverage). Raw Score shows the unadjusted result.",
            "Breadth counts pillars scoring 60+; Rank Range shows the best/worst rank across alternative pillar weightings.",
            "Lens tabs rank funds within peer groups and colour each group on its own.",
            "Full method, weights and data caveats: Metric Guide, Methodology and Data Quality tabs.",
        ]
        for i, ln in enumerate(lines):
            cc = ws.cell(r + 1 + i, 1, f"•  {ln}")
            cc.font = st.font(size=9)

        # Themes
        r += len(lines) + 2
        th = c.recency.get("themes")
        ws.cell(r, 1, "Themes that may be skewing the latest year").font = st.font(bold=True, color=NAVY, size=12)
        flagged = [] if th is None or th.empty else [x for _, x in th[th["flag"] != ""].head(6).iterrows()]
        if not flagged:
            ws.cell(r + 1, 1, "No theme, tilt or group had an outlier latest year in this comparison.").font = st.font(size=9)
        for i, x in enumerate(flagged):
            ws.cell(r + 1 + i, 1, f"•  {x['flag']}: {x['dimension']} '{x['group']}' ({x['n']} funds) - 1Y {x['excess_1y']:+.1f} pts "
                                  f"vs comparison" + (f", prior 2Y {x['excess_prior']:+.1f} pts" if pd.notna(x['excess_prior']) else "")
                    ).font = st.font(size=9)
        r += max(1, len(flagged)) + 2

        # Caveats
        ws.cell(r, 1, "Key data caveats").font = st.font(bold=True, color=NAVY, size=12)
        warns = [e for e in c.log.sorted() if e.severity in ("error", "warning")][:8]
        if not warns:
            ws.cell(r + 1, 1, "No warnings.").font = st.font(size=9)
        for i, e in enumerate(warns):
            ws.cell(r + 1 + i, 1, f"•  [{e.area}{' ' + e.ticker if e.ticker else ''}] {e.message}").font = st.font(size=9)
        self._print_setup(ws)
        ws.sheet_view.zoomScale = 100

    def _mini_table(self, ws, r, cols, tickers):
        st = self.st
        for k, col in enumerate(cols, start=1):
            cc = ws.cell(r, k, col.header)
            cc.font, cc.fill, cc.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), HDR_ALIGN
        ws.row_dimensions[r].height = 30
        for i, t in enumerate(tickers, start=1):
            for k, col in enumerate(cols, start=1):
                v = _clean(col.get(t))
                if isinstance(v, bool):
                    v = "Yes" if v else ""
                cc = ws.cell(r + i, k, v)
                cc.font = st.font(bold=(k == 2))
                cc.alignment = LEFT if col.width >= 14 else CENTER
                if v is not None and not isinstance(v, str):
                    cc.number_format = FORMATS.get(col.fmt, "General")
                if col.color == "score" and isinstance(v, (int, float)):
                    cc.fill = st.heat(v)

    # 2. Master ranking ---------------------------------------------------------------------------
    def _master_cols(self) -> list[Col]:
        c = self.c
        desc = [Col("Objective", lambda t: self.v(t, "objective"), "text", 16, band="Fund", band_key="identity")]
        if "category" in c.df and c.df["category"].nunique() > 1:
            desc.insert(0, Col("Category", lambda t: (self.v(t, "category") or "").replace("US Fund ", "") or None,
                               "text", 14, band="Fund", band_key="identity"))
        if "sb_group" in c.df:
            desc.append(Col("Strategic Beta Group", lambda t: self.group_label(t, "sb_group"), "text", 16,
                            band="Fund", band_key="identity"))
        if "index_weighting" in c.df:
            desc.append(Col("Index Weighting", lambda t: self.v(t, "index_weighting") or NOT_CLASSIFIED, "text", 16,
                            band="Fund", band_key="identity"))
        scored = self.scored_metric_cols()
        used = {st.metric for st in c.scores.statuses if st.status == "scored"}
        disp = [self.metric_col(m, band="Reference", band_key="display")
                for m in c.prof.get("display_metrics", []) if m in c.df and m not in used]
        return self.id_cols() + desc + self.composite_cols() + self.pillar_cols() + scored + disp

    def master(self):
        ws = self._ws("Master Ranking", "0F4C5C")
        self.table(ws, self._master_cols(), self.c.ranked, top=1, freeze_cols=4, bench=True)

    def shortlist(self):
        ws = self._ws("Shortlist", "C9A227")
        tick = [t for t in self.c.ranked if t in set(self.c.shortlist)]
        self.table(ws, self._master_cols(), tick, top=1, freeze_cols=4)

    # 3. Pillar scorecard ---------------------------------------------------------------------------
    def scorecard(self):
        s = self.c.scores
        ws = self._ws("Pillar Scorecard", "2F7D6B")
        cols = self.id_cols(approved=False) + [
            Col("Composite\n(Adjusted)", lambda t: s.adjusted[t], "score", 10, "score", band="Composite", band_key="composite")]
        for p in s.pillar_weights:
            lab = s.pillar_labels[p]
            cols.append(Col(f"{lab}\nPILLAR", lambda t, p=p: s.pillar_scores.at[t, p], "score", 10, "score", band=lab, band_key=p))
            for stt in s.statuses:
                if stt.pillar == p and stt.status == "scored":
                    m = self.c.d.by_id[stt.metric]
                    cols.append(Col(m.label, lambda t, mid=stt.metric: s.metric_scores.at[t, mid], "score", 9, "score",
                                    band=lab, band_key=p))
        self.table(ws, cols, self.c.ranked, top=1, freeze_cols=4)

    # 4. Lens tabs --------------------------------------------------------------------------------
    LENS_METRICS = ["nav_ret_3y", "nav_ret_5y", "nav_ret_10y", "nav_ret_15y", "sharpe_3y", "sharpe_5y", "sharpe_10y",
                    "down_capture_5y", "down_capture_10y", "std_dev_5y", "std_dev_10y", "worst_year_5y",
                    "top_half_pct", "net_er", "aum"]

    def lens(self, L):
        s = self.c.scores
        ws = self._ws(f"Lens - {L.label}", "2E5E8C")
        tab = L.table
        rows = list(tab.index)
        tick_of = dict(zip(tab.index, tab["ticker"]))
        groups = [tab.at[i, "group"] for i in rows]
        # rows are (group, ticker) pairs, so getters index by row id
        T = lambda i: tick_of[i]
        cols = [
            Col("Group", lambda i: tab.at[i, "group"], "text", 24, None, band=L.label, band_key="group"),
            Col("Group\nSize", lambda i: int(tab.at[i, "group_n"]), "int", 7, None, band=L.label, band_key="group"),
            Col("Within\nRank", lambda i: _clean(tab.at[i, "within_rank"]), "rank", 7, "rank", band=L.label, band_key="group"),
            Col("Overall\nRank", lambda i: s.rank[T(i)], "rank", 7, None, band=L.label, band_key="group"),
            Col("Ticker", lambda i: T(i), "text", 8, None, band=L.label, band_key="group"),
            Col("Fund", lambda i: self.v(T(i), "name"), "text", 34, None, band=L.label, band_key="group"),
            Col("Composite\n(Adjusted)", lambda i: s.adjusted[T(i)], "score", 10, "dir", band="Composite", band_key="composite"),
        ]
        for p in s.pillar_weights:
            cols.append(Col(s.pillar_labels[p], lambda i, p=p: s.pillar_scores.at[T(i), p], "score", 10, "dir",
                            band="Pillar Scores", band_key="pillars"))
        for mc in self.LENS_METRICS:
            if mc in self.c.df:
                base = self.metric_col(mc, band="Key Metrics", band_key="display")
                base.get = (lambda i, g=base.get: g(T(i)))
                cols.append(base)
        self.note(ws, 1, f"{L.label} lens: funds ranked within each group by composite score. Colours are relative "
                         f"WITHIN each group (groups of fewer than 3 funds are coloured against the whole table). "
                         f"A fund can appear in several groups when Morningstar lists several values.")
        self.table(ws, cols, rows, top=2, groups=groups, freeze_cols=6, group_col=5)

    def leaders(self):
        s, dg = self.c.scores, self.c.diag
        ws = self._ws("Group Leaders", "2E5E8C")
        Lt = self.c.leaders
        if Lt.empty:
            ws["A1"] = "No lenses available."
            return
        idx = list(Lt.index)
        T = lambda i: Lt.at[i, "ticker"]
        cols = [
            Col("Lens", lambda i: Lt.at[i, "lens"], "text", 20, band="Group", band_key="group"),
            Col("Group", lambda i: Lt.at[i, "group"], "text", 26, band="Group", band_key="group"),
            Col("Group\nSize", lambda i: int(Lt.at[i, "group_n"]), "int", 7, band="Group", band_key="group"),
            Col("Within\nRank", lambda i: int(Lt.at[i, "within_rank"]), "int", 7, band="Group", band_key="group"),
            Col("Ticker", lambda i: T(i), "text", 8, band="Fund", band_key="identity"),
            Col("Fund", lambda i: self.v(T(i), "name"), "text", 34, band="Fund", band_key="identity"),
            Col("Overall\nRank", lambda i: s.rank[T(i)], "rank", 8, "rank", band="Fund", band_key="identity"),
            Col("Composite", lambda i: s.adjusted[T(i)], "score", 10, "score", band="Fund", band_key="identity"),
            Col("Tier", lambda i: s.tier[T(i)], "text", 8, band="Fund", band_key="identity"),
            Col("Robust", lambda i: bool(dg.robust[T(i)]), "text", 8, band="Fund", band_key="identity"),
            Col("Note", lambda i: "Small group (n<3)" if Lt.at[i, "group_n"] < 3 else None, "text", 18,
                band="Fund", band_key="identity"),
        ]
        groups = [f"{Lt.at[i, 'lens']}|{Lt.at[i, 'group']}" for i in idx]
        self.table(ws, cols, idx, top=1, groups=groups, freeze_cols=6, group_col=3)

    # 5. Calendar years -------------------------------------------------------------------------
    def calendar(self):
        s = self.c.scores
        years = annual_years(self.c.df)[: int(self.c.prof["scoring"]["calendar_years"])]
        if not years:
            return
        ws = self._ws("Calendar Years", "6B5B95")
        cols = self.id_cols(approved=False) + [
            Col("Composite", lambda t: s.adjusted[t], "score", 10, "score", band="Composite", band_key="composite")]
        for y in years:
            cols.append(Col(str(y) + (" (stress)" if y == 2022 else ""), lambda t, y=y: self.v(t, f"nav_annual_{y}"),
                            "pct", 9, "dir", band="Calendar-Year NAV Return (coloured within each year)", band_key="returns",
                            key=f"nav_annual_{y}"))
        for m in ("top_half_pct", "worst_year_5y", "history_years"):
            if m in self.c.df:
                cols.append(self.metric_col(m, band="Consistency", band_key="consistency", width=10))
        self.table(ws, cols, self.c.ranked, top=1, freeze_cols=4, bench=True)

    # 6. Risk & fit -----------------------------------------------------------------------------
    def risk_fit(self):
        c, s = self.c, self.c.scores
        ws = self._ws("Risk & Fit", "4A6274")
        cols = self.id_cols(approved=False)
        for m in ("best_fit_index", "best_fit_beta", "best_fit_r2"):
            if m in c.df:
                cols.append(self.metric_col(m, band="Best Fit", band_key="fit", width=12 if m != "best_fit_index" else 26))
        fams = [("beta", "Beta"), ("r_squared", "R-Squared"), ("corr_est", "Correlation est. (sqrt R2)"),
                ("te_ratio", "Tracking Error"), ("up_capture", "Upside Capture"), ("down_capture", "Downside Capture"),
                ("capture_spread", "Capture Spread (Up - Down)")]
        for fam, lab in fams:
            for h in SCORING_HORIZONS:
                col = f"{fam}_{h}"
                if col in c.df:
                    cc = self.metric_col(col, band=lab, band_key="fit")
                    cc.header = HORIZON_LABEL[h]
                    cols.append(cc)
        if "fit" in s.pillar_scores:
            cols.append(Col(s.pillar_labels["fit"], lambda t: s.pillar_scores.at[t, "fit"], "score", 10, "score",
                            band="Pillar", band_key="pillars"))
        b = c.prof["overrides"].get("beta") or {}
        r2 = c.prof["overrides"].get("r_squared") or {}
        bdesc = (f"beta closest to {b.get('target')} is green" if b.get("direction") == "target"
                 else f"{b.get('direction', 'neutral')} beta is green")
        self.note(ws, 1, f"Profile '{c.prof['name']}': {bdesc}; {r2.get('direction', 'neutral')} R-squared is green. "
                         f"Morningstar computes these against the category's standard index; sqrt(R2) estimates correlation.")
        self.table(ws, cols, self.c.ranked, top=2, freeze_cols=4, bench=True)

    def correlation(self):
        """Indicative correlation of calendar-year returns among the top-ranked funds."""
        c = self.c
        years = annual_years(c.df)[:10]
        top = [t for t in c.ranked if pd.notna(c.scores.rank[t])][:25]
        if len(years) < 5 or len(top) < 3:
            return
        block = c.df.loc[top, [f"nav_annual_{y}" for y in years]].apply(pd.to_numeric, errors="coerce").T
        corr = block.corr(min_periods=5)
        ws = self._ws("Calendar Correlation", "4A6274")
        st = self.st
        direction, _ = effective_direction("corr_est", c.d, c.prof)
        self.note(ws, 1, f"Indicative only: correlation of calendar-year NAV returns ({min(years)}-{max(years)}, "
                         f"min 5 overlapping years) among the top {len(top)} funds. "
                         + ("Lower correlation is green (diversification) for this profile." if direction == "lower"
                            else "Shaded by level (neutral) - within a single category high correlation is expected."))
        ws.cell(2, 1, "Ticker").font = st.font(bold=True, color="FFFFFF", size=9)
        ws.cell(2, 1).fill = st.fill(NAVY)
        ws.column_dimensions["A"].width = 9
        for j, t in enumerate(top, start=2):
            h = ws.cell(2, j, t)
            h.font, h.fill, h.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), CENTER
            ws.column_dimensions[get_column_letter(j)].width = 7
        for i, t in enumerate(top, start=3):
            a = ws.cell(i, 1, t)
            a.font, a.fill = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY)
            for j, u in enumerate(top, start=2):
                v = corr.at[t, u]
                cc = ws.cell(i, j)
                if pd.isna(v):
                    cc.value, cc.fill = "—", st.fill(MISSING_FILL)
                    cc.font, cc.alignment = st.font(color="A0A6B0"), CENTER
                    continue
                cc.value, cc.number_format, cc.alignment, cc.font = float(v), "0.00", CENTER, st.font(size=9)
                pos = (1 - v) / 2 * 100 if direction == "lower" else (v + 1) / 2 * 100
                cc.fill = st.heat(pos, neutral=(direction != "lower"))
        ws.freeze_panes = "B3"
        self._print_setup(ws)

    # 7. Sensitivity --------------------------------------------------------------------------------
    def sensitivity(self):
        s, dg = self.c.scores, self.c.diag
        ws = self._ws("Sensitivity", "0F4C5C")
        cols = self.id_cols(approved=False) + [
            Col("Composite", lambda t: s.adjusted[t], "score", 10, "score", band="Composite", band_key="composite")]
        for name in dg.scheme_ranks.columns:
            cols.append(Col(name, lambda t, n=name: dg.scheme_ranks.at[t, n], "rank", 10, "rank",
                            band="Rank under each pillar weighting", band_key="pillars"))
        cols += [Col("Best", lambda t: dg.rank_best[t], "rank", 7, "rank", band="Stability", band_key="composite"),
                 Col("Worst", lambda t: dg.rank_worst[t], "rank", 7, "rank", band="Stability", band_key="composite"),
                 Col("Range", lambda t: (dg.rank_worst[t] - dg.rank_best[t]) if pd.notna(dg.rank_best[t]) else None,
                     "int", 7, "dir", "lower", band="Stability", band_key="composite"),
                 Col("Robust\nLeader", lambda t: bool(dg.robust[t]), "text", 8, band="Stability", band_key="composite")]
        self.note(ws, 1, "How much each fund's rank depends on the pillar weights. 'x2' doubles one pillar's weight. "
                         "Robust Leader = top quintile under every scheme.")
        self.table(ws, cols, self.c.ranked, top=2, freeze_cols=4)

    # 7b. Recency check ---------------------------------------------------------------------------
    def recency(self):
        c, s = self.c, self.c.scores
        if "recency_skew" not in c.df:
            return
        rc = c.recency
        ws = self._ws("Recency Check", "C9A227")
        off = rc.get("rank_off", s.rank)
        cols = self.id_cols(approved=False) + [
            Col("Composite", lambda t: s.adjusted[t], "score", 10, "score", band="Composite", band_key="composite"),
            Col("Rank with\nrecency OFF", lambda t: off.get(t), "rank", 10, "rank", band="Rank impact", band_key="composite"),
            Col("Rank change\n(+ = up)", lambda t: (off[t] - s.rank[t]) if pd.notna(off.get(t)) and pd.notna(s.rank[t]) else None,
                "int", 10, "dir", "higher", band="Rank impact", band_key="composite"),
            Col("Recency Flag", lambda t: self.recency_flag(t), "text", 13, band="Rank impact", band_key="composite"),
            Col("Hot-Theme\nExposure", lambda t: ", ".join(rc.get("fund_hot", {}).get(t, [])) or None, "text", 18,
                band="Rank impact", band_key="composite"),
            Col("Momentum\nStrategy", lambda t: self.v(t, "momentum_tilt"), "text", 10, band="Rank impact", band_key="composite"),
        ]
        for col in ("nav_ret_ytd", "nav_ret_12m", "nav_ret_3y", "nav_ret_5y", "nav_ret_10y"):
            if col in c.df:
                cols.append(self.metric_col(col, band="Trailing NAV return", band_key="returns"))
        heads = {"3y": "Prior 2Y\n(of 3Y)", "5y": "Prior 4Y\n(of 5Y)", "10y": "Prior 9Y\n(of 10Y)"}
        for h in ("3y", "5y", "10y"):
            col = f"nav_ret_xr_{h}"
            if col in c.df:
                cc = self.metric_col(col, band="Ex-latest-year return (latest 12M removed)", band_key="returns")
                cc.header = heads[h]
                cols.append(cc)
        cols.append(Col("Recency Skew\n(pct pts)", lambda t: self.v(t, "recency_skew"), "score", 11, "dir", "lower",
                        band="Diagnostic", band_key="consistency"))
        mode_txt = {"off": "OFF - standard trailing returns are scored",
                    "balanced": "BALANCED - Returns pillar blends standard and ex-latest-year returns 50/50; a recent-surge penalty is scored",
                    "strict": "STRICT - only ex-latest-year returns are scored; recent-surge penalty weight 2"}[rc.get("mode", "off")]
        th = rc.get("themes")
        hot = [] if th is None or th.empty else list(th.loc[th["flag"] == "Hot year", "group"])
        mom = (f" Hot-year themes: {', '.join(hot[:6])} (see Theme Scan)." if hot else " No hot-year themes detected.")
        self.note(ws, 1, f"Recency control: {mode_txt}. Recency Skew = 1Y peer percentile minus ex-latest-year percentile; "
                         f"+{rc.get('surge_flag', 35):.0f} or more = 'Recent surge' (standing leans on one strong year).{mom}")
        self.table(ws, cols, self.c.ranked, top=2, freeze_cols=4, bench=True)

    # 7c. Theme scan ------------------------------------------------------------------------------
    def theme_tab(self):
        th = self.c.recency.get("themes")
        if th is None or th.empty:
            return
        st, rc = self.st, self.c.prof["recency"]
        ws = self._ws("Theme Scan", "C9A227")
        idx = list(th.index)
        G = lambda col: (lambda i: _clean(th.at[i, col]))
        cols = [Col("Flag", G("flag"), "text", 16, band="Theme", band_key="group"),
                Col("Dimension", G("dimension"), "text", 26, band="Theme", band_key="group"),
                Col("Group / Theme", G("group"), "text", 28, band="Theme", band_key="group"),
                Col("Funds", G("n"), "int", 7, band="Theme", band_key="group"),
                Col("Median 1Y\nReturn", G("median_1y"), "pct", 10, "dir", band="Latest year vs comparison", band_key="returns"),
                Col("1Y Excess\n(pts)", G("excess_1y"), "ratio", 10, "dir", band="Latest year vs comparison", band_key="returns"),
                Col("YTD Excess\n(pts)", G("excess_ytd"), "ratio", 10, "dir", band="Latest year vs comparison", band_key="returns"),
                Col("Prior 2Y Excess\n(pts)", G("excess_prior"), "ratio", 12, "dir", band="Before the latest year", band_key="consistency"),
                Col("Median Recency\nSkew (pts)", G("median_skew"), "score", 12, "dir", "lower", band="Before the latest year", band_key="consistency"),
                Col("Top-quintile\nfunds (recency off)", G("top_quintile_off"), "int", 12, band="Ranking impact", band_key="composite"),
                Col("Top-quintile\nfunds (now)", G("top_quintile_now"), "int", 11, band="Ranking impact", band_key="composite"),
                Col("Avg rank change\n(+ = up)", G("avg_rank_change"), "ratio", 12, "dir", band="Ranking impact", band_key="composite"),
                Col("Funds in group", G("tickers"), "text", 60, band="Members", band_key="identity")]
        self.note(ws, 1, f"Which themes, tilts and groups had an outlier latest year? 'Hot year' = group median 1Y return "
                         f"{rc['theme_gap']:+.0f}+ pts above this comparison AND median recency skew {rc['theme_skew']:+.0f}+ "
                         "(this year is unusual for them). 'Persistent leader' = strong now and in the prior 2 years. "
                         "Diagnostic only - scores change only through the recency control setting.")
        last = self.table(ws, cols, idx, top=2, freeze_cols=3)
        fills = {"Hot year": "F4B183", "Cold year": "9DC3E6", "Persistent leader": "A9D18E"}
        for r in range(4, last + 1):
            f = fills.get(ws.cell(r, 1).value)
            if f:
                ws.cell(r, 1).fill = st.fill(f)
                ws.cell(r, 1).font = st.font(bold=True)

    # 8. Approval audit -------------------------------------------------------------------------
    def audit(self):
        a = self.c.audit
        ws = self._ws("Approval Audit", "3A7D44")
        st = self.st
        idx = list(a.index)
        cols = [Col("Ticker", lambda i: a.at[i, "ticker"], "text", 9), Col("Fund", lambda i: a.at[i, "name"], "text", 38),
                Col("Status", lambda i: a.at[i, "status"], "text", 24), Col("Reason", lambda i: a.at[i, "reason"], "text", 48),
                Col("Approved Category", lambda i: a.at[i, "approved_category"], "text", 18),
                Col("Solicitation", lambda i: a.at[i, "solicitation"], "text", 11),
                Col("Complexity", lambda i: a.at[i, "complexity"], "text", 10),
                Col("Active", lambda i: a.at[i, "active"], "text", 8)]
        last = self.table(ws, cols, idx, top=1, band_row=False, freeze_cols=2)
        colors = {"Included": "C6EFCE", "Approved - screened out": "FFEB9C", "Approved - not solicitable": "FFEB9C",
                  "Approved - not in data": "FFC7CE"}
        for r in range(2, last + 1):
            f = colors.get(ws.cell(r, 3).value)
            if f:
                ws.cell(r, 3).fill = st.fill(f)

    # 5b. Portfolio -------------------------------------------------------------------------------
    SECTOR_COLS = ["sec_cyclical", "sec_sensitive", "sec_defensive", "sec_basic_materials", "sec_consumer_cyclical",
                   "sec_financial", "sec_real_estate", "sec_communication", "sec_energy", "sec_industrials",
                   "sec_technology", "sec_consumer_defensive", "sec_healthcare", "sec_utilities"]
    CHAR_COLS = ["equity_style_box", "style_consistency", "avg_mkt_cap", "pe", "pb", "pc", "ps", "roe", "roa",
                 "net_margin", "debt_cap", "eps_growth_3y", "med_top10", "med_n_hold", "turnover"]
    BOX_COLS = ["sbox_lv", "sbox_lb", "sbox_lg", "sbox_mv", "sbox_mb", "sbox_mg", "sbox_sv", "sbox_sb", "sbox_sg"]

    def portfolio(self):
        c = self.c
        sec = [x for x in self.SECTOR_COLS if x in c.df and c.df[x].notna().any()]
        chars = [x for x in self.CHAR_COLS if x in c.df and c.df[x].notna().any()]
        box = [x for x in self.BOX_COLS if x in c.df and c.df[x].notna().any()]
        if not (sec or chars or box):
            return
        ws = self._ws("Portfolio", "4A6274")
        cols = self.id_cols(approved=False)
        if "sector_tilt" in c.df:
            cols.append(Col("Sector Overweights\n(5+ pts vs bench)", lambda t: self.v(t, "sector_tilt"), "text", 26,
                            band="Tilts", band_key="group"))
        if "sector_under" in c.df:
            cols.append(Col("Sector Underweights\n(5+ pts vs bench)", lambda t: self.v(t, "sector_under"), "text", 26,
                            band="Tilts", band_key="group"))
        for x in sec:
            cols.append(self.metric_col(x, band="Sector weights (% of equity)", band_key="fit"))
        act = [x for x in sec if f"{x}_active" in c.df]
        for x in act:
            cc = Col(c.d.by_id[x].label.replace(" %", ""), lambda t, k=f"{x}_active": self.v(t, k), "ratio", 9, "dir",
                     "none", band="Active weight vs benchmark (pts)", band_key="returns")
            cols.append(cc)
        for x in chars:
            cols.append(self.metric_col(x, band="Equity portfolio statistics", band_key="consistency",
                                        width=10 if x not in ("equity_style_box", "style_consistency") else 14))
        for x in box:
            cols.append(self.metric_col(x, band="Style box exposure (%)", band_key="cost", width=8))
        self.note(ws, 1, "Portfolio characteristics are shaded by level (neutral blue: not good or bad in themselves). "
                         "Active weight = fund sector weight minus the benchmark row in the input. "
                         "Use with the Theme Scan to see which tilts drove the latest year.")
        self.table(ws, cols, c.ranked, top=2, freeze_cols=4, bench=True)

    # 5c. Medalist inputs ---------------------------------------------------------------------------
    MEDALIST = [("People pillar", ["med_mgr_exp_rk", "med_mgr_ir_rk_12m", "med_mgr_ir_rk_3y", "med_mgr_ir_rk_5y",
                                   "med_brand_ir_rk_3y", "med_brand_ir_rk_5y", "med_brand_retention_rk",
                                   "med_brand_tenure_rk", "med_mgr_ownership"]),
                ("Process pillar", ["med_proc_ir_rk_3y", "med_proc_ir_rk_5y", "med_success_rk_3y", "med_success_rk_5y",
                                    "med_success_rk_10y", "med_style_drift_rk_3y", "med_ap_ir_rk_10y", "med_pas_te_rk_3y"]),
                ("Parent pillar", ["med_fee_rank", "med_parent_success_3y", "med_parent_success_5y",
                                   "med_parent_success_10y", "med_brand_retention", "med_obsolete"]),
                ("Price", ["med_fee", "med_fee_rk"])]

    def medalist(self):
        c, s = self.c, self.c.scores
        groups = [(g, [x for x in cols_ if x in c.df and c.df[x].notna().any()]) for g, cols_ in self.MEDALIST]
        groups = [(g, x) for g, x in groups if x]
        if not groups:
            return
        ws = self._ws("Medalist Inputs", "6B5B95")
        cols = self.id_cols(approved=False)
        if "stewardship" in s.pillar_scores:
            cols.append(Col(s.pillar_labels["stewardship"], lambda t: s.pillar_scores.at[t, "stewardship"], "score", 10,
                            "score", band="FundLens pillar", band_key="pillars"))
        for g, xs in groups:
            for x in xs:
                cc = self.metric_col(x, band=g, band_key={"People pillar": "returns", "Process pillar": "risk_adjusted",
                                                          "Parent pillar": "consistency"}.get(g, "cost"), width=10)
                cols.append(cc)
        self.note(ws, 1, "Morningstar Medalist Rating inputs. '(rank)' columns are Morningstar category percentiles where "
                         "0 = best, so lower is green. Index funds have no active-manager data; FundLens does not penalise them for it.")
        self.table(ws, cols, c.ranked, top=2, freeze_cols=4)

    # 8b. Data coverage ------------------------------------------------------------------------------
    def coverage(self):
        """Data Sources & Coverage: every input file / pasted page, what it held, and any fund it lacked."""
        ds, st = self.c.dataset, self.st
        pages = ds.pages
        if len(pages) < 2 and not any(p.get("listed") for p in pages):
            return
        ws = self._ws("Data Sources", "8C4A3A")
        sets = [set(p["tickers"]) for p in pages]
        allt = sorted(set().union(*sets)) if sets else []
        listed = max([p.get("listed") or 0 for p in pages] + [len(allt)])
        pasted = any(p.get("listed") for p in pages)
        self.note(ws, 1, f"Inputs merged by ticker ({len(allt)} funds). Files are read in name order, so the same files always "
                         "give the same workbook. "
                         + ("Pasted grids only copy rows rendered on screen - zoom the browser out before copying. " if pasted else ""))
        r = 3
        hdr = ["Data set", "File", "Print date", "Funds", "Of total", "Share", "Benchmark values"]
        for k, h in enumerate(hdr, start=1):
            cc = ws.cell(r, k, h)
            cc.font, cc.fill, cc.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), CENTER
        for p in pages:
            r += 1
            of = p.get("listed") or len(allt)
            share = len(p["tickers"]) / of if of else None
            vals = [p["kind"], p["file"], p.get("print_date"), len(p["tickers"]), of, share, p.get("bench", 0)]
            for k, v in enumerate(vals, start=1):
                cc = ws.cell(r, k, v)
                cc.font, cc.alignment = st.font(size=9), LEFT if k <= 2 else CENTER
            ws.cell(r, 6).number_format = "0%"
            if share is not None:
                ws.cell(r, 6).fill = st.heat(share * 100)
        r += 2
        bname = ds.benchmark_name or "none found"
        info = [f"Benchmark: {bname} ({len(ds.benchmark)} values used for reference rows and active weights)."]
        ni = ds.name_info or {}
        if ni:
            info.append(f"Names matched to tickers with certainty: {ni.get('resolved', 0)} of {ni.get('n_tickers', 0)} "
                        f"(the rest show {NAME_MISSING}; add a file with Name + Ticker to fill them).")
        complete = all(len(x) == len(allt) for x in sets) and listed == len(allt)
        if complete:
            info.append(f"All {len(allt)} funds are present in every input.")
        for line in info:
            ws.cell(r, 1, line).font = st.font(size=9, italic=True)
            r += 1
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 46
        for k in range(3, 8):
            ws.column_dimensions[get_column_letter(k)].width = 13
        self._print_setup(ws)
        if complete:
            return
        r += 1
        hdr = ["Ticker", "Fund"] + [p["kind"] for p in pages] + ["Inputs", "Ranked"]
        for k, h in enumerate(hdr, start=1):
            cc = ws.cell(r, k, h)
            cc.font, cc.fill, cc.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), HDR_ALIGN
        ws.row_dimensions[r].height = 30
        hdr_row = r
        for t in allt:
            r += 1
            n = sum(t in x for x in sets)
            nm = (self.v(t, "name") if t in self.c.df.index else
                  (ds.df.at[t, "name"] if t in ds.df.index and pd.notna(ds.df.at[t, "name"]) else NAME_MISSING))
            vals = [t, nm] + ["Yes" if t in x else "-" for x in sets] + [
                f"{n}/{len(pages)}", "Yes" if t in self.c.df.index and pd.notna(self.c.scores.rank.get(t)) else "No"]
            for k, v in enumerate(vals, start=1):
                cc = ws.cell(r, k, v)
                cc.font, cc.alignment = st.font(size=9), LEFT if k == 2 else CENTER
                if 3 <= k < 3 + len(pages):
                    cc.fill = st.fill("C6EFCE" if v == "Yes" else "FFC7CE")
            ws.cell(r, len(vals) - 1).fill = st.heat(n / len(pages) * 100)
        ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(len(hdr))}{r}"
        ws.freeze_panes = ws.cell(hdr_row + 1, 3)

    # 9. Original data --------------------------------------------------------------------------
    def original(self):
        ds, st = self.c.dataset, self.st
        for gi, g in enumerate(ds.grids):
            kind = next((p["kind"] for p in ds.pages if p["file"] == g.file), None)
            title = (f"Src - {kind}" if kind and len(ds.grids) > 1 else
                     "Original Data" if gi == 0 else f"Original Data {gi + 1}")[:31]
            if title in self.wb.sheetnames:
                title = f"{title[:27]} {gi + 1}"
            ws = self._ws(title, "595959")
            note = ("Original Morningstar export (values unchanged). Rows are coloured left-to-right where a direction "
                    "applies (green = better); neutral blue = characteristic; dates converted.")
            if self.c.kind == "approved":
                note += " Full original universe retained here; ranking tabs are approved-only."
            self.note(ws, 1, note)
            maps = {rm.row: rm for rm in ds.rowmaps if rm.grid_index == gi}
            label_col = None
            for r, row in enumerate(g.rows):
                for k, v in enumerate(row):
                    if v is None:
                        continue
                    cell = ws.cell(r + 2, k + 1)
                    rm = maps.get(r)
                    if (rm is not None and rm.match is not None and rm.match.metric.kind == "date"
                            and not (isinstance(v, str) and norm_label(v) == norm_label(rm.label))):
                        d = to_date(v)
                        if d is not None:
                            cell.value, cell.number_format = d, FORMATS["date"]
                            cell.font = st.font(size=9)
                            continue
                    cell.value = v
                    cell.font = st.font(size=9)
                    if isinstance(v, (dt.date, dt.datetime)):
                        cell.number_format = FORMATS["date"]
            # column-wise colouring for funds-in-rows tables (pasted grids)
            for cm in (x for x in ds.colmaps if x["grid"] == gi):
                for k, m in cm["cols"].items():
                    if m.metric.kind not in ("directional", "characteristic"):
                        continue
                    direction, target = effective_direction(m.metric.id, self.c.d, self.c.prof)
                    vals = {r: g.rows[r][k] for r in cm["rows"] if k < len(g.rows[r])
                            and isinstance(g.rows[r][k], (int, float)) and not isinstance(g.rows[r][k], bool)}
                    if len(vals) < 3:
                        continue
                    pr = pct_rank(pd.Series(vals, dtype=float), direction if direction != "none" else "higher", target)
                    for r, p in pr.items():
                        if pd.notna(p):
                            ws.cell(r + 2, k + 1).fill = st.heat(p, neutral=(direction == "none"))
            if ds.colmaps and not maps:
                for k in range(1, max((len(x) for x in g.rows), default=0) + 1):
                    ws.column_dimensions[get_column_letter(k)].width = 12
            # row-wise colouring
            for r, rm in maps.items():
                label_col = label_col if label_col is not None else next(
                    (k for k, v in enumerate(g.rows[r]) if v is not None), 0)
                ws.cell(r + 2, label_col + 1).font = st.font(bold=True, size=9)
                if rm.match is None:
                    continue
                m = rm.match.metric
                if m.kind not in ("directional", "characteristic"):
                    continue
                direction, target = effective_direction(m.id, self.c.d, self.c.prof)
                row = g.rows[r]
                vals = {k: row[k] for k in range(label_col + 1, len(row))
                        if isinstance(row[k], (int, float)) and not isinstance(row[k], bool)}
                if len(vals) < 3:
                    continue
                ser = pd.Series(vals, dtype=float)
                pr = pct_rank(ser, direction if direction != "none" else "higher", target)
                for k, p in pr.items():
                    if pd.notna(p):
                        ws.cell(r + 2, k + 1).fill = st.heat(p, neutral=(direction == "none"))
            if label_col is not None:
                ws.column_dimensions[get_column_letter(label_col + 1)].width = 36
                for k in range(label_col + 2, (max((len(x) for x in g.rows), default=0)) + 2):
                    ws.column_dimensions[get_column_letter(k)].width = 13
                tr = next((rm.row for rm in ds.rowmaps if rm.grid_index == gi and rm.match is not None
                           and rm.match.metric.id == "ticker"), 1)
                ws.freeze_panes = ws.cell(tr + 3, label_col + 2)
            ws.sheet_view.zoomScale = 85

    # 10. Metric guide ---------------------------------------------------------------------------
    def metric_guide(self):
        c, s, st = self.c, self.c.scores, self.st
        ws = self._ws("Metric Guide", "595959")
        hdr = ["Pillar", "Metric", "Columns Used", "Direction (this profile)", "Pillar Weight", "Weight in Pillar",
               "Effective Weight", "Status", "Reason / Note", "What it means"]
        widths = [20, 26, 34, 22, 10, 10, 11, 10, 30, 70]
        for k, (h, w) in enumerate(zip(hdr, widths), start=1):
            cc = ws.cell(1, k, h)
            cc.font, cc.fill, cc.alignment = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY), HDR_ALIGN
            ws.column_dimensions[get_column_letter(k)].width = w
        r = 1

        def dir_text(mid):
            d_, tgt = effective_direction(mid, c.d, c.prof)
            return {"higher": "Higher is better", "lower": "Lower is better", "none": "Neutral (not good/bad)",
                    "target": f"Closest to {tgt} is better"}.get(d_, d_)

        for stt in s.statuses:
            r += 1
            m = c.d.by_id.get(stt.metric)
            pw = s.pillar_weights.get(stt.pillar, 0)
            psum = sum(x.weight for x in s.statuses if x.pillar == stt.pillar and x.status == "scored") or 1
            vals = [s.pillar_labels[stt.pillar], m.label if m else stt.metric,
                    ", ".join(HORIZON_LABEL.get(x.rsplit("_", 1)[-1], x) if m and m.horizons else self.label(x)
                              for x in stt.columns) or "—",
                    dir_text(stt.metric), pw if stt.status == "scored" else None,
                    stt.weight / psum if stt.status == "scored" else None,
                    stt.effective_weight if stt.status == "scored" else None,
                    "Scored" if stt.status == "scored" else "Skipped", stt.reason or None,
                    m.description if m else ""]
            for k, v in enumerate(vals, start=1):
                cc = ws.cell(r, k, v)
                cc.font = st.font(size=9, bold=(k == 2))
                cc.alignment = WRAP
                if k in (5, 6, 7) and v is not None:
                    cc.number_format = "0.0%"
                    cc.alignment = CENTER
            ws.cell(r, 8).fill = st.fill("C6EFCE" if stt.status == "scored" else "FFEB9C")
        r += 2
        ws.cell(r, 1, "Displayed (not scored) metrics in this data set").font = st.font(bold=True, color=NAVY, size=11)
        scored = {x.metric for x in s.statuses}
        seen = set()
        for col in c.df.columns:
            m, h, y = c.d.column_meta(col)
            if m is None or m.id in scored or m.id in seen or m.kind in ("id",):
                continue
            seen.add(m.id)
            r += 1
            vals = ["Display", m.label, "", dir_text(m.id) if m.kind not in ("text", "date") else m.kind.title(),
                    None, None, None, "Display", None, m.description]
            for k, v in enumerate(vals, start=1):
                cc = ws.cell(r, k, v)
                cc.font, cc.alignment = st.font(size=9, bold=(k == 2)), WRAP
        ws.freeze_panes = "C2"
        self._print_setup(ws)

    # 11. Methodology --------------------------------------------------------------------------
    def methodology(self):
        c, s, dg, st = self.c, self.c.scores, self.c.diag, self.st
        ws = self._ws("Methodology", "595959")
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 130
        p = c.prof
        sc = p["scoring"]
        hw = ", ".join(f"{HORIZON_LABEL[h]} {w:.0%}" for h, w in p["horizon_weights"].items() if w > 0)
        pillars = "; ".join(f"{s.pillar_labels[k]} {w:.0%}" for k, w in s.pillar_weights.items())
        scr = {k: v for k, v in (p.get("screens") or {}).items() if v not in (None, False, [], "")}
        b = p["overrides"].get("beta") or {}
        rows = [
            ("Purpose", f"Systematic, reproducible comparison of {c.category} {c.vehicle.lower()} for shortlisting. "
                        "A starting point for due diligence, not a recommendation."),
            ("Universe", (f"{c.meta['n_input']} funds in the input; {int(s.rank.notna().sum())} ranked in this workbook"
                          + (" (approved list only, re-ranked among themselves)." if c.kind == "approved" else ".")
                          + f" Screens: {scr if scr else 'none active'}; {len(c.excluded)} fund(s) excluded (see Data Quality).")),
            ("Data", f"Files: {'; '.join(f'{n} [sha {h}]' for n, h in c.meta['files'].items())}. "
                     f"Returns {c.meta['returns_basis']} as of {c.asof}. Portfolio data as of {c.meta.get('portfolio_date')}. "
                     "Trailing returns are NAV total returns (consistent for ETFs and mutual funds); market-price returns are display only."),
            ("Step 1 - Metric scores", f"Each metric column is converted to a 0-100 percentile among the funds being compared "
                                       f"(100 = best; ties share the average rank). Lower-is-better metrics are inverted; target metrics "
                                       f"are scored by distance from the target. A column is skipped when fewer than "
                                       f"{sc['min_funds_per_metric']} funds have data."),
            ("Step 2 - Horizon blend", f"The 3/5/10/15-year versions of one metric are blended into ONE score ({hw}), with weights "
                                       "renormalised over the horizons a fund has. This avoids counting the same evidence many times."),
            ("Step 3 - Pillars", f"Metric scores are averaged (metric weights in the Metric Guide) into pillar scores. Pillar weights: {pillars}."),
            ("Step 4 - Composite", f"Raw Score = pillar-weighted mean over pillars with data. Coverage = share of the total scoring "
                                   f"weight for which the fund has data. Composite (Adjusted) = 50 + (Raw - 50) x Coverage^{sc['shrinkage_k']}: "
                                   "funds with thin evidence (e.g. no 10/15-year record) are pulled toward the median in proportion "
                                   "to what is missing, so a short hot streak cannot top the list on its own."),
            ("Step 5 - Rank & tiers", "Rank by Composite (Adjusted); ties broken by Raw Score, then lower net expense ratio, then ticker. "
                                      "Tiers are quintiles of rank (Tier 1 = top 20%). Funds with coverage below "
                                      f"{sc['low_evidence_coverage']:.0%} are labelled 'Low evidence' instead of a tier."),
            ("Well-rounded diagnostics", f"Breadth = number of pillars scoring {sc['breadth_threshold']}+. Weakest Pillar = lowest pillar score. "
                                         f"Sensitivity re-ranks every fund under {len(dg.schemes)} pillar weightings (profile, equal, and each pillar doubled); "
                                         "Robust Leader = top quintile under every scheme."),
            ("Benchmark-relative metrics", (
                "Alpha, beta, R-squared, capture ratios, information ratio, tracking error and batting average are computed by "
                "Morningstar against the category's standard index, so they are comparable across these funds and are scored. "
                "(Prospectus benchmarks differ by fund but are not the basis of these statistics.)"
                if p.get("relative_metrics_basis") == "standard_index" else
                "Profile set to prospectus-benchmark basis: benchmark-relative metrics are scored only when 80%+ of funds share one benchmark.")),
            ("Portfolio fit", (f"Beta: closest to {b.get('target')} is best." if b.get("direction") == "target" else
                               f"Beta: {b.get('direction', 'neutral')} is better.")
                              + f" R-squared: {(p['overrides'].get('r_squared') or {}).get('direction', 'neutral')} is better. "
                              "sqrt(R-squared) is shown as a correlation estimate with the index."),
            ("Crisis-tested drawdown", f"Worst 3-month return since inception is scored only for funds with a full {sc['stress_year']} "
                                       "calendar-year return; younger funds have not been through a comparable sell-off and would look artificially mild."),
            ("Consistency", f"% Years Top Half = share of the last {sc['calendar_years']} calendar years (min 3) in which the fund "
                            "beat the median fund in this comparison. Worst Year (5Y) = worst of the last five calendar years (all five required)."),
            ("Recency bias", {
                "off": "Recency control OFF: standard trailing returns scored. Recency Skew is still reported on the Recency Check tab.",
                "balanced": "Recency control BALANCED: the Returns pillar blends standard trailing returns with ex-latest-year returns "
                            "(trailing return with the latest 12 months removed; exact, same end date) 50/50, and a Recent Surge penalty "
                            "(positive part of 1Y percentile minus ex-latest-year percentile) is scored, so one hot year - e.g. a momentum "
                            "run - cannot carry a fund's rank. Recency Check shows each fund's rank with the control off.",
                "strict": "Recency control STRICT: only ex-latest-year returns are scored in Returns and the Recent Surge penalty has double "
                          "weight. Recency Check shows each fund's rank with the control off.",
            }[p["recency"]["mode"]]),
            ("Yield", "Displayed but not scored (avoids rewarding value traps). Can be enabled in the profile."),
            ("Lenses", "Lens tabs rank funds within groups (objective, strategic-beta group, index selection/weighting). Multi-value "
                       "Morningstar entries are split so a fund can appear in several groups. Morningstar numeric codes are shown "
                       "only with a verified label (codes.yaml); otherwise 'Code N (unmapped)'. Nothing is inferred."),
            ("Colours", "Static fills computed by FundLens: green = better in the metric's good direction, red = worse; positioned by "
                        "percentile so one outlier does not wash out a column. Neutral blue = characteristic with no good/bad. "
                        "Grey '—' = no data. Lens tabs colour within each group. Editing a value does not recolour - re-run FundLens."),
            ("Limitations", "No monthly return series, so true correlations and max drawdown are not computed; the calendar-year "
                            "correlation tab is indicative only. Scores are relative to the funds supplied. Morningstar data accuracy applies."),
            ("Data rights", "Underlying data (c) Morningstar, Inc., supplied under the user's licence. FundLens reformats and scores "
                            "it for internal research; Morningstar's terms on copying and redistribution continue to apply to "
                            "this workbook and anything derived from it."),
            ("Reproducibility", f"FundLens engine v{c.meta['engine_version']}; profile '{c.meta['profile_id']}' [hash {c.meta['profile_hash']}]. "
                                "The same inputs and profile always produce the same rankings and colours."),
        ]
        if c.approved is not None:
            rows.insert(2, ("Approved list", f"{c.approved.file} [sha {c.approved.hash}]: {len(c.approved.tickers)} tickers; "
                                              "matched on ticker. See Approval Audit."))
        for k, h in enumerate(["Topic", "Explanation"], start=1):
            cc = ws.cell(1, k, h)
            cc.font, cc.fill = st.font(bold=True, color="FFFFFF"), st.fill(NAVY)
        for i, (a, b_) in enumerate(rows, start=2):
            ws.cell(i, 1, a).font = st.font(bold=True, size=9)
            ws.cell(i, 1).alignment = WRAP
            cc = ws.cell(i, 2, b_)
            cc.font, cc.alignment = st.font(size=9), WRAP
            ws.row_dimensions[i].height = max(15, 13 * math.ceil(len(b_) / 150))
        self._print_setup(ws)

    # 12. Data quality --------------------------------------------------------------------------
    def quality(self):
        c, st = self.c, self.st
        ws = self._ws("Data Quality", "8C4A3A")
        for k, (h, w) in enumerate(zip(["Severity", "Area", "Ticker", "Message"], [10, 16, 9, 140]), start=1):
            cc = ws.cell(1, k, h)
            cc.font, cc.fill = st.font(bold=True, color="FFFFFF", size=9), st.fill(NAVY)
            ws.column_dimensions[get_column_letter(k)].width = w
        sev_fill = {"error": "FFC7CE", "warning": "FFEB9C", "info": "EEF2F8"}
        r = 1
        for e in c.log.sorted():
            r += 1
            for k, v in enumerate([e.severity.upper(), e.area, e.ticker or None, e.message], start=1):
                cc = ws.cell(r, k, v)
                cc.font, cc.alignment = st.font(size=9), WRAP
            ws.cell(r, 1).fill = st.fill(sev_fill.get(e.severity, "FFFFFF"))
        r += 2
        ws.cell(r, 1, "Screened-out funds").font = st.font(bold=True, color=NAVY, size=11)
        if not c.excluded:
            ws.cell(r + 1, 1, "None (no screens active or no fund failed a screen).").font = st.font(size=9)
        for i, e in enumerate(c.excluded, start=1):
            ws.cell(r + i, 3, e["ticker"]).font = st.font(size=9, bold=True)
            ws.cell(r + i, 4, f"{e['name']}: {e['reason']}").font = st.font(size=9)
        ws.freeze_panes = "A2"
        self._print_setup(ws)


def output_name(c: Comparison) -> str:
    asof = c.asof.isoformat() if c.asof else "undated"
    base = f"FundLens - {c.category} - {c.vehicle} - {asof}"
    if c.kind == "approved":
        base += " - Approved"
    return re.sub(r'[<>:"/\\|?*]', "-", base) + ".xlsx"


def write_workbook(c: Comparison, out_dir: str | Path, palette: str = "standard") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = Writer(c, palette)
    wb = w.build()
    stamp = dt.datetime.combine(c.asof or dt.date(2000, 1, 1), dt.time())
    wb.properties.creator = "FundLens"
    wb.properties.title = f"FundLens - {c.title}"
    wb.properties.created = stamp
    wb.properties.modified = stamp
    path = out_dir / output_name(c)
    wb.save(path)
    return path

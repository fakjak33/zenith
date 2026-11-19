"""Funds-in-rows tables, including grids copy-pasted from Morningstar Advisor Workstation.

A pasted sheet is a stack of "pages" (one per Morningstar data set view), each:

    <benchmark title>                     e.g. 'S&P 500 (TR) (1970)'   (optional)
    <#>  <fund name>                      numbered name list (the grid's frozen left pane)
    Ticker | Portfolio | Tot Ret | ...    header, wrapped over 1-6 rows
           | Risk Score| 3 Yr Annlzd
           |           | (mo-end)
    -      | -         | 22.89            benchmark row (ticker '-')
    ABLYX  | 66        | 16.18            data rows

Two quirks of copy-paste are handled explicitly and reported, never guessed around:
  * the web grid only renders the rows on screen, so each page may hold a different subset of funds
    (coverage is reported per page and per fund);
  * names live in a separate pane, so a name is attached to a ticker only when the alignment is
    provably unique (see `resolve_names`).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .dictionary import Dictionary, Match, norm_label

TICKER_RX = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
TICKER_LABELS = {"ticker", "symbol", "ticker symbol"}
JUNK = re.compile(r"[ⓘℹ®™]")          # info icons, (R), (TM) pasted from the web UI


@dataclass
class Page:
    header_row: int
    labels: list[str]                       # composed header label per column
    matches: list[Match | None]
    data_rows: list[int]
    tickers: list[str]
    names: dict[int, str]                   # numbered name list captured above this page
    title: str | None                       # benchmark title line, when present
    benchmark: dict[str, object] = field(default_factory=dict)
    kind: str = "Table"
    list_name: str | None = None            # export title row: e.g. 'LARGE VALUE BLEND LIST'
    print_date: str | None = None

    @property
    def n_listed(self) -> int:
        return max(self.names) if self.names else 0


def _is_ticker(v) -> bool:
    return isinstance(v, str) and bool(TICKER_RX.match(v.strip()))


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _txt(v) -> str:
    if _blank(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return JUNK.sub("", str(v)).strip()


def page_kind(labels: list[str]) -> str:
    j = " | ".join(labels).lower()
    for key, name in (("pillar", "Medalist inputs"), ("sharpe", "Risk analysis"), ("annual return", "Annual returns"),
                      ("cyclical", "Sector exposure"), ("p/e", "Equity portfolio"), ("since incep", "Trailing returns"),
                      ("risk score", "Basics"), ("expense", "Basics")):
        if key in j:
            return name
    return "Table"


PRINT_DATE_RX = re.compile(r"print date\s*:?\s*(.+)", re.I)


def _title_block(rows: list[list], start: int, end: int) -> dict:
    """Advisor Workstation export title row: list name | universe | data set | 'For advisor use only' | Print Date."""
    for i in range(start, end):
        cells = [_txt(v) for v in rows[i] if not _blank(v)]
        pd_cell = next((c for c in cells if PRINT_DATE_RX.match(c)), None)
        if pd_cell and len(cells) >= 3:
            rest = [c for c in cells if c is not pd_cell and "advisor use" not in c.lower()]
            return {"list_name": rest[0] if rest else None,
                    "data_set": rest[2] if len(rest) > 2 else (rest[-1] if len(rest) > 1 else None),
                    "print_date": PRINT_DATE_RX.match(pd_cell).group(1).strip()}
    return {}


def _name_column(rows: list[list], data_rows: list[int], labels: list[str], tcol: int) -> int | None:
    """An unlabelled column holding fund names (e.g. a paste with '#' and name next to Ticker)."""
    best = None
    for c, lab in enumerate(labels + [""] * 3):
        if lab or c == tcol:
            continue
        vals = [rows[r][c] for r in data_rows if c < len(rows[r])]
        texty = [v for v in vals if isinstance(v, str) and v.strip() and not _is_ticker(v) and v.strip() not in ("-", "--")]
        if vals and len(texty) >= 0.8 * len(data_rows) and len(texty) >= 2:
            best = c
            break
    return best


def find_pages(rows: list[list], tcol: int, d: Dictionary) -> list[Page]:
    hdrs = [r for r, row in enumerate(rows) if len(row) > tcol and norm_label(row[tcol]) in TICKER_LABELS]
    pages: list[Page] = []
    prev_end = 0
    title = None
    name_match = d.match("name")
    for k, h in enumerate(hdrs):
        nxt = hdrs[k + 1] if k + 1 < len(hdrs) else len(rows)
        # header continuation rows: until the first row with something in the ticker column
        r = h + 1
        while r < nxt and (len(rows[r]) <= tcol or _blank(rows[r][tcol])) and any(not _blank(x) for x in rows[r]):
            r += 1
        width = max(len(rows[i]) for i in range(h, r)) if r > h else len(rows[h])
        labels = []
        for c in range(width):
            parts = [_txt(rows[i][c]) for i in range(h, r) if c < len(rows[i])]
            labels.append(re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip())
        matches = [d.match(lab) if lab else None for lab in labels]
        meta = _title_block(rows, prev_end, h)
        # name list / title above this header (between previous page's data and this header)
        names, ttl = {}, None
        for i in range(prev_end, h):
            row = rows[i]
            nz = [(c, v) for c, v in enumerate(row) if not _blank(v)]
            if len(nz) == 2 and isinstance(nz[0][1], (int, float)) and isinstance(nz[1][1], str):
                names[int(nz[0][1])] = _txt(nz[1][1])
            elif len(nz) == 1 and isinstance(nz[0][1], str) and not names:
                ttl = _txt(nz[0][1])
        title = ttl or title
        data_rows, tickers, bench, bench_row = [], [], {}, None
        for i in range(r, nxt):
            row = rows[i]
            first = next((_txt(v) for v in row if not _blank(v)), "")
            if first.startswith("©") or first.lower().startswith("(c)"):
                break                                            # '(c) Morningstar' footer ends the table
            v = row[tcol] if tcol < len(row) else None
            if _is_ticker(v):
                data_rows.append(i)
                tickers.append(v.strip().upper())
            elif isinstance(v, str) and v.strip() in ("-", "--"):
                # benchmark / index row: above the data (pasted grids) or below it (exports, after an
                # all-dash 'Average' row) - keep the dash row carrying the most numbers
                n_num = sum(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row)
                if n_num and n_num > sum(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bench.values()):
                    bench = {c: row[c] for c in range(len(row)) if c != tcol}
                    bench_row = i
        # unlabelled fund-name column (pasted grids with '#' + name next to Ticker)
        if name_match is not None and not any(m is not None and m.metric.id == "name" for m in matches):
            nc = _name_column(rows, data_rows, labels, tcol)
            if nc is not None:
                while len(matches) <= nc:
                    matches.append(None)
                    labels.append("")
                matches[nc] = name_match
        # benchmark label: the name-column text of the benchmark row
        if bench_row is not None:
            nc = next((c for c, m in enumerate(matches) if m is not None and m.metric.id == "name"), None)
            lab = _txt(rows[bench_row][nc]) if nc is not None and nc < len(rows[bench_row]) else ""
            if lab and lab not in ("-", "--"):
                title = title or lab
            if nc is not None:
                bench.pop(nc, None)
        pages.append(Page(h, labels, matches, data_rows, tickers, names, title, bench,
                          meta.get("data_set") or page_kind(labels), meta.get("list_name"), meta.get("print_date")))
        prev_end = (max(data_rows + ([bench_row] if bench_row is not None else [])) + 1) if data_rows else r
    return pages


def merged_order(seqs: list[list[str]]) -> list[str] | None:
    """Global fund order consistent with every page's (sub)sequence, or None if they conflict."""
    succ: dict[str, set] = {}
    indeg: Counter = Counter()
    first: dict[str, int] = {}
    n = 0
    for s in seqs:
        for t in s:
            if t not in first:
                first[t] = n
                n += 1
            succ.setdefault(t, set())
        for a, b in zip(s, s[1:]):
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    ready = sorted((t for t in succ if indeg[t] == 0), key=first.get)
    out = []
    while ready:
        t = ready.pop(0)
        out.append(t)
        for b in sorted(succ[t], key=first.get):
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
        ready.sort(key=first.get)
    return out if len(out) == len(succ) else None


def resolve_names(pages: list[Page]) -> tuple[dict[str, str], dict]:
    """Attach names to tickers only where every alignment consistent with the paste agrees.

    Facts used: the name pane and the data pane list funds in the same grid order (positions 1..N);
    a page's data row for a fund can only exist if that fund's name row was captured on the same page.
    For each ticker the set of feasible positions is computed by forward/backward dynamic programming;
    a name is assigned only when that set has exactly one position.
    """
    named_pages = [p for p in pages if p.names]
    info = {"listed": 0, "resolved": 0, "ambiguous": 0, "order_conflict": False, "uncaptured_positions": []}
    if not named_pages:
        return {}, info
    N = max(p.n_listed for p in named_pages)
    full_names: dict[int, str] = {}
    for p in named_pages:
        for k, v in p.names.items():
            full_names.setdefault(k, v)
    order = merged_order([p.tickers for p in pages if p.tickers])
    info["listed"] = N
    if order is None:
        info["order_conflict"] = True
        return {}, info
    M = len(order)
    allowed = []
    for t in order:
        s = set(range(1, N + 1))
        for p in pages:
            if t in p.tickers and p.names:
                s &= set(p.names)
        allowed.append(s)
    # forward: positions reachable for ticker i given tickers 0..i placed increasingly
    fwd = []
    prev = {0}
    for i in range(M):
        m = min(prev) if prev else None
        cur = {q for q in allowed[i] if m is not None and q > m}
        fwd.append(cur)
        prev = cur
    bwd = [set() for _ in range(M)]
    nxt = {N + 1}
    for i in range(M - 1, -1, -1):
        m = max(nxt) if nxt else None
        cur = {q for q in allowed[i] if m is not None and q < m}
        bwd[i] = cur
        nxt = cur
    names: dict[str, str] = {}
    used = set()
    for i, t in enumerate(order):
        feas = fwd[i] & bwd[i]
        if len(feas) == 1:
            q = next(iter(feas))
            if q in full_names:
                names[t] = full_names[q]
                used.add(q)
    info["resolved"] = len(names)
    info["ambiguous"] = M - len(names)
    if M + 0 < N and len(names) == M:
        info["uncaptured_positions"] = sorted(set(range(1, N + 1)) - used)
    info["uncaptured_names"] = [full_names[q] for q in info["uncaptured_positions"] if q in full_names]
    info["n_tickers"] = M
    return names, info

"""TREND FOLLOWING persistence: daily signal history + the event log.

Two append-only stores per universe, both sharded by calendar year:

HISTORY  data/trend/<universe>/history/<YYYY>.json   (current year, plain JSON)
                                       <YYYY>.json.gz (closed years, gzipped once)

    {"year": 2026, "fields": ["s10", "f2_8", ..., "f128_512", "bull_mask"],
     "tickers": ["AAPL", "MSFT", ...],                 # append-only within the shard
     "days": [{"date": "2026-01-02", "v": [[...9 ints...] | null, ...]}, ...]}

  One row per ticker per trading day, aligned to `tickers` (a ticker added
  mid-year simply has no entry on earlier days):
    s10        Trend Score x 10, rounded (0.1 resolution) -- null if unscored
    f<speed>   each speed's forecast rounded to an integer on the -20..+20
               scale -- null until that speed is valid
    bull_mask  bit j set when speed j's fast EMA is ABOVE its slow EMA, taken
               from the unrounded raw EWMAC. This is what makes every
               historical crossover date exact even where a forecast rounds
               to 0 -- "when did 32/128 turn bullish?" is a bit flip here.

  Date-major so that a nightly append only adds to the END of the file (git
  stores that as a small delta). Closed years are gzipped once at year
  rollover: they never change again, and ~7 MB/year of plain JSON becomes
  ~1.5 MB. The current year stays plain precisely because it is appended
  to every night -- a gzipped file rewritten daily would defeat git's
  delta compression entirely.

EVENTS   data/trend/<universe>/events/<YYYY>.json(.gz)
    {"fields": EVENT_FIELDS, "rows": [[...], ...]}
  Triggers, upgrades/downgrades and multi-speed confirmations (see events.py).
  Single-speed crossovers are NOT duplicated here -- the history shards'
  bull_mask already records every one of them exactly; recent ones are also
  in recent_events.json for the Triggers view.

Every store function takes an optional directory override (resolved inside
the function, never as a default argument) so tests can redirect it, the
same convention as mom/history.py.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TREND_EVENT_DIRS, TREND_HISTORY_DIRS
from . import SPEED_KEYS

FIELDS = ["s10", *[f"f{k}" for k in SPEED_KEYS], "bull_mask"]
EVENT_FIELDS = ["date", "ticker", "type", "dir", "speeds", "horizon",
                "score_before", "score_after", "from", "to"]


# ------------------------------------------------------------------ shards --
def _dir(kind: str, universe: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return (TREND_HISTORY_DIRS if kind == "history" else TREND_EVENT_DIRS)[universe]


def _years(d: Path) -> list[int]:
    if not d.exists():
        return []
    ys = set()
    for p in d.iterdir():
        stem = p.name.split(".")[0]
        if stem.isdigit() and (p.name.endswith(".json") or p.name.endswith(".json.gz")):
            ys.add(int(stem))
    return sorted(ys)


def _read(d: Path, year: int, default: dict) -> dict:
    plain, gz = d / f"{year}.json", d / f"{year}.json.gz"
    try:
        if plain.exists():
            return json.loads(plain.read_text(encoding="utf-8"))
        if gz.exists():
            with gzip.open(gz, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return default


def _write(d: Path, year: int, doc: dict, compress: bool) -> None:
    d.mkdir(parents=True, exist_ok=True)
    plain, gz = d / f"{year}.json", d / f"{year}.json.gz"
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    if compress:
        # mtime=0 -> byte-identical output for identical input, so re-running
        # a backfill does not produce a spurious git change.
        with gzip.GzipFile(gz, "wb", mtime=0) as fh:
            fh.write(payload.encode("utf-8"))
        if plain.exists():
            plain.unlink()
    else:
        plain.write_text(payload, encoding="utf-8")
        if gz.exists():
            gz.unlink()


def roll_closed_years(universe: str, today: date | None = None,
                      history_dir: Path | None = None, events_dir: Path | None = None) -> int:
    """Gzip any plain shard from a year that has ended. Returns shards rolled."""
    today = today or date.today()
    n = 0
    for kind, override in (("history", history_dir), ("events", events_dir)):
        d = _dir(kind, universe, override)
        for y in _years(d):
            if y < today.year and (d / f"{y}.json").exists():
                _write(d, y, _read(d, y, {}), compress=True)
                n += 1
    return n


def clear(universe: str, history_dir: Path | None = None, events_dir: Path | None = None) -> None:
    """Delete every shard (backfill starts from a clean slate)."""
    for kind, override in (("history", history_dir), ("events", events_dir)):
        d = _dir(kind, universe, override)
        if d.exists():
            for p in d.iterdir():
                if p.name.endswith(".json") or p.name.endswith(".json.gz"):
                    p.unlink()


# ----------------------------------------------------------------- encoding --

_NULL = -32768    # in-memory sentinel for JSON null


class Encoded:
    """One ticker's encoded rows, held as a compact int32 matrix (n_days x 9)
    until a shard actually needs them. A 10-year backfill of ~2,000 tickers
    as Python lists would be ~1 GB; as int32 arrays it is ~180 MB."""
    __slots__ = ("dates", "arr")

    def __init__(self, dates: list[str], arr: np.ndarray):
        self.dates, self.arr = dates, arr

    def __len__(self) -> int:
        return len(self.dates)

    def after(self, since: str | None) -> "Encoded":
        if since is None:
            return self
        i = int(np.searchsorted(np.array(self.dates), since, side="right"))
        return Encoded(self.dates[i:], self.arr[i:])

    def rows(self) -> dict[str, list]:
        return {d: [None if v == _NULL else int(v) for v in row]
                for d, row in zip(self.dates, self.arr.tolist())}


def encode_rows(forecast: pd.DataFrame, raw: pd.DataFrame, score: pd.Series) -> Encoded:
    """Encode every date with at least one valid speed as
    [s10, f..., bull_mask] (see the module docstring for the field meanings)."""
    f = forecast[list(SPEED_KEYS)].to_numpy(dtype=float)
    r = raw[list(SPEED_KEYS)].to_numpy(dtype=float)
    s = score.reindex(forecast.index).to_numpy(dtype=float)
    keep = np.isfinite(f).any(axis=1)
    bits = (1 << np.arange(len(SPEED_KEYS)))
    masks = ((np.nan_to_num(r, nan=0.0) > 0) * bits).sum(axis=1)
    arr = np.full((len(f), 2 + len(SPEED_KEYS)), _NULL, dtype=np.int32)
    ok_s = np.isfinite(s)
    arr[ok_s, 0] = np.round(s[ok_s] * 10).astype(np.int32)
    ok_f = np.isfinite(f)
    fr = np.round(np.where(ok_f, f, 0.0)).astype(np.int32)
    arr[:, 1:-1] = np.where(ok_f, fr, _NULL)
    arr[:, -1] = masks
    dates = [d.strftime("%Y-%m-%d") for d in forecast.index[keep]]
    return Encoded(dates, arr[keep])


# ------------------------------------------------------------------ history --
def last_date(universe: str, history_dir: Path | None = None) -> str | None:
    d = _dir("history", universe, history_dir)
    for y in reversed(_years(d)):
        days = _read(d, y, {}).get("days") or []
        if days:
            return days[-1]["date"]
    return None


def _year_rows(series: dict[str, "Encoded | dict"], y: int) -> dict[str, dict[str, list]]:
    """{date: {ticker: row}} for one calendar year, materialized only now."""
    lo, hi = f"{y}-01-01", f"{y}-12-31"
    out: dict[str, dict[str, list]] = {}
    for t, enc in series.items():
        if isinstance(enc, Encoded):
            ds = np.array(enc.dates)
            i0, i1 = np.searchsorted(ds, lo, "left"), np.searchsorted(ds, hi, "right")
            rows = Encoded(enc.dates[i0:i1], enc.arr[i0:i1]).rows()
        else:
            rows = {k: v for k, v in enc.items() if lo <= k <= hi}
        for ds_, row in rows.items():
            out.setdefault(ds_, {})[t] = row
    return out


def append(universe: str, series: dict[str, "Encoded | dict"], today: date | None = None,
           history_dir: Path | None = None) -> dict:
    """Append encoded rows ({ticker: Encoded | {date: row}}) to the yearly shards.

    Idempotent: a date already present in a shard is never rewritten (history
    is a record of what the system said that day). Dates are kept ascending.
    Returns {"dates_added": n, "rows_added": n, "breadth": [breadth_row, ...]}
    -- the per-date breadth summary is computed here, while each day's rows
    are in memory anyway, so breadth_history never needs a second pass."""
    today = today or date.today()
    d = _dir("history", universe, history_dir)
    years_present: set[int] = set()
    for enc in series.values():
        ds = enc.dates if isinstance(enc, Encoded) else list(enc)
        years_present.update(int(x[:4]) for x in (ds[:1] + ds[-1:]) if x)
    if years_present:
        years_present = set(range(min(years_present), max(years_present) + 1))
    added_dates = added_rows = 0
    breadth: list[dict] = []
    for y in sorted(years_present):
        year_rows = _year_rows(series, y)
        if not year_rows:
            continue
        by_year = {y: year_rows}
        doc = _read(d, y, {"year": y, "fields": FIELDS, "tickers": [], "days": []})
        tickers = doc.get("tickers", [])
        pos = {t: i for i, t in enumerate(tickers)}
        have = {day["date"] for day in doc.get("days", [])}
        new_days = []
        for ds in sorted(by_year[y]):
            if ds in have:
                continue
            day_rows = by_year[y][ds]
            for t in sorted(day_rows):
                if t not in pos:
                    pos[t] = len(tickers)
                    tickers.append(t)
            v = [None] * len(tickers)
            for t, row in day_rows.items():
                v[pos[t]] = row
            new_days.append({"date": ds, "v": v})
            breadth.append(breadth_row(ds, list(day_rows.values())))
            added_dates += 1
            added_rows += len(day_rows)
        if not new_days:
            continue
        doc["tickers"] = tickers
        doc["days"] = sorted(doc.get("days", []) + new_days, key=lambda x: x["date"])
        doc["fields"] = FIELDS
        _write(d, y, doc, compress=y < today.year)
    return {"dates_added": added_dates, "rows_added": added_rows, "breadth": breadth}


def breadth_row(ds: str, rows: list[list]) -> dict:
    """Universe-level summary of one day's encoded rows: how many assets are
    scored, the score distribution, and the share bullish AT EACH SPEED --
    the market's own trend term structure."""
    scores = np.array([r[0] / 10.0 for r in rows if r[0] is not None], dtype=float)
    speed_bull = []
    for j in range(len(SPEED_KEYS)):
        valid = [r for r in rows if r[1 + j] is not None]
        speed_bull.append(round(sum(1 for r in valid if (r[-1] >> j) & 1) / len(valid), 4)
                          if valid else None)
    if not len(scores):
        return {"date": ds, "n": 0, "speed_bull": speed_bull}
    return {"date": ds, "n": int(len(scores)),
            "mean": round(float(scores.mean()), 3), "median": round(float(np.median(scores)), 3),
            "pct_bull": round(float((scores >= 5).mean()), 4),
            "pct_bear": round(float((scores < -5).mean()), 4),
            "speed_bull": speed_bull}


def series_for(universe: str, ticker: str, start_year: int | None = None,
               history_dir: Path | None = None, shard_loader=None) -> pd.DataFrame:
    """One ticker's persisted history as a DataFrame indexed by date with
    columns score, f<speed>..., and bull_<speed> (+1/-1, NaN if invalid).
    `shard_loader(year) -> doc` lets the view pass a cached reader."""
    d = _dir("history", universe, history_dir)
    years = [y for y in _years(d) if start_year is None or y >= start_year]
    recs = []
    for y in years:
        doc = shard_loader(y) if shard_loader else _read(d, y, {})
        tickers = doc.get("tickers") or []
        if ticker not in tickers:
            continue
        j = tickers.index(ticker)
        for day in doc.get("days", []):
            v = day["v"]
            if j < len(v) and v[j] is not None:
                recs.append((day["date"], v[j]))
    if not recs:
        return pd.DataFrame()
    return decode(recs)


def decode(recs: list[tuple[str, list]]) -> pd.DataFrame:
    idx = pd.to_datetime([r[0] for r in recs])
    arr = [r[1] for r in recs]
    out = {"score": [None if a[0] is None else a[0] / 10.0 for a in arr]}
    for j, k in enumerate(SPEED_KEYS):
        out[f"f{k}"] = [a[1 + j] for a in arr]
        out[f"bull_{k}"] = [(None if a[1 + j] is None else (1 if (a[-1] >> j) & 1 else -1))
                            for a in arr]
    return pd.DataFrame(out, index=idx).astype(float)


def read_shard(universe: str, year: int, history_dir: Path | None = None) -> dict:
    return _read(_dir("history", universe, history_dir), year, {})


def years(universe: str, history_dir: Path | None = None) -> list[int]:
    return _years(_dir("history", universe, history_dir))


# ------------------------------------------------------------------- events --
def encode_event(e: dict) -> list:
    return [e["date"], e["ticker"], e["type"], e["dir"],
            "".join(str(SPEED_KEYS.index(k)) for k in e.get("speeds") or []),
            e.get("horizon"), e.get("score_before"), e.get("score_after"),
            e.get("from"), e.get("to")]


def decode_event(row: list) -> dict:
    e = dict(zip(EVENT_FIELDS, row))
    e["speeds"] = [SPEED_KEYS[int(c)] for c in (e.get("speeds") or "")]
    e["speed"] = e["speeds"][0] if e["type"] == "cross" and e["speeds"] else None
    e["n_speeds"] = len(e["speeds"])
    return e


def append_events(universe: str, events: list[dict], today: date | None = None,
                  events_dir: Path | None = None) -> int:
    """Append non-crossover events to the yearly event shards. Idempotent on
    (date, ticker, type, dir, speeds)."""
    today = today or date.today()
    d = _dir("events", universe, events_dir)
    by_year: dict[int, list[list]] = {}
    for e in events:
        if e["type"] == "cross":
            continue
        by_year.setdefault(int(e["date"][:4]), []).append(encode_event(e))
    added = 0
    for y, rows in sorted(by_year.items()):
        doc = _read(d, y, {"year": y, "fields": EVENT_FIELDS, "rows": []})
        seen = {tuple(r[:5]) for r in doc.get("rows", [])}
        new = []
        for r in rows:                         # dedupe against the shard AND within this batch
            k = tuple(r[:5])
            if k not in seen:
                seen.add(k)
                new.append(r)
        if not new:
            continue
        doc["rows"] = sorted(doc.get("rows", []) + new, key=lambda r: (r[0], r[1], r[2]))
        doc["fields"] = EVENT_FIELDS
        _write(d, y, doc, compress=y < today.year)
        added += len(new)
    return added


def events_for(universe: str, ticker: str | None = None, start_year: int | None = None,
               events_dir: Path | None = None) -> list[dict]:
    d = _dir("events", universe, events_dir)
    out = []
    for y in _years(d):
        if start_year is not None and y < start_year:
            continue
        for r in _read(d, y, {}).get("rows", []):
            if ticker is None or r[1] == ticker:
                out.append(decode_event(r))
    return out


def merge_recent(existing: list[list], new_events: list[dict], as_of: str,
                 keep_days: int, keep_cross_days: int) -> list[list]:
    """recent_events.json rows: prior rows + new events, deduped, pruned to
    the recent window (crossovers to the shorter window)."""
    asof = pd.Timestamp(as_of)
    cut = (asof - pd.tseries.offsets.BDay(keep_days)).strftime("%Y-%m-%d")
    cut_x = (asof - pd.tseries.offsets.BDay(keep_cross_days)).strftime("%Y-%m-%d")
    rows = {tuple(r[:5]): r for r in existing}
    for e in new_events:
        r = encode_event(e)
        rows[tuple(r[:5])] = r
    kept = [r for r in rows.values() if r[0] > (cut_x if r[2] == "cross" else cut)]
    kept.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
    return kept

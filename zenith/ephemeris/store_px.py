"""Local price store: the Release-asset Parquet, read one ticker at a time.

`ensure_local()` downloads the nightly asset into EPHEMERIS_PX_DIR when the
local copy is missing or older than EPHEMERIS_PX_MAX_AGE_HOURS, and falls
back to a stale copy if the download fails -- the game keeps working offline.
`PxStore` maps ticker -> row group (prefetch writes one per ticker), so a
chart load reads ~one row group, never the whole file.

Weekly and Monthly bars are resampled from Daily (Friday / month-end close).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EPHEMERIS_PX_DIR, EPHEMERIS_PX_MAX_AGE_HOURS, EPHEMERIS_RELEASE_URL

# timeframe -> (source file, resample rule or None)
SOURCES = {"Daily": ("daily.parquet", None), "Weekly": ("daily.parquet", "W-FRI"),
           "Monthly": ("daily.parquet", "ME"), "1H": ("hourly.parquet", None),
           "4H": ("hourly.parquet", "4H")}
# rough bars per source bar, for eligibility estimates before exact loading
BAR_RATIO = {"Daily": 1.0, "Weekly": 1 / 5, "Monthly": 1 / 21, "1H": 1.0, "4H": 2 / 7}


def _fresh(p: Path, max_age_h: float) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600


def ensure_local(name: str, max_age_h: float = EPHEMERIS_PX_MAX_AGE_HOURS) -> Path | None:
    """Path to a usable local copy of a release asset, downloading if stale."""
    p = EPHEMERIS_PX_DIR / name
    if _fresh(p, max_age_h):
        return p
    try:
        import requests
        EPHEMERIS_PX_DIR.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with requests.get(EPHEMERIS_RELEASE_URL.format(name=name), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(p)
    except Exception:
        pass                                   # offline / no release yet -> stale copy if any
    return p if p.exists() else None


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "4H":
        # Hourly stamps are exchange wall-clock (US/Eastern). Session-only series
        # (equities, ETFs, indices: 09:30..15:30) bin from the 09:30 open into
        # 09:30-13:30 and 13:30-16:00 (a short bar). Round-the-clock series
        # (crypto, FX) bin on plain 4-hour boundaries.
        hours = df.index.hour
        rth = len(hours) > 0 and hours.min() >= 9 and hours.max() <= 16
        g = df.resample("4h", offset="9h30min" if rth else "0h", label="left", closed="left")
    else:
        g = df.resample(rule, label="right", closed="right")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


class PxStore:
    """Read-through cache over one or more timeframe Parquet files."""

    def __init__(self, root: Path | None = None, cache_size: int = 64):
        self.root = root
        self._files: dict[str, object] = {}
        self._index: dict[str, dict[str, list[int]]] = {}
        self._lock = threading.Lock()
        self._lru: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------- files ----
    def _pf(self, name: str):
        if name not in self._files:
            import pyarrow.parquet as pq
            p = (self.root / name) if self.root else ensure_local(name)
            if p is None or not Path(p).exists():
                self._files[name] = None
                self._index[name] = {}
            else:
                pf = pq.ParquetFile(p)
                self._files[name] = pf
                self._index[name] = self._build_index(pf)
        return self._files[name]

    @staticmethod
    def _build_index(pf) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = {}
        md = pf.metadata
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(0).statistics
            if st is not None and st.has_min_max and st.min == st.max:
                t = st.min.decode() if isinstance(st.min, bytes) else st.min
                idx.setdefault(t, []).append(i)
            else:                              # fall back: read the ticker column of this group
                for t in set(pf.read_row_group(i, columns=["ticker"]).column(0).to_pylist()):
                    idx.setdefault(t, []).append(i)
        return idx

    def available(self, tf: str = "Daily") -> bool:
        return self._pf(SOURCES[tf][0]) is not None

    def counts(self, tf: str = "Daily") -> dict[str, int]:
        """Approximate bar count per ticker at this timeframe (cheap: metadata only)."""
        name, _ = SOURCES[tf]
        pf = self._pf(name)
        if pf is None:
            return {}
        md = pf.metadata
        ratio = BAR_RATIO[tf]
        return {t: int(sum(md.row_group(i).num_rows for i in rgs) * ratio)
                for t, rgs in self._index[name].items()}

    # ------------------------------------------------------------- bars -----
    def bars(self, ticker: str, tf: str = "Daily") -> pd.DataFrame | None:
        """OHLCV DataFrame (float64) indexed by ts, or None."""
        key = (ticker, tf)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
            name, rule = SOURCES[tf]
            pf = self._pf(name)
            rgs = self._index.get(name, {}).get(ticker)
            if pf is None or not rgs:
                return None
            import pyarrow as pa
            tbl = pa.concat_tables([pf.read_row_group(i) for i in rgs])
            df = tbl.to_pandas()
            df = df[df["ticker"] == ticker].drop(columns="ticker").set_index("ts").sort_index()
            df = df.astype("float64")
            if rule:
                df = resample(df, rule)
            self._lru[key] = df
            if len(self._lru) > self._cache_size:
                self._lru.popitem(last=False)
            return df


def arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"ts": df.index.values, **{c: df[c].to_numpy(np.float64) for c in
                                      ("open", "high", "low", "close", "volume")}}

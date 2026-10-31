"""Cboe index history CSVs (free, no key), cached for the day.

Two layouts exist: strategy benchmarks ship `DATE,<NAME>` and the volatility
family ships `DATE,OPEN,HIGH,LOW,CLOSE`. Both are read as one close series —
the CLOSE column when present, else the single value column. (The older
vault/repro/data.cboe_index assumes two columns and would mislabel the
five-column files; this reader is the general one.)
"""

from __future__ import annotations

import io
import time

import pandas as pd

from ..config import BROWSER_HEADERS, CBOE_INDEX_CSV, STRATIDX_CACHE_DIR


def parse_csv(text: str, name: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "DATE" not in df.columns:
        raise ValueError(f"{name}: no DATE column")
    col = "CLOSE" if "CLOSE" in df.columns else (name.upper() if name.upper() in df.columns
                                                 else [c for c in df.columns if c != "DATE"][-1])
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="mixed", errors="coerce"), name=name)
    s = s[s.index.notna()].dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def cboe_csv(name: str, max_age_h: float = 12.0, cache_dir=STRATIDX_CACHE_DIR) -> pd.Series:
    """Close series for one Cboe index; empty Series if unavailable."""
    p = cache_dir / f"{name}.csv"
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600:
        try:
            return parse_csv(p.read_text(encoding="utf-8"), name)
        except Exception:
            pass
    try:
        import requests
        r = requests.get(CBOE_INDEX_CSV.format(name=name), timeout=45,
                         headers={"User-Agent": BROWSER_HEADERS["User-Agent"]})
        r.raise_for_status()
        s = parse_csv(r.text, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.text, encoding="utf-8")
        time.sleep(0.2)
        return s
    except Exception:
        if p.exists():
            try:
                return parse_csv(p.read_text(encoding="utf-8"), name)
            except Exception:
                pass
        return pd.Series(dtype=float, name=name)


def many(names, **kw) -> dict[str, pd.Series]:
    return {n: cboe_csv(n, **kw) for n in names}

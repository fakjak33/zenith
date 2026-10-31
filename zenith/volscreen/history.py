"""The IV history: one row per ticker per day, kept on the `ephemeris-px` release.

Free historical implied vol does not exist, so this table is the only record
of it — append-only, deduped on (ticker, date), and a write that would SHRINK
it is refused. `src` says where a row came from:
    0 observed   from that night's real chain
    1 index      Cboe's own volatility index for the name (VIX9D..VIX1Y for
                 SPX/SPY/XSP, VXN for NDX/QQQ, RVX, VXD, VXEEM, GVZ, OVX)
    2 modeled    THETA RUN's modeled IV (HAR-lite x index VRP), so ranks have
                 a provisional base until observed rows accrue
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import VOLSCREEN_TENORS

TERM_COLS = (["src", "spot"] + [f"iv_{t}" for t in VOLSCREEN_TENORS]
             + ["ivx_30", "skew25", "bf25", "rv_cc20", "rv_yz20", "garch_30", "vrp_30",
                "opt_vol", "oi", "atm_spread"])
SURF_COLS = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"]
SEED_YEARS = 3

# Cboe volatility indices -> (underlyings they price, maturity in calendar days)
SEED_MAP = {
    "SPX": {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 183, "VIX1Y": 365},
    "NDX": {"VXN": 30}, "RUT": {"RVX": 30}, "DIA": {"VXD": 30}, "EEM": {"VXEEM": 30},
    "GLD": {"GVZ": 30}, "USO": {"OVX": 30},
}
SEED_ALIAS = {"SPY": "SPX", "XSP": "SPX", "QQQ": "NDX", "IWM": "RUT"}


def empty_term() -> pd.DataFrame:
    df = pd.DataFrame(columns=["ticker", "date"] + TERM_COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_term(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return empty_term()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return empty_term()
    df["date"] = pd.to_datetime(df["date"])
    for c in TERM_COLS:
        if c not in df:
            df[c] = np.nan
    return df[["ticker", "date"] + TERM_COLS]


def read_ticker(path: Path | None, ticker: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One ticker's rows without loading the whole file (row-group filter)."""
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, filters=[("ticker", "=", ticker)],
                             columns=(["ticker", "date"] + columns) if columns else None)
    except Exception:
        return pd.DataFrame()
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def append(hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Union, newest write wins on (ticker, date); observed rows outrank seeds."""
    if rows is None or rows.empty:
        return hist
    both = pd.concat([hist, rows], ignore_index=True) if len(hist) else rows.copy()
    both["date"] = pd.to_datetime(both["date"])
    both["_o"] = -both["src"].fillna(9).astype(float)        # src 0 sorts last -> kept
    both = both.sort_values(["ticker", "date", "_o"]).drop_duplicates(["ticker", "date"], keep="last")
    return both.drop(columns="_o").sort_values(["ticker", "date"]).reset_index(drop=True)


def write(df: pd.DataFrame, path: Path, prev_rows: int, cols: list[str]) -> int:
    """One row group per ticker, float32 + zstd. Refuses to shrink the table."""
    if len(df) < prev_rows:
        raise RuntimeError(f"history would shrink {prev_rows} -> {len(df)} rows; refusing to write")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    out = df.copy()
    for c in cols:
        if c not in out:
            out[c] = np.nan
        if c in ("src", "tenor"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int16")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    writer = None
    try:
        for _, g in out.groupby("ticker", sort=True):
            tbl = pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return 0
    tmp.replace(path)
    return path.stat().st_size


def load_surf(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["ticker", "date", "tenor"] + SURF_COLS)
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "tenor"] + SURF_COLS)


def append_surf(hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return hist
    both = pd.concat([hist, rows], ignore_index=True) if len(hist) else rows.copy()
    both["date"] = pd.to_datetime(both["date"])
    return (both.drop_duplicates(["ticker", "date", "tenor"], keep="last")
            .sort_values(["ticker", "date", "tenor"]).reset_index(drop=True))


# ---------------------------------------------------------------- seeding --

def seed_family(ticker: str) -> dict | None:
    return SEED_MAP.get(SEED_ALIAS.get(ticker, ticker))


def _har_lite(close: pd.Series) -> pd.Series:
    """0.2 / 0.4 / 0.4 blend of 5 / 21 / 63-day realized vol (THETA's forecast)."""
    r = np.log(close.astype(float)).diff()
    def rv(n):
        return np.sqrt((r * r).rolling(n, min_periods=max(3, n // 2)).mean() * 252)
    return 0.2 * rv(5) + 0.4 * rv(21) + 0.4 * rv(63)


def seed_from_indices(ticker: str, vol_idx: dict[str, pd.Series], close: pd.Series | None,
                      scale: float = 1.0) -> pd.DataFrame:
    """src=1 rows from Cboe vol indices, interpolated to our tenors in total variance."""
    from .surface import cm_interp

    fam = seed_family(ticker)
    if not fam:
        return pd.DataFrame()
    cols = {name: vol_idx.get(name) for name in fam}
    cols = {k: v for k, v in cols.items() if v is not None and len(v)}
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame({k: v / 100.0 for k, v in cols.items()}).sort_index()
    panel = panel[panel.index >= panel.index.max() - pd.DateOffset(years=SEED_YEARS)]
    Ts = np.array([fam[k] / 365.0 for k in panel.columns])
    rows = []
    for d, vals in panel.iterrows():
        v = vals.to_numpy(float)
        rec = {"ticker": ticker, "date": d, "src": 1}
        for t in VOLSCREEN_TENORS:
            if len(Ts) == 1:
                rec[f"iv_{t}"] = float(v[0]) if t == 30 else np.nan
            else:
                iv_t, ext = cm_interp(Ts, v, t / 365.0)
                rec[f"iv_{t}"] = iv_t if not ext or t in (7,) else np.nan
        rec["ivx_30"] = rec["iv_30"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    if close is not None and len(close):
        c = close.sort_index()
        fc = _har_lite(c).reindex(out["date"]).to_numpy()
        out["spot"] = c.reindex(out["date"]).to_numpy() * scale
        out["vrp_30"] = out["iv_30"].to_numpy() - fc
    return out


def seed_from_theta(ticker: str, theta_path: Path | None) -> pd.DataFrame:
    """src=2 rows from THETA RUN's modeled IV table (release asset)."""
    if theta_path is None or not Path(theta_path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(theta_path, filters=[("ticker", "=", ticker)],
                             columns=["ticker", "ts", "close", "iv30", "rv_fc", "rv21"])
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"ts": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= df["date"].max() - pd.DateOffset(years=SEED_YEARS)]
    out = pd.DataFrame({"ticker": ticker, "date": df["date"].to_numpy(), "src": 2,
                        "spot": df["close"].to_numpy(float), "iv_30": df["iv30"].to_numpy(float),
                        "ivx_30": df["iv30"].to_numpy(float), "rv_cc20": df["rv21"].to_numpy(float)})
    out["vrp_30"] = out["iv_30"] - df["rv_fc"].to_numpy(float)
    return out[np.isfinite(out["iv_30"])]


def coverage(hist: pd.DataFrame, ticker: str) -> dict:
    """Observed-day count and the source mix behind a ticker's ranks."""
    g = hist[hist["ticker"] == ticker]
    if g.empty:
        return {"observed": 0, "seed": None, "rows": 0}
    src = g["src"].fillna(0).astype(int)
    seed = None
    if (src == 1).any():
        seed = "index"
    elif (src == 2).any():
        seed = "modeled"
    return {"observed": int((src == 0).sum()), "seed": seed, "rows": int(len(g))}


def finite(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None

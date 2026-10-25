"""THETA RUN data build: universe, earnings calendar and the modeled-IV table.

    python -m zenith.theta.build            # full build -> EPHEMERIS_PX_DIR/theta.parquet
    python -m zenith.theta.build --limit 20 # quick dev build

Inputs are all free: the EPHEMERIS daily price store (GitHub Release asset),
^VIX and ^IRX closes from Yahoo, and yfinance earnings dates (cached by the
vault's repro layer). Outputs: data/theta/universe.json, earnings.json and
build.json (committed, small) plus theta.parquet (release asset, gitignored
locally). The nightly workflow re-runs this after the EPHEMERIS prefetch.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import THETA_FILES, THETA_START
from . import ivmodel, load, save

ETFS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "HYG", "LQD", "GLD", "SLV", "USO", "UNG",
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XOP", "XBI", "SMH",
        "KRE", "GDX", "GDXJ", "EEM", "EFA", "FXI", "EWZ", "EWJ", "ARKK", "VNQ", "IBB", "KWEB",
        "IYR", "OIH", "XME", "XRT"]
N_STOCKS = 110
DV_WINDOW = 252
SERIES = ("_VIX", "_RATE")           # pseudo-tickers carried in the parquet (close only)
COLS = ["open", "high", "low", "close", "volume", "rv21", "rv_fc", "iv_base", "em",
        "next_earn", "iv30", "iv_rank", "iv_pct", "skew", "beta"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dollar_volume(df: pd.DataFrame, n: int = DV_WINDOW) -> float:
    tail = df.tail(n)
    return float((tail["close"] * tail["volume"]).median()) if len(tail) else 0.0


def tiers(dv: dict[str, float], core: set[str]) -> dict[str, str]:
    """A/B/C by dollar-volume tercile; the core index ETFs are always A."""
    if not dv:
        return {}
    ranked = sorted(dv, key=dv.get, reverse=True)
    n = len(ranked)
    out = {}
    for i, t in enumerate(ranked):
        out[t] = "A" if (t in core or i < n / 3) else ("B" if i < 2 * n / 3 else "C")
    return out


def pick_universe(store, rows: list[dict], n_stocks: int = N_STOCKS,
                  limit: int | None = None) -> list[dict]:
    meta = {r["ticker"]: r for r in rows}
    etfs = [t for t in ETFS if t in meta]
    stocks = [r["ticker"] for r in rows if r.get("cls") == "Russell 1000"]
    dv: dict[str, float] = {}
    first: dict[str, str] = {}
    for t in etfs + stocks:
        df = store.bars(t)
        if df is None or len(df) < DV_WINDOW:
            continue
        dv[t] = dollar_volume(df)
        first[t] = df.index[0].date().isoformat()
    top = sorted((t for t in stocks if t in dv), key=dv.get, reverse=True)[:n_stocks]
    chosen = [t for t in etfs if t in dv] + top
    if limit:
        chosen = chosen[:limit]
    tier = tiers({t: dv[t] for t in chosen}, core={"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"})
    out = []
    for t in chosen:
        m = meta[t]
        kind = "index" if t == "SPY" else ("etf" if t in ETFS else "stock")
        out.append({"ticker": t, "name": m.get("name", t), "kind": kind, "tier": tier[t],
                    "sector": m.get("sector") or m.get("category") or "",
                    "listed_from": max(first[t], THETA_START)})
    return out


def frame_for(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df.index >= pd.Timestamp(THETA_START), COLS].copy()
    for c in COLS:
        out[c] = out[c].astype("float32")
    return out


def write_parquet(frames: dict[str, pd.DataFrame], path) -> int:
    """One row group per ticker (PxStore-style), float32 + zstd."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    writer = None
    try:
        for t, df in frames.items():
            d = df.copy()
            d.insert(0, "ticker", t)
            d.index.name = "ts"
            tbl = pa.Table.from_pandas(d.reset_index(), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(path)
    return path.stat().st_size


def calibration(frames: dict[str, pd.DataFrame], universe: list[dict]) -> dict:
    """IV vs subsequently realized vol: the premium the model builds in."""
    kinds = {u["ticker"]: u["kind"] for u in universe}
    ratios = {"index": [], "etf": [], "stock": []}
    for t, df in frames.items():
        if t in SERIES:
            continue
        r = np.log(df["close"].astype(float)).diff()
        fwd = np.sqrt((r.pow(2)).rolling(21).mean().shift(-21) * 252)
        q = (df["iv_base"] / fwd).replace([np.inf, -np.inf], np.nan).dropna()
        if len(q):
            ratios[kinds[t]].append(float(q.median()))
    spy = frames.get("SPY")
    vix = frames.get("_VIX")
    spy_err = None
    if spy is not None and vix is not None:
        j = pd.concat([spy["iv_base"], vix["close"]], axis=1).dropna()
        spy_err = float((j.iloc[:, 0] - j.iloc[:, 1]).abs().max()) if len(j) else None
    return {"median_iv_over_fwd_rv": {k: (round(float(np.median(v)), 3) if v else None)
                                     for k, v in ratios.items()},
            "spy_iv_minus_vix_max_abs": spy_err}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="zenith.theta.build")
    ap.add_argument("--limit", type=int, default=None, help="dev: only the first N tickers")
    ap.add_argument("--refresh-earnings", action="store_true")
    a = ap.parse_args(argv)

    from ..ephemeris import universe as eph_universe
    from ..ephemeris.store_px import PxStore
    from ..vault.repro import data as rdata

    store = PxStore()
    if not store.available("Daily"):
        raise SystemExit("EPHEMERIS daily.parquet not available; run the ephemeris prefetch first")
    rows = eph_universe.load()
    universe = pick_universe(store, rows, limit=a.limit)
    print(f"[theta] universe: {len(universe)} tickers")

    vix = rdata.yahoo_close("^VIX") / 100.0
    rate = (rdata.yahoo_close("^IRX") / 100.0).clip(lower=0.0)

    earnings = load("earnings", {}).get("dates", {})
    for u in universe:
        if u["kind"] != "stock":
            continue
        if a.refresh_earnings or u["ticker"] not in earnings:
            earnings[u["ticker"]] = rdata.earnings_dates(u["ticker"])

    spy_px = store.bars("SPY")
    spy_r = ivmodel.log_returns(spy_px["close"])
    mult = ivmodel.vrp_multiplier(vix.reindex(spy_px.index).ffill(), ivmodel.har_forecast(spy_r))

    frames: dict[str, pd.DataFrame] = {}
    for u in universe:
        px = store.bars(u["ticker"])
        if px is None or px.empty:
            continue
        f = ivmodel.features(px, earnings.get(u["ticker"], []), mult, vix, spy_r, u["kind"])
        frames[u["ticker"]] = frame_for(f)
    cal = spy_px.index[spy_px.index >= pd.Timestamp(THETA_START)]
    for name, s in (("_VIX", vix), ("_RATE", rate)):
        df = pd.DataFrame({c: np.nan for c in COLS}, index=cal)
        df["close"] = s.reindex(cal).ffill().astype(float)
        frames[name] = frame_for(df)

    size = write_parquet(frames, THETA_FILES["local_parquet"])
    save("universe", {"generated": _now(), "start": THETA_START, "rows": universe})
    save("earnings", {"generated": _now(), "source": "yfinance get_earnings_dates (~2002-)",
                      "dates": {t: earnings.get(t, []) for t in sorted(earnings)}})
    report = {"generated": _now(), "tickers": len(universe), "rows": int(sum(len(f) for f in frames.values())),
              "parquet_mb": round(size / 1e6, 1),
              "first_date": str(cal[0].date()), "last_date": str(cal[-1].date()),
              "earnings_coverage": {t: len(v) for t, v in earnings.items()},
              "calibration": calibration(frames, universe)}
    save("build", report)
    print(json.dumps({k: v for k, v in report.items() if k != "earnings_coverage"}, indent=2))


if __name__ == "__main__":
    main()

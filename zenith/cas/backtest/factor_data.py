"""Load long-history academic factor returns for the factor-momentum backtest.

Two free / low-friction sources, both optional and degrade-gracefully:

  * Ken French Data Library (via ``pandas-datareader``) — regional 5-factor sets
    (Mkt-RF, SMB, HML, RMW, CMA) plus momentum (WML), monthly. No key needed.
  * AQR Data Library (xlsx) — e.g. QMJ (Quality-minus-Junk), BAB (Betting-
    Against-Beta). No API, so we read a workbook the user drops into
    ``data/cas/cache/`` (named ``aqr_qmj.xlsx`` / ``aqr_bab.xlsx``). Absent → skipped.

Everything returns a dict ``{factor_label -> monthly return pd.Series}`` with the
returns expressed as decimals (0.01 = 1%). Labels are ``REGION:FACTOR`` so the
backtest can group/report by region.
"""

from __future__ import annotations

import datetime as _dt
import io

import pandas as pd

from .. import store_cas
from ...config import CAS_CACHE_DIR

# Pull full history, not pandas-datareader's default trailing 5 years. US factors
# start 1963; developed/EM regions begin ~1990 and are returned from their own start.
_START = _dt.datetime(1963, 1, 1)

# Ken French regional dataset names. (Some are skipped silently if the library
# renames them; the loader never raises.)
FF_5F = {
    "US": "North_America_5_Factors",
    "DEV": "Developed_ex_US_5_Factors",
    "EU": "Europe_5_Factors",
    "JP": "Japan_5_Factors",
    "AP": "Asia_Pacific_ex_Japan_5_Factors",
    "EM": "Emerging_5_Factors",
}
FF_MOM = {
    "US": "North_America_Mom_Factor",
    "DEV": "Developed_ex_US_Mom_Factor",
    "EU": "Europe_Mom_Factor",
    "JP": "Japan_Mom_Factor",
    "AP": "Asia_Pacific_ex_Japan_Mom_Factor",
    "EM": "Emerging_MOM_Factor",
}

_CACHE_KEY = "ff_factors"


def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ken French tables come back with a monthly PeriodIndex; normalise it."""
    if hasattr(df.index, "to_timestamp"):
        df = df.copy()
        df.index = df.index.to_timestamp()
    return df


def load_french_factors(max_age_hours: float = 720.0) -> dict[str, pd.Series]:
    """Regional Fama-French factor return series (decimals), monthly. Cached for
    ~30 days (they only update monthly). Returns {} if the library/network is
    unavailable."""
    cached = store_cas.cache_get(_CACHE_KEY, max_age_hours)
    if cached:
        return {k: pd.read_json(io.StringIO(v), typ="series", orient="split")
                for k, v in cached.items()}

    try:
        import pandas_datareader.data as web
    except Exception:
        return {}

    out: dict[str, pd.Series] = {}
    for region, ds in FF_5F.items():
        try:
            tbl = _to_ts_index(web.DataReader(ds, "famafrench", start=_START)[0] / 100.0)
        except Exception:
            continue
        for col in tbl.columns:
            name = str(col).strip()
            if name == "RF":
                continue
            out[f"{region}:{name}"] = tbl[col].dropna()
    for region, ds in FF_MOM.items():
        try:
            tbl = _to_ts_index(web.DataReader(ds, "famafrench", start=_START)[0] / 100.0)
        except Exception:
            continue
        if len(tbl.columns):
            out[f"{region}:WML"] = tbl[tbl.columns[0]].dropna()

    if out:
        store_cas.cache_put(_CACHE_KEY, {k: v.to_json(orient="split") for k, v in out.items()})
    return out


# AQR public monthly factor workbooks — auto-downloaded (no manual step).
# label -> (filename on aqr.com, sheet name inside the workbook)
_AQR_BASE = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
_AQR_FILES = {
    "QMJ": ("Quality-Minus-Junk-Factors-Monthly.xlsx", "QMJ Factors"),
    "BAB": ("Betting-Against-Beta-Equity-Factors-Monthly.xlsx", "BAB Factors"),
}
# Region columns we try to pull from an AQR sheet, mapped to our region keys.
_AQR_REGIONS = {"USA": "US", "Global": "GLB", "Japan": "JP", "Europe": "EU"}


def _fresh(path, max_age_days: float) -> bool:
    import time
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400


def _find_header_row(raw: pd.DataFrame, limit: int = 40) -> int | None:
    """AQR sheets carry a disclaimer block before the table; find the row whose
    first cell is 'DATE'."""
    for i in range(min(limit, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() == "date":
            return i
    return None


def load_aqr_factors(max_age_days: float = 30.0) -> dict[str, pd.Series]:
    """Auto-download AQR QMJ/BAB monthly factors from their public URLs, cache to
    data/cas/cache/, and parse the regional columns. Degrades to {} on any failure
    (offline, layout change, etc.) so the backtest still runs on French data."""
    out: dict[str, pd.Series] = {}
    for label, (fname, sheet) in _AQR_FILES.items():
        path = CAS_CACHE_DIR / fname
        if not _fresh(path, max_age_days):
            try:
                import requests
                r = requests.get(_AQR_BASE + fname,
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
            except Exception:
                pass
        if not path.exists():
            continue
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
            hdr = _find_header_row(raw)
            if hdr is None:
                continue
            tbl = raw.iloc[hdr + 1:].copy()
            tbl.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
            tbl = tbl.rename(columns={tbl.columns[0]: "DATE"})
            tbl["DATE"] = pd.to_datetime(tbl["DATE"], errors="coerce")
            tbl = tbl.dropna(subset=["DATE"]).set_index("DATE")
            for col, region in _AQR_REGIONS.items():
                if col in tbl.columns:
                    s = pd.to_numeric(tbl[col], errors="coerce").dropna()
                    if len(s) > 24:
                        out[f"{region}:{label}"] = s
        except Exception:
            continue
    return out


def load_all() -> dict[str, pd.Series]:
    """All available academic factor series (French + AQR)."""
    out = load_french_factors()
    out.update(load_aqr_factors())
    return out

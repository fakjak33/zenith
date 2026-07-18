"""OSAP factor family: Open Source Asset Pricing (Chen-Zimmermann) predictors.

~212 published long-short predictor returns ('op' = original-paper
implementations, port == LS) plus the SignalDoc — descriptions, the exact
"Detailed Definition" calculation, and the sign — for every signal. The
dataset updates roughly ONCE A YEAR (current release: data through Dec 2024),
so signals here form on the latest published month and are stamped with it;
Gupta & Kelly find factor momentum persists at 1-60 month formation horizons,
so a months-old signal still carries documented information. Per the user's
survivorship constraint the recreated-signal layer (OSAP_CHAR_MAP -> Russell
1000 screens) produces holdings/screens only; the backtest shown for this
family uses the published returns, which include dead stocks.

OSAP models are excluded from the monthly pick tracker — their evaluation
would sit pending until the next annual release.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .. import save
from ...cas import store_cas
from ...cas.backtest import factor_data

_DOC_CACHE_KEY = "osap_signaldoc"
_DOC_TTL_HOURS = 60 * 24.0

# SignalDoc columns we keep for the committed catalog
_DOC_FIELDS = {
    "Acronym": "signal", "LongDescription": "name", "Authors": "authors",
    "Year": "year", "Cat.Economic": "category", "Sign": "sign",
    "SampleStartYear": "sample_start", "SampleEndYear": "sample_end",
    "Return": "op_monthly_ret", "T-Stat": "op_tstat",
    "Detailed Definition": "definition",
}


def build_panel(_px=None) -> tuple[pd.DataFrame, dict]:
    """Published LS monthly return panel (annual vintage)."""
    from . import aqr_published
    panel, meta = aqr_published.build_osap_panel()
    meta["annual"] = True
    meta["max_lag"] = None          # accept each factor's own last month
    return panel, meta


def load_doc(max_age_hours: float = _DOC_TTL_HOURS) -> dict[str, dict]:
    """SignalDoc rows keyed by acronym. Cached; degrades to {} offline."""
    cached = store_cas.cache_get(_DOC_CACHE_KEY, max_age_hours)
    if cached:
        return cached
    df = None
    csv = factor_data.CAS_CACHE_DIR / "SignalDoc.csv"
    if csv.exists():
        try:
            df = pd.read_csv(csv)
        except Exception:
            df = None
    if df is None:
        try:
            import openassetpricing as oap
            df = oap.OpenAP().dl_signal_doc("pandas")
        except Exception:
            return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        rec = {}
        for src, dst in _DOC_FIELDS.items():
            v = row.get(src)
            rec[dst] = None if pd.isna(v) else (
                int(v) if dst in ("year", "sample_start", "sample_end")
                else float(v) if dst in ("sign", "op_monthly_ret", "op_tstat")
                else str(v).strip())
        if rec.get("signal"):
            out[rec["signal"]] = rec
    if out:
        store_cas.cache_put(_DOC_CACHE_KEY, out)
    return out


def refresh_catalog(panel_columns: list[str]) -> dict:
    """Write data/fmom/osap_catalog.json for the signals actually present in
    the returns panel (definitions, sign, provenance for the UI)."""
    doc = load_doc()
    signals = {}
    for sig in panel_columns:
        rec = doc.get(sig) or {"signal": sig, "name": sig, "definition": None,
                               "sign": None, "category": None}
        signals[sig] = rec
    obj = {"as_of": date.today().isoformat(),
           "source": ("Open Source Asset Pricing (Chen & Zimmermann), "
                      "openassetpricing.com — original-paper LS ports + "
                      "SignalDoc"),
           "n": len(signals), "n_documented": sum(1 for s in signals.values()
                                                  if s.get("definition")),
           "signals": signals}
    save("osap_catalog", obj)
    return obj

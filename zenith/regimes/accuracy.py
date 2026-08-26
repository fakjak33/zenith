"""REGIMES forecast-accuracy tracking (spec section 32): calibrate this
engine's own historical calls against NBER's recession dating (USREC) — an
EXTERNAL label this engine cannot influence, exactly the "genuine
out-of-sample" anchor spec section 42 asks for.

Two honest limitations stated up front, not buried:

  1. The "recession signal" used here is DECLARED regime == "Deflation /
     Slowdown" — a reasonable proxy (falling growth AND falling inflation
     is a recessionary-leaning read) but NOT identical to NBER's own
     multi-indicator recession definition. Lead/lag and false-positive/
     negative counts should be read as "does this engine's own regime call
     track recessions", not as a claim of NBER-equivalence.
  2. The Brier score uses transition.py's OWN empirical base-rate table as
     the "forecast probability" — which means it is evaluated IN-SAMPLE
     (the same historical months that generated the table are the months
     it's scored against). This is explicitly labelled "in-sample" in
     every place it's shown. A genuine held-out/walk-forward evaluation is
     future work, not something this module pretends to already do.
"""

from __future__ import annotations

import pandas as pd

from ..cas.sources import fred as cas_fred

USREC_SERIES_ID = "USREC"
RECESSION_REGIME = "Deflation / Slowdown"
LOOKAROUND_MONTHS = 12
BRIER_HORIZON_MONTHS = 6


def fetch_usrec(sleep: float = 0.2) -> pd.Series:
    """NBER US recession indicator (1 = recession month), monthly, back to
    1854 on the free fredgraph.csv endpoint. Its own cache key so it never
    collides with the macro registry's fetch cadence — USREC is dated with
    a real lag (NBER announces recessions retrospectively) but that lag is
    irrelevant here: this module evaluates HISTORICAL calibration, where the
    label being "known late" in real time is precisely the point of the
    lead/lag statistic, not a look-ahead problem to correct for."""
    raw, status = cas_fred.get_series([USREC_SERIES_ID], max_age_hours=120.0, sleep=sleep,
                                      limit=None, cache_key="regimes_usrec")
    pts = raw.get(USREC_SERIES_ID, [])
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series([p["value"] for p in pts], index=pd.to_datetime([p["date"] for p in pts]))
    return s.sort_index()


def align_monthly(usrec: pd.Series, ends: pd.DatetimeIndex) -> pd.Series:
    if usrec.empty:
        return pd.Series(index=ends, dtype=float)
    return usrec.reindex(usrec.index.union(ends)).ffill().reindex(ends)


def lead_lag(declared_regime: pd.Series, usrec_monthly: pd.Series) -> dict:
    """For every NBER recession START month, the nearest month our own
    recession_signal first turned on within +/-LOOKAROUND_MONTHS. Positive
    lead = this engine flagged it before NBER's own start month."""
    signal = (declared_regime == RECESSION_REGIME).astype(int)
    usrec_starts = usrec_monthly[(usrec_monthly == 1) & (usrec_monthly.shift(1).fillna(0) == 0)].index

    leads, false_negatives = [], 0
    for start in usrec_starts:
        pos = usrec_monthly.index.get_loc(start)
        window_start = max(0, pos - LOOKAROUND_MONTHS)
        window_end = min(len(signal) - 1, pos + LOOKAROUND_MONTHS)
        window = signal.iloc[window_start:window_end + 1]
        on = window[window == 1]
        if on.empty:
            false_negatives += 1
            continue
        first_on = on.index[0]
        lead_months = int((start.to_period("M") - first_on.to_period("M")).n)
        leads.append({"nber_start": start.date().isoformat(), "our_first_signal": first_on.date().isoformat(),
                     "lead_months": lead_months})

    # false positives: our signal turned on but no NBER recession started within +/-LOOKAROUND_MONTHS
    our_starts = signal[(signal == 1) & (signal.shift(1).fillna(0) == 0)].index
    false_positives = 0
    for start in our_starts:
        pos = usrec_monthly.index.get_loc(start) if start in usrec_monthly.index else None
        if pos is None:
            continue
        window_start, window_end = max(0, pos - LOOKAROUND_MONTHS), min(len(usrec_monthly) - 1, pos + LOOKAROUND_MONTHS)
        if not (usrec_monthly.iloc[window_start:window_end + 1] == 1).any():
            false_positives += 1

    avg_lead = round(sum(r["lead_months"] for r in leads) / len(leads), 1) if leads else None
    return {"n_nber_recessions": len(usrec_starts), "n_matched": len(leads),
           "n_false_negatives": false_negatives, "n_our_signal_episodes": len(our_starts),
           "n_false_positives": false_positives, "avg_lead_months": avg_lead, "matches": leads}


def brier_score(declared_regime: pd.Series, transition_tables: dict,
                horizon_months: int = BRIER_HORIZON_MONTHS) -> dict:
    """IN-SAMPLE Brier score: at each month t, p_hat = the empirical
    (unconditional) probability that regime(t)'s historical successors
    landed in RECESSION_REGIME `horizon_months` later; actual = whether that
    ACTUALLY happened this time. See module docstring for why this is
    explicitly in-sample, not held-out."""
    months_to_days = {1: 30, 3: 90, 6: 180, 12: 365}
    horizon_days = months_to_days.get(horizon_months)
    if horizon_days is None:
        return {"n": 0, "brier": None, "in_sample": True, "note": f"no table for {horizon_months}mo horizon"}
    uncond = transition_tables.get("unconditional", {})
    errs = []
    for i in range(len(declared_regime) - horizon_months):
        reg = declared_regime.iloc[i]
        if reg is None:
            continue
        cell = uncond.get(reg, {}).get(str(horizon_days), {})
        dest = cell.get("destinations", {}).get(RECESSION_REGIME, {}) if cell else {}
        p_hat = dest.get("p")
        if p_hat is None:
            continue
        actual_reg = declared_regime.iloc[i + horizon_months]
        actual = 1.0 if actual_reg == RECESSION_REGIME else 0.0
        errs.append((p_hat - actual) ** 2)
    if not errs:
        return {"n": 0, "brier": None, "in_sample": True}
    return {"n": len(errs), "brier": round(sum(errs) / len(errs), 4), "in_sample": True,
           "horizon_months": horizon_months,
           "note": "IN-SAMPLE: evaluated against the same historical months that generated the "
                   "probability table. A lower score is better (0=perfect, 0.25=no-skill coin-flip "
                   "baseline at p=0.5, higher=worse than that). Not a held-out/walk-forward test."}


def build(declared_regime: pd.Series, transition_tables: dict, sleep: float = 0.2) -> dict:
    usrec = fetch_usrec(sleep=sleep)
    usrec_monthly = align_monthly(usrec, declared_regime.index)
    if usrec_monthly.dropna().empty:
        return {"available": False, "reason": "USREC fetch failed or returned no data this run."}
    ll = lead_lag(declared_regime, usrec_monthly)
    brier = brier_score(declared_regime, transition_tables)
    return {"available": True, "lead_lag": ll, "brier": brier}

"""FMOM pick history: append-only rows, deferred evaluation, summaries.

One row per (formation month, model). A row is appended when the month's
signals are formed and evaluated on a later run once the NEXT month's factor
returns exist in that family's panel — AQR published data can lag, so rows
stay `evaluated: false` ("pending") rather than being filled with fabricated
numbers. Old rows are never rewritten except to fill `realized`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import DISCLAIMER, load, save

MODELS = ("etf_tsfm", "etf_csfm", "man_tsfm", "man_csfm", "aqr_tsfm", "aqr_csfm")


def family_of(model: str) -> str:
    return model.split("_", 1)[0]


def lens_of(model: str) -> str:
    return model.split("_", 1)[1]


def make_row(month: str, model: str, weights: pd.Series,
             backfilled: bool = False) -> dict:
    """A history row from one month's leg weights (core.leg_weights output)."""
    w = weights.dropna()
    longs = [{"factor": f, "w": round(float(v), 4)} for f, v in
             w[w > 0].sort_values(ascending=False).items()]
    shorts = [{"factor": f, "w": round(float(v), 4)} for f, v in
              w[w < 0].sort_values().items()]
    return {"month": month, "model": model, "backfilled": backfilled,
            "picks": {"long": longs, "short": shorts},
            "degenerate": not longs or not shorts,
            "evaluated": False, "realized": {}}


def append_rows(new_rows: list[dict]) -> int:
    """Append rows, skipping (month, model) pairs already present. Idempotent
    under cron retries. Returns the number actually appended."""
    hist = load("history", {"rows": []})
    rows = hist.get("rows", [])
    seen = {(r["month"], r["model"]) for r in rows}
    added = 0
    for r in new_rows:
        if (r["month"], r["model"]) not in seen:
            rows.append(r)
            seen.add((r["month"], r["model"]))
            added += 1
    rows.sort(key=lambda r: (r["month"], r["model"]))
    hist.update({"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
                 "rows": rows})
    save("history", hist)
    return added


def _next_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y + (m == 12):04d}-{(m % 12) + 1:02d}"


def evaluate_pending(panels: dict[str, pd.DataFrame]) -> int:
    """Fill `realized` on every unevaluated row whose next-month factor returns
    now exist in the family's panel. panels: family -> monthly panel."""
    hist = load("history", {"rows": []})
    rows = hist.get("rows", [])
    filled = 0
    monthly_by_family = {
        fam: {ts.strftime("%Y-%m"): row for ts, row in panel.iterrows()}
        for fam, panel in panels.items() if panel is not None and not panel.empty}
    for r in rows:
        if r.get("evaluated"):
            continue
        fam = family_of(r["model"])
        realized = monthly_by_family.get(fam, {}).get(_next_month(r["month"]))
        if realized is None:
            continue
        per, model_ret, long_ret, short_ret = {}, 0.0, 0.0, 0.0
        ok = False
        for leg, sign in (("long", 1.0), ("short", 1.0)):
            for p in r["picks"][leg]:
                v = realized.get(p["factor"])
                if v is None or pd.isna(v):
                    continue
                ok = True
                per[p["factor"]] = round(float(v), 4)
                model_ret += p["w"] * float(v)
                if leg == "long":
                    long_ret += p["w"] * float(v)
                else:
                    short_ret += abs(p["w"]) * float(v)   # return OF the shorted side
        if not ok:
            continue
        r["realized"] = {"model_ret": round(model_ret, 4),
                         "long_ret": round(long_ret, 4),
                         "short_ret": round(short_ret, 4),
                         "hit": model_ret > 0, "per_factor": per,
                         "eval_date": date.today().isoformat()}
        r["evaluated"] = True
        filled += 1
    if filled:
        hist["as_of"] = date.today().isoformat()
        save("history", hist)
    return filled


def summarize(rows: list[dict]) -> dict:
    """Per-model tracker stats + equity-curve points for charting."""
    out: dict[str, dict] = {}
    curve: list[dict] = []
    for model in MODELS:
        mine = sorted((r for r in rows if r["model"] == model and r.get("evaluated")),
                      key=lambda r: r["month"])
        if not mine:
            continue
        rets = [r["realized"]["model_ret"] for r in mine]
        cum = 1.0
        for r in mine:
            cum *= 1.0 + r["realized"]["model_ret"]
            curve.append({"month": r["month"], "model": model,
                          "ret": r["realized"]["model_ret"],
                          "cum": round(cum - 1.0, 4),
                          "backfilled": r.get("backfilled", False)})
        n = len(rets)
        out[model] = {
            "n_months": n,
            "n_live": sum(1 for r in mine if not r.get("backfilled")),
            "hit_rate": round(sum(1 for x in rets if x > 0) / n, 4),
            "avg_ret": round(sum(rets) / n, 4),
            "cum_ret": round(cum - 1.0, 4),
            "last_month": mine[-1]["month"],
        }
    pending = sorted({r["month"] for r in rows if not r.get("evaluated")})
    return {"models": out, "curve": curve, "pending_months": pending}

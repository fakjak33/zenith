"""FMOM compute orchestrator.

  python -m zenith.fmom.compute --action auto|backfill|replicate

auto (the monthly cron path, days 1-4) is idempotent and self-healing: it
rebuilds the latest signals for every available family, evaluates any pending
history rows (a row for month M fills once M+1 factor returns exist), appends
the new formation month's rows exactly once, and refreshes the backtest.
backfill simulates every historical month from the same panels (published/ETF
returns only — no survivorship-biased self-computed history) so the tracker
has depth from day one. replicate is the Phase-3 Russell 1000 holdings build.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from . import DISCLAIMER, save, load, archive_month
from . import core, history
from .catalog import ETF_PROXIES
from .families import aqr_published, etf_proxy, man_styles, osap
from ..cas.sources import prices

# families excluded from the monthly pick tracker (annual data — evaluation
# would sit pending for up to a year)
UNTRACKED_FAMILIES = {"osap"}


def _build_panels(status: list[dict]) -> dict[str, tuple[pd.DataFrame, dict]]:
    out: dict[str, tuple[pd.DataFrame, dict]] = {}
    # one combined pull so SPY and shared tickers download once
    wanted = sorted(set(etf_proxy.tickers()) | set(man_styles.tickers()))
    px, st = prices.get_history(wanted, period="max")
    status.append({"segment": "prices", "ok": st.get("ok", False),
                   "n": st.get("n", 0), "requested": len(wanted),
                   "error": st.get("error", "")})
    for family, builder in (("etf", etf_proxy.build_panel),
                            ("man", man_styles.build_panel),
                            ("aqr", aqr_published.build_panel),
                            ("osap", osap.build_panel)):
        panel, meta = builder(px)
        status.append({"segment": f"panel({family})", "ok": meta.get("ok", False),
                       "n": meta.get("n_factors", 0),
                       "error": meta.get("error", ""),
                       "missing": meta.get("missing", [])})
        out[family] = (panel, meta)
    if out["osap"][1].get("ok"):
        try:
            osap.refresh_catalog(list(out["osap"][0].columns))
            status.append({"segment": "osap_catalog", "ok": True,
                           "n": out["osap"][0].shape[1]})
        except Exception as e:
            status.append({"segment": "osap_catalog", "ok": False,
                           "error": str(e)[:200]})
    return out


_PROXY = {p["factor"]: p["ticker"] for p in ETF_PROXIES}


def _model_block(panel: pd.DataFrame, family: str, lens: str,
                 max_lag: int | None = 3) -> dict | None:
    """Signal rows for the latest formation month of one (family, lens).

    Formation uses each factor's OWN most recent published month (within
    ``max_lag`` months of the panel end) — sources like AQR's workbooks lag
    Ken French by a month or two, and dropping them would silently shrink the
    cross-section. ``max_lag=None`` accepts any age (OSAP's annual vintage).
    Each row carries its ``source_month``."""
    if panel is None or panel.empty or len(panel) < 2:
        return None
    t = panel.index[-1]
    vol_all = core.ann_vol(panel)
    r1, vols, src = {}, {}, {}
    for f in panel.columns:
        col = panel[f].dropna()
        if col.empty:
            continue
        if max_lag is not None and col.index[-1] < t - pd.offsets.MonthEnd(max_lag):
            continue
        m = col.index[-1]
        v = vol_all.at[m, f]
        if pd.isna(v) or v == 0:
            continue
        r1[f], vols[f], src[f] = float(col.iloc[-1]), float(v), m
    if len(r1) < 3:
        return None
    r1 = pd.Series(r1)
    form = r1 - r1.mean() if lens == "csfm" else r1
    s = (form / pd.Series(vols)).clip(-core.CLIP, core.CLIP).dropna()
    w = core.leg_weights(s)
    deciles = core.decile_flags(s)
    order = s.sort_values(ascending=False)
    rows = []
    for rank, (factor, sv) in enumerate(order.items(), start=1):
        rows.append({
            "factor": factor,
            "proxy": _PROXY.get(factor) if family == "etf" else None,
            "r_1m": round(r1[factor], 4),
            "r_1m_dm": round(float(r1[factor] - r1.mean()), 4),
            "vol_ann": round(vols[factor], 4),
            "s": round(float(sv), 4),
            "weight": round(float(w.get(factor, 0.0)), 4),
            "leg": "long" if w.get(factor, 0.0) > 0 else "short",
            "decile": deciles.get(factor, ""),
            "rank": rank,
            "source_month": src[factor].strftime("%Y-%m"),
        })
    n_long = sum(1 for r in rows if r["weight"] > 0)
    n_short = sum(1 for r in rows if r["weight"] < 0)
    return {"family": family, "lens": lens,
            "formation_month": t.strftime("%Y-%m"),
            "rows": rows, "n_factors": len(rows),
            "n_long": n_long, "n_short": n_short,
            "degenerate": n_long == 0 or n_short == 0}


def _backtest_block(panel: pd.DataFrame, meta: dict) -> dict | None:
    """Backtest + per-factor persistence stats for one family panel."""
    if panel is None or panel.empty:
        return None
    from ..cas.backtest.factor_momentum import ar1
    bt = core.backtest(panel)
    series = [{"d": ts.strftime("%Y-%m"),
               **{c: (round(float(v), 5) if pd.notna(v) else None)
                  for c, v in row.items()}}
              for ts, row in bt["series"].iterrows()]
    per_factor = {}
    for f in panel.columns:
        b, tstat = ar1(panel[f])
        per_factor[f] = {
            "ar1": None if pd.isna(b) else round(float(b), 4),
            "t_stat": None if pd.isna(tstat) else round(float(tstat), 3),
            "n_months": int(panel[f].dropna().shape[0]),
        }
    recent = [{"d": ts.strftime("%Y-%m"), "factor": f,
               "ret": round(float(v), 4)}
              for ts, row in panel.tail(24).iterrows()
              for f, v in row.items() if pd.notna(v)]
    return {"series": series, "stats": bt["stats"], "per_factor": per_factor,
            "recent_panel": recent,
            "benchmark_note": ("'ew' = equal-weight all factors, the paper's "
                               "untimed benchmark"),
            "source_note": meta.get("benchmark", ""),
            "start": series[0]["d"] if series else None,
            "end": series[-1]["d"] if series else None}


def _rebuild_outputs(panels: dict[str, tuple[pd.DataFrame, dict]],
                     status: list[dict]) -> tuple[dict, dict]:
    """signals_latest.json + backtest.json from fresh panels."""
    models: dict[str, dict] = {}
    backtests: dict[str, dict] = {}
    for family, (panel, meta) in panels.items():
        max_lag = meta.get("max_lag", 3)
        for lens in ("tsfm", "csfm"):
            block = _model_block(panel, family, lens, max_lag=max_lag)
            if block:
                if meta.get("annual"):
                    block["annual"] = True
                models[f"{family}_{lens}"] = block
        bt = _backtest_block(panel, meta)
        if bt:
            backtests[family] = bt
    sources = _sources_stamp(panels)
    signals = {"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
               "models": models, "sources": sources}
    backtest = {"as_of": date.today().isoformat(), "disclaimer": DISCLAIMER,
                "families": backtests}
    save("signals", signals)
    save("backtest", backtest)
    status.append({"segment": "signals", "ok": bool(models), "n": len(models)})
    return signals, backtest


def _sources_stamp(panels: dict[str, tuple[pd.DataFrame, dict]]) -> dict:
    out = {}
    for family, (panel, meta) in panels.items():
        rec = {"last_month": meta.get("last_month"),
               "n_factors": meta.get("n_factors"),
               "missing": meta.get("missing", [])}
        if meta.get("members"):
            rec["members"] = meta["members"]
        if meta.get("coverage"):
            rec["coverage"] = meta["coverage"]
        if meta.get("annual"):
            rec["annual"] = True
        out[family] = rec
    return out


def run_auto() -> dict:
    today = date.today()
    status: list[dict] = []
    panels = _build_panels(status)
    signals, _ = _rebuild_outputs(panels, status)

    # evaluate pending picks now that a new month of returns may exist
    filled = history.evaluate_pending({f: p for f, (p, _m) in panels.items()})
    status.append({"segment": "evaluate", "ok": True, "n": filled})

    # append this formation month's rows (idempotent) + archive the snapshot
    new_rows = []
    for key, block in signals["models"].items():
        if block["family"] in UNTRACKED_FAMILIES:
            continue                       # annual data — no monthly tracking
        w = pd.Series({r["factor"]: r["weight"] for r in block["rows"]})
        new_rows.append(history.make_row(block["formation_month"], key, w,
                                         backfilled=False))
    added = history.append_rows(new_rows)
    status.append({"segment": "append", "ok": True, "n": added})
    months = {b["formation_month"] for b in signals["models"].values()}
    for m in months:
        archive_month(m, {"month": m, "as_of": today.isoformat(),
                          "models": {k: v for k, v in signals["models"].items()
                                     if v["formation_month"] == m}})

    out = {"date": today.isoformat(), "segments": status,
           "sources": {**signals["sources"],
                       "computed": today.isoformat()}}
    save("status", out)
    print(f"[fmom] {today} auto: models={len(signals['models'])} "
          f"evaluated={filled} appended={added}")
    for s in status:
        flag = "ok " if s.get("ok") else "ERR"
        print(f"  {flag} {s['segment']}: n={s.get('n', '')}"
              + (f"  ({s.get('error')})" if s.get("error") else ""))
    return out


def run_backfill() -> dict:
    """Simulate every historical formation month from the current panels and
    evaluate them immediately — instant tracker depth, survivorship-free
    (published/ETF return series include every month the funds traded)."""
    today = date.today()
    status: list[dict] = []
    panels = _build_panels(status)
    _rebuild_outputs(panels, status)

    new_rows: list[dict] = []
    for family, (panel, _meta) in panels.items():
        if panel is None or panel.empty or family in UNTRACKED_FAMILIES:
            continue
        for lens in ("tsfm", "csfm"):
            sig_all = (core.tsfm_signals(panel) if lens == "tsfm"
                       else core.csfm_signals(panel))
            for t in panel.index:
                s = sig_all.loc[t].dropna()
                if len(s) < 3:
                    continue
                w = core.leg_weights(s)
                new_rows.append(history.make_row(
                    t.strftime("%Y-%m"), f"{family}_{lens}", w, backfilled=True))
    added = history.append_rows(new_rows)
    filled = history.evaluate_pending({f: p for f, (p, _m) in panels.items()})
    status.append({"segment": "backfill", "ok": True, "n": added})
    print(f"[fmom] {today} backfill: rows={added} evaluated={filled}")
    return {"ok": True, "written": added, "evaluated": filled}


def run_replicate() -> dict:
    """Russell 1000 replication layer: current holdings per replicable factor.
    Heavy (~1000 yfinance info calls, cached ~25 days) — its own action so a
    failure here never blocks the signal run."""
    from .families import replication
    from ..pretom import universe as uni

    today = date.today()
    universe, uni_status = uni.russell1000()
    if not universe:
        print("[fmom] replicate aborted: no universe")
        return {"ok": False, "error": "no universe"}
    tickers = [u["ticker"] for u in universe]
    px, st = prices.get_history(tickers + ["SPY"], period="2y")
    print(f"[fmom] replicate: universe={len(universe)} "
          f"({uni_status.get('source')}), prices={st.get('n')}")
    fund = replication.get_fundamentals(tickers)
    stmt = replication.get_statement_chars(tickers)
    chars = replication.compute_chars(universe, px, fund, stmt)
    holdings = replication.build_holdings(universe, px, fund, chars)
    save("holdings", holdings)
    screens = replication.build_screens(universe, chars)
    screens["etf_holdings"] = replication.get_etf_top_holdings(
        [p["ticker"] for p in ETF_PROXIES])
    save("screens", screens)
    print(f"[fmom] replicate: holdings for {holdings['n_factors']} factors, "
          f"{screens['n_screens']} screens ({today})")
    return {"ok": True, "n_factors": holdings["n_factors"],
            "n_screens": screens["n_screens"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="auto",
                    choices=["auto", "backfill", "replicate"])
    args = ap.parse_args()
    if args.action == "backfill":
        run_backfill()
    elif args.action == "replicate":
        run_replicate()
    else:
        run_auto()


if __name__ == "__main__":
    main()

"""The auto-generated "Read": plain-language strengths, weaknesses and next
drills, from deterministic rules over stats.py -- no LLM.

Rules (every claim carries its n and 95% CI; nothing is said about n < 30):
  * Overall edge vs the base rate and vs the trend rule.
  * Cells (timeframe x horizon, class, regime tags, indicator set, side)
    whose base-rate edge is significant (|z| >= 1.96) become strengths or
    weaknesses, strongest first.
  * Conviction calibration (is High > Medium > Low?) and Brier vs a coin.
  * Stops: stopped-then-reversed rate; the hindsight-best ATR stop.
  * Next drill = the weakest well-sampled cell; with too little data, the
    drill is simply "play more" in the least-sampled area.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone

import numpy as np

from ..config import THEME
from . import stats as S

Z = S.Z95

DIMENSIONS = [  # (columns, label template) -- most specific first wins ties
    (["asset_class", "timeframe", "horizon"], "{timeframe} / {horizon}-candle calls in {asset_class}"),
    (["timeframe", "horizon"], "{timeframe} / {horizon}-candle calls"),
    (["asset_class"], "{asset_class} calls"),
    (["asset_class", "timeframe"], "{timeframe} calls in {asset_class}"),
    (["rg_trend"], "charts {rg_trend}"),
    (["rg_vol"], "{rg_vol} charts"),
    (["rg_rsi"], "charts at {rg_rsi}"),
    (["rg_dist_high"], "charts {rg_dist_high} from the 52-week high"),
    (["indicator_set"], "calls with indicators: {indicator_set}"),
]


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}pp"


def _ci(lo: float, hi: float) -> str:
    return f"95% CI {lo * 100:+.1f} to {hi * 100:+.1f}pp"


def build(d, resets=None) -> dict:
    """{"summary": [...], "strengths": [...], "weaknesses": [...], "notes": [...], "drills": [...]}
    -- each item {"text", "n"}."""
    out = {"summary": [], "strengths": [], "weaknesses": [], "notes": [], "drills": []}
    n = len(d)
    if n == 0:
        out["notes"].append({"text": "No calls yet. Play some charts and the Read writes itself.", "n": 0})
        return out
    h = S.headline(d)
    b = h["base"]
    out["summary"].append({"n": n, "text": (
        f"{n} calls, hit rate {h['hit']:.1%} (95% CI {h['hit_lo']:.1%}–{h['hit_hi']:.1%}). "
        f"The always-long base rate on these same charts is {b['base']:.1%}, so your edge is "
        f"{_pp(b['edge'])} ({_ci(b['lo'], b['hi'])}).")})
    if n < S.MIN_N:
        out["notes"].append({"n": n, "text": f"Only {n} calls — fewer than {S.MIN_N}. Every number here is "
                             "noise-dominated; no strengths or weaknesses are claimed yet."})
    else:
        if b["z"] >= Z:
            out["strengths"].append({"n": n, "text": f"Overall you beat the base rate by {_pp(b['edge'])} "
                                     f"(n={n}, {_ci(b['lo'], b['hi'])} — excludes zero)."})
        elif b["z"] <= -Z:
            out["weaknesses"].append({"n": n, "text": f"Overall you trail the base rate by {_pp(b['edge'])} "
                                      f"(n={n}, {_ci(b['lo'], b['hi'])}). Simply buying would have hit more often."})
        else:
            out["notes"].append({"n": n, "text": "Overall edge vs the base rate is not distinguishable from zero "
                                 f"yet ({_ci(b['lo'], b['hi'])})."})
        r = h["rule"]
        if np.isfinite(r["z"]):
            verdict = ("beat" if r["z"] >= Z else "trail" if r["z"] <= -Z else "are level with")
            bucket = "strengths" if r["z"] >= Z else "weaknesses" if r["z"] <= -Z else "notes"
            out[bucket].append({"n": r["n"], "text": (
                f"You {verdict} the trend rule on the same charts: {r['mean'] * 100:+.2f}% per call "
                f"(n={r['n']}, 95% CI {r['lo'] * 100:+.2f} to {r['hi'] * 100:+.2f}%).")})

    cells = []
    for cols, tmpl in DIMENSIONS:
        t = S.breakdown(d, cols)
        if t.empty:
            continue
        for _, row in t[~t["small_n"]].iterrows():
            if not np.isfinite(row["z"]) or abs(row["z"]) < Z:
                continue
            label = tmpl.format(**{c: row[c] for c in cols})
            cells.append((row["z"], label, row, cols))
    seen = set()
    # strongest first; when two lenses see the same calls, the more specific label wins
    for z, label, row, cols in sorted(cells, key=lambda x: (-round(abs(x[0]), 9), -len(x[3]))):
        sig = (int(row["n"]), round(float(row["edge"]), 6))   # same calls seen through two lenses
        if label in seen or sig in seen:
            continue
        seen.update({label, sig})
        txt = (f"Your {label} {'beat' if z > 0 else 'trail'} the base rate by {_pp(row['edge'])} "
               f"(n={int(row['n'])}, {_ci(row['edge_lo'], row['edge_hi'])} — excludes zero).")
        (out["strengths"] if z > 0 else out["weaknesses"]).append({"n": int(row["n"]), "text": txt})
    out["strengths"] = out["strengths"][:6]
    out["weaknesses"] = out["weaknesses"][:6]

    ls = S.long_short(d)
    if ls and n >= S.MIN_N and abs(ls["bias"]) >= 0.15:
        lean = "long" if ls["bias"] > 0 else "short"
        out["notes"].append({"n": n, "text": (
            f"You lean {lean}: {ls['long_share']:.0%} of calls are UP while the tape goes up "
            f"{ls['base']:.0%} of the time on these charts.")})

    cal = S.calibration(d)
    t = cal["table"][~cal["table"]["small_n"]] if not cal["table"].empty else cal["table"]
    detail = "; ".join(f"{r.conviction} {r.hit:.0%} (n={r.n}, 95% CI {r.hit_lo:.0%}–{r.hit_hi:.0%})"
                       for r in t.itertuples()) if len(t) else ""
    if cal["monotonic"] is True:
        out["strengths"].append({"n": n, "text": f"Your conviction is calibrated — hit rate rises with it: {detail}."})
    elif cal["monotonic"] is False:
        out["weaknesses"].append({"n": n, "text": f"Conviction is not calibrated — {detail}. Size up only "
                                  "when the read is genuinely clearer."})
    if np.isfinite(cal["brier"]) and n >= S.MIN_N:
        better = cal["brier"] < cal["brier_coin"]
        out["notes"].append({"n": n, "text": f"Brier score {cal['brier']:.3f} vs 0.250 for a coin "
                             f"({'better' if better else 'worse'} than guessing; lower is better)."})

    st = S.stops_analysis(d)
    if st.get("stopped_n", 0) >= S.MIN_N:
        out["notes"].append({"n": st["stopped_n"], "text": (
            f"{st['stopped_then_reversed']:.0%} of your stop-outs would have finished in your favour by the "
            f"horizon (n={st['stopped_n']}).")})
    hs = st.get("hindsight")
    if hs is not None and len(hs) and hs["n"].iloc[0] >= S.MIN_N:
        best = hs[hs["best"]].iloc[0]
        out["notes"].append({"n": int(best["n"]), "text": (
            f"Hindsight only: across your calls a {best['stop']} stop would have maximised average return "
            f"({best['exp_ret'] * 100:+.2f}% per call). Treat as a hypothesis to test, not a rule.")})

    out["drills"] = _drills(d)
    return out


def _drills(d) -> list[dict]:
    t = S.breakdown(d, ["asset_class", "timeframe", "horizon"])
    if t.empty:
        return []
    ok = t[~t["small_n"]]
    if len(ok):
        w = ok.sort_values("edge").iloc[0]
        return [{"n": int(w["n"]), "text": (
            f"Next drill: {w['asset_class']}, {w['timeframe']}, {int(w['horizon'])}-candle "
            f"(your weakest well-sampled area: {_pp(w['edge'])} vs base, n={int(w['n'])}).")}]
    lo = t.sort_values("n").iloc[0]
    need = S.MIN_N - int(t["n"].max())
    return [{"n": int(lo["n"]), "text": (
        f"Next drill: keep one setting fixed until a cell reaches {S.MIN_N} calls "
        f"({max(need, 0)} more in your busiest cell) — e.g. {t.iloc[0]['asset_class']}, "
        f"{t.iloc[0]['timeframe']}, {int(t.iloc[0]['horizon'])}-candle.")}]


# ------------------------------------------------------------ export ----
def to_html(rep: dict, player: str, filters: str = "") -> str:
    sec = [("Summary", "summary", THEME.text), ("Strengths", "strengths", THEME.teal),
           ("Weaknesses", "weaknesses", THEME.coral), ("Notes", "notes", THEME.mustard),
           ("Recommended drills", "drills", THEME.navy)]
    body = []
    for title, key, col in sec:
        items = rep.get(key) or []
        if not items:
            continue
        lis = "".join(f"<li>{html.escape(x['text'])}</li>" for x in items)
        body.append(f'<h2 style="color:{col}">{title}</h2><ul>{lis}</ul>')
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>EPHEMERIS Read — {html.escape(player)}</title>
<style>body{{background:{THEME.bg};color:{THEME.text};font-family:'Space Mono','Courier New',monospace;
max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.5}}h1{{font-family:'VT323',monospace;letter-spacing:.14em}}
h2{{font-size:1rem;letter-spacing:.14em;text-transform:uppercase;border-bottom:1px solid #333}}
li{{margin:.35rem 0}} .m{{color:{THEME.muted};font-size:.8rem}}
@media print{{body{{background:#fff;color:#000}}}}</style></head><body>
<h1>EPHEMERIS · THE READ</h1><div class="m">{html.escape(player)} · {when}{(' · ' + html.escape(filters)) if filters else ''}</div>
{''.join(body)}
<p class="m">Deterministic rules over your logged calls. Claims require n ≥ {S.MIN_N} and a 95% interval that
excludes zero. Base rate = share of UP outcomes for the chart's class × timeframe × horizon. Print this page to PDF
from your browser.</p></body></html>"""

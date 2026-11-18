"""The verdict card: every guard rail and test for one strategy, graded.

Each check is PASS / WARN / FAIL, or NOT RUN when its test hasn't been run for
this exact strategy (spec hash). The overall tier uses ZENITH's evidence scale:
A = robust on everything that matters, B = promising with gaps, C = not yet
evidence of an edge. Permutation and out-of-sample are *critical*: failing
either caps the grade at C, and the grade can't reach A until both have run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS, WARN, FAIL, NONE = "PASS", "WARN", "FAIL", "NOT RUN"
SCORE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
CRITICAL = {"Permutation test", "Out-of-sample"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _band(v, good, ok, higher=True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NONE
    if not higher:
        v, good, ok = -v, -good, -ok
    return PASS if v >= good else WARN if v >= ok else FAIL


def build(facts: dict) -> list[Check]:
    """``facts`` holds whatever has been computed; missing keys → NOT RUN."""
    f = facts
    out = [
        Check("Hypothesis written", PASS if f.get("hypothesis") else WARN,
              "economic reason recorded before testing" if f.get("hypothesis") else "no written reason for the edge"),
        Check("Enough evidence", PASS if f["trades"] >= 30 and f["years"] >= 5 else WARN if f["trades"] >= 15 else FAIL,
              f"{f['trades']} trades over {f['years']:.1f} in-sample years"),
        Check("Deflated Sharpe", _band(f.get("dsr"), 0.9, 0.5),
              f"{f['dsr'] * 100:.0f}% probability the Sharpe is real after {f['trials']} trials"),
        Check("Survives 2× costs", _band(f.get("sharpe_2x"), 0.3, 0.0),
              f"Sharpe {f['sharpe_2x']:.2f} at double costs"),
    ]
    a = f.get("attr")
    out.append(Check("Beats its benchmark", NONE if a is None else _band(a["alpha_t"], 2.0, 1.0),
                     "open ATTRIBUTION" if a is None else
                     f"alpha {a['alpha_ann'] * 100:+.1f}%/yr, t = {a['alpha_t']:.1f} vs {a['bench']}"))
    out.append(Check("Not just beta", NONE if a is None else
                     (PASS if np.isfinite(a["beta_share"]) and a["beta_share"] < 0.5 else
                      WARN if np.isfinite(a["beta_share"]) and a["beta_share"] < 0.8 else FAIL),
                     "open ATTRIBUTION" if a is None else
                     f"beta {a['beta']:.2f} explains {a['beta_share'] * 100:.0f}% of the average return"
                     if np.isfinite(a["beta_share"]) else "average return ≈ 0"))
    s = f.get("sensitivity")
    out.append(Check("Parameter plateau", NONE if s is None else _band(s["robustness"], 0.75, 0.5),
                     "run SENSITIVITY" if s is None else
                     f"robustness {s['robustness']:.2f}, {s['share_pos'] * 100:.0f}% of the grid has Sharpe > 0"))
    p = f.get("perm_p")
    out.append(Check("Permutation test", NONE if p is None else _band(p, 0.05, 0.10, higher=False),
                     "run ROBUSTNESS › PERMUTATION" if p is None else f"p = {p:.3f}"))
    k = f.get("noise_kept")
    out.append(Check("Noise tolerance", NONE if k is None else _band(k, 0.6, 0.3),
                     "run ROBUSTNESS › NOISE" if k is None else f"{k * 100:.0f}% of Sharpe kept at k = 0.25"))
    w = f.get("wf_eff")
    out.append(Check("Walk-forward", NONE if w is None else _band(w, 0.5, 0.25),
                     "run ROBUSTNESS › WALK-FORWARD" if w is None else f"efficiency {w * 100:.0f}%"))
    fl = f.get("filters")
    if fl is None:
        out.append(Check("Filters earn their place", PASS if not f.get("n_filters") else NONE,
                         "no filters used" if not f.get("n_filters") else "run FILTERS"))
    else:
        bad = [v for v in fl if v != "adds value"]
        out.append(Check("Filters earn their place", PASS if not bad else FAIL if "hurts" in bad else WARN,
                         "every filter adds value" if not bad else f"{len(bad)} filter set(s) without clear value"))
    o = f.get("oos")
    if o is None:
        out.append(Check("Out-of-sample", NONE, "reveal once in OUT-OF-SAMPLE when tuning is done"))
    elif o.get("contaminated"):
        out.append(Check("Out-of-sample", FAIL, "strategy changed after the reveal — no clean test left"))
    else:
        out.append(Check("Out-of-sample", _band(o["kept"], 0.5, 0.0),
                         f"OOS Sharpe {o['oos_sharpe']:.2f} vs IS {o['is_sharpe']:.2f}"))
    return out


def grade(checks: list[Check]) -> tuple[str, str]:
    run = [c for c in checks if c.status != NONE]
    if not run:
        return "C", "nothing tested yet"
    score = sum(SCORE[c.status] for c in run) / len(run)
    crit = {c.name: c.status for c in checks if c.name in CRITICAL}
    missing = sum(c.status == NONE for c in checks)
    if missing > 3:
        return "C", f"only {len(run)} of {len(checks)} checks run — not enough testing to call it an edge yet"
    if FAIL in crit.values():
        return "C", f"failed a critical test ({', '.join(k for k, v in crit.items() if v == FAIL)})"
    if score >= 0.85 and NONE not in crit.values() and missing <= 1:
        return "A", f"{score * 100:.0f}% of checks passed, including permutation and out-of-sample"
    if score >= 0.6:
        tail = f"; {missing} check(s) not run" if missing else ""
        return "B", f"{score * 100:.0f}% of the checks run passed{tail}"
    return "C", f"only {score * 100:.0f}% of the checks run passed"

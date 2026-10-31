"""Per-underlying checklist, overall score, and the VRP state machine.

STATE MACHINE — encodes the practitioner rule the user brought: "the slope of
VRP is the alarm, VRP is the confirmation, the direction of the slope is the
all clear". Inputs each day: v = VRP30, s = OLS slope of VRP30 over the last
`slope_days` observations, s_z = s / std(daily VRP changes, `z_window`).

  CONFIRMED   v <= -buffer: realized has caught up with implied (no premium)
  ALARM       v > 0 but s_z <= alarm_z: the premium is collapsing fast
  RECOVERING  from ALARM / CONFIRMED, the slope has turned up for
              `recover_days` straight days
  ALL-CLEAR   v > 0 and the slope has been >= 0 for `clear_days` straight days
  (otherwise the previous state holds — hysteresis, so one noisy print
   cannot flip it; NEUTRAL until the first decisive reading)

This rule is NOT a published result; it is shown as a heuristic with every
threshold exposed in config.VOLSCREEN_STATE.

CHECKLIST — each rule returns pass / warn / fail with the value and a reason;
the score is the weight-sum (pass = 1, warn = 0.5, fail = 0) on a 0-100 scale
(config.VOLSCREEN_CHECK_WEIGHTS). SELL mode rewards rich, stable premium;
BUY mode rewards cheap implied vol against rising realized vol.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import VOLSCREEN_CHECK_WEIGHTS, VOLSCREEN_STATE
from .vrp import ols_slope

STATES = ("ALL-CLEAR", "RECOVERING", "NEUTRAL", "ALARM", "CONFIRMED")
PASS, WARN, FAIL = "pass", "warn", "fail"
CREDIT = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
TARGET_DTE = 45                       # the tenor a seller is screening for


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def vrp_states(vrp, cfg: dict = VOLSCREEN_STATE) -> list[str]:
    """State for every observation of a VRP30 series (oldest first)."""
    v = np.asarray([x if _ok(x) else np.nan for x in vrp], float)
    n, zw = cfg["slope_days"], cfg["z_window"]
    out, state = [], "NEUTRAL"
    pos_run = nonneg_run = 0
    for t in range(len(v)):
        x = v[t]
        if not np.isfinite(x):
            out.append(state)
            continue
        s = ols_slope(v[max(0, t - n + 1): t + 1])
        d = np.diff(v[max(0, t - zw): t + 1])
        d = d[np.isfinite(d)]
        sd = float(np.std(d, ddof=1)) if len(d) >= 10 else float("nan")
        sz = s / sd if np.isfinite(s) and np.isfinite(sd) and sd > 0 else float("nan")
        if np.isfinite(s):
            pos_run = pos_run + 1 if s > 0 else 0
            nonneg_run = nonneg_run + 1 if s >= 0 else 0
        if x <= -cfg["confirm_buffer"]:
            state = "CONFIRMED"
        elif np.isfinite(sz) and sz <= cfg["alarm_z"]:
            state = "ALARM"
        elif state in ("ALARM", "CONFIRMED"):
            if pos_run >= cfg["recover_days"]:
                state = "RECOVERING"
        elif state in ("RECOVERING", "NEUTRAL"):
            if nonneg_run >= cfg["clear_days"]:
                state = "ALL-CLEAR"
        out.append(state)
    return out


def days_in_state(states: list[str]) -> int:
    if not states:
        return 0
    last, k = states[-1], 0
    for s in reversed(states):
        if s != last:
            break
        k += 1
    return k


def _item(key, label, status, value, why) -> dict:
    return {"key": key, "label": label, "status": status, "value": value, "why": why}


def _pct(x) -> str:
    return f"{x * 100:.1f}" if _ok(x) else "—"


def rules_sell(r: dict) -> list[dict]:
    items = []
    st = r.get("stars") or 1
    items.append(_item("liquidity", "Liquid, tight market", PASS if st >= 3 else WARN if st == 2 else FAIL,
                       f"{st}★", "3-4★ chains cost little to enter and adjust; 1★ spreads eat the premium."))
    v, ivrv = r.get("vrp30"), r.get("iv_rv")
    s = PASS if _ok(v) and v > 0 and _ok(ivrv) and ivrv >= 1.1 else WARN if _ok(v) and v > 0 else FAIL
    items.append(_item("vrp", "VRP positive (IV over forecast RV)", s,
                       f"{_pct(v)} pts · IV/RV {ivrv:.2f}" if _ok(ivrv) else f"{_pct(v)} pts",
                       "Ex-earnings 30D IV minus the GARCH forecast of 30D realized vol."))
    state = r.get("state", "NEUTRAL")
    s = {"ALL-CLEAR": PASS, "RECOVERING": PASS, "NEUTRAL": WARN, "ALARM": WARN, "CONFIRMED": FAIL}[state]
    items.append(_item("state", "VRP slope: all clear", s, state,
                       "Slope is the alarm, VRP the confirmation, slope direction the all-clear."))
    ivr, ivp = r.get("ivr"), r.get("ivp")
    if not _ok(ivr):
        s, why = WARN, "No year of IV history yet for this name."
    else:
        s = PASS if ivr >= 0.30 or (_ok(ivp) and ivp >= 0.50) else WARN if ivr >= 0.15 else FAIL
        why = "Premium is rich against this name's own past year."
    items.append(_item("ivr", "IV rank / percentile elevated", s, f"IVR {_pct(ivr)} · IVP {_pct(ivp)}", why))
    ratio, explained = r.get("ratio_30_90"), r.get("kink_explained")
    if not _ok(ratio):
        s, why = WARN, "Term structure unavailable."
    elif ratio <= 1.0:
        s, why = PASS, "Contango: front cheaper than back — the calm regime short premium earns in."
    elif explained:
        s, why = WARN, "Inverted, but the earnings event explains the kink."
    else:
        s = FAIL if ratio > 1.10 else WARN
        why = "Backwardation without an event: the market is pricing near-term stress."
    items.append(_item("term", "Term structure favourable", s, f"30/90 {ratio:.2f}" if _ok(ratio) else "—", why))
    a, b = r.get("yz10"), r.get("yz60")
    s = WARN if not (_ok(a) and _ok(b)) else PASS if a <= b else WARN if a <= 1.2 * b else FAIL
    items.append(_item("rv_trend", "Realized vol settling", s,
                       f"RV10 {_pct(a)} / RV60 {_pct(b)}", "Short-window realized at or below long-window."))
    d = r.get("earn_days")
    s = PASS if not _ok(d) or d < 0 or d > TARGET_DTE else FAIL if d <= 7 else WARN
    items.append(_item("earnings", f"No earnings inside {TARGET_DTE} DTE", s,
                       f"in {int(d)}d" if _ok(d) and d >= 0 else "none",
                       "An earnings gap is a separate, priced event risk."))
    sp = r.get("skew_pct")
    s = WARN if not _ok(sp) else PASS if sp >= 0.5 else WARN if sp >= 0.25 else FAIL
    items.append(_item("skew", "Put skew rich", s, f"pct {_pct(sp)}",
                       "Steep 25Δ put-minus-call skew pays more for selling the downside."))
    v30, v60 = r.get("vrp_t", {}).get("30"), r.get("vrp_t", {}).get("60")
    k = sum(1 for x in (v30, v60) if _ok(x) and x > 0)
    items.append(_item("garch", "GARCH-VRP positive at 30 & 60D", PASS if k == 2 else WARN if k == 1 else FAIL,
                       f"{_pct(v30)} / {_pct(v60)}", "IV above the GARCH term forecast at both tenors."))
    return items


def rules_buy(r: dict) -> list[dict]:
    items = []
    st = r.get("stars") or 1
    items.append(_item("liquidity", "Liquid, tight market", PASS if st >= 3 else WARN if st == 2 else FAIL,
                       f"{st}★", "Long premium pays the spread twice; it has to be tight."))
    v = r.get("vrp30")
    s = PASS if _ok(v) and v <= 0 else WARN if _ok(v) and v < 0.01 else FAIL
    items.append(_item("vrp", "IV at or below forecast RV", s, f"{_pct(v)} pts",
                       "Options are cheap only when implied sits below expected realized."))
    state = r.get("state", "NEUTRAL")
    s = {"CONFIRMED": PASS, "ALARM": WARN, "NEUTRAL": WARN, "RECOVERING": FAIL, "ALL-CLEAR": FAIL}[state]
    items.append(_item("state", "VRP state favours owning vol", s, state,
                       "CONFIRMED = realized has overtaken implied."))
    ivr = r.get("ivr")
    s = WARN if not _ok(ivr) else PASS if ivr <= 0.20 else WARN if ivr <= 0.35 else FAIL
    items.append(_item("ivr", "IV rank low", s, f"IVR {_pct(ivr)}", "Buy premium when it is cheap vs its year."))
    ratio = r.get("ratio_30_90")
    s = WARN if not _ok(ratio) else PASS if ratio <= 0.95 else WARN if ratio <= 1.02 else FAIL
    items.append(_item("term", "Front month cheap vs back", s, f"30/90 {ratio:.2f}" if _ok(ratio) else "—",
                       "A steep contango makes near-dated long vol (and calendars) cheap."))
    a, b = r.get("yz10"), r.get("yz60")
    s = WARN if not (_ok(a) and _ok(b)) else PASS if a > 1.1 * b else WARN if a > b else FAIL
    items.append(_item("rv_trend", "Realized vol rising", s, f"RV10 {_pct(a)} / RV60 {_pct(b)}",
                       "Rising realized vol is what a long-vol position needs."))
    d = r.get("earn_days")
    s = PASS if not _ok(d) or d < 0 or d > TARGET_DTE else WARN
    items.append(_item("earnings", "No earnings pump in the price", s,
                       f"in {int(d)}d" if _ok(d) and d >= 0 else "none",
                       "Pre-earnings IV is inflated and crushes after the report."))
    sp = r.get("skew_pct")
    s = WARN if not _ok(sp) else PASS if sp <= 0.5 else WARN if sp <= 0.75 else FAIL
    items.append(_item("skew", "Downside protection not overpriced", s, f"pct {_pct(sp)}",
                       "Flat skew = cheaper puts."))
    return items


def score(items: list[dict], mode: str) -> tuple[float, str]:
    w = VOLSCREEN_CHECK_WEIGHTS[mode]
    tot = sum(w.get(i["key"], 0) for i in items)
    got = sum(w.get(i["key"], 0) * CREDIT[i["status"]] for i in items)
    sc = 100.0 * got / tot if tot else 0.0
    grade = "A" if sc >= 80 else "B" if sc >= 65 else "C" if sc >= 50 else "D" if sc >= 35 else "F"
    return round(sc, 1), grade


VARIABLE_WHY = {"term", "ivr"}            # the only rules whose explanation depends on the data
RULES = {"sell": rules_sell, "buy": rules_buy}


def text(mode: str) -> dict[str, tuple[str, str]]:
    """{key: (label, default reason)} — written once, not into every row."""
    return {i["key"]: (i["label"], i["why"]) for i in RULES[mode]({})}


def evaluate(r: dict, compact: bool = True) -> dict:
    out = {}
    for mode, fn in RULES.items():
        items = fn(r)
        sc, gr = score(items, mode)
        if compact:
            items = [{"key": i["key"], "status": i["status"], "value": i["value"],
                      **({"why": i["why"]} if i["key"] in VARIABLE_WHY else {})} for i in items]
        out[mode] = {"score": sc, "grade": gr, "items": items}
    return out

"""REGIMES alerts (spec section 26) — purely computed from the existing
journal history (history.py's sharded daily record), diffed against the
reading from ~30 days ago. No new storage: the journal already carries
regime/confidence/momentum_score/transitioning per day, extended here to
also carry the Regime Change Score so alerts can track it.
"""

from __future__ import annotations

from datetime import date, timedelta

CONFIDENCE_THRESHOLDS = (30.0, 70.0)


def _nearest_row(rows: list[dict], target: date) -> dict | None:
    if not rows:
        return None
    best, best_diff = None, None
    for r in rows:
        try:
            d = date.fromisoformat(r["date"])
        except Exception:
            continue
        diff = abs((d - target).days)
        if best_diff is None or diff < best_diff:
            best, best_diff = r, diff
    return best


def evaluate(journal_rows: list[dict], current: dict, change_score: dict,
            lookback_days: int = 30) -> list[dict]:
    """journal_rows: this year's + (if needed) prior year's rows, ascending
    by date (history.load_journal()'s output). Returns a list of triggered
    alerts, each {id, severity, title, description}."""
    alerts: list[dict] = []
    today = date.today()
    past = _nearest_row(journal_rows, today - timedelta(days=lookback_days))
    if past is None:
        return alerts

    regime_now, regime_then = current.get("regime"), past.get("regime")
    if regime_now and regime_then and regime_now != regime_then:
        alerts.append({"id": "regime_changed", "severity": "high",
                       "title": f"Regime changed: {regime_then} → {regime_now}",
                       "description": f"The declared regime has changed over the trailing "
                                      f"{lookback_days} days.",
                       "what_changed": f"{regime_then} → {regime_now}",
                       "watch_next": "Confirm via the Growth & Inflation tab which axis breadth "
                                    "actually flipped."})

    conf_now, conf_then = current.get("confidence"), past.get("confidence")
    if conf_now is not None and conf_then is not None:
        for thresh in CONFIDENCE_THRESHOLDS:
            crossed_up = conf_then < thresh <= conf_now
            crossed_down = conf_then >= thresh > conf_now
            if crossed_up or crossed_down:
                direction = "above" if crossed_up else "below"
                alerts.append({"id": f"confidence_crossed_{int(thresh)}", "severity": "medium",
                               "title": f"Confidence crossed {direction} {thresh:.0f}",
                               "description": f"Regime confidence moved from {conf_then:.0f} to "
                                              f"{conf_now:.0f} over {lookback_days} days.",
                               "what_changed": f"{conf_then:.0f} → {conf_now:.0f}",
                               "watch_next": "Check whether breadth is genuinely broadening or "
                                            "concentrated in a few indicators."})

    trans_now, trans_then = current.get("transitioning"), past.get("transitioning")
    if trans_now and not trans_then:
        alerts.append({"id": "transition_underway", "severity": "high",
                       "title": f"Transition underway toward {current.get('raw_regime')}",
                       "description": "The raw (unpersisted) regime signal has newly diverged "
                                      "from the declared regime.",
                       "what_changed": f"raw regime now reads {current.get('raw_regime')}",
                       "watch_next": f"If this persists {2} more monthly readings it will become "
                                    f"the new declared regime."})

    score_now = change_score.get("score")
    band_now = change_score.get("band")
    score_then = past.get("change_score")
    if score_now is not None and score_then is not None and score_now - score_then >= 20:
        alerts.append({"id": "change_score_jump", "severity": "medium",
                       "title": f"Regime Change Score rose sharply: {score_then:.0f} → {score_now:.0f}",
                       "description": f"Now in the '{band_now}' band.",
                       "what_changed": f"{score_then:.0f} → {score_now:.0f}",
                       "watch_next": "Check the What Is Changing panel for which indicators drove it."})

    mtm_now, mtm_then = current.get("momentum", {}).get("score"), past.get("momentum_score")
    if mtm_now is not None and mtm_then is not None and (mtm_now >= 0) != (mtm_then >= 0):
        alerts.append({"id": "momentum_sign_flip", "severity": "low",
                       "title": f"Regime momentum flipped sign: {mtm_then:+.0f} → {mtm_now:+.0f}",
                       "description": "The forces defining the current regime have reversed direction.",
                       "what_changed": f"{mtm_then:+.0f} → {mtm_now:+.0f}",
                       "watch_next": "Watch for the raw regime call to follow if this persists."})

    return alerts

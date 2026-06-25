"""Contingency planning segment — proactive/reactive playbooks.

Each scenario pairs a trigger (evaluated against the live regime summary and
signals) with pre-planned proactive and reactive actions/hedges. Stored as
editable JSON; the compute step stamps each scenario's current trigger status so
the UI can highlight what's active now.
"""

from __future__ import annotations

from . import store_cas

DEFAULT_SCENARIOS = [
    {
        "id": "vol_spike",
        "name": "Volatility spike / risk-off",
        "trigger": "regime risk-off or VIX percentile > 0.85",
        "proactive": "Trim high-beta & crowded-long themes; raise cash; pre-buy "
                     "downside hedges while cheap (SPY/QQQ puts, VIX calls).",
        "reactive": "Add to quality/min-vol (USMV/QUAL); rotate toward defensives "
                    "(XLP/XLU); avoid catching falling knives in lev-long COT extremes.",
    },
    {
        "id": "rate_shock",
        "name": "Rate shock / curve move",
        "trigger": "10y yield percentile > 0.85 or sharp 10y-2y change",
        "proactive": "Reduce long-duration (TLT) and rate-sensitive (XLU/XLRE) "
                     "exposure; favor financials (XLF/KRE) on steepening.",
        "reactive": "Reassess growth/tech (QQQ/IGV) sensitivity; hedge duration.",
    },
    {
        "id": "crowding_unwind",
        "name": "Crowded-trade unwind",
        "trigger": "leveraged/managed-money COT at positive extreme with weakening "
                   "price trend",
        "proactive": "Lighten the crowded theme into strength; set trailing stops.",
        "reactive": "Expect sharp mean-reversion; size down; wait for positioning "
                    "to reset before re-entering.",
    },
    {
        "id": "theme_reversal",
        "name": "Leading-theme reversal",
        "trigger": "top relative-strength theme rolls below its 50d MA",
        "proactive": "Take partial profits on extended leaders (e.g. semis/gold).",
        "reactive": "Rotate into the next emerging relative-strength theme.",
    },
    {
        "id": "liquidity_event",
        "name": "Liquidity / event window",
        "trigger": "triple-witching / quarter-end / index rebalance within 5 days",
        "proactive": "Expect flow-driven dislocation; reduce new risk into the print.",
        "reactive": "Fade exaggerated rebalance moves once the window passes.",
    },
]


def load() -> list[dict]:
    sc = store_cas.load("contingency", [])
    if not sc:
        sc = [dict(s, active=False, status="") for s in DEFAULT_SCENARIOS]
        store_cas.save("contingency", sc)
    return sc


def evaluate(regime: dict, fred: dict, overlap_ranked: list[dict]) -> list[dict]:
    """Stamp each scenario with current active/status based on live state."""
    scenarios = load()
    risk = regime.get("label", "")
    vix_pct = regime.get("vix_percentile", 0.5)

    def _pct(series_id):
        s = fred.get(series_id, [])
        if len(s) < 20:
            return 0.5
        vals = [p["value"] for p in s][-252:]
        return sum(1 for v in vals if v <= vals[-1]) / len(vals)

    dgs10_pct = _pct("DGS10")

    for s in scenarios:
        active, status = False, ""
        if s["id"] == "vol_spike":
            active = (risk == "risk-off") or vix_pct > 0.85
            status = f"regime={risk}, VIX pct={vix_pct:.0%}"
        elif s["id"] == "rate_shock":
            active = dgs10_pct > 0.85
            status = f"10y yield pct={dgs10_pct:.0%}"
        elif s["id"] == "liquidity_event":
            from .sources import calendar as cal
            soon = [e for e in cal.upcoming() if e.get("days_until", 99) <= 5]
            active = bool(soon)
            status = (f"{soon[0]['event']} in {soon[0]['days_until']}d" if soon
                      else "no event within 5d")
        elif s["id"] == "crowding_unwind":
            crowded = [r for r in overlap_ranked
                       if r["sell_count"] >= 2 and "flows" in r.get("segments", [])]
            active = bool(crowded)
            status = (f"{crowded[0]['asset']} flashing crowded-sell" if crowded
                      else "no crowded extreme detected")
        elif s["id"] == "theme_reversal":
            status = "monitor leading themes vs 50d MA"
        s["active"], s["status"] = active, status
    store_cas.save("contingency", scenarios)
    return scenarios

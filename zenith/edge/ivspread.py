"""IV-spread screen (Cremers & Weinbaum 2010) — pure math.

Per stock: IV spread = open-interest-weighted mean of (call IV - put IV) across
matched strikes near the money on the nearest standard expiry. Positive =
relatively expensive calls (bullish option positioning); negative = expensive
puts (bearish / often hard-to-borrow, per Muravyev-Pearson-Pollet 2025).

The `options.iv_spread_rows()` fetcher produces the raw per-ticker measure; this
module ranks it and flags the borrow-fee caveat cross-section.
"""

from __future__ import annotations

from .common import assemble

HORIZON_TD = 5          # Cremers-Weinbaum predictability is ~1 week


def build(rows: list[dict], hard_to_borrow: set[str] | None = None) -> dict:
    """rows: [{ticker, name, sector, iv_spread, n_pairs, spot}]. Returns the
    ranked screen. LONG = expensive calls (top), SHORT = expensive puts."""
    htb = hard_to_borrow or set()
    a = assemble([dict(r) for r in rows], "iv_spread", higher_is_long=True)
    for r in a["ranked"]:
        r["iv_spread_bp"] = round(r["iv_spread"] * 1e4, 1)
        # Muravyev flag: a wide negative spread that is ALSO high-short-interest
        # is most likely a borrow-fee artifact, not a clean directional short.
        r["borrow_flag"] = bool(r.get("side") == "short" and r["ticker"] in htb)
    return {"screen": "ivspread", "horizon_td": HORIZON_TD, **a}

"""ETF MOMENTUM universe assembly + the ETF taxonomy that replaces GICS.

The universe itself is NOT rebuilt here. `mom/mvt/universe.py::etf_universe()`
already assembles the ~935-ticker union and applies three layers of
leveraged/inverse gating (an explicit reviewed ticker list, a name regex, and
a self-healing yfinance metadata cache); this module consumes that and adds
what a ranked momentum table needs on top:

  * `refresh_meta=False`, ALWAYS. mom.yml refreshes data/mom/mvt/etf_meta.json
    nightly and commits it. If this package refreshed it too, two workflows
    would write the same committed file and race on push. ETF MOMENTUM is a
    consumer of that cache, never a second writer. Do not "fix" this by
    flipping the flag.
  * AUM and expense ratio from the committed Morningstar catalog (574/574 and
    573/574 populated) — the closest ETF analogue to MOMENTUM's market-cap
    column.
  * A NORMALIZED category and a mechanical asset-class rollup (below).

Near-duplicate funds are deliberately NOT dropped here — see the package
docstring. The one empirical gate this feature does apply lives in
compute.py, which filters `duplicate_and_leverage_gate`'s output down to its
leverage verdicts only.
"""

from __future__ import annotations

from ..mom.mvt import universe as mvt_universe

# Morningstar's own vocabulary reaches this repo through two different doors
# with two different spellings: the committed strategic-beta catalog says
# "US Fund Large Blend" while the yfinance .info cache says "Large Blend".
# Left alone that splits one bucket in two -- measured live, stripping this
# prefix collapses 117 distinct category strings to 77.
_CATEGORY_PREFIX = "US Fund "

# Ordered asset-class rollup. Deliberately NOT cas.universe.asset_class_of(),
# which returns sector/theme/factor/macro/equity -- a ROLE taxonomy, and
# unknown for 546 of the 912 included names anyway.
#
# These rules are mechanical over Morningstar's own category vocabulary; no
# new taxonomy is invented. Order matters, and the two subtle cases are
# called out because they are easy to get backwards:
#   * "Digital Assets" is matched EXACTLY. "Equity Digital Assets" is a fund
#     holding crypto-adjacent EQUITIES, not crypto, and must fall through to
#     Equity.
#   * "Equity Precious Metals", "Equity Energy", "Natural Resources",
#     "Infrastructure" and "Energy Limited Partnership" are equity SECTOR
#     funds, not commodity exposure -- they must stay Equity. Only the
#     "Commodities *" categories are genuine commodity exposure.
_FIXED_INCOME_TOKENS = ("Bond", "Muni", "Government", "Bank Loan", "Preferred Stock")
_ALTERNATIVE_EXACT = frozenset({"Equity Market Neutral", "Derivative Income",
                                "Trading--Miscellaneous"})

ASSET_CLASSES = ("Equity", "Fixed Income", "Commodity", "Real Estate",
                 "Allocation", "Currency", "Digital Assets", "Alternative", "Unknown")


def normalize_category(category: str | None) -> str:
    """Morningstar category with the source-specific 'US Fund ' prefix removed."""
    c = (category or "").strip()
    return c[len(_CATEGORY_PREFIX):].strip() if c.startswith(_CATEGORY_PREFIX) else c


def asset_class_of(category: str | None) -> str:
    """Coarse asset class from a NORMALIZED Morningstar category string.
    Pass the output of normalize_category(), not a raw catalog value."""
    c = (category or "").strip()
    if not c:
        return "Unknown"
    if "Allocation" in c:
        return "Allocation"
    if c == "Single Currency":
        return "Currency"
    if c in _ALTERNATIVE_EXACT:
        return "Alternative"
    if c == "Digital Assets":          # exact -- "Equity Digital Assets" is equity
        return "Digital Assets"
    if any(tok in c for tok in _FIXED_INCOME_TOKENS):
        return "Fixed Income"
    if "Commodities" in c:             # not "Equity Precious Metals"/"Natural Resources"
        return "Commodity"
    if "Real Estate" in c:
        return "Real Estate"
    return "Equity"


def constituents() -> tuple[list[dict], dict]:
    """The ETF universe as (rows, status). Every raw ticker survives into
    `rows` with an `included` flag and, when excluded, a stated reason --
    nothing is silently dropped, exactly as mom._build_rows does for the
    Russell 1000."""
    raw_rows, status = mvt_universe.etf_universe(refresh_meta=False)   # see module docstring
    catalog = mvt_universe.etf_catalog()

    rows = []
    for r in raw_rows:
        cat = normalize_category(r.get("morningstar_category"))
        cat_row = catalog.get(r["ticker"], {})
        rows.append({
            "ticker": r["ticker"],
            "name": r.get("name") or r["ticker"],
            "category": cat,
            "asset_class": asset_class_of(cat),
            "region": r.get("region") or "",
            "label": r.get("label") or "",
            "groups": r.get("morningstar_groups") or [],
            "aum_m": cat_row.get("aum_m"),
            "er": cat_row.get("er"),
            "included": r.get("included", False),
            "exclusion_reason": r.get("exclusion_reason"),
        })

    included = [r for r in rows if r["included"]]
    by_class: dict[str, int] = {}
    for r in included:
        by_class[r["asset_class"]] = by_class.get(r["asset_class"], 0) + 1
    status = {
        **status,
        "n_included": len(included),
        "n_categories": len({r["category"] for r in included if r["category"]}),
        "by_asset_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "n_without_category": sum(1 for r in included if not r["category"]),
    }
    return rows, status

"""Fund registry — one entry per tracked fund.

Adding a second fund (KMLM, CTA, other systematic strategies) is a `Fund`
entry here plus one adapter module under `sources/`. Nothing else in the
package, the Action or the view is fund-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Fund:
    key: str                     # slug used for the data dir and UI keys
    ticker: str
    name: str
    adviser: str
    subadviser: str
    adapter: str                 # module name under zenith.holdings.sources
    source_url: str              # the fund's own daily holdings page
    source_label: str            # short provenance line for the UI
    strategy: str                # what the strategy does, in one paragraph
    reads_as: str                # how to read this fund's published file
    caveats: tuple[str, ...] = field(default_factory=tuple)
    legacy_urls: tuple[str, ...] = field(default_factory=tuple)  # for Wayback backfill
    enabled: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


DBMF = Fund(
    key="dbmf",
    ticker="DBMF",
    name="iMGP DBi Managed Futures Strategy ETF",
    adviser="iM Global Partner",
    subadviser="Dynamic Beta investments",
    adapter="imgp",
    source_url=("https://www.imgp.com/us/fund/"
                "US53700T8273-imgp-dbi-managed-futures-strategy-etf/"),
    source_label="iM Global Partner — fund page, daily holdings (SEC Rule 6c-11)",
    strategy=(
        "DBi's Dynamic Beta Engine regresses the trailing 60-day returns of the "
        "largest managed-futures hedge funds against a basket of liquid futures, "
        "then holds the best-fit portfolio. It replicates the target group's "
        "performance — explicitly not its positions — at an 0.85% flat fee."
    ),
    reads_as=(
        "Every futures row's market value is signed NOTIONAL exposure and its "
        "weight is that notional as a fraction of NAV. A weight of -0.96 means "
        "the fund is short 96% of NAV in notional terms. Treasury bills are "
        "margin collateral, not a directional view."
    ),
    caveats=(
        "These are DBMF's own positions, which estimate — but are not — the "
        "positions of the CTA funds it tracks.",
        "Notional exposure is not risk-equivalent across asset classes: a large "
        "2-year note position carries far less risk than a smaller 10-year one.",
        "Treasury bills are collateral for futures margin and are excluded from "
        "the long/short exposure figures.",
        "Futures roll quarterly. Positions are tracked by contract root, so a "
        "Sep-to-Dec roll is one continuous position, not a close plus an open.",
    ),
    legacy_urls=(
        "https://imgpfunds.com/im-dbi-managed-futures-strategy-etf/",
    ),
)

FUNDS: dict[str, Fund] = {f.key: f for f in (DBMF,)}

DEFAULT_FUND = "dbmf"


def get(key: str) -> Fund:
    if key not in FUNDS:
        raise KeyError(f"unknown fund {key!r}; known: {sorted(FUNDS)}")
    return FUNDS[key]


def enabled_funds() -> list[Fund]:
    return [f for f in FUNDS.values() if f.enabled]


def registry_snapshot() -> dict:
    """Committed so the app can render fund cards without importing adapters."""
    return {"funds": [f.as_dict() for f in FUNDS.values()],
            "default": DEFAULT_FUND}

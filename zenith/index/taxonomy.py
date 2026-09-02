"""INDEX taxonomy — the controlled vocabularies entries are classified against.

DESIGN RULE: this file is DATA, not logic. Every vocabulary is a plain dict of
``slug -> (label, aliases)``. Adding "Insurance-Linked Securities" as a strategy
is one dict entry and nothing else — no new branch, no view change, no
migration. The spec's own requirement was that the taxonomy be "flexible and
extensible, rather than hard-coded in a way that makes future expansion
difficult", and the test suite enforces it: ``test_taxonomy_extends_without_code``
adds a term at runtime and asserts resolution works.

Aliases exist because the same idea arrives spelled many ways — "quant",
"quantitative", "systematic quant" — and a directory whose filters only match
one spelling is a directory whose filters do not work. ``resolve()`` maps any
spelling to the canonical slug; unknown terms are returned normalised rather
than dropped, so a new tag is never silently lost.

THREE PRIMARY CATEGORIES (institutional / academic / tool) come straight from
the user's own source list, which was already divided under exactly those three
headings. Honouring that division rather than re-deriving one keeps the import
faithful to what was supplied.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Primary category — the top-level split, mirroring the seed list's own sections
# --------------------------------------------------------------------------
PRIMARY_CATEGORIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "institutional": ("Institutional", ("institution", "firm", "industry")),
    "academic": ("Academic / Research", ("academic", "research", "journal", "university")),
    "tool": ("Tool", ("tools", "platform", "software", "data")),
}

# --------------------------------------------------------------------------
# Entity type — WHAT a row is, orthogonal to which category it belongs to.
# (A podcast can be institutional; a person can be academic.)
# --------------------------------------------------------------------------
ENTITY_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "organisation": ("Organisation", ("org", "organization", "firm", "company")),
    "person": ("Person", ("people", "individual")),
    "podcast": ("Podcast", ("show", "youtube channel")),
    "academic_source": ("Academic source", ("journal", "repository", "working papers")),
    "tool": ("Tool", ("platform", "screener", "software")),
    "publication": ("Publication", ("newsletter", "blog", "report")),
}

# Sub-type refines entity_type where the distinction is genuinely useful for
# filtering (a hedge fund and a central bank are both organisations, but nobody
# searching for one wants the other).
ORG_SUBTYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "hedge_fund": ("Hedge fund", ("hedge fund manager",)),
    "asset_manager": ("Asset manager", ("asset management", "fund manager")),
    "private_markets": ("Private markets", ("private equity firm", "buyout firm")),
    "bank": ("Bank / broker-dealer", ("investment bank", "broker", "sell-side")),
    "trading_firm": ("Trading firm", ("prop shop", "market maker", "proprietary trading")),
    "cta": ("CTA / managed futures", ("managed futures firm", "commodity trading advisor")),
    "exchange": ("Exchange / venue", ("exchange", "clearing")),
    "data_provider": ("Data / analytics provider", ("data vendor", "index provider")),
    "research_house": ("Research house", ("research firm", "independent research")),
    "wealth_platform": ("Wealth / brokerage platform", ("brokerage", "wealth manager")),
    "official_sector": ("Official sector", ("central bank", "regulator", "supranational")),
    "think_tank": ("Think tank / institute", ("institute", "policy centre", "policy center")),
    "university": ("University", ("school", "faculty")),
    "media": ("Media", ("publisher", "press")),
}

# --------------------------------------------------------------------------
# Investment approach — the spec's "Investment Approach" secondary taxonomy
# --------------------------------------------------------------------------
INVESTMENT_APPROACH: dict[str, tuple[str, tuple[str, ...]]] = {
    "quantitative": ("Quantitative", ("quant", "quantitative research")),
    "systematic": ("Systematic", ("rules-based", "systematic trading")),
    "discretionary": ("Discretionary", ("judgemental", "judgmental")),
    "fundamental": ("Fundamental", ("bottom-up",)),
    "macro": ("Macro", ("macroeconomic",)),
    "global_macro": ("Global macro", ("global-macro",)),
    "long_short_equity": ("Long/short equity", ("l/s equity", "equity long short")),
    "market_neutral": ("Equity market neutral", ("market neutral", "emn")),
    "stat_arb": ("Statistical arbitrage", ("statarb", "stat arb", "arbitrage")),
    "trend_following": ("Trend following", ("trend", "momentum trading")),
    "managed_futures": ("Managed futures", ("cta", "commodity trading advisor")),
    "multi_strategy": ("Multi-strategy", ("multistrat", "multi strat", "platform")),
    "event_driven": ("Event driven", ("merger arb", "special situations")),
    "value": ("Value", ("deep value",)),
    "growth": ("Growth", ()),
    "quality": ("Quality", ()),
    "momentum": ("Momentum", ("cross-sectional momentum",)),
    "factor_investing": ("Factor investing", ("factors", "smart beta", "strategic beta")),
    "alternative_risk_premia": ("Alternative risk premia", ("arp", "risk premia")),
    "volatility": ("Volatility", ("vol", "vol trading")),
    "options": ("Options", ("option strategies",)),
    "derivatives": ("Derivatives", ("futures", "swaps")),
    "fixed_income": ("Fixed income", ("bonds", "rates")),
    "credit": ("Credit", ("high yield", "leveraged finance")),
    "commodities": ("Commodities", ("commodity",)),
    "crypto": ("Crypto", ("digital assets", "cryptocurrency")),
    "private_equity": ("Private equity", ("buyout", "pe")),
    "venture_capital": ("Venture capital", ("vc", "venture")),
    "real_estate": ("Real estate", ("property", "reits")),
    "infrastructure": ("Infrastructure", ("infra", "real assets")),
    "asset_allocation": ("Asset allocation", ("allocation", "strategic asset allocation")),
    "portfolio_construction": ("Portfolio construction", ("portfolio design",)),
    "risk_management": ("Risk management", ("risk",)),
    "behavioral_finance": ("Behavioural finance", ("behavioral finance", "behavioural")),
    "market_microstructure": ("Market microstructure", ("microstructure", "execution")),
    "market_making": ("Market making", ("liquidity provision",)),
    "replication": ("Replication", ("hedge fund replication", "index replication")),
    "passive": ("Passive / index", ("indexing", "index funds", "beta")),
    "evidence_based": ("Evidence-based investing", ("academic investing",)),
}

# --------------------------------------------------------------------------
# Asset class
# --------------------------------------------------------------------------
ASSET_CLASSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "equities": ("Equities", ("equity", "stocks", "shares")),
    "fixed_income": ("Fixed income", ("bonds", "rates", "treasuries")),
    "credit": ("Credit", ("corporate credit", "high yield")),
    "commodities": ("Commodities", ("commodity", "energy", "metals")),
    "fx": ("FX / currencies", ("currencies", "foreign exchange", "forex")),
    "crypto": ("Crypto", ("digital assets",)),
    "volatility": ("Volatility", ("vix", "vol surface")),
    "derivatives": ("Derivatives", ("futures", "options", "swaps")),
    "alternatives": ("Alternatives", ("alts", "alternative investments")),
    "private_markets": ("Private markets", ("private equity", "private credit")),
    "real_assets": ("Real assets", ("real estate", "infrastructure")),
    "multi_asset": ("Multi-asset", ("cross-asset", "balanced")),
}

# --------------------------------------------------------------------------
# Insight type — WHAT KIND OF KNOWLEDGE the source provides (spec section 3)
# --------------------------------------------------------------------------
INSIGHT_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "market_data": ("Market data", ("data", "prices", "quotes")),
    "research": ("Research", ("insights", "commentary")),
    "academic_research": ("Academic research", ("papers", "working papers", "journal articles")),
    "investment_ideas": ("Investment ideas", ("ideas", "stock picks")),
    "portfolio_construction": ("Portfolio construction", ("portfolio design",)),
    "asset_allocation": ("Asset allocation", ("allocation",)),
    "trading": ("Trading", ("execution", "trade ideas")),
    "quantitative_research": ("Quantitative research", ("quant research",)),
    "macroeconomics": ("Macroeconomics", ("macro", "economics")),
    "monetary_policy": ("Monetary policy", ("central banking", "fed")),
    "fiscal_policy": ("Fiscal policy", ("government finance",)),
    "options": ("Options", ("options analytics",)),
    "volatility": ("Volatility", ("vol research",)),
    "derivatives": ("Derivatives", ("futures research",)),
    "technical_analysis": ("Technical analysis", ("charting", "ta")),
    "fundamental_analysis": ("Fundamental analysis", ("company analysis", "valuation")),
    "alternative_investments": ("Alternative investments", ("alts research",)),
    "institutional_investing": ("Institutional investing", ("allocators", "endowments")),
    "manager_selection": ("Manager selection", ("due diligence", "manager research")),
    "risk_management": ("Risk management", ("risk analytics",)),
    "behavioral_finance": ("Behavioural finance", ("behavioral finance",)),
    "market_history": ("Market history", ("financial history",)),
    "financial_education": ("Financial education", ("education", "learning")),
    "interviews": ("Interviews", ("conversations",)),
    "podcasts": ("Podcasts", ("audio", "podcast")),
    "books": ("Books", ("book",)),
    "news": ("News", ("headlines",)),
    "screening": ("Screening", ("screener", "stock screener")),
    "backtesting": ("Backtesting", ("backtest", "simulation")),
    "portfolio_monitoring": ("Portfolio monitoring", ("tracking", "portfolio analytics")),
    "positioning": ("Positioning / flows", ("flows", "cot", "positioning data")),
    "index_data": ("Index & benchmark data", ("benchmarks", "indices", "indexes")),
    "regulatory": ("Regulatory / filings", ("filings", "disclosure")),
}

# Lifecycle states an entry moves through (spec section 16). `archived` exists
# so a defunct source is retired WITHOUT deleting the historical record.
LIFECYCLE_STATES: dict[str, str] = {
    "new": "New — imported, not yet reviewed",
    "verified": "Verified — link confirmed live and metadata checked",
    "needs_review": "Needs review — missing, ambiguous or unconfirmed information",
    "updated": "Updated — changed since it was last verified",
    "archived": "Archived — defunct or superseded, kept for the historical record",
}

CONFIDENCE_LEVELS: dict[str, str] = {
    "high": "Identity and official source both confirmed",
    "medium": "Identity confident; some metadata unconfirmed",
    "low": "Ambiguous entry — identity or official source not established",
}

# Relationship (edge) types for the knowledge graph. Phase 1 populates the
# first six; `appeared_on` / `guest_of` are reserved for Phase 2's podcast
# harvest and declared here so the vocabulary does not fork later.
RELATIONSHIP_TYPES: dict[str, str] = {
    "works_at": "Person works at organisation (current)",
    "worked_at": "Person previously worked at organisation",
    "founded": "Person founded organisation",
    "subsidiary_of": "Organisation is a division/brand of a parent organisation",
    "publishes": "Organisation or person publishes this research source or podcast",
    "related_to": "Related resource (topical, non-hierarchical)",
    "appeared_on": "Person appeared as a guest on a podcast (Phase 2)",
    "hosts": "Person hosts a podcast (Phase 2)",
}

# The registry every lookup goes through. Adding a whole NEW vocabulary is one
# entry here plus one dict above.
VOCABULARIES: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "primary_category": PRIMARY_CATEGORIES,
    "entity_type": ENTITY_TYPES,
    "org_subtype": ORG_SUBTYPES,
    "investment_approach": INVESTMENT_APPROACH,
    "asset_class": ASSET_CLASSES,
    "insight_type": INSIGHT_TYPES,
}

VOCABULARY_LABELS: dict[str, str] = {
    "primary_category": "Primary category",
    "entity_type": "Entity type",
    "org_subtype": "Organisation sub-type",
    "investment_approach": "Investment approach",
    "asset_class": "Asset class",
    "insight_type": "Type of insight",
}


def slugify(text: str) -> str:
    """Canonical slug: lowercase, non-alphanumerics collapsed to underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return s.strip("_")


def resolve(vocab: str, term: str) -> str | None:
    """Map any spelling of ``term`` to its canonical slug in ``vocab``.

    Returns the slugified term itself when unrecognised, rather than None, so a
    tag that is simply not in the vocabulary yet is preserved instead of
    silently dropped — the caller can then see it via ``unknown_terms()``.
    Returns None only for empty input.
    """
    if not term:
        return None
    table = VOCABULARIES.get(vocab)
    if table is None:
        return slugify(term)
    key = slugify(term)
    if key in table:
        return key
    raw = str(term).strip().lower()
    for slug, (label, aliases) in table.items():
        if raw == label.lower() or slugify(label) == key:
            return slug
        for alias in aliases:
            if raw == alias.lower() or slugify(alias) == key:
                return slug
    return key


def label_of(vocab: str, slug: str) -> str:
    """Human label for a slug, falling back to a title-cased version of the
    slug so an unrecognised term still displays readably in the UI."""
    table = VOCABULARIES.get(vocab) or {}
    if slug in table:
        return table[slug][0]
    return str(slug or "").replace("_", " ").strip().title()


def resolve_many(vocab: str, terms) -> list[str]:
    """Resolve a list of terms, de-duplicated, order preserved."""
    out: list[str] = []
    for t in terms or []:
        r = resolve(vocab, t)
        if r and r not in out:
            out.append(r)
    return out


def known(vocab: str) -> list[str]:
    return list((VOCABULARIES.get(vocab) or {}).keys())


def unknown_terms(vocab: str, terms) -> list[str]:
    """Terms that resolved to something NOT in the vocabulary — i.e. tags that
    are in use but undeclared. Surfaced in the Taxonomy view so the vocabulary
    can be grown deliberately instead of drifting."""
    table = VOCABULARIES.get(vocab) or {}
    return [t for t in resolve_many(vocab, terms) if t not in table]

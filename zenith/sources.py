"""Source registry for Zenith.

Each source: name, category (insight|research|news), kind (rss), url, enabled.
Feeds-first: we list the best free RSS/Atom feed we know of. Some firms publish
no machine-readable free feed (JS-only / LinkedIn / paywalled) — those are kept
here ``enabled=False`` with a note so the catalog is complete and easy to extend.
The scraper validates feeds at runtime and records per-source status, so a stale
URL simply yields zero items rather than breaking the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    category: str          # insight | research | news
    kind: str              # rss (only supported kind for now)
    url: str
    enabled: bool = True
    note: str = ""


SOURCES: list[Source] = [
    # ---------------- INSIGHTS (firms / banks / managers) ----------------
    Source("Apollo Academy", "insight", "rss", "https://www.apolloacademy.com/feed/"),
    Source("Alpha Architect", "insight", "rss", "https://alphaarchitect.com/feed/"),
    Source("Newfound (Flirting with Models)", "insight", "rss", "https://blog.thinknewfound.com/feed/"),
    Source("A Wealth of Common Sense", "insight", "rss", "https://awealthofcommonsense.com/feed/"),
    Source("The Irrelevant Investor", "insight", "rss", "https://theirrelevantinvestor.com/feed/"),
    Source("Klement on Investing", "insight", "rss", "https://klementoninvesting.substack.com/feed"),
    Source("Verdad Capital", "insight", "rss", "https://verdadcap.com/archive?format=rss"),
    Source("Mauldin Economics (Thoughts from the Frontline)", "insight", "rss",
           "https://www.mauldineconomics.com/feed"),
    Source("Top Traders Unplugged (podcast)", "insight", "rss", "https://toptradersunplugged.com/feed/"),
    Source("Hedgeye", "insight", "rss", "https://app.hedgeye.com/feed_items.rss",
           enabled=False, note="403 blocked"),
    Source("Morningstar", "insight", "rss", "https://www.morningstar.com/rss/news",
           enabled=False, note="no working public RSS (404)"),
    Source("Man Institute", "insight", "rss", "https://www.man.com/rss", enabled=False,
           note="no confirmed public RSS"),
    Source("AQR", "insight", "rss", "https://www.aqr.com/rss", enabled=False,
           note="no confirmed public RSS"),
    Source("Research Affiliates", "insight", "rss", "https://www.researchaffiliates.com/rss",
           enabled=False, note="no confirmed public RSS"),
    Source("BlackRock Investment Institute", "insight", "rss", "https://www.blackrock.com/rss",
           enabled=False, note="JS site, no confirmed RSS"),
    Source("State Street / SSGA", "insight", "rss", "https://www.ssga.com/rss",
           enabled=False, note="no confirmed RSS"),
    Source("Charles Schwab", "insight", "rss", "https://www.schwab.com/rss",
           enabled=False, note="no confirmed RSS"),
    Source("Citadel / Jane Street / Bridgewater / hedge funds", "insight", "rss", "",
           enabled=False, note="publish little/no free machine-readable feed"),

    # ---------------- RESEARCH (federal / academic / working papers) ------
    Source("NBER Working Papers", "research", "rss", "https://back.nber.org/rss/new.xml"),
    Source("BIS Working Papers", "research", "rss", "https://www.bis.org/doclist/wppubls.rss"),
    Source("Federal Reserve (FEDS Notes)", "research", "rss",
           "https://www.federalreserve.gov/feeds/feds_notes.xml"),
    Source("Federal Reserve (Working Papers)", "research", "rss",
           "https://www.federalreserve.gov/feeds/working_papers.xml"),
    Source("SF Fed Economic Letter", "research", "rss",
           "https://www.frbsf.org/economic-research/feed/"),
    Source("OFR", "research", "rss", "https://www.financialresearch.gov/feeds/", enabled=False,
           note="no confirmed RSS"),
    Source("CEPR", "research", "rss", "https://cepr.org/rss.xml", enabled=False,
           note="verify at runtime"),
    Source("SSRN FEN", "research", "rss", "https://papers.ssrn.com/sol3/Jeljour_results.cfm",
           enabled=False, note="SSRN RSS restricted"),
    # Journal table-of-contents (abstracts free; articles may be paywalled but link out)
    Source("Journal of Finance (TOC)", "research", "rss",
           "https://onlinelibrary.wiley.com/feed/15406261/most-recent"),
    Source("Journal of Financial Economics (TOC)", "research", "rss",
           "https://rss.sciencedirect.com/publication/science/0304405X"),
    Source("Review of Financial Studies (TOC)", "research", "rss",
           "https://academic.oup.com/rfs/rss", enabled=False, note="Oxford RSS URL unstable (404)"),
    Source("Financial Analysts Journal (TOC)", "research", "rss",
           "https://www.tandfonline.com/feed/rss/ufaj20"),
    Source("Quantitative Finance (TOC)", "research", "rss",
           "https://www.tandfonline.com/feed/rss/rquf20"),

    # ---------------- NEWS / TOOLS ----------------------------------------
    Source("Yahoo Finance", "news", "rss", "https://finance.yahoo.com/news/rssindex"),
    Source("Federal Reserve Press", "news", "rss",
           "https://www.federalreserve.gov/feeds/press_all.xml"),
    Source("CME Group", "news", "rss", "https://www.cmegroup.com/rss", enabled=False,
           note="no confirmed RSS"),
]


def enabled_sources() -> list[Source]:
    return [s for s in SOURCES if s.enabled and s.url]


def by_name(name: str) -> Source | None:
    return next((s for s in SOURCES if s.name == name), None)

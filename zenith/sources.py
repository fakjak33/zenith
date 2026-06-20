"""Source registry for Zenith.

Each source: name, category (insight|research|news), kind (rss), url, enabled.
Feeds-first: we list the best free RSS/Atom feed we know of. Some firms publish
no machine-readable free feed (JS-only / LinkedIn / Cloudflare-walled / paywalled)
— those are kept here ``enabled=False`` with a note so the catalog is complete and
easy to extend. The scraper validates feeds at runtime and records per-source
status, so a stale URL simply yields zero items rather than breaking the run.

Every enabled URL below was probed and confirmed to return feed entries. News is
intentionally minimized (see NEWS section): the focus is durable *insights* and
*research*, not the daily headline churn.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    category: str          # insight | research | news
    kind: str              # "rss" feed, or "html" hub page scraped for links
    url: str               # feed URL (rss) or insights/research index URL (html)
    enabled: bool = True
    note: str = ""
    # html-only: regex restricting which link paths count as articles, e.g.
    # r"/insights/|/research/". Empty = accept any same-domain headline link.
    link_pattern: str = ""


SOURCES: list[Source] = [
    # ================= INSIGHTS (firms / managers / analysts) =================
    # Apollo Academy carries Torsten Slok's "Daily Spark" in its main feed.
    Source("Apollo Academy (incl. Daily Spark)", "insight", "rss",
           "https://www.apolloacademy.com/feed/"),
    Source("Bespoke Investment Group", "insight", "rss", "https://www.bespokepremium.com/feed/"),
    Source("Alpha Architect", "insight", "rss", "https://alphaarchitect.com/feed/"),
    Source("Newfound (Flirting with Models)", "insight", "rss", "https://blog.thinknewfound.com/feed/"),
    Source("Verdad Capital", "insight", "rss", "https://verdadcap.com/archive?format=rss"),
    Source("Klement on Investing", "insight", "rss", "https://klementoninvesting.substack.com/feed"),
    Source("A Wealth of Common Sense", "insight", "rss", "https://awealthofcommonsense.com/feed/"),
    Source("The Irrelevant Investor", "insight", "rss", "https://theirrelevantinvestor.com/feed/",
           enabled=False, note="feed now returns 0 entries (blog dormant/moved)"),
    Source("The Big Picture (Ritholtz)", "insight", "rss", "https://ritholtz.com/feed/"),
    Source("Calculated Risk", "insight", "rss",
           "https://www.calculatedriskblog.com/feeds/posts/default"),
    Source("FT Alphaville", "insight", "rss", "https://www.ft.com/alphaville?format=rss"),
    Source("Musings on Markets (Damodaran)", "insight", "rss",
           "https://aswathdamodaran.blogspot.com/feeds/posts/default"),
    Source("Robert Carver (Systematic)", "insight", "rss",
           "https://qoppac.blogspot.com/feeds/posts/default"),
    Source("Philosophical Economics", "insight", "rss",
           "https://www.philosophicaleconomics.com/feed/"),
    Source("Pragmatic Capitalism (Discipline Funds)", "insight", "rss",
           "https://disciplinefunds.com/feed/"),
    Source("Marginal Revolution", "insight", "rss", "https://marginalrevolution.com/feed"),
    Source("Conversable Economist", "insight", "rss", "https://conversableeconomist.com/feed/"),
    Source("Econbrowser", "insight", "rss", "https://econbrowser.com/feed"),
    Source("Quantian's Newsletter", "insight", "rss", "https://quantian.substack.com/feed"),
    Source("Project Syndicate (Economics)", "insight", "rss", "https://www.project-syndicate.org/rss"),
    Source("Mauldin Economics (Frontline)", "insight", "rss", "https://www.mauldineconomics.com/feed",
           enabled=False, note="feed 404 (no current public RSS)"),
    Source("Top Traders Unplugged (podcast)", "insight", "rss", "https://toptradersunplugged.com/feed/"),
    # --- new verified RSS feeds (probed 2026-06) ---
    Source("Meb Faber / Cambria", "insight", "rss", "https://mebfaber.com/feed/"),
    Source("Simplify Asset Management", "insight", "rss", "https://www.simplify.us/blog/rss.xml"),
    # --- HTML-hub sources: no RSS, but their insights index scrapes cleanly via
    #     the direct (browser-UA) tier — confirmed returning article links 2026-06 ---
    Source("Blackstone (Insights)", "insight", "html", "https://www.blackstone.com/insights/",
           link_pattern=r"/insights/(article|insight)/"),
    Source("Goldman Sachs (Insights)", "insight", "html", "https://www.goldmansachs.com/insights",
           link_pattern=r"/insights/(articles|pages)/"),
    Source("AQR (Insights)", "insight", "html", "https://www.aqr.com/Insights",
           link_pattern=r"/Insights/"),
    Source("Man Institute", "insight", "html", "https://www.man.com/insights",
           link_pattern=r"/insights/"),
    Source("BlackRock Investment Institute", "insight", "html",
           "https://www.blackrock.com/us/individual/insights",
           link_pattern=r"/insights/"),
    # --- HTML-hub sources that the direct tier can't reach: fall back to Apify
    #     ONLY when APIFY_TOKEN is configured & under budget (else report blocked) ---
    Source("Morningstar (Insights)", "insight", "html", "https://www.morningstar.com/insights",
           link_pattern=r"/(articles|insights)/", enabled=False,
           note="anti-bot SPA; empty via direct + Apify datacenter proxy. Needs "
                "Apify residential proxy (APIFY_RESIDENTIAL=1) — disabled to protect the $5/mo tier"),
    Source("Research Affiliates", "insight", "html", "https://www.researchaffiliates.com/publications",
           link_pattern=r"/publications/", enabled=False,
           note="anti-bot SPA; empty via direct + Apify datacenter proxy. Needs "
                "Apify residential proxy (APIFY_RESIDENTIAL=1) — disabled to protect the $5/mo tier"),
    # --- registered but genuinely unreachable (kept for completeness) ---
    Source("Citadel", "insight", "html", "https://www.citadel.com/insights/", enabled=False,
           note="Cloudflare-walled (403 on all paths incl. browser UA); Apify residential proxy only"),
    Source("KKR (Insights)", "insight", "html", "https://www.kkr.com/insights", enabled=False,
           note="robots.txt disallows crawling /insights — respected, not scraped"),
    Source("Two Sigma (Insights)", "insight", "html", "https://www.twosigma.com/insights/",
           enabled=False, note="robots.txt disallows crawling /insights — respected"),
    Source("Jane Street", "insight", "rss", "", enabled=False,
           note="no free machine-readable feed (user: low insight value anyway)"),
    Source("Deutsche Bank (Chart of the Day)", "insight", "rss",
           "https://www.dbresearch.com/", enabled=False,
           note="DB Research requires login; no free public feed"),
    Source("Hedgeye", "insight", "rss", "https://app.hedgeye.com/feed_items.rss",
           enabled=False, note="403 blocked; subscriber-gated"),

    # ============ RESEARCH (federal / central banks / academic) ============
    Source("Liberty Street Economics (NY Fed)", "research", "rss",
           "https://libertystreeteconomics.newyorkfed.org/feed/"),
    Source("St. Louis Fed (On the Economy)", "research", "rss",
           "https://www.stlouisfed.org/on-the-economy/rss"),
    Source("NBER Working Papers", "research", "rss", "https://back.nber.org/rss/new.xml"),
    Source("BIS Working Papers", "research", "rss", "https://www.bis.org/doclist/wppubls.rss"),
    Source("ECB Working Papers", "research", "rss", "https://www.ecb.europa.eu/rss/wppub.html"),
    Source("Federal Reserve (FEDS Notes)", "research", "rss",
           "https://www.federalreserve.gov/feeds/feds_notes.xml"),
    Source("Federal Reserve (Working Papers)", "research", "rss",
           "https://www.federalreserve.gov/feeds/working_papers.xml"),
    Source("Federal Reserve (Speeches)", "research", "rss",
           "https://www.federalreserve.gov/feeds/speeches.xml"),
    Source("SF Fed Economic Letter", "research", "rss",
           "https://www.frbsf.org/economic-research/feed/", enabled=False,
           note="feed 404 (FRBSF dropped this RSS)"),
    Source("Bank Underground (Bank of England)", "research", "rss",
           "https://bankunderground.co.uk/feed/"),
    Source("arXiv q-fin (Quant Finance)", "research", "rss", "http://export.arxiv.org/rss/q-fin"),
    # Journal table-of-contents (abstracts free; full text links out)
    Source("Journal of Finance (TOC)", "research", "rss",
           "https://onlinelibrary.wiley.com/feed/15406261/most-recent"),
    Source("Journal of Financial Economics (TOC)", "research", "rss",
           "https://rss.sciencedirect.com/publication/science/0304405X"),
    Source("Financial Analysts Journal (TOC)", "research", "rss",
           "https://www.tandfonline.com/feed/rss/ufaj20"),
    Source("Quantitative Finance (TOC)", "research", "rss",
           "https://www.tandfonline.com/feed/rss/rquf20"),
    Source("Journal of Empirical Finance (TOC)", "research", "rss",
           "https://rss.sciencedirect.com/publication/science/09275398"),
    # --- registered but no confirmed free feed ---
    Source("Review of Financial Studies (TOC)", "research", "rss",
           "https://academic.oup.com/rfs/rss", enabled=False, note="Oxford RSS URL unstable (404)"),
    Source("CEPR / VoxEU", "research", "rss", "https://cepr.org/rss.xml", enabled=False,
           note="no stable public RSS (404)"),
    Source("IMF Working Papers", "research", "rss",
           "https://www.imf.org/en/Publications/RSS?series=IMF+Working+Papers",
           enabled=False, note="403/empty"),
    Source("SSRN FEN", "research", "rss", "https://papers.ssrn.com/sol3/Jeljour_results.cfm",
           enabled=False, note="SSRN RSS restricted"),

    # ===================== NEWS (intentionally minimal) =====================
    # Per user request we minimize news. Only low-noise *official* sources stay
    # enabled; commercial headline feeds (Yahoo/CNBC/Nasdaq) are disabled.
    Source("Federal Reserve (Press)", "news", "rss",
           "https://www.federalreserve.gov/feeds/press_all.xml"),
    Source("Yahoo Finance", "news", "rss", "https://finance.yahoo.com/news/rssindex",
           enabled=False, note="disabled: news minimized per user"),
    Source("CNBC / commercial headline feeds", "news", "rss", "", enabled=False,
           note="disabled: news minimized per user"),
]


def enabled_sources() -> list[Source]:
    return [s for s in SOURCES if s.enabled and s.url]


def by_name(name: str) -> Source | None:
    return next((s for s in SOURCES if s.name == name), None)

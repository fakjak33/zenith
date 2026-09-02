"""INDEX seed catalog — the user's supplied resource list, parsed and enriched.

PROVENANCE. The raw list lives at ``data/index/seed/seed_list_2026-09-01.csv``
and is committed verbatim. This module is the ENRICHED reading of it: every row
below traces back to a line in that file, plus the metadata needed to make it
useful (official URL, what the thing actually is, what it covers, who runs it).

WHAT THE RAW LIST ACTUALLY IS. Despite the .csv extension it is a
section-delimited text list under three headings — "RESEARCH + INSTITUTIONS +
INSIGHTS RESOURCES", "JOURNALS + ACADEMIC RESOURCES + RESEARCH", and
"NEWS + TOOLS" — which map almost exactly onto the requested primary taxonomy
(institutional / academic / tool). That division is honoured rather than
re-derived, because it is the user's own classification.

PEOPLE COME FROM THE PARENTHETICALS. Rows such as
``"TIER1 ALPHA (E.G., MIKE GREEN, CRAIG PETERSON, DAVID PEGLER)"`` encode people
AND their affiliation. Those become Person entities joined to the firm by a
``works_at`` edge, which is why the knowledge graph has real edges on day one,
before any podcast harvesting.

HONESTY. Every entry carries a ``confidence``. Entries marked ``low`` are ones
whose identity or official source could not be established — their ``url`` is
left EMPTY and they ship as ``needs_review`` rather than pointing at a
plausible-looking domain that may belong to someone else. URLs recorded here
were resolved by probing the domain during authoring; ``links.py`` then
re-checks every one of them and records the real HTTP result, so nothing is
marked ``verified`` on an author's say-so.
"""

from __future__ import annotations

from . import model as m
from . import taxonomy as tx

SEED_PROVENANCE = "user seed list 2026-09-01"
PRIOR_DB_PROVENANCE = "prior manual compilation (ttu_fwm, 2026-08-08)"

# ---------------------------------------------------------------------------
# Organisations, academic sources and tools.
#
# Tuple shape: (name, entity_type, org_subtype, primary_category, url,
#               description, approach[], asset_classes[], insight_types[],
#               key_people[], extra{})
# `extra` carries anything optional: aliases, location, founded, research_url,
# confidence, notes, status.
# ---------------------------------------------------------------------------
_E = "organisation"
_INST, _ACAD, _TOOL = "institutional", "academic", "tool"

ENTRIES: list[tuple] = [
    # ===================== ALTERNATIVE / HEDGE FUND MANAGERS =====================
    ("Citadel", _E, "hedge_fund", _INST, "https://www.citadel.com",
     "One of the largest multi-strategy hedge funds, running equities, commodities, "
     "fixed income & macro, credit and quantitative strategies from Miami and Chicago.",
     ["multi_strategy", "quantitative", "fundamental"], ["multi_asset", "equities"],
     ["research", "institutional_investing"], ["Ken Griffin"],
     {"location": "Miami, FL", "founded": "1990", "aliases": ["Citadel LLC"],
      "notes": "Zenith cannot scrape Citadel insights: robots.txt disallows the "
               "/insights path (documented in zenith/sources.py). Listed here as a "
               "reference source, not an ingestible feed."}),
    ("Citadel Securities", _E, "trading_firm", _INST, "https://www.citadelsecurities.com",
     "Market maker and liquidity provider across equities, options, fixed income and FX; "
     "a separate firm from the Citadel hedge fund, sharing a founder.",
     ["market_making", "quantitative"], ["equities", "derivatives", "fixed_income"],
     ["market_data", "research"], ["Ken Griffin"],
     {"aliases": ["Citadel Global Market Intelligence"],
      "notes": "The seed list's 'CITADEL GLOBAL MARKET INTELLIGENCE' refers to Citadel's "
               "market-commentary output; kept as an alias rather than a separate entity."}),
    ("Bridgewater Associates", _E, "hedge_fund", _INST, "https://www.bridgewater.com",
     "The largest hedge fund by assets, known for systematic global macro, the All Weather "
     "risk-parity portfolio and extensive published research on economic machine dynamics.",
     ["global_macro", "systematic", "asset_allocation"], ["multi_asset"],
     ["research", "macroeconomics", "asset_allocation"], ["Ray Dalio", "Bob Prince", "Greg Jensen"],
     {"location": "Westport, CT", "founded": "1975",
      "research_url": "https://www.bridgewater.com/research-and-insights"}),
    ("Two Sigma", _E, "hedge_fund", _INST, "https://www.twosigma.com",
     "Quantitative investment manager applying machine learning, distributed computing and "
     "large alternative datasets across systematic strategies.",
     ["quantitative", "systematic", "stat_arb"], ["equities", "multi_asset"],
     ["quantitative_research", "research"], ["John Overdeck", "David Siegel"],
     {"location": "New York, NY", "founded": "2001",
      "research_url": "https://www.twosigma.com/insights/"}),
    ("Renaissance Technologies", _E, "hedge_fund", _INST, "https://www.rentec.com",
     "Pioneering quantitative hedge fund; its Medallion fund is the most celebrated "
     "track record in systematic trading. Publishes essentially nothing.",
     ["quantitative", "systematic", "stat_arb"], ["multi_asset"],
     ["quantitative_research"], ["Jim Simons", "Peter Brown"],
     {"location": "East Setauket, NY", "founded": "1982",
      "notes": "Included for completeness as an institutional reference. Publishes no "
               "public research — there is nothing to monitor."}),
    ("D. E. Shaw & Co.", _E, "hedge_fund", _INST, "https://www.deshaw.com",
     "Quantitative and qualitative multi-strategy investment firm, one of the original "
     "computational finance shops.",
     ["quantitative", "systematic", "multi_strategy"], ["multi_asset"],
     ["quantitative_research", "research"], ["David E. Shaw"],
     {"location": "New York, NY", "founded": "1988", "aliases": ["D.E. Shaw", "DE Shaw"]}),
    ("Point72", _E, "hedge_fund", _INST, "https://www.point72.com",
     "Multi-strategy manager built around discretionary long/short equity pods alongside "
     "macro and systematic units.",
     ["long_short_equity", "multi_strategy", "discretionary"], ["equities", "multi_asset"],
     ["research", "investment_ideas"], ["Steven A. Cohen"],
     {"location": "Stamford, CT", "aliases": ["Point 72"]}),
    ("Balyasny Asset Management", _E, "hedge_fund", _INST, "https://www.bamfunds.com",
     "Multi-strategy platform running equities long/short, macro, commodities and "
     "systematic teams.",
     ["multi_strategy", "long_short_equity", "macro"], ["multi_asset", "equities"],
     ["research", "institutional_investing"], ["Dmitry Balyasny"],
     {"location": "Chicago, IL", "aliases": ["BAM", "Balyasny"]}),
    ("Man Group", _E, "asset_manager", _INST, "https://www.man.com",
     "Listed alternative investment manager and the parent of AHL, Numeric, GLG and "
     "FRM; publishes an unusually deep public research library through Man Institute.",
     ["systematic", "quantitative", "multi_strategy", "trend_following"],
     ["multi_asset", "equities", "credit"],
     ["research", "quantitative_research", "academic_research"], [],
     {"location": "London, UK", "founded": "1783",
      "research_url": "https://www.man.com/insights"}),
    ("Man Institute", _E, "research_house", _INST, "https://www.man.com/insights",
     "Man Group's research arm — the publishing front for AHL/Numeric/GLG quantitative "
     "and macro research, white papers and academic collaborations.",
     ["quantitative", "systematic", "factor_investing"], ["multi_asset"],
     ["research", "quantitative_research", "academic_research"], [],
     {"notes": "Man Group's research publication brand, not a separate firm — modelled "
               "as a subsidiary_of edge to Man Group. man.com/institute returns 404; the "
               "live research hub is man.com/insights (link sweep, 2026-09-01)."}),
    ("Man AHL", _E, "cta", _INST, "https://www.man.com/ahl",
     "Man Group's systematic managed-futures and trend-following division, one of the "
     "longest-running CTA programmes.",
     ["trend_following", "managed_futures", "systematic", "quantitative"],
     ["derivatives", "commodities", "fx", "fixed_income"],
     ["quantitative_research", "trading"], [],
     {"aliases": ["AHL", "Man Group AHL"], "founded": "1987"}),
    ("Winton", _E, "cta", _INST, "https://www.winton.com",
     "Systematic investment manager applying statistical research to futures and equity "
     "markets; long-running trend and multi-strategy programmes.",
     ["systematic", "quantitative", "trend_following", "managed_futures"],
     ["derivatives", "commodities", "equities"], ["quantitative_research", "research"],
     ["David Harding"],
     {"location": "London, UK", "founded": "1997", "aliases": ["Winton Capital", "Winton Group"]}),
    ("Aspect Capital", _E, "cta", _INST, "https://www.aspectcapital.com",
     "London systematic manager best known for its Diversified Programme, a broad "
     "medium-term trend-following strategy.",
     ["systematic", "trend_following", "managed_futures", "quantitative"],
     ["derivatives", "commodities", "fx"], ["quantitative_research", "research"], [],
     {"location": "London, UK", "founded": "1997"}),
    ("Transtrend", _E, "cta", _INST, "https://www.transtrend.com",
     "Dutch systematic manager running the Diversified Trend Program; publishes thoughtful "
     "commentary on the limits of trend-model interpretation.",
     ["systematic", "trend_following", "managed_futures"], ["derivatives", "commodities", "fx"],
     ["quantitative_research", "research"], [],
     {"location": "Rotterdam, Netherlands", "founded": "1991"}),
    ("DUNN Capital Management", _E, "cta", _INST, "https://dunncapital.com",
     "One of the oldest continuously-operating trend-following CTAs, running a high-volatility "
     "systematic programme since the 1970s.",
     ["trend_following", "managed_futures", "systematic"], ["derivatives", "commodities"],
     ["trading", "quantitative_research"], ["Bill Dunn", "Marty Bergin"],
     {"founded": "1974", "aliases": ["DUNN Capital"]}),
    ("Lynx Asset Management", _E, "cta", _INST, "https://www.lynxhedge.se",
     "Swedish systematic manager combining trend following with complementary "
     "diversifying models.",
     ["systematic", "trend_following", "managed_futures", "quantitative"],
     ["derivatives", "commodities", "fx"], ["quantitative_research", "research"],
     ["Svante Bergström"],
     {"location": "Stockholm, Sweden", "aliases": ["Lynx"]}),
    ("Graham Capital Management", _E, "cta", _INST, "https://www.grahamcapital.com",
     "Alternative manager running systematic macro/trend programmes alongside "
     "discretionary global macro traders.",
     ["global_macro", "systematic", "trend_following", "managed_futures"],
     ["multi_asset", "derivatives"], ["research", "quantitative_research"], ["Kenneth Tropin"],
     {"location": "Rowayton, CT", "founded": "1994"}),
    ("AlphaSimplex Group", _E, "cta", _INST, "https://www.alphasimplex.com",
     "Systematic manager founded by Andrew Lo, known for adaptive-markets-informed "
     "managed futures and prolific public research on trend following.",
     ["systematic", "trend_following", "managed_futures", "quantitative"],
     ["derivatives", "multi_asset"], ["quantitative_research", "research", "academic_research"],
     ["Andrew Lo", "Katy Kaminski"],
     {"aliases": ["AlphaSimplex"], "founded": "1999",
      "notes": "Zenith probed alphasimplex.com for an HTML insights hub during an earlier "
               "source sweep and it did not yield a scrapeable feed (see sources.py)."}),
    ("Gresham Investment Management", _E, "asset_manager", _INST, "https://www.greshamllc.com",
     "Commodity and real-asset specialist; its quantitative division publishes research "
     "on commodity factor investing.",
     ["commodities", "quantitative", "systematic"], ["commodities", "real_assets"],
     ["research", "quantitative_research"], [],
     {"aliases": ["Gresham Investment Management LLC"]}),
    ("GreshamQuant", _E, "research_house", _INST, "",
     "Quantitative research arm of Gresham Investment Management, publishing opinion "
     "pieces and research articles on systematic commodity strategies.",
     ["quantitative", "systematic", "commodities"], ["commodities"],
     ["quantitative_research", "research"], ["Yoav Git"],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "notes": "No standalone greshamquant.com resolved when probed (2026-09-01). Content "
               "appears to live under the Gresham Investment Management site; URL left empty "
               "rather than guessed."}),
    ("Aquantum", _E, "cta", _INST, "",
     "German systematic manager running quantitative commodity and futures strategies.",
     ["systematic", "quantitative", "commodities", "managed_futures"],
     ["commodities", "derivatives"], ["quantitative_research", "research"], ["Moritz Seibert"],
     {"confidence": "low", "lifecycle_state": "needs_review", "location": "Germany",
      "notes": "Candidate domains probed 2026-09-01 (aquantum.de, aquantum-group.com, "
               "aquantum-invest.com) did not resolve cleanly. URL left empty pending "
               "confirmation. Moritz Seibert's affiliation is from the seed list itself."}),
    ("Mount Lucas Management", _E, "cta", _INST, "https://www.mtlucas.com",
     "Systematic manager behind the MLM Index, an early published benchmark for "
     "trend-following returns.",
     ["systematic", "trend_following", "managed_futures"], ["commodities", "derivatives"],
     ["quantitative_research", "index_data"], [],
     {"aliases": ["MT Lucas", "Mount Lucas Management LP"],
      "notes": "Seed list said 'MT LUCAS'; the firm's own name is Mount Lucas Management."}),
    ("Efficient Capital Management", _E, "asset_manager", _INST, "https://www.efficient.com",
     "Managed-futures multi-manager platform allocating across CTA programmes; publishes "
     "research on manager selection and portfolio construction in alternatives.",
     ["managed_futures", "multi_strategy", "portfolio_construction"], ["derivatives", "alternatives"],
     ["manager_selection", "research", "portfolio_construction"], ["Marat Molyboga"],
     {"aliases": ["Efficient Capital Markets", "Efficient Capital"],
      "notes": "Seed list said 'EFFICIENT CAPITAL MARKETS'; efficientcapital.com redirects "
               "to efficient.com, the firm's current site (probed 2026-09-01)."}),
    ("Dynamic Beta Investments", _E, "asset_manager", _INST, "https://dbi.co",
     "Hedge-fund and managed-futures replication specialist; the sub-adviser behind the "
     "DBMF managed-futures replication ETF that Zenith's HOLDINGS tab tracks.",
     ["replication", "systematic", "managed_futures", "quantitative"],
     ["derivatives", "alternatives", "multi_asset"], ["research", "quantitative_research"],
     ["Andrew Beer", "Mathias Mamou-Mani"],
     {"aliases": ["DBI", "Dynamic Beta"],
      "notes": "Directly relevant to Zenith: DBMF is the first fund tracked in the "
               "HOLDINGS tab. dynamicbeta.com redirects to dbi.co (probed 2026-09-01)."}),
    ("Convex Asset Management", _E, "hedge_fund", _INST, "",
     "Options and volatility-focused trading firm.",
     ["volatility", "options", "derivatives"], ["volatility", "derivatives"],
     ["options", "volatility", "trading"], ["Noel Smith"],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "notes": "Candidate domains probed 2026-09-01 did not resolve. URL left empty. "
               "Noel Smith's affiliation is from the seed list itself."}),
    ("The Ambrus Group", _E, "hedge_fund", _INST, "",
     "Volatility-focused trading firm known for convex/tail-risk equity volatility strategies.",
     ["volatility", "options", "derivatives"], ["volatility", "equities", "derivatives"],
     ["options", "volatility", "trading"], ["Kris Sidial"],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "aliases": ["The Ambros Group", "Ambrus Group"],
      "notes": "Seed list spelled it 'AMBROS'. ambrosgroup.com returned HTTP 200 with no page "
               "title on a first probe and then timed out on the link sweep, both on "
               "2026-09-01, so the domain is NOT confirmed as this firm's and no URL "
               "is asserted. Kris Sidial's affiliation is from the seed list itself."}),
    ("Archive Capital Advisors", _E, "research_house", _INST,
     "https://www.archivecapitaladvisors.ie",
     "Investment advisory firm run by Alan Dunne, co-host of Top Traders Unplugged's "
     "Allocator series; focuses on institutional asset allocation and alternatives.",
     ["asset_allocation", "managed_futures", "portfolio_construction"], ["multi_asset", "alternatives"],
     ["asset_allocation", "institutional_investing", "manager_selection"], ["Alan Dunne"],
     {"location": "Ireland", "aliases": ["Archive Capital"]}),
    ("Tier1 Alpha", _E, "research_house", _INST, "https://tier1alpha.com",
     "Research shop analysing market structure, dealer positioning, gamma exposure and "
     "volatility-driven flows.",
     ["volatility", "options", "market_microstructure", "derivatives"],
     ["volatility", "equities", "derivatives"],
     ["research", "positioning", "options", "volatility"],
     ["Mike Green", "Craig Peterson", "David Pegler"], {}),
    ("Hedgeye Risk Management", _E, "research_house", _INST, "https://www.hedgeye.com",
     "Independent subscription research firm publishing macro, sector and single-name "
     "views through a quantitative-overlay process.",
     ["macro", "discretionary"], ["multi_asset", "equities"],
     ["research", "macroeconomics", "investment_ideas"], ["Keith McCullough"],
     {"aliases": ["Hedgeye"]}),
    ("Newfound Research", _E, "asset_manager", _INST, "https://www.thinknewfound.com",
     "Quantitative asset manager and prolific publisher on portfolio construction, "
     "rebalance timing luck, trend following and return stacking.",
     ["quantitative", "systematic", "portfolio_construction", "trend_following"],
     ["multi_asset"], ["quantitative_research", "portfolio_construction", "research"],
     ["Corey Hoffstein"],
     {"research_url": "https://blog.thinknewfound.com",
      "notes": "Already ingested by Zenith — see the 'Newfound (Flirting with Models)' feed "
               "in sources.py. Corey Hoffstein hosts the Flirting with Models podcast."}),
    ("Jane Street", _E, "trading_firm", _INST, "https://www.janestreet.com",
     "Global proprietary trading firm and major ETF liquidity provider, notable for its "
     "public technical writing and puzzles.",
     ["market_making", "quantitative", "systematic"], ["equities", "derivatives", "fixed_income"],
     ["market_data", "trading"], [],
     {"notes": "No public research feed to ingest (confirmed during an earlier Zenith "
               "source sweep)."}),
    ("Optiver", _E, "trading_firm", _INST, "https://www.optiver.com",
     "Dutch market maker specialising in options and derivatives liquidity provision.",
     ["market_making", "options", "derivatives", "quantitative"],
     ["derivatives", "volatility", "equities"], ["trading", "options", "market_data"], [],
     {"location": "Amsterdam, Netherlands",
      "notes": "Probed as an HTML source during an earlier Zenith sweep; no usable feed."}),
    ("Baker Bros. Advisors", _E, "hedge_fund", _INST, "",
     "Biotechnology-focused long-term investment manager.",
     ["fundamental", "long_short_equity"], ["equities"], ["investment_ideas", "research"],
     ["Julian Baker", "Felix Baker"],
     {"aliases": ["Bakers Brothers Advisors", "Baker Brothers"],
      "confidence": "low", "lifecycle_state": "needs_review",
      "notes": "Seed list said 'BAKERS BROTHERS ADVISORS'; the firm's registered name is "
               "Baker Bros. Advisors LP. No official site resolved when probed 2026-09-01 "
               "(bbabio.com and bakerbrosadvisors.com both failed; bakerbros.com is an "
               "unrelated flooring retailer), so no URL is asserted. The firm is a real "
               "13F filer and can be confirmed through SEC EDGAR."}),
    ("Coatue Management", _E, "hedge_fund", _INST, "https://www.coatue.com",
     "Technology-focused crossover investor spanning public long/short equity and "
     "private growth investing; publishes widely-circulated tech/AI market decks.",
     ["long_short_equity", "growth", "venture_capital", "fundamental"],
     ["equities", "private_markets"], ["research", "investment_ideas"], ["Philippe Laffont"],
     {"aliases": ["Coatue"]}),
    ("GQG Partners", _E, "asset_manager", _INST, "https://gqg.com",
     "Global and emerging-market equity manager running concentrated, quality-oriented "
     "portfolios with an unusually public research posture.",
     ["fundamental", "quality", "growth"], ["equities"], ["research", "investment_ideas"],
     ["Rajiv Jain"],
     {"notes": "Probed as an HTML source during an earlier Zenith sweep; no usable feed."}),

    # ===================== PRIVATE MARKETS =====================
    ("Apollo Global Management", _E, "private_markets", _INST, "https://www.apollo.com",
     "Alternative asset manager spanning private credit, private equity and retirement "
     "services; its Apollo Academy is a widely-read free macro research channel.",
     ["private_equity", "credit", "macro"], ["private_markets", "credit"],
     ["research", "macroeconomics", "alternative_investments"], ["Marc Rowan", "Torsten Sløk"],
     {"aliases": ["Apollo"]}),
    ("Apollo Academy", _E, "research_house", _INST, "https://www.apolloacademy.com",
     "Apollo's free public education and research channel — the home of Torsten Sløk's "
     "Daily Spark macro chartbook.",
     ["macro"], ["multi_asset"], ["macroeconomics", "research", "financial_education"],
     ["Torsten Sløk"],
     {"aliases": ["Apollo Acadamy", "Apollo Daily Spark"],
      "notes": "Seed list contained both 'APOLLO ACADAMY' [sic] and 'APOLLO DAILY SPARK'. "
               "The Daily Spark is a publication OF Apollo Academy, so both fold into this "
               "entry as aliases. Zenith's feed for this is currently disabled: the RSS "
               "endpoint returns HTTP 200 with zero entries after a 2026 site restructure "
               "(see sources.py)."}),
    ("Blackstone", _E, "private_markets", _INST, "https://www.blackstone.com",
     "The largest alternative asset manager, spanning real estate, private equity, credit "
     "and hedge fund solutions; publishes regular macro and market commentary.",
     ["private_equity", "real_estate", "credit"], ["private_markets", "real_assets", "credit"],
     ["research", "alternative_investments", "macroeconomics"],
     ["Stephen Schwarzman", "Jon Gray", "Joe Zidle"], {"location": "New York, NY"}),
    ("KKR", _E, "private_markets", _INST, "https://www.kkr.com",
     "Global investment firm across private equity, infrastructure, real estate and credit; "
     "its Global Macro team publishes influential asset-allocation research.",
     ["private_equity", "infrastructure", "credit", "asset_allocation"],
     ["private_markets", "real_assets", "credit"],
     ["research", "macroeconomics", "asset_allocation"], ["Henry McVey"],
     {"research_url": "https://www.kkr.com/insights",
      "notes": "Probed as an HTML source during an earlier Zenith sweep; robots/JS blocked."}),

    # ===================== BANKS / SELL-SIDE =====================
    ("JPMorgan Chase", _E, "bank", _INST, "https://www.jpmorgan.com",
     "Global bank whose research and asset-management arms publish the Guide to the Markets "
     "and Long-Term Capital Market Assumptions.",
     ["macro", "fundamental"], ["multi_asset"], ["research", "macroeconomics", "market_data"], [],
     {"aliases": ["JP Morgan", "J.P. Morgan"]}),
    ("J.P. Morgan Asset Management", _E, "asset_manager", _INST,
     "https://am.jpmorgan.com",
     "Asset-management arm publishing the Guide to the Markets, one of the most widely used "
     "free market chartbooks, plus Long-Term Capital Market Assumptions.",
     ["asset_allocation", "fundamental"], ["multi_asset"],
     ["asset_allocation", "research", "market_data"], [],
     {"aliases": ["JPMAM", "J.P. Morgan AM"],
      "notes": "Already ingested by Zenith as an HTML source (see sources.py)."}),
    ("Goldman Sachs", _E, "bank", _INST, "https://www.goldmansachs.com",
     "Global investment bank; its Global Investment Research and Asset Management arms "
     "publish widely-followed macro and thematic work.",
     ["macro", "fundamental", "quantitative"], ["multi_asset"],
     ["research", "macroeconomics", "market_data"], [],
     {"research_url": "https://www.goldmansachs.com/insights"}),
    ("Bank of America", _E, "bank", _INST, "https://www.bankofamerica.com",
     "Global bank whose BofA Global Research publishes the closely-watched Global Fund "
     "Manager Survey and Flow Show positioning work.",
     ["macro", "fundamental"], ["multi_asset", "equities"],
     ["research", "positioning", "macroeconomics"], ["Michael Hartnett"],
     {"aliases": ["BofA", "BofA Research", "Bank of America Merrill Lynch"],
      "notes": "Seed list had 'BANK OF AMERICA' and 'BOFA RESEARCH' as separate rows; "
               "BofA Global Research is a division, folded in as an alias. bofa.com does "
               "not resolve — the corporate site is bankofamerica.com (link sweep, 2026-09-01)."}),
    ("Citigroup", _E, "bank", _INST, "https://www.citigroup.com",
     "Global bank; Citi Research publishes macro, rates and equity strategy.",
     ["macro", "fundamental"], ["multi_asset"], ["research", "macroeconomics"], [],
     {"aliases": ["Citi"]}),
    ("UBS", _E, "bank", _INST, "https://www.ubs.com",
     "Swiss global bank and wealth manager; publishes the Global Investment Returns "
     "Yearbook (Dimson-Marsh-Staunton) and extensive CIO research.",
     ["macro", "asset_allocation", "fundamental"], ["multi_asset"],
     ["research", "market_history", "asset_allocation"], [],
     {"notes": "Publisher of the DMS Global Investment Returns Yearbook — the standard "
               "long-run asset class return dataset."}),
    ("Deutsche Bank", _E, "bank", _INST, "https://www.db.com",
     "German global bank; DB Research produces the Chart of the Day and long-run "
     "historical market studies.",
     ["macro", "fundamental"], ["multi_asset"], ["research", "macroeconomics"],
     ["Jim Reid"],
     {"aliases": ["Deutsche Bank Chart of the Day", "DB Research"],
      "notes": "Seed list had 'DEUTSCHE BANK' and 'DEUTSCHE BANK CHART OF THE DAY' as "
               "separate rows; the latter is a DB Research publication, folded in as an "
               "alias. Zenith cannot ingest it — DB Research is login-gated (sources.py). "
               "Charts are frequently mirrored on isabelnet.com."}),
    ("Barclays", _E, "bank", _INST, "https://www.barclays.com",
     "UK global bank; Barclays Research publishes macro, rates, credit and systematic "
     "strategy work.",
     ["macro", "quantitative", "fixed_income"], ["multi_asset", "fixed_income", "credit"],
     ["research", "macroeconomics"], [], {}),
    ("HSBC", _E, "bank", _INST, "https://www.hsbc.com",
     "Global bank with a strong emerging-market and Asia research franchise.",
     ["macro", "fundamental"], ["multi_asset", "fx"], ["research", "macroeconomics"], [], {}),
    ("Wells Fargo", _E, "bank", _INST, "https://www.wellsfargo.com",
     "US bank; its Investment Institute publishes asset-allocation and macro commentary.",
     ["macro", "asset_allocation"], ["multi_asset"], ["research", "macroeconomics"], [], {}),

    # ===================== ASSET MANAGERS / ETF ISSUERS =====================
    ("BlackRock", _E, "asset_manager", _INST, "https://www.blackrock.com",
     "The world's largest asset manager and the parent of the iShares ETF platform; the "
     "BlackRock Investment Institute publishes macro and asset-allocation research.",
     ["passive", "factor_investing", "asset_allocation"], ["multi_asset", "equities", "fixed_income"],
     ["research", "asset_allocation", "market_data", "index_data"], ["Larry Fink"],
     {"aliases": ["BlackRock / iShares", "BLK"],
      "research_url": "https://www.blackrock.com/corporate/insights/blackrock-investment-institute"}),
    ("iShares", _E, "asset_manager", _INST, "https://www.ishares.com",
     "BlackRock's ETF platform — the largest ETF family globally, and a primary source of "
     "fund-level holdings and flow data.",
     ["passive", "factor_investing"], ["equities", "fixed_income", "multi_asset"],
     ["market_data", "index_data", "portfolio_monitoring"], [],
     {"notes": "A brand of BlackRock, not a separate firm — modelled as a subsidiary_of edge."}),
    ("State Street Global Advisors", _E, "asset_manager", _INST, "https://www.ssga.com",
     "Asset manager behind the SPDR ETF family, including SPY — the first US-listed ETF.",
     ["passive", "factor_investing", "asset_allocation"], ["equities", "fixed_income", "multi_asset"],
     ["market_data", "index_data", "research"], [],
     {"aliases": ["SSGA", "State Street Global Advisors", "State Street"],
      "notes": "Seed list had 'SPDR', 'STATE STREET' and 'SSGA' as three separate rows. "
               "SSGA is the asset-management arm; SPDR is its ETF brand; State Street Corp "
               "is the listed parent. Modelled as SSGA (this entry) with SPDR as a "
               "subsidiary brand."}),
    ("SPDR ETFs", _E, "asset_manager", _INST, "https://www.ssga.com/us/en/intermediary/etfs",
     "State Street's ETF brand — the SPDR family, including SPY and the sector SPDRs that "
     "Zenith's own CAS and MOMENTUM engines use as sector proxies.",
     ["passive", "factor_investing"], ["equities", "multi_asset"],
     ["market_data", "index_data"], [],
     {"aliases": ["SPDR"],
      "notes": "Directly relevant to Zenith: the sector SPDRs are the sector benchmark set "
               "used by MOMENTUM's cross-universe comparison."}),
    ("Vanguard", _E, "asset_manager", _INST, "https://www.vanguard.com",
     "Client-owned asset manager that popularised low-cost index investing; publishes "
     "long-horizon capital markets research.",
     ["passive", "asset_allocation", "evidence_based"], ["equities", "fixed_income", "multi_asset"],
     ["research", "asset_allocation", "financial_education"], ["John C. Bogle"],
     {"founded": "1975",
      "notes": "Probed as an HTML source during an earlier Zenith sweep; no usable feed."}),
    ("Capital Group", _E, "asset_manager", _INST, "https://www.capitalgroup.com",
     "One of the largest active managers, running the American Funds via a multi-manager "
     "portfolio system.",
     ["fundamental", "growth", "discretionary"], ["equities", "fixed_income"],
     ["research", "investment_ideas"], [], {"founded": "1931"}),
    ("AQR Capital Management", _E, "asset_manager", _INST, "https://www.aqr.com",
     "Quantitative manager and one of the most prolific publishers of factor, value, "
     "momentum, trend and alternative-risk-premia research in the industry.",
     ["quantitative", "systematic", "factor_investing", "alternative_risk_premia",
      "value", "momentum", "trend_following"],
     ["multi_asset", "equities"],
     ["academic_research", "quantitative_research", "research", "portfolio_construction"],
     ["Cliff Asness", "John Liew", "Antti Ilmanen"],
     {"aliases": ["AQR", "AQR Research"], "founded": "1998", "location": "Greenwich, CT",
      "research_url": "https://www.aqr.com/Insights"}),
    ("Dimensional Fund Advisors", _E, "asset_manager", _INST, "https://www.dimensional.com",
     "Systematic manager built directly on academic asset-pricing research (Fama-French "
     "factors); a foundational evidence-based investing firm.",
     ["factor_investing", "systematic", "evidence_based", "value", "quality"],
     ["equities", "fixed_income"], ["academic_research", "research", "financial_education"],
     ["David Booth", "Eugene Fama", "Kenneth French"],
     {"aliases": ["Dimensional", "DFA"], "founded": "1981"}),
    ("Avantis Investors", _E, "asset_manager", _INST, "https://www.avantisinvestors.com",
     "Systematic factor-tilted ETF and fund manager founded by former Dimensional staff.",
     ["factor_investing", "systematic", "evidence_based", "value"], ["equities", "fixed_income"],
     ["research", "financial_education"], ["Eduardo Repetto"], {"aliases": ["Avantis"]}),
    ("Alpha Architect", _E, "asset_manager", _INST, "https://alphaarchitect.com",
     "Quantitative manager and research publisher translating academic asset-pricing papers "
     "into practitioner-readable summaries; runs concentrated value and momentum ETFs.",
     ["quantitative", "factor_investing", "value", "momentum", "evidence_based"],
     ["equities"], ["academic_research", "quantitative_research", "financial_education", "research"],
     ["Wesley Gray", "Jack Vogel"],
     {"notes": "Already ingested by Zenith — see the 'Alpha Architect' RSS feed in sources.py."}),
    ("Research Affiliates", _E, "research_house", _INST, "https://www.researchaffiliates.com",
     "Research-driven firm behind RAFI fundamental indexation; publishes extensively on "
     "valuation-aware factor investing and long-horizon expected returns.",
     ["factor_investing", "value", "asset_allocation", "quantitative"],
     ["equities", "multi_asset"], ["research", "academic_research", "asset_allocation"],
     ["Rob Arnott", "Cam Harvey"],
     {"notes": "Enabled in Zenith's source registry as an anti-bot SPA that is attempted "
               "via Firecrawl nightly (see sources.py)."}),
    ("PIMCO", _E, "asset_manager", _INST, "https://www.pimco.com",
     "Fixed-income specialist manager; publishes Secular and Cyclical Outlooks alongside "
     "quantitative research on bonds and macro.",
     ["fixed_income", "macro", "credit"], ["fixed_income", "credit", "multi_asset"],
     ["research", "macroeconomics", "academic_research"], [],
     {"founded": "1971", "research_url": "https://www.pimco.com/en-us/insights"}),
    ("Simplify Asset Management", _E, "asset_manager", _INST, "https://www.simplify.us",
     "ETF issuer specialising in options-overlay, convexity, managed-futures and "
     "alternative-return-stream strategies.",
     ["options", "volatility", "derivatives", "managed_futures", "replication"],
     ["derivatives", "volatility", "multi_asset"], ["research", "options", "volatility"], [],
     {"aliases": ["Simplify"],
      "notes": "Already ingested by Zenith — see the 'Simplify Asset Management' feed in "
               "sources.py."}),
    ("Cambria Investment Management", _E, "asset_manager", _INST,
     "https://www.cambriainvestments.com",
     "Quantitative ETF manager founded by Meb Faber, running trend, value and "
     "shareholder-yield strategies; a prolific free-research publisher.",
     ["quantitative", "systematic", "value", "trend_following", "asset_allocation"],
     ["equities", "multi_asset"], ["research", "quantitative_research", "financial_education"],
     ["Meb Faber"],
     {"aliases": ["Cambria Funds", "Cambria"],
      "notes": "Already ingested by Zenith via the 'Meb Faber / Cambria' feed (sources.py). "
               "Meb Faber hosts The Meb Faber Show."}),
    ("Columbia Threadneedle Investments", _E, "asset_manager", _INST,
     "https://www.columbiathreadneedle.com",
     "Global asset manager across equities, fixed income and alternatives.",
     ["fundamental", "asset_allocation"], ["multi_asset", "equities", "fixed_income"],
     ["research"], [], {"aliases": ["Columbia Threadneedle"]}),
    ("New York Life Investment Management", _E, "asset_manager", _INST,
     "https://www.newyorklifeinvestments.com",
     "Multi-boutique asset manager; parent of IndexIQ, whose liquid-alternative ETFs "
     "include hedge-fund replication strategies.",
     ["replication", "asset_allocation"], ["multi_asset", "alternatives"], ["research"], [],
     {"aliases": ["NYLIM", "New York Life Investments"]}),
    ("AGF Management", _E, "asset_manager", _INST, "https://www.agf.com",
     "Canadian asset manager; AGF Investments publishes macro and quantitative research, "
     "including systematic strategies via AGFiQ.",
     ["quantitative", "factor_investing", "macro"], ["multi_asset", "equities"],
     ["research", "quantitative_research"], [],
     {"aliases": ["AGF Investments", "AGF Management Limited"],
      "notes": "Probed for RSS during an earlier Zenith sweep; feed returned empty."}),
    ("Virtus Investment Partners", _E, "asset_manager", _INST, "https://www.virtus.com",
     "Multi-boutique asset manager distributing a range of affiliated investment managers.",
     ["multi_strategy", "fundamental"], ["multi_asset"], ["research"], [],
     {"aliases": ["Virtus"],
      "notes": "Probed for RSS during an earlier Zenith sweep; feed returned empty."}),
    ("Convergence Investment Partners", _E, "asset_manager", _INST, "",
     "Quantitative manager publishing insights, research and signal analysis.",
     ["quantitative", "systematic"], ["equities"], ["quantitative_research", "research"], [],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "notes": "Candidate domains probed 2026-09-01 did not resolve. URL left empty rather "
               "than guessed; the descriptive text is taken from the seed list's own "
               "parenthetical ('INSIGHTS, RESEARCH, SIGNAL ANALYSIS')."}),
    ("Charles Schwab", _E, "wealth_platform", _INST, "https://www.schwab.com",
     "Brokerage and wealth platform; the Schwab Center for Financial Research publishes "
     "free market and behavioural commentary.",
     ["asset_allocation", "behavioral_finance"], ["multi_asset"],
     ["research", "financial_education", "market_data"], ["Liz Ann Sonders", "Kathy Jones"],
     {"aliases": ["Schwab"],
      "notes": "Probed for RSS during an earlier Zenith sweep; feed returned empty."}),

    # ===================== EXCHANGES, DATA & INDEX PROVIDERS =====================
    ("CME Group", _E, "exchange", _INST, "https://www.cmegroup.com",
     "The largest futures exchange operator; home of the CME FedWatch tool, which infers "
     "policy-rate probabilities from fed funds futures.",
     ["derivatives", "macro"], ["derivatives", "commodities", "fixed_income", "fx"],
     ["market_data", "monetary_policy", "derivatives", "index_data"], [],
     {"aliases": ["CME", "CME FedWatch"],
      "notes": "Appeared TWICE in the seed list — once under institutions and once under "
               "tools as 'CME (E.G., FED WATCH)'. Merged into one entity; FedWatch is "
               "listed separately as a tool because it is used as a tool."}),
    ("Cboe Global Markets", _E, "exchange", _INST, "https://www.cboe.com",
     "Options and volatility exchange operator; publisher of the VIX and the wider "
     "volatility index family.",
     ["options", "volatility", "derivatives"], ["volatility", "derivatives", "equities"],
     ["market_data", "volatility", "options", "index_data"], [],
     {"aliases": ["CBOE"],
      "notes": "Probed as an HTML source during an earlier Zenith sweep; no usable feed."}),
    ("Nasdaq", _E, "exchange", _INST, "https://www.nasdaq.com",
     "Exchange operator and market-data provider; also an index provider.",
     ["passive"], ["equities", "derivatives"], ["market_data", "index_data", "news"], [], {}),
    ("S&P Global", _E, "data_provider", _INST, "https://www.spglobal.com",
     "Index provider (S&P Dow Jones Indices), credit rating agency and market-data business; "
     "publisher of the S&P 500 and the SPIVA active-vs-passive scorecards.",
     ["passive", "factor_investing"], ["multi_asset", "equities", "credit"],
     ["index_data", "market_data", "research", "regulatory"], [],
     {"aliases": ["SPGI", "S&P Dow Jones Indices", "S&P"],
      "notes": "Seed list had both 'S&P GLOBAL' and 'SPGI' (its ticker) as separate rows — "
               "the same entity, merged. Probed as an HTML source during an earlier Zenith "
               "sweep; no usable feed."}),
    ("FactSet", _E, "data_provider", _INST, "https://www.factset.com",
     "Financial data and analytics platform; FactSet Insight publishes free market and "
     "earnings research, including the widely-cited Earnings Insight series.",
     ["fundamental", "quantitative"], ["equities", "multi_asset"],
     ["market_data", "research", "fundamental_analysis"], ["John Butters"],
     {"research_url": "https://insight.factset.com",
      "notes": "Already ingested by Zenith — see the 'FactSet Insight' RSS feed in sources.py."}),
    ("LSEG", _E, "data_provider", _INST, "https://www.lseg.com",
     "London Stock Exchange Group — exchange operator and, via the former Refinitiv "
     "business, a major market-data and analytics provider.",
     [], ["multi_asset"], ["market_data", "index_data", "news"], [],
     {"aliases": ["London Stock Exchange Group", "Refinitiv"]}),
    ("CFRA", _E, "research_house", _INST, "https://www.cfraresearch.com",
     "Independent equity and fund research provider; successor to the S&P Capital IQ "
     "equity research franchise.",
     ["fundamental"], ["equities"], ["research", "fundamental_analysis", "investment_ideas"], [], {}),
    ("Morningstar", _E, "data_provider", _INST, "https://www.morningstar.com",
     "Fund research, ratings and data provider; publishes extensive free research on funds, "
     "asset allocation and investor behaviour, and hosts The Long View podcast.",
     ["evidence_based", "asset_allocation", "factor_investing"], ["multi_asset", "equities"],
     ["manager_selection", "research", "market_data", "financial_education", "podcasts"],
     ["Christine Benz", "Dan Lefkovitz", "Amy Arnott"],
     {"notes": "Enabled in Zenith's source registry as an anti-bot SPA attempted via "
               "Firecrawl nightly (see sources.py). Also the publisher of The Long View, "
               "one of the podcasts INDEX monitors."}),
    ("BCA Research", _E, "research_house", _INST, "https://www.bcaresearch.com",
     "Independent macro research house producing global investment strategy across "
     "asset classes.",
     ["macro", "global_macro", "asset_allocation"], ["multi_asset"],
     ["research", "macroeconomics", "asset_allocation"], [], {"founded": "1949"}),
    ("HedgeNordic", _E, "media", _INST, "https://hedgenordic.com",
     "News and research service covering the Nordic hedge fund industry, including an "
     "annual industry report series.",
     ["multi_strategy", "managed_futures"], ["alternatives"],
     ["news", "manager_selection", "research"], [],
     {"location": "Nordics"}),

    # ===================== OFFICIAL SECTOR =====================
    ("Bank for International Settlements", _E, "official_sector", _ACAD,
     "https://www.bis.org",
     "The central banks' bank — publishes the Quarterly Review, working papers and the "
     "triennial FX and derivatives market surveys.",
     ["macro"], ["multi_asset", "fx", "fixed_income"],
     ["academic_research", "macroeconomics", "monetary_policy", "regulatory"], [],
     {"aliases": ["BIS"],
      "notes": "Already ingested by Zenith — see the BIS feed in sources.py."}),
    ("Office of Financial Research", _E, "official_sector", _INST,
     "https://www.financialresearch.gov",
     "US Treasury body publishing financial-stability research and free public data "
     "series, including money-market and repo monitors.",
     ["macro", "risk_management"], ["fixed_income", "credit", "multi_asset"],
     ["research", "market_data", "regulatory", "risk_management"], [],
     {"aliases": ["OFR"],
      "notes": "Appeared TWICE in the seed list — under institutions AND under tools. "
               "Merged into one entity; it is genuinely both, so it carries both "
               "'research' and 'market_data' insight types."}),

    # ===================== PODCASTS (Phase 2 harvest targets) =====================
    ("Top Traders Unplugged", "podcast", "", _INST, "https://toptradersunplugged.com",
     "Long-running podcast on systematic investing, trend following and managed futures, "
     "spanning the Systematic Investor, Allocator and Investment Legends series.",
     ["systematic", "trend_following", "managed_futures", "global_macro"],
     ["derivatives", "multi_asset", "commodities"],
     ["podcasts", "interviews", "quantitative_research", "institutional_investing"],
     ["Niels Kaastrup-Larsen", "Alan Dunne", "Rich Brennan", "Mark Rzepczynski"],
     {"aliases": ["TTU"],
      "notes": "Seed list placed this under institutions; it is a podcast. Already ingested "
               "by Zenith as an RSS insight source (sources.py). Full archive confirmed: "
               "961 episodes back to 2014 via feeds.captivate.fm — Phase 2 harvest target."}),
    ("Flirting with Models", "podcast", "", _INST, "https://www.flirtingwithmodels.com",
     "Corey Hoffstein's podcast interviewing quantitative researchers and systematic "
     "portfolio managers about how they actually build their models.",
     ["quantitative", "systematic", "portfolio_construction", "factor_investing"],
     ["multi_asset", "equities"],
     ["podcasts", "interviews", "quantitative_research", "portfolio_construction"],
     ["Corey Hoffstein"],
     {"notes": "Published by Newfound Research. Full archive confirmed: 126 episodes via "
               "feeds.captivate.fm — Phase 2 harvest target."}),
    ("The Derivative", "podcast", "", _INST, "https://www.rcmalternatives.com/podcast/",
     "RCM Alternatives' podcast on managed futures, volatility, options and alternative "
     "investments, hosted by Jeff Malec.",
     ["managed_futures", "volatility", "options", "derivatives", "trend_following"],
     ["derivatives", "commodities", "volatility", "alternatives"],
     ["podcasts", "interviews", "alternative_investments", "options"], ["Jeff Malec"],
     {"aliases": ["The Derivative with Jeff Malec"],
      "notes": "Full archive confirmed: 237 episodes back to 2020 — Phase 2 harvest target."}),
    ("Alpha Exchange", "podcast", "", _INST, "https://alphaexchange.com",
     "Dean Curnutt's podcast on volatility, risk, market structure and the price of "
     "financial risk, featuring senior derivatives and macro practitioners.",
     ["volatility", "options", "derivatives", "risk_management", "market_microstructure"],
     ["volatility", "derivatives", "multi_asset"],
     ["podcasts", "interviews", "volatility", "options", "risk_management"], ["Dean Curnutt"],
     {"aliases": ["Alpha Exchange with Dean Curnutt"],
      "notes": "Published by Macro Risk Advisors. Full archive confirmed: 268 episodes back "
               "to 2018 — Phase 2 harvest target. Episode titles carry 'Name, Role, Firm' "
               "structure, the richest guest metadata of any monitored show."}),
    ("Excess Returns", "podcast", "", _INST, "https://excessreturnspod.com",
     "Jack Forehand and Justin Carbonneau's podcast on evidence-based and quantitative "
     "investing, factor strategies and investor behaviour.",
     ["quantitative", "factor_investing", "evidence_based", "value", "behavioral_finance"],
     ["equities", "multi_asset"],
     ["podcasts", "interviews", "quantitative_research", "behavioral_finance"],
     ["Jack Forehand", "Justin Carbonneau"],
     {"notes": "Full archive confirmed: 554 episodes back to 2019 — Phase 2 harvest target."}),
    ("Capital Allocators", "podcast", "", _INST, "https://capitalallocators.com",
     "Ted Seides' podcast on institutional investing — endowments, foundations, pensions, "
     "manager selection and the craft of allocating capital.",
     ["asset_allocation", "multi_strategy", "portfolio_construction"],
     ["multi_asset", "alternatives", "private_markets"],
     ["podcasts", "interviews", "institutional_investing", "manager_selection", "asset_allocation"],
     ["Ted Seides"],
     {"aliases": ["Capital Allocators with Ted Seides"],
      "notes": "Full archive confirmed: 820 episodes back to 2017 — Phase 2 harvest target."}),
    ("Other People's Money", "podcast", "", _INST, "https://otherpeoplesmoney.substack.com",
     "Max Wiethe's podcast on the business of asset management — how funds are actually "
     "built, raised and run.",
     ["multi_strategy", "discretionary"], ["alternatives", "multi_asset"],
     ["podcasts", "interviews", "institutional_investing", "manager_selection"], ["Max Wiethe"],
     {"aliases": ["Other People's Money with Max Wiethe", "OPM"],
      "notes": "Full archive confirmed: 76 episodes back to 2024 — Phase 2 harvest target. "
               "opmpod.com does not resolve; the live official home is the Substack "
               "(link sweep, 2026-09-01)."}),
    ("Rational Reminder", "podcast", "", _INST, "https://rationalreminder.ca",
     "Benjamin Felix and Cameron Passmore's evidence-based investing podcast, heavily "
     "grounded in academic asset-pricing literature and frequently interviewing the "
     "researchers themselves.",
     ["evidence_based", "factor_investing", "passive", "behavioral_finance", "asset_allocation"],
     ["equities", "multi_asset"],
     ["podcasts", "interviews", "academic_research", "financial_education", "behavioral_finance"],
     ["Benjamin Felix", "Cameron Passmore"],
     {"aliases": ["The Rational Reminder Podcast"],
      "notes": "Full archive confirmed: 447 episodes — Phase 2 harvest target. An unusually "
               "high proportion of guests are published academics, making this the highest-"
               "value show for the academic side of the graph."}),
    ("The Meb Faber Show", "podcast", "", _INST, "https://mebfaber.com/podcast/",
     "Meb Faber's podcast interviewing fund managers, allocators and researchers across "
     "quantitative and global value investing.",
     ["quantitative", "value", "trend_following", "asset_allocation"], ["multi_asset", "equities"],
     ["podcasts", "interviews", "investment_ideas", "asset_allocation"], ["Meb Faber"],
     {"notes": "Published by Cambria. Full archive confirmed: 715 episodes — Phase 2 target."}),
    ("Money Maze Podcast", "podcast", "", _INST, "https://www.moneymazepodcast.com",
     "Simon Brewer's podcast interviewing leading investors and business builders across "
     "public and private markets.",
     ["fundamental", "private_equity", "asset_allocation", "discretionary"],
     ["multi_asset", "private_markets", "equities"],
     ["podcasts", "interviews", "institutional_investing"], ["Simon Brewer"],
     {"aliases": ["Money Maze"],
      "notes": "Full archive confirmed: 241 episodes back to 2020 — Phase 2 harvest target."}),
    ("Monetary Matters", "podcast", "", _INST, "https://www.youtube.com/@monetarymatters",
     "Jack Farley's podcast on monetary plumbing, macro, rates and the mechanics of the "
     "financial system.",
     ["macro", "global_macro", "fixed_income"], ["fixed_income", "multi_asset", "fx"],
     ["podcasts", "interviews", "macroeconomics", "monetary_policy"], ["Jack Farley"],
     {"aliases": ["Monetary Matters with Jack Farley"],
      "notes": "Full archive confirmed: 297 episodes back to 2024 — Phase 2 harvest target. "
               "monetarymatters.net does not resolve; the live official channel is YouTube "
               "(link sweep, 2026-09-01)."}),
    ("Odd Lots", "podcast", "", _INST,
     "https://www.bloomberg.com/oddlots",
     "Joe Weisenthal and Tracy Alloway's Bloomberg podcast on the odd corners of markets "
     "and the real economy — supply chains, commodities, plumbing and policy.",
     ["macro", "commodities", "global_macro"], ["multi_asset", "commodities", "fixed_income"],
     ["podcasts", "interviews", "macroeconomics", "news"],
     ["Joe Weisenthal", "Tracy Alloway"],
     {"notes": "Full archive confirmed: 1,265 episodes back to 2015 — the deepest archive of "
               "any monitored show, and a Phase 2 harvest target."}),
    ("The Long View", "podcast", "", _INST,
     "https://www.morningstar.com/podcasts/the-long-view",
     "Morningstar's podcast with Christine Benz and Dan Lefkovitz, interviewing investors, "
     "advisers and researchers on long-horizon investing and retirement.",
     ["evidence_based", "asset_allocation", "behavioral_finance", "passive"],
     ["multi_asset", "equities"],
     ["podcasts", "interviews", "financial_education", "asset_allocation"],
     ["Christine Benz", "Dan Lefkovitz"],
     {"notes": "Published by Morningstar. Full archive confirmed: 389 episodes back to 2019 "
               "— Phase 2 harvest target."}),
    ("COMPLEXITY", "podcast", "", _ACAD, "https://www.santafe.edu/culture/podcasts",
     "The Santa Fe Institute's podcast on complex systems science — networks, emergence, "
     "evolution and complexity economics.",
     ["behavioral_finance", "quantitative"], ["multi_asset"],
     ["podcasts", "interviews", "academic_research", "financial_education"], [],
     {"aliases": ["Complexity", "Complexity from the Santa Fe Institute"],
      "notes": "Published by the Santa Fe Institute. Archive confirmed: 119 episodes back to "
               "2019 — Phase 2 harvest target. Directly relevant to Zenith's own CAS "
               "(Complex Adaptive Systems) tab."}),

    # ===================== ACADEMIC / RESEARCH =====================
    ("SSRN", "academic_source", "", _ACAD, "https://www.ssrn.com",
     "The Social Science Research Network — the primary preprint repository for finance "
     "and economics working papers.",
     [], ["multi_asset"], ["academic_research", "research"], [],
     {"notes": "Cannot be ingested by Zenith: SSRN is login-gated (documented in "
               "sources.py). arXiv q-fin is used as the substitute open repository."}),
    ("NBER", "academic_source", "", _ACAD, "https://www.nber.org",
     "The National Bureau of Economic Research — publisher of the canonical US working "
     "paper series and the official arbiter of US business-cycle (recession) dating.",
     ["macro"], ["multi_asset"], ["academic_research", "macroeconomics", "market_history"], [],
     {"aliases": ["National Bureau of Economic Research"],
      "notes": "Already ingested by Zenith (sources.py). Its USREC recession series is the "
               "external label Zenith's REGIMES tab calibrates itself against."}),
    ("CEPR", "academic_source", "", _ACAD, "https://cepr.org",
     "The Centre for Economic Policy Research — European working paper series and the "
     "VoxEU commentary portal.",
     ["macro"], ["multi_asset"], ["academic_research", "macroeconomics", "fiscal_policy"], [],
     {"aliases": ["Centre for Economic Policy Research", "VoxEU"]}),
    ("Santa Fe Institute", _E, "think_tank", _ACAD, "https://www.santafe.edu",
     "Independent research institute for complex systems science; the intellectual home of "
     "complexity economics and the publisher of the COMPLEXITY podcast.",
     ["behavioral_finance", "quantitative"], ["multi_asset"],
     ["academic_research", "podcasts", "financial_education"],
     ["W. Brian Arthur", "Doyne Farmer"],
     {"aliases": ["SFI"], "founded": "1984", "location": "Santa Fe, NM",
      "notes": "Conceptually foundational to Zenith's own CAS tab."}),
    ("Mercatus Center", _E, "think_tank", _ACAD, "https://www.mercatus.org",
     "Research centre at George Mason University; hosts David Beckworth's Macro Musings "
     "and publishes monetary-policy research.",
     ["macro"], ["multi_asset", "fixed_income"],
     ["academic_research", "macroeconomics", "monetary_policy", "podcasts"],
     ["David Beckworth", "Tyler Cowen"],
     {"aliases": ["Mercatus Center at George Mason University"],
      "notes": "Probed for RSS during an earlier Zenith sweep; feed returned empty."}),
    ("Journal of Finance", "academic_source", "", _ACAD,
     "https://onlinelibrary.wiley.com/journal/15406261",
     "The flagship journal of the American Finance Association — the top-ranked academic "
     "finance journal.",
     [], ["multi_asset"], ["academic_research"], [],
     {"notes": "Already ingested by Zenith as a table-of-contents feed (sources.py)."}),
    ("Journal of Financial Economics", "academic_source", "", _ACAD,
     "https://www.sciencedirect.com/journal/journal-of-financial-economics",
     "Top-tier empirical finance journal; the original home of much of the factor "
     "asset-pricing literature.",
     ["factor_investing"], ["multi_asset"], ["academic_research"], [],
     {"aliases": ["JFE"],
      "notes": "Already ingested by Zenith as a table-of-contents feed (sources.py)."}),
    ("Review of Financial Studies", "academic_source", "", _ACAD,
     "https://academic.oup.com/rfs",
     "Top-three academic finance journal published by the Society for Financial Studies.",
     [], ["multi_asset"], ["academic_research"], [], {"aliases": ["RFS"]}),
    ("Journal of Portfolio Management", "academic_source", "", _ACAD,
     "https://www.pm-research.com/content/iijpormgmt",
     "Practitioner-facing journal on portfolio construction, asset allocation and "
     "quantitative strategy.",
     ["portfolio_construction", "asset_allocation", "factor_investing"], ["multi_asset"],
     ["academic_research", "portfolio_construction", "asset_allocation"], [],
     {"aliases": ["JPM", "JoPM"],
      "notes": "Probed for RSS during an earlier Zenith sweep; pm-research feeds returned empty."}),
    ("Financial Analysts Journal", "academic_source", "", _ACAD,
     "https://www.tandfonline.com/toc/ufaj20/current",
     "CFA Institute's practitioner-academic journal bridging investment research and "
     "portfolio practice.",
     ["factor_investing", "portfolio_construction"], ["multi_asset"],
     ["academic_research", "research"], [],
     {"aliases": ["FAJ"],
      "notes": "Already ingested by Zenith as a table-of-contents feed (sources.py)."}),
    ("Review of Asset Pricing Studies", "academic_source", "", _ACAD,
     "https://academic.oup.com/raps",
     "Asset-pricing journal published by the Society for Financial Studies.",
     ["factor_investing"], ["multi_asset"], ["academic_research"], [], {"aliases": ["RAPS"]}),
    ("Journal of Empirical Finance", "academic_source", "", _ACAD,
     "https://www.sciencedirect.com/journal/journal-of-empirical-finance",
     "Journal focused on empirical and econometric research in finance.",
     ["quantitative"], ["multi_asset"], ["academic_research", "quantitative_research"], [], {}),
    ("Quantitative Finance", "academic_source", "", _ACAD,
     "https://www.tandfonline.com/toc/rquf20/current",
     "Journal covering mathematical and computational finance, derivatives pricing and "
     "market microstructure.",
     ["quantitative", "derivatives", "market_microstructure"], ["derivatives", "multi_asset"],
     ["academic_research", "quantitative_research"], [],
     {"notes": "Already ingested by Zenith as a table-of-contents feed (sources.py)."}),
    ("Journal of Financial Data Science", "academic_source", "", _ACAD,
     "https://www.pm-research.com/content/iijjfds",
     "Journal on machine learning, alternative data and data science applied to investment "
     "management.",
     ["quantitative", "systematic"], ["multi_asset"],
     ["academic_research", "quantitative_research"], [], {}),
    ("Journal of Alternative Investments", "academic_source", "", _ACAD,
     "https://www.pm-research.com/content/iijaltinv",
     "Journal covering hedge funds, managed futures, private markets and other "
     "alternative strategies.",
     ["managed_futures", "multi_strategy", "private_equity"], ["alternatives", "private_markets"],
     ["academic_research", "alternative_investments"], [], {"aliases": ["JAI"]}),
    ("Journal of Risk", "academic_source", "", _ACAD, "",
     "Journal on risk measurement, management and regulation, published by Risk.net.",
     ["risk_management", "quantitative"], ["multi_asset"],
     ["academic_research", "risk_management"], [],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "notes": "Every risk.net path probed on 2026-09-01 returned 404, so no landing page "
               "is asserted. The journal exists; its current URL needs confirming by hand."}),

    # ===================== TOOLS =====================
    ("Koyfin", "tool", "", _TOOL, "https://www.koyfin.com",
     "Market data and charting platform covering equities, macro series, fundamentals and "
     "cross-asset dashboards; a widely-used low-cost Bloomberg alternative.",
     [], ["multi_asset", "equities", "fixed_income"],
     ["market_data", "screening", "fundamental_analysis", "portfolio_monitoring"], [], {}),
    ("Market Chameleon", "tool", "", _TOOL, "https://marketchameleon.com",
     "Options analytics platform: implied vs realised volatility, earnings-move statistics, "
     "unusual options activity and volatility screening.",
     ["options", "volatility", "derivatives"], ["volatility", "derivatives", "equities"],
     ["options", "volatility", "screening", "market_data"], [],
     {"aliases": ["MarketChameleon"]}),
    ("Barchart", "tool", "", _TOOL, "https://www.barchart.com",
     "Market data platform covering futures, options and equities, with extensive free "
     "screeners and commodity data.",
     ["derivatives", "commodities"], ["commodities", "derivatives", "equities"],
     ["market_data", "screening"], [], {"aliases": ["Barcharts"]}),
    ("TradingView", "tool", "", _TOOL, "https://www.tradingview.com",
     "Charting and social analysis platform with a large scripting community (Pine Script) "
     "and broad cross-asset coverage.",
     [], ["multi_asset", "equities", "crypto"],
     ["market_data", "technical_analysis", "screening", "backtesting"], [], {}),
    ("Finviz", "tool", "", _TOOL, "https://finviz.com",
     "Fast equity screener and market-map visualiser with fundamental and technical filters.",
     [], ["equities"], ["screening", "market_data", "technical_analysis"], [], {}),
    ("OptionStrat", "tool", "", _TOOL, "https://optionstrat.com",
     "Options strategy builder and payoff visualiser with unusual-activity flow scanning.",
     ["options", "derivatives"], ["derivatives", "volatility", "equities"],
     ["options", "screening", "trading"], [], {}),
    ("OptionCharts", "tool", "", _TOOL, "https://optioncharts.io",
     "Options data visualisation platform covering chains, volatility surfaces, open "
     "interest and skew.",
     ["options", "volatility"], ["derivatives", "volatility", "equities"],
     ["options", "volatility", "market_data"], [], {"aliases": ["OptionCharts.io"]}),
    ("Portfolio Visualizer", "tool", "", _TOOL, "https://www.portfoliovisualizer.com",
     "Portfolio backtesting and analytics platform: asset allocation backtests, factor "
     "regressions, Monte Carlo and optimisation.",
     ["asset_allocation", "portfolio_construction", "factor_investing"], ["multi_asset"],
     ["backtesting", "portfolio_construction", "asset_allocation", "portfolio_monitoring"], [], {}),
    ("Portfolio Charts", "tool", "", _TOOL, "https://portfoliocharts.com",
     "Long-horizon asset-allocation visualisation site, known for its distinctive charts of "
     "portfolio outcome distributions across decades.",
     ["asset_allocation", "portfolio_construction"], ["multi_asset"],
     ["asset_allocation", "portfolio_construction", "financial_education", "market_history"], [],
     {}),
    ("testfol.io", "tool", "", _TOOL, "https://testfol.io",
     "Free portfolio backtester for ETFs and asset allocations, with leverage and "
     "synthetic-series modelling.",
     ["asset_allocation", "portfolio_construction"], ["multi_asset", "equities"],
     ["backtesting", "portfolio_construction", "asset_allocation"], [],
     {"aliases": ["testfolio", "Testfol.io"]}),
    ("Foliolytic", "tool", "", _TOOL, "https://foliolytic.com",
     "Free portfolio analyser computing 70+ quantitative metrics on user portfolios.",
     ["portfolio_construction", "risk_management", "quantitative"], ["multi_asset"],
     ["portfolio_monitoring", "portfolio_construction", "risk_management"], [],
     {"notes": "Identity confirmed by probing foliolytic.com on 2026-09-01 (site title: "
               "'Foliolytic — Free Portfolio Analyzer | 70+ Quant Metrics')."}),
    ("FactorsToday", "tool", "", _TOOL, "https://www.factorstoday.com",
     "Quantitative factor analysis platform providing modern factor exposure and "
     "performance analytics.",
     ["factor_investing", "quantitative"], ["equities", "multi_asset"],
     ["quantitative_research", "screening", "portfolio_monitoring"], [],
     {"notes": "Identity confirmed by probing factorstoday.com on 2026-09-01 (site title: "
               "'Modern Quantitative Analysis | FactorsToday')."}),
    ("Open Source Asset Pricing", "tool", "", _ACAD, "https://www.openassetpricing.com",
     "Chen & Zimmermann's open replication of the published cross-sectional asset-pricing "
     "literature — downloadable return series for 200+ documented anomalies.",
     ["factor_investing", "quantitative", "systematic"], ["equities"],
     ["academic_research", "quantitative_research", "backtesting", "market_data"],
     ["Andrew Chen", "Tom Zimmermann"],
     {"aliases": ["OSAP", "Open Source Asset Pricing (Chen-Zimmermann)"],
      "notes": "Directly used by Zenith: the openassetpricing package is a declared "
               "dependency and powers FACTOR MOMENTUM's academic 65-factor set."}),
    ("Citrindex", "tool", "", _TOOL, "https://www.citrindex.com",
     "Market index and analytics service.",
     [], ["multi_asset"], ["index_data", "market_data"], [],
     {"confidence": "low", "lifecycle_state": "needs_review",
      "notes": "Domain resolves and is live (probed 2026-09-01, page title 'Citrindex'), but "
               "the site gave too little detail to describe the service confidently. Flagged "
               "for review rather than described speculatively."}),
    ("SG CTA Trend Indicator", "tool", "", _TOOL,
     "https://wholesale.banking.societegenerale.com/en/prime-services-indices/",
     "Société Générale's family of CTA and trend-following benchmark indices — the "
     "industry-standard reference for managed-futures performance.",
     ["trend_following", "managed_futures", "systematic"], ["derivatives", "commodities"],
     ["index_data", "market_data", "research"], [],
     {"aliases": ["SG CTA Index", "SG Trend Index", "SG CTA Trend Indicator Daily Report"],
      "notes": "Seed list referred to a 'daily report'. SG publishes the SG CTA / SG Trend "
               "index family from its Prime Services indices pages; the exact daily-report "
               "URL may sit behind client access."}),
    ("Forex Factory", "tool", "", _TOOL, "https://www.forexfactory.com",
     "FX-focused community site whose economic calendar is widely used for tracking "
     "high-impact macro data releases.",
     ["macro"], ["fx", "multi_asset"], ["market_data", "macroeconomics", "news"], [],
     {"aliases": ["ForexFactory"]}),
    ("CME FedWatch", "tool", "", _TOOL,
     "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
     "CME's tool inferring implied Federal Reserve policy-rate probabilities from fed funds "
     "futures pricing.",
     ["macro", "fixed_income"], ["fixed_income", "derivatives"],
     ["monetary_policy", "market_data"], [],
     {"aliases": ["FedWatch"],
      "notes": "Listed separately from CME Group because it is used as a standalone tool; "
               "linked to CME Group by a subsidiary_of edge."}),
    ("Yahoo Finance", "tool", "", _TOOL, "https://finance.yahoo.com",
     "Free market data portal; its undocumented price API underpins a great deal of "
     "retail and research tooling.",
     [], ["multi_asset", "equities"], ["market_data", "news", "screening"], [],
     {"notes": "Directly relevant to Zenith: yfinance (which wraps Yahoo's endpoints) is the "
               "price source for CAS, MOMENTUM, PEAD, IDEAS and more. Its occasional data "
               "gaps have caused real Zenith bugs."}),
    ("Google Finance", "tool", "", _TOOL, "https://www.google.com/finance",
     "Free market data portal and quote service.",
     [], ["multi_asset", "equities"], ["market_data", "news"], [], {}),
]

# ---------------------------------------------------------------------------
# People. Every one of these comes from the seed list itself — either as a
# named row or inside a "(E.G., ...)" parenthetical — so the affiliations below
# are the user's own, not inferred.
#
# Tuple shape: (name, current_affiliation, role, url, description,
#               approach[], insight_types[], extra{})
# ---------------------------------------------------------------------------
PEOPLE: list[tuple] = [
    ("Robert Carver", "Independent (own capital)", "Systematic trader, author and blogger",
     "https://qoppac.blogspot.com",
     "Former institutional systematic trader (AHL) who now runs his own capital and writes "
     "one of the most detailed public bodies of work on systematic trading system design — "
     "position sizing, risk targeting, forecast combination and portfolio construction for "
     "futures traders.",
     ["systematic", "trend_following", "managed_futures", "quantitative", "risk_management"],
     ["quantitative_research", "financial_education", "books", "research"],
     {"aliases": ["Rob Carver"],
      "historical_affiliations": ["Man AHL", "Barclays"],
      "asset_classes": ["derivatives", "commodities", "fx", "fixed_income", "equities"],
      "books": ["Systematic Trading", "Smart Portfolios", "Leveraged Trading",
                "Advanced Futures Trading Strategies"],
      "notes": "The worked reference example for this directory. His blog (Investment Idiocy "
               "at qoppac.blogspot.com) is already ingested by Zenith as the 'Robert Carver "
               "(Systematic)' RSS feed. He is a recurring Top Traders Unplugged guest — those "
               "episode-level links arrive with the Phase 2 harvest.",
      "confidence": "high"}),
    ("Andrew Beer", "Dynamic Beta Investments", "Co-founder and Managing Member",
     "https://dbi.co",
     "Co-founder of Dynamic Beta Investments and architect of the managed-futures "
     "replication approach behind the DBMF ETF; a frequent commentator on hedge fund fees "
     "and replication.",
     ["replication", "managed_futures", "systematic"],
     ["research", "alternative_investments", "interviews"],
     {"asset_classes": ["derivatives", "alternatives"],
      "notes": "Named in the seed list's own parenthetical for Dynamic Beta Investments. "
               "DBMF, which his firm sub-advises, is the fund tracked in Zenith's HOLDINGS tab.",
      "confidence": "high"}),
    ("Katy Kaminski", "AlphaSimplex Group", "Chief Research Strategist",
     "https://www.alphasimplex.com",
     "Researcher and portfolio manager specialising in trend following and crisis alpha; "
     "co-author of 'Trend Following with Managed Futures'.",
     ["trend_following", "managed_futures", "systematic", "quantitative"],
     ["quantitative_research", "academic_research", "books", "interviews"],
     {"asset_classes": ["derivatives", "commodities", "multi_asset"],
      "books": ["Trend Following with Managed Futures"],
      "notes": "Named in the seed list's own parenthetical for AlphaSimplex.",
      "confidence": "high"}),
    ("Mike Green", "Tier1 Alpha", "Researcher and strategist",
     "https://tier1alpha.com",
     "Market structure researcher best known for work on the effects of passive-flow growth "
     "on price formation and volatility.",
     ["market_microstructure", "volatility", "passive", "macro"],
     ["research", "positioning", "volatility", "interviews"],
     {"asset_classes": ["equities", "volatility"],
      "notes": "Named in the seed list's own parenthetical for Tier1 Alpha.",
      "confidence": "medium"}),
    ("Craig Peterson", "Tier1 Alpha", "Researcher", "https://tier1alpha.com",
     "Researcher at Tier1 Alpha working on market structure and dealer-positioning analytics.",
     ["market_microstructure", "volatility", "options"],
     ["research", "positioning", "options"],
     {"asset_classes": ["equities", "volatility", "derivatives"],
      "notes": "Named in the seed list's own parenthetical for Tier1 Alpha. Only the "
               "affiliation is confirmed; role description is generic by necessity.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("David Pegler", "Tier1 Alpha", "Researcher", "https://tier1alpha.com",
     "Researcher at Tier1 Alpha working on market structure and dealer-positioning analytics.",
     ["market_microstructure", "volatility", "options"],
     ["research", "positioning", "options"],
     {"asset_classes": ["equities", "volatility", "derivatives"],
      "notes": "Named in the seed list's own parenthetical for Tier1 Alpha. Only the "
               "affiliation is confirmed; role description is generic by necessity.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("Kris Sidial", "The Ambrus Group", "Co-Chief Investment Officer", "",
     "Volatility trader focused on tail-risk and convex equity volatility strategies.",
     ["volatility", "options", "derivatives"],
     ["volatility", "options", "trading", "interviews"],
     {"asset_classes": ["volatility", "derivatives", "equities"],
      "notes": "Named in the seed list's own parenthetical for The Ambros Group. Firm URL "
               "unconfirmed, so no personal URL is asserted either.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("Noel Smith", "Convex Asset Management", "Chief Investment Officer", "",
     "Options and volatility trader.",
     ["volatility", "options", "derivatives"], ["volatility", "options", "trading"],
     {"asset_classes": ["volatility", "derivatives"],
      "notes": "Named in the seed list's own parenthetical for Convex Asset Management. "
               "Firm URL unconfirmed, so no personal URL is asserted either.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("Alan Dunne", "Archive Capital Advisors", "Founder and Managing Director",
     "https://www.archivecapitaladvisors.ie",
     "Investment adviser focused on institutional asset allocation and alternatives; "
     "co-hosts the Allocator series on Top Traders Unplugged.",
     ["asset_allocation", "managed_futures", "global_macro", "portfolio_construction"],
     ["asset_allocation", "institutional_investing", "podcasts", "interviews"],
     {"asset_classes": ["multi_asset", "alternatives"],
      "notes": "Named in the seed list's own parenthetical for Archive Capital Advisors.",
      "confidence": "high"}),
    ("Rich Brennan", "ATS Trading Solutions", "Founder",
     "https://www.atstradingsolutions.com",
     "Systematic trend-following practitioner and educator; publishes a reading list and "
     "trading-research library, and contributes to Top Traders Unplugged.",
     ["trend_following", "systematic", "managed_futures"],
     ["financial_education", "trading", "podcasts", "books"],
     {"aliases": ["Richard Brennan"], "asset_classes": ["derivatives", "commodities"],
      "notes": "Named in the seed list's own parenthetical for ATS Trading Solutions "
               "(alongside 'READING LIST + THE VAULT'). The domain is live but returns a "
               "Cloudflare anti-bot challenge to automated requests (probed 2026-09-01), so "
               "the link checker will report it BLOCKED rather than broken.",
      "confidence": "medium"}),
    ("Yoav Git", "Gresham Investment Management", "Quantitative researcher", "",
     "Quantitative researcher publishing on systematic commodity strategies through "
     "GreshamQuant.",
     ["quantitative", "systematic", "commodities"],
     ["quantitative_research", "research"],
     {"asset_classes": ["commodities"],
      "notes": "Named in the seed list's own parenthetical for GreshamQuant.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("Moritz Seibert", "Aquantum", "Systematic trader and researcher", "",
     "Systematic futures trader and writer on trend following and managed futures.",
     ["trend_following", "systematic", "managed_futures"],
     ["trading", "research", "podcasts"],
     {"asset_classes": ["derivatives", "commodities"],
      "notes": "Named in the seed list's own parenthetical for Aquantum. Firm URL "
               "unconfirmed, so no personal URL is asserted.",
      "confidence": "low", "lifecycle_state": "needs_review"}),
    ("Marat Molyboga", "Efficient Capital Management", "Chief Risk Officer and Director of Research",
     "https://www.efficient.com",
     "Researcher on managed-futures manager selection, portfolio construction and hedge "
     "fund performance measurement; a frequent academic-journal author.",
     ["managed_futures", "portfolio_construction", "risk_management", "quantitative"],
     ["academic_research", "manager_selection", "quantitative_research"],
     {"asset_classes": ["derivatives", "alternatives"],
      "notes": "Named in the seed list's own parenthetical for Efficient Capital Markets.",
      "confidence": "medium"}),
    ("David Beckworth", "Mercatus Center", "Senior Research Fellow",
     "https://www.mercatus.org/people/david-beckworth",
     "Monetary economist and host of the Macro Musings podcast; writes on nominal GDP "
     "targeting and monetary policy frameworks.",
     ["macro"], ["macroeconomics", "monetary_policy", "podcasts", "academic_research"],
     {"asset_classes": ["fixed_income", "multi_asset"],
      "notes": "Named in the seed list's own parenthetical for the Mercatus Center.",
      "confidence": "medium"}),
    ("Svante Bergström", "Lynx Asset Management", "Co-founder and CEO",
     "https://www.lynxhedge.se",
     "Co-founder of Lynx Asset Management, a Swedish systematic trend-following manager.",
     ["systematic", "trend_following", "managed_futures"],
     ["trading", "quantitative_research", "interviews"],
     {"aliases": ["Svante Bergstrom"], "asset_classes": ["derivatives", "commodities", "fx"],
      "notes": "Named in the seed list's own parenthetical for Lynx Asset Management.",
      "confidence": "medium"}),
    ("Torsten Sløk", "Apollo Global Management", "Chief Economist",
     "https://www.apolloacademy.com",
     "Chief Economist at Apollo and author of the Daily Spark, a widely-circulated free "
     "daily macro chartbook.",
     ["macro"], ["macroeconomics", "research", "market_data"],
     {"aliases": ["Torsten Slok"], "historical_affiliations": ["Deutsche Bank"],
      "asset_classes": ["multi_asset"],
      "notes": "The Daily Spark appeared as its own row in the seed list; it is his "
               "publication, so it is folded into Apollo Academy with him as the author.",
      "confidence": "high"}),
    ("Corey Hoffstein", "Newfound Research", "Co-founder and Chief Investment Officer",
     "https://www.thinknewfound.com",
     "Quantitative researcher and host of Flirting with Models; known for work on "
     "rebalance timing luck, trend following and return stacking.",
     ["quantitative", "systematic", "portfolio_construction", "trend_following"],
     ["quantitative_research", "portfolio_construction", "podcasts", "research"],
     {"asset_classes": ["multi_asset", "equities"],
      "notes": "Host of Flirting with Models, one of the podcasts INDEX monitors.",
      "confidence": "high"}),
    ("Meb Faber", "Cambria Investment Management", "Co-founder and Chief Investment Officer",
     "https://mebfaber.com",
     "Quantitative investor, author and host of The Meb Faber Show; known for early public "
     "work on tactical asset allocation and shareholder yield.",
     ["quantitative", "value", "trend_following", "asset_allocation"],
     ["quantitative_research", "podcasts", "books", "research"],
     {"asset_classes": ["equities", "multi_asset"],
      "books": ["Global Asset Allocation", "The Ivy Portfolio", "Shareholder Yield"],
      "notes": "Host of The Meb Faber Show. His blog is already ingested by Zenith "
               "(sources.py).",
      "confidence": "high"}),
    ("Niels Kaastrup-Larsen", "Top Traders Unplugged", "Founder and host",
     "https://toptradersunplugged.com",
     "Host and founder of Top Traders Unplugged, and a managed-futures practitioner.",
     ["trend_following", "managed_futures", "systematic"],
     ["podcasts", "interviews", "trading"],
     {"asset_classes": ["derivatives", "commodities"],
      "notes": "Host of Top Traders Unplugged, the deepest-archive podcast INDEX monitors.",
      "confidence": "high"}),
    ("Ted Seides", "Capital Allocators", "Founder and host",
     "https://capitalallocators.com",
     "Host of Capital Allocators and a former institutional allocator; writes on manager "
     "selection and the craft of allocating capital.",
     ["asset_allocation", "multi_strategy"],
     ["podcasts", "interviews", "institutional_investing", "manager_selection", "books"],
     {"asset_classes": ["multi_asset", "alternatives"],
      "historical_affiliations": ["Protégé Partners", "Yale University Investments Office"],
      "books": ["Capital Allocators", "So You Want to Start a Hedge Fund"],
      "confidence": "high"}),
    ("Dean Curnutt", "Macro Risk Advisors", "Founder and CEO",
     "https://alphaexchange.com",
     "Founder of Macro Risk Advisors and host of Alpha Exchange, focused on volatility, "
     "risk pricing and market structure.",
     ["volatility", "options", "risk_management", "derivatives"],
     ["podcasts", "interviews", "volatility", "options"],
     {"asset_classes": ["volatility", "derivatives"], "confidence": "high"}),
    ("Jeff Malec", "RCM Alternatives", "CEO and host",
     "https://www.rcmalternatives.com",
     "CEO of RCM Alternatives and host of The Derivative, covering managed futures, "
     "volatility and alternative investments.",
     ["managed_futures", "volatility", "derivatives"],
     ["podcasts", "interviews", "alternative_investments"],
     {"asset_classes": ["derivatives", "commodities", "alternatives"], "confidence": "high"}),
    ("Jack Forehand", "Validea Capital Management", "Co-founder and host",
     "https://excessreturnspod.com",
     "Quantitative investor and co-host of Excess Returns, focused on evidence-based and "
     "factor investing.",
     ["quantitative", "factor_investing", "evidence_based", "value"],
     ["podcasts", "interviews", "quantitative_research"],
     {"asset_classes": ["equities"], "confidence": "medium"}),
    ("Justin Carbonneau", "Validea Capital Management", "Partner and host",
     "https://excessreturnspod.com",
     "Co-host of Excess Returns, writing on quantitative and factor-based investing.",
     ["quantitative", "factor_investing", "evidence_based"],
     ["podcasts", "interviews", "quantitative_research"],
     {"asset_classes": ["equities"], "confidence": "medium"}),
    ("Max Wiethe", "Other People's Money", "Founder and host",
     "https://otherpeoplesmoney.substack.com",
     "Host of Other People's Money, covering how asset management businesses are built "
     "and run.",
     ["multi_strategy"], ["podcasts", "interviews", "institutional_investing"],
     {"asset_classes": ["alternatives"], "confidence": "medium"}),
    ("Benjamin Felix", "PWL Capital", "Chief Investment Officer and host",
     "https://rationalreminder.ca",
     "Evidence-based investing communicator and co-host of the Rational Reminder podcast; "
     "translates academic asset-pricing research for practitioners.",
     ["evidence_based", "factor_investing", "passive", "asset_allocation"],
     ["podcasts", "interviews", "academic_research", "financial_education"],
     {"asset_classes": ["equities", "multi_asset"], "confidence": "high"}),
    ("Cameron Passmore", "PWL Capital", "Portfolio Manager and host",
     "https://rationalreminder.ca",
     "Co-host of the Rational Reminder podcast and portfolio manager at PWL Capital.",
     ["evidence_based", "passive", "asset_allocation"],
     ["podcasts", "interviews", "financial_education"],
     {"asset_classes": ["equities", "multi_asset"], "confidence": "high"}),
    ("Simon Brewer", "Money Maze Podcast", "Host",
     "https://www.moneymazepodcast.com",
     "Host of the Money Maze Podcast; a former CIO interviewing leading investors across "
     "public and private markets.",
     ["fundamental", "asset_allocation", "discretionary"],
     ["podcasts", "interviews", "institutional_investing"],
     {"asset_classes": ["multi_asset"], "confidence": "medium"}),
    ("Jack Farley", "Monetary Matters", "Founder and host",
     "https://www.youtube.com/@monetarymatters",
     "Host of Monetary Matters, covering monetary plumbing, rates and macro mechanics.",
     ["macro", "global_macro", "fixed_income"],
     ["podcasts", "interviews", "macroeconomics", "monetary_policy"],
     {"asset_classes": ["fixed_income", "multi_asset"],
      "historical_affiliations": ["Blockworks"], "confidence": "medium"}),
    ("Joe Weisenthal", "Bloomberg", "Co-host, Odd Lots",
     "https://www.bloomberg.com/oddlots",
     "Bloomberg journalist and co-host of Odd Lots.",
     ["macro"], ["podcasts", "interviews", "macroeconomics", "news"],
     {"asset_classes": ["multi_asset"], "confidence": "high"}),
    ("Tracy Alloway", "Bloomberg", "Co-host, Odd Lots",
     "https://www.bloomberg.com/oddlots",
     "Bloomberg journalist and co-host of Odd Lots.",
     ["macro"], ["podcasts", "interviews", "macroeconomics", "news"],
     {"asset_classes": ["multi_asset"], "confidence": "high"}),
    ("Christine Benz", "Morningstar", "Director of Personal Finance and Retirement Planning",
     "https://www.morningstar.com/podcasts/the-long-view",
     "Morningstar's personal finance lead and co-host of The Long View; writes on "
     "retirement planning and portfolio withdrawal strategy.",
     ["asset_allocation", "evidence_based", "passive"],
     ["podcasts", "interviews", "financial_education", "asset_allocation", "books"],
     {"asset_classes": ["multi_asset"], "confidence": "high"}),
    ("Dan Lefkovitz", "Morningstar", "Strategist and co-host, The Long View",
     "https://www.morningstar.com/podcasts/the-long-view",
     "Morningstar Indexes strategist and co-host of The Long View.",
     ["passive", "factor_investing"], ["podcasts", "interviews", "index_data"],
     {"asset_classes": ["equities", "multi_asset"], "confidence": "medium"}),
    ("Cliff Asness", "AQR Capital Management", "Founder, Managing Principal and CIO",
     "https://www.aqr.com",
     "Co-founder of AQR and a prolific writer on value, momentum, factor investing and the "
     "behaviour of markets and investors.",
     ["quantitative", "factor_investing", "value", "momentum", "systematic"],
     ["academic_research", "quantitative_research", "research", "behavioral_finance"],
     {"asset_classes": ["equities", "multi_asset"],
      "historical_affiliations": ["Goldman Sachs Asset Management"], "confidence": "high"}),
    ("Andrew Lo", "MIT Sloan School of Management", "Professor of Finance",
     "https://alo.mit.edu",
     "MIT finance professor, author of the Adaptive Markets Hypothesis, and founder of "
     "AlphaSimplex.",
     ["quantitative", "systematic", "behavioral_finance"],
     ["academic_research", "quantitative_research", "books"],
     {"asset_classes": ["multi_asset"], "books": ["Adaptive Markets", "A Non-Random Walk Down Wall Street"],
      "historical_affiliations": ["AlphaSimplex Group"], "confidence": "high"}),
    ("David Booth", "Dimensional Fund Advisors", "Founder and Chairman",
     "https://www.dimensional.com",
     "Founder of Dimensional Fund Advisors, the firm that translated Fama-French "
     "asset-pricing research into an investment business.",
     ["factor_investing", "systematic", "evidence_based", "passive"],
     ["academic_research", "research", "interviews"],
     {"asset_classes": ["equities", "fixed_income"], "confidence": "high"}),
    ("Rob Arnott", "Research Affiliates", "Founder and Chairman",
     "https://www.researchaffiliates.com",
     "Founder of Research Affiliates and originator of fundamental indexation (RAFI); "
     "writes on valuation-aware factor investing and long-horizon returns.",
     ["factor_investing", "value", "asset_allocation", "quantitative"],
     ["academic_research", "research", "asset_allocation"],
     {"asset_classes": ["equities", "multi_asset"], "confidence": "high"}),
    ("Wesley Gray", "Alpha Architect", "Founder and CEO",
     "https://alphaarchitect.com",
     "Founder of Alpha Architect; former Marine and finance PhD who publishes accessible "
     "summaries of academic asset-pricing research.",
     ["quantitative", "factor_investing", "value", "momentum", "evidence_based"],
     ["academic_research", "quantitative_research", "financial_education", "books"],
     {"asset_classes": ["equities"], "books": ["Quantitative Value", "Quantitative Momentum"],
      "confidence": "high"}),
]

# Explicit parent -> child brand/division relationships found in the seed list.
# Modelled as edges rather than merges because both names are genuinely used and
# a researcher may look for either.
SUBSIDIARY_EDGES: list[tuple[str, str, str]] = [
    ("Man Institute", "Man Group", "Man Group's research publication arm"),
    ("Man AHL", "Man Group", "Man Group's systematic managed-futures division"),
    ("iShares", "BlackRock", "BlackRock's ETF brand"),
    ("SPDR ETFs", "State Street Global Advisors", "State Street's ETF brand"),
    ("GreshamQuant", "Gresham Investment Management", "Gresham's quantitative research arm"),
    ("Apollo Academy", "Apollo Global Management", "Apollo's public research & education channel"),
    ("CME FedWatch", "CME Group", "A CME Group tool"),
    ("Citadel Securities", "Citadel", "Separate firm sharing a founder with the hedge fund"),
    ("J.P. Morgan Asset Management", "JPMorgan Chase",
     "JPMorgan's asset-management arm — publisher of the Guide to the Markets"),
]

# Podcast -> publishing organisation.
PUBLISHES_EDGES: list[tuple[str, str, str]] = [
    ("Newfound Research", "Flirting with Models", "Publishes the podcast"),
    ("Morningstar", "The Long View", "Publishes the podcast"),
    ("Santa Fe Institute", "COMPLEXITY", "Publishes the podcast"),
    ("Cambria Investment Management", "The Meb Faber Show", "Publishes the podcast"),
]

# Person -> podcast hosting relationships (Phase 2 adds the guest edges).
HOSTS_EDGES: list[tuple[str, str]] = [
    ("Niels Kaastrup-Larsen", "Top Traders Unplugged"),
    ("Alan Dunne", "Top Traders Unplugged"),
    ("Rich Brennan", "Top Traders Unplugged"),
    ("Corey Hoffstein", "Flirting with Models"),
    ("Jeff Malec", "The Derivative"),
    ("Dean Curnutt", "Alpha Exchange"),
    ("Jack Forehand", "Excess Returns"),
    ("Justin Carbonneau", "Excess Returns"),
    ("Ted Seides", "Capital Allocators"),
    ("Max Wiethe", "Other People's Money"),
    ("Benjamin Felix", "Rational Reminder"),
    ("Cameron Passmore", "Rational Reminder"),
    ("Meb Faber", "The Meb Faber Show"),
    ("Simon Brewer", "Money Maze Podcast"),
    ("Jack Farley", "Monetary Matters"),
    ("Joe Weisenthal", "Odd Lots"),
    ("Tracy Alloway", "Odd Lots"),
    ("Christine Benz", "The Long View"),
    ("Dan Lefkovitz", "The Long View"),
]

FOUNDED_EDGES: list[tuple[str, str]] = [
    ("Andrew Lo", "AlphaSimplex Group"),
    ("David Booth", "Dimensional Fund Advisors"),
    ("Rob Arnott", "Research Affiliates"),
    ("Wesley Gray", "Alpha Architect"),
    ("Cliff Asness", "AQR Capital Management"),
    ("Corey Hoffstein", "Newfound Research"),
    ("Meb Faber", "Cambria Investment Management"),
    ("Alan Dunne", "Archive Capital Advisors"),
    ("Andrew Beer", "Dynamic Beta Investments"),
    ("Svante Bergström", "Lynx Asset Management"),
]


# Entities whose official URL was ACTUALLY PROBED while authoring this file
# (2026-09-01), or that Zenith already ingests through a confirmed feed in
# sources.py. Only these default to `high` confidence.
#
# The distinction matters and is the reason this set exists rather than a blanket
# default: writing down a URL that looks right is not the same as checking it.
# Everything else defaults to `medium` — identity confident, official source
# asserted but not individually verified — and links.py then records the real
# HTTP result in `link_status`, which is a measurement rather than an opinion.
_PROBED_OR_INGESTED = {
    # probed directly during authoring
    "Citrindex", "Foliolytic", "FactorsToday", "Mount Lucas Management",
    "Efficient Capital Management", "HedgeNordic", "Open Source Asset Pricing",
    "testfol.io", "Portfolio Charts", "OptionCharts", "Market Chameleon", "Koyfin",
    "Tier1 Alpha", "Archive Capital Advisors", "Gresham Investment Management",
    "Dynamic Beta Investments", "Lynx Asset Management", "DUNN Capital Management",
    "Transtrend",
    # already ingested by Zenith through a confirmed feed (sources.py)
    "Alpha Architect", "Newfound Research", "Simplify Asset Management", "FactSet",
    "Cambria Investment Management", "Top Traders Unplugged", "NBER",
    "Bank for International Settlements", "J.P. Morgan Asset Management",
    "Journal of Finance", "Journal of Financial Economics", "Financial Analysts Journal",
    "Quantitative Finance",
}


def _org_entity(row: tuple) -> dict:
    (name, etype, subtype, category, url, desc, approach,
     assets, insights, people, extra) = row
    extra = dict(extra or {})
    if url:
        default_conf = "high" if name in _PROBED_OR_INGESTED else "medium"
    else:
        default_conf = "low"
    return m.make(
        name,
        entity_type=etype,
        org_subtype=subtype,
        primary_category=category,
        url=url,
        description=desc,
        investment_approach=approach,
        asset_classes=assets,
        insight_types=insights,
        key_people=people,
        provenance=extra.pop("provenance", SEED_PROVENANCE),
        confidence=extra.pop("confidence", default_conf),
        lifecycle_state=extra.pop("lifecycle_state", "new"),
        **extra,
    )


def _person_entity(row: tuple) -> dict:
    name, affiliation, role, url, desc, approach, insights, extra = row
    extra = dict(extra or {})
    return m.make(
        name,
        entity_type="person",
        primary_category=extra.pop("primary_category", "institutional"),
        url=url,
        description=desc,
        role=role,
        current_affiliation=affiliation,
        investment_approach=approach,
        insight_types=insights,
        provenance=extra.pop("provenance", SEED_PROVENANCE),
        confidence=extra.pop("confidence", "medium"),
        lifecycle_state=extra.pop("lifecycle_state", "new"),
        **extra,
    )


def build() -> tuple[list[dict], list[dict]]:
    """Build the seed entity list and its relationship edges.

    Returns ``(entities, relationships)``. Deduplication is NOT done here —
    ``dedupe.py`` owns that, so the seed stays a plain declaration of what was
    supplied and the merge logic stays testable in one place.
    """
    entities = [_org_entity(r) for r in ENTRIES]
    entities += [_person_entity(r) for r in PEOPLE]

    by_name: dict[str, dict] = {}
    for ent in entities:
        by_name[ent["name"].lower()] = ent
        for alias in ent.get("aliases", []):
            by_name.setdefault(str(alias).lower(), ent)

    rels: list[dict] = []

    def _link(a: str, b: str, rel: str, note: str = "") -> None:
        ea, eb = by_name.get(a.lower()), by_name.get(b.lower())
        if ea and eb and ea["id"] != eb["id"]:
            rels.append(m.edge(ea["id"], eb["id"], rel, note))

    for child, parent, note in SUBSIDIARY_EDGES:
        _link(child, parent, "subsidiary_of", note)
    for publisher, work, note in PUBLISHES_EDGES:
        _link(publisher, work, "publishes", note)
    for person, show in HOSTS_EDGES:
        _link(person, show, "hosts")
    for person, org in FOUNDED_EDGES:
        _link(person, org, "founded")

    # works_at / worked_at, straight from each person's own affiliation fields
    for ent in entities:
        if ent["entity_type"] != "person":
            continue
        if ent.get("current_affiliation"):
            _link(ent["name"], ent["current_affiliation"], "works_at")
        for old in ent.get("historical_affiliations", []):
            _link(ent["name"], str(old), "worked_at")

    # key_people -> works_at, the reverse direction of the same fact. Only
    # created when the person is actually an entity in the catalog, so a name
    # mentioned in passing does not manufacture a dangling edge.
    seen = {(r["source"], r["target"], r["type"]) for r in rels}
    for ent in entities:
        for person in ent.get("key_people", []):
            pe = by_name.get(str(person).lower())
            if pe and pe.get("entity_type") == "person":
                key = (pe["id"], ent["id"], "works_at")
                if key not in seen:
                    seen.add(key)
                    rels.append(m.edge(pe["id"], ent["id"], "works_at"))
    return entities, rels

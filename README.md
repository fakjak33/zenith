# ZENITH — Daily Insights & Research Aggregator

> Scrapes credible, **free** financial **insights** and **academic research**
> every night, dedupes so nothing repeats, sorts insights into **text-only** vs
> **visual** (charts/tables), and keeps a browsable **archive**. Parallax vintage
> aesthetic. Links out to everything.

## How it works
- **Nightly GitHub Action** (`.github/workflows/scrape.yml`, ~02:30 UTC) runs
  `python -m zenith.scrape`, which fetches each source's feed, drops anything in
  the `seen` index, classifies insights, and commits the updated JSON in `data/`.
- The **Streamlit app** (`app.py`) just reads that JSON — TODAY, ARCHIVE, SOURCES.

## Run locally
```bash
.venv/Scripts/python -m zenith.scrape       # fetch + compile today's items
.venv/Scripts/python -m streamlit run app.py
```

## Tests
```bash
.venv/Scripts/python -m pytest tests/ -q
```

## Sources
`zenith/sources.py` is a registry (insight / research / news). Each entry is
either `kind="rss"` (its best free RSS/Atom feed) or `kind="html"` (an
insights/research **hub page** scraped for article links via `link_pattern`).
The SOURCES tab shows per-source status from the last run, including how each was
fetched (`rss` / `direct` / `apify`).

## Reaching "blocked" sources — hybrid fetcher
Many firms publish no RSS. For those, `kind="html"` sources are fetched in tiers
(`zenith/fetch.py::get_html`):
1. **Direct (free):** a realistic browser User-Agent. This alone unblocks many
   "403" firms — confirmed working for **Blackstone, Goldman Sachs, AQR, Man
   Institute, BlackRock**, which have no feed.
2. **Apify fallback (paid, budget-gated):** only when the direct tier is blocked.
   `zenith/apify.py` runs Apify's `website-content-crawler` and is gated on
   Apify's *own* reported monthly spend so it can never exceed the FREE plan.

### Apify setup
- Set an `APIFY_TOKEN` env var locally / Streamlit secret / GitHub repo secret
  (the nightly Action passes `secrets.APIFY_TOKEN`). No token ⇒ direct-only (free).
- Env knobs: `APIFY_CRAWLER=cheerio|playwright` (cheerio = cheapest, no browser),
  `APIFY_MONTHLY_BUDGET_USD` (default 4.0; stops calling Apify above this),
  `APIFY_RESIDENTIAL=1` (residential proxy for the hardest anti-bot SPAs — costs
  well above the free tier, off by default).
- Usage is written to `data/apify_usage.json` and printed each run so you can
  watch consumption against the $5/mo free credits.
- **robots.txt is respected before any Apify call** — robots-disallowed sources
  (e.g. KKR, Two Sigma `/insights`) are never scraped.

## Honest constraints
- **Not every firm has a free, machine-readable feed.** Hedge funds / market
  makers (Citadel, Jane Street, Bridgewater, …) publish little or nothing
  syndicated. **Citadel in particular is Cloudflare-walled** (every path returns
  403, even with a browser User-Agent) and **Deutsche Bank's "Chart of the Day"**
  lives behind a DB Research login — neither exposes a free feed, so both are
  registered but disabled with a note. Apollo's **"Daily Spark"** *is* covered —
  it rides in the main Apollo Academy feed.
- **~40 working sources** now → a few hundred fresh items/day. Coverage spans firm
  commentary (Apollo, Bespoke, Alpha Architect, Verdad, Newfound, Ritholtz,
  Calculated Risk, FT Alphaville, Damodaran, Carver, Klement, Meb Faber/Cambria,
  Simplify, plus **feed-less firms via the HTML tier: Blackstone, Goldman Sachs,
  AQR, Man Institute, BlackRock**), central-bank / academic research (NY Fed
  Liberty Street, St. Louis Fed, NBER, BIS, ECB, Fed FEDS/WP/Speeches, Bank
  Underground, arXiv q-fin) and journal TOC feeds (incl. J. Empirical Finance).
  The registry is easy to extend — add a `Source(...)` line.
- **Some sites resist everything affordable.** Citadel (Cloudflare), Research
  Affiliates and Morningstar (anti-bot SPAs) return nothing via the direct tier
  *or* Apify's datacenter proxy; they'd need Apify **residential** proxy, which
  costs well above the $5/mo free credits — so they're registered but disabled.
- **News is minimized** by design. Commercial headline feeds (Yahoo Finance,
  CNBC, Nasdaq) are disabled; only low-noise official notices (Fed press) remain,
  and the viewer tucks news into a collapsed expander.
- **robots.txt**: respected for article-page fetches (used to classify
  text-vs-visual). RSS/Atom **feed endpoints** are treated as syndication
  endpoints (as all feed readers do) and fetched directly.
- **Text vs visual** is a heuristic (feed media + a light page check for
  figures/tables/charts) — good, not perfect.
- We **link out** and store only title/link/source/date/snippet — no republished
  full text. Free, non-paywalled content only.

## Disclaimer
Personal research aggregator. Respect each source's terms; content belongs to its
publishers. Not investment advice.

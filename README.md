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
`zenith/sources.py` is a feeds-first registry (insight / research / news). Each
entry has its best free RSS/Atom feed and an `enabled` flag. The SOURCES tab shows
per-source status from the last run.

## Honest constraints
- **Not every firm has a free, machine-readable feed.** Hedge funds / market
  makers (Citadel, Jane Street, Bridgewater, …) publish little or nothing
  syndicated. **Citadel in particular is Cloudflare-walled** (every path returns
  403, even with a browser User-Agent) and **Deutsche Bank's "Chart of the Day"**
  lives behind a DB Research login — neither exposes a free feed, so both are
  registered but disabled with a note. Apollo's **"Daily Spark"** *is* covered —
  it rides in the main Apollo Academy feed.
- **~30 working sources** now → a few hundred fresh items/day. Coverage spans firm
  commentary (Apollo, Bespoke, Alpha Architect, Verdad, Newfound, Ritholtz,
  Calculated Risk, FT Alphaville, Damodaran, Carver, Klement, …), central-bank /
  academic research (NY Fed Liberty Street, St. Louis Fed, NBER, BIS, ECB, Fed
  FEDS/WP/Speeches, Bank Underground, arXiv q-fin), and journal TOC feeds. The
  registry is easy to extend — add a `Source(...)` line.
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

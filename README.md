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
- **Not every firm has a free, machine-readable feed.** Many hedge funds / market
  makers (Citadel, Jane Street, Bridgewater, …) publish little or nothing
  syndicated, and some sites are JS-only or paywalled — those are registered but
  disabled with a note. Real daily volume comes from banks/asset managers with
  RSS, exchanges, **BIS / NBER / Fed / SSRN / journals (TOC)**, and news feeds.
  Currently ~17 working sources → a few hundred items/day; the registry is easy
  to extend.
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

# Phase 01 — Harvest the tastylive corpus
Read 00_master_brief.md and HANDOFF.md first.

Goal: a complete, re-runnable inventory of the target shows' episodes, with transcripts cached
locally, plus the written learning and mechanics pages.

Targets (chosen by the user):
1. Market Measures: the full archive.
2. Research shows: From Theory to Practice; Options Jive (including Research Corner segments);
   The Skinny on Options (every variant, e.g. Data Science, Math, Abstract Applications);
   Best Practices; Tasty Bites; any other research-team series you find. Confirm the final
   list at GATE A.
3. Written sources:
   - tastylive Learn Center (concepts & strategies, glossary);
   - research-style news/insights articles;
   - tastytrade help-center pages on margin requirements by strategy, assignment/exercise,
     rolling, account types (cash, IRA incl. limited margin, Reg-T, portfolio margin,
     futures) and options approval levels.

Found during planning (2026-10-23). Verify; don't trust:
- Episode URLs look like tastylive.com/shows/<show-slug>/episodes/<slug>-MM-DD-YYYY. Show pages
  paginate with ?page=N, 25 per page.
- Recent episode pages render only a generic ~50-word blurb, with no transcript and no
  per-episode video id. Inspect the raw HTML and any embedded JSON (__NEXT_DATA__ / JSON-LD)
  for video ids. Check whether older episodes carry real descriptions or segment titles.
- The YouTube channel @tastyliveshow (formerly tastytrade) is the expected transcript source.
- feeds.tastylive.com/podcast.rss is one network-wide feed (~627 items): audio for recent
  episodes only.
- APIFY_TOKEN is read from the environment (zenith/config.py `APIFY_TOKEN`), and zenith/apify.py
  already wraps the Apify API. It is a possible transcript fallback.

## Step 1 — Discover (metadata only)
- Enumerate each target show's tastylive.com pages: URL, date, title, description.
- Enumerate the YouTube channel's uploads and playlists with yt-dlp --flat-playlist: id, title,
  upload date, duration, description, playlist. Classify each video to a show from its
  title/description/playlist. Unclassified videos go to a review list.
- Match tastylive episodes to YouTube videos on show, date (±2 days) and title similarity.
  Store match_confidence.
- Write data/vault/manifest.json (schema in brief) and data/vault/coverage.json: per show ×
  year, the count on tastylive.com, matched videos and unmatched ones.

GATE A — show:
- the coverage table;
- the classifier rules;
- 20 unclassified titles;
- an estimate of fetch time.

Ask the user to confirm shows and years before any bulk fetch.

## Step 2 — Fetch (incremental: skip anything already cached; resumable)
Transcript sources, in order of preference:
1. YouTube manual captions.
2. YouTube auto captions: yt-dlp --skip-download --write-subs --write-auto-subs
   --sub-langs "en.*" --sub-format vtt, with --sleep-requests and --sleep-subtitles.
3. If YouTube blocks or rate-limits: the Apify YouTube-transcript actor, through
   zenith/apify.py. Ask before spending.
4. Last resort, for high-value gaps only: local faster-whisper on podcast audio. Ask before
   installing.

Processing:
- Parse VTT into de-duplicated segments with start times. Auto captions repeat rolling lines;
  collapse them. Store data/vault/cache/raw/<episode_id>.json as
  {"source", "segments": [{"t", "text"}]}.
- Save written pages to data/vault/cache/pages/<slug>.json as {url, fetched, title, text},
  using main-content extraction only.
- Run long fetches in the background with a progress log.

## Step 3 — Report
Update coverage.json with:
- transcript coverage % per show × year;
- median transcript length;
- failures by reason.

Spot-check 5 transcripts for quality (ASR noise, truncation).

## Code
- zenith/vault/harvest.py — CLI:
  python -m zenith.vault.harvest discover|fetch|report [--show S] [--since YYYY] [--limit N]
- zenith/vault/corpus.py — pure helpers: VTT parse and dedupe, show classifier, episode
  matching, id building.
- tests/test_vault.py — offline tests for those helpers, with small fixtures in
  tests/fixtures/vault/.
- requirements-vault.txt — new research-only dependencies.
- .gitignore — add data/vault/cache/.

GATE B — show:
- the final coverage table;
- the failure summary;
- the first 30 cleaned lines of 3 sample transcripts.

Then update HANDOFF.md.

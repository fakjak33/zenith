# Phase 02 — Taxonomy + per-episode extraction (Claude API, Message Batches)
Read 00_master_brief.md and HANDOFF.md first. Load the claude-api skill before writing any API
code. Take model ids, Message Batches, structured outputs, prompt caching and token counting
from it, not from memory. ANTHROPIC_API_KEY comes from the environment only.

## Step 1 — Taxonomy
Draw a stratified sample: ~60 transcripts across shows and years, weighted toward Market
Measures. Read them in-session and draft data/vault/taxonomy.json:
- strategy slugs with aliases;
- topic slugs;
- episode types.

It must at least include:
- Strategies: short strangle, short straddle, iron condor, iron fly, jade lizard and variants,
  short put, cash-secured put, covered call, covered strangle, wheel, poor man's covered call,
  credit/debit verticals, butterflies, broken-wing butterfly, ratio spreads, calendars,
  diagonals, long straddle/strangle, risk reversal, collar, protective put / index put
  spread, VIX hedges, earnings trades, 0DTE structures, futures options, pairs.
- Topics: IV rank/percentile, expected move, probability of profit/touch, delta as
  probability, entry DTE, 21-DTE management, profit targets, rolling (duration/credit),
  defending the tested side, inverting, stop/loss rules, number of occurrences, buying-power
  allocation vs VIX, position sizing, theta/NLV, beta-weighted delta,
  correlation/diversification, liquidity, skew, term structure, VRP, earnings IV crush,
  post-earnings drift, assignment/dividends, margin types, taxes, costs/slippage.

GATE A — show the taxonomy for the user's edits, then freeze it as v1.

## Step 2 — Extraction script + pilot
zenith/vault/extract.py:
- one request per episode;
- the system prompt (instructions + taxonomy + schema) is a cached shared prefix;
- the user turn is episode metadata + the timestamped transcript;
- output is structured JSON matching the extraction schema in the brief;
- record model, prompt version and token usage per episode;
- chunk only if a transcript exceeds the context budget.

Evaluate two designs:
- (a) single-tier: full extraction with one model;
- (b) two-tier: a cheap triage pass (smaller model: episode_type, signal 0–3,
  strategies/topics), then full extraction only for signal ≥ 1 with the stronger model.

Pilot on 40 episodes with each candidate model and design. Validate:
- the output is schema-valid;
- every slug is in the taxonomy (otherwise it goes to unmapped);
- grounding: every number in studies/rules/claims appears in the transcript within ±60 s of
  its timestamp. Automate this; failures are suspected hallucinations;
- hand-audit 10 episodes in-session against their transcripts (missed studies, wrong numbers).

Project the full-run cost with token counting.

GATE B — show, per configuration:
- schema pass %, grounding pass %, audit notes, missed-study count;
- projected cost and runtime;
- your recommendation.

Do not submit the full run without the user's approval.

## Step 3 — Full run
- Batch ids persist in data/vault/cache/batches.json. Poll politely and re-submit only
  failures, so the run is resumable.
- Write data/vault/extract/<episode_id>.json. These are committed: paraphrased notes, no
  transcript text.
- Re-runs process only episodes with a new transcript or a stale extract version.
- Update each manifest row's extract block (status, version, model).

## Step 4 — Aggregate
- data/vault/research_catalog.json — every study flattened: question, strategies,
  underlyings, period, entry/management, metrics, findings, caveats, source ref
  (ep:<episode_id>@mm:ss) and episode url.
- data/vault/rules_index.json — rules grouped by strategy/topic and dated, so changes in
  guidance over time are visible.

Report:
- the top 30 strategies and topics by episode count;
- the unmapped list;
- rules whose guidance changed over time.

GATE C — show:
- counts, failure rate and grounding pass rate;
- the top lists;
- 5 random extractions beside the transcript passages they came from.

Then update HANDOFF.md.

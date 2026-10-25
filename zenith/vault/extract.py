"""OPTIONS VAULT Phase 02 — per-episode extraction with the Claude API (Message Batches).

One request per episode. The system prompt (instructions + frozen taxonomy) is a
cached shared prefix; the user turn is the episode metadata plus its timestamped
transcript. Output is constrained to a JSON schema (structured outputs), then
converted to the canonical extraction record of 00_master_brief.md and checked:

  schema     required keys and types; every strategy/topic slug in the taxonomy
             (anything else is moved to ``unmapped``)
  grounding  every number in studies/rules/claims must appear in the transcript
             within +/- GROUND_WINDOW_S of the item's timestamp; failures are
             suspected hallucinations
  verbatim   no run of VERBATIM_WORDS words copied from the transcript
             (committed extracts are paraphrase only)

Two designs are piloted: single-tier (one model does the full extraction) and
two-tier (a cheap triage pass, then full extraction only where signal >= 1).
The two-tier result is assembled from the triage outputs plus the full-tier
outputs of the same pilot episodes, so the pilot pays for each model once.

Pure helpers (prompt, schema, grounding, validation, costs) need no network and
are exercised by tests/test_vault_extract.py. The API layer imports ``anthropic``
lazily; the key comes from ANTHROPIC_API_KEY (or an ``ant auth login`` profile),
never from a file.

IN-SESSION PATH (no API spend; the user's choice on 2026-10-24). Claude Code reader subagents
read the transcripts and write outputs in the same structured shape; ``ingest`` runs them through
the same conversion and checks and writes the canonical records:
  .venv/Scripts/python -m zenith.vault.extract kit --out <dir>            # instructions + schema
  .venv/Scripts/python -m zenith.vault.extract pilot-set --n 20 --out <dir>   # transcripts as text
  .venv/Scripts/python -m zenith.vault.extract ingest <reader_dir> [--dest pilot|run]
  .venv/Scripts/python -m zenith.vault.extract report

API PATH (Message Batches; needs ANTHROPIC_API_KEY and spends money):
  .venv/Scripts/python -m zenith.vault.extract pilot [--configs opus sonnet haiku-triage]
  .venv/Scripts/python -m zenith.vault.extract poll          # waits politely, collects results
  .venv/Scripts/python -m zenith.vault.extract report        # Gate B table
  .venv/Scripts/python -m zenith.vault.extract project       # token-counted full-run cost
  .venv/Scripts/python -m zenith.vault.extract run --config opus   # full run (after Gate B)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import VAULT_CACHE_DIR, VAULT_DIR
from .corpus import fmt_ts

PROMPT_VERSION = "x1"
TAXONOMY_PATH = VAULT_DIR / "taxonomy.json"
MANIFEST_PATH = VAULT_DIR / "manifest.json"
EXTRACT_DIR = VAULT_DIR / "extract"            # committed: paraphrased notes, no transcript text
RAW_DIR = VAULT_CACHE_DIR / "raw"              # transcripts (gitignored)
PILOT_DIR = VAULT_CACHE_DIR / "pilot"          # pilot outputs per configuration
BATCHES_PATH = VAULT_CACHE_DIR / "batches.json"
TOKENS_PATH = VAULT_CACHE_DIR / "token_counts.json"

GROUND_WINDOW_S = 60
STUDY_SETUP_BEFORE_S = 300   # a study's setup (DTE, sample size) is described before its results ts
FINDINGS_WINDOW_S = 300      # numbers inside findings text, checked +/- this (secondary measure)
VERBATIM_WORDS = 12
CHUNK_CHARS = 600_000      # ~150k tokens; the longest transcript is ~47k chars, so this is a guard
POLL_SECS = 60
PILOT_N = 40

# $/MTok, Claude API list prices (claude-api skill, cached 2026-10-06). Batches bill 50%.
# A 1-hour cache write costs 2x input. Haiku 5.5's cache-read price is assumed at 0.1x input.
PRICES = {
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01},
}
BATCH_DISCOUNT = 0.5

# tier -> (model, effort, max_tokens). Opus 5.5 thinking cannot be disabled; effort is the control.
CONFIGS = {
    "opus": ("full", "claude-opus-5-5", "medium", 16000),
    "sonnet": ("full", "claude-sonnet-5-5", "medium", 16000),
    "haiku-triage": ("triage", "claude-haiku-5-5", "low", 4000),
}
TWO_TIER = {"haiku>opus": ("haiku-triage", "opus"), "haiku>sonnet": ("haiku-triage", "sonnet")}

EPISODE_TYPES = ["study", "mechanics", "concept", "commentary", "trade_review", "other"]
RULE_KINDS = ["entry", "manage", "exit", "roll", "defend", "size", "select", "avoid"]
METRICS = ["win_rate", "avg_pnl", "pnl_per_day", "max_loss", "roc", "stdev"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jread(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[vault-extract] {msg}", flush=True)


# =============================================================================
# PROMPT
# =============================================================================

INSTRUCTIONS = """\
You extract structured research notes from one tastylive episode transcript for OPTIONS VAULT, an
evidence-graded reference of options strategies, edges and portfolio mechanics.

The transcript is YouTube auto-captions: no punctuation guarantees, misheard words ("strangel",
"I see" for IC, "pop" for probability of profit), and numbers written as digits or words. Each line
starts with its timestamp [mm:ss].

What to extract:
- episode_type: study (a backtest or data study with a defined sample/period/metrics), mechanics
  (how margin, assignment, products, orders or taxes work), concept (explains an idea without a
  dataset), commentary (market or news reaction), trade_review (walks through specific trades),
  other (little options content).
- signal: 0 = no usable options content; 1 = passing remarks only; 2 = concrete rules or numbers;
  3 = at least one study with sample, period and metrics.
- strategies / topics: taxonomy slugs for what the episode substantively discusses (not every
  passing mention). Use only slugs from the taxonomy below. A strategy or concept that no slug
  covers goes in unmapped as a short canonical name.
- studies: one entry per distinct backtest or data study presented. question = what was tested;
  underlyings; period (e.g. "2005-2024"); entry.dte and entry.rule (strikes/deltas/conditions);
  management (how trades were managed or exited); sample_n (number of trades or occurrences if
  stated); metrics (win_rate and roc in percent, avg_pnl and max_loss in dollars per trade or as
  stated, pnl_per_day, stdev; null when not stated); findings (each a short paraphrased result
  with its numbers); caveats (limits the hosts state or that are evident, e.g. one underlying,
  short window, mid-price fills); ts = where the study's results are presented.
- rules: actionable guidance (entry, manage, exit, roll, defend, size, select, avoid) stated or
  clearly endorsed by the hosts, with conditions and numbers, and the strategies/topics each
  applies to.
- claims: other factual or quantitative statements worth checking (e.g. "the expected move
  contains the price about 68% of the time"), with numbers.
- terms: jargon defined on air, with a one-line definition in your own words.

Numbers:
- Every value you put in metrics, sample_n, entry.dte or a numbers list must be a number the
  hosts actually say (or show and read out) near that item's ts. Do not compute, convert or infer
  new numbers. Percentages stay in percent (16 delta -> 16, 50% -> 50).
- ts is the [mm:ss] of the line where that number or statement is spoken. Use the transcript's
  timestamps exactly.

Copyright: write everything as short paraphrase in your own words. Never copy sentences from the
transcript; at most a few words of a phrase.

If the episode has no options content, return signal 0 with empty lists.
"""

TRIAGE_INSTRUCTIONS = """\
You triage one tastylive episode transcript for OPTIONS VAULT. Read it and return only:
episode_type (study | mechanics | concept | commentary | trade_review | other), signal
(0 = no usable options content; 1 = passing remarks only; 2 = concrete rules or numbers;
3 = at least one backtest/data study with sample, period and metrics), and the taxonomy slugs of
the strategies and topics it substantively discusses. Use only slugs from the taxonomy below.
"""


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    return _jread(path, {})


def taxonomy_block(tax: dict) -> str:
    """Compact, deterministic taxonomy text for the cached system prefix."""
    lines = [f"TAXONOMY {tax.get('version', '')}", "", "Strategies (slug: name | aliases):"]
    for s in tax.get("strategies", []):
        lines.append(f"- {s['slug']}: {s['name']} | {', '.join(s.get('aliases', []))}")
    lines += ["", "Topics (slug: name | aliases):"]
    for t in tax.get("topics", []):
        lines.append(f"- {t['slug']}: {t['name']} | {', '.join(t.get('aliases', []))}")
    return "\n".join(lines)


def build_system(tax: dict, tier: str = "full") -> list[dict]:
    """System prompt as one cached block. Byte-stable for a given taxonomy + tier."""
    head = INSTRUCTIONS if tier == "full" else TRIAGE_INSTRUCTIONS
    return [{"type": "text", "text": head + "\n" + taxonomy_block(tax),
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def episode_url(row: dict) -> str:
    if row.get("url"):
        return row["url"]
    return f"https://www.youtube.com/watch?v={row['youtube_id']}" if row.get("youtube_id") else ""


def transcript_lines(raw: dict) -> list[str]:
    return [f"[{fmt_ts(g['t'])}] {g['text']}" for g in raw.get("segments", [])]


def chunk_lines(lines: list[str], max_chars: int = CHUNK_CHARS) -> list[list[str]]:
    """Split transcript lines into pieces under max_chars (one piece in practice)."""
    out, cur, n = [], [], 0
    for ln in lines:
        if cur and n + len(ln) + 1 > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur or not out:
        out.append(cur)
    return out


def build_user(row: dict, lines: list[str], part: tuple[int, int] = (1, 1)) -> str:
    title = row.get("title") or row.get("yt_title") or ""
    tags = ", ".join((row.get("tl") or {}).get("tags") or [])
    head = [f"episode_id: {row['episode_id']}", f"show: {row.get('show', '')}",
            f"air_date: {row.get('air_date', '')}", f"title: {title}"]
    if tags:
        head.append(f"tastylive tags: {tags}")
    if part[1] > 1:
        head.append(f"transcript part {part[0]} of {part[1]} (extract only from this part)")
    return "\n".join(head) + "\n\nTRANSCRIPT\n" + "\n".join(lines)


# =============================================================================
# OUTPUT SCHEMA (structured outputs) and canonical conversion
# =============================================================================

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def _nullable(t: str) -> dict:
    return {"anyOf": [{"type": t}, {"type": "null"}]}


def api_schema(tax: dict, tier: str = "full") -> dict:
    """JSON schema sent as output_config.format. Slugs are enums, so they are valid by construction;
    free-form ``numbers`` objects become [{name, value, unit}] lists (converted back to dicts)."""
    strat = {"type": "string", "enum": [s["slug"] for s in tax["strategies"]]}
    topic = {"type": "string", "enum": [t["slug"] for t in tax["topics"]]}
    s = {"type": "string"}
    base = {"episode_type": {"type": "string", "enum": EPISODE_TYPES},
            "signal": {"type": "integer", "enum": [0, 1, 2, 3]},
            "strategies": _arr(strat), "topics": _arr(topic)}
    if tier == "triage":
        return _obj(base)
    nums = _arr(_obj({"name": s, "value": {"type": "number"}, "unit": s}))
    study = _obj({
        "question": s, "strategies": _arr(strat), "underlyings": _arr(s), "period": s,
        "entry": _obj({"dte": _nullable("number"), "rule": s}), "management": s,
        "sample_n": _nullable("integer"),
        "metrics": _obj({m: _nullable("number") for m in METRICS}),
        "findings": _arr(s), "caveats": _arr(s), "ts": s})
    rule = _obj({"kind": {"type": "string", "enum": RULE_KINDS}, "text": s, "conditions": s,
                 "strategies": _arr(strat), "topics": _arr(topic), "numbers": nums, "ts": s})
    claim = _obj({"text": s, "topics": _arr(topic), "numbers": nums, "ts": s})
    return _obj({**base, "studies": _arr(study), "rules": _arr(rule), "claims": _arr(claim),
                 "terms": _arr(_obj({"term": s, "definition": s})), "unmapped": _arr(s)})


def _num_dict(items) -> dict:
    if isinstance(items, dict):
        return {str(k): v for k, v in items.items() if isinstance(v, (int, float))}
    out = {}
    for it in items or []:
        if isinstance(it, dict) and isinstance(it.get("value"), (int, float)):
            key = str(it.get("name") or "value").strip() or "value"
            if it.get("unit"):
                key = f"{key} ({it['unit']})"
            while key in out:
                key += "'"
            out[key] = it["value"]
    return out


def to_canonical(out: dict, row: dict) -> dict:
    """Structured output -> canonical extraction record (brief schema plus added fields)."""
    rec = {"episode_id": row["episode_id"], "show": row.get("show", ""),
           "air_date": row.get("air_date", ""),
           "title": row.get("title") or row.get("yt_title") or "", "url": episode_url(row),
           "episode_type": out.get("episode_type", "other"), "signal": out.get("signal", 0),
           "strategies": list(out.get("strategies", [])), "topics": list(out.get("topics", [])),
           "studies": [], "rules": [], "claims": [],
           "terms": list(out.get("terms", [])), "unmapped": list(out.get("unmapped", []))}
    for st in out.get("studies", []):
        rec["studies"].append({**st, "entry": dict(st.get("entry") or {"dte": None, "rule": ""}),
                               "metrics": {m: (st.get("metrics") or {}).get(m) for m in METRICS}})
    for r in out.get("rules", []):
        rec["rules"].append({**r, "numbers": _num_dict(r.get("numbers"))})
    for c in out.get("claims", []):
        rec["claims"].append({**c, "numbers": _num_dict(c.get("numbers"))})
    return rec


def merge_parts(parts: list[dict]) -> dict:
    """Merge chunked outputs: union slugs, concatenate items, max signal."""
    if len(parts) == 1:
        return parts[0]
    m = dict(parts[0])
    m["signal"] = max(p.get("signal", 0) for p in parts)
    for k in ("strategies", "topics", "unmapped"):
        m[k] = list(dict.fromkeys(x for p in parts for x in p.get(k, [])))
    for k in ("studies", "rules", "claims", "terms"):
        m[k] = [x for p in parts for x in p.get(k, [])]
    return m


# =============================================================================
# CHECKS: schema, grounding, verbatim
# =============================================================================

def parse_ts(ts) -> int | None:
    """'mm:ss' or 'h:mm:ss' -> seconds; None when unparseable."""
    if not isinstance(ts, str):
        return None
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d{1,3}):(\d{2})\s*", ts)
    if not m:
        return None
    h, mm, ss = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return h * 3600 + mm * 60 + ss


def validate(rec: dict, tax: dict) -> list[str]:
    """Schema + taxonomy check of a canonical record. Unknown slugs are moved to unmapped in place;
    returns the list of errors (empty = pass)."""
    errs = []
    known = {"strategies": {s["slug"] for s in tax["strategies"]},
             "topics": {t["slug"] for t in tax["topics"]}}
    for k in ("episode_id", "episode_type", "signal", "strategies", "topics", "studies", "rules",
              "claims", "terms", "unmapped"):
        if k not in rec:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if rec["episode_type"] not in EPISODE_TYPES:
        errs.append(f"episode_type {rec['episode_type']!r}")
    if rec["signal"] not in (0, 1, 2, 3):
        errs.append(f"signal {rec['signal']!r}")

    def _slugs(holder: dict, where: str):
        for kind, ok in known.items():
            if kind not in holder:
                continue
            bad = [x for x in holder[kind] if x not in ok]
            if bad:
                holder[kind] = [x for x in holder[kind] if x in ok]
                rec["unmapped"] += [b for b in bad if b not in rec["unmapped"]]

    _slugs(rec, "episode")
    for i, st in enumerate(rec["studies"]):
        _slugs(st, f"studies[{i}]")
        for k in ("question", "findings", "metrics", "ts"):
            if k not in st:
                errs.append(f"studies[{i}] missing {k}")
        if set(st.get("metrics", {})) != set(METRICS):
            errs.append(f"studies[{i}].metrics keys")
        if parse_ts(st.get("ts")) is None:
            errs.append(f"studies[{i}].ts {st.get('ts')!r}")
    for i, r in enumerate(rec["rules"]):
        _slugs(r, f"rules[{i}]")
        if r.get("kind") not in RULE_KINDS:
            errs.append(f"rules[{i}].kind {r.get('kind')!r}")
        if not isinstance(r.get("numbers"), dict):
            errs.append(f"rules[{i}].numbers")
        if parse_ts(r.get("ts")) is None:
            errs.append(f"rules[{i}].ts {r.get('ts')!r}")
    for i, c in enumerate(rec["claims"]):
        _slugs(c, f"claims[{i}]")
        if not isinstance(c.get("numbers"), dict):
            errs.append(f"claims[{i}].numbers")
        if parse_ts(c.get("ts")) is None:
            errs.append(f"claims[{i}].ts {c.get('ts')!r}")
    if rec["signal"] == 3 and not rec["studies"]:
        errs.append("signal 3 without studies")
    return errs


_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
          "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_FRACS = {"half": 0.5, "quarter": 0.25, "third": 1 / 3}
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)?(\.\d+)?(?:\s*(k|mm|m|million|billion|thousand)\b)?",
                     re.I)


_MONEY = {"dollar", "dollars", "buck", "bucks"}


def _word_spans(words: list[str]) -> list[tuple[int, int, float]]:
    """(start, end, value) for runs of spelled-out number words and bare digit tokens."""
    spans, i = [], 0
    while i < len(words):
        w = words[i]
        if re.fullmatch(r"\d+(?:\.\d+)?", w):
            spans.append((i, i + 1, float(w)))
            i += 1
        elif w in _WORDS:
            start, cur, total = i, 0.0, 0.0
            while i < len(words):
                w = words[i]
                if w in _WORDS:
                    cur += _WORDS[w]
                elif w in _SCALES:
                    cur = max(cur, 1) * _SCALES[w]
                    if _SCALES[w] >= 1000:
                        total, cur = total + cur, 0.0
                elif not (w == "and" and cur >= 100 and i + 1 < len(words) and words[i + 1] in _WORDS):
                    break
                i += 1
            spans.append((start, i, total + cur))
        else:
            i += 1
    return spans


def numbers_in(text: str) -> list[float]:
    """Numbers in caption text: digits (1,000 / 2.5 / .16 / 10k), spelled-out numbers, spoken money
    ("a dollar sixty", "three dollars and sixty cents"), and caption forms that drop the decimal
    point or separator ("283" for 2.83, "7030" for 70/30)."""
    out = []
    for m in _NUM_RE.finditer(text):
        ip, fp, suf = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not ip and not fp:
            continue
        v = float((ip or "0").replace(",", "") + (fp or ""))
        out.append(v)
        mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "billion": 1e9}.get(suf)
        if mult:
            out.append(v * mult)
        if ip and not fp and not suf and "," not in ip and 3 <= len(ip) <= 4:
            out.append(v / 100)
            if len(ip) == 4:
                out += [float(ip[:2]), float(ip[2:])]
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    spans = _word_spans(words)
    out += [v for a, b, v in spans if not words[a][0].isdigit()]
    out += [_FRACS[w] for w in words if w in _FRACS]
    by_start = {a: (b, v) for a, b, v in spans}
    heads = spans + [(i, i + 1, 1.0) for i, w in enumerate(words) if w == "a"]
    for a, b, v in heads:
        if b < len(words) and words[b] in _MONEY:
            out.append(v)
            j = b + 1 + int(b + 1 < len(words) and words[b + 1] == "and")
            if j in by_start and by_start[j][1] < 100:
                out.append(v + by_start[j][1] / 100)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.015 * abs(b), 0.051)


def number_matches(v: float, candidates: list[float]) -> bool:
    """A stated value matches a transcript number directly, unsigned, or as fraction<->percent
    (only for values <= 1, so 12.5 does not match 1,250)."""
    for t in candidates:
        for x in (v, abs(v)):
            if (_close(x, t) or (x <= 1 and _close(x * 100, t))
                    or (t <= 1 and _close(x, t * 100))):
                return True
    return False


def window_numbers(raw: dict, ts: int, before: int = GROUND_WINDOW_S,
                   after: int = GROUND_WINDOW_S) -> list[float]:
    nums = []
    for g in raw.get("segments", []):
        if ts - before <= g["t"] <= ts + after:
            nums += numbers_in(g["text"])
    return nums


def item_numbers(rec: dict):
    """Yield (where, value, ts, seconds_before) for every structured number in studies/rules/claims.
    A study's setup fields may be said up to STUDY_SETUP_BEFORE_S before its results ts."""
    for i, st in enumerate(rec.get("studies", [])):
        vals = [("sample_n", st.get("sample_n"), STUDY_SETUP_BEFORE_S),
                ("entry.dte", (st.get("entry") or {}).get("dte"), STUDY_SETUP_BEFORE_S)]
        vals += [(f"metrics.{k}", v, GROUND_WINDOW_S) for k, v in (st.get("metrics") or {}).items()]
        for name, v, before in vals:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                yield f"studies[{i}].{name}", float(v), st.get("ts"), before
    for kind in ("rules", "claims"):
        for i, it in enumerate(rec.get(kind, [])):
            for name, v in (it.get("numbers") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    yield f"{kind}[{i}].{name}", float(v), it.get("ts"), GROUND_WINDOW_S


def ground(rec: dict, raw: dict, window: int = GROUND_WINDOW_S) -> dict:
    """Check every structured number against the transcript near its timestamp. Each failure says
    whether the number occurs anywhere in the transcript (misplaced ts) or nowhere (suspected
    hallucination)."""
    checked, failures = 0, []
    cache: dict[tuple, list[float]] = {}
    everywhere = [x for g in raw.get("segments", []) for x in numbers_in(g["text"])]
    for where, v, ts, before in item_numbers(rec):
        checked += 1
        sec = parse_ts(ts)
        if sec is None:
            failures.append({"where": where, "value": v, "ts": ts, "why": "bad ts",
                             "anywhere": number_matches(v, everywhere)})
            continue
        key = (sec, before)
        if key not in cache:
            cache[key] = window_numbers(raw, sec, before, window)
        if not number_matches(v, cache[key]):
            failures.append({"where": where, "value": v, "ts": ts, "why": "not near ts",
                             "anywhere": number_matches(v, everywhere)})
    f_checked = f_passed = 0
    for st in rec.get("studies", []):
        sec = parse_ts(st.get("ts"))
        if sec is None:
            continue
        near = window_numbers(raw, sec, FINDINGS_WINDOW_S, FINDINGS_WINDOW_S)
        for text in st.get("findings", []):
            for m in _NUM_RE.finditer(text):
                if m.group(1) or m.group(2):
                    f_checked += 1
                    f_passed += number_matches(float((m.group(1) or "0").replace(",", "")
                                                     + (m.group(2) or "")), near)
    return {"checked": checked, "passed": checked - len(failures), "failures": failures,
            "nowhere": sum(not f["anywhere"] for f in failures),
            "findings_numbers": {"checked": f_checked, "passed": f_passed}}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _texts(rec: dict):
    for st in rec.get("studies", []):
        yield st.get("question", "")
        yield from st.get("findings", [])
        yield from st.get("caveats", [])
        yield st.get("management", "")
    for k in ("rules", "claims"):
        for it in rec.get(k, []):
            yield it.get("text", "")
    for t in rec.get("terms", []):
        yield t.get("definition", "")


def verbatim(rec: dict, raw: dict, n: int = VERBATIM_WORDS) -> list[str]:
    """Extract strings that share an n-word run with the transcript (copyright check)."""
    words = _tokens(" ".join(g["text"] for g in raw.get("segments", [])))
    grams = {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    hits = []
    for s in _texts(rec):
        w = _tokens(s)
        if any(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1)):
            hits.append(s)
    return hits


def check(rec: dict, raw: dict, tax: dict) -> dict:
    errs = validate(rec, tax)
    g = ground(rec, raw)
    v = verbatim(rec, raw)
    return {"schema_ok": not errs, "errors": errs, "grounding": g, "verbatim": v}


# =============================================================================
# SELECTION, STALENESS, COSTS
# =============================================================================

def extract_version(tax: dict) -> str:
    return f"{PROMPT_VERSION}+tax-{tax.get('version', '')}"


def needs_extract(row: dict, existing: dict | None, version: str) -> bool:
    """True when there is a transcript and no current extract for it."""
    if (row.get("transcript") or {}).get("source", "none") == "none":
        return False
    if not existing:
        return True
    meta = existing.get("extract", {})
    return (meta.get("version") != version
            or meta.get("transcript_chars") != (row.get("transcript") or {}).get("chars"))


def pilot_set(rows: list[dict], n: int = PILOT_N, seed: int = 20261024) -> list[str]:
    """Stratified pilot sample: shows x years, Market Measures weighted 2x."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault((r["show"], r["air_date"][:4]), []).append(r)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    weights = {k: (2 if k[0] == "market-measures" else 1) for k in keys}
    picked: list[str] = []
    while len(picked) < min(n, len(rows)):
        for k in keys:
            for _ in range(weights[k]):
                if strata[k] and len(picked) < n:
                    picked.append(strata[k].pop()["episode_id"])
    return sorted(picked)


def cost_usd(model: str, usage: dict, batch: bool = True) -> float:
    """Dollar cost of one request from its usage block."""
    p = PRICES[model]
    c = (usage.get("input_tokens", 0) * p["in"]
         + usage.get("cache_creation_input_tokens", 0) * p["in"] * 2.0
         + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
         + usage.get("output_tokens", 0) * p["out"]) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


def project_cost(model: str, user_tokens: list[int], system_tokens: int, out_tokens: float,
                 batch: bool = True) -> float:
    """Projected cost: one cache write of the system prefix, cache reads after, plus user turns and
    the pilot's mean output (thinking included)."""
    p = PRICES[model]
    n = len(user_tokens)
    if not n:
        return 0.0
    c = (system_tokens * p["in"] * 2.0 + (n - 1) * system_tokens * p["cache_read"]
         + sum(user_tokens) * p["in"] + n * out_tokens * p["out"]) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


# =============================================================================
# DATA ACCESS
# =============================================================================

def load_rows() -> list[dict]:
    man = _jread(MANIFEST_PATH, {"episodes": []})
    return [r for r in man["episodes"]
            if (r.get("transcript") or {}).get("source", "none") != "none"
            and (RAW_DIR / f"{r['episode_id']}.json").exists()]


def load_raw(episode_id: str) -> dict:
    return _jread(RAW_DIR / f"{episode_id}.json", {"segments": []})


def custom_id(episode_id: str, part: int = 1, parts: int = 1) -> str:
    cid = episode_id if parts == 1 else f"{episode_id}__p{part}"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid), cid
    return cid


def requests_for(rows: list[dict], tax: dict, config: str) -> list[dict]:
    """Batch requests (plain dicts) for these episodes under one configuration."""
    tier, model, effort, max_tokens = CONFIGS[config]
    system = build_system(tax, tier)
    fmt = {"type": "json_schema", "schema": api_schema(tax, tier)}
    reqs = []
    for row in rows:
        pieces = chunk_lines(transcript_lines(load_raw(row["episode_id"])))
        for i, lines in enumerate(pieces, 1):
            reqs.append({"custom_id": custom_id(row["episode_id"], i, len(pieces)),
                         "params": {"model": model, "max_tokens": max_tokens, "system": system,
                                    "output_config": {"effort": effort, "format": fmt},
                                    "messages": [{"role": "user", "content":
                                                  build_user(row, lines, (i, len(pieces)))}]}})
    return reqs


# =============================================================================
# API LAYER (lazy anthropic import)
# =============================================================================

def _client():
    import os

    import anthropic  # research-only dependency (requirements-vault.txt)
    if not os.environ.get("ANTHROPIC_API_KEY") and sys.platform == "win32":
        # A key saved with SetEnvironmentVariable(..., 'User') only reaches processes started after
        # it was set; read it from the user's registry environment (never from or to a file).
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
        except OSError:
            pass
    return anthropic.Anthropic()


def _state() -> dict:
    return _jread(BATCHES_PATH, {"batches": []})


def submit(rows: list[dict], tax: dict, config: str, kind: str) -> str:
    reqs = requests_for(rows, tax, config)
    b = _client().messages.batches.create(requests=reqs)
    st = _state()
    st["batches"].append({"id": b.id, "kind": kind, "config": config, "created": _now(),
                          "version": extract_version(tax), "n": len(reqs),
                          "status": b.processing_status, "collected": False, "failed": []})
    _jwrite(BATCHES_PATH, st)
    _log(f"submitted {b.id}: {kind}/{config}, {len(reqs)} requests")
    return b.id


def _out_dir(kind: str, config: str) -> Path:
    return EXTRACT_DIR if kind == "run" else PILOT_DIR / config


def _usage(msg) -> dict:
    u = msg.usage
    return {k: getattr(u, k, 0) or 0 for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def collect(entry: dict, tax: dict) -> None:
    """Write every succeeded result of an ended batch; record failures for resubmission."""
    client = _client()
    rows = {r["episode_id"]: r for r in load_rows()}
    tier, model, _, _ = CONFIGS[entry["config"]]
    outdir = _out_dir(entry["kind"], entry["config"])
    parts: dict[str, list] = {}
    failed = []
    for res in client.messages.batches.results(entry["id"]):
        eid = res.custom_id.split("__p")[0]
        if res.result.type != "succeeded":
            failed.append({"episode_id": eid, "why": res.result.type})
            continue
        msg = res.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        if msg.stop_reason != "end_turn":
            failed.append({"episode_id": eid, "why": f"stop_reason {msg.stop_reason}"})
            continue
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            failed.append({"episode_id": eid, "why": "invalid json"})
            continue
        parts.setdefault(eid, []).append((res.custom_id, out, _usage(msg)))
    for eid, got in parts.items():
        got.sort(key=lambda g: g[0])
        out = merge_parts([g[1] for g in got])
        usage = {k: sum(g[2][k] for g in got) for k in got[0][2]}
        row = rows.get(eid, {"episode_id": eid})
        meta = {"version": extract_version(tax), "prompt_version": PROMPT_VERSION,
                "taxonomy_version": tax.get("version", ""), "model": model, "tier": tier,
                "config": entry["config"], "batch_id": entry["id"], "created": _now(),
                "transcript_chars": (row.get("transcript") or {}).get("chars"),
                "usage": usage, "cost_usd": round(cost_usd(model, usage), 5)}
        if tier == "triage":
            rec = {"episode_id": eid, **{k: out.get(k) for k in
                                         ("episode_type", "signal", "strategies", "topics")}}
            rec["extract"] = meta
        else:
            rec = to_canonical(out, row)
            rec["extract"] = meta
            rec["checks"] = check(rec, load_raw(eid), tax)
        _jwrite(outdir / f"{eid}.json", rec)
    entry["collected"], entry["failed"] = True, failed
    _log(f"collected {entry['id']}: {len(parts)} written to {outdir}, {len(failed)} failed")


def poll(once: bool = False) -> None:
    """Poll open batches politely (one request per batch per POLL_SECS) and collect ended ones."""
    tax = load_taxonomy()
    client = _client()
    while True:
        st = _state()
        open_ = [b for b in st["batches"] if not b["collected"]]
        if not open_:
            _log("no open batches")
            return
        for b in open_:
            cur = client.messages.batches.retrieve(b["id"])
            b["status"] = cur.processing_status
            c = cur.request_counts
            _log(f"{b['id']} {b['kind']}/{b['config']}: {cur.processing_status} "
                 f"(processing {c.processing}, ok {c.succeeded}, err {c.errored}, exp {c.expired})")
            if cur.processing_status == "ended":
                collect(b, tax)
        _jwrite(BATCHES_PATH, st)
        if once or all(b["collected"] for b in st["batches"]):
            return
        time.sleep(POLL_SECS)


def count_tokens(rows: list[dict], tax: dict, model: str) -> dict:
    """Token-count each episode's user turn (and the system prefix once); cached on disk."""
    client = _client()
    cache = _jread(TOKENS_PATH, {})
    key = f"{model}|{extract_version(tax)}"
    got = cache.setdefault(key, {})
    sys_full = build_system(tax, "full")
    if "_system_full" not in got:
        n0 = client.messages.count_tokens(model=model, system=sys_full,
                                          messages=[{"role": "user", "content": "x"}]).input_tokens
        got["_system_full"] = n0
    for i, row in enumerate(rows):
        eid = row["episode_id"]
        if eid in got:
            continue
        user = build_user(row, transcript_lines(load_raw(eid)))
        n = client.messages.count_tokens(model=model, system=sys_full,
                                         messages=[{"role": "user", "content": user}]).input_tokens
        got[eid] = n - got["_system_full"]
        if i % 50 == 0:
            _jwrite(TOKENS_PATH, cache)
            _log(f"  counted {i}/{len(rows)}")
        time.sleep(0.2)
    _jwrite(TOKENS_PATH, cache)
    return got


# =============================================================================
# IN-SESSION PATH: reader kit, transcript export, ingest
# =============================================================================

INLINE_MODEL = "claude-sonnet (Claude Code subagent)"

READER_FORMAT = """
OUTPUT FORMAT
Write one JSON object per episode matching reader_schema.json, plus "episode_id". In rules and
claims, "numbers" is a list of {"name", "value", "unit"} objects (value is a bare number). Every
slug must come from the taxonomy above; anything else goes in "unmapped".
"""


def write_kit(tax: dict, out: Path) -> list[Path]:
    """Reader instructions (the same text the API path caches) and the output schema."""
    out.mkdir(parents=True, exist_ok=True)
    instr = out / "reader_instructions.md"
    instr.write_text(INSTRUCTIONS + "\n" + taxonomy_block(tax) + "\n" + READER_FORMAT,
                     encoding="utf-8")
    schema = out / "reader_schema.json"
    sch = api_schema(tax)
    sch["properties"] = {"episode_id": {"type": "string"}, **sch["properties"]}
    sch["required"] = ["episode_id"] + sch["required"]
    _jwrite(schema, sch)
    return [instr, schema]


def export_transcripts(rows: list[dict], out: Path) -> list[Path]:
    """Write each episode's metadata + timestamped transcript as text for a reader (third-party
    text: keep `out` outside the repo or under data/vault/cache/)."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        path = out / f"{row['episode_id']}.txt"
        path.write_text(build_user(row, transcript_lines(load_raw(row["episode_id"]))),
                        encoding="utf-8")
        paths.append(path)
    return paths


def ingest(src: Path, dest: Path, tax: dict, model: str = INLINE_MODEL) -> list[dict]:
    """Reader outputs (one object or a list per file) -> canonical records with checks in `dest`."""
    rows = {r["episode_id"]: r for r in load_rows()}
    done = []
    for f in sorted(src.glob("*.json")):
        data = _jread(f, [])
        for out in (data if isinstance(data, list) else [data]):
            eid = out.get("episode_id", "")
            if eid not in rows:
                _log(f"  skip {f.name}: unknown episode_id {eid!r}")
                continue
            rec = to_canonical(out, rows[eid])
            rec["extract"] = {"version": extract_version(tax), "prompt_version": PROMPT_VERSION,
                              "taxonomy_version": tax.get("version", ""), "model": model,
                              "tier": "full", "config": "inline", "created": _now(),
                              "transcript_chars": (rows[eid].get("transcript") or {}).get("chars"),
                              "usage": {"output_tokens": 0}, "cost_usd": 0.0}
            rec["checks"] = check(rec, load_raw(eid), tax)
            _jwrite(dest / f"{eid}.json", rec)
            done.append({"episode_id": eid, "schema_ok": rec["checks"]["schema_ok"],
                         "grounding": f"{rec['checks']['grounding']['passed']}/"
                                      f"{rec['checks']['grounding']['checked']}"})
    _log(f"ingested {len(done)} episodes into {dest}")
    return done


# =============================================================================
# PILOT REPORT (Gate B)
# =============================================================================

def _load_dir(d: Path) -> dict[str, dict]:
    return {p.stem: _jread(p, {}) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def summarize_full(recs: dict[str, dict]) -> dict:
    n = len(recs)
    if not n:
        return {"n": 0}
    schema = sum(r["checks"]["schema_ok"] for r in recs.values())
    chk = sum(r["checks"]["grounding"]["checked"] for r in recs.values())
    ok = sum(r["checks"]["grounding"]["passed"] for r in recs.values())
    ep_ok = sum(r["checks"]["grounding"]["checked"] == r["checks"]["grounding"]["passed"]
                for r in recs.values())
    out_tok = [r["extract"]["usage"]["output_tokens"] for r in recs.values()]
    nowhere = sum(r["checks"]["grounding"]["nowhere"] for r in recs.values())
    return {"n": n, "schema_pass": schema / n,
            "grounding_numbers": (ok / chk) if chk else None, "numbers_checked": chk,
            "numbers_nowhere": nowhere,
            "grounding_episodes": ep_ok / n,
            "studies": sum(len(r["studies"]) for r in recs.values()),
            "rules": sum(len(r["rules"]) for r in recs.values()),
            "claims": sum(len(r["claims"]) for r in recs.values()),
            "unmapped": sum(len(r["unmapped"]) for r in recs.values()),
            "verbatim_hits": sum(len(r["checks"]["verbatim"]) for r in recs.values()),
            "mean_output_tokens": sum(out_tok) / n,
            "cost_usd": sum(r["extract"]["cost_usd"] for r in recs.values())}


def summarize_two_tier(triage: dict[str, dict], full: dict[str, dict]) -> dict:
    """Two-tier = triage on all; full only where triage signal >= 1. 'missed' = episodes triage
    dropped (signal 0) whose full-tier extract has studies or rules."""
    common = sorted(set(triage) & set(full))
    kept = [e for e in common if (triage[e].get("signal") or 0) >= 1]
    missed = [e for e in common if e not in kept and (full[e]["studies"] or full[e]["rules"])]
    missed_studies = sum(len(full[e]["studies"]) for e in missed)
    cost = (sum(triage[e]["extract"]["cost_usd"] for e in common)
            + sum(full[e]["extract"]["cost_usd"] for e in kept))
    sub = summarize_full({e: full[e] for e in kept})
    return {**sub, "n_pilot": len(common), "kept": len(kept), "missed_episodes": missed,
            "missed_studies": missed_studies, "cost_usd": cost}


def pilot_report() -> dict:
    full = {c: _load_dir(PILOT_DIR / c) for c, v in CONFIGS.items() if v[0] == "full"}
    triage = {c: _load_dir(PILOT_DIR / c) for c, v in CONFIGS.items() if v[0] == "triage"}
    inline = _load_dir(PILOT_DIR / "inline")
    if inline:
        full["inline"] = inline
    rep = {"generated": _now(), "single_tier": {c: summarize_full(r) for c, r in full.items() if r},
           "two_tier": {}}
    for name, (t, f) in TWO_TIER.items():
        if triage.get(t) and full.get(f):
            rep["two_tier"][name] = summarize_two_tier(triage[t], full[f])
    # studies a configuration missed relative to the union of configurations, per episode
    eps = set().union(*[set(r) for r in full.values()]) if full else set()
    rep["study_counts"] = {e: {c: len(full[c][e]["studies"]) for c in full if e in full[c]}
                           for e in sorted(eps)}
    return rep


# =============================================================================
# CLI
# =============================================================================

def cmd_pilot(args) -> None:
    tax = load_taxonomy()
    rows = load_rows()
    path = PILOT_DIR / "pilot_set.json"
    ids = _jread(path, None) or pilot_set(rows)
    _jwrite(path, ids)
    sel = [r for r in rows if r["episode_id"] in set(ids)]
    _log(f"pilot: {len(sel)} episodes, configs {args.configs}")
    for c in args.configs:
        submit(sel, tax, c, kind="pilot")


def cmd_run(args) -> None:
    tax = load_taxonomy()
    version = extract_version(tax)
    inflight = {b["config"] for b in _state()["batches"] if b["kind"] == "run" and not b["collected"]}
    if inflight:
        _log("a run batch is still open; poll first")
        return
    rows = [r for r in load_rows()
            if needs_extract(r, _jread(EXTRACT_DIR / f"{r['episode_id']}.json", None), version)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        _log("nothing to extract")
        return
    submit(rows, tax, args.config, kind="run")


def cmd_project(args) -> None:
    tax = load_taxonomy()
    rows = load_rows()
    rep = pilot_report()
    out = {}
    for c, (tier, model, _, _) in CONFIGS.items():
        counts = count_tokens(rows, tax, model)
        users = [counts[r["episode_id"]] for r in rows if r["episode_id"] in counts]
        pil = rep["single_tier"].get(c) or {}
        mean_out = pil.get("mean_output_tokens") or (300 if tier == "triage" else 4000)
        out[c] = {"episodes": len(users), "user_tokens": sum(users),
                  "system_tokens": counts["_system_full"], "mean_output_tokens": mean_out,
                  "cost_usd": round(project_cost(model, users, counts["_system_full"], mean_out), 2)}
    _jwrite(PILOT_DIR / "projection.json", out)
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="zenith.vault.extract")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p = sub.add_parser("poll")
    p.add_argument("--once", action="store_true")
    sub.add_parser("report")
    sub.add_parser("project")
    p = sub.add_parser("kit")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pilot-set")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ingest")
    p.add_argument("src", type=Path)
    p.add_argument("--dest", choices=["pilot", "run"], default="pilot")
    p.add_argument("--model", default=INLINE_MODEL)
    p = sub.add_parser("run")
    p.add_argument("--config", default="opus", choices=[c for c, v in CONFIGS.items() if v[0] == "full"])
    p.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if args.cmd == "pilot":
        cmd_pilot(args)
    elif args.cmd == "poll":
        poll(args.once)
    elif args.cmd == "report":
        rep = pilot_report()
        _jwrite(PILOT_DIR / "report.json", rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "study_counts"}, indent=2))
    elif args.cmd == "project":
        cmd_project(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "kit":
        for path in write_kit(load_taxonomy(), args.out):
            _log(f"wrote {path}")
    elif args.cmd == "pilot-set":
        rows = load_rows()
        ids = pilot_set(rows, args.n)
        _jwrite(PILOT_DIR / f"pilot_set_{args.n}.json", ids)
        paths = export_transcripts([r for r in rows if r["episode_id"] in set(ids)], args.out)
        _log(f"exported {len(paths)} transcripts to {args.out}")
    elif args.cmd == "ingest":
        dest = PILOT_DIR / "inline" if args.dest == "pilot" else EXTRACT_DIR
        for d in ingest(args.src, dest, load_taxonomy(), args.model):
            print(d)


if __name__ == "__main__":
    sys.exit(main())

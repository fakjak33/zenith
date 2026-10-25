"""OPTIONS VAULT corpus helpers — pure functions, no network.

Four jobs, all exercised offline by tests/test_vault.py:

1. SHOW REGISTRY. Which tastylive shows are in scope, under which tastylive.com
   slugs, YouTube title suffixes and playlists they appear.
2. CLASSIFIER. Assign a YouTube video to a show from its title, playlists and
   description. Rules are ordered and each hit records the rule that fired, so
   the classification is auditable (GATE A shows the rule list).
3. MATCHING. Pair tastylive.com episodes with YouTube videos on show, date
   (+/- MATCH_WINDOW_DAYS) and title similarity, one-to-one, with a confidence.
4. VTT. YouTube auto captions are "rolling": every cue repeats the previous line
   and adds a few words, with 10 ms bridge cues in between. parse_vtt collapses
   that into de-duplicated lines with start times; segments() merges lines into
   citation-sized chunks.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Show:
    slug: str                        # canonical id: prefix of every episode_id
    name: str
    tl_slugs: tuple[str, ...] = ()   # tastylive.com /shows/<slug>
    suffixes: tuple[str, ...] = ()   # normalized YouTube "Title | <suffix>" forms
    playlists: tuple[str, ...] = ()  # YouTube playlist ids that hold only this show
    note: str = ""


# Target shows (prompts/vault/01_harvest.md). Suffixes are matched after
# _norm(), so they are lowercase with punctuation stripped. The Skinny on Options
# variants are separate shows: their segments, hosts and eras differ.
SHOWS: tuple[Show, ...] = (
    Show("market-measures", "Market Measures",
         tl_slugs=("market-measures",),
         suffixes=("market measures", "market measure"),
         playlists=("PLPVve34yolHauPYzRmpV0XV8D9AZaimu6",)),
    Show("from-theory-to-practice", "From Theory to Practice",
         tl_slugs=("from-theory-to-practice",),
         suffixes=("from theory to practice", "from theory to practice options")),
    Show("options-jive", "Options Jive",
         tl_slugs=("options-jive",),
         suffixes=("options jive", "research corner"),
         playlists=("PLPVve34yolHb_AA-bnhxCVsxOW9N_WkHS",),
         note="tastylive.com show page retired (redirects to /on-demand)."),
    Show("skinny-math", "The Skinny on Options Math",
         tl_slugs=("the-skinny-on-options-math", "Skinny-on-Options-Math"),
         suffixes=("the skinny on options math", "skinny on options math")),
    Show("skinny-abstract", "The Skinny on Options: Abstract Applications",
         tl_slugs=("the-skinny-on-options-abstract-applications",),
         suffixes=("the skinny on options abstract applications",
                   "skinny on options abstract applications", "abstract applications"),
         playlists=("PLPVve34yolHaO4afeZDMLQ4wfuodRokpv",)),
    # Dropped at GATE A (2026-10-24, user): Best Practices, Tasty Bites and The Skinny
    # on Options: Data Science -- show pages retired and nothing on YouTube.
    # Research-team series found during discovery; kept at GATE A.
    Show("math-check", "The Math Check",
         suffixes=("the math check", "math check"),
         playlists=("PLPVve34yolHZZz5pBAEkdmmjQree-YKQa",),
         note="YouTube-only series (found 2026-10-23)."),
    Show("strange-stats", "Strange Stats",
         suffixes=("strange stats",), playlists=("PLALXAFqsohtc",),
         note="YouTube-only series (found 2026-10-23)."),
    Show("research-team-live", "Trades from the Research Team LIVE",
         tl_slugs=("trades-from-the-research-team-live",),
         suffixes=("trades from the research team live", "trades from the research team")),
)
SHOW_BY_SLUG = {s.slug: s for s in SHOWS}
TL_SLUG_TO_SHOW = {t: s.slug for s in SHOWS for t in s.tl_slugs}

# Playlists that mix shows but are research by construction: a video here that
# no show rule claims lands in the "research" bucket for review, not in a show.
RESEARCH_PLAYLISTS = {
    "PLPVve34yolHZ1lcMnUb8LKOm3NyrvkYZr": "Options Trading Backtesting & Research",
    "PLPVve34yolHZ2SOBQANt4wi1lZcZrF5bv": "tastylive Best of Research 2025",
}
RESEARCH_BUCKET = "research"

MATCH_WINDOW_DAYS = 2          # tastylive air date vs YouTube upload date
MATCH_MIN_CONFIDENCE = 0.40    # below this a candidate pair is not accepted
# A research-bucket video has no show to constrain it, so its title must carry the
# match: 0.75 needs ~all of the shorter title's content words (two shared generic
# words such as "iron condor" score ~0.67 and are rejected).
RESEARCH_MIN_TITLE = 0.75


# --- normalization -----------------------------------------------------------

_STOP = frozenset("a an the and or of to in on for with is are be by at vs our we you your "
                  "how what why when do does it this that s".split())


def _norm(s: str) -> str:
    s = _html.unescape(s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def clean_html(s: str) -> str:
    """Strip tags/entities from a sitemap or page description."""
    s = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(_html.unescape(s).replace("�", "-").split())


def split_title(title: str) -> tuple[str, str]:
    """'Skewed Iron Condor Breakdown | Market Measures' -> (core, normalized suffix)."""
    if "|" not in (title or ""):
        return title or "", ""
    core, _, suffix = title.rpartition("|")
    return core.strip(), _norm(suffix)


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def tokens(s: str) -> set[str]:
    return {_stem(t) for t in _norm(s).split() if t not in _STOP and len(t) > 1}


def title_similarity(a: str, b: str) -> float:
    """0..1 on stemmed content tokens: max(Dice, 0.9 x containment).

    Containment covers the common case of a short tastylive title ('Skewed Iron
    Condors') inside a longer YouTube rewrite ('Skewed Iron Condor Breakdown').
    Character-sequence ratios were rejected: they score unrelated titles ~0.4.
    """
    ta, tb = tokens(split_title(a)[0]), tokens(split_title(b)[0])
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    dice = 2 * inter / (len(ta) + len(tb))
    contain = inter / min(len(ta), len(tb)) if min(len(ta), len(tb)) >= 2 else 0.0
    return round(max(dice, 0.9 * contain), 3)


def is_generic_title(title: str, show: str) -> bool:
    """tastylive titles some episodes just 'Market Measures' (no topic)."""
    s = SHOW_BY_SLUG.get(show)
    return bool(s) and _norm(title) in {_norm(s.name), *s.suffixes}


# --- tastylive URLs and ids --------------------------------------------------

_EP_URL = re.compile(r"/shows/([^/]+)/episodes/([^/?#]+?)-(\d{2})-(\d{2})-(\d{4})(?:-(\d+))?/?$")


def parse_episode_url(url: str) -> dict | None:
    """tastylive episode URL -> {tl_show, slug, air_date}; None if not an episode."""
    m = _EP_URL.search(url or "")
    if not m:
        return None
    tl_show, slug, mm, dd, yyyy, _ = m.groups()
    try:
        d = date(int(yyyy), int(mm), int(dd))
    except ValueError:
        return None
    return {"tl_show": tl_show, "slug": slug, "air_date": d.isoformat()}


def slug_title(slug: str) -> str:
    return slug.replace("-", " ").strip()


def assign_ids(rows: list[dict], existing: dict[str, str] | None = None) -> None:
    """Set row['episode_id'] = '<show>-<YYYY-MM-DD>[-n]' in place.

    ``existing`` maps a row key (tastylive url or youtube id) to an id assigned on
    an earlier run, so ids stay stable when new episodes appear. New rows take the
    next free suffix for their show and day, in (url, youtube_id) order.
    """
    existing = existing or {}
    taken: set[str] = set()
    for r in rows:
        eid = existing.get(row_key(r))
        if eid:
            r["episode_id"] = eid
            taken.add(eid)
    pending = [r for r in rows if not r.get("episode_id")]
    pending.sort(key=lambda r: (r["show"], r["air_date"], r.get("url") or "", r.get("youtube_id") or ""))
    for r in pending:
        base = f"{r['show']}-{r['air_date']}"
        eid, n = base, 1
        while eid in taken:
            n += 1
            eid = f"{base}-{n}"
        r["episode_id"] = eid
        taken.add(eid)


def row_key(r: dict) -> str:
    return r.get("url") or f"yt:{r.get('youtube_id')}"


# --- classifier --------------------------------------------------------------

def classify_video(title: str, playlists: list[str] | tuple = (),
                   description: str = "") -> tuple[str | None, str]:
    """Return (show_slug | 'research' | None, rule_that_fired).

    Ordered rules; the first that fires wins:
      1. suffix      — the text after the last '|' is a known show suffix;
      2. playlist    — the video sits in a single-show playlist;
      3. title       — a show name appears anywhere in the title;
      4. description — a show name appears in the first 400 chars of the description;
      5. research    — the video sits in a mixed research playlist (review bucket);
    otherwise (None, 'unclassified').
    """
    _, suffix = split_title(title)
    if suffix:
        for s in SHOWS:
            if suffix in s.suffixes:
                return s.slug, "suffix"
    pls = set(playlists or ())
    for s in SHOWS:
        if pls & set(s.playlists):
            return s.slug, "playlist"
    nt = f" {_norm(title)} "
    for s in SHOWS:
        if any(f" {x} " in nt for x in s.suffixes if len(x.split()) >= 2):
            return s.slug, "title"
    nd = f" {_norm((description or '')[:400])} "
    for s in SHOWS:
        if any(f" {x} " in nd for x in s.suffixes if len(x.split()) >= 2):
            return s.slug, "description"
    if pls & set(RESEARCH_PLAYLISTS):
        return RESEARCH_BUCKET, "research-playlist"
    return None, "unclassified"


# --- matching ----------------------------------------------------------------

def _days(a: str, b: str) -> int:
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def match_score(ep: dict, vid: dict, window: int = MATCH_WINDOW_DAYS) -> float:
    """Confidence 0..1 that tastylive episode ``ep`` is YouTube video ``vid``.

    Both carry 'show', a date ('air_date' / 'upload_date', ISO) and 'title';
    'description' is optional. Different show or outside the window -> 0.
    A generic tastylive title ('Market Measures') carries no topic, so the
    date alone decides and confidence is capped at 0.55.
    """
    research = vid.get("show") == RESEARCH_BUCKET
    if (ep["show"] != vid.get("show") and not research) or not vid.get("upload_date"):
        return 0.0
    dd = _days(ep["air_date"], vid["upload_date"])
    if dd > window:
        return 0.0
    date_s = 1.0 - dd / (window + 1)
    generic = is_generic_title(ep.get("title", ""), ep["show"])
    if generic:
        return 0.0 if research else round(0.55 * date_s, 3)
    t = max(title_similarity(ep.get("title", ""), vid.get("title", "")),
            title_similarity(slug_title(ep.get("slug", "")), vid.get("title", "")))
    if research and t < RESEARCH_MIN_TITLE:
        return 0.0     # a show-less research video needs the title to say which episode
    desc = 0.0
    if ep.get("description") and vid.get("description"):
        te, tv = tokens(ep["description"]), tokens(vid["description"][:600])
        desc = len(te & tv) / len(te) if te else 0.0
    return round(0.55 * t + 0.15 * min(desc, 1.0) + 0.30 * date_s, 3)


def match_episodes(episodes: list[dict], videos: list[dict],
                   window: int = MATCH_WINDOW_DAYS,
                   min_conf: float = MATCH_MIN_CONFIDENCE) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one assignment by descending score.

    Returns {episode_index: (video_index, confidence)}. A generic-titled episode
    is accepted on date alone only when it is the sole same-show candidate
    within the window on both sides (no ambiguity).
    """
    by_show: dict[str, list[int]] = {}
    for j, v in enumerate(videos):
        if v.get("show") and v.get("upload_date"):
            by_show.setdefault(v["show"], []).append(j)
    research = by_show.get(RESEARCH_BUCKET, [])
    pairs: list[tuple[float, int, int]] = []
    ep_cands: dict[int, int] = {}
    vid_cands: dict[int, int] = {}
    for i, ep in enumerate(episodes):
        for j in by_show.get(ep["show"], []) + research:
            s = match_score(ep, videos[j], window)
            if s > 0:
                pairs.append((s, i, j))
                ep_cands[i] = ep_cands.get(i, 0) + 1
                vid_cands[j] = vid_cands.get(j, 0) + 1
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    out: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for s, i, j in pairs:
        if i in out or j in used:
            continue
        generic = is_generic_title(episodes[i].get("title", ""), episodes[i]["show"])
        if generic:
            if ep_cands.get(i) != 1 or vid_cands.get(j) != 1:
                continue
        elif s < min_conf:
            continue
        out[i] = (j, s)
        used.add(j)
    return out


# --- VTT ---------------------------------------------------------------------

_CUE_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})[.,](\d{3})")
_TAG = re.compile(r"<[^>]+>")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text: str) -> list[dict]:
    """WebVTT -> [{'t': seconds, 'text': line}] with rolling repeats collapsed.

    Bridge cues shorter than 50 ms are skipped; a line equal to one of the last
    three emitted lines is a repeat. Bracketed sound tags ([Music]) are dropped.
    """
    out: list[dict] = []
    recent: list[str] = []
    # Cues are separated by truly empty lines; auto-caption cues CONTAIN
    # whitespace-only lines, so a looser split would detach text from its timing.
    blocks = (text or "").lstrip("﻿").replace("\r\n", "\n").split("\n\n")
    for b in blocks:
        lines = b.strip().splitlines()
        k = next((n for n, ln in enumerate(lines) if _CUE_TIME.search(ln)), None)
        if k is None:
            continue
        m = _CUE_TIME.search(lines[k])
        start = _secs(*m.groups()[:4])
        end = _secs(*m.groups()[4:])
        if end - start < 0.05:
            continue
        for ln in lines[k + 1:]:
            t = " ".join(_html.unescape(_TAG.sub("", ln)).split())
            t = re.sub(r"\[(?:music|applause|laughter|__)\]", "", t, flags=re.I).strip()
            if not t or t in recent:
                continue
            out.append({"t": round(start, 1), "text": t})
            recent = (recent + [t])[-3:]
    return out


def segments(lines: list[dict], max_secs: float = 20.0, max_chars: int = 320) -> list[dict]:
    """Merge caption lines into chunks of <= max_secs / max_chars; t = chunk start (int s)."""
    out: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        if cur and (ln["t"] - cur["t"] < max_secs) and len(cur["text"]) + len(ln["text"]) < max_chars:
            cur["text"] += " " + ln["text"]
        else:
            cur = {"t": int(ln["t"]), "text": ln["text"]}
            out.append(cur)
    return out


def strip_boilerplate(texts: dict[str, str], frac: float = 0.3) -> dict[str, str]:
    """Drop lines that recur in more than ``frac`` of the documents.

    tastylive pages have no <main>/<article>, so extracted text carries the
    Learn Center sidebar and the legal footer. Those lines repeat across the
    site; article lines do not. Needs a corpus (>= 5 docs) to be meaningful.
    """
    if len(texts) < 5:
        return dict(texts)
    df: dict[str, int] = {}
    for t in texts.values():
        for ln in set(t.splitlines()):
            df[ln] = df.get(ln, 0) + 1
    cut = frac * len(texts)
    return {k: "\n".join(ln for ln in t.splitlines() if df.get(ln, 0) <= cut)
            for k, t in texts.items()}


def fmt_ts(seconds: float) -> str:
    """Seconds -> 'mm:ss' (minutes may exceed 59 for long shows)."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"

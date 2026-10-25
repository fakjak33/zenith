"""OPTIONS VAULT Phase 01 — harvest the tastylive corpus (CLI).

    python -m zenith.vault.harvest discover [--refresh] [--no-meta]
    python -m zenith.vault.harvest fetch    [--show S] [--since YYYY] [--limit N]
    python -m zenith.vault.harvest pages    [--kind concepts|definitions|news] [--limit N] [--all-news]
    python -m zenith.vault.harvest report

DISCOVER (metadata only). tastylive.com's sitemap-episode.xml is the episode
inventory: it lists every episode back to 2012 (title, blurb, tags, date), the
show pages page through exactly the same set, and it is one request instead of
~250 page fetches. Show pages are still read from page 1 until they reach
episodes the sitemap already has, which picks up the newest episodes. YouTube
is enumerated with yt-dlp --flat-playlist (uploads, streams, every playlist);
videos the classifier assigns to a target show then get full metadata (upload
date, description, caption tracks) because matching needs exact dates.

FETCH. YouTube captions only (manual first, then auto "en-orig"), one video at a
time with yt-dlp's request/subtitle sleeps, a pause between videos and a hard
stop after repeated 429s. Incremental: rows whose raw/<episode_id>.json exists
are skipped. Paid (Apify) and local-ASR fallbacks are NOT wired in: both need
the user's go-ahead (prompts/vault/01_harvest.md).

Politeness follows zenith/index/podcasts.py: one source at a time, backoff, no
hammering. Third-party text is written only under data/vault/cache/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .. import fetch
from ..config import VAULT_CACHE_DIR
from . import corpus, load, save

SITEMAP = "https://www.tastylive.com/sitemap-episode.xml"
SITE = "https://www.tastylive.com"
CHANNEL = "https://www.youtube.com/@tastyliveshow"
CHANNEL_TABS = ("videos", "streams", "playlists")

DISC_DIR = VAULT_CACHE_DIR / "discovery"
FLAT_DIR = DISC_DIR / "yt_flat"
META_DIR = DISC_DIR / "yt_meta"
RAW_DIR = VAULT_CACHE_DIR / "raw"
PAGES_DIR = VAULT_CACHE_DIR / "pages"
TMP_DIR = VAULT_CACHE_DIR / "tmp"

_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9",
       "v": "http://www.google.com/schemas/sitemap-video/1.1"}

# Seconds. yt-dlp sleeps between its own requests; we add a jittered pause
# between videos. Measured cost per caption fetch is ~6-9 s end to end.
YT_SLEEP_REQUESTS = 1.0
YT_SLEEP_SUBTITLES = 2.0
YT_PAUSE_BETWEEN = (1.5, 3.5)
YT_MAX_429 = 3
PAGE_PAUSE = (1.0, 2.0)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jread(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _get(url: str):
    """Zenith UA first, browser UA as the retry (like fetch.parse_feed)."""
    r = fetch.get(url, timeout=60)
    return r if r is not None else fetch.get(url, timeout=60, browser_ua=True)


def _log(msg: str) -> None:
    print(f"[vault] {msg}", flush=True)


# =============================================================================
# DISCOVER — tastylive.com
# =============================================================================

def tastylive_sitemap() -> list[dict]:
    """Every target-show episode in the sitemap (all shows kept in the cache)."""
    r = _get(SITEMAP)
    if r is None:
        raise RuntimeError("sitemap-episode.xml unreachable")
    root = ET.fromstring(r.content)
    rows = []
    for u in root.findall("s:url", _NS):
        loc = u.findtext("s:loc", namespaces=_NS) or ""
        p = corpus.parse_episode_url(loc)
        if not p:
            continue
        v = u.find("v:video", _NS)
        rows.append({
            "url": loc, **p,
            "title": ((v.findtext("v:title", namespaces=_NS) if v is not None else "")
                      or corpus.slug_title(p["slug"]).title()),
            "description": corpus.clean_html(v.findtext("v:description", namespaces=_NS) or "")
            if v is not None else "",
            "tags": [t.text for t in v.findall("v:tag", _NS)] if v is not None else [],
            "in_sitemap": True,
        })
    return rows


def tastylive_show_pages(tl_slug: str, known: set[str], max_pages: int = 120) -> list[dict]:
    """Episodes from /shows/<slug>?page=N until a page adds nothing new."""
    from bs4 import BeautifulSoup

    out, seen = [], set()
    for page in range(1, max_pages + 1):
        r = _get(f"{SITE}/shows/{tl_slug}?page={page}")
        if r is None or "expiredRedirect" in r.url or f"/shows/{tl_slug}" not in r.url:
            break
        soup = BeautifulSoup(r.text, "lxml")
        fresh = 0
        for a in soup.select(f'a[href*="/shows/{tl_slug}/episodes/"]'):
            href = a.get("href", "").split("?")[0]
            url = href if href.startswith("http") else SITE + href
            p = corpus.parse_episode_url(url)
            if not p or url in seen:
                continue
            seen.add(url)
            card = a.find_parent(attrs={"data-testid": True})
            title = ""
            if card is not None:
                h = card.find(["h2", "h3", "h4", "p"])
                title = h.get_text(" ", strip=True) if h else ""
            if url not in known:
                fresh += 1
                out.append({"url": url, **p, "title": title or corpus.slug_title(p["slug"]).title(),
                            "description": "", "tags": [], "in_sitemap": False})
        if not fresh:
            break
        time.sleep(random.uniform(*PAGE_PAUSE))
    return out


def discover_tastylive() -> list[dict]:
    rows = tastylive_sitemap()
    _jwrite(DISC_DIR / "tastylive_sitemap.json", rows)
    known = {r["url"] for r in rows}
    extra = []
    for s in corpus.SHOWS:
        for tl in s.tl_slugs:
            got = tastylive_show_pages(tl, known)
            if got:
                _log(f"show pages {tl}: +{len(got)} episodes not in sitemap")
            extra += got
            known |= {r["url"] for r in got}
    rows += extra
    _jwrite(DISC_DIR / "tastylive.json", rows)
    return rows


# =============================================================================
# DISCOVER — YouTube
# =============================================================================

def _ydl(extra: dict | None = None):
    import yt_dlp  # research-only dependency (requirements-vault.txt)

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True,
            "sleep_interval_requests": YT_SLEEP_REQUESTS,
            "js_runtimes": {"node": {}, "deno": {}}, "ignoreerrors": True}
    opts.update(extra or {})
    return yt_dlp.YoutubeDL(opts)


_FLAT_KEYS = ("id", "title", "duration", "playlist_id", "playlist_title", "availability")


def _flat_list(url: str, name: str, refresh: bool) -> list[dict]:
    """yt-dlp --flat-playlist for one tab/playlist, cached as jsonl."""
    path = FLAT_DIR / f"{name}.jsonl"
    if path.exists() and not refresh:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    with _ydl({"extract_flat": "in_playlist"}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    rows = [{k: e.get(k) for k in _FLAT_KEYS} | {"playlist_id": info.get("id"),
                                                  "playlist_title": info.get("title")}
            for e in (info.get("entries") or []) if e]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rows


def youtube_flat(refresh: bool = False) -> tuple[dict[str, dict], list[dict]]:
    """{video_id: {id, title, duration, playlists, tabs}} and the playlist list."""
    vids: dict[str, dict] = {}
    for tab in ("videos", "streams"):
        for e in _flat_list(f"{CHANNEL}/{tab}", tab, refresh):
            if e.get("id") and e.get("title"):
                v = vids.setdefault(e["id"], {"id": e["id"], "title": e["title"],
                                              "duration": e.get("duration"),
                                              "playlists": [], "tabs": []})
                v["tabs"].append(tab)
    playlists = [p for p in _flat_list(f"{CHANNEL}/playlists", "playlists", refresh) if p.get("id")]
    for p in playlists:
        for e in _flat_list(f"https://www.youtube.com/playlist?list={p['id']}", f"pl_{p['id']}", refresh):
            if not e.get("id") or not e.get("title"):
                continue          # private / deleted entries carry no title
            v = vids.setdefault(e["id"], {"id": e["id"], "title": e["title"],
                                          "duration": e.get("duration"),
                                          "playlists": [], "tabs": []})
            if p["id"] not in v["playlists"]:
                v["playlists"].append(p["id"])
    return vids, playlists


_META_KEYS = ("id", "title", "upload_date", "release_date", "duration", "availability",
              "live_status")


def youtube_meta(vid: str) -> dict | None:
    """Full metadata for one video, cached in yt_meta/<id>.json."""
    path = META_DIR / f"{vid}.json"
    cached = _jread(path, None)
    if cached is not None:
        return cached
    # metadata only: skipping the player JS/configs and stream manifests cuts a
    # call from ~7 s to ~4 s and still returns upload date, description, captions
    light = {"extractor_args": {"youtube": {"skip": ["dash", "hls", "translated_subs"],
                                            "player_skip": ["js", "configs"]}}}
    with _ydl(light) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    if not info:
        meta = {"id": vid, "error": "unavailable"}
    else:
        meta = {k: info.get(k) for k in _META_KEYS}
        meta["description"] = (info.get("description") or "")[:3000]
        meta["subs_manual"] = sorted(k for k in (info.get("subtitles") or {}) if k.startswith("en"))
        meta["subs_auto"] = sorted(k for k in (info.get("automatic_captions") or {})
                                   if k in ("en", "en-orig", "en-US"))
    _jwrite(path, meta)
    return meta


def _iso(yyyymmdd: str | None) -> str | None:
    if not yyyymmdd or len(yyyymmdd) != 8:
        return None
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def discover_youtube(refresh: bool = False, with_meta: bool = True) -> list[dict]:
    vids, playlists = youtube_flat(refresh)
    _jwrite(DISC_DIR / "yt_playlists.json",
            [{"id": p["id"], "title": p.get("title")} for p in playlists])
    out = []
    for v in vids.values():
        show, rule = corpus.classify_video(v["title"], v["playlists"])
        out.append(v | {"show": show, "rule": rule})
    todo = [v for v in out if v["show"]]
    if with_meta:
        missing = [v for v in todo if not (META_DIR / f"{v['id']}.json").exists()]
        _log(f"youtube: {len(vids)} videos, {len(todo)} classified, "
             f"{len(missing)} need full metadata")
        for n, v in enumerate(missing, 1):
            youtube_meta(v["id"])
            if n % 50 == 0:
                _log(f"  metadata {n}/{len(missing)}")
            time.sleep(random.uniform(0.3, 0.8))
    for v in out:
        m = _jread(META_DIR / f"{v['id']}.json", None)
        if m:
            v["upload_date"] = _iso(m.get("upload_date") or m.get("release_date"))
            v["description"] = m.get("description", "")
            v["subs_manual"] = m.get("subs_manual", [])
            v["subs_auto"] = m.get("subs_auto", [])
            v["availability"] = m.get("availability") or m.get("error")
            if v["show"] in (None, corpus.RESEARCH_BUCKET):   # description rule needs metadata
                v["show"], v["rule"] = corpus.classify_video(v["title"], v["playlists"],
                                                             v["description"])
    _jwrite(DISC_DIR / "youtube.json", out)
    return out


# =============================================================================
# DISCOVER — manifest + coverage + review
# =============================================================================

def build_manifest(tl_rows: list[dict], yt_rows: list[dict]) -> dict:
    eps = []
    for r in tl_rows:
        show = corpus.TL_SLUG_TO_SHOW.get(r["tl_show"])
        if show:
            eps.append(r | {"show": show})
    # research-bucket videos may match any show's episode on title (corpus.match_score);
    # unmatched ones get their own show='research' rows so their captions are fetched
    # and Phase 02 can attribute them by content (GATE A, user)
    vids = [v for v in yt_rows
            if v.get("show") in corpus.SHOW_BY_SLUG or v.get("show") == corpus.RESEARCH_BUCKET]
    matches = corpus.match_episodes(eps, vids)
    matched_vids = {j for j, _ in matches.values()}

    old = {corpus.row_key(r): r for r in load("manifest", {}).get("episodes", [])}
    rows = []
    for i, ep in enumerate(eps):
        j, conf = matches.get(i, (None, 0.0))
        v = vids[j] if j is not None else {}
        rows.append(_row(ep["show"], ep["air_date"], ep["title"], ep["url"], v, conf,
                         tl={"slug": ep["slug"], "tags": ep.get("tags", []),
                             "in_sitemap": ep.get("in_sitemap", True)}))
    for j, v in enumerate(vids):
        if j in matched_vids or not v.get("upload_date"):
            continue
        rows.append(_row(v["show"], v["upload_date"], corpus.split_title(v["title"])[0],
                         "", v, 0.0, tl=None))
    # carry fetch / extract state from the previous manifest
    for r in rows:
        prev = old.get(corpus.row_key(r))
        if prev:
            for k in ("transcript", "extract"):
                if prev.get(k):
                    r[k] = prev[k]
    corpus.assign_ids(rows, {k: r["episode_id"] for k, r in old.items() if r.get("episode_id")})
    rows.sort(key=lambda r: (r["show"], r["air_date"], r["episode_id"]))
    return {"generated": _now(), "shows": [s.slug for s in corpus.SHOWS], "episodes": rows}


def _row(show, air_date, title, url, v, conf, tl) -> dict:
    yid = v.get("id", "")
    return {
        "episode_id": "", "show": show, "air_date": air_date, "title": title, "url": url,
        "youtube_id": yid, "duration_s": int(v.get("duration") or 0),
        "match_confidence": conf if url and yid else (1.0 if yid else 0.0),
        "transcript": {"source": "none", "chars": 0},
        "extract": {"status": "pending", "version": "", "model": ""},
        # additions (Phase 01)
        "origin": "both" if url and yid else ("tastylive" if url else "youtube"),
        "yt_title": v.get("title", ""), "yt_upload_date": v.get("upload_date"),
        "yt_rule": v.get("rule", ""),
        "captions": {"manual": bool(v.get("subs_manual")), "auto": bool(v.get("subs_auto"))}
        if yid else {"manual": False, "auto": False},
        "tl": tl,
    }


_RESEARCHY = re.compile(r"\b(stud(y|ied|ies)|data|backtest\w*|research|years of|tested|"
                        r"statistic\w*|probabilit\w*|win rate|dte|strangle|condor|iv ?r)\b", re.I)


def build_review(yt_rows: list[dict], manifest: dict) -> dict:
    unclassified = [v for v in yt_rows if v.get("show") is None]
    researchy = [v for v in unclassified if _RESEARCHY.search(v["title"])
                 and "live trading" not in v["title"].lower()]
    return {
        "generated": _now(),
        "unclassified_total": len(unclassified),
        "unclassified_researchy": [{"id": v["id"], "title": v["title"], "playlists": v["playlists"]}
                                   for v in researchy],
        "research_unattributed": [{"episode_id": r["episode_id"], "title": r["yt_title"]}
                                  for r in manifest["episodes"] if r["show"] == corpus.RESEARCH_BUCKET],
        "youtube_only": [{"episode_id": r["episode_id"], "title": r["yt_title"]}
                         for r in manifest["episodes"] if r["origin"] == "youtube"],
    }


def build_coverage(manifest: dict, yt_rows: list[dict]) -> dict:
    cells: dict[tuple, Counter] = defaultdict(Counter)
    for r in manifest["episodes"]:
        c = cells[(r["show"], r["air_date"][:4])]
        if r["url"]:
            c["tastylive"] += 1
        if r["youtube_id"]:
            c["youtube"] += 1
        if r["origin"] == "both":       # 'tastylive'/'youtube' origins are derived below
            c["both"] += 1
        if r["youtube_id"] and (r["captions"]["manual"] or r["captions"]["auto"]):
            c["captioned"] += 1
        if r["transcript"]["source"] != "none":
            c["transcript"] += 1
    shows = []
    research = corpus.Show(corpus.RESEARCH_BUCKET, "Research videos (show unattributed)")
    for s in (*corpus.SHOWS, research):
        years = sorted({y for (sh, y) in cells if sh == s.slug})
        rows = []
        for y in years:
            c = cells[(s.slug, y)]
            total = c["both"] + (c["tastylive"] - c["both"]) + (c["youtube"] - c["both"])
            rows.append({"year": int(y), "episodes": total, "tastylive": c["tastylive"],
                         "youtube": c["youtube"], "matched": c["both"],
                         "tastylive_only": c["tastylive"] - c["both"],
                         "youtube_only": c["youtube"] - c["both"], "captioned": c["captioned"],
                         "transcript": c["transcript"],
                         "transcript_pct": round(100 * c["transcript"] / total, 1) if total else 0.0})
        tot = {k: sum(r[k] for r in rows) for k in ("episodes", "tastylive", "youtube", "matched",
                                                     "tastylive_only", "youtube_only",
                                                     "captioned", "transcript")}
        tot["transcript_pct"] = round(100 * tot["transcript"] / tot["episodes"], 1) if tot["episodes"] else 0.0
        shows.append({"show": s.slug, "name": s.name, "years": rows, "total": tot})
    rules = Counter(v.get("rule") for v in yt_rows)
    prev = load("coverage", {})
    out = {"generated": _now(), "shows": shows,
           "youtube": {"videos": len(yt_rows), "rules": dict(rules),
                       "classified_target": sum(1 for v in yt_rows
                                                if v.get("show") in corpus.SHOW_BY_SLUG)},
           "match": {"window_days": corpus.MATCH_WINDOW_DAYS,
                     "min_confidence": corpus.MATCH_MIN_CONFIDENCE}}
    for k in ("transcripts", "failures"):
        if k in prev:
            out[k] = prev[k]
    return out


def run_discover(refresh: bool = False, with_meta: bool = True) -> None:
    tl = discover_tastylive()
    _log(f"tastylive: {len(tl)} episodes in sitemap + show pages")
    yt = discover_youtube(refresh, with_meta)
    man = build_manifest(tl, yt)
    save("manifest", man)
    save("review", build_review(yt, man))
    save("coverage", build_coverage(man, yt))
    _log(f"manifest: {len(man['episodes'])} rows -> data/vault/manifest.json")
    inv = page_inventory()
    save("pages", {"generated": _now(), "counts": dict(Counter(p["kind"] for p in inv)),
                   "pages": inv})
    _log(f"pages inventory: {dict(Counter(p['kind'] for p in inv))}")


# =============================================================================
# FETCH — YouTube captions
# =============================================================================

class RateLimited(RuntimeError):
    pass


def _pick_track(row: dict, meta: dict) -> tuple[str, str] | None:
    """(kind, lang): manual English first, then original-language auto captions."""
    manual = meta.get("subs_manual") or []
    if manual:
        return "yt_manual", ("en" if "en" in manual else manual[0])
    auto = meta.get("subs_auto") or []
    for lang in ("en-orig", "en", "en-US"):
        if lang in auto:
            return "yt_auto", lang
    return None


def fetch_captions(row: dict) -> dict:
    """Download + parse one video's captions into raw/<episode_id>.json."""
    vid = row["youtube_id"]
    meta = youtube_meta(vid) or {}
    if meta.get("error"):
        return {"source": "none", "chars": 0, "error": meta["error"]}
    track = _pick_track(row, meta)
    if not track:
        return {"source": "none", "chars": 0, "error": "no_captions"}
    kind, lang = track
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for f in TMP_DIR.glob(f"{vid}.*"):
        f.unlink()
    opts = {"writesubtitles": kind == "yt_manual", "writeautomaticsub": kind == "yt_auto",
            "subtitleslangs": [lang], "subtitlesformat": "vtt",
            "sleep_interval_subtitles": YT_SLEEP_SUBTITLES,
            "outtmpl": str(TMP_DIR / "%(id)s.%(ext)s"), "ignoreerrors": False}
    import yt_dlp

    try:
        with _ydl(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except yt_dlp.utils.DownloadError as ex:
        msg = str(ex)
        if "429" in msg or "Too Many Requests" in msg:
            raise RateLimited(msg) from ex
        reason = ("private" if "Private video" in msg else
                  "unavailable" if "unavailable" in msg.lower() else "download_error")
        return {"source": "none", "chars": 0, "error": reason}
    files = sorted(TMP_DIR.glob(f"{vid}.*.vtt"))
    if not files:
        return {"source": "none", "chars": 0, "error": "no_file"}
    lines = corpus.parse_vtt(files[0].read_text(encoding="utf-8", errors="replace"))
    for f in files:
        f.unlink()
    if not lines:
        return {"source": "none", "chars": 0, "error": "empty"}
    segs = corpus.segments(lines)
    chars = sum(len(s["text"]) for s in segs)
    _jwrite(RAW_DIR / f"{row['episode_id']}.json",
            {"episode_id": row["episode_id"], "source": kind, "lang": lang,
             "youtube_id": vid, "fetched": _now(), "segments": segs})
    return {"source": kind, "chars": chars, "lang": lang}


def run_fetch(show: str | None = None, since: int | None = None, limit: int | None = None) -> None:
    man = load("manifest", {})
    rows = man.get("episodes", [])
    todo = [r for r in rows if r["youtube_id"]
            and (show is None or r["show"] == show)
            and (since is None or int(r["air_date"][:4]) >= since)
            and not (RAW_DIR / f"{r['episode_id']}.json").exists()
            and r["transcript"].get("error") not in ("private", "unavailable", "no_captions")]
    todo.sort(key=lambda r: r["air_date"], reverse=True)
    if limit:
        todo = todo[:limit]
    _log(f"fetch: {len(todo)} videos to fetch")
    hits429, done = 0, 0
    for n, r in enumerate(todo, 1):
        try:
            r["transcript"] = fetch_captions(r)
            hits429 = 0
        except RateLimited:
            hits429 += 1
            wait = 60 * 2 ** hits429
            _log(f"  429 from YouTube ({hits429}/{YT_MAX_429}); backing off {wait}s")
            if hits429 >= YT_MAX_429:
                _log("  stopping: YouTube is rate-limiting. Re-run later (resumable).")
                break
            time.sleep(wait)
            continue
        done += 1
        t = r["transcript"]
        _log(f"  [{n}/{len(todo)}] {r['episode_id']}: {t['source']} {t['chars']} chars"
             + (f" ({t['error']})" if t.get("error") else ""))
        if done % 10 == 0:
            save("manifest", man)
        time.sleep(random.uniform(*YT_PAUSE_BETWEEN))
    save("manifest", man)
    run_report(quiet=True)


# =============================================================================
# PAGES — written sources
# =============================================================================

def page_inventory() -> list[dict]:
    """Learn Center concepts, definitions, news/insights and help-center URLs."""
    inv: list[dict] = []
    r = _get(f"{SITE}/concepts-strategies/strangle")
    if r is not None:
        h = r.text.replace("\\u002F", "/").replace("\\/", "/")
        for slug in sorted(set(re.findall(r"/concepts-strategies/([a-z0-9-]+)", h))):
            inv.append({"kind": "concepts", "slug": f"concepts-{slug}",
                        "url": f"{SITE}/concepts-strategies/{slug}"})
    for kind, sm, prefix in (("definitions", "sitemap-definition.xml", "def"),
                             ("news", "sitemap-blog_post.xml", "news")):
        r = _get(f"{SITE}/{sm}")
        if r is None:
            continue
        root = ET.fromstring(r.content)
        for u in root.findall("s:url", _NS):
            loc = u.findtext("s:loc", namespaces=_NS) or ""
            inv.append({"kind": kind, "slug": f"{prefix}-{loc.rstrip('/').rsplit('/', 1)[-1]}",
                        "url": loc, "lastmod": (u.findtext("s:lastmod", namespaces=_NS) or "")[:10]})
    r = _get("https://support.tastytrade.com/support/s/sitemap-topicarticle-1.xml")
    if r is not None:
        root = ET.fromstring(r.content)
        for u in root.findall("s:url", _NS):
            loc = u.findtext("s:loc", namespaces=_NS) or ""
            inv.append({"kind": "help", "slug": f"help-{loc.rstrip('/').rsplit('/', 1)[-1]}",
                        "url": loc, "lastmod": (u.findtext("s:lastmod", namespaces=_NS) or "")[:10]})
    return inv


def _main_text(html: str) -> tuple[str, str]:
    """(title, main-content text): <main>/<article> only, chrome stripped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "").split("|")[0].strip()
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]):
        t.decompose()
    node = soup.find("article") or soup.find("main") or soup.body
    text = node.get_text("\n", strip=True) if node else ""
    text = "\n".join(ln for ln in (l.strip().replace("​", "") for l in text.splitlines()) if ln)
    return title, text


# News/insights is mostly market commentary; GATE A (user) kept only the
# research-style articles, selected by URL slug (~260 of ~2,900).
_NEWS_RESEARCH = re.compile(
    r"(data-science|stud(y|ies)|research|backtest|strangle|iron-condor|straddle|0-dte|zero-dte|"
    r"dte|implied-volatility|iv-rank|vix-|theta|delta|gamma|vega|expected-move|options-|option-|"
    r"credit|premium|roll|assignment|margin|buying-power|probability|win-rate|put-spread|"
    r"call-spread|covered-call|wheel|calendar|diagonal|butterfly|skew)")


def wanted_page(p: dict, all_news: bool = False) -> bool:
    if p["kind"] == "help":
        return False          # JS-rendered; skipped at GATE A (needs Playwright or Apify)
    if p["kind"] == "news" and not all_news:
        return bool(_NEWS_RESEARCH.search(p["slug"].removeprefix("news-")))
    return True


def run_pages(kind: str | None = None, limit: int | None = None, all_news: bool = False) -> None:
    inv = load("pages", {}).get("pages") or page_inventory()
    todo = [p for p in inv if (kind is None or p["kind"] == kind) and wanted_page(p, all_news)
            and not (PAGES_DIR / f"{p['slug']}.json").exists()]
    if limit:
        todo = todo[:limit]
    _log(f"pages: {len(todo)} to fetch (help-center pages need a JS render; not fetched)")
    for n, p in enumerate(todo, 1):
        r = _get(p["url"])
        if r is None:
            _log(f"  [{n}] {p['slug']}: failed")
            continue
        r.encoding = r.apparent_encoding or r.encoding
        title, text = _main_text(r.text)
        _jwrite(PAGES_DIR / f"{p['slug']}.json",
                {"url": p["url"], "fetched": _now(), "title": title, "text": text})
        if n % 25 == 0:
            _log(f"  pages {n}/{len(todo)}")
        time.sleep(random.uniform(*PAGE_PAUSE))
    clean_pages()


def clean_pages() -> dict:
    """Strip site chrome from every cached page and mark duplicate articles.

    Keeps the extracted text as ``text_raw`` so cleaning is re-runnable. Many
    /definitions/ URLs serve the same article as a /concepts-strategies/ page;
    the later slug (concepts sort first) gets ``duplicate_of``.
    """
    docs = {p.stem: _jread(p, None) for p in sorted(PAGES_DIR.glob("*.json"))}
    docs = {k: d for k, d in docs.items() if d}
    clean = corpus.strip_boilerplate({k: d.get("text_raw", d["text"]) for k, d in docs.items()})
    seen: dict[str, str] = {}
    dups = 0
    for k in sorted(docs, key=lambda s: (not s.startswith("concepts-"), s)):
        d = docs[k]
        d.setdefault("text_raw", d["text"])
        d["text"] = clean[k]
        h = hash(d["text"])
        d["duplicate_of"] = seen.get(h)
        dups += bool(d["duplicate_of"])
        seen.setdefault(h, k)
        _jwrite(PAGES_DIR / f"{k}.json", d)
    stats = {"pages": len(docs), "duplicates": dups,
             "median_chars": int(statistics.median([len(d["text"]) for d in docs.values()])) if docs else 0}
    _log(f"pages cleaned: {stats}")
    return stats


# =============================================================================
# REPORT
# =============================================================================

def run_report(quiet: bool = False) -> dict:
    man = load("manifest", {})
    rows = man.get("episodes", [])
    for r in rows:              # the cache is the truth for transcript state
        raw = _jread(RAW_DIR / f"{r['episode_id']}.json", None)
        if raw and r["transcript"]["source"] == "none":
            r["transcript"] = {"source": raw["source"], "chars": sum(len(s["text"]) for s in raw["segments"])}
    save("manifest", man)
    yt = _jread(DISC_DIR / "youtube.json", [])
    cov = build_coverage(man, yt)
    have = [r["transcript"]["chars"] for r in rows if r["transcript"]["source"] != "none"]
    fails = Counter(r["transcript"].get("error") for r in rows
                    if r["youtube_id"] and r["transcript"]["source"] == "none" and r["transcript"].get("error"))
    no_video = Counter(r["show"] for r in rows if not r["youtube_id"])
    cov["transcripts"] = {
        "n": len(have), "median_chars": int(statistics.median(have)) if have else 0,
        "by_source": dict(Counter(r["transcript"]["source"] for r in rows
                                  if r["transcript"]["source"] != "none")),
    }
    cov["failures"] = {"fetch": dict(fails), "no_youtube_video": dict(no_video),
                       "not_attempted": sum(1 for r in rows if r["youtube_id"]
                                            and r["transcript"]["source"] == "none"
                                            and not r["transcript"].get("error"))}
    save("coverage", cov)
    if not quiet:
        print_coverage(cov)
    return cov


def print_coverage(cov: dict) -> None:
    hdr = f"{'show':<26}{'year':>5}{'eps':>6}{'tl':>6}{'yt':>5}{'match':>6}{'tl-only':>8}{'yt-only':>8}{'capt':>6}{'txt':>5}{'txt%':>7}"
    print(hdr)
    for s in cov["shows"]:
        for y in s["years"]:
            print(f"{s['show']:<26}{y['year']:>5}{y['episodes']:>6}{y['tastylive']:>6}{y['youtube']:>5}"
                  f"{y['matched']:>6}{y['tastylive_only']:>8}{y['youtube_only']:>8}{y['captioned']:>6}"
                  f"{y['transcript']:>5}{y['transcript_pct']:>7}")
        t = s["total"]
        print(f"{s['show'] + ' TOTAL':<26}{'':>5}{t['episodes']:>6}{t['tastylive']:>6}{t['youtube']:>5}"
              f"{t['matched']:>6}{t['tastylive_only']:>8}{t['youtube_only']:>8}{t['captioned']:>6}"
              f"{t['transcript']:>5}{t['transcript_pct']:>7}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="OPTIONS VAULT Phase 01 harvest")
    ap.add_argument("action", choices=["discover", "fetch", "pages", "report"])
    ap.add_argument("--show")
    ap.add_argument("--since", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--kind", choices=["concepts", "definitions", "news", "help"])
    ap.add_argument("--refresh", action="store_true", help="re-enumerate YouTube lists")
    ap.add_argument("--no-meta", action="store_true", help="skip per-video metadata")
    ap.add_argument("--all-news", action="store_true", help="every news article, not the research subset")
    a = ap.parse_args()
    if a.action == "discover":
        run_discover(a.refresh, not a.no_meta)
    elif a.action == "fetch":
        run_fetch(a.show, a.since, a.limit)
    elif a.action == "pages":
        run_pages(a.kind, a.limit, a.all_news)
    else:
        run_report()


if __name__ == "__main__":
    main()

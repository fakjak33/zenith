"""Daily scrape orchestrator: fetch → dedupe → classify → store.

Run:  python -m zenith.scrape
Designed to be invoked nightly by the GitHub Action, which commits the updated
data/ JSON back to the repo. Idempotent within a day; items already in the 'seen'
index are skipped so nothing repeats across days.
"""

from __future__ import annotations

import html
import re
import time

from . import store, classify as classifier, apify, firecrawl, extract
from .config import MAX_ITEMS_PER_SOURCE, CLASSIFY_FETCH, CLASSIFY_MAX_FETCH
from .fetch import parse_feed, get_html
from .sources import enabled_sources


def _clean_text(s: str, limit: int = 320) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")        # strip tags
    s = html.unescape(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:limit] + ("…" if len(s) > limit else "")


def _published(entry) -> str:
    for key in ("published", "updated", "created"):
        if entry.get(key):
            return str(entry.get(key))
    return ""


def _entries_for(src) -> tuple[list | None, int, str]:
    """Return (entries, total_seen, via) for a source, or (None, 0, error).

    ``via`` records how the content was obtained: 'rss', 'direct', 'apify', or an
    error string. RSS uses feedparser; HTML hubs are fetched (direct → Apify
    fallback) and link-extracted.
    """
    if src.kind == "html":
        html, via = get_html(src.url)
        if not html:
            return None, 0, via            # 'blocked' / 'robots' / 'apify:budget' …
        entries = extract.extract_links(html, src.url, src.link_pattern or None,
                                        max_items=MAX_ITEMS_PER_SOURCE)
        return entries, len(entries), via
    parsed = parse_feed(src.url)
    if parsed is None:
        return None, 0, "feed unreachable"
    return list(parsed.entries), len(parsed.entries), "rss"


def run() -> dict:
    seen = store.load_seen()
    fetch_budget = [CLASSIFY_MAX_FETCH]
    all_items: list[dict] = []
    status: list[dict] = []

    for src in enabled_sources():
        entries, total, via = _entries_for(src)
        if entries is None:
            status.append({"source": src.name, "ok": False, "new": 0,
                           "error": via, "via": via})
            continue

        new_count = 0
        for entry in entries[:MAX_ITEMS_PER_SOURCE]:
            link = entry.get("link", "")
            title = _clean_text(entry.get("title", ""), 240)
            if not link or not title:
                continue
            iid = store.item_id(link, title)
            if iid in seen:
                continue
            seen.add(iid)
            new_count += 1

            if src.category == "insight":
                kind = classifier.classify(entry, link, CLASSIFY_FETCH, fetch_budget)
            else:
                kind = "n/a"   # research/news not split

            all_items.append({
                "id": iid,
                "source": src.name,
                "category": src.category,
                "visual": kind,                 # 'visual' | 'text' | 'n/a'
                "title": title,
                "link": link,
                "summary": _clean_text(entry.get("summary", "")),
                "published": _published(entry),
            })
        status.append({"source": src.name, "ok": True, "new": new_count,
                       "total_feed": total, "via": via})
        time.sleep(0.2)   # be polite between sources

    day = store.today_str()
    # merge with any items already captured earlier today (re-runs)
    existing = {i["id"]: i for i in store.load_archive(day)}
    for it in all_items:
        existing[it["id"]] = it
    merged = list(existing.values())

    store.write_archive(day, merged)
    store.write_latest(all_items)        # latest = just this run's new items
    store.save_seen(seen)
    apify_use = apify.usage_summary()
    store.save_status(status)
    store.save_apify_usage(apify_use)

    summary = {"date": day, "new_items": len(all_items),
               "sources_ok": sum(s["ok"] for s in status),
               "sources_total": len(status),
               "apify_calls": apify_use["calls_this_run"],
               "apify_spend_usd": apify_use["monthly_spend_usd"]}
    print(f"[zenith] {summary}")
    for s in status:
        flag = "ok " if s["ok"] else "ERR"
        via = s.get("via", "")
        print(f"  {flag} {s['source']}: +{s.get('new', 0)} [{via}]"
              + ("" if s["ok"] else f"  ({s.get('error')})"))
    fc_use = firecrawl.usage_summary()
    if fc_use["enabled"]:
        print(f"  firecrawl: {fc_use['calls_this_run']} call(s) this run")
    if apify_use["enabled"]:
        print(f"  apify: {apify_use['calls_this_run']} call(s) this run; "
              f"month spend ~${apify_use['monthly_spend_usd']:.3f} / "
              f"${apify_use['budget_usd']:.2f} budget")
    return summary


if __name__ == "__main__":
    run()

"""OPTIONS VAULT Phase 05 helpers — evidence for the strategy rolodex.

The user chose (2026-10-24) to build the rolodex from tastylive's written Trader Resources
(Learn Center "Concepts & Strategies" pages, glossary, research articles) cross-checked against
Phase 03 literature and Phase 04 reproductions, instead of extracting every transcript. Work is
done in-session (no API spend); this module holds the deterministic parts:

  popularity   how often each taxonomy strategy/topic appears in the titles, descriptions and
               tags of the whole episode archive (2012-), by period. Metadata only, so it covers
               the ~5,500 episodes, not just the ~700 with transcripts.
  page notes   paraphrased notes per Learn Center page (written by reader subagents), checked:
               slugs in the taxonomy, stat numbers present in the page text, no 12-word runs
               copied from the page, quote under 15 words. Cited as ``page:<slug>``.

Run:
  .venv/Scripts/python -m zenith.vault.rolodex popularity
  .venv/Scripts/python -m zenith.vault.rolodex notes <notes_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..config import VAULT_CACHE_DIR, VAULT_DIR
from .extract import (_jread, _jwrite, _log, _now, _tokens, load_taxonomy, number_matches, numbers_in,
                      VERBATIM_WORDS)

MANIFEST_PATH = VAULT_DIR / "manifest.json"
DISCOVERY_PATH = VAULT_CACHE_DIR / "discovery" / "tastylive.json"
PAGES_DIR = VAULT_CACHE_DIR / "pages"
POPULARITY_PATH = VAULT_DIR / "popularity.json"
PAGE_NOTES_PATH = VAULT_DIR / "page_notes.json"

PERIODS = [("2012-2016", 2012, 2016), ("2017-2021", 2017, 2021), ("2022-2026", 2022, 2026)]
# Short aliases that are ambiguous in free text; matched only when listed here.
SHORT_OK = {"0dte", "bwb", "pmcc", "csp", "ivr", "pop", "vix", "pot", "bpr", "pead", "roc", "1x2",
            "xsp", "/vx", "/es"}


def _alias_pattern(item: dict) -> re.Pattern | None:
    words = {item["name"].lower(), item["slug"].replace("-", " ")}
    words |= {a.lower() for a in item.get("aliases", [])}
    words = {w for w in words if len(w) >= 4 or w in SHORT_OK}
    if not words:
        return None
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?:e?s)?(?![a-z0-9])")


def episode_texts() -> list[dict]:
    """Title + description + tags per manifest episode (description from the tastylive sitemap)."""
    man = _jread(MANIFEST_PATH, {"episodes": []})["episodes"]
    desc = {d["url"]: d.get("description") or "" for d in _jread(DISCOVERY_PATH, [])}
    out = []
    for e in man:
        tl = e.get("tl") or {}
        text = " ".join([e.get("title") or "", e.get("yt_title") or "", desc.get(e.get("url"), ""),
                         " ".join(tl.get("tags") or []), (tl.get("slug") or "").replace("-", " ")])
        out.append({"episode_id": e["episode_id"], "show": e["show"],
                    "year": int(e["air_date"][:4]), "text": text.lower()})
    return out


def popularity(tax: dict, eps: list[dict]) -> dict:
    """Episode counts per strategy/topic slug, overall and per period."""
    res = {"n_episodes": len(eps),
           "note": "Mentions in episode title/description/tags, not trade counts. Side is not "
                   "distinguished ('strangle' counts long and short; 'butterfly' also matches iron "
                   "butterflies). Generic show descriptions undercount some shows.",
           "periods": {p: sum(lo <= e["year"] <= hi for e in eps) for p, lo, hi in PERIODS}}
    for kind in ("strategies", "topics"):
        rows = []
        for item in tax[kind]:
            pat = _alias_pattern(item)
            hits = [e for e in eps if pat and pat.search(e["text"])] if pat else []
            by = {p: sum(lo <= e["year"] <= hi for e in hits) for p, lo, hi in PERIODS}
            rows.append({"slug": item["slug"], "name": item["name"], "episodes": len(hits),
                         "by_period": by,
                         "share_by_period": {p: round(by[p] / res["periods"][p], 4)
                                             if res["periods"][p] else 0.0 for p in by},
                         "by_show": dict(Counter(e["show"] for e in hits).most_common())})
        res[kind] = sorted(rows, key=lambda r: -r["episodes"])
    return res


# =============================================================================
# PAGE NOTES
# =============================================================================

def page_text(slug: str) -> str:
    d = _jread(PAGES_DIR / f"{slug}.json", {})
    return d.get("text", "")


def _note_texts(note: dict):
    yield note.get("summary", "")
    for sn in note.get("strategy_notes", []):
        for v in sn.values():
            if isinstance(v, str):
                yield v
            elif isinstance(v, dict):
                yield from (x for x in v.values() if isinstance(x, str))
            elif isinstance(v, list):
                yield from (x for x in v if isinstance(x, str))
    for k in ("rules", "stats"):
        for it in note.get(k, []):
            yield it.get("text", "") or it.get("label", "")
            yield it.get("conditions", "")
    for m in note.get("mechanics", []):
        yield m.get("point", "")
    for t in note.get("terms", []):
        yield t.get("definition", "")


def check_note(note: dict, text: str, tax: dict) -> dict:
    """Slugs, stat numbers against the page text, verbatim runs, quote length."""
    known = {"strategies": {s["slug"] for s in tax["strategies"]},
             "topics": {t["slug"] for t in tax["topics"]}}
    names = {a.lower(): (kind, it["slug"]) for kind in known for it in tax[kind]
             for a in [it["name"], it["slug"].replace("-", " ")] + it.get("aliases", [])}
    moved = []

    def fix(holder: dict):
        for kind, ok in known.items():
            if kind in holder:
                bad = [x for x in holder[kind] if x not in ok]
                holder[kind] = [x for x in holder[kind] if x in ok]
                moved.extend(bad)

    fix(note)
    for k in ("rules", "stats"):
        for it in note.get(k, []):
            fix(it)
    for sn in note.get("strategy_notes", []):
        if sn.get("strategy") not in known["strategies"]:
            moved.append(sn.get("strategy"))
    # unmapped names that match a taxonomy name or alias are mapped back
    still = []
    for u in note.get("unmapped", []) + [m for m in moved if m]:
        hit = names.get(str(u).lower())
        if hit and hit[1] not in note.setdefault(hit[0], []):
            note[hit[0]].append(hit[1])
        elif not hit and u not in still:
            still.append(u)
    note["unmapped"] = still
    page_nums = numbers_in(text)
    stats_checked = stats_ok = 0
    failures = []
    for s in note.get("stats", []) + [n for r in note.get("rules", []) for n in r.get("numbers", [])]:
        v = s.get("number", s.get("value"))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            stats_checked += 1
            if number_matches(float(v), page_nums):
                stats_ok += 1
            else:
                failures.append(s.get("label") or s.get("name"))
    words = _tokens(text)
    grams = {tuple(words[i:i + VERBATIM_WORDS]) for i in range(len(words) - VERBATIM_WORDS + 1)}
    verbatim = [s for s in _note_texts(note) if s and any(
        tuple(w[i:i + VERBATIM_WORDS]) in grams
        for w in [_tokens(s)] for i in range(len(w) - VERBATIM_WORDS + 1))]
    quote = note.get("quote") or ""
    return {"numbers_checked": stats_checked, "numbers_ok": stats_ok, "number_failures": failures,
            "verbatim": verbatim, "quote_ok": len(quote.split()) < 15,
            "quote_in_page": (not quote) or " ".join(_tokens(quote)) in " ".join(words)}


def ingest_notes(src: Path, tax: dict) -> dict:
    notes, report = {}, {}
    for f in sorted(src.glob("*.json")):
        note = _jread(f, {})
        slug = note.get("page") or f.stem
        text = page_text(slug)
        if not text:
            _log(f"  skip {f.name}: no cached page {slug}")
            continue
        note["page"] = slug
        note["source"] = f"page:{slug}"
        note["url"] = _jread(PAGES_DIR / f"{slug}.json", {}).get("url", "")
        note["checks"] = check_note(note, text, tax)
        notes[slug] = note
        c = note["checks"]
        report[slug] = f"{c['numbers_ok']}/{c['numbers_checked']} numbers, " \
                       f"{len(c['verbatim'])} verbatim, quote_ok={c['quote_ok'] and c['quote_in_page']}"
    return {"notes": notes, "report": report}


# =============================================================================
# CARD PLAN + EVIDENCE BUNDLES
# =============================================================================

LITERATURE_PATH = VAULT_DIR / "literature.json"
REPRO_DIR = VAULT_DIR / "repro"
PILOT_INLINE_DIR = VAULT_CACHE_DIR / "pilot" / "inline"
STRATEGIES_PATH = VAULT_DIR / "strategies.json"

# slug -> (family, variant slugs folded into the card, literature topics, repro names, Cboe indices)
CARDS = {
    "short-strangle": ("neutral short premium", [], ["vrp", "option-selling", "management-rules"],
                       ["strangle_mgmt", "vrp_spx", "expected_move", "cboe_short_vol"], []),
    "short-straddle": ("neutral short premium", [], ["vrp", "option-selling", "management-rules"],
                       ["vrp_spx", "expected_move"], []),
    "iron-condor": ("neutral short premium", [], ["option-selling", "management-rules", "trading-costs"],
                    ["cboe_short_vol", "expected_move"], ["CNDR"]),
    "iron-fly": ("neutral short premium", [], ["option-selling", "management-rules"], [], []),
    "zero-dte": ("0DTE", [], ["zero-dte", "trading-costs"], [], []),
    "earnings-trade": ("earnings", [], ["earnings"], ["earnings_drift"], []),
    "short-put": ("directional short premium", [], ["option-selling", "skew", "vrp"],
                  ["cboe_short_vol"], ["PUT", "WPUT"]),
    "cash-secured-put": ("directional short premium", ["wheel"], ["option-selling", "mechanics"],
                         ["cboe_short_vol"], ["PUT"]),
    "put-credit-spread": ("directional short premium", [], ["option-selling", "skew"], [], []),
    "call-credit-spread": ("directional short premium", [], ["option-selling", "skew"], [], []),
    "short-call": ("directional short premium", [], ["option-selling", "mechanics"], [], []),
    "jade-lizard": ("directional short premium", ["big-lizard", "reverse-jade-lizard"],
                    ["option-selling", "skew"], [], []),
    "ratio-spread": ("ratio and butterfly", [], ["skew", "option-selling"], [], []),
    "butterfly": ("ratio and butterfly", [], ["option-selling"], ["cboe_short_vol"], ["BFLY"]),
    "broken-wing-butterfly": ("ratio and butterfly", [], ["option-selling", "skew"], [], []),
    "covered-call": ("stock-based", ["covered-strangle"], ["option-selling", "mechanics"],
                     ["cboe_short_vol"], ["BXM", "BXMD"]),
    "collar": ("stock-based", [], ["tail-hedging"], ["cboe_tail_hedge"], ["CLL"]),
    "poor-mans-covered-call": ("stock-based", [], ["option-selling"], [], []),
    "leaps-stock-replacement": ("stock-based", [], ["option-anomalies"], [], []),
    "calendar": ("time spreads", [], ["iv-predictors"], [], []),
    "diagonal": ("time spreads", [], ["iv-predictors"], [], []),
    "debit-spread": ("directional long premium", [], ["option-anomalies"], [], []),
    "long-call": ("long volatility and directional", [], ["option-anomalies", "vrp"], [], []),
    "long-put": ("long volatility and directional", [], ["option-anomalies", "tail-hedging"], [], []),
    "long-straddle": ("long volatility and directional", [], ["vrp", "earnings"], ["vrp_spx"], []),
    "long-strangle": ("long volatility and directional", [], ["vrp", "earnings"], ["vrp_spx"], []),
    "protective-put": ("hedges", [], ["tail-hedging"], ["cboe_tail_hedge"], ["PPUT"]),
    "tail-hedge": ("hedges", [], ["tail-hedging"], ["cboe_tail_hedge", "vix_spikes"], []),
    "vix-hedge": ("hedges", [], ["tail-hedging"], ["vix_spikes"], []),
    "futures-options": ("futures options", [], ["mechanics", "trading-costs"], [], []),
}
THIN = {"pairs-trade": "no Learn Center page; 48 archive mentions, no study in the pilot",
        "super-bull": "no Learn Center page; 14 archive mentions",
        "short-risk-reversal": "no Learn Center page; 11 archive mentions",
        "strangle-swap": "single host experiment; no page"}
SHARED_PAGES = ["concepts-managing-winners", "concepts-defending-positions",
                "concepts-number-of-occurrences", "concepts-options-liquidity", "concepts-assignment",
                "concepts-options-expiration", "concepts-implied-volatility-rank-percentile",
                "concepts-risk-management", "concepts-naked-options", "concepts-standard-deviation",
                "concepts-order-types", "concepts-pattern-day-trader-pdt-rule"]


def _note_items(note: dict, slugs: set[str]) -> dict:
    """The parts of one page note that concern these strategy slugs."""
    tagged = lambda it: bool(slugs & set(it.get("strategies", [])))
    return {"source": note["source"], "summary": note.get("summary", ""),
            "strategy_notes": [s for s in note.get("strategy_notes", []) if s.get("strategy") in slugs],
            "rules": [r for r in note.get("rules", []) if tagged(r)],
            "stats": [s for s in note.get("stats", []) if tagged(s)],
            "quote": note.get("quote", "")}


def _episode_items(slugs: set[str]) -> list[dict]:
    out = []
    for f in sorted(PILOT_INLINE_DIR.glob("*.json")):
        r = _jread(f, {})
        eid = r.get("episode_id")
        studies = [{**s, "source": f"ep:{eid}@{s['ts']}"} for s in r.get("studies", [])
                   if slugs & set(s.get("strategies", []))]
        rules = [{**x, "source": f"ep:{eid}@{x['ts']}"} for x in r.get("rules", [])
                 if slugs & set(x.get("strategies", []))]
        if studies or rules:
            out.append({"episode": eid, "title": r.get("title"), "air_date": r.get("air_date"),
                        "studies": studies, "rules": rules})
    return out


def _lit_topic(t: dict) -> dict:
    return {"topic": t["id"], "tier": t.get("tier"), "tier_note": t.get("tier_note"),
            "agreement": t.get("agreement_with_tasty"), "consensus": t.get("consensus"),
            "agreement_detail": t.get("agreement_detail", []),
            "implications": t.get("implications", []),
            "papers": [{"ref": f"lit:{p['id']}", "title": p.get("title", ""),
                        "finding": str(p.get("finding", ""))[:400]} for p in t.get("papers", [])]}


def _repro(name: str, indices: list[str]) -> dict:
    d = _jread(REPRO_DIR / f"{name}.json", {})
    out = {"ref": f"repro:{name}", "period": d.get("period"), "caveats": d.get("caveats", []),
           "claims": [{k: c.get(k) for k in ("text", "verdict", "our_value")} for c in d.get("claims", [])]}
    if indices and d.get("table"):
        out["index_rows"] = [r for r in d["table"] if r.get("index") in set(indices) | {"SPTR", "TBILL"}]
    return out


def bundles(tax: dict) -> dict[str, dict]:
    notes = _jread(PAGE_NOTES_PATH, {"notes": {}})["notes"]
    lit = {t["id"]: t for t in _jread(LITERATURE_PATH, {"topics": []})["topics"]}
    pop = {r["slug"]: r for r in _jread(POPULARITY_PATH, {"strategies": []})["strategies"]}
    meta = {s["slug"]: s for s in tax["strategies"]}
    out = {}
    for slug, (family, variants, topics, repros, indices) in CARDS.items():
        slugs = {slug, *variants}
        pages = [_note_items(n, slugs) for n in notes.values() if slugs & set(n.get("strategies", []))]
        out[slug] = {"slug": slug, "name": meta[slug]["name"], "family": family, "variants": variants,
                     "taxonomy": {s: meta[s] for s in slugs},
                     "pages": [p for p in pages if p["strategy_notes"] or p["rules"] or p["stats"]],
                     "episodes": _episode_items(slugs),
                     "literature": [_lit_topic(lit[t]) for t in topics if t in lit],
                     "repro": [_repro(r, indices) for r in repros],
                     "popularity": {s: {k: pop[s][k] for k in ("episodes", "by_period")}
                                    for s in slugs if s in pop}}
    return out


def shared_bundle() -> dict:
    notes = _jread(PAGE_NOTES_PATH, {"notes": {}})["notes"]
    lit = {t["id"]: t for t in _jread(LITERATURE_PATH, {"topics": []})["topics"]}
    return {"pages": {s: notes[s] for s in SHARED_PAGES if s in notes},
            "literature": [_lit_topic(lit[t]) for t in ("mechanics", "sizing-risk", "management-rules")
                           if t in lit],
            "allocation": "25-50% of buying power in options, max 75%; higher in high vol and small "
                          "accounts [ep:options-jive-2023-12-27@06:56]"}


# =============================================================================
# INTEGRITY CHECK
# =============================================================================

MECHANICS_PATH = VAULT_DIR / "mechanics.json"
CARD_KEYS = ["slug", "name", "family", "aliases", "one_liner", "short_what", "greeks", "risk", "accounts",
             "legs", "when", "avoid", "setup", "manage", "size", "stats", "evidence", "history",
             "mistakes", "example", "related", "sources"]
REF_RE = re.compile(r"\b(ep|lit|repro|page):([A-Za-z0-9._-]*[A-Za-z0-9])(?:@(\d+:\d{2}))?(?=[\]\s,;.\"')]|$)")


def resolver() -> dict:
    man = {e["episode_id"]: e for e in _jread(MANIFEST_PATH, {"episodes": []})["episodes"]}
    lit = {p["id"] for t in _jread(LITERATURE_PATH, {"topics": []})["topics"] for p in t.get("papers", [])}
    return {"man": man, "lit": lit}


def ref_ok(kind: str, key: str, ts: str | None, res: dict) -> bool:
    if kind == "ep":
        e = res["man"].get(key)
        if not e:
            return False
        if ts:
            m, s = ts.split(":")
            dur = e.get("duration_s") or 0
            return not dur or int(m) * 60 + int(s) <= dur
        return True
    if kind == "lit":
        return key in res["lit"]
    if kind == "repro":
        return (REPRO_DIR / f"{key}.json").exists()
    return (PAGES_DIR / f"{key}.json").exists()


def refs_in(obj) -> list[tuple]:
    return REF_RE.findall(json.dumps(obj, ensure_ascii=False))


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or v == ""


def check_all(cards: list[dict], trees: list[dict], tax: dict, res: dict | None = None) -> dict:
    res = res or resolver()
    slugs = {s["slug"] for s in tax["strategies"]}
    problems: list[str] = []
    n_refs = 0
    for c in cards:
        sid = c.get("slug", "?")
        for k in CARD_KEYS:
            if k not in c:
                problems.append(f"{sid}: missing {k}")
        if c.get("slug") not in slugs:
            problems.append(f"{sid}: slug not in taxonomy")
        if (c.get("evidence") or {}).get("tier") not in ("A", "B", "C"):
            problems.append(f"{sid}: tier {(c.get('evidence') or {}).get('tier')!r}")
        for k in ("one_liner", "risk", "family"):
            if _empty(c.get(k)):
                problems.append(f"{sid}: empty {k}")
        bad_rel = [r for r in c.get("related", []) if r not in slugs]
        if bad_rel:
            problems.append(f"{sid}: related not in taxonomy {bad_rel}")
        for st in c.get("stats", []):
            if not st.get("sources"):
                problems.append(f"{sid}: stat without sources: {st.get('label')}")
        if "nan" in json.dumps(c).lower().replace("financ", ""):
            problems.append(f"{sid}: NaN in card")
        for kind, key, ts in refs_in(c):
            n_refs += 1
            if not ref_ok(kind, key, ts or None, res):
                problems.append(f"{sid}: unresolved {kind}:{key}{'@' + ts if ts else ''}")
    for t in trees:
        ids = {n["id"] for n in t.get("nodes", [])}
        for e in t.get("edges", []):
            if e["from"] not in ids or e["to"] not in ids:
                problems.append(f"tree {t['id']}: edge {e['from']}->{e['to']} to unknown node")
        for kind, key, ts in refs_in(t):
            n_refs += 1
            if not ref_ok(kind, key, ts or None, res):
                problems.append(f"tree {t['id']}: unresolved {kind}:{key}{'@' + ts if ts else ''}")
    return {"cards": len(cards), "trees": len(trees), "refs": n_refs, "problems": problems}


EDGE_KEYS = ["slug", "title", "finding", "numbers", "conditions", "act_with", "tasty_sources", "literature",
             "repro", "agreement", "tier", "kind", "caveats"]


def check_extras(edges: list[dict], portfolio: dict, glossary: list[dict], tax: dict,
                 res: dict | None = None) -> dict:
    """Integrity of edges.json, portfolio.json and glossary.json: refs, slugs, required keys."""
    res = res or resolver()
    slugs = {s["slug"] for s in tax["strategies"]} | {t["slug"] for t in tax["topics"]}
    problems, n_refs = [], 0

    def refs(obj, where):
        nonlocal n_refs
        for kind, key, ts in refs_in(obj):
            n_refs += 1
            if not ref_ok(kind, key, ts or None, res):
                problems.append(f"{where}: unresolved {kind}:{key}{'@' + ts if ts else ''}")

    for e in edges:
        sid = f"edge {e.get('slug', '?')}"
        problems += [f"{sid}: missing {k}" for k in EDGE_KEYS if k not in e]
        if e.get("tier") not in ("A", "B", "C"):
            problems.append(f"{sid}: tier {e.get('tier')!r}")
        if e.get("kind") not in ("EDGE", "CONVENTION"):
            problems.append(f"{sid}: kind {e.get('kind')!r}")
        if e.get("agreement") not in ("agrees", "mixed", "contradicts", "untested"):
            problems.append(f"{sid}: agreement {e.get('agreement')!r}")
        bad = [s for s in e.get("act_with", []) if s not in slugs]
        if bad:
            problems.append(f"{sid}: act_with not in taxonomy {bad}")
        for n in e.get("numbers", []):
            if not n.get("sources"):
                problems.append(f"{sid}: number without sources: {n.get('label')}")
        refs(e, sid)
    for k in ("account_matrix", "allocation", "sleeve_playbook", "account_tiers", "taxes", "unverified"):
        if k not in portfolio:
            problems.append(f"portfolio: missing {k}")
    refs(portfolio, "portfolio")
    for t in glossary:
        if not t.get("definition"):
            problems.append(f"glossary {t.get('term')}: empty definition")
        bad = [s for s in t.get("related", []) if s not in slugs]
        if bad:
            problems.append(f"glossary {t.get('term')}: related not in taxonomy {bad}")
        refs(t, f"glossary {t.get('term')}")
    return {"edges": len(edges), "glossary": len(glossary), "refs": n_refs, "problems": problems}


def accounts_for(legs: list[dict]) -> tuple[dict, str]:
    """Account permissions derived from the legs (tastytrade IRA rules [lit:tastytrade-ira-2026]):
    naked short puts need a margin account; uncovered calls are allowed in a tastytrade IRA only at
    "The Works" with $25,000 start-of-day net liq; defined-risk spreads are IRA-eligible (cash-account
    spreads not verified); long options and covered/secured positions are cash- and IRA-eligible.
    Legs typed "call or put" (structures that can be built either way) count as one generic kind."""
    def kind(l):
        t = l.get("type")
        return t if t in ("call", "put", "stock") else "option"

    def qty(side, k):
        return sum(abs(l.get("qty") or 1) for l in legs if l.get("side") == side and kind(l) == k)

    short_c, short_p, short_o = qty("short", "call"), qty("short", "put"), qty("short", "option")
    long_c, long_p, long_o, stock = qty("long", "call"), qty("long", "put"), qty("long", "option"), qty("long", "stock")
    if not (short_c or short_p or short_o):
        return {"cash": True, "ira": True, "reg_t": True, "pm": True}, \
            "Long options and stock only: cash and IRA eligible [lit:tastytrade-ira-2026]."
    naked_c = max(0, short_c - long_c - stock)
    naked_p = max(0, short_p - long_p)
    naked_o = max(0, short_o - long_o)
    if stock and not (naked_c or naked_p or naked_o) and not short_p and not short_o:
        return {"cash": True, "ira": True, "reg_t": True, "pm": True}, \
            "Covered by stock or long options: cash and IRA eligible [lit:tastytrade-ira-2026]."
    if not (naked_c or naked_p or naked_o):
        return {"cash": False, "ira": True, "reg_t": True, "pm": True}, \
            "Defined-risk spread: IRA eligible [lit:tastytrade-ira-2026]; cash-account spreads not verified."
    if naked_c and not (naked_p or naked_o):
        return {"cash": False, "ira": True, "reg_t": True, "pm": True}, \
            ("Uncovered calls: margin account, or a tastytrade IRA at 'The Works' with $25,000 start-of-day "
             "net liq [lit:tastytrade-ira-2026].")
    return {"cash": False, "ira": False, "reg_t": True, "pm": True}, \
        "Naked short puts need a margin account; not allowed in IRA or cash accounts [lit:tastytrade-ira-2026]."


# Cards whose permissions do not follow from the legs alone (secured by cash, or a product wrapper).
ACCOUNT_OVERRIDES = {"cash-secured-put": ({"cash": True, "ira": True, "reg_t": True, "pm": True},
                                         "Fully cash-secured: cash and IRA eligible [lit:tastytrade-ira-2026]."),
                     "futures-options": None}


def assemble(card_dir: Path) -> dict:
    """Merge draft card files into strategies.json (keeping cards already there unless replaced),
    refreshing each card's popularity from popularity.json."""
    pop = {r["slug"]: r for r in _jread(POPULARITY_PATH, {"strategies": []})["strategies"]}
    cur = {c["slug"]: c for c in _jread(STRATEGIES_PATH, {"cards": []})["cards"]}
    for f in sorted(card_dir.glob("*.json")):
        c = _jread(f, {})
        if c.get("slug"):
            cur[c["slug"]] = c
    order = list(CARDS)
    cards = sorted(cur.values(), key=lambda c: order.index(c["slug"]) if c["slug"] in order else 999)
    for c in cards:
        if c["slug"] in ACCOUNT_OVERRIDES:
            if ACCOUNT_OVERRIDES[c["slug"]]:
                c["accounts"], c["accounts_note"] = ACCOUNT_OVERRIDES[c["slug"]]
        elif c.get("legs"):
            c["accounts"], c["accounts_note"] = accounts_for(c["legs"])
        p = pop.get(c["slug"])
        if p:
            c["popularity"] = {"episodes": p["episodes"], "by_period": p["by_period"],
                               "note": "Mentions in titles/descriptions/tags; long and short not distinguished."}
    return {"status": "draft", "generated": _now()[:10],
            "note": "Phase 05 strategy cards (Trader Resources + literature + reproductions).",
            "thin_cuts": THIN, "cards": cards}


RANKING_PATH = VAULT_DIR / "ranking.json"
POP_PATH = VAULT_DIR / "pop.json"
TIER_ORDER = {"A": 0, "B": 1, "C": 2}
EARNING_FAMILIES = {"neutral short premium", "directional short premium", "ratio and butterfly", "stock-based"}
HEDGE_SLUGS = {"collar"}   # stock-based, but its evidence is the cost of protection [repro:cboe_tail_hedge]


def ranking(cards: list[dict], pop_rows: list[dict]) -> dict:
    """Popularity, POP and reliability tables. Reliability = evidence tier first, then the model's
    worst-5% loss (smaller is better), then archive popularity as a tie-break."""
    pop_by = {r["slug"]: r for r in pop_rows}
    rows = []
    for c in cards:
        p = pop_by.get(c["slug"], {})
        rows.append({"slug": c["slug"], "name": c["name"], "family": c["family"], "risk": c["risk"],
                     "tier": c["evidence"]["tier"], "kind": c["evidence"]["kind"],
                     "thin_evidence": c.get("thin_evidence", ""),
                     "episodes": (c.get("popularity") or {}).get("episodes", 0),
                     "pop_theoretical": p.get("pop_theoretical"), "pop_realized": p.get("pop_realized"),
                     "mean_units_realized": p.get("mean_units_realized"),
                     "cvar5_units_realized": p.get("cvar5_units_realized"), "unit": p.get("unit"),
                     "model_setup": p.get("setup"), "evidence_note": c["evidence"].get("note", "")})

    def reliability_key(r):
        # tail figures are in different units across rows (credit, debit, position cost), so they
        # are shown but not used for ordering
        return (TIER_ORDER.get(r["tier"], 9), bool(r["thin_evidence"]), -r["episodes"])

    # Reliability is about a positive expected outcome, so it ranks only the families whose evidence
    # claims one (premium selling, stock-based income). Hedges and long premium are expected to cost
    # money on average [repro:cboe_tail_hedge] [lit:coval-shumway-2001]; they get their own list.
    earners = [r for r in rows if r["family"] in EARNING_FAMILIES and r["slug"] not in HEDGE_SLUGS]
    return {
        "note": "POP and tail figures come from the flat-vol model in zenith/vault/pop.py (held to "
                "expiration, no skew, no costs) [page:concepts-vix] [repro:vrp_spx]; tiers from the "
                "strategy cards; popularity from archive titles/tags. Reliability ranks evidence first "
                "and covers only strategies with a claimed positive edge.",
        "most_popular": sorted(rows, key=lambda r: -r["episodes"]),
        "highest_pop": sorted([r for r in rows if r["pop_theoretical"] is not None],
                              key=lambda r: -r["pop_theoretical"]),
        "most_reliable_defined": sorted([r for r in earners if r["risk"] == "defined"], key=reliability_key),
        "most_reliable_undefined": sorted([r for r in earners if r["risk"] == "undefined"], key=reliability_key),
        "insurance_and_long_premium": sorted([r for r in rows if r not in earners
                                              and r["family"] not in ("0DTE", "earnings", "futures options")],
                                             key=reliability_key),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="zenith.vault.rolodex")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("popularity")
    p = sub.add_parser("notes")
    p.add_argument("src", type=Path)
    p = sub.add_parser("bundle")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("assemble")
    p.add_argument("card_dir", type=Path)
    p = sub.add_parser("check")
    p.add_argument("--cards", type=Path, help="directory of draft card files instead of strategies.json")
    args = ap.parse_args(argv)
    tax = load_taxonomy()
    if args.cmd == "popularity":
        pop = popularity(tax, episode_texts())
        _jwrite(POPULARITY_PATH, pop)
        for r in pop["strategies"][:30]:
            print(f"{r['episodes']:5d}  {r['slug']:26s} {r['by_period']}")
    elif args.cmd == "notes":
        res = ingest_notes(args.src, tax)
        _jwrite(PAGE_NOTES_PATH, {"taxonomy_version": tax.get("version"), "notes": res["notes"]})
        for k, v in res["report"].items():
            print(f"{k:60s} {v}")
    elif args.cmd == "bundle":
        for slug, b in bundles(tax).items():
            _jwrite(args.out / f"{slug}.json", b)
            print(f"{slug:26s} pages {len(b['pages']):2d}  episodes {len(b['episodes']):2d}  "
                  f"lit {len(b['literature'])}  repro {len(b['repro'])}")
        _jwrite(args.out / "_shared.json", shared_bundle())
    elif args.cmd == "assemble":
        out = assemble(args.card_dir)
        _jwrite(STRATEGIES_PATH, out)
        print(f"{len(out['cards'])} cards -> {STRATEGIES_PATH}")
        rk = ranking(out["cards"], _jread(POP_PATH, {"rows": []})["rows"])
        _jwrite(RANKING_PATH, rk)
        print(f"ranking -> {RANKING_PATH}")
    elif args.cmd == "check":
        if args.cards:
            cards = [_jread(f, {}) for f in sorted(args.cards.glob("*.json"))]
        else:
            cards = _jread(STRATEGIES_PATH, {"cards": []})["cards"]
        rep = check_all(cards, _jread(MECHANICS_PATH, {"trees": []})["trees"], tax)
        print(f"{rep['cards']} cards, {rep['trees']} trees, {rep['refs']} refs, "
              f"{len(rep['problems'])} problems")
        for p in rep["problems"]:
            print("  -", p)
        base = args.cards.parent if args.cards else VAULT_DIR
        ex = check_extras(_jread(base / "edges.json", {"edges": []})["edges"],
                          _jread(base / "portfolio.json", {}),
                          _jread(base / "glossary.json", {"terms": []})["terms"], tax)
        print(f"{ex['edges']} edges, glossary {ex['glossary']} terms, {ex['refs']} refs, "
              f"{len(ex['problems'])} problems")
        for p in ex["problems"]:
            print("  -", p)


if __name__ == "__main__":
    sys.exit(main())

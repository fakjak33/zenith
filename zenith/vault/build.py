"""OPTIONS VAULT — validate the phase outputs and pack what the view reads.

The view must not load the 5.8 MB manifest or anything in data/vault/cache/ (gitignored), so this
step resolves every source reference used by the cards, edges, trees, portfolio, glossary, POP
table and reproductions into a small link table, and writes it with the counts to
data/vault/build.json:

  ep:<id>@mm:ss  -> YouTube at the timestamp when the episode has a video, else the tastylive page
  page:<slug>    -> the tastylive Learn Center URL (pages.json)
  lit:<id>       -> the paper's URL / DOI (literature.json)
  repro:<name>   -> the reproduction's claim (shown in EDGES & RESEARCH)

Checks: every reference resolves; the card / edge / tree schemas (rolodex.CARD_KEYS, EDGE_KEYS);
and, when the local page cache exists, the full rolodex integrity check as well.

Run: .venv/Scripts/python -m zenith.vault.build
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date

from . import DISCLAIMER, load, save
from ..config import VAULT_FILES
from .rolodex import CARD_KEYS, EDGE_KEYS, PAGES_DIR, refs_in

SHOW_NAMES = {"market-measures": "Market Measures", "from-theory-to-practice": "From Theory to Practice",
              "options-jive": "Options Jive", "skinny-math": "Skinny on Options Math",
              "skinny-abstract": "Skinny on Options Abstract", "math-check": "The Math Check",
              "strange-stats": "Strange Stats", "research-team-live": "Research Team Live",
              "research": "tastylive research"}


def _repro_all() -> dict[str, dict]:
    d = VAULT_FILES["repro_dir"]
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def _secs(ts: str | None) -> int | None:
    if not ts:
        return None
    m, s = ts.split(":")
    return int(m) * 60 + int(s)


def pack(art: dict | None = None) -> dict:
    """art: optional {name: obj} override (tests); otherwise loads VAULT_FILES."""
    a = art or {}

    def get(name, default):
        return a[name] if name in a else load(name, default)

    cards = get("strategies", {"cards": []})
    edges = get("edges", {"edges": []})
    mech = get("mechanics", {"trees": []})
    port = get("portfolio", {})
    gloss = get("glossary", {"terms": []})
    pop = get("pop", {"rows": []})
    rank = get("ranking", {})
    lit = get("literature", {"topics": []})
    man = get("manifest", {"episodes": []})
    pages = get("pages", {"pages": []})
    repro = a["repro"] if "repro" in a else _repro_all()

    episodes = {e["episode_id"]: e for e in man.get("episodes", [])}
    page_urls = {p["slug"]: p.get("url", "") for p in pages.get("pages", [])}
    papers = {p["id"]: p for t in lit.get("topics", []) for p in t.get("papers", [])}
    topics = {t["id"]: t for t in lit.get("topics", [])}

    problems: list[str] = []
    for c in cards.get("cards", []):
        problems += [f"card {c.get('slug')}: missing {k}" for k in CARD_KEYS if k not in c]
        if (c.get("evidence") or {}).get("tier") not in ("A", "B", "C"):
            problems.append(f"card {c.get('slug')}: bad tier")
    for e in edges.get("edges", []):
        problems += [f"edge {e.get('slug')}: missing {k}" for k in EDGE_KEYS if k not in e]
    for t in mech.get("trees", []):
        ids = {n["id"] for n in t.get("nodes", [])}
        problems += [f"tree {t.get('id')}: edge {x['from']}->{x['to']}" for x in t.get("edges", [])
                     if x["from"] not in ids or x["to"] not in ids]

    refs: dict[str, dict] = {}
    used = [cards, edges, mech, port, gloss, pop, rank,
            [{k: r.get(k) for k in ("tasty_sources", "literature", "claims")} for r in repro.values()]]
    for kind, key, ts in refs_in(used):
        ref = f"{kind}:{key}"
        if ref in refs or f"unresolved {ref}" in problems:
            continue
        if kind == "ep":
            e = episodes.get(key)
            if not e:
                problems.append(f"unresolved {ref}")
                continue
            refs[ref] = {"label": f"{SHOW_NAMES.get(e.get('show'), e.get('show'))} "
                                  f"{e.get('air_date', '')}: {e.get('title') or e.get('yt_title') or ''}".strip(),
                         "url": e.get("url") or "", "yt": e.get("youtube_id") or ""}
        elif kind == "page":
            if key not in page_urls:
                problems.append(f"unresolved {ref}")
                continue
            refs[ref] = {"label": "tastylive: " + key.split("-", 1)[-1].replace("-", " "),
                         "url": page_urls[key]}
        elif kind == "lit":
            p = papers.get(key)
            if not p and key in topics:          # repro claim_source "lit:<topic>"
                refs[ref] = {"label": f"literature review topic: {topics[key].get('title', key)}",
                             "url": ""}
                continue
            if not p:
                problems.append(f"unresolved {ref}")
                continue
            yr = f" ({p['year']})" if p.get("year") else ""
            refs[ref] = {"label": f"{p.get('authors') or ''}{yr}: {p.get('title') or ''}",
                         "url": p.get("url") or (f"https://doi.org/{p['doi']}" if p.get("doi") else "")}
        else:
            r = repro.get(key)
            if not r:
                problems.append(f"unresolved {ref}")
                continue
            refs[ref] = {"label": f"our reproduction {key}: {r.get('claim', '')}", "url": ""}

    # timestamps beyond the episode's runtime (same rule as rolodex.ref_ok)
    for kind, key, ts in refs_in(used):
        if kind == "ep" and ts and key in episodes:
            dur = episodes[key].get("duration_s") or 0
            if dur and _secs(ts) > dur:
                problems.append(f"ep:{key}@{ts} beyond runtime")

    rolodex_problems = None
    if PAGES_DIR.exists() and art is None:
        from .rolodex import check_all, check_extras, load_taxonomy
        tax = load_taxonomy()
        r1 = check_all(cards.get("cards", []), mech.get("trees", []), tax)
        r2 = check_extras(edges.get("edges", []), port, gloss.get("terms", []), tax)
        rolodex_problems = r1["problems"] + r2["problems"]
        problems += [f"rolodex: {p}" for p in rolodex_problems]

    cl = cards.get("cards", [])
    return {
        "generated": date.today().isoformat(),
        "status": cards.get("status", "draft"),
        "disclaimer": DISCLAIMER,
        "counts": {
            "strategies": len(cl), "edges": len(edges.get("edges", [])),
            "trees": len(mech.get("trees", [])), "glossary": len(gloss.get("terms", [])),
            "papers": len(papers), "topics": len(lit.get("topics", [])), "repro": len(repro),
            "tiers": dict(sorted(Counter(c["evidence"]["tier"] for c in cl).items())),
            "edge_tiers": dict(sorted(Counter(e["tier"] for e in edges.get("edges", [])).items())),
            "thin_evidence": sum(1 for c in cl if c.get("thin_evidence")),
            "refs": len(refs),
        },
        "rolodex_check": "skipped (no local page cache)" if rolodex_problems is None
        else f"{len(rolodex_problems)} problems",
        "problems": problems,
        "refs": dict(sorted(refs.items())),
    }


def main(argv=None) -> int:
    out = pack()
    save("build", out)
    c = out["counts"]
    print(f"{c['strategies']} cards (tiers {c['tiers']}, thin {c['thin_evidence']}), {c['edges']} edges, "
          f"{c['trees']} trees, {c['glossary']} terms, {c['papers']} papers, {c['repro']} repro")
    print(f"{c['refs']} refs resolved; rolodex check: {out['rolodex_check']}; "
          f"{len(out['problems'])} problems -> {VAULT_FILES['build']}")
    for p in out["problems"]:
        print("  -", p)
    return 1 if out["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

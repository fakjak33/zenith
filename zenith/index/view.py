"""INDEX tab — the Master List directory surface.

Reads only committed JSON (data/index/*) — no network calls from the view, the
convention every Zenith package follows.

WHY THERE IS NO A/B/C EVIDENCE BADGE HERE. Every other tab opens with
``ui_theme.evidence_rating``, but that badge grades a SIGNAL's out-of-sample
predictive strength. A directory has no such property, and stamping a letter on
it would manufacture exactly the false precision this feature is supposed to
avoid. The verification strip below is the honest analogue: counts of what has
actually been checked, when, and what still needs review — measurements, not a
grade. See zenith/index/__init__.py for the full posture.

st.dataframe's canvas grid renders no accessible text, so the Directory table's
contents cannot be verified by scraping the browser; the tests exercise the data
path directly and via AppTest instead (a standing repo gotcha).
"""

from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import help_badge, key_findings, section, stamp
from . import DISCLAIMER, load
from . import export as idx_export
from . import guests as idx_guests
from . import quality as idx_quality
from . import taxonomy as tx

SUBVIEWS = ["Overview", "Directory", "Entity detail", "Podcasts", "Guests",
            "Taxonomy", "Data management"]

_FINDINGS = [
    {"stat": "Every link is checked by an actual HTTP request, and a site that blocks "
             "robots is reported as BLOCKED — not as broken.",
     "cite": "links.py"},
    {"stat": "Where a fact could not be verified the field is left empty and the entry is "
             "flagged for review, rather than filled with a plausible guess.",
     "cite": "honesty posture, §2"},
    {"stat": "When a person changes firm the old affiliation is preserved, not overwritten.",
     "cite": "model.merge()"},
]

_LINK_COLORS = {"ok": THEME.teal, "blocked": THEME.mustard, "error": THEME.coral,
                "missing": THEME.coral, "unchecked": THEME.muted}
_STATE_COLORS = {"verified": THEME.teal, "updated": THEME.navy, "new": THEME.mint,
                 "needs_review": THEME.mustard, "archived": THEME.muted}


@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(cache_bust: str = "") -> dict:
    # `episodes` is deliberately NOT loaded here: it is thousands of rows and
    # only the Podcasts view needs it, so it gets its own cached loader.
    return {"entities": load("entities", []), "relationships": load("relationships", []),
            "status": load("status", {}), "links": load("links", {}),
            "podcasts": load("podcasts", {})}


@st.cache_data(ttl=600, show_spinner=False)
def _episodes(cache_bust: str = "") -> list[dict]:
    return load("episodes", [])


def today_badge() -> str | None:
    try:
        status = load("status", {})
        total = status.get("total")
        if not total:
            return None
        needs = status.get("needs_review", 0)
        verified = status.get("verified", 0)
        color = THEME.mustard if needs > total * 0.25 else THEME.teal
        return uc.chip(f"INDEX — {total} sources indexed", color=color,
                       sub=f"{verified} verified · {needs} need review")
    except Exception:
        return None


# ------------------------------------------------------------------ helpers --
def _rel_map(entities: list[dict], relationships: list[dict]) -> dict[str, list[tuple]]:
    """entity id -> [(label, other_name, other_id)], both edge directions."""
    names = {e["id"]: e.get("name", "") for e in entities}
    inverse = {"works_at": "employs", "worked_at": "previously employed",
               "founded": "founded by", "subsidiary_of": "parent of",
               "publishes": "published by", "hosts": "hosted by",
               "appeared_on": "guest", "related_to": "related to"}
    out: dict[str, list[tuple]] = {}
    for r in relationships:
        s, t = r.get("source"), r.get("target")
        typ = str(r.get("type") or "related_to").replace("_", " ")
        if s in names and t in names:
            out.setdefault(s, []).append((typ, names[t], t))
            out.setdefault(t, []).append(
                (inverse.get(r.get("type"), r.get("type")).replace("_", " "), names[s], s))
    return out


def _tag_labels(ent: dict, field: str, vocab: str) -> list[str]:
    return [tx.label_of(vocab, s) for s in ent.get(field, [])]


def _link_chip(status: str) -> str:
    return uc.chip(str(status or "unchecked"),
                   color=_LINK_COLORS.get(status, THEME.muted))


def _filter_options(entities: list[dict], field: str, vocab: str) -> list[str]:
    used: set[str] = set()
    for e in entities:
        used.update(e.get(field, []))
    return sorted((tx.label_of(vocab, s) for s in used), key=str.lower)


# ----------------------------------------------------------------- overview --
def _overview(art: dict) -> None:
    ents, status = art["entities"], art["status"]
    total = len(ents)
    verified = sum(1 for e in ents if e.get("lifecycle_state") == "verified")
    needs = sum(1 for e in ents if e.get("lifecycle_state") == "needs_review")
    link_ok = sum(1 for e in ents if e.get("link_status") == "ok")
    oldest = status.get("oldest_verification_days")

    st.markdown(section("Verification — what has actually been checked", 0,
                        help="These are measurements, not a quality grade. 'Links live' "
                             "means an HTTP request succeeded; sites that block robots are "
                             "counted separately rather than as failures."),
                unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Entries", "value": f"{total:,}", "color": THEME.text,
         "sub": f"{status.get('relationships', 0)} relationships"},
        {"label": "Links live", "value": f"{link_ok:,}", "color": THEME.teal,
         "sub": f"{(link_ok / total * 100):.0f}% of entries responded" if total else ""},
        {"label": "Verified", "value": f"{verified:,}", "color": THEME.mint,
         "sub": "link confirmed + reviewed"},
        {"label": "Need review", "value": f"{needs:,}", "color": THEME.mustard,
         "sub": "unconfirmed or incomplete"},
        {"label": "Oldest check", "value": (f"{oldest}d" if oldest is not None else "—"),
         "color": THEME.navy, "sub": "since last verification"},
    ]), unsafe_allow_html=True)

    # The catalog has two populations with very different verification stories,
    # and reporting one number for both is misleading: the curated entries were
    # hand-written with checked URLs, while the harvested ones are names parsed
    # out of podcast metadata that nobody has verified yet. Showing "1,976 need
    # review" without that split reads as decay when it is actually just the
    # shape of a large automated import.
    def _discovered(ent: dict) -> bool:
        prov = str(ent.get("provenance", ""))
        return ("harvest" in prov or "prior manual compilation" in prov) and \
               "seed list" not in prov

    harvested = [e for e in ents if _discovered(e)]
    curated = [e for e in ents if not _discovered(e)]
    if harvested:
        cur_needs = sum(1 for e in curated if e.get("lifecycle_state") == "needs_review")
        st.markdown(uc.note_strip("Two populations, verified differently", [
            f"{len(curated):,} CURATED entries from the supplied resource list — "
            f"hand-written, link-checked, {cur_needs} still needing review.",
            f"{len(harvested):,} DISCOVERED entries from the podcast archives and the "
            "prior compilation — real names parsed from publisher metadata, but no "
            "website or biography verified, so all of them are flagged for review by "
            "design rather than by failure.",
            "Filter the Directory by Status or search by provenance to work with "
            "either population on its own.",
        ]), unsafe_allow_html=True)

    ls = status.get("links", {})
    if ls.get("checked"):
        # These count distinct URLs, whereas "Links live" above counts ENTRIES.
        # The two differ because several entries legitimately share one URL — a
        # person's link normally points at their employer's site — so the counts
        # are labelled explicitly rather than left looking contradictory.
        st.markdown(uc.note_strip(
            f"How to read the link statuses — {ls.get('checked', 0)} distinct URLs checked",
            [f"{ls.get('ok', 0)} URLs responded normally (covering {link_ok} entries — several "
             "entries share a URL, since a person's link usually points at their firm).",
             f"{ls.get('blocked', 0)} are alive but refuse automated requests (Cloudflare, "
             "403, or a robots.txt rule). These are NOT broken — they open fine in a browser.",
             f"{ls.get('error', 0)} genuinely did not resolve and are flagged for review."],
        ), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(section("By category", 2), unsafe_allow_html=True)
        cat = pd.DataFrame(
            [{"Category": tx.label_of("primary_category", k), "Entries": v}
             for k, v in (status.get("by_primary_category") or {}).items()])
        if not cat.empty:
            uc.hbar(cat, "Entries", "Category", cap=max(cat["Entries"].max(), 1),
                    fmt="d", px_per_row=30, min_height=140)
    with c2:
        st.markdown(section("By entity type", 3), unsafe_allow_html=True)
        typ = pd.DataFrame(
            [{"Type": tx.label_of("entity_type", k), "Entries": v}
             for k, v in (status.get("by_entity_type") or {}).items()])
        if not typ.empty:
            uc.hbar(typ, "Entries", "Type", cap=max(typ["Entries"].max(), 1),
                    fmt="d", px_per_row=30, min_height=140)

    linked = status.get("zenith_sources_linked", 0)
    if linked:
        st.markdown(section("Already flowing into Zenith", 4), unsafe_allow_html=True)
        st.caption(
            f"{linked} of these entries are already ingested as live sources by Zenith's "
            "own scraper — their items appear in the TODAY and ARCHIVE tabs. Filter the "
            "Directory by 'Zenith source' to see which.")
        chips = [uc.chip(e["name"], color=THEME.teal, sub=e["zenith_source"])
                 for e in ents if e.get("zenith_source")]
        st.markdown("".join(chips), unsafe_allow_html=True)

    pod = art.get("podcasts") or {}
    if pod.get("shows"):
        st.markdown(section("Podcast intelligence", 1,
                            help="Full archives of the 14 monitored shows, with guests "
                                 "parsed from episode titles and show notes."),
                    unsafe_allow_html=True)
        total_eps = sum(s["episodes"] for s in pod["shows"])
        st.markdown(uc.numeric_slab([
            {"label": "Shows monitored", "value": f"{len(pod['shows'])}", "color": THEME.text},
            {"label": "Episodes", "value": f"{total_eps:,}", "color": THEME.teal,
             "sub": "full archives, not recent items"},
            {"label": "Guests in directory", "value": f"{pod.get('guests_promoted', 0):,}",
             "color": THEME.mint, "sub": f"of {pod.get('guests_total', 0):,} parsed"},
            {"label": "Appearances", "value": f"{pod.get('records', 0):,}",
             "color": THEME.navy, "sub": "guest-episode links"},
        ]), unsafe_allow_html=True)
        st.caption("See the Podcasts and Guests views for per-show coverage, the guest "
                   "database and newly detected episodes.")

    recent = sorted(ents, key=lambda e: str(e.get("date_added") or ""), reverse=True)[:12]
    if recent:
        st.markdown(section("Recently added", 5), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"Name": e["name"],
                           "Type": tx.label_of("entity_type", e.get("entity_type", "")),
                           "Category": tx.label_of("primary_category", e.get("primary_category", "")),
                           "Added": e.get("date_added", "")} for e in recent]),
            use_container_width=True, hide_index=True, height=min(420, 38 * len(recent) + 40))


# ---------------------------------------------------------------- directory --
def _directory(art: dict) -> None:
    ents = art["entities"]
    rels = _rel_map(ents, art["relationships"])

    st.markdown(section("Search the index", 0,
                        help="Searches names, aliases, descriptions, people, affiliations "
                             "and every tag — by label as well as internal slug."),
                unsafe_allow_html=True)
    q = st.text_input("Search", "", key="idx_q",
                      placeholder="e.g. trend following, volatility, Carver, options analytics")

    c1, c2, c3 = st.columns(3)
    cat = c1.multiselect("Category", [tx.label_of("primary_category", k)
                                      for k in tx.known("primary_category")], key="idx_cat")
    etype = c2.multiselect("Entity type", [tx.label_of("entity_type", k)
                                           for k in tx.known("entity_type")], key="idx_type")
    state = c3.multiselect("Status", [k.replace("_", " ") for k in tx.LIFECYCLE_STATES],
                           key="idx_state")

    c4, c5, c6 = st.columns(3)
    approach = c4.multiselect("Investment approach",
                              _filter_options(ents, "investment_approach", "investment_approach"),
                              key="idx_appr")
    assets = c5.multiselect("Asset class",
                            _filter_options(ents, "asset_classes", "asset_class"), key="idx_asset")
    insight = c6.multiselect("Type of insight",
                             _filter_options(ents, "insight_types", "insight_type"), key="idx_insight")

    c7, c8 = st.columns([1, 3])
    only_zenith = c7.checkbox("Zenith source only", value=False, key="idx_zsrc",
                              help="Show only entries Zenith's scraper already ingests.")
    layout = c8.radio("View as", ["Table", "Cards"], horizontal=True, key="idx_layout")

    rows = ents
    if q.strip():
        from . import model as m
        needle = q.strip().lower()
        rows = [e for e in rows if needle in m.search_blob(e)]
    if cat:
        want = {tx.resolve("primary_category", c) for c in cat}
        rows = [e for e in rows if e.get("primary_category") in want]
    if etype:
        want = {tx.resolve("entity_type", c) for c in etype}
        rows = [e for e in rows if e.get("entity_type") in want]
    if state:
        want = {c.replace(" ", "_") for c in state}
        rows = [e for e in rows if e.get("lifecycle_state") in want]
    for picked, field, vocab in ((approach, "investment_approach", "investment_approach"),
                                 (assets, "asset_classes", "asset_class"),
                                 (insight, "insight_types", "insight_type")):
        if picked:
            want = {tx.resolve(vocab, c) for c in picked}
            rows = [e for e in rows if want & set(e.get(field, []))]
    if only_zenith:
        rows = [e for e in rows if e.get("zenith_source")]

    # Record the result set BEFORE the empty-result early return. Setting it
    # only on the success path left the previous query's names in session state
    # whenever a search matched nothing — stale state that reads as "the filter
    # did nothing" rather than "the filter matched nothing".
    st.session_state["idx_result_names"] = [e["name"] for e in rows]

    st.caption(f"{len(rows)} of {len(ents)} entries match.")
    if not rows:
        st.info("Nothing matches those filters. Try removing one.")
        return

    if layout == "Table":
        _directory_table(rows, rels)
    else:
        _directory_cards(rows, rels)


_COL_HELP = {
    "Name": "Entity name, normalised from the source list.",
    "Type": "What this is — organisation, person, podcast, academic source or tool.",
    "Category": "Top-level classification: institutional, academic or tool.",
    "Specialty": "Investment approach tags.",
    "Asset classes": "Asset classes this source covers.",
    "Insight": "What kind of knowledge it provides.",
    "Institution": "Current affiliation (people) or parent/sub-type (organisations).",
    "Link": "Official website. Blank means no URL could be verified.",
    "Link status": "ok = responded; blocked = alive but refuses robots; error = unreachable.",
    "Status": "Lifecycle state — verified, new, updated, needs review or archived.",
    "Confidence": "How sure we are of the identity and official source.",
    "Zenith source": "The matching feed in Zenith's own scraper, if it ingests this already.",
}


def _directory_table(rows: list[dict], rels: dict) -> None:
    df = pd.DataFrame([{
        "Name": e["name"],
        "Type": tx.label_of("entity_type", e.get("entity_type", "")),
        "Category": tx.label_of("primary_category", e.get("primary_category", "")),
        "Specialty": ", ".join(_tag_labels(e, "investment_approach", "investment_approach")),
        "Asset classes": ", ".join(_tag_labels(e, "asset_classes", "asset_class")),
        "Insight": ", ".join(_tag_labels(e, "insight_types", "insight_type")),
        "Institution": e.get("current_affiliation") or tx.label_of("org_subtype", e.get("org_subtype", "")),
        "Link": e.get("url", ""),
        "Link status": e.get("link_status", "unchecked"),
        "Status": str(e.get("lifecycle_state", "")).replace("_", " "),
        "Confidence": e.get("confidence", ""),
        "Zenith source": e.get("zenith_source", ""),
    } for e in rows])
    cfg = uc.colcfg(df.columns, _COL_HELP)
    cfg["Link"] = st.column_config.LinkColumn("Link", help=_COL_HELP["Link"], display_text="open ↗")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(760, 36 * len(df) + 40), column_config=cfg)


def _directory_cards(rows: list[dict], rels: dict) -> None:
    cols = THEME.section_colors
    cards = []
    for i, e in enumerate(rows[:120]):
        c = cols[i % len(cols)]
        name = _html.escape(e["name"])
        link = e.get("url", "")
        title = (f'<a href="{_html.escape(link)}" target="_blank" rel="noopener" '
                 f'style="color:#fff; text-decoration:none; font-weight:700;">{name} ↗</a>'
                 if link else f'<span style="color:#fff; font-weight:700;">{name}</span>')
        meta = " · ".join(x for x in (
            tx.label_of("entity_type", e.get("entity_type", "")),
            tx.label_of("primary_category", e.get("primary_category", "")),
            e.get("current_affiliation") or "") if x)
        tags = (_tag_labels(e, "investment_approach", "investment_approach")[:4]
                + _tag_labels(e, "insight_types", "insight_type")[:3])
        tag_html = "".join(uc.chip(t, color=c) for t in tags)
        related = rels.get(e["id"], [])[:4]
        rel_html = ("".join(
            f'<div style="font-size:0.74rem; color:{THEME.muted};">{_html.escape(lbl)}: '
            f'<span style="color:#ddd;">{_html.escape(nm)}</span></div>'
            for lbl, nm, _i in related)) if related else ""
        zsrc = (f'<div style="font-size:0.72rem; color:{THEME.teal}; margin-top:0.3rem;">'
                f'in Zenith: {_html.escape(e["zenith_source"])}</div>') if e.get("zenith_source") else ""
        status_bits = uc.chip(str(e.get("lifecycle_state", "")).replace("_", " "),
                              color=_STATE_COLORS.get(e.get("lifecycle_state"), THEME.muted))
        status_bits += _link_chip(e.get("link_status"))
        cards.append(
            f'<div class="z-card" style="border-left:4px solid {c};">'
            f'<div class="z-src" style="color:{c}">{_html.escape(meta)}</div>'
            f'<div class="z-title">{title}</div>'
            f'<div class="z-sum">{_html.escape(e.get("description", ""))}</div>'
            f'<div style="margin-top:0.4rem;">{tag_html}</div>'
            f'{rel_html}{zsrc}'
            f'<div style="margin-top:0.35rem;">{status_bits}</div>'
            f'</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)
    if len(rows) > 120:
        st.caption(f"Showing the first 120 of {len(rows)} — narrow the filters, or use the "
                   "Table view, to see the rest.")


# ------------------------------------------------------------ entity detail --
def _entity_detail(art: dict) -> None:
    ents = art["entities"]
    if not ents:
        st.info("No entries yet.")
        return
    names = sorted((e["name"] for e in ents), key=str.lower)
    default = 0
    prev = st.session_state.get("idx_detail_name")
    if prev in names:
        default = names.index(prev)
    picked = st.selectbox("Entity", names, index=default, key="idx_detail_name")
    ent = next(e for e in ents if e["name"] == picked)
    rels = _rel_map(ents, art["relationships"]).get(ent["id"], [])

    st.markdown(section(ent["name"], 0), unsafe_allow_html=True)
    meta = " · ".join(x for x in (
        tx.label_of("entity_type", ent.get("entity_type", "")),
        tx.label_of("primary_category", ent.get("primary_category", "")),
        tx.label_of("org_subtype", ent["org_subtype"]) if ent.get("org_subtype") else "",
        ent.get("location", ""),
        f"founded {ent['founded']}" if ent.get("founded") else "") if x)
    st.caption(meta)
    if ent.get("description"):
        st.markdown(ent["description"])

    chips = [uc.chip(str(ent.get("lifecycle_state", "")).replace("_", " "),
                     color=_STATE_COLORS.get(ent.get("lifecycle_state"), THEME.muted)),
             _link_chip(ent.get("link_status")),
             uc.chip(f"confidence: {ent.get('confidence', '—')}",
                     color=THEME.mint if ent.get("confidence") == "high" else THEME.mustard)]
    st.markdown("".join(chips), unsafe_allow_html=True)

    for label, field in (("Website", "url"), ("Research", "research_url"), ("Social", "social_url")):
        if ent.get(field):
            st.markdown(f"**{label}:** [{ent[field]}]({ent[field]})")
    if ent.get("zenith_source"):
        st.markdown(f"**Ingested by Zenith as:** `{ent['zenith_source']}` — its items appear "
                    "in the TODAY and ARCHIVE tabs.")

    c1, c2 = st.columns(2)
    with c1:
        for label, field, vocab in (("Investment approach", "investment_approach", "investment_approach"),
                                    ("Asset classes", "asset_classes", "asset_class"),
                                    ("Type of insight", "insight_types", "insight_type")):
            vals = _tag_labels(ent, field, vocab)
            if vals:
                st.markdown(f"**{label}**")
                st.markdown("".join(uc.chip(v, color=THEME.navy) for v in vals),
                            unsafe_allow_html=True)
    with c2:
        if ent.get("current_affiliation"):
            st.markdown(f"**Current affiliation:** {ent['current_affiliation']}")
        if ent.get("role"):
            st.markdown(f"**Role:** {ent['role']}")
        if ent.get("historical_affiliations"):
            st.markdown("**Previously:** " + ", ".join(ent["historical_affiliations"]))
        if ent.get("key_people"):
            st.markdown("**Key people:** " + ", ".join(ent["key_people"]))
        if ent.get("books"):
            st.markdown("**Books:** " + ", ".join(ent["books"]))
        if ent.get("aliases"):
            st.markdown("**Also known as:** " + ", ".join(ent["aliases"]))

    if rels:
        st.markdown(section("Relationships", 3,
                            help="Both directions of the knowledge graph — edges pointing at "
                                 "this entity as well as from it."), unsafe_allow_html=True)
        grouped: dict[str, list[str]] = {}
        for lbl, nm, _i in rels:
            grouped.setdefault(lbl, []).append(nm)
        for lbl, targets in sorted(grouped.items()):
            st.markdown(f"**{lbl}:** " + ", ".join(sorted(set(targets))))

    issues = idx_quality.issues(ent)
    st.markdown(section("Provenance & verification", 5), unsafe_allow_html=True)
    st.caption(f"Source: {ent.get('provenance') or '—'} · added {ent.get('date_added') or '—'} · "
               f"last verified {ent.get('date_last_verified') or 'never'} · "
               f"completeness {idx_quality.completeness(ent):.0%}")
    if ent.get("notes"):
        st.markdown(f"> {ent['notes']}")
    if issues:
        st.markdown("".join(uc.chip(i, color=THEME.mustard) for i in issues),
                    unsafe_allow_html=True)


# ----------------------------------------------------------------- podcasts --
def _podcasts(art: dict) -> None:
    doc = art["podcasts"]
    shows = doc.get("shows") or []
    if not shows:
        st.info("No podcast harvest yet. Run "
                "`python -m zenith.index.compute --action podcasts` to pull the "
                "monitored archives and extract their guests.")
        return

    total_eps = sum(s["episodes"] for s in shows)
    ok_feeds = sum(1 for s in shows if s.get("ok"))
    st.markdown(section("Monitored podcast archives", 0,
                        help="Every feed was resolved through Apple's keyless iTunes "
                             "Search API and then probed. These are full archives, not "
                             "the latest few episodes."),
                unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Shows", "value": f"{len(shows)}", "color": THEME.text,
         "sub": f"{ok_feeds} feeds healthy"},
        {"label": "Episodes", "value": f"{total_eps:,}", "color": THEME.teal,
         "sub": "harvested and stored"},
        {"label": "Guests found", "value": f"{doc.get('guests_total', 0):,}",
         "color": THEME.mint, "sub": f"{doc.get('guests_promoted', 0):,} confident enough "
                                     "to enter the directory"},
        {"label": "Appearances", "value": f"{doc.get('records', 0):,}",
         "color": THEME.navy, "sub": "guest-episode links"},
    ]), unsafe_allow_html=True)

    st.markdown(uc.note_strip("How to read guest coverage", [
        "Coverage is the share of a show's episodes where a guest could be parsed from "
        "the title or notes. A low number means editorial titles, not a broken feed — "
        "Odd Lots names its guest in the title far less often than Alpha Exchange does.",
        "Only guests confident enough to clear the promotion bar become directory "
        "entries. The rest stay recorded against their episode.",
        "Cross-show guests are people who have appeared on more than one monitored "
        "podcast — usually the most interesting names in the graph.",
    ]), unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Podcast": s["podcast"], "Episodes": s["episodes"],
        "First": s.get("earliest", ""), "Latest": s.get("latest", ""),
        "With a guest": s["episodes_with_guest"],
        "Coverage": s["coverage"],
        "Unique guests": s["unique_guests"],
        "Cross-show": s["cross_show_guests"],
        "Feed": "ok" if s.get("ok") else (s.get("error") or "failed"),
    } for s in shows])
    st.dataframe(
        df.style.format({"Coverage": "{:.0%}"}).map(
            lambda v: uc.grad_teal(v, 1.0), subset=["Coverage"]),
        use_container_width=True, hide_index=True, height=min(620, 36 * len(df) + 40),
        column_config=uc.colcfg(df.columns, {
            "Episodes": "Episodes harvested from the show's full archive.",
            "Coverage": "Share of episodes where a guest could be parsed.",
            "Cross-show": "Guests who also appear on another monitored show.",
            "Feed": "Health of the RSS feed on the last harvest.",
        }))

    st.markdown(section("Show detail", 3), unsafe_allow_html=True)
    names = [s["podcast"] for s in shows]
    picked = st.selectbox("Podcast", names, key="idx_pod_pick")
    show = next(s for s in shows if s["podcast"] == picked)

    c1, c2 = st.columns([2, 1])
    with c1:
        if show.get("note"):
            st.caption(show["note"])
        st.markdown(f"**Feed:** `{show['feed_url']}`")
        st.caption(f"{show['episodes']:,} episodes · {show.get('earliest', '?')} → "
                   f"{show.get('latest', '?')} · {show['unique_guests']} unique guests "
                   f"· {show['coverage']:.0%} guest coverage")
        if show.get("hosts"):
            st.markdown("**Hosts:** " + ", ".join(show["hosts"]))
            st.caption("Hosts are excluded from guest extraction — they are the show's "
                       "own voice, and are linked to it by a 'hosts' edge instead.")
    with c2:
        if not show.get("ok"):
            st.markdown(uc.state_banner(THEME.coral, "FEED",
                                        show.get("error") or "failed"),
                        unsafe_allow_html=True)

    guests = doc.get("guests") or {}
    on_show = [g for g in guests.values() if picked in (g.get("podcasts") or [])]
    on_show.sort(key=lambda g: (-g.get("n_appearances", 0), g.get("name", "")))
    if on_show:
        st.markdown(section(f"Guests on {picked} ({len(on_show)})", 4),
                    unsafe_allow_html=True)
        gdf = pd.DataFrame([{
            "Guest": g["name"],
            "Appearances": g.get("n_appearances", 0),
            "Also on": ", ".join(p for p in g.get("podcasts", []) if p != picked),
            "Affiliation": g.get("current_firm", ""),
            "Role": g.get("role", ""),
            "Most recent": (g.get("appearances") or [{}])[0].get("published", ""),
            "Confidence": g.get("confidence", ""),
        } for g in on_show[:400]])
        st.dataframe(gdf, use_container_width=True, hide_index=True,
                     height=min(560, 36 * len(gdf) + 40))
        if len(on_show) > 400:
            st.caption(f"Showing the 400 most frequent of {len(on_show)} guests.")

    eps = [e for e in _episodes(str(doc.get("date", ""))) if e.get("podcast") == picked]
    eps.sort(key=lambda e: e.get("published", ""), reverse=True)
    if eps:
        st.markdown(section("Most recent episodes", 5), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Published": e.get("published", ""), "Episode": e.get("title", ""),
            "Link": e.get("url", ""),
        } for e in eps[:25]]), use_container_width=True, hide_index=True,
            height=min(480, 36 * min(len(eps), 25) + 40),
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="open ↗")})


# ------------------------------------------------------------------- guests --
def _guests(art: dict) -> None:
    doc = art["podcasts"]
    guests = doc.get("guests") or {}
    if not guests:
        st.info("No podcast harvest yet. Run "
                "`python -m zenith.index.compute --action podcasts`.")
        return

    promoted = [g for g in guests.values()
                if idx_guests.meets_threshold(g.get("confidence", "low"))]
    multi = [g for g in promoted if g.get("n_podcasts", 0) >= 2]
    with_firm = [g for g in promoted if g.get("current_firm")]

    st.markdown(section("The guest database", 0,
                        help="Built by parsing the title and show notes of every "
                             "harvested episode. Every field traces to text the "
                             "publisher wrote — nothing is inferred."),
                unsafe_allow_html=True)
    st.markdown(uc.numeric_slab([
        {"label": "Guests", "value": f"{len(promoted):,}", "color": THEME.teal,
         "sub": f"of {len(guests):,} parsed"},
        {"label": "On 2+ shows", "value": f"{len(multi):,}", "color": THEME.mint,
         "sub": "cross-show researchers"},
        {"label": "With an affiliation", "value": f"{len(with_firm):,}",
         "color": THEME.navy, "sub": "firm stated in metadata"},
        {"label": "Appearances", "value": f"{doc.get('records', 0):,}",
         "color": THEME.mustard, "sub": "guest-episode links"},
    ]), unsafe_allow_html=True)

    q = st.text_input("Search guests", "", key="idx_guest_q",
                      placeholder="name, firm or podcast")
    c1, c2 = st.columns([1, 1])
    only_multi = c1.checkbox("Only guests on 2+ shows", value=False, key="idx_guest_multi")
    min_apps = c2.slider("Minimum appearances", 1, 10, 1, key="idx_guest_min")

    rows = promoted
    if q.strip():
        needle = q.strip().lower()
        rows = [g for g in rows
                if needle in g["name"].lower()
                or needle in str(g.get("current_firm", "")).lower()
                or any(needle in p.lower() for p in g.get("podcasts", []))]
    if only_multi:
        rows = [g for g in rows if g.get("n_podcasts", 0) >= 2]
    rows = [g for g in rows if g.get("n_appearances", 0) >= min_apps]
    rows.sort(key=lambda g: (-g.get("n_appearances", 0), g.get("name", "")))

    st.caption(f"{len(rows):,} of {len(promoted):,} guests match.")
    st.session_state["idx_guest_names"] = [g["name"] for g in rows]
    if not rows:
        st.info("No guests match those filters.")
        return

    gdf = pd.DataFrame([{
        "Guest": g["name"],
        "Appearances": g.get("n_appearances", 0),
        "Shows": g.get("n_podcasts", 0),
        "Podcasts": ", ".join(g.get("podcasts", [])),
        "Affiliation": g.get("current_firm", ""),
        "Role": g.get("role", ""),
        "Most recent": (g.get("appearances") or [{}])[0].get("published", ""),
    } for g in rows[:500]])
    st.dataframe(gdf, use_container_width=True, hide_index=True,
                 height=min(620, 36 * len(gdf) + 40),
                 column_config=uc.colcfg(gdf.columns, {
                     "Shows": "How many different monitored podcasts this person has "
                              "appeared on.",
                     "Affiliation": "Firm as stated in the episode metadata — not "
                                    "independently verified.",
                 }))
    if len(rows) > 500:
        st.caption(f"Showing the 500 most frequent of {len(rows):,}.")

    st.markdown(section("Guest detail", 3), unsafe_allow_html=True)
    pick = st.selectbox("Guest", [g["name"] for g in rows[:500]], key="idx_guest_pick")
    prof = next(g for g in rows if g["name"] == pick)
    st.caption(" · ".join(x for x in (
        f"{prof.get('n_appearances', 0)} appearance(s)",
        f"{prof.get('n_podcasts', 0)} show(s)",
        prof.get("role", ""), prof.get("current_firm", ""),
        f"confidence {prof.get('confidence', '')}") if x))
    if prof.get("past_firms"):
        st.markdown("**Previously associated with:** " + ", ".join(prof["past_firms"]))
    apps = prof.get("appearances") or []
    if apps:
        st.dataframe(pd.DataFrame([{
            "Published": a.get("published", ""), "Podcast": a.get("podcast", ""),
            "Episode": a.get("title", ""), "Link": a.get("url", ""),
        } for a in apps]), use_container_width=True, hide_index=True,
            height=min(420, 36 * len(apps) + 40),
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="open ↗")})
        st.caption("Appearances shown here are the most recent stored on the profile; "
                   "the full archive lives in episodes.json.")

    new_eps = art["status"].get("new_episodes") or []
    if new_eps:
        st.markdown(section("Newly detected episodes", 5,
                            help="Episodes that appeared in a feed for the first time on "
                                 "the most recent harvest."), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Published": e.get("published", ""), "Podcast": e.get("podcast", ""),
            "Episode": e.get("title", ""),
        } for e in new_eps]), use_container_width=True, hide_index=True,
            height=min(360, 36 * len(new_eps) + 40))


# ----------------------------------------------------------------- taxonomy --
def _taxonomy(art: dict) -> None:
    ents = art["entities"]
    st.markdown(section("Controlled vocabularies", 0,
                        help="The taxonomy is data, not code — adding a term is a single "
                             "dictionary entry, so it can grow without a migration."),
                unsafe_allow_html=True)
    st.caption("Counts show how many entries currently carry each term. Terms with zero "
               "entries are declared but unused — they cost nothing and keep the vocabulary "
               "stable as the index grows.")

    field_for = {"primary_category": "primary_category", "entity_type": "entity_type",
                 "org_subtype": "org_subtype", "investment_approach": "investment_approach",
                 "asset_class": "asset_classes", "insight_type": "insight_types"}
    vocab = st.selectbox("Vocabulary", list(tx.VOCABULARIES),
                         format_func=lambda v: tx.VOCABULARY_LABELS.get(v, v), key="idx_vocab")
    field = field_for[vocab]
    counts: dict[str, int] = {}
    for e in ents:
        v = e.get(field)
        for item in (v if isinstance(v, list) else [v]):
            if item:
                counts[item] = counts.get(item, 0) + 1

    df = pd.DataFrame([{"Term": tx.label_of(vocab, slug), "Slug": slug,
                        "Entries": counts.get(slug, 0),
                        "Aliases": ", ".join(aliases)}
                       for slug, (_lbl, aliases) in tx.VOCABULARIES[vocab].items()])
    df = df.sort_values("Entries", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(700, 36 * len(df) + 40))

    undeclared = (art["status"].get("undeclared_tags") or {}).get(vocab, [])
    if undeclared:
        st.markdown(section("In use but not declared", 2), unsafe_allow_html=True)
        st.caption("These tags appear on entries but are not in the vocabulary yet. They are "
                   "preserved rather than dropped, and are the queue for growing the taxonomy "
                   "deliberately.")
        st.markdown("".join(uc.chip(t, color=THEME.mustard) for t in undeclared),
                    unsafe_allow_html=True)


# ---------------------------------------------------------- data management --
def _data_management(art: dict) -> None:
    ents, rels, status = art["entities"], art["relationships"], art["status"]

    st.markdown(section("Download the Master List", 0,
                        help="Self-contained exports. The spreadsheet formats flatten lists "
                             "with '; ' and include human labels alongside internal slugs; "
                             "JSON preserves the full nested structure."),
                unsafe_allow_html=True)
    st.caption("The Excel workbook carries three sheets — the directory, the relationship "
               "graph with names resolved, and the taxonomy key — so the file explains "
               "itself outside Zenith.")
    c1, c2, c3 = st.columns(3)
    day = status.get("date")
    try:
        c1.download_button("⤓  CSV", data=idx_export.to_csv(ents, rels),
                           file_name=idx_export.filename("csv", day), mime="text/csv",
                           use_container_width=True)
        c2.download_button("⤓  Excel (XLSX)", data=idx_export.to_xlsx(ents, rels, status),
                           file_name=idx_export.filename("xlsx", day),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        c3.download_button("⤓  JSON", data=idx_export.to_json(ents, rels, status),
                           file_name=idx_export.filename("json", day),
                           mime="application/json", use_container_width=True)
    except Exception as exc:                                   # pragma: no cover
        st.error(f"Export failed: {type(exc).__name__}: {exc}")

    st.markdown(section("Link health", 2), unsafe_allow_html=True)
    ls = status.get("links", {})
    if not ls.get("checked"):
        st.info("No link sweep has run yet. Run `python -m zenith.index.compute --action links`.")
    else:
        st.markdown(uc.numeric_slab([
            {"label": "Responded", "value": f"{ls.get('ok', 0)}", "color": THEME.teal},
            {"label": "Blocked to robots", "value": f"{ls.get('blocked', 0)}",
             "color": THEME.mustard, "sub": "alive, but refuses automation"},
            {"label": "Unreachable", "value": f"{ls.get('error', 0)}", "color": THEME.coral},
            {"label": "No URL", "value": f"{status.get('missing_url', 0)}", "color": THEME.muted},
        ]), unsafe_allow_html=True)
        bad = [e for e in ents if e.get("link_status") in ("error", "missing", "blocked")]
        if bad:
            st.dataframe(pd.DataFrame([{
                "Name": e["name"], "Link status": e.get("link_status"),
                "URL": e.get("url", ""), "Note": e.get("notes", "")[:120],
            } for e in sorted(bad, key=lambda e: (e.get("link_status", ""), e["name"]))]),
                use_container_width=True, hide_index=True, height=min(460, 36 * len(bad) + 40))

    st.markdown(section("Review queue", 3,
                        help="Worst first — most issues, then least complete. These are the "
                             "entries where a few minutes of work removes the most uncertainty."),
                unsafe_allow_html=True)
    queue = status.get("review_queue") or idx_quality.review_queue(ents)
    if not queue:
        st.success("Nothing flagged — every entry has a description, tags and a checked link.")
    else:
        st.dataframe(pd.DataFrame([{
            "Name": r["name"], "Type": tx.label_of("entity_type", r.get("entity_type", "")),
            "Confidence": r.get("confidence", ""),
            "Completeness": f"{r.get('completeness', 0):.0%}",
            "Issues": "; ".join(r.get("issues", [])),
        } for r in queue]), use_container_width=True, hide_index=True,
            height=min(560, 36 * len(queue) + 40))

    st.markdown(section("Import audit", 4,
                        help="What the importer did to the supplied list — every merge, and "
                             "every near-duplicate it deliberately left alone."),
                unsafe_allow_html=True)
    report = status.get("dedupe_report") or []
    if report:
        st.dataframe(pd.DataFrame([{
            "Action": r["action"].replace("_", " "), "Kept": r["primary"],
            "Other": r["other"], "Reason": r["reason"],
        } for r in report]), use_container_width=True, hide_index=True,
            height=min(360, 36 * len(report) + 40))
    else:
        st.caption("No merges or near-duplicates recorded in the last import.")

    cands = status.get("duplicate_candidates") or []
    if cands:
        st.caption("Possible duplicates sharing a domain with nothing declared to explain it:")
        for c in cands:
            st.markdown(f"- `{c['domain']}` — " + ", ".join(c["names"]))

    orphans = status.get("orphan_edges", 0)
    if orphans:
        st.markdown(uc.state_banner(THEME.coral, "GRAPH INTEGRITY",
                                    f"{orphans} relationship(s) point at a missing entity."),
                    unsafe_allow_html=True)

    st.markdown(section("Provenance", 5), unsafe_allow_html=True)
    st.caption("The raw supplied lists are committed verbatim so every entry can be traced "
               "back to what was actually provided:")
    for f in status.get("seed_files", []):
        st.markdown(f"- `{f}`")
    st.caption(f"Last run: {status.get('date', '—')} (`--action {status.get('action', '—')}`) · "
               f"mean completeness {status.get('mean_completeness', 0):.0%}")


# ------------------------------------------------------------------ render --
def render() -> None:
    st.caption(DISCLAIMER)
    status = load("status", {})
    art = _artefacts(cache_bust=str(status.get("date", "")))

    if not art["entities"]:
        st.info("The index has not been built yet. Run "
                "`python -m zenith.index.compute --action seed` to import the seed list, "
                "then `--action links` to verify every URL.")
        return

    st.markdown(stamp(status.get("date", "—"), "Index"), unsafe_allow_html=True)
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)

    sub = st.radio("View", SUBVIEWS, horizontal=True, key="idx_sub", label_visibility="collapsed")
    if sub == "Overview":
        _overview(art)
    elif sub == "Directory":
        _directory(art)
    elif sub == "Entity detail":
        _entity_detail(art)
    elif sub == "Podcasts":
        _podcasts(art)
    elif sub == "Guests":
        _guests(art)
    elif sub == "Taxonomy":
        _taxonomy(art)
    else:
        _data_management(art)

    with st.expander("Methodology, scope and what is coming next"):
        st.markdown(
            "**What this is.** A curated directory of the financial information ecosystem — "
            "institutions, people, academic sources, podcasts and tools — imported from a "
            "supplied resource list, deduplicated, classified against an extensible "
            "taxonomy, and connected to itself by a relationship graph.\n\n"
            "**Why there is no A/B/C evidence badge.** Every other Zenith tab carries one, "
            "but that badge grades a signal's out-of-sample predictive strength. A directory "
            "has no such property, so stamping a letter on it would be inventing precision. "
            "The verification strip on the Overview is the honest equivalent: what has "
            "actually been checked, and when.\n\n"
            "**How links are judged.** Every URL is probed by a real HTTP request with a "
            "browser user-agent. A host that is alive but refuses automated requests "
            "(Cloudflare, HTTP 403, or a robots.txt rule) is recorded as **blocked**, not "
            "broken — Citadel, SSRN and several journal publishers are in this category and "
            "open perfectly well in a browser.\n\n"
            "**What is deliberately empty.** Where an entity's identity or official source "
            "could not be established, the field is left blank and the entry is flagged "
            "*needs review*, rather than pointed at a plausible-looking domain. The review "
            "queue in Data Management lists every one of them.\n\n"
            "**History is kept.** When a person changes firm the previous affiliation moves "
            "into their history rather than being overwritten, so the record of where "
            "someone came from survives.\n\n"
            "**Coming next.** *Phase 2* — podcast intelligence: harvesting the full episode "
            "archives of the 14 monitored shows (~5,700 episodes, all confirmed available) "
            "and extracting a guest database linked to firms, strategies and episodes. "
            "*Phase 3* — the network visualisation, a discovery surface for finding "
            "researchers worth following, and transparent relevance ranking.")

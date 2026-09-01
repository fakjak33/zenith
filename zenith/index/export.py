"""INDEX export — the Master List as CSV, Excel or JSON.

Zenith had no download path anywhere before this (no ``st.download_button``,
``to_csv`` or ``BytesIO`` in the codebase), so this module is written to be
reusable by other tabs later rather than wired into the INDEX view alone.

THE EXPORT MUST STAND ALONE. The requirement is that the downloaded file is
"clean enough to use outside Zenith", which rules out dumping internal shapes:

  * list fields are joined with "; " — a stable, spreadsheet-safe separator that
    survives a round-trip and does not collide with the commas inside names
    like "Baker Bros. Advisors, LP";
  * tag slugs are ALSO rendered as human labels, because ``trend_following`` is
    an internal key and "Trend following" is what a reader needs;
  * relationships are resolved to NAMES, not ids — an edge list of hex ids is
    useless in a spreadsheet — and shipped both as a per-entity summary column
    and as their own sheet/array for anyone who wants the real graph;
  * the column order is ``model.FIELDS`` order, which is arranged identity →
    classification → links → people → tags → provenance, i.e. the order someone
    reading left-to-right would want.

XLSX uses openpyxl, already a Zenith dependency (FMOM's Morningstar catalog
reader), so this adds nothing to requirements.txt.
"""

from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd

from . import model as m
from . import taxonomy as tx

LIST_SEP = "; "

# Tag fields rendered twice: raw slugs (machine-readable, round-trippable) and
# human labels (readable). Both, because the export serves both audiences.
_LABELLED = {"investment_approach": "investment_approach",
             "asset_classes": "asset_class",
             "insight_types": "insight_type"}

# Internal-only fields that would just be noise in a spreadsheet.
_SKIP = {"slug"}


def _join(v) -> str:
    if isinstance(v, list):
        return LIST_SEP.join(
            str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in v if x not in (None, ""))
    return "" if v is None else str(v)


def _relationship_names(entities: list[dict], relationships: list[dict]) -> dict[str, list[str]]:
    """Per-entity human-readable relationship summaries, both directions.

    Both directions matter: a firm's row should show the people who work there
    even though the ``works_at`` edge is stored pointing the other way.
    """
    names = {e["id"]: e.get("name", "") for e in entities}
    out: dict[str, list[str]] = {e["id"]: [] for e in entities}
    inverse = {"works_at": "employs", "worked_at": "previously employed",
               "founded": "founded by", "subsidiary_of": "parent of",
               "publishes": "published by", "hosts": "hosted by",
               "appeared_on": "guest", "related_to": "related to"}
    for r in relationships:
        s, t, typ = r.get("source"), r.get("target"), str(r.get("type") or "related_to")
        if s in out and t in names:
            out[s].append(f"{typ.replace('_', ' ')}: {names[t]}")
        if t in out and s in names:
            out[t].append(f"{inverse.get(typ, typ).replace('_', ' ')}: {names[s]}")
    return out


def to_rows(entities: list[dict], relationships: list[dict] | None = None) -> list[dict]:
    """Flatten entities into export-ready rows."""
    rels = _relationship_names(entities, relationships or [])
    rows: list[dict] = []
    for ent in entities:
        row: dict[str, object] = {}
        for field in m.FIELDS:
            if field in _SKIP:
                continue
            row[field] = _join(ent.get(field))
        for field, vocab in _LABELLED.items():
            row[f"{field}_labels"] = LIST_SEP.join(
                tx.label_of(vocab, s) for s in ent.get(field, []))
        row["entity_type_label"] = tx.label_of("entity_type", ent.get("entity_type", ""))
        row["primary_category_label"] = tx.label_of("primary_category",
                                                    ent.get("primary_category", ""))
        row["relationships"] = LIST_SEP.join(rels.get(ent["id"], []))
        row["relationship_count"] = len(rels.get(ent["id"], []))
        rows.append(row)
    return rows


def to_dataframe(entities: list[dict], relationships: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(to_rows(entities, relationships))


def relationships_frame(entities: list[dict], relationships: list[dict]) -> pd.DataFrame:
    """The edge list with names resolved — the graph, usable on its own."""
    names = {e["id"]: e.get("name", "") for e in entities}
    types = {e["id"]: e.get("entity_type", "") for e in entities}
    return pd.DataFrame([{
        "source": names.get(r.get("source"), r.get("source")),
        "source_type": types.get(r.get("source"), ""),
        "relationship": str(r.get("type") or "").replace("_", " "),
        "target": names.get(r.get("target"), r.get("target")),
        "target_type": types.get(r.get("target"), ""),
        "note": r.get("note", ""),
        "source_id": r.get("source"), "target_id": r.get("target"),
    } for r in relationships])


def to_csv(entities: list[dict], relationships: list[dict] | None = None) -> bytes:
    """UTF-8 BOM so Excel opens accented names correctly on Windows — without
    it "Torsten Sløk" and "Svante Bergström" arrive mojibaked."""
    return to_dataframe(entities, relationships).to_csv(index=False).encode("utf-8-sig")


def to_json(entities: list[dict], relationships: list[dict] | None = None,
            status: dict | None = None) -> bytes:
    """Full-fidelity export: nested structures preserved, nothing flattened."""
    payload = {
        "generated": date.today().isoformat(),
        "source": "Zenith INDEX — Master List",
        "counts": {"entities": len(entities), "relationships": len(relationships or [])},
        "status": status or {},
        "taxonomy": {
            vocab: {slug: label for slug, (label, _a) in table.items()}
            for vocab, table in tx.VOCABULARIES.items()
        },
        "entities": entities,
        "relationships": relationships or [],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def to_xlsx(entities: list[dict], relationships: list[dict] | None = None,
            status: dict | None = None) -> bytes:
    """Multi-sheet workbook: the directory, the graph, and the taxonomy key.

    The taxonomy sheet is included so the file is self-describing — a reader who
    has never seen Zenith can still tell what ``alternative_risk_premia`` means.
    """
    relationships = relationships or []
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        to_dataframe(entities, relationships).to_excel(xl, sheet_name="Master List", index=False)
        rf = relationships_frame(entities, relationships)
        if not rf.empty:
            rf.to_excel(xl, sheet_name="Relationships", index=False)
        tax_rows = [{"vocabulary": tx.VOCABULARY_LABELS.get(vocab, vocab),
                     "slug": slug, "label": label,
                     "aliases": LIST_SEP.join(aliases)}
                    for vocab, table in tx.VOCABULARIES.items()
                    for slug, (label, aliases) in table.items()]
        pd.DataFrame(tax_rows).to_excel(xl, sheet_name="Taxonomy", index=False)
        if status:
            pd.DataFrame([{"metric": k, "value": _join(v) if isinstance(v, list) else
                           (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)}
                          for k, v in status.items()]).to_excel(
                xl, sheet_name="Status", index=False)
    return buf.getvalue()


def filename(ext: str, day: str | None = None) -> str:
    return f"zenith_master_list_{day or date.today().isoformat()}.{ext}"

"""Build and validate one daily snapshot.

Validation is FAIL CLOSED: if a snapshot does not pass, `compute` refuses to
archive it or to overwrite `latest.json`. A parser that silently starts
returning three rows because the fund company changed its markup must show up
as "data unavailable", never as "DBMF closed most of its book today".
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from . import normalize

# --- validation thresholds --------------------------------------------------
MIN_ROWS = 5                 # a real file is ~18 rows; fewer means a parse break
MAX_STALE_DAYS = 10          # a live fetch older than this is a broken source
MAX_NAV_MOVE = 0.25          # day-over-day NAV move that implies a bad parse
GROSS_BAND = (0.2, 8.0)      # plausible gross notional exposure for a CTA book
MIN_HTML_BYTES = 200_000     # the source page is ~6 MB; a stub means a partial GET
MAX_WEIGHT = 5.0             # no single position should be 500% of NAV
TURNOVER_WARN = 0.90         # >90% of gross changing in a day is suspicious


def _parse_value_date(raw: str) -> date | None:
    """The custodian publishes MM/DD/YYYY; accept ISO too."""
    s = (raw or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def build(fund: str, rows: list[dict], meta: dict,
          previous: dict | None = None, live: bool = True) -> dict:
    """Assemble a validated snapshot from parsed rows.

    ``meta`` carries source provenance from the adapter: ``value_date``,
    ``url``, ``kind``, ``via``, ``html_bytes``, ``retrieved_at``.
    ``live`` is False for Wayback replays, which are legitimately old.
    """
    errors: list[str] = []
    warnings: list[str] = []

    as_of = _parse_value_date(meta.get("value_date", ""))
    if as_of is None:
        errors.append(f"unparseable value_date {meta.get('value_date')!r}")
    else:
        today = date.today()
        if as_of > today:
            errors.append(f"value_date {as_of} is in the future")
        elif live and (today - as_of).days > MAX_STALE_DAYS:
            errors.append(f"value_date {as_of} is {(today - as_of).days} days old")

    if len(rows) < MIN_ROWS:
        errors.append(f"only {len(rows)} rows parsed (min {MIN_ROWS}) — parse break?")

    html_bytes = int(meta.get("html_bytes") or 0)
    if html_bytes and html_bytes < MIN_HTML_BYTES:
        errors.append(f"source page only {html_bytes} bytes — partial download?")

    norm = normalize.normalize_rows(rows, asof=as_of)
    positions, nav = norm["positions"], norm["nav"]

    if nav is None:
        errors.append("no TOTAL NET ASSETS row — parse break?")
    elif nav <= 0:
        errors.append(f"non-positive NAV {nav}")
    elif previous and previous.get("nav"):
        move = abs(nav / previous["nav"] - 1.0)
        if move > MAX_NAV_MOVE:
            errors.append(f"NAV moved {move:.1%} vs {previous.get('as_of')}")

    summary = normalize.exposure_summary(positions)
    if nav and positions:
        if not GROSS_BAND[0] <= summary["gross"] <= GROSS_BAND[1]:
            errors.append(f"gross exposure {summary['gross']:.2f}x outside "
                          f"{GROSS_BAND[0]}-{GROSS_BAND[1]}x band")
        hot = [p["id"] for p in positions if abs(p["weight"]) > MAX_WEIGHT]
        if hot:
            errors.append(f"implausible weight on {', '.join(hot)}")

    for f in norm["flags"]:
        warnings.append(f)

    if previous and previous.get("positions") and positions:
        turnover = _turnover(previous["positions"], positions)
        if turnover > TURNOVER_WARN:
            warnings.append(f"suspicious_turnover:{turnover:.0%}")
    else:
        turnover = None

    snap = {
        "fund": fund,
        "as_of": as_of.isoformat() if as_of else None,
        "retrieved_at": meta.get("retrieved_at")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"kind": meta.get("kind", "unknown"), "url": meta.get("url", ""),
                   "via": meta.get("via", "direct"),
                   # For Wayback replays: which capture this came from, so a
                   # re-run can skip it without re-downloading 3 MB of HTML.
                   "capture_day": meta.get("capture_day"),
                   "html_bytes": html_bytes or None},
        "nav": round(nav, 2) if nav else None,
        "n_rows": len(rows),
        "n_positions": len(positions),
        "positions": positions,
        "summary": summary,
        "quality": {
            "ok": not errors,
            "errors": errors,
            "warnings": sorted(set(warnings)),
            "turnover": round(turnover, 4) if turnover is not None else None,
        },
    }
    return snap


def _turnover(old: list[dict], new: list[dict]) -> float:
    """Share of gross exposure that changed between two books, 0-1+."""
    ow = {p["id"]: p["weight"] for p in old
          if p["asset_class"] in normalize.DIRECTIONAL}
    nw = {p["id"]: p["weight"] for p in new
          if p["asset_class"] in normalize.DIRECTIONAL}
    gross = sum(abs(v) for v in ow.values()) or 1.0
    moved = sum(abs(nw.get(k, 0.0) - ow.get(k, 0.0)) for k in set(ow) | set(nw))
    return moved / gross


def is_ok(snap: dict) -> bool:
    return bool(snap.get("quality", {}).get("ok"))


def reason(snap: dict) -> str:
    errs = snap.get("quality", {}).get("errors", [])
    return "; ".join(errs)[:300] if errs else ""

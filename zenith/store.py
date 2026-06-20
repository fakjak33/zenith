"""JSON storage: per-day archive, latest run, dedup 'seen' index, source status."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from urllib.parse import urlsplit, urlunsplit

from .config import ARCHIVE_DIR, LATEST_JSON, SEEN_JSON, STATUS_JSON, USAGE_JSON


def normalize_url(url: str) -> str:
    """Strip query/fragment + lowercase host for stable dedup keys."""
    try:
        s = urlsplit(url.strip())
        return urlunsplit((s.scheme.lower(), s.netloc.lower(), s.path.rstrip("/"), "", ""))
    except Exception:
        return (url or "").strip().lower()


def item_id(url: str, title: str = "") -> str:
    """Stable hash for dedup — URL-based, title fallback."""
    key = normalize_url(url) or (title or "").strip().lower()
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load_seen() -> set[str]:
    return set(_read(SEEN_JSON, []))


def save_seen(seen: set[str]) -> None:
    SEEN_JSON.write_text(json.dumps(sorted(seen)), encoding="utf-8")


def write_latest(items: list[dict]) -> None:
    LATEST_JSON.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def load_latest() -> list[dict]:
    return _read(LATEST_JSON, [])


def write_archive(day: str, items: list[dict]) -> None:
    (ARCHIVE_DIR / f"{day}.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def load_archive(day: str) -> list[dict]:
    return _read(ARCHIVE_DIR / f"{day}.json", [])


def archive_dates() -> list[str]:
    return sorted((p.stem for p in ARCHIVE_DIR.glob("*.json")), reverse=True)


def save_status(status: list[dict]) -> None:
    STATUS_JSON.write_text(json.dumps(status, indent=2), encoding="utf-8")


def load_status() -> list[dict]:
    return _read(STATUS_JSON, [])


def save_apify_usage(usage: dict) -> None:
    USAGE_JSON.write_text(json.dumps(usage, indent=2), encoding="utf-8")


def load_apify_usage() -> dict:
    return _read(USAGE_JSON, {})


def today_str() -> str:
    return date.today().isoformat()

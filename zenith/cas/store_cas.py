"""JSON IO for the CAS monitor — mirrors zenith.store but writes under data/cas/.

Each artefact (signals, positioning, themes, consensus, overlap, …) is one JSON
file named in config.CAS_FILES. Per-day archives of the full signal set go under
data/cas/archive/<date>.json so the viewer can show history.
"""

from __future__ import annotations

import json
from datetime import date

from ..config import CAS_FILES, CAS_ARCHIVE_DIR, CAS_CACHE_DIR


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(name: str, default=None):
    """Load a named CAS artefact (see config.CAS_FILES)."""
    return _read(CAS_FILES[name], default if default is not None else [])


def save(name: str, obj) -> None:
    _write(CAS_FILES[name], obj)


def archive_signals(day: str, signals: list[dict]) -> None:
    _write(CAS_ARCHIVE_DIR / f"{day}.json", signals)


def archive_dates() -> list[str]:
    return sorted((p.stem for p in CAS_ARCHIVE_DIR.glob("*.json")), reverse=True)


def load_archive(day: str) -> list[dict]:
    return _read(CAS_ARCHIVE_DIR / f"{day}.json", [])


# --- raw-data cache (prices, COT, …) so reruns within a day don't re-download ---
def cache_get(key: str, max_age_hours: float = 18.0):
    p = CAS_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        import time
        if (time.time() - p.stat().st_mtime) > max_age_hours * 3600:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_put(key: str, obj) -> None:
    _write(CAS_CACHE_DIR / f"{key}.json", obj)


def today_str() -> str:
    return date.today().isoformat()

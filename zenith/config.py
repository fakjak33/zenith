"""Zenith config: paths + ported Parallax theme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
LATEST_JSON = DATA_DIR / "latest.json"
SEEN_JSON = DATA_DIR / "seen.json"
STATUS_JSON = DATA_DIR / "status.json"
for _d in (DATA_DIR, ARCHIVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# polite scraping
USER_AGENT = "ZenithResearchAggregator/0.1 (+personal research; respects robots.txt)"
REQUEST_TIMEOUT = 15
MAX_ITEMS_PER_SOURCE = 25       # cap per source per run
CLASSIFY_FETCH = True           # fetch article pages to classify text vs visual
CLASSIFY_MAX_FETCH = 120        # cap page fetches per run (politeness/time)


@dataclass(frozen=True)
class Theme:
    bg: str = "#000000"
    panel: str = "#0b0b0b"
    grid: str = "#2c2c2c"
    border: str = "#ffffff"
    teal: str = "#2ec4b6"
    coral: str = "#ff5a3c"
    orange: str = "#ff8c2b"
    mustard: str = "#ffc857"
    mauve: str = "#c46b8b"
    mint: str = "#7bdcb5"
    navy: str = "#2a9bc4"
    text: str = "#ffffff"
    muted: str = "#b8b8b8"
    font_display: str = "'VT323', 'Space Mono', 'Courier New', monospace"
    font_body: str = "'Space Mono', 'Share Tech Mono', 'Courier New', monospace"
    section_colors: tuple = ("#2ec4b6", "#ffc857", "#ff5a3c", "#c46b8b", "#2a9bc4", "#7bdcb5")


THEME = Theme()

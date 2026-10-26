"""Write the ZENITH brand files from zenith.art (re-run after changing the mark).

    python scripts/build_brand.py

Outputs: static/zenith_lockup.svg (sidebar logo), static/zenith_mark.svg (collapsed
sidebar icon), assets/favicon.png and static/logo.png (browser tab).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zenith import art  # noqa: E402
from zenith.config import THEME as T  # noqa: E402


def lockup_svg() -> str:
    mark = art.mark(56, animate=False)
    inner = mark[mark.index(">") + 1: mark.rindex("</svg>")]
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="64" viewBox="0 0 300 64">'
            f'<g transform="translate(4,4) scale(0.4667)">{inner}</g>'
            f'<text x="62" y="36" fill="{T.text}" font-family="JetBrains Mono, IBM Plex Mono, monospace" '
            f'font-weight="300" font-size="22" letter-spacing="11">ZENITH</text>'
            f'<text x="63" y="53" fill="{T.muted}" font-family="JetBrains Mono, IBM Plex Mono, monospace" '
            f'font-size="7.5" letter-spacing="3.2">HIGHER PERSPECTIVE. CLEARER SIGNALS.</text></svg>')


def main() -> None:
    static = ROOT / "static"
    static.mkdir(exist_ok=True)
    (static / "zenith_lockup.svg").write_text(lockup_svg(), encoding="utf-8")
    (static / "zenith_mark.svg").write_text(art.mark(64, animate=False), encoding="utf-8")
    art.favicon_png(ROOT / "assets" / "favicon.png", 128)
    art.favicon_png(static / "logo.png", 256)
    print("wrote static/zenith_lockup.svg, static/zenith_mark.svg, assets/favicon.png, static/logo.png")


if __name__ == "__main__":
    main()

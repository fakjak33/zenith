"""Build the Zenith disc logo: palette-gradient colour fields, fully transparent
ribbon + background.

Reuses the exact ribbon/disc geometry of ``assets/logo_raw.png`` (the original
art) but repaints the coloured fields with Zenith-palette gradients and makes all
cream (the 'Z' ribbon and the outside-disc background) transparent.

  fields = disc & ~cream        # the coloured area of the original
  - paint each `fields` pixel by ANGLE from the disc centre (conic), matching the
    original orientation: top=orange, right=coral, bottom=navy, left=mint/teal
  - a gentle radial light->deep gradient adds depth
  - ribbon and outside-disc stay alpha=0 (fully transparent)

No cairo / SVG renderer needed — pure Pillow + numpy.

Usage:  python scripts/build_logo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
STATIC = ROOT / "static"

# Zenith palette (config.Theme) as RGB
MINT = (123, 220, 181)
TEAL = (46, 196, 182)
NAVY = (42, 155, 196)
ORANGE = (255, 140, 43)
MUSTARD = (255, 200, 87)
CORAL = (255, 90, 60)
DEEP_NAVY = (22, 63, 84)


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _quantize_palette(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Snap a colour to the nearest Zenith-palette anchor so the pixel art reads
    as a limited palette rather than a smooth gradient."""
    anchors = (MINT, TEAL, NAVY, ORANGE, MUSTARD, CORAL, DEEP_NAVY)
    return min(anchors, key=lambda a: sum((rgb[i] - a[i]) ** 2 for i in range(3)))


def _retro(img: Image.Image, grid: int = 56) -> Image.Image:
    """8-bit / CRT post-process: pixelate to a small grid, quantize to the
    palette, then add scanlines + grain + a subtle RGB-split glitch. Alpha-aware
    so the transparent ribbon/background stay transparent."""
    size = img.size[0]
    # 1) pixelate — downsample then nearest-neighbour upscale
    small = img.resize((grid, grid), Image.BILINEAR)
    a = np.asarray(small).astype(np.int16)
    op = a[:, :, 3] > 110                      # treat as opaque pixel-cell
    # 2) quantize opaque cells to palette anchors
    for j in range(grid):
        for i in range(grid):
            if op[j, i]:
                q = _quantize_palette((int(a[j, i, 0]), int(a[j, i, 1]), int(a[j, i, 2])))
                a[j, i, :3] = q
            else:
                a[j, i, 3] = 0                 # hard alpha edge = crisp pixels
    # 3) grain — subtle per-cell brightness noise on opaque cells
    rng = np.random.default_rng(7)
    noise = rng.integers(-18, 19, size=(grid, grid, 1))
    a[:, :, :3] = np.clip(a[:, :, :3] + noise * op[:, :, None], 0, 255)
    # 4) scanlines — darken every other row
    a[0::2, :, :3] = (a[0::2, :, :3] * 0.82).astype(np.int16)
    px = Image.fromarray(a.astype(np.uint8), "RGBA").resize((size, size), Image.NEAREST)

    # 5) subtle glitch — shift the red channel left on a couple of horizontal bands
    arr = np.asarray(px).astype(np.uint8).copy()
    band = max(2, size // grid)
    for y0 in (int(size * 0.34), int(size * 0.62)):
        sl = slice(y0, y0 + band)
        arr[sl, :-3, 0] = arr[sl, 3:, 0]       # red tear right→left
    return Image.fromarray(arr, "RGBA")


def _angular_color(deg: float) -> tuple[int, int, int]:
    """Map an angle (0=up, clockwise) to a palette colour, blending between the
    four anchors at top / right / bottom / left so the wheel is continuous."""
    a = deg % 360.0
    # anchors: 0=top(orange), 90=right(coral), 180=bottom(navy), 270=left(teal)
    if a < 90:      # top -> right
        return _lerp(ORANGE, CORAL, a / 90.0)
    if a < 180:     # right -> bottom
        return _lerp(CORAL, NAVY, (a - 90) / 90.0)
    if a < 270:     # bottom -> left
        return _lerp(NAVY, TEAL, (a - 180) / 90.0)
    return _lerp(TEAL, ORANGE, (a - 270) / 90.0)   # left -> top (via mint/teal)


def build() -> None:
    raw = Image.open(ASSETS / "logo_raw.png").convert("RGBA")
    arr = np.asarray(raw).astype(np.int16)
    h, w = arr.shape[:2]

    # cream reference = median of the four corners
    corners = np.array([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]])
    bg = np.median(corners, axis=0)
    cream = np.abs(arr[:, :, :3] - bg).max(axis=2) <= 45   # ribbon + background

    # disc = circle fit from the bbox of the non-cream (coloured) pixels
    ys, xs = np.where(~cream)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 + 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = dist <= r

    fields = disc & (~cream)        # paint only here; everything else transparent

    # angle (0=up, clockwise) and radius for every pixel
    ang = (np.degrees(np.arctan2(xx - cx, -(yy - cy)))) % 360.0
    rad = np.clip(dist / r, 0, 1)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    fy, fx = np.where(fields)
    for y, x in zip(fy.tolist(), fx.tolist()):
        base = _angular_color(ang[y, x])
        # radial depth: lighter near centre, slightly deeper toward the rim
        col = _lerp(_lerp(base, (255, 255, 255), 0.10),
                    _lerp(base, DEEP_NAVY, 0.18), rad[y, x])
        out[y, x, 0], out[y, x, 1], out[y, x, 2], out[y, x, 3] = col[0], col[1], col[2], 255

    img = Image.fromarray(out, "RGBA")
    # feather the field edges so the cut against the transparent ribbon is smooth
    alpha = img.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
    img.putalpha(alpha)

    # trim to the disc bbox (square)
    L, T, R, B = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
    img = img.crop((L, T, R, B))

    # retro/8-bit CRT restyle to match the Zenith theme
    img = _retro(img, grid=56)

    ASSETS.mkdir(exist_ok=True)
    STATIC.mkdir(exist_ok=True)
    img.save(ASSETS / "logo.png")
    # NEAREST on downscale keeps the chunky pixels crisp instead of blurring them
    img.resize((256, 256), Image.NEAREST).save(ASSETS / "logo_256.png")
    img.resize((64, 64), Image.NEAREST).save(ASSETS / "favicon.png")
    img.resize((256, 256), Image.NEAREST).save(STATIC / "logo.png")

    cream_left = int(((out[:, :, 3] > 10) &
                      (out[:, :, 0] > 225) & (out[:, :, 1] > 220) & (out[:, :, 2] > 195)).sum())
    print(f"disc center=({cx:.0f},{cy:.0f}) r={r:.0f}  painted={len(fy):,}px  "
          f"residual_cream={cream_left}px  -> assets/logo*.png, static/logo.png")


if __name__ == "__main__":
    build()

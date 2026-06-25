"""Knock out the cream background of the Zenith logo PNG to transparency.

The outer background AND the inner 'Z' ribbon are both cream, so we cannot
delete cream by color alone. Instead we flood-fill from the image border across
the contiguous cream region — the ribbon is enclosed by colored pixels, so it is
never reached and stays intact. A light feather on the alpha edge removes the
anti-aliased rim halo around the disc.

Usage:  python scripts/make_logo_transparent.py SRC DST
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

TOL = 40  # max per-channel distance from the sampled cream to count as background


def main(src: str, dst: str) -> None:
    im = Image.open(src).convert("RGBA")
    arr = np.asarray(im).astype(np.int16)
    h, w = arr.shape[:2]

    # reference cream = median of the four corners
    corners = np.array([arr[0, 0, :3], arr[0, -1, :3],
                        arr[-1, 0, :3], arr[-1, -1, :3]])
    bg = np.median(corners, axis=0)
    cream = (np.abs(arr[:, :, :3] - bg).max(axis=2) <= TOL)

    # BFS flood fill from every border pixel that is cream
    visited = np.zeros((h, w), dtype=bool)
    dq: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if cream[y, x] and not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if cream[y, x] and not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))

    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and cream[ny, nx]:
                visited[ny, nx] = True
                dq.append((ny, nx))

    out = arr.astype(np.uint8).copy()
    out[visited, 3] = 0  # background -> transparent

    result = Image.fromarray(out, "RGBA")
    # feather the alpha channel slightly to soften the cut edge, then re-apply
    alpha = result.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
    result.putalpha(alpha)
    result.save(dst)

    removed = int(visited.sum())
    print(f"bg cream ~ {tuple(int(c) for c in bg)}  tol={TOL}")
    print(f"removed {removed:,} / {h*w:,} px ({removed/(h*w):.1%}) -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Cap-aware executor for many backtests.

Uses a thread pool: numpy/pandas release the GIL in their heavy kernels, and
threads avoid Windows process-spawn re-importing the Streamlit app. Locally the
pool uses every core; on the cloud it is small and the caps keep jobs short.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from ..config import is_local


def workers() -> int:
    return max(1, (os.cpu_count() or 2)) if is_local() else 2


def pmap(fn: Callable, items: Iterable, progress: Callable[[float], None] | None = None) -> list:
    """Ordered parallel map with an optional 0..1 progress callback."""
    items = list(items)
    if not items:
        return []
    out = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers()) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if progress:
                progress(done / len(items))
    return out

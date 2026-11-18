"""Holdings-overlap math between user-built baskets.

Third-party ETF holdings aren't available for free, so overlap is computed
between funds you build here (and any indices we ship constituents for).
"""

from __future__ import annotations

import pandas as pd


def jaccard(a, b) -> float:
    """Name overlap: |A∩B| / |A∪B| (ignores weights)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def weight_overlap(wa: dict, wb: dict) -> float:
    """Weighted overlap = Σ min(|wa_i|, |wb_i|) over common names (0..1).

    1.0 = identical portfolios; 0.0 = no shared holdings. Uses absolute weights
    normalised to gross 1 so long/short funds compare on exposure.
    """
    def _norm(w):
        s = sum(abs(v) for v in w.values())
        return {k: abs(v) / s for k, v in w.items()} if s > 0 else {}
    na, nb = _norm(wa), _norm(wb)
    return float(sum(min(na.get(k, 0.0), nb.get(k, 0.0)) for k in set(na) | set(nb)))


def overlap_matrix(baskets: dict) -> pd.DataFrame:
    """Pairwise weighted-overlap matrix. ``baskets`` maps name -> {ticker: weight}."""
    names = list(baskets)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            m.loc[i, j] = 1.0 if i == j else weight_overlap(baskets[i], baskets[j])
    return m

"""Walk-forward with re-tuning (upgraded from PARALLAX, which never re-tuned).

Anchored folds inside the in-sample data. In each fold the parameter is chosen
on the training data only — by the *plateau* rule, not the raw peak — then run
forward on the next unseen slice. The stitched out-of-fold returns are what the
tuning procedure would really have earned; compare them with the in-sample
Sharpe of the best variant to see how much of that Sharpe was the tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import metrics
from ..spec import StrategySpec, run_spec
from .sensitivity import plateau, sweep, to_grid


def walk_forward(spec: StrategySpec, ohlcv: pd.DataFrame, param: str, values: list, n_folds: int = 5,
                 first_train: float = 0.4, aux: dict | None = None, progress=None) -> dict:
    n = len(ohlcv)
    edges = np.linspace(int(n * first_train), n, n_folds + 1, dtype=int)
    rows, pieces, hashes = [], [], []
    for k in range(n_folds):
        tr_end, te_end = edges[k], edges[k + 1]
        train = ohlcv.iloc[:tr_end]
        df, hs = sweep(spec, train, param, values, aux=aux)
        hashes += hs
        pl = plateau(to_grid(df, "Sharpe"))
        pick = pl["plateau"]["x"] if pl.get("ok") else spec.params.get(param)
        res = run_spec(spec.with_params(**{param: pick}), ohlcv.iloc[:te_end], aux)
        oos = res.returns.iloc[tr_end:te_end]
        pieces.append(oos)
        rows.append({"fold": k + 1, "train_to": str(train.index[-1].date()),
                     "test": f"{ohlcv.index[tr_end].date()} → {ohlcv.index[te_end - 1].date()}",
                     "pick": pick, "train_sharpe": float(df["Sharpe"].max()),
                     "test_sharpe": metrics.sharpe(oos, res.periods)})
        if progress:
            progress((k + 1) / n_folds)
    stitched = pd.concat(pieces)
    periods = res.periods
    full, _ = sweep(spec, ohlcv, param, values, aux=aux)
    is_best = float(full["Sharpe"].max())
    wf = metrics.sharpe(stitched, periods)
    return {"folds": pd.DataFrame(rows), "oos": stitched, "wf_sharpe": wf, "is_best_sharpe": is_best,
            "efficiency": wf / is_best if is_best > 1e-9 else np.nan, "periods": periods, "hashes": hashes}

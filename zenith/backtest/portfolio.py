"""Combine strategies into one vol-targeted portfolio, Carver-style.

Each strategy is already sized to its own vol target, so combining them with
weights that sum to 1 *under*-shoots the target by the diversification they
add. The instrument diversification multiplier IDM = 1/√(w'Cw) (capped at
2.5) scales the blend back up. Everything is causal: weights and IDM at bar t
use return history up to t−1 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IDM_CAP = 2.5


def _weights(rets: pd.DataFrame, method: str, window: int) -> pd.DataFrame:
    n = rets.shape[1]
    if method == "equal":
        return pd.DataFrame(1.0 / n, index=rets.index, columns=rets.columns)
    vol = rets.rolling(window, min_periods=max(20, window // 4)).std().shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.fillna(1.0 / n)


def _idm(rets: pd.DataFrame, w: pd.DataFrame, window: int) -> pd.Series:
    out = pd.Series(1.0, index=rets.index)
    arr = rets.to_numpy(float)
    W = w.to_numpy(float)
    minp = max(30, window // 4)
    for i in range(minp + 1, len(rets)):
        lo = max(0, i - window)
        c = np.corrcoef(arr[lo:i].T)
        c = np.nan_to_num(c, nan=0.0)
        np.fill_diagonal(c, 1.0)
        c = np.clip(c, 0.0, 1.0)        # Carver floors negative correlations at 0 for the IDM
        v = float(W[i] @ c @ W[i])
        out.iloc[i] = min(IDM_CAP, 1.0 / np.sqrt(v)) if v > 1e-12 else 1.0
    return out


def combine(rets: pd.DataFrame, method: str = "equal", window: int = 252) -> dict:
    rets = rets.dropna(how="any")
    w = _weights(rets, method, window)
    idm = _idm(rets, w, window)
    port = (rets * w).sum(axis=1) * idm
    cov = rets.cov().to_numpy()
    wl = w.iloc[-1].to_numpy()
    tot = float(wl @ cov @ wl)
    rc = (wl * (cov @ wl)) / tot if tot > 0 else np.full(len(wl), np.nan)
    sd = rets.std()
    div_ratio = float((wl * sd.to_numpy()).sum() / np.sqrt(tot)) if tot > 0 else np.nan
    return {"returns": port, "weights": w, "idm": idm, "corr": rets.corr(),
            "risk_contrib": pd.Series(rc, index=rets.columns), "div_ratio": div_ratio, "components": rets}

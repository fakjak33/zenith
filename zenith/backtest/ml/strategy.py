"""An ML model as a LAB strategy (spec.mode == "ml").

The model is re-trained every ``retrain`` bars on data whose targets were
already known (rows <= s − h), and predicts the next block only — so the
forecast at bar t never depends on anything after t. That makes an ML model
a normal saved strategy: LIBRARY, PORTFOLIO, OOS reveal and every robustness
step treat it like a rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..rules.base import Strategy
from .core import MODELS, build_features, make_model, make_target, proba, to_position

DEFAULT = {"features": {"indicators": ["rsi", "tsmom", "rv_yz", "er"], "lags": 3, "memory": 0, "calendar": False,
                        "ref_lags": 0},
           "target": "direction", "h": 1, "k": 1.0, "model": "logistic", "C": 1.0, "retrain": 126,
           "min_train": 750, "window": "expanding", "train_len": 1000, "mapping": "threshold", "hi": 0.55,
           "lo": 0.45, "long_only": False}


def task_of(cfg: dict) -> str:
    return "reg" if cfg.get("target") == "return" else "clf"


class MLStrategy(Strategy):
    key = "_ml"
    label = "ML model"
    params: dict = {}

    def __init__(self, cfg: dict, tf: str = "Daily"):
        self.values = {}
        self.cfg = {**DEFAULT, **(cfg or {})}
        self.tf = tf

    def predictions(self, ohlcv: pd.DataFrame) -> pd.Series:
        c = self.cfg
        task = task_of(c)
        X = build_features(ohlcv, c["features"], self.tf)
        y = make_target(ohlcv, c["target"], int(c["h"]), float(c["k"]))
        h = int(c["h"])
        ok_x = X.notna().all(axis=1).to_numpy()
        ok_y = y.notna().to_numpy()
        Xv, yv = X.to_numpy(), y.to_numpy()
        n = len(X)
        pred = np.full(n, np.nan)
        start = int(c["min_train"]) + int(np.argmax(ok_x)) if ok_x.any() else n
        step = max(5, int(c["retrain"]))
        for s in range(start, n, step):
            hi_tr = s - h                                   # targets of rows < s - h are known at s
            lo_tr = max(0, hi_tr - int(c["train_len"])) if c.get("window") == "rolling" else 0
            tr = np.arange(lo_tr, max(hi_tr, lo_tr))
            tr = tr[ok_x[tr] & ok_y[tr]]
            te = np.arange(s, min(s + step, n))
            te = te[ok_x[te]]
            if len(tr) < 100 or len(te) == 0 or (task == "clf" and len(np.unique(yv[tr])) < 2):
                continue
            m = make_model(c["model"], task, C=float(c["C"])).fit(Xv[tr], yv[tr])
            pred[te] = proba(m, Xv[te]) if task == "clf" else m.predict(Xv[te])
        return pd.Series(pred, index=X.index)

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        c = self.cfg
        pos = to_position(self.predictions(ohlcv), task_of(c), c["mapping"], float(c["hi"]), float(c["lo"]),
                          long_only=bool(c.get("long_only")))
        return (pos * 10.0).reindex(ohlcv.index).fillna(0.0)

    def raw_forecast(self, ohlcv):
        return self.forecast(ohlcv)

    def describe(self) -> str:
        c = self.cfg
        f = c["features"]
        return (f"ML {MODELS[c['model']].label.split(' (')[0]} · {c['target']} h={c['h']} · "
                f"{len(f.get('indicators', []))} ind + {f.get('lags', 0)} lags · retrain {c['retrain']}")


def needs(cfg: dict) -> set:
    from ..indicators import CATALOG
    f = {**DEFAULT, **(cfg or {})}["features"]
    out = set()
    for k in f.get("indicators", []):
        out |= set(CATALOG[k].needs)
    if int(f.get("ref_lags", 0)):
        out.add("ref_close")
    return out

"""ML LAB engine: features, targets, time-ordered validation and models.

Design rules (MemLabs + SharpeEdge + López de Prado):
  - predict *returns / direction*, never price levels;
  - features at bar t use bars <= t only; the target looks h bars ahead, so a
    training sample is only usable once its target is known (t + h <= now) —
    every split purges those h bars (and an optional embargo);
  - time order is never shuffled; the score that matters is the trading one
    (expected value per bar of acting on the prediction, net of costs), not
    MSE or accuracy on their own;
  - always compare with a naive baseline (always predict the majority class /
    the training mean).

scikit-learn and statsmodels are imported lazily so the rest of Zenith never
loads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import scale_params_for_tf
from ..indicators import CATALOG


# =================================================================== FEATURES
def build_features(df: pd.DataFrame, cfg: dict, tf: str = "Daily") -> pd.DataFrame:
    """cfg keys: indicators [keys], lags int, memory int (0=off), calendar bool, ref_lags int.
    Every column is known at the close of bar t."""
    lr = np.log(df["close"]).diff()
    cols: dict[str, pd.Series] = {}
    for k in cfg.get("indicators", []):
        ind = CATALOG[k]
        p = scale_params_for_tf(ind.defaults(), tf)
        v = ind(df, **p)
        if ind.overlay:                                   # price-level lines → distance from price
            v = df["close"] / v - 1
        cols[k] = v
    for i in range(1, int(cfg.get("lags", 3)) + 1):
        cols[f"ret_lag{i}"] = lr.shift(i - 1)
    if "ref_close" in df and int(cfg.get("ref_lags", 0)):
        rr = np.log(df["ref_close"]).diff()
        for i in range(1, int(cfg["ref_lags"]) + 1):
            cols[f"ref_lag{i}"] = rr.shift(i - 1)
        cols["spread_lag1"] = (lr - rr)
    m = int(cfg.get("memory", 0))
    if m:
        cols[f"memory_{m}"] = lr.rolling(m, min_periods=m).mean()          # MemLabs regime 'memory'
        cols[f"ret_vs_memory_{m}"] = lr - cols[f"memory_{m}"]
    if cfg.get("calendar"):
        for d in range(5):
            cols[f"dow_{d}"] = pd.Series((df.index.dayofweek == d).astype(float), index=df.index)
    X = pd.DataFrame(cols, index=df.index).replace([np.inf, -np.inf], np.nan)
    return X


# ==================================================================== TARGETS
TARGETS = {
    "direction": "Next h-bar direction (up / down) — classification",
    "return": "Next h-bar log return — regression",
    "triple_barrier": "Which is hit first: +k·σ, −k·σ or the time limit — classification (López de Prado)",
}


def fwd_return(df: pd.DataFrame, h: int = 1) -> pd.Series:
    c = df["close"]
    return np.log(c.shift(-h) / c)


def make_target(df: pd.DataFrame, kind: str, h: int = 1, k: float = 1.0, vol_n: int = 20) -> pd.Series:
    fr = fwd_return(df, h)
    if kind == "return":
        return fr
    if kind == "direction":
        return (fr > 0).astype(float).where(fr.notna())
    if kind == "triple_barrier":
        lr = np.log(df["close"]).diff()
        sig = lr.rolling(vol_n, min_periods=vol_n).std() * np.sqrt(h)
        c = np.log(df["close"].to_numpy())
        s = sig.to_numpy()
        n = len(c)
        out = np.full(n, np.nan)
        for t in range(n - h):
            if not np.isfinite(s[t]) or s[t] <= 0:
                continue
            path = c[t + 1: t + h + 1] - c[t]
            up = np.where(path >= k * s[t])[0]
            dn = np.where(path <= -k * s[t])[0]
            fu, fd = (up[0] if len(up) else h + 1), (dn[0] if len(dn) else h + 1)
            if fu == fd == h + 1:
                out[t] = 1.0 if path[-1] > 0 else 0.0           # time barrier: sign of the move
            else:
                out[t] = 1.0 if fu < fd else 0.0
        return pd.Series(out, index=df.index)
    raise ValueError(kind)


# ===================================================================== SPLITS
CV_SCHEMES = {
    "expanding": "Expanding window: train on everything so far, test on the next block (walk-forward).",
    "rolling": "Rolling window: train on the last N bars only, test on the next block — adapts to regime change.",
    "split": "One chronological split: first part trains, the rest tests (MemLabs' 70/30).",
    "purged_kfold": "Purged k-fold: every block is tested once; training skips h bars (+ embargo) around it.",
}


def splits(n: int, scheme: str = "expanding", n_folds: int = 5, min_train: int = 500, train_len: int = 750,
           frac: float = 0.7, purge: int = 1, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered (train, test) index pairs. ``purge`` = bars whose target overlaps the test start."""
    idx = np.arange(n)
    out = []
    if scheme == "split":
        cut = int(n * frac)
        return [(idx[: max(cut - purge, 0)], idx[cut:])]
    if scheme == "purged_kfold":
        for b in np.array_split(idx, n_folds):
            lo, hi = b[0], b[-1]
            mask = (idx < lo - purge) | (idx > hi + embargo)
            out.append((idx[mask], b))
        return out
    start = min(min_train, n // 2)
    blocks = np.array_split(idx[start:], n_folds)
    for b in blocks:
        if len(b) == 0:
            continue
        end = b[0] - purge
        lo = max(0, end - train_len) if scheme == "rolling" else 0
        out.append((idx[lo:max(end, lo)], b))
    return out


# ===================================================================== MODELS
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    task: str          # "clf" | "reg" | "both"
    help: str


MODELS = {
    "logistic": ModelSpec("logistic", "Logistic regression (L2)", "clf",
                          "A straight-line score turned into a probability. Simple, hard to overfit, and its "
                          "weights are readable. SharpeEdge's workhorse."),
    "logistic_l1": ModelSpec("logistic_l1", "Logistic regression (L1 / lasso)", "clf",
                             "Like logistic, but pushes useless features' weights to exactly zero — automatic "
                             "feature selection."),
    "linear_svm": ModelSpec("linear_svm", "Linear SVM (L1)", "clf",
                            "Finds the straight boundary with the widest margin between up and down bars. "
                            "SharpeEdge's meta-labeling model."),
    "random_forest": ModelSpec("random_forest", "Random forest", "both",
                               "Many shallow decision trees voting. Captures interactions and thresholds; easy "
                               "to overfit on noisy returns — watch the CV gap."),
    "gbm": ModelSpec("gbm", "Gradient boosting (HistGB)", "both",
                     "Trees built one after another, each fixing the last one's errors — the XGBoost family "
                     "(MemLabs). Powerful and the easiest of these to overfit."),
    "linear": ModelSpec("linear", "Linear regression (OLS)", "reg",
                        "The classic straight line. On returns the sign of each weight tells you momentum (+) "
                        "or mean reversion (−) — MemLabs' AR models."),
    "ridge": ModelSpec("ridge", "Ridge regression", "reg", "OLS with weights shrunk toward zero: steadier."),
    "lasso": ModelSpec("lasso", "Lasso regression", "reg", "OLS that zeroes out weak features."),
    "stack": ModelSpec("stack", "Stack (logistic + forest + boosting → logistic)", "clf",
                       "MemLabs' model stacking: three different models predict out-of-fold, a simple model "
                       "learns how much to trust each (Wolpert 1992)."),
}


def make_model(key: str, task: str, C: float = 1.0, seed: int = 0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if key == "logistic":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
    if key == "logistic_l1":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=500))
    if key == "linear_svm":
        from sklearn.svm import LinearSVC
        return make_pipeline(StandardScaler(), LinearSVC(C=C * 0.1, penalty="l1", dual=False, max_iter=5000))
    if key == "random_forest":
        if task == "clf":
            from sklearn.ensemble import RandomForestClassifier as RF
        else:
            from sklearn.ensemble import RandomForestRegressor as RF
        return RF(n_estimators=150, max_depth=4, min_samples_leaf=50, n_jobs=-1, random_state=seed)
    if key == "gbm":
        if task == "clf":
            from sklearn.ensemble import HistGradientBoostingClassifier as GB
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor as GB
        return GB(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50, random_state=seed)
    if key == "linear":
        from sklearn.linear_model import LinearRegression
        return make_pipeline(StandardScaler(), LinearRegression())
    if key == "ridge":
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0 / max(C, 1e-6)))
    if key == "lasso":
        from sklearn.linear_model import Lasso
        return make_pipeline(StandardScaler(), Lasso(alpha=1e-4 / max(C, 1e-6), max_iter=5000))
    if key == "stack":
        return TimeStack(seed=seed)
    raise KeyError(key)


class TimeStack:
    """Stacking without shuffling: base models' out-of-fold probabilities come from expanding splits
    inside the training window; the meta logistic regression learns on those."""

    def __init__(self, seed: int = 0, inner: int = 4):
        self.seed, self.inner = seed, inner
        self.bases = ["logistic", "random_forest", "gbm"]

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        n = len(X)
        oof = np.full((n, len(self.bases)), np.nan)
        for tr, te in splits(n, "expanding", n_folds=self.inner, min_train=max(100, n // 3)):
            if len(tr) < 50 or len(np.unique(y[tr])) < 2:
                continue
            for j, b in enumerate(self.bases):
                m = make_model(b, "clf", seed=self.seed).fit(X[tr], y[tr])
                oof[te, j] = m.predict_proba(X[te])[:, 1]
        ok = np.isfinite(oof).all(axis=1)
        self.meta = LogisticRegression(max_iter=500).fit(oof[ok], y[ok]) if ok.sum() > 30 else None
        self.models = [make_model(b, "clf", seed=self.seed).fit(X, y) for b in self.bases]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X):
        P = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        p = self.meta.predict_proba(P)[:, 1] if self.meta is not None else P.mean(axis=1)
        return np.column_stack([1 - p, p])

    def weights(self) -> dict:
        if self.meta is None:
            return {}
        return dict(zip(self.bases, self.meta.coef_[0]))


def proba(model, X) -> np.ndarray:
    """P(class 1) for any classifier (SVMs: logistic squash of the margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1 / (1 + np.exp(-2 * d))


# =========================================================== WALK-FORWARD RUN
def walk_forward(X: pd.DataFrame, y: pd.Series, model_key: str, task: str, scheme: str = "expanding",
                 n_folds: int = 6, h: int = 1, embargo: int = 0, min_train: int = 500, train_len: int = 750,
                 frac: float = 0.7, C: float = 1.0) -> dict:
    """Out-of-sample predictions on every test bar (P(up) for clf, predicted return for reg)."""
    data = pd.concat([X, y.rename("_y")], axis=1).dropna()
    Xv, yv = data.drop(columns="_y").to_numpy(), data["_y"].to_numpy()
    idx = data.index
    pred = pd.Series(np.nan, index=idx)
    folds = []
    model = None
    for tr, te in splits(len(data), scheme, n_folds, min_train, train_len, frac, purge=h, embargo=embargo):
        if len(tr) < 100 or len(te) == 0 or (task == "clf" and len(np.unique(yv[tr])) < 2):
            continue
        model = make_model(model_key, task, C=C).fit(Xv[tr], yv[tr])
        p = proba(model, Xv[te]) if task == "clf" else model.predict(Xv[te])
        pred.iloc[te] = p
        base = (yv[tr].mean() > 0.5) if task == "clf" else yv[tr].mean()
        folds.append({"train_from": idx[tr[0]], "train_to": idx[tr[-1]], "test_from": idx[te[0]],
                      "test_to": idx[te[-1]], "n_train": len(tr), "n_test": len(te), "base": float(base),
                      "te": te})
    return {"pred": pred, "folds": folds, "model": model, "X": data.drop(columns="_y"), "y": data["_y"]}


# =================================================================== MAPPING
MAPPINGS = {
    "threshold": "Long above the upper probability, short below the lower, flat in between.",
    "sign": "Always in: long if the prediction says up, short if down.",
    "tanh": "Smooth: size grows with confidence, capped (MemLabs' tanh sizing).",
    "hardtanh": "Linear in confidence, clipped at full size (hard-tanh).",
}


def to_position(pred: pd.Series, task: str, mapping: str = "threshold", hi: float = 0.55, lo: float = 0.45,
                gain: float = 10.0, long_only: bool = False) -> pd.Series:
    """Prediction → position in [-1, 1]."""
    if task == "clf":
        z = pred - 0.5
    else:
        sd = pred.expanding(min_periods=50).std().shift(1)
        z = (pred / sd.where(sd > 0)).fillna(0.0) * 0.1
    if mapping == "threshold":
        if task == "clf":
            pos = pd.Series(np.where(pred >= hi, 1.0, np.where(pred <= lo, -1.0, 0.0)), index=pred.index)
        else:
            pos = pd.Series(np.where(z >= (hi - 0.5), 1.0, np.where(z <= (lo - 0.5), -1.0, 0.0)), index=pred.index)
    elif mapping == "sign":
        pos = np.sign(z)
    elif mapping == "tanh":
        pos = np.tanh(gain * z)
    else:
        pos = (gain * z).clip(-1, 1)
    pos = pos.where(pred.notna(), 0.0).fillna(0.0)
    return pos.clip(lower=0.0) if long_only else pos


# ================================================================ SCORECARD
def scorecard(pred: pd.Series, y: pd.Series, fwd: pd.Series, pos: pd.Series, task: str, folds: list,
              cost_bps: float = 2.5, periods: int = 252) -> dict:
    ok = pred.notna()
    p, yy, fr = pred[ok], y.reindex(pred.index)[ok], fwd.reindex(pred.index)[ok]
    out = {"n": int(ok.sum())}
    if task == "clf":
        hit = ((p >= 0.5) == (yy >= 0.5)).mean()
        naive = []
        for f in folds:
            te = pred.index[f["te"]]
            yt = y.reindex(te).dropna()
            naive.append(((yt >= 0.5) == f["base"]).sum())
        out["accuracy"] = float(hit)
        out["naive"] = float(np.sum(naive) / max(ok.sum(), 1))
    else:
        out["accuracy"] = float((np.sign(p) == np.sign(fr)).mean())
        out["naive"] = float((np.sign(fr) == np.sign(np.mean([f["base"] for f in folds]) or 1)).mean())
        out["oos_r2"] = float(1 - ((fr - p) ** 2).sum() / ((fr - fr.mean()) ** 2).sum())
    pos_t = pos.reindex(pred.index).fillna(0.0)
    gross = pos_t * fwd.reindex(pred.index).fillna(0.0)
    cost = pos_t.diff().abs().fillna(pos_t.abs()) * cost_bps / 1e4
    net = (gross - cost)[ok]
    out["ev_bp"] = float(net[pos_t[ok] != 0].mean() * 1e4) if (pos_t[ok] != 0).any() else 0.0
    sd = net.std()
    out["sharpe"] = float(net.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    out["exposure"] = float((pos_t[ok] != 0).mean())
    out["net"] = net
    return out


def calibration(pred: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"p": pred, "y": y.reindex(pred.index)}).dropna()
    if len(d) < bins * 5:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True)
    return pd.DataFrame({"predicted": g["p"].mean(), "actual": g["y"].mean(), "n": g.size()}).reset_index(drop=True)


def coefficients(model, cols: list[str]) -> pd.Series | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        return pd.Series(np.ravel(est.coef_), index=cols)
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=cols)
    return None


def hac_table(X: pd.DataFrame, y: pd.Series, task: str, lags: int = 5) -> pd.DataFrame | None:
    """statsmodels OLS / Logit on standardised features with Newey-West (HAC) errors."""
    try:
        import statsmodels.api as sm
        Z = (X - X.mean()) / X.std().replace(0, np.nan)
        Z = sm.add_constant(Z.dropna(axis=1, how="any").fillna(0.0))
        if task == "clf":
            res = sm.Logit(y, Z).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": lags})
        else:
            res = sm.OLS(y, Z).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return pd.DataFrame({"weight": res.params, "std err (HAC)": res.bse, "t": res.tvalues, "p": res.pvalues})
    except Exception:  # noqa: BLE001
        return None


def perm_importance(model, X: pd.DataFrame, y: pd.Series, task: str, seed: int = 0) -> pd.Series | None:
    try:
        from sklearn.inspection import permutation_importance
        score = "roc_auc" if task == "clf" else "r2"
        r = permutation_importance(model, X.to_numpy(), y.to_numpy(), n_repeats=5, random_state=seed, scoring=score)
        return pd.Series(r.importances_mean, index=X.columns).sort_values()
    except Exception:  # noqa: BLE001
        return None

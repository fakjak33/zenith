"""ML LAB extras: online learning, meta-labeling and decomposed returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .core import make_model, proba, splits


# ================================================================ ONLINE
def online_weights(df: pd.DataFrame, lags: int = 3, memory: int = 40, lr: float = 0.01) -> pd.DataFrame:
    """MemLabs' regime-change fix: an SGD regressor updated one bar at a time (partial_fit) on lagged
    returns. Returns its weights after each bar — a positive lag-1 weight means momentum, negative means
    mean reversion — plus its one-step-ahead prediction (made before seeing the bar)."""
    from sklearn.linear_model import SGDRegressor
    r = np.log(df["close"]).diff()
    sd = r.rolling(max(memory, 20), min_periods=20).std().shift(1)
    z = (r / sd).clip(-5, 5)
    X = pd.concat({f"lag{i}": z.shift(i) for i in range(1, lags + 1)}, axis=1)
    if memory:
        X[f"memory{memory}"] = z.shift(1).rolling(memory, min_periods=memory).mean()
    data = pd.concat([X, z.rename("y")], axis=1).dropna()
    m = SGDRegressor(learning_rate="constant", eta0=lr, penalty="l2", alpha=1e-4, random_state=0)
    W, P = [], []
    Xv, yv = data.drop(columns="y").to_numpy(), data["y"].to_numpy()
    first = True
    for i in range(len(data)):
        x = Xv[i: i + 1]
        P.append(float(m.predict(x)[0]) if not first else 0.0)
        m.partial_fit(x, yv[i: i + 1])
        first = False
        W.append(m.coef_.copy())
    out = pd.DataFrame(W, index=data.index, columns=data.columns.drop("y"))
    out["prediction"] = P
    return out


# ============================================================ META-LABEL
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    if min(n1, n2) == 0:
        return (np.nan, np.nan)
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (np.nan, np.nan)
    z = (k1 / n1 - k2 / n2) / se
    return float(z), float(2 * (1 - sps.norm.cdf(abs(z))))


def trades_table(position: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """One row per trade of the base strategy: entry bar, exit bar, side, compound return."""
    pos = np.sign(position.reindex(returns.index).fillna(0.0))
    held = pos.shift(1).fillna(0.0)
    grp = (held != held.shift()).cumsum()
    rows = []
    for _, g in returns.groupby(grp):
        side = held.loc[g.index[0]]
        if side == 0:
            continue
        entry = returns.index[returns.index.get_loc(g.index[0]) - 1]     # decided at the previous close
        rows.append({"entry": entry, "first": g.index[0], "exit": g.index[-1], "side": side,
                     "ret": float(np.prod(1 + g.to_numpy()) - 1)})
    return pd.DataFrame(rows)


def meta_label(trades: pd.DataFrame, X: pd.DataFrame, model_key: str = "logistic", threshold: float = 0.5,
               n_folds: int = 5, min_train: int = 60) -> dict:
    """SharpeEdge's meta-labeling: a classifier predicts whether each base-strategy trade will win, using
    features known at entry; trades are taken only when P(win) >= threshold. Walk-forward over trades, and a
    trade only joins the training set once it has closed before the test trade opens (purging)."""
    if trades.empty:
        return {}
    t = trades.copy()
    t["win"] = (t["ret"] > 0).astype(float)
    F = X.reindex(t["entry"]).to_numpy()
    F = np.column_stack([F, t["side"].to_numpy()])
    ok = np.isfinite(F).all(axis=1)
    t, F = t[ok].reset_index(drop=True), F[ok]
    p = np.full(len(t), np.nan)
    for tr, te in splits(len(t), "expanding", n_folds=n_folds, min_train=min_train, purge=0):
        first_entry = t["entry"].iloc[te[0]]
        tr = tr[t["exit"].iloc[tr].to_numpy() < first_entry]             # closed before the test trade opened
        if len(tr) < 30 or len(np.unique(t["win"].iloc[tr])) < 2:
            continue
        m = make_model(model_key, "clf").fit(F[tr], t["win"].iloc[tr].to_numpy())
        p[te] = proba(m, F[te])
    t["p_win"] = p
    tested = t[np.isfinite(p)]
    take = tested[tested["p_win"] >= threshold]
    k_all, n_all = int(tested["win"].sum()), len(tested)
    k_f, n_f = int(take["win"].sum()), len(take)
    z, pv = two_prop_z(k_f, n_f, k_all - k_f, n_all - n_f) if n_f < n_all else (np.nan, np.nan)
    return {"trades": t, "tested": tested, "taken": take, "win_all": k_all / max(n_all, 1),
            "win_taken": k_f / max(n_f, 1), "ci_all": wilson(k_all, n_all), "ci_taken": wilson(k_f, n_f),
            "z": z, "p": pv, "n_all": n_all, "n_taken": n_f}


def filtered_position(position: pd.Series, trades: pd.DataFrame) -> pd.Series:
    """Base position with skipped trades removed."""
    keep = position.copy() * 0.0
    for _, r in trades.iterrows():
        if r.get("take", False):
            seg = (position.index >= r["entry"]) & (position.index < r["exit"])
            keep[seg] = position[seg]
    return keep


# =========================================================== DECOMPOSE
def decompose(X: pd.DataFrame, fwd: pd.Series, n_folds: int = 6, min_train: int = 500, h: int = 1) -> dict:
    """MemLabs' decomposed returns: E[r] = P(up)·E[|r| | up] − (1 − P(up))·E[|r| | down], each part learnt
    separately (logistic for the sign, ridge regressions for the two magnitudes), vs one direct regression."""
    data = pd.concat([X, fwd.rename("_r")], axis=1).dropna()
    Xv, r = data.drop(columns="_r").to_numpy(), data["_r"].to_numpy()
    e_dec, e_dir = np.full(len(r), np.nan), np.full(len(r), np.nan)
    pu = np.full(len(r), np.nan)
    for tr, te in splits(len(r), "expanding", n_folds=n_folds, min_train=min_train, purge=h):
        if len(tr) < 200:
            continue
        up = r[tr] > 0
        clf = make_model("logistic", "clf").fit(Xv[tr], up.astype(float))
        mu = make_model("ridge", "reg").fit(Xv[tr][up], np.abs(r[tr][up]))
        md = make_model("ridge", "reg").fit(Xv[tr][~up], np.abs(r[tr][~up]))
        p = proba(clf, Xv[te])
        pu[te] = p
        e_dec[te] = p * np.clip(mu.predict(Xv[te]), 0, None) - (1 - p) * np.clip(md.predict(Xv[te]), 0, None)
        e_dir[te] = make_model("ridge", "reg").fit(Xv[tr], r[tr]).predict(Xv[te])
    ok = np.isfinite(e_dec)
    rmse = lambda e: float(np.sqrt(np.mean((r[ok] - e[ok]) ** 2)))  # noqa: E731
    ev = lambda e: float(np.mean(np.sign(e[ok]) * r[ok]) * 1e4)      # noqa: E731
    # independence: are up-move sizes different from down-move sizes beyond chance?
    rng = np.random.default_rng(0)
    a = np.abs(r)
    obs = a[r > 0].mean() - a[r < 0].mean()
    sims = np.array([(lambda s: a[s > 0].mean() - a[s < 0].mean())(rng.permutation(np.sign(r))) for _ in range(500)])
    naive = float(np.mean(r > 0) * a[r > 0].mean() - np.mean(r < 0) * a[r < 0].mean())
    return {"index": data.index, "p_up": pu, "e_dec": e_dec, "e_dir": e_dir, "rmse_dec": rmse(e_dec),
            "rmse_dir": rmse(e_dir), "rmse_zero": float(np.sqrt(np.mean(r[ok] ** 2))),
            "ev_dec_bp": ev(e_dec), "ev_dir_bp": ev(e_dir), "size_gap": float(obs),
            "size_gap_p": float((np.abs(sims) >= abs(obs)).mean()), "drift": float(r.mean()),
            "naive_drift": naive, "up_size": float(a[r > 0].mean()), "down_size": float(a[r < 0].mean())}

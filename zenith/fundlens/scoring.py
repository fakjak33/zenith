"""Composite scoring.

  1. metric column  -> 0-100 percentile among the comparison universe (direction-aware)
  2. horizon blend  -> one score per metric (weights renormalised over available horizons)
  3. pillar score   -> weighted mean of metric scores
  4. raw composite  -> weighted mean of pillar scores
  5. adjusted       -> 50 + (raw - 50) * coverage^k   (shrinks thin evidence toward the median)
  6. rank           -> adjusted desc, raw desc, net expense asc, ticker asc  (fully deterministic)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dictionary import SCORING_HORIZONS, Dictionary
from .quality import QualityLog


# --------------------------------------------------------------------------- directions
def effective_direction(metric_id: str, d: Dictionary, prof: dict) -> tuple[str, float | None]:
    """('higher'|'lower'|'target'|'none', target)."""
    ov = (prof.get("overrides") or {}).get(metric_id)
    if ov:
        return ov.get("direction", "none"), ov.get("target")
    m = d.by_id.get(metric_id)
    return (m.direction if m else "none"), None


def pct_rank(values: pd.Series, direction: str, target: float | None = None) -> pd.Series:
    """Percentile 0-100 where 100 = best. Ties share the average rank. NaN stays NaN."""
    x = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    v = x.dropna()
    if v.empty or direction == "none":
        return out
    if direction == "target":
        v = -(v - float(target or 0.0)).abs()
    elif direction == "lower":
        v = -v
    n = len(v)
    if n == 1:
        out[v.index] = 50.0
        return out
    r = v.rank(method="average")
    out[v.index] = (r - 1.0) / (n - 1.0) * 100.0
    return out


# --------------------------------------------------------------------------- results
@dataclass
class MetricStatus:
    metric: str
    pillar: str
    weight: float                    # within-pillar weight
    status: str                      # scored | skipped
    reason: str = ""
    columns: list[str] = field(default_factory=list)
    effective_weight: float = 0.0    # share of composite (0-1), for the Metric Guide


@dataclass
class ScoreResult:
    column_scores: pd.DataFrame      # percentile per canonical column (sharpe_3y, ...)
    metric_scores: pd.DataFrame      # blended per metric (sharpe, ...)
    metric_avail: pd.DataFrame       # 0-1 horizon-weight availability per metric
    pillar_scores: pd.DataFrame
    pillar_cov: pd.DataFrame         # 0-1 availability within each pillar
    raw: pd.Series
    coverage: pd.Series
    adjusted: pd.Series
    rank: pd.Series                  # 1..n (NaN = insufficient data)
    tier: pd.Series
    order: list[str]                 # tickers in rank order (unranked last)
    statuses: list[MetricStatus]
    pillar_weights: dict[str, float] # normalised over pillars with >=1 scored metric
    pillar_labels: dict[str, str]
    optional: frozenset = frozenset()  # pillars that do not reduce coverage when a fund has no data (e.g. Medalist)


def _metric_columns(metric_id: str, d: Dictionary, df: pd.DataFrame, prof: dict) -> list[str]:
    m = d.by_id.get(metric_id)
    if m is not None and m.horizons:
        return [f"{metric_id}_{h}" for h in SCORING_HORIZONS
                if prof["horizon_weights"].get(h, 0) > 0 and f"{metric_id}_{h}" in df]
    return [metric_id] if metric_id in df else []


def benchmark_relative_ok(df: pd.DataFrame, prof: dict, log: QualityLog) -> bool:
    if prof.get("relative_metrics_basis", "standard_index") != "prospectus":
        return True
    b = df.get("prospectus_benchmark")
    if b is None or b.dropna().empty:
        return True
    share = b.value_counts(normalize=True).iloc[0]
    if share >= 0.8:
        return True
    log.add("warning", "Scoring",
            f"Benchmark-relative metrics (alpha, IR, captures, beta, R2) not scored: only {share:.0%} of funds share "
            f"the most common prospectus benchmark (need 80%)")
    return False


def score(df: pd.DataFrame, prof: dict, d: Dictionary, log: QualityLog) -> ScoreResult:
    idx = df.index
    sc = prof["scoring"]
    hw = prof["horizon_weights"]
    min_n = int(sc["min_funds_per_metric"])
    rel_ok = benchmark_relative_ok(df, prof, log)

    col_scores: dict[str, pd.Series] = {}
    metric_scores: dict[str, pd.Series] = {}
    metric_avail: dict[str, pd.Series] = {}
    statuses: list[MetricStatus] = []
    pillar_scores: dict[str, pd.Series] = {}
    pillar_cov: dict[str, pd.Series] = {}
    active_pillars: dict[str, float] = {}

    for pid, p in prof["pillars"].items():
        num = pd.Series(0.0, index=idx)
        den = pd.Series(0.0, index=idx)
        cov_num = pd.Series(0.0, index=idx)
        cov_den = 0.0
        for mid, mw in p["metrics"].items():
            st = MetricStatus(mid, pid, mw, "skipped")
            statuses.append(st)
            if mw <= 0:
                st.reason = "weight 0"
                continue
            if p["weight"] <= 0:
                st.reason = "pillar weight 0"
                continue
            meta = d.by_id.get(mid)
            if meta is None:
                st.reason = "unknown metric id"
                log.add("error", "Profile", f"Profile references unknown metric '{mid}'")
                continue
            if meta.benchmark_relative and not rel_ok:
                st.reason = "benchmarks differ across funds"
                continue
            direction, target = effective_direction(mid, d, prof)
            if direction == "none":
                st.reason = "no direction for this profile"
                continue
            cols = []
            for c in _metric_columns(mid, d, df, prof):
                n = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
                if n >= min_n:
                    cols.append(c)
                    col_scores[c] = pct_rank(df[c], direction, target)
            if not cols:
                st.reason = f"fewer than {min_n} funds have data"
                continue
            st.status, st.columns = "scored", cols
            # horizon blend
            if meta.horizons:
                w = {c: hw[c.rsplit("_", 1)[1]] for c in cols}
            else:
                w = {cols[0]: 1.0}
            wsum = sum(w.values())
            s_num = pd.Series(0.0, index=idx)
            s_den = pd.Series(0.0, index=idx)
            for c, wc in w.items():
                s = col_scores[c]
                has = s.notna()
                s_num += s.fillna(0.0) * wc
                s_den += has.astype(float) * wc
            ms = (s_num / s_den).where(s_den > 0)
            avail = s_den / wsum
            metric_scores[mid], metric_avail[mid] = ms, avail
            num += ms.fillna(0.0) * mw
            den += ms.notna().astype(float) * mw
            cov_num += avail * mw
            cov_den += mw
        if cov_den > 0:
            pillar_scores[pid] = (num / den).where(den > 0)
            pillar_cov[pid] = cov_num / cov_den
            active_pillars[pid] = p["weight"]
        else:
            log.add("info", "Scoring", f"Pillar '{p['label']}' has no scorable metrics in this data set; ignored")

    tot = sum(active_pillars.values()) or 1.0
    pw = {k: v / tot for k, v in active_pillars.items()}
    ps = pd.DataFrame(pillar_scores, index=idx)
    pc = pd.DataFrame(pillar_cov, index=idx)
    optional = frozenset(k for k, p in prof["pillars"].items() if p.get("optional"))
    raw, cov, adj = composite(ps, pc, pw, float(sc["shrinkage_k"]), optional)

    # effective weights for the Metric Guide
    for st in statuses:
        if st.status == "scored":
            pmetrics = [s for s in statuses if s.pillar == st.pillar and s.status == "scored"]
            st.effective_weight = pw.get(st.pillar, 0) * st.weight / sum(s.weight for s in pmetrics)

    order, rank = rank_order(df, adj, raw)
    n = int(rank.notna().sum())
    low_cov = float(sc.get("low_evidence_coverage", 0.25))
    tier = pd.Series({t: ("n/a" if pd.isna(rank[t]) or not n else
                          "Low evidence" if cov[t] < low_cov else f"Tier {math.ceil(5 * rank[t] / n)}")
                      for t in idx})
    n_low = int((tier == "Low evidence").sum())
    if n_low:
        log.add("info", "Scoring", f"{n_low} fund(s) have under {low_cov:.0%} coverage (mostly very short histories); "
                                   "tier shown as 'Low evidence' - their composite sits near the median by construction")
    unranked = [t for t in idx if pd.isna(rank[t])]
    if unranked:
        log.add("warning", "Scoring", f"{len(unranked)} fund(s) have no scorable data (e.g. only sector or portfolio "
                                      f"data was captured) and are listed unranked: {', '.join(unranked)}")

    return ScoreResult(
        column_scores=pd.DataFrame(col_scores, index=idx),
        metric_scores=pd.DataFrame(metric_scores, index=idx),
        metric_avail=pd.DataFrame(metric_avail, index=idx),
        pillar_scores=ps, pillar_cov=pc, raw=raw, coverage=cov, adjusted=adj,
        rank=rank, tier=tier, order=order, statuses=statuses, pillar_weights=pw,
        pillar_labels={k: p["label"] for k, p in prof["pillars"].items()},
        optional=optional,
    )


def composite(ps: pd.DataFrame, pc: pd.DataFrame, pw: dict[str, float], k: float,
              optional: frozenset = frozenset()):
    """Optional pillars count toward coverage only for funds that have data in them (e.g. Medalist
    people/process inputs do not exist for index funds - that is not missing evidence)."""
    num = pd.Series(0.0, index=ps.index)
    den = pd.Series(0.0, index=ps.index)
    cov = pd.Series(0.0, index=ps.index)
    wt = pd.Series(0.0, index=ps.index)
    for pid, w in pw.items():
        if w <= 0 or pid not in ps:
            continue
        s = ps[pid]
        c = pc[pid].fillna(0.0)
        num += s.fillna(0.0) * w
        den += s.notna().astype(float) * w
        cov += c * w
        wt += (c > 0).astype(float) * w if pid in optional else w
    raw = (num / den).where(den > 0)
    cov = (cov / wt.where(wt > 0)).fillna(0.0).clip(0, 1)
    adj = 50.0 + (raw - 50.0) * cov.pow(k)
    return raw, cov, adj


def rank_order(df: pd.DataFrame, adj: pd.Series, raw: pd.Series) -> tuple[list[str], pd.Series]:
    er = pd.to_numeric(df.get("net_er", pd.Series(np.nan, index=df.index)), errors="coerce")
    keys = []
    for t in df.index:
        a, r = adj[t], raw[t]
        ranked = pd.notna(a)
        keys.append((0 if ranked else 1,
                     -round(float(a), 9) if ranked else 0.0,
                     -round(float(r), 9) if ranked else 0.0,
                     float(er[t]) if pd.notna(er[t]) else math.inf,
                     str(t)))
    order = [k[-1] for k in sorted(keys)]
    rank = pd.Series(np.nan, index=df.index, dtype=float)
    i = 0
    for t in order:
        if pd.notna(adj[t]):
            i += 1
            rank[t] = i
    return order, rank

"""The THETA RUN coach: OPTIONS VAULT turned into in-game nudges.

Every nudge quotes (paraphrased, as stored) a vault field and carries its
reference, so the game teaches the same evidence-graded material the vault
shows: the tastylive rule, and -- where our tests disagree -- the caveat.

  ref formats   card:<slug>.<field>  tree:<tree>.<node>  edge:<slug>  alloc
Tests resolve every reference against data/vault/*.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..vault import load as vload
from .engine.clock import BP_GUIDE
from .engine.strategies import TEMPLATES

_REF = re.compile(r"\s*\[[^\]]+\]")
SEV_ORDER = {"alert": 0, "warn": 1, "info": 2, "good": 3}
FOLLOW_DAYS = 3                       # a management call counts as followed within 3 trading days


@dataclass
class Nudge:
    severity: str        # alert | warn | info | good
    title: str
    text: str
    ref: str
    sources: list[str]

    def to_dict(self) -> dict:
        return {"severity": self.severity, "title": self.title, "text": self.text,
                "ref": self.ref, "sources": self.sources}


def strip_refs(s: str) -> str:
    return " ".join(_REF.sub("", s or "").split())


def refs_in(s: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", s or "")


def first_sentence(s: str) -> str:
    s = strip_refs(s)
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    return m.group(1) if m else s


class Vault:
    """Lazy view of the vault artefacts the coach needs."""

    def __init__(self, strategies=None, mechanics=None, edges=None, portfolio=None):
        self.cards = {c["slug"]: c for c in (strategies or vload("strategies", {})).get("cards", [])}
        trees = (mechanics or vload("mechanics", {})).get("trees", [])
        self.nodes = {f"{t['id']}.{n['id']}": n for t in trees for n in t["nodes"]}
        self.edges = {e["slug"]: e for e in (edges or vload("edges", {})).get("edges", [])}
        self.portfolio = portfolio or vload("portfolio", {})

    def resolve(self, ref: str) -> str | None:
        """Text behind a reference, or None if it does not exist."""
        kind, _, key = ref.partition(":")
        if kind == "card":
            slug, _, path = key.partition(".")
            node = self.cards.get(slug)
            for part in path.split(".") if path else []:
                if isinstance(node, dict):
                    node = node.get(part)
                elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                    node = node[int(part)]
                else:
                    return None
            return node if isinstance(node, str) else None
        if kind == "tree":
            n = self.nodes.get(key)
            return n["text"] if n else None
        if kind == "edge":
            e = self.edges.get(key)
            return e.get("finding") if e else None
        if ref == "alloc":
            return ((self.portfolio.get("allocation") or {}).get("bp_vs_vix") or {}).get("guidance")
        return None

    def nudge(self, severity: str, title: str, ref: str, lead: str = "") -> Nudge | None:
        raw = self.resolve(ref)
        if raw is None:
            return None
        text = (lead + " " if lead else "") + first_sentence(raw)
        return Nudge(severity, title, text.strip(), ref, refs_in(raw))


# ---------------------------------------------------------------- pre-trade --

def pretrade(v: Vault, row: dict, template: str, preview: dict, dte: int, nlv: float,
             bp_used: float, open_tickers: list[str], earn_in_expiry: bool) -> list[Nudge]:
    """Nudges on a trade ticket. row = Market.snapshot row for the ticker."""
    spec = TEMPLATES[template]
    card = spec["vault"]
    out: list[Nudge | None] = []
    selling = preview["credit"] > 0
    ivr = row.get("ivr")
    if selling and ivr is not None and ivr == ivr:
        if ivr < 0.25:
            out.append(v.nudge("warn", f"IV rank {ivr:.0%}: thin premium", f"card:{card}.avoid.0"))
            out.append(v.nudge("info", "What our test found", "edge:iv-rank-timing"))
        elif ivr >= 0.5:
            out.append(v.nudge("good", f"IV rank {ivr:.0%}: tastylive's sell zone", f"card:{card}.when.iv"))
    if earn_in_expiry and row.get("kind") == "stock":
        out.append(v.nudge("warn", "Earnings before expiry", "edge:earnings-iv-crush",
                           "This trade holds through a report."))
    if dte < 25 or dte > 70:
        out.append(v.nudge("info", f"{dte} DTE is off the 45-DTE default", f"card:{card}.setup.dte"))
    if nlv > 0:
        share = preview["bpr"] / nlv
        if share > 0.05:
            out.append(v.nudge("alert" if share > 0.10 else "warn",
                               f"This trade uses {share:.0%} of net liq", "tree:opening.o5"))
        total = (bp_used + preview["bpr"]) / nlv
        if total > BP_GUIDE:
            out.append(v.nudge("alert", f"Book would use {total:.0%} of net liq in BP", "alloc"))
    if spec["risk"] == "undefined":
        out.append(v.nudge("info", "Undefined risk", f"card:{card}.size.per_trade"))
    if row.get("tier") == "C":
        out.append(v.nudge("warn", "Wide markets on this name", "tree:opening.o1"))
    if row.get("ticker") in open_tickers:
        out.append(v.nudge("info", "Already exposed to this name", "edge:number-of-occurrences",
                           "Concentration adds no occurrences."))
    return sorted([n for n in out if n], key=lambda n: SEV_ORDER[n.severity])


# ------------------------------------------------------------------ in-run --

EVENT_REFS = {
    "dte21": ("warn", "21 DTE: close or roll", "tree:managing-winners.w4"),
    "target": ("good", "Profit target hit", "tree:managing-winners.w2"),
    "stop": ("alert", "Loss at 2x the credit", "tree:taking-losses.l2"),
    "earnings": ("warn", "Report tomorrow", "edge:earnings-iv-crush"),
    "bp": ("alert", "Buying power over the guideline", "alloc"),
    "expiry": ("info", "Expiration", "tree:expiration-assignment.e3"),
}
EVENT_CAVEATS = {"target": "edge:managing-winners", "dte21": "edge:exit-21-dte",
                 "stop": "edge:stop-loss-2x"}


def on_events(v: Vault, events: list[dict]) -> list[Nudge]:
    out = []
    seen = set()
    for e in events:
        spec = EVENT_REFS.get(e["kind"])
        if not spec or (e["kind"], e.get("pid")) in seen:
            continue
        seen.add((e["kind"], e.get("pid")))
        sev, title, ref = spec
        n = v.nudge(sev, title, ref, e["msg"] + ".")
        if n:
            out.append(n)
        cav = EVENT_CAVEATS.get(e["kind"])
        if cav and (e["kind"], "cav") not in seen:
            seen.add((e["kind"], "cav"))
            c = v.nudge("info", "Evidence check", cav)
            if c:
                out.append(c)
    return out


# ----------------------------------------------------------------- debrief --

def adherence(log: list[dict]) -> tuple[float | None, list[dict]]:
    """Share of management calls (21 DTE, profit target) acted on within FOLLOW_DAYS."""
    calls = [e for e in log if e["kind"] in ("dte21", "target") and e.get("pid")]
    acts = [e for e in log if e["kind"] in ("close", "roll", "expire")]
    rows = []
    for c in calls:
        ok = any(a.get("pid") == c["pid"] and c["i"] <= a["i"] <= c["i"] + FOLLOW_DAYS for a in acts)
        rows.append({"i": c["i"], "kind": c["kind"], "pid": c["pid"], "followed": ok})
    if not rows:
        return None, rows
    return sum(r["followed"] for r in rows) / len(rows), rows

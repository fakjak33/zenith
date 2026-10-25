"""THETA RUN persistence on the EPHEMERIS repository (same players, same database).

One extra table, created on first use (IF NOT EXISTS; row-level security on
Postgres like the EPHEMERIS tables). State is a compact JSON of the Run, saved
on week/event advances, on exit and at the end -- a handful of small writes per
run, well inside a free Supabase project.
"""

from __future__ import annotations

import json

from ..ephemeris.repo import DIALECTS, Repository, now_iso

_DDL = """
CREATE TABLE IF NOT EXISTS theta_runs (
    id {pk},
    player_id BIGINT NOT NULL REFERENCES players(id),
    mode TEXT NOT NULL,
    challenge TEXT,
    difficulty TEXT NOT NULL,
    scenario_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    status TEXT NOT NULL,
    ret {real},
    max_dd {real},
    xp INTEGER,
    achievements_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class ThetaStore:
    def __init__(self, repo: Repository):
        self.r = repo
        repo._exec(_DDL.format(**DIALECTS[repo.dialect]).strip())
        if repo.dialect == "postgres":
            repo._exec("ALTER TABLE theta_runs ENABLE ROW LEVEL SECURITY")

    def create_run(self, player_id: int, mode: str, scenario: dict, state: dict) -> int:
        t = now_iso()
        return self.r._exec(
            "INSERT INTO theta_runs (player_id, mode, challenge, difficulty, scenario_json, "
            "state_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "RETURNING id",
            (player_id, mode, scenario.get("challenge"), scenario["difficulty"],
             json.dumps(scenario), json.dumps(state), "active", t, t), returning=True)

    def save_state(self, run_id: int, state: dict) -> None:
        self.r._exec("UPDATE theta_runs SET state_json = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(state), now_iso(), run_id))

    def finish(self, run_id: int, state: dict, ret: float, max_dd: float, xp: int,
               achievements: list[str]) -> None:
        self.r._exec("UPDATE theta_runs SET state_json = ?, status = 'done', ret = ?, max_dd = ?, "
                     "xp = ?, achievements_json = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(state), ret, max_dd, xp, json.dumps(achievements), now_iso(), run_id))

    def abandon(self, run_id: int) -> None:
        self.r._exec("UPDATE theta_runs SET status = 'abandoned', updated_at = ? WHERE id = ?",
                     (now_iso(), run_id))

    def active_run(self, player_id: int) -> dict | None:
        rows = self.r._query("SELECT * FROM theta_runs WHERE player_id = ? AND status = 'active' "
                             "ORDER BY updated_at DESC LIMIT 1", (player_id,))
        if not rows:
            return None
        row = rows[0]
        row["scenario"] = json.loads(row["scenario_json"])
        row["state"] = json.loads(row["state_json"])
        return row

    def history(self, player_id: int, limit: int = 50) -> list[dict]:
        rows = self.r._query("SELECT id, mode, challenge, difficulty, ret, max_dd, xp, "
                             "achievements_json, updated_at FROM theta_runs WHERE player_id = ? "
                             "AND status = 'done' ORDER BY updated_at DESC LIMIT ?", (player_id, limit))
        for r in rows:
            r["achievements"] = json.loads(r.pop("achievements_json") or "[]")
        return rows

    def total_xp(self, player_id: int) -> int:
        rows = self.r._query("SELECT COALESCE(SUM(xp), 0) AS xp FROM theta_runs "
                             "WHERE player_id = ? AND status = 'done'", (player_id,))
        return int(rows[0]["xp"] or 0) if rows else 0

    def achievements(self, player_id: int) -> set[str]:
        out: set[str] = set()
        for r in self.history(player_id, limit=500):
            out.update(r["achievements"])
        return out

    def leaderboard(self, challenge: str | None = None, limit: int = 10) -> list[dict]:
        sql = ("SELECT p.display, r.challenge, r.difficulty, r.ret, r.max_dd, r.xp, r.updated_at "
               "FROM theta_runs r JOIN players p ON p.id = r.player_id WHERE r.status = 'done'")
        params: list = []
        if challenge:
            sql += " AND r.challenge = ?"
            params.append(challenge)
        sql += " ORDER BY r.xp DESC LIMIT ?"
        params.append(limit)
        return self.r._query(sql, tuple(params))

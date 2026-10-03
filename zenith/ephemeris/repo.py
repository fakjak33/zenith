"""Persistence behind a small repository interface.

    get_repo()  ->  PostgresRepo  when st.secrets["ephemeris_db_url"] is set
                    (Supabase session-pooler URI; Community Cloud is IPv4-only)
                ->  SqliteRepo    otherwise (data/ephemeris/ephemeris.sqlite3;
                    ":memory:" in tests)

Both share one SQL dialect subset: `?` placeholders (translated to `%s` for
psycopg2), ISO-8601 UTC text timestamps, and RETURNING id. Tables are
created on first connect (IF NOT EXISTS), so a fresh Supabase project needs
no manual SQL. On Postgres, row-level security is enabled with no policies:
the app's own connection (table owner) bypasses it, while Supabase's public
REST API sees nothing.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TABLES = ("players", "guesses", "daily_results", "account_resets", "settings_presets")

_DDL = """
CREATE TABLE IF NOT EXISTS players (
    id {pk},
    handle TEXT NOT NULL UNIQUE,
    display TEXT NOT NULL,
    pin_hash TEXT,
    pin_salt TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS guesses (
    id {pk},
    player_id BIGINT NOT NULL REFERENCES players(id),
    mode TEXT NOT NULL,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    asset_class TEXT,
    sector TEXT,
    timeframe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    lookback INTEGER NOT NULL,
    window_start TEXT,
    decision_date TEXT NOT NULL,
    chart_key TEXT NOT NULL,
    indicators_json TEXT,
    blinding_json TEXT,
    direction INTEGER NOT NULL,
    conviction TEXT,
    stake {real},
    entry {real},
    sl {real},
    tp {real},
    sl_spec TEXT,
    tp_spec TEXT,
    note TEXT,
    outcome TEXT,
    market_ret {real},
    trade_ret {real},
    pnl {real},
    win INTEGER,
    r_mult {real},
    atr_ret {real},
    mfe {real},
    mae {real},
    mfe_full {real},
    mae_full {real},
    candles_held INTEGER,
    exit_reason TEXT,
    ambiguous INTEGER,
    gap_fill INTEGER,
    base_rate {real},
    rule_call INTEGER,
    rule_ret {real},
    regime_json TEXT,
    setup_tag TEXT,
    decision_ms INTEGER
);
CREATE INDEX IF NOT EXISTS guesses_player_mode ON guesses (player_id, mode);
CREATE TABLE IF NOT EXISTS daily_results (
    player_id BIGINT NOT NULL REFERENCES players(id),
    game_date TEXT NOT NULL,
    slot INTEGER NOT NULL,
    guess_id BIGINT NOT NULL REFERENCES guesses(id),
    PRIMARY KEY (player_id, game_date, slot)
);
CREATE TABLE IF NOT EXISTS account_resets (
    id {pk},
    player_id BIGINT NOT NULL REFERENCES players(id),
    mode TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings_presets (
    id {pk},
    player_id BIGINT NOT NULL REFERENCES players(id),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (player_id, kind, name)
);
"""

DIALECTS = {
    "sqlite": {"pk": "INTEGER PRIMARY KEY AUTOINCREMENT", "real": "REAL"},
    "postgres": {"pk": "BIGSERIAL PRIMARY KEY", "real": "DOUBLE PRECISION"},
}

GUESS_FIELDS = ("player_id", "mode", "ts", "ticker", "name", "asset_class", "sector", "timeframe",
                "horizon", "lookback", "window_start", "decision_date", "chart_key", "indicators_json",
                "blinding_json", "direction", "conviction", "stake", "entry", "sl", "tp", "sl_spec",
                "tp_spec", "note", "outcome", "market_ret", "trade_ret", "pnl", "win", "r_mult",
                "atr_ret", "mfe", "mae", "mfe_full", "mae_full", "candles_held", "exit_reason",
                "ambiguous", "gap_fill", "base_rate", "rule_call", "rule_ret", "regime_json",
                "setup_tag", "decision_ms")


def ddl(dialect: str) -> str:
    sql = _DDL.format(**DIALECTS[dialect])
    if dialect == "postgres":
        sql += "".join(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;\n" for t in TABLES)
    return sql


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_handle(handle: str) -> str:
    return " ".join((handle or "").split()).lower()


def valid_handle(handle: str) -> str | None:
    """None if fine, else a short reason. Deliberately permissive."""
    h = " ".join((handle or "").split())
    if not 2 <= len(h) <= 24:
        return "2–24 characters"
    if not all(ch.isalnum() or ch in " ._-" for ch in h):
        return "letters, numbers, space, . _ - only"
    return None


def _pin_hash(pin: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{pin}".encode()).hexdigest()


class Repository:
    """Shared logic; subclasses supply _exec/_query and the dialect."""
    dialect = "sqlite"
    ph = "?"

    def _sql(self, sql: str) -> str:
        return sql if self.ph == "?" else sql.replace("?", self.ph)

    # subclasses implement
    def _exec(self, sql: str, params=(), returning: bool = False):
        raise NotImplementedError

    def _query(self, sql: str, params=()) -> list[dict]:
        raise NotImplementedError

    def init(self) -> None:
        for stmt in [s.strip() for s in ddl(self.dialect).split(";") if s.strip()]:
            self._exec(stmt)

    # ---------------------------------------------------------- players -----
    def players(self) -> list[dict]:
        return self._query("SELECT id, handle, display, pin_hash IS NOT NULL AS has_pin, created_at "
                           "FROM players ORDER BY display")

    def player(self, handle: str) -> dict | None:
        rows = self._query("SELECT * FROM players WHERE handle = ?", (normalize_handle(handle),))
        return rows[0] if rows else None

    def get_or_create_player(self, handle: str) -> dict:
        p = self.player(handle)
        if p:
            return p
        display = " ".join(handle.split())
        try:
            self._exec("INSERT INTO players (handle, display, created_at) VALUES (?, ?, ?)",
                       (normalize_handle(handle), display, now_iso()))
        except Exception:
            pass                       # lost a race to the same handle -> just read it
        return self.player(handle)

    def set_pin(self, player_id: int, pin: str | None) -> None:
        if pin:
            salt = secrets.token_hex(8)
            self._exec("UPDATE players SET pin_hash = ?, pin_salt = ? WHERE id = ?",
                       (_pin_hash(pin, salt), salt, player_id))
        else:
            self._exec("UPDATE players SET pin_hash = NULL, pin_salt = NULL WHERE id = ?", (player_id,))

    @staticmethod
    def check_pin(player: dict, pin: str) -> bool:
        if not player.get("pin_hash"):
            return True
        return secrets.compare_digest(_pin_hash(pin or "", player.get("pin_salt") or ""), player["pin_hash"])

    # ---------------------------------------------------------- guesses -----
    def log_guess(self, row: dict) -> int:
        row = {k: row.get(k) for k in GUESS_FIELDS}
        row["ts"] = row["ts"] or now_iso()
        for k in ("win", "ambiguous", "gap_fill"):
            if row[k] is not None:
                row[k] = int(bool(row[k]))
        for k in ("indicators_json", "blinding_json", "regime_json"):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], default=str)
        cols = ", ".join(GUESS_FIELDS)
        qs = ", ".join("?" for _ in GUESS_FIELDS)
        return self._exec(f"INSERT INTO guesses ({cols}) VALUES ({qs}) RETURNING id",
                          tuple(row[k] for k in GUESS_FIELDS), returning=True)

    def guesses_df(self, player_id: int, mode: str | None = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM guesses WHERE player_id = ?", [player_id]
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        df = pd.DataFrame(self._query(sql + " ORDER BY id", tuple(params)))
        if df.empty:
            return pd.DataFrame(columns=("id",) + GUESS_FIELDS)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def seen_charts(self, player_id: int) -> dict:
        out: dict = {}
        for r in self._query("SELECT ticker, timeframe, decision_date FROM guesses WHERE player_id = ?",
                             (player_id,)):
            out.setdefault((r["ticker"], r["timeframe"]), []).append(pd.Timestamp(r["decision_date"]))
        return out

    # ---------------------------------------------------------- account -----
    def reset_account(self, player_id: int, mode: str) -> None:
        self._exec("INSERT INTO account_resets (player_id, mode, ts) VALUES (?, ?, ?)",
                   (player_id, mode, now_iso()))

    def resets(self, player_id: int, mode: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM account_resets WHERE player_id = ?", [player_id]
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        return self._query(sql + " ORDER BY id", tuple(params))

    # ---------------------------------------------------------- daily -------
    def save_daily(self, player_id: int, game_date: str, slot: int, guess_id: int) -> bool:
        try:
            self._exec("INSERT INTO daily_results (player_id, game_date, slot, guess_id) VALUES (?, ?, ?, ?)",
                       (player_id, game_date, slot, guess_id))
            return True
        except Exception:
            return False               # already played this slot

    def daily_for(self, player_id: int) -> list[dict]:
        return self._query("SELECT d.game_date, d.slot, g.* FROM daily_results d JOIN guesses g "
                           "ON g.id = d.guess_id WHERE d.player_id = ? ORDER BY d.game_date, d.slot",
                           (player_id,))

    # ---------------------------------------------------------- presets -----
    def presets(self, player_id: int, kind: str) -> dict[str, dict]:
        return {r["name"]: json.loads(r["payload_json"]) for r in self._query(
            "SELECT name, payload_json FROM settings_presets WHERE player_id = ? AND kind = ? ORDER BY name",
            (player_id, kind))}

    def save_preset(self, player_id: int, kind: str, name: str, payload: dict) -> None:
        self._exec("DELETE FROM settings_presets WHERE player_id = ? AND kind = ? AND name = ?",
                   (player_id, kind, name))
        self._exec("INSERT INTO settings_presets (player_id, kind, name, payload_json, created_at) "
                   "VALUES (?, ?, ?, ?, ?)", (player_id, kind, name, json.dumps(payload), now_iso()))

    def delete_preset(self, player_id: int, kind: str, name: str) -> None:
        self._exec("DELETE FROM settings_presets WHERE player_id = ? AND kind = ? AND name = ?",
                   (player_id, kind, name))


class SqliteRepo(Repository):
    dialect, ph = "sqlite", "?"

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self.init()

    def _exec(self, sql, params=(), returning=False):
        with self._lock:
            cur = self._con.execute(self._sql(sql), params)
            out = cur.fetchone()[0] if returning else None
            self._con.commit()
            return out

    def _query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._con.execute(self._sql(sql), params).fetchall()]

    @property
    def label(self) -> str:
        return "local SQLite"


def parse_db_url(url: str) -> dict:
    """Postgres URI -> psycopg2 keyword args, tolerant of how people paste
    Supabase strings: the password is taken LITERALLY (everything between the
    first ':' after the scheme and the LAST '@'), so '@ # / : ?' need no
    percent-encoding, and leftover [brackets] from '[YOUR-PASSWORD]' are
    stripped. `password_alt` is the percent-decoded form, tried second."""
    from urllib.parse import unquote
    u = (url or "").strip().strip('"').strip("'")
    scheme, sep, rest = u.partition("://")
    if not sep or scheme not in ("postgres", "postgresql"):
        raise ValueError("ephemeris_db_url must start with postgresql://")
    creds, at, hostpart = rest.rpartition("@")
    if not at:
        raise ValueError("ephemeris_db_url has no user:password@ part")
    user, _, password = creds.partition(":")
    if len(password) > 2 and password.startswith("[") and password.endswith("]"):
        password = password[1:-1]
    hostport, _, tail = hostpart.partition("/")
    dbname = (tail.split("?", 1)[0] or "postgres")
    host, _, port = hostport.rpartition(":") if hostport.count(":") == 1 else (hostport, "", "")
    out = {"host": host or hostport, "port": int(port) if port.isdigit() else 5432, "user": unquote(user),
           "password": password, "dbname": dbname, "sslmode": "require"}
    alt = unquote(password)
    if alt != password:
        out["password_alt"] = alt
    return out


def db_url_problem(url: str) -> str | None:
    """A plain-English diagnosis of a Supabase URL, never echoing the password."""
    try:
        kw = parse_db_url(url)
    except ValueError as exc:
        return str(exc)
    if "[YOUR-PASSWORD]" in url or kw["password"] in ("YOUR-PASSWORD", ""):
        return "the password placeholder was not replaced"
    if "@db." in url and ".supabase.co" in url:
        return ("this is Supabase's Direct connection string (IPv6-only); use the Session pooler URI "
                "(host ends in pooler.supabase.com)")
    if "pooler.supabase.com" in kw["host"] and "." not in kw["user"]:
        return "pooler user must be postgres.<project-ref>, not plain 'postgres'"
    return None


class PostgresRepo(Repository):
    dialect, ph = "postgres", "%s"

    def __init__(self, url: str):
        self.url = url
        self._kw = parse_db_url(url)
        self._con = None
        self._lock = threading.Lock()
        self.init()

    def _conn(self):
        import psycopg2
        if self._con is None or self._con.closed:
            kw = {k: v for k, v in self._kw.items() if k != "password_alt"}
            try:
                self._con = psycopg2.connect(connect_timeout=10, **kw)
            except psycopg2.OperationalError as exc:
                alt = self._kw.get("password_alt")
                if not alt or "password authentication failed" not in str(exc):
                    raise
                # the pasted password was percent-encoded -> try the decoded form
                self._con = psycopg2.connect(connect_timeout=10, **(kw | {"password": alt}))
                self._kw["password"] = alt
        return self._con

    def _run(self, fn):
        with self._lock:
            for attempt in (0, 1):                 # one reconnect on a dropped pooler link
                try:
                    con = self._conn()
                    with con.cursor() as cur:
                        out = fn(cur)
                    con.commit()
                    return out
                except Exception as exc:
                    import psycopg2
                    try:
                        self._con.rollback()
                    except Exception:
                        pass
                    if attempt == 0 and isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError)):
                        self._con = None
                        continue
                    raise

    def _exec(self, sql, params=(), returning=False):
        def fn(cur):
            cur.execute(self._sql(sql), params)
            return cur.fetchone()[0] if returning else None
        return self._run(fn)

    def _query(self, sql, params=()):
        def fn(cur):
            cur.execute(self._sql(sql), params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        return self._run(fn)

    @property
    def label(self) -> str:
        return "Supabase Postgres"


def get_repo(db_url: str | None = None, sqlite_path: str | Path | None = None) -> Repository:
    if db_url:
        return PostgresRepo(db_url)
    from ..config import EPHEMERIS_FILES
    return SqliteRepo(sqlite_path or EPHEMERIS_FILES["local_db"])

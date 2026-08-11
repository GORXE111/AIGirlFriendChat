"""SQLite 持久化。

一个存档 = 一个 save 行 + 它的消息、事实、情节、状态。
用标准库 sqlite3，不引 ORM —— 表很少，查询很简单，多一层抽象只是负担。

WAL 模式：前端轮询主动消息时会有并发读，WAL 下读不阻塞写。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS saves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    character_id    TEXT NOT NULL DEFAULT 'h01',
    player_surname  TEXT NOT NULL DEFAULT '',
    player_given    TEXT NOT NULL DEFAULT '',
    stage           TEXT NOT NULL DEFAULT 'S0',
    affinity        REAL NOT NULL DEFAULT 0,
    emotions        TEXT NOT NULL DEFAULT '{}',
    emotions_at     TEXT NOT NULL,
    chapter         INTEGER NOT NULL DEFAULT 0,
    flags           TEXT NOT NULL DEFAULT '[]',
    beat_progress   TEXT NOT NULL DEFAULT '{}',
    pending_options TEXT NOT NULL DEFAULT '[]',
    pending_topics  TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    save_id      INTEGER NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    role         TEXT NOT NULL,              -- user | assistant
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    deliver_at   TEXT,                       -- 计划送达时间；NULL = 立即
    delivered    INTEGER NOT NULL DEFAULT 1,
    proactive    INTEGER NOT NULL DEFAULT 0, -- 是否她主动发起
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_save ON messages(save_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_pending
    ON messages(save_id, delivered, deliver_at);

-- 语义记忆：关于玩家的事实
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    save_id     INTEGER NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'other',
    learned_at  TEXT NOT NULL,
    UNIQUE(save_id, content)
);
CREATE INDEX IF NOT EXISTS idx_facts_save ON facts(save_id);

-- 情节记忆：带时间戳的事件。「你上周三说嗓子疼」全靠这张表
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    save_id     INTEGER NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    happened_at TEXT NOT NULL,
    importance  INTEGER NOT NULL DEFAULT 1,  -- 1..5
    recalled    INTEGER NOT NULL DEFAULT 0,  -- 被回指过几次
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_save ON episodes(save_id, happened_at DESC);

-- 洞察：从若干情节里综合出来的更高层结论。
--
-- 抽取产出「他有胃病」，综合产出「他每次考试前都会胃疼」。
-- **前者是「她记得」，后者才是「她懂你」。**
-- 内梗也放在这里（kind='joke'）—— 它就是"我们之间反复出现的东西"。
CREATE TABLE IF NOT EXISTS insights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    save_id     INTEGER NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'him',   -- him | us | joke
    weight      INTEGER NOT NULL DEFAULT 1,    -- 被印证过几次
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(save_id, content)
);
CREATE INDEX IF NOT EXISTS idx_insights_save ON insights(save_id, weight DESC);

CREATE TABLE IF NOT EXISTS reflect_marks (
    save_id          INTEGER PRIMARY KEY REFERENCES saves(id) ON DELETE CASCADE,
    last_message_id  INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    # (表, 列, DDL)。CREATE TABLE IF NOT EXISTS 不会给已有表补列，
    # 加字段时必须在这里登记一条，否则老存档会崩。
    ("saves", "flags", "ALTER TABLE saves ADD COLUMN flags TEXT NOT NULL DEFAULT '[]'"),
    ("saves", "beat_progress",
     "ALTER TABLE saves ADD COLUMN beat_progress TEXT NOT NULL DEFAULT '{}'"),
    ("saves", "pending_options",
     "ALTER TABLE saves ADD COLUMN pending_options TEXT NOT NULL DEFAULT '[]'"),
    ("saves", "pending_topics",
     "ALTER TABLE saves ADD COLUMN pending_topics TEXT NOT NULL DEFAULT '[]'"),
)


class Database:
    def __init__(self, path: str | Path = "saves.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        for table, column, ddl in MIGRATIONS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(ddl)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ---------------- 存档 ----------------

    def create_save(
        self,
        name: str,
        *,
        character_id: str = "h01",
        surname: str = "",
        given: str = "",
    ) -> int:
        now = utcnow()
        with self.connect() as c:
            cur = c.execute(
                """INSERT INTO saves
                   (name, character_id, player_surname, player_given,
                    emotions, emotions_at, created_at, updated_at)
                   VALUES (?,?,?,?,'{}',?,?,?)""",
                (name, character_id, surname, given, now, now, now),
            )
            return int(cur.lastrowid)

    def get_save(self, save_id: int) -> dict[str, Any] | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM saves WHERE id=?", (save_id,)).fetchone()
        return dict(row) if row else None

    def list_saves(self) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM saves ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_save(self, save_id: int, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "name", "player_surname", "player_given", "stage",
            "affinity", "emotions", "emotions_at", "chapter",
            "flags", "beat_progress", "pending_options", "pending_topics",
        }
        json_fields = {"emotions", "flags", "beat_progress",
                       "pending_options", "pending_topics"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"不可更新的字段：{k}")
            sets.append(f"{k}=?")
            if k in json_fields and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            vals.append(v)
        sets.append("updated_at=?")
        vals.extend([utcnow(), save_id])
        with self.connect() as c:
            c.execute(f"UPDATE saves SET {','.join(sets)} WHERE id=?", vals)

    def delete_save(self, save_id: int) -> None:
        with self.connect() as c:
            c.execute("DELETE FROM saves WHERE id=?", (save_id,))

    # ---------------- 消息 ----------------

    def add_message(
        self,
        save_id: int,
        role: str,
        content: str,
        *,
        deliver_at: str | None = None,
        delivered: bool = True,
        proactive: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as c:
            cur = c.execute(
                """INSERT INTO messages
                   (save_id, role, content, created_at, deliver_at,
                    delivered, proactive, meta)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    save_id, role, content, utcnow(), deliver_at,
                    int(delivered), int(proactive),
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def recent_messages(
        self, save_id: int, limit: int = 30, *, delivered_only: bool = True
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE save_id=?"
        if delivered_only:
            sql += " AND delivered=1"
        sql += " ORDER BY id DESC LIMIT ?"
        with self.connect() as c:
            rows = c.execute(sql, (save_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def due_messages(self, save_id: int, now: str | None = None) -> list[dict[str, Any]]:
        """到点该送达但还没送达的消息。"""
        now = now or utcnow()
        with self.connect() as c:
            rows = c.execute(
                """SELECT * FROM messages
                   WHERE save_id=? AND delivered=0 AND deliver_at<=?
                   ORDER BY id""",
                (save_id, now),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_delivered(self, message_ids: list[int]) -> None:
        if not message_ids:
            return
        with self.connect() as c:
            c.executemany(
                "UPDATE messages SET delivered=1 WHERE id=?",
                [(i,) for i in message_ids],
            )

    def last_user_message_at(self, save_id: int) -> datetime | None:
        with self.connect() as c:
            row = c.execute(
                """SELECT created_at FROM messages
                   WHERE save_id=? AND role='user'
                   ORDER BY id DESC LIMIT 1""",
                (save_id,),
            ).fetchone()
        return parse_ts(row["created_at"]) if row else None

    # ---------------- 记忆 ----------------

    def add_fact(self, save_id: int, content: str, category: str = "other") -> None:
        with self.connect() as c:
            c.execute(
                """INSERT OR IGNORE INTO facts (save_id, content, category, learned_at)
                   VALUES (?,?,?,?)""",
                (save_id, content.strip(), category, utcnow()),
            )

    def get_facts(self, save_id: int, limit: int = 60) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM facts WHERE save_id=? ORDER BY id DESC LIMIT ?",
                (save_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_episode(
        self, save_id: int, summary: str, happened_at: str, importance: int = 1
    ) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO episodes
                   (save_id, summary, happened_at, importance, created_at)
                   VALUES (?,?,?,?,?)""",
                (save_id, summary.strip(), happened_at, importance, utcnow()),
            )

    def get_episodes(self, save_id: int, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute(
                """SELECT * FROM episodes WHERE save_id=?
                   ORDER BY happened_at DESC LIMIT ?""",
                (save_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- 洞察 ----------------

    def upsert_insight(
        self, save_id: int, content: str, kind: str = "him"
    ) -> None:
        """同一条洞察被再次印证时加权，而不是插重复。

        weight 就是「这个规律出现过几次」—— 越高越可信，也越该被她挂在嘴边。
        """
        now = utcnow()
        with self.connect() as c:
            c.execute(
                """INSERT INTO insights
                   (save_id, content, kind, weight, created_at, updated_at)
                   VALUES (?,?,?,1,?,?)
                   ON CONFLICT(save_id, content) DO UPDATE
                   SET weight = weight + 1, updated_at = excluded.updated_at""",
                (save_id, content.strip(), kind, now, now),
            )

    def get_insights(
        self, save_id: int, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM insights WHERE save_id=?"
        args: list[Any] = [save_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY weight DESC, id DESC LIMIT ?"
        args.append(limit)
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # ---------------- 慢回路水位 ----------------

    def get_reflect_mark(self, save_id: int) -> int:
        with self.connect() as c:
            row = c.execute(
                "SELECT last_message_id FROM reflect_marks WHERE save_id=?", (save_id,)
            ).fetchone()
        return int(row["last_message_id"]) if row else 0

    def set_reflect_mark(self, save_id: int, message_id: int) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO reflect_marks (save_id, last_message_id, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(save_id) DO UPDATE
                   SET last_message_id=excluded.last_message_id,
                       updated_at=excluded.updated_at""",
                (save_id, message_id, utcnow()),
            )

    def messages_since(self, save_id: int, after_id: int) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute(
                """SELECT * FROM messages
                   WHERE save_id=? AND id>? AND delivered=1
                   ORDER BY id""",
                (save_id, after_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- 存档导出 ----------------

    def export_save(self, save_id: int) -> dict[str, Any]:
        save = self.get_save(save_id)
        if save is None:
            raise KeyError(f"存档不存在：{save_id}")
        return {
            "save": save,
            "messages": self.recent_messages(save_id, limit=100000, delivered_only=False),
            "facts": self.get_facts(save_id, limit=100000),
            "episodes": self.get_episodes(save_id, limit=100000),
            "reflect_mark": self.get_reflect_mark(save_id),
        }

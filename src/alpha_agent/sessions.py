"""Conversation session store (SQLite) for multi-turn chat.

Each session stores the distilled conversation: pairs of
(user question, assistant final answer).  The full history is re-injected
into the agent on the next turn so it can answer follow-ups like
"and what about MSFT?".
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 20  # keep last 20 Q/A pairs to bound context size


class SessionStore:
    """Persistent multi-turn session store (SQLite)."""

    def __init__(self, db_path: str | Path = "sessions.sqlite3") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new empty session and return its id."""
        session_id = secrets.token_urlsafe(16)
        with self._lock:
            self._db().execute(
                "INSERT INTO sessions (id, messages, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, json.dumps([]), time.time(), time.time()),
            )
            self._db().commit()
        return session_id

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        """Return the conversation history for *session_id* (empty if unknown)."""
        with self._lock:
            row = self._db().execute(
                "SELECT messages FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return []
        try:
            messages = json.loads(row[0])
        except json.JSONDecodeError:
            return []
        return [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if m.get("content")
        ]

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        """Append a (question, answer) pair, trimming to the last N turns."""
        messages = self.load_messages(session_id)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        # keep only the most recent turns to bound the prompt size
        max_msgs = _MAX_HISTORY_TURNS * 2
        if len(messages) > max_msgs:
            messages = messages[-max_msgs:]
        with self._lock:
            self._db().execute(
                "UPDATE sessions SET messages = ?, updated_at = ? WHERE id = ?",
                (json.dumps(messages), time.time(), session_id),
            )
            self._db().commit()

    def exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._db().execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._db().execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._db().commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "  id TEXT PRIMARY KEY,"
                "  messages TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  updated_at REAL NOT NULL"
                ")"
            )
            db.commit()

    def _db(self) -> sqlite3.Connection:
        if not hasattr(self, "_conn"):
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def close(self) -> None:
        if hasattr(self, "_conn"):
            self._conn.close()

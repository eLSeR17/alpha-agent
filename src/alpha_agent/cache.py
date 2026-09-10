"""SQLite-backed response cache.

Caches full ``AgentResponse`` payloads keyed by a hash of the query.
Repeated identical questions (e.g. "AAPL price" asked every morning)
are answered from cache, saving model tokens and latency.

The cache is optional: if it is never enabled, the agent behaves
exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from .schemas import AgentResponse, ToolCall, ToolResult

logger = logging.getLogger(__name__)

_TTL_SECONDS = 3600.0  # 1 hour default


class ResponseCache:
    """Tiny SQLite cache keyed by SHA-256 of the query.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  ``:memory:`` for tests.
    ttl_seconds:
        How long a cached entry stays valid.
    """

    def __init__(self, db_path: str | Path = "cache.sqlite3", ttl_seconds: float = _TTL_SECONDS) -> None:
        self.db_path = str(db_path)
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> AgentResponse | None:
        """Return a cached :class:`AgentResponse` or ``None`` on miss."""
        key = self._hash(query)
        with self._lock:
            row = self._db().execute(
                "SELECT payload, ts FROM responses WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            return None

        payload, ts = row
        if time.time() - ts > self.ttl:
            with self._lock:
                self._db().execute("DELETE FROM responses WHERE key = ?", (key,))
                self._db().commit()
            return None

        try:
            return self._deserialize(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Cached payload for %s unparseable: %s", query[:40], exc)
            return None

    def set(self, response: AgentResponse) -> None:
        """Store a response under its query key."""
        key = self._hash(response.query)
        payload = self._serialize(response)
        with self._lock:
            self._db().execute(
                "INSERT OR REPLACE INTO responses (key, payload, ts) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._db().commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "  key TEXT PRIMARY KEY,"
                "  payload TEXT NOT NULL,"
                "  ts REAL NOT NULL"
                ")"
            )
            db.commit()

    def _db(self) -> sqlite3.Connection:
        """Open a connection (lazily cached on first use)."""
        if not hasattr(self, "_conn"):
            db_path = self.db_path
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        return self._conn

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize(response: AgentResponse) -> str:
        data = {
            "query": response.query,
            "final_answer": response.final_answer,
            "reasoning": response.reasoning,
            "tool_calls": [
                {
                    "tool_call": {"name": tr.tool_call.name, "arguments": tr.tool_call.arguments},
                    "output": tr.output,
                    "success": tr.success,
                    "error": tr.error,
                }
                for tr in response.tool_calls
            ],
            "iterations_used": response.iterations_used,
            "data": response.data,
        }
        return json.dumps(data)

    @staticmethod
    def _deserialize(payload: str) -> AgentResponse:
        data = json.loads(payload)
        return AgentResponse(
            query=data["query"],
            final_answer=data["final_answer"],
            reasoning=data.get("reasoning", []),
            tool_calls=[
                ToolResult(
                    tool_call=ToolCall(
                        name=tr["tool_call"]["name"],
                        arguments=tr["tool_call"].get("arguments", {}),
                    ),
                    output=tr.get("output", ""),
                    success=tr.get("success", True),
                    error=tr.get("error"),
                )
                for tr in data.get("tool_calls", [])
            ],
            iterations_used=data.get("iterations_used", 0),
            data=data.get("data", {}),
        )

    def close(self) -> None:
        if hasattr(self, "_conn"):
            self._conn.close()

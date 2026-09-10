"""Unit tests for the multi-turn session store."""

from __future__ import annotations

from alpha_agent.sessions import SessionStore


class TestSessionStore:
    def test_create_and_load(self) -> None:
        store = SessionStore(db_path=":memory:")
        sid = store.create_session()
        assert store.exists(sid)
        assert store.load_messages(sid) == []
        store.close()

    def test_append_and_load_turns(self) -> None:
        store = SessionStore(db_path=":memory:")
        sid = store.create_session()
        store.append_turn(sid, "What is AAPL price?", "AAPL is at 319.97.")
        store.append_turn(sid, "And MSFT?", "MSFT is at 428.5.")
        history = store.load_messages(sid)
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "What is AAPL price?"}
        assert history[1]["role"] == "assistant"
        assert history[3]["content"] == "MSFT is at 428.5."
        store.close()

    def test_unknown_session_empty(self) -> None:
        store = SessionStore(db_path=":memory:")
        assert store.load_messages("nope") == []
        assert not store.exists("nope")
        store.close()

    def test_delete_session(self) -> None:
        store = SessionStore(db_path=":memory:")
        sid = store.create_session()
        store.delete_session(sid)
        assert not store.exists(sid)
        store.close()

    def test_history_is_bounded(self) -> None:
        store = SessionStore(db_path=":memory:")
        sid = store.create_session()
        for i in range(30):
            store.append_turn(sid, f"Q{i}", f"A{i}")
        history = store.load_messages(sid)
        assert len(history) <= 40  # _MAX_HISTORY_TURNS * 2
        store.close()

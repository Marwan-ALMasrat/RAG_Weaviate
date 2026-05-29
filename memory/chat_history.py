"""
memory/chat_history.py — In-session conversation context.

Stores the turn-by-turn chat so the LLM has context for follow-up questions.
Stored in Streamlit session_state — lives for the duration of the browser session.

Design note:
  Intentionally simple in-memory store.
  For persistence across sessions, swap _store for a DB-backed implementation
  without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


class ChatHistory:
    """
    Manages a list of conversation messages.

    Parameters
    ----------
    max_turns : int
        Keep only the last N complete turns (user+assistant pairs).
        Prevents unbounded context growth.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_turns * 2  # each turn = 2 messages

    def add_user(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append(Message(role="assistant", content=text))
        self._trim()

    def as_list(self) -> list[dict]:
        """Return history as list of dicts for LLM prompt building."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    def __len__(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        if len(self._messages) > self._max_messages:
            # Remove oldest pair (keep most recent context)
            self._messages = self._messages[-self._max_messages:]

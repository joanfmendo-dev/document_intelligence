"""The copilot conversations, kept on the server (one per browser session) so that they survive a visit to another page.
They live in memory: restarting Medical Records starts empty, and nothing about a patient is written to disk by the chat."""
from __future__ import annotations

import secrets
from collections import OrderedDict
from threading import Lock

MAX_TURNS_KEPT = 50      # per conversation: the oldest turns are dropped
MAX_CONVERSATIONS = 20   # Medical Records is a local app: a few browsers at most


class ChatStore:
    def __init__(self) -> None:
        self._chats: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def new_id() -> str:
        return secrets.token_urlsafe(16)

    def _chat(self, chat_id: str) -> list[dict]:  # call with the lock held
        chat = self._chats.setdefault(chat_id, [])
        self._chats.move_to_end(chat_id)
        while len(self._chats) > MAX_CONVERSATIONS:
            self._chats.popitem(last=False)
        return chat

    def add(self, chat_id: str, question: str, plan: dict | None, reply: str) -> int:
        """Record a turn and return its id (the html is set once it is rendered, and changes when the user turns the page)."""
        with self._lock:
            chat = self._chat(chat_id)
            turn = (chat[-1]["id"] + 1) if chat else 0
            chat.append({"id": turn, "question": question, "plan": plan, "reply": reply, "html": ""})
            del chat[:-MAX_TURNS_KEPT]
            return turn

    def set_html(self, chat_id: str, turn: int, html: str) -> None:
        with self._lock:
            for t in self._chats.get(chat_id, []):
                if t["id"] == turn:
                    t["html"] = html

    def turns(self, chat_id: str) -> list[dict]:
        with self._lock:
            return [dict(t) for t in self._chats.get(chat_id, [])]

    def history(self, chat_id: str, last: int) -> list[dict]:
        """What the model sees of the conversation: the last turns that produced a plan, oldest first."""
        return [{k: t[k] for k in ("question", "plan", "reply")} for t in self.turns(chat_id) if t["plan"]][-last:]

    def clear(self, chat_id: str) -> None:
        with self._lock:
            self._chats.pop(chat_id, None)

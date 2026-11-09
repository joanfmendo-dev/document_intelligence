import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

from docintel.settings import s


class FakeOllama:
    """A local HTTP server that speaks enough of Ollama's API to test the real request path
    (our code -> litellm -> HTTP). It records every /api/chat request body."""

    def __init__(self):
        self.requests: list[dict] = []
        self._replies: list[dict] = []
        self.default = {"content": "{}", "done_reason": "stop"}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/api/chat"):
                    owner.requests.append(body)
                    reply = owner._replies.pop(0) if owner._replies else owner.default
                else:
                    reply = {"content": "", "done_reason": "stop"}
                status = reply.get("status", 200)
                out = json.dumps({"model": body.get("model"), "created_at": "2026-01-01T00:00:00Z",
                                  "message": {"role": "assistant", "content": reply["content"]},
                                  "done": True, "done_reason": reply["done_reason"],
                                  "prompt_eval_count": 123, "eval_count": 45}) if status == 200 else "boom"
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out.encode())))
                self.end_headers()
                self.wfile.write(out.encode())

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def reply(self, content, done_reason="stop", status=200):
        self._replies.append({"content": content, "done_reason": done_reason, "status": status})

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def ollama(monkeypatch):
    server = FakeOllama()
    monkeypatch.setattr(s, "LLM_BASE_URL", server.url)
    monkeypatch.setattr(s, "VLM_BASE_URL", server.url)
    yield server
    server.stop()

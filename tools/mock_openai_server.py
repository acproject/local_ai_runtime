import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "mock-model", "object": "model", "created": 0, "owned_by": "mock-openai"}
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        j = None
        try:
            j = json.loads(body) if body else {}
        except Exception:
            j = {}

        if self.path == "/v1/chat/completions":
            model = str(j.get("model") or "mock-model")
            msg = ""
            messages = j.get("messages") or []
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    msg = str(last.get("content") or "")
            self._send(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"mock:{msg}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                },
            )
            return

        if self.path == "/v1/embeddings":
            model = str(j.get("model") or "mock-model")
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
                    "model": model,
                    "usage": {"prompt_tokens": None, "total_tokens": None},
                },
            )
            return

        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19002)
    args = ap.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()


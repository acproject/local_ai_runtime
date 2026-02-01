import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ENGINE = None
MOCK_BYTES = 0


class HfEngine:
    def __init__(
        self,
        model_path: str,
        model_id: str,
        device: str,
        trust_remote_code: bool,
        max_new_tokens_default: int,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.model_id = model_id
        self.device = device
        self.max_new_tokens_default = max_new_tokens_default

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
        )
        self.model.to(device)
        self.model.eval()
        self.torch = torch

    def list_models(self):
        return [
            {
                "id": self.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "hf-transformers",
            }
        ]

    def _build_prompt(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def chat(self, messages, max_new_tokens=None, temperature=None, top_p=None):
        prompt = self._build_prompt(messages)

        max_new_tokens = int(max_new_tokens or self.max_new_tokens_default)
        if max_new_tokens <= 0:
            max_new_tokens = self.max_new_tokens_default

        temperature = float(temperature) if temperature is not None else 0.7
        top_p = float(top_p) if top_p is not None else 0.9

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with self.torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


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
            if ENGINE is not None:
                self._send(200, {"object": "list", "data": ENGINE.list_models()})
                return
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
            model = str(j.get("model") or (ENGINE.model_id if ENGINE is not None else "mock-model"))
            messages = j.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            try:
                if ENGINE is not None:
                    content = ENGINE.chat(
                        messages,
                        max_new_tokens=j.get("max_tokens") or j.get("max_completion_tokens"),
                        temperature=j.get("temperature"),
                        top_p=j.get("top_p"),
                    )
                    finish_reason = "stop"
                else:
                    msg = ""
                    if messages:
                        last = messages[-1]
                        if isinstance(last, dict):
                            msg = str(last.get("content") or "")
                    max_req = j.get("max_tokens") or j.get("max_completion_tokens")
                    max_req_i = None
                    try:
                        if max_req is not None:
                            max_req_i = int(max_req)
                    except Exception:
                        max_req_i = None

                    if MOCK_BYTES and MOCK_BYTES > 0:
                        cap = MOCK_BYTES
                        finish_reason = "stop"
                        if max_req_i is not None and max_req_i > 0 and cap > max_req_i:
                            cap = max_req_i
                            finish_reason = "length"
                        head = f"mock-long:n={len(messages)} bytes={cap} last={msg}\n"
                        fill = ("0123456789abcdef" * 1024) + "\n"
                        out = head
                        while len(out) < cap:
                            out += fill
                        content = out[:cap]
                    else:
                        content = f"mock:n={len(messages)} last={msg}"
                        finish_reason = "stop"
                        extra = []
                        if "temperature" in j:
                            extra.append(f"temp={j.get('temperature')}")
                        if "top_p" in j:
                            extra.append(f"top_p={j.get('top_p')}")
                        if "min_p" in j:
                            extra.append(f"min_p={j.get('min_p')}")
                        if extra:
                            content += " " + " ".join(extra)

                self._send(
                    200,
                    {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    },
                )
            except Exception as e:
                self._send(
                    500,
                    {
                        "error": {
                            "message": f"inference failed: {e}",
                            "type": "server_error",
                        }
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
    ap.add_argument("--mock-bytes", type=int, default=0)
    ap.add_argument("--hf-model-path")
    ap.add_argument("--hf-model-id", default="hf-model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    global ENGINE, MOCK_BYTES
    MOCK_BYTES = int(args.mock_bytes or 0)
    if args.hf_model_path:
        ENGINE = HfEngine(
            model_path=args.hf_model_path,
            model_id=args.hf_model_id,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens_default=args.max_new_tokens,
        )

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()


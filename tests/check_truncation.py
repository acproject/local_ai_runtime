import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple
from http.client import HTTPConnection
from urllib.parse import urlparse


def _no_proxy_env() -> None:
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        if k in os.environ:
            os.environ.pop(k, None)


def _wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            st, _, body = http_json("GET", f"{base_url}/v1/models", None, timeout_s=5.0)
            if 200 <= st < 300:
                return
            last_err = f"http {st}: {body[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.2)
    raise RuntimeError(f"runtime not ready: {last_err or 'unknown error'}")


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]],
    timeout_s: float,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], str]:
    data = None
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            headers_out = {k: v for k, v in resp.headers.items()}
            return int(getattr(resp, "status", 0) or 0), headers_out, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        headers_out = {k: v for k, v in getattr(e, "headers", {}).items()} if getattr(e, "headers", None) else {}
        return int(getattr(e, "code", 0) or 0), headers_out, body


def _sse_chat_once(
    base_url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    max_deltas: int = 0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str], Dict[str, str], int, bool]:
    u = urlparse(f"{base_url}/v1/chat/completions")
    if u.scheme != "http":
        raise RuntimeError(f"only http is supported, got {u.scheme}")
    conn = HTTPConnection(u.hostname, u.port, timeout=timeout_s)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if extra_headers:
        headers.update(extra_headers)
    conn.request("POST", u.path, body=body, headers=headers)
    resp = conn.getresponse()
    status = resp.status
    headers_out = {k: v for k, v in resp.getheaders()}
    if status < 200 or status >= 300:
        raw = resp.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat stream failed: http {status}: {raw[:500]}")

    acc = ""
    finish_reason: Optional[str] = None
    deltas = 0
    done = False
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            continue
        if not s.startswith("data:"):
            continue
        data = s[len("data:") :].strip()
        if data == "[DONE]":
            done = True
            break
        j = json.loads(data)
        if isinstance(j, dict):
            choices = j.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    fr = c0.get("finish_reason")
                    if isinstance(fr, str):
                        finish_reason = fr
                    delta = c0.get("delta")
                    if isinstance(delta, dict):
                        txt = delta.get("content")
                        if isinstance(txt, str):
                            acc += txt
                            deltas += 1
                            if max_deltas > 0 and deltas >= max_deltas:
                                break
        if max_deltas > 0 and deltas >= max_deltas:
            break
    conn.close()
    return acc, finish_reason, headers_out, deltas, done


def _make_system_prompt_bytes(n: int) -> str:
    base = (
        "You are a coding agent running in the opencode, a terminal-based coding assistant.\n"
        "Follow the user's instructions.\n"
    )
    if n <= len(base):
        return base[:n]
    fill = "AGENTS.md spec\n" * 2048
    s = base + fill
    while len(s.encode("utf-8")) < n:
        s += fill
    out = s.encode("utf-8")[:n].decode("utf-8", errors="ignore")
    return out


def run_one(
    base_url: str,
    model: str,
    system_bytes: int,
    max_tokens: int,
    stream: bool,
    prompt: str,
    timeout_s: float,
    stream_max_deltas: int,
) -> None:
    system_prompt = _make_system_prompt_bytes(system_bytes)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt,
        },
    ]
    payload: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if stream:
        payload["stream"] = True
        text, fr, _, deltas, done = _sse_chat_once(base_url, payload, timeout_s=timeout_s, max_deltas=stream_max_deltas)
        partial = False
        if stream_max_deltas > 0 and deltas >= stream_max_deltas and not done:
            partial = True
        print(
            json.dumps(
                {
                    "stream": True,
                    "system_bytes": system_bytes,
                    "max_tokens": max_tokens,
                    "completion_chars": len(text),
                    "finish_reason": fr,
                    "partial": partial,
                    "deltas": deltas,
                    "sample": text[:120],
                },
                ensure_ascii=True,
            )
        )
        return

    payload["stream"] = False
    st, _, body = http_json("POST", f"{base_url}/v1/chat/completions", payload, timeout_s=timeout_s)
    if st < 200 or st >= 300:
        raise RuntimeError(f"chat failed: http {st}: {body[:500]}")
    j = json.loads(body)
    fr = None
    txt = ""
    if isinstance(j, dict):
        choices = j.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            fr0 = choices[0].get("finish_reason")
            if isinstance(fr0, str):
                fr = fr0
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    txt = c
    print(
        json.dumps(
            {
                "stream": False,
                "system_bytes": system_bytes,
                "max_tokens": max_tokens,
                "completion_chars": len(txt),
                "finish_reason": fr,
                "sample": txt[:120],
            },
            ensure_ascii=True,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument("--model", default="")
    ap.add_argument("--runtime-exe", default="")
    ap.add_argument("--llama-model-root", default="")
    ap.add_argument("--provider", default="llama_cpp")
    ap.add_argument("--system-bytes", type=int, default=70000)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--stream-max-deltas", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument(
        "--prompt",
        default="Write a Markdown bullet list with 200 items. Each item must be ASCII and unique.",
    )
    args = ap.parse_args()

    _no_proxy_env()

    rt: Optional[subprocess.Popen] = None
    try:
        if args.runtime_exe:
            u = urlparse(args.base_url)
            if not u.hostname or not u.port:
                raise RuntimeError("--base-url must include host:port when --runtime-exe is set")
            renv = os.environ.copy()
            renv["PYTHONUNBUFFERED"] = "1"
            renv["RUNTIME_LISTEN_HOST"] = u.hostname
            renv["RUNTIME_LISTEN_PORT"] = str(u.port)
            renv["RUNTIME_PROVIDER"] = str(args.provider)
            if args.llama_model_root:
                renv["LLAMA_CPP_MODEL"] = args.llama_model_root
            rt = subprocess.Popen([args.runtime_exe], env=renv)

        _wait_ready(args.base_url, timeout_s=60.0)

        model = args.model
        if not model:
            st, _, body = http_json("GET", f"{args.base_url}/v1/models", None, timeout_s=30.0)
            if st < 200 or st >= 300:
                raise RuntimeError(f"/v1/models failed: http {st}: {body[:200]}")
            jm = json.loads(body)
            if not isinstance(jm, dict) or "data" not in jm or not isinstance(jm["data"], list) or not jm["data"]:
                raise RuntimeError("invalid /v1/models response")
            prefixed = None
            unprefixed = None
            first = None
            prefer_prefix = f"{args.provider}:" if args.provider else ""
            for it in jm["data"]:
                if not isinstance(it, dict):
                    continue
                mid = it.get("id")
                if not isinstance(mid, str) or not mid:
                    continue
                if first is None:
                    first = mid
                if prefer_prefix and prefixed is None and mid.startswith(prefer_prefix):
                    prefixed = mid
                if unprefixed is None and ":" not in mid:
                    unprefixed = mid
            picked = prefixed or unprefixed or first
            if not picked:
                raise RuntimeError("no model id found in /v1/models")
            model = picked

        if args.matrix:
            for system_bytes in [0, 2048, 8192, 20000, 70000]:
                for max_tokens in [128, 512, 2048, 4096]:
                    run_one(
                        args.base_url,
                        model,
                        system_bytes,
                        max_tokens,
                        args.stream,
                        args.prompt,
                        args.timeout_s,
                        args.stream_max_deltas,
                    )
            return 0

        run_one(
            args.base_url,
            model,
            args.system_bytes,
            args.max_tokens,
            args.stream,
            args.prompt,
            args.timeout_s,
            args.stream_max_deltas,
        )
        return 0
    finally:
        if rt is not None:
            rt.terminate()
            try:
                rt.wait(timeout=5.0)
            except Exception:
                rt.kill()
                rt.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())

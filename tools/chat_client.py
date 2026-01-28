import argparse
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


def _clean_text_for_json(s: str) -> str:
    try:
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _no_proxy_env() -> None:
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        if k in os.environ:
            os.environ.pop(k, None)


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]],
    timeout_s: float,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], str]:
    data = None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(getattr(resp, "status", 0) or 0), dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        return int(getattr(e, "code", 0) or 0), dict(getattr(e, "headers", {}) or {}), body


def _list_models(base_url: str, timeout_s: float) -> List[str]:
    st, _, body = _http_json("GET", f"{base_url}/v1/models", None, timeout_s=timeout_s)
    if st < 200 or st >= 300:
        raise RuntimeError(f"/v1/models failed: http {st}: {body[:500]}")
    j = json.loads(body)
    out: List[str] = []
    if isinstance(j, dict) and isinstance(j.get("data"), list):
        for it in j["data"]:
            if isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"]:
                out.append(it["id"])
    return out


def _pick_model_interactive(models: List[str]) -> str:
    if not models:
        raise RuntimeError("no models available")
    while True:
        for i, m in enumerate(models, start=1):
            print(f"{i:>3}. {m}")
        s = input("选择模型（序号或模型 id）：").strip()
        if not s:
            continue
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print("序号不合法")
            continue
        if s in models:
            return s
        print("模型不存在")


def _sse_chat(
    base_url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str],
) -> Tuple[str, Optional[str], bool, Optional[str]]:
    u = urlparse(f"{base_url}/v1/chat/completions")
    if u.scheme != "http":
        raise RuntimeError(f"only http is supported, got {u.scheme}")
    conn = HTTPConnection(u.hostname, u.port, timeout=timeout_s)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    h.update(headers)
    conn.request("POST", u.path, body=body, headers=h)
    resp = conn.getresponse()
    next_session_id = resp.getheader("x-session-id") or resp.getheader("X-Session-Id")
    if resp.status < 200 or resp.status >= 300:
        raw = resp.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"chat stream failed: http {resp.status}: {raw[:500]}")

    acc = ""
    finish_reason: Optional[str] = None
    done = False
    try:
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
            if not isinstance(j, dict):
                continue
            choices = j.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            c0 = choices[0]
            if not isinstance(c0, dict):
                continue
            fr = c0.get("finish_reason")
            if isinstance(fr, str):
                finish_reason = fr
            delta = c0.get("delta")
            if isinstance(delta, dict):
                txt = delta.get("content")
                if isinstance(txt, str) and txt:
                    acc += txt
                    sys.stdout.write(txt)
                    sys.stdout.flush()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return acc, finish_reason, done, next_session_id


@dataclass
class ChatState:
    base_url: str
    model: str
    session_id: str
    system_prompt: str
    stream: bool
    max_tokens: int
    timeout_s: float
    messages: List[Dict[str, str]]

    def headers(self) -> Dict[str, str]:
        return {"x-session-id": self.session_id}

    def reset_history(self) -> None:
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": _clean_text_for_json(self.system_prompt)})

    def ensure_system(self) -> None:
        if not self.system_prompt:
            return
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": _clean_text_for_json(self.system_prompt)}
            return
        self.messages.insert(0, {"role": "system", "content": _clean_text_for_json(self.system_prompt)})


def _send_one(state: ChatState, role: str, content: str) -> str:
    state.ensure_system()
    state.messages.append({"role": role, "content": _clean_text_for_json(content)})
    payload: Dict[str, Any] = {
        "model": state.model,
        "messages": state.messages,
        "max_tokens": int(state.max_tokens),
    }
    if state.stream:
        payload["stream"] = True
        text, fr, done, next_session_id = _sse_chat(state.base_url, payload, timeout_s=state.timeout_s, headers=state.headers())
        if next_session_id and next_session_id != state.session_id:
            state.session_id = next_session_id
        sys.stdout.write("\n")
        sys.stdout.flush()
        if fr is None and done:
            fr = "stop"
        state.messages.append({"role": "assistant", "content": _clean_text_for_json(text)})
        return text

    st, resp_headers, body = _http_json(
        "POST", f"{state.base_url}/v1/chat/completions", payload, timeout_s=state.timeout_s, headers=state.headers()
    )
    next_session_id = resp_headers.get("x-session-id") or resp_headers.get("X-Session-Id")
    if next_session_id and next_session_id != state.session_id:
        state.session_id = next_session_id
    if st < 200 or st >= 300:
        raise RuntimeError(f"chat failed: http {st}: {body[:500]}")
    j = json.loads(body)
    txt = ""
    if isinstance(j, dict):
        choices = j.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                txt = msg["content"]
    state.messages.append({"role": "assistant", "content": _clean_text_for_json(txt)})
    return txt


def _print_help() -> None:
    print(
        "\n".join(
            [
                "命令：",
                "  /help                      显示帮助",
                "  /models                    列出可用模型",
                "  /model <id|序号>           切换模型（不会清空历史）",
                "  /system <text>             设置 system prompt（会写入历史首条）",
                "  /stream on|off             打开/关闭流式输出",
                "  /max_tokens <n>            设置 max_tokens",
                "  /session <id>              切换 session_id（不会清空本地历史）",
                "  /clear                     清空本地对话历史（保留 system）",
                "  /msg <role> <text>         以指定 role 发送一条消息（role: user/system/assistant）",
                "  /exit                      退出",
                "",
                "直接输入文本：作为 user 消息发送。",
            ]
        )
    )


def _parse_on_off(s: str) -> Optional[bool]:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return None


def _interactive(state: ChatState) -> None:
    _print_help()
    while True:
        try:
            line = input("> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if not line.strip():
            continue

        if line.startswith("/"):
            parts = line.strip().split(" ", 2)
            cmd = parts[0].lower()
            arg1 = parts[1] if len(parts) > 1 else ""
            arg2 = parts[2] if len(parts) > 2 else ""

            if cmd in ("/exit", "/quit"):
                return
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/models":
                models = _list_models(state.base_url, timeout_s=state.timeout_s)
                for i, m in enumerate(models, start=1):
                    mark = "*" if m == state.model else " "
                    print(f"{mark} {i:>3}. {m}")
                continue
            if cmd == "/model":
                models = _list_models(state.base_url, timeout_s=state.timeout_s)
                if not arg1:
                    state.model = _pick_model_interactive(models)
                    print(f"已切换模型：{state.model}")
                    continue
                if arg1.isdigit():
                    idx = int(arg1)
                    if 1 <= idx <= len(models):
                        state.model = models[idx - 1]
                        print(f"已切换模型：{state.model}")
                        continue
                    print("序号不合法")
                    continue
                if arg1 in models:
                    state.model = arg1
                    print(f"已切换模型：{state.model}")
                    continue
                print("模型不存在")
                continue
            if cmd == "/system":
                state.system_prompt = (arg1 + (" " + arg2 if arg2 else "")).strip()
                state.ensure_system()
                print("已更新 system prompt")
                continue
            if cmd == "/stream":
                b = _parse_on_off(arg1)
                if b is None:
                    print("用法：/stream on|off")
                    continue
                state.stream = b
                print(f"stream={1 if state.stream else 0}")
                continue
            if cmd == "/max_tokens":
                try:
                    n = int(arg1)
                except Exception:
                    n = 0
                if n <= 0:
                    print("用法：/max_tokens <正整数>")
                    continue
                state.max_tokens = n
                print(f"max_tokens={state.max_tokens}")
                continue
            if cmd == "/session":
                if not arg1:
                    print("用法：/session <id>")
                    continue
                state.session_id = arg1.strip()
                print(f"session_id={state.session_id}")
                continue
            if cmd == "/clear":
                state.reset_history()
                print("已清空对话历史")
                continue
            if cmd == "/msg":
                role = arg1.strip()
                content = arg2.strip()
                if role not in ("user", "system", "assistant"):
                    print("role 只能是 user/system/assistant")
                    continue
                if not content:
                    print("用法：/msg <role> <text>")
                    continue
                print("")
                try:
                    out = _send_one(state, role, content)
                except Exception as e:
                    print(f"\n请求失败：{e}")
                    continue
                if not state.stream:
                    print(out)
                print("")
                continue

            print("未知命令，输入 /help 查看帮助")
            continue

        print("")
        try:
            out = _send_one(state, "user", line)
        except Exception as e:
            print(f"\n请求失败：{e}")
            continue
        if not state.stream:
            print(out)
        print("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument("--model", default="")
    ap.add_argument("--session-id", default="")
    ap.add_argument("--system", default="")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--once", default="")
    args = ap.parse_args()

    _no_proxy_env()

    session_id = args.session_id.strip() or f"sess-{uuid.uuid4().hex[:16]}"
    models = _list_models(args.base_url, timeout_s=float(args.timeout_s))
    model = args.model.strip()
    if not model:
        model = models[0] if models else ""
    if model not in models and models:
        model = models[0]

    state = ChatState(
        base_url=args.base_url.rstrip("/"),
        model=model,
        session_id=session_id,
        system_prompt=args.system or "",
        stream=bool(args.stream),
        max_tokens=int(args.max_tokens),
        timeout_s=float(args.timeout_s),
        messages=[],
    )
    state.reset_history()

    print(f"base_url={state.base_url}")
    print(f"session_id={state.session_id}")
    print(f"model={state.model}")
    print(f"stream={1 if state.stream else 0}")
    print(f"max_tokens={state.max_tokens}")

    if args.once:
        t0 = time.time()
        out = _send_one(state, "user", args.once)
        if not state.stream:
            print(out)
        dt = time.time() - t0
        print(f"\n(done in {dt:.2f}s, chars={len(out)})")
        return 0

    _interactive(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

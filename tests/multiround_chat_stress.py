import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 300.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], str]:
    data = None
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), dict(resp.headers.items()), body
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return int(e.code), dict(e.headers.items()) if e.headers else {}, body
    except Exception as e:
        return 0, {}, str(e)


def wait_ready(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        st, _, body = http_json("GET", f"{base_url}/v1/models", None, timeout_s=5.0)
        if 200 <= st < 300:
            return
        last_err = (st, body)
        time.sleep(0.3)
    raise RuntimeError(f"runtime not ready: {last_err}")


def parse_models(body: str) -> List[str]:
    j = json.loads(body)
    data = j.get("data") or []
    out: List[str] = []
    for it in data:
        if isinstance(it, dict):
            mid = it.get("id")
            if isinstance(mid, str) and mid:
                out.append(mid)
    return out


def choose_model(models: List[str], preferred: Optional[str], contains: Optional[str]) -> str:
    if preferred:
        return preferred
    if contains:
        needle = contains.lower()
        candidates = [m for m in models if needle in m.lower()]
        if candidates:
            candidates.sort(key=lambda x: (0 if x.startswith("llama_cpp:") else 1, len(x), x))
            return candidates[0]
    if not models:
        raise RuntimeError("no models returned by /v1/models")
    models = sorted(models)
    return models[0]


@dataclass
class ProcMem:
    working_set_mb: float
    private_mb: float


def get_process_memory_mb_windows(pid: int) -> Optional[ProcMem]:
    if os.name != "nt":
        return None

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_VM_READ = 0x0010

    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    if not ok:
        return None
    return ProcMem(
        working_set_mb=float(counters.WorkingSetSize) / (1024.0 * 1024.0),
        private_mb=float(counters.PrivateUsage) / (1024.0 * 1024.0),
    )


def find_pid_by_listen_port_windows(port: int) -> Optional[int]:
    if os.name != "nt":
        return None
    try:
        p = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if p.returncode != 0:
        return None

    want_suffix = f":{port}"
    for raw_line in (p.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        state = parts[3]
        pid_str = parts[4]
        if state != "LISTENING":
            continue
        if not local_addr.endswith(want_suffix):
            continue
        try:
            return int(pid_str)
        except Exception:
            continue
    return None


def start_process(argv: List[str], env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stop_process(p: Optional[subprocess.Popen]) -> None:
    if not p:
        return
    try:
        p.terminate()
        p.wait(timeout=5)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def drain_output_nonblocking(p: subprocess.Popen, max_lines: int = 200) -> None:
    if not p.stdout:
        return
    lines = 0
    while lines < max_lines:
        line = p.stdout.readline()
        if not line:
            break
        sys.stdout.write(line)
        lines += 1


def chat_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_s: float = 300.0,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, str]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if extra_payload:
        payload.update(extra_payload)
    st, resp_headers, body = http_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        payload,
        timeout_s=timeout_s,
        extra_headers=extra_headers,
    )
    if st < 200 or st >= 300:
        raise RuntimeError(f"chat failed: http {st}: {body}")
    j = json.loads(body)
    choices = j.get("choices") or []
    if not choices or not isinstance(choices, list):
        raise RuntimeError(f"invalid response: {body}")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        raise RuntimeError(f"invalid response: {body}")
    return content, resp_headers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--model")
    ap.add_argument("--model-contains", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--turns-per-round", type=int, default=2)
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--pid", type=int, default=0, help="If set, use this PID for memory printing (Windows).")
    ap.add_argument("--content-format", choices=["string", "array"], default="string")
    ap.add_argument("--session-mode", choices=["client", "server"], default="client")
    ap.add_argument("--runtime-exe", help="If set, start local-ai-runtime.exe for this run.")
    ap.add_argument("--provider", default="llama_cpp", help="Only used when --runtime-exe is set.")
    ap.add_argument("--llama-model-root", help="Only used when --runtime-exe is set.")
    ap.add_argument("--lmdeploy-host", help="Only used when --runtime-exe is set.")
    ap.add_argument("--ollama-host", help="Only used when --runtime-exe is set.")
    ap.add_argument("--mnn-host", help="Only used when --runtime-exe is set.")
    ap.add_argument("--llama-flash-attn", default="disabled", help="Only used when --runtime-exe is set.")
    ap.add_argument("--llama-n-batch", default="256", help="Only used when --runtime-exe is set.")
    ap.add_argument("--llama-n-ubatch", default="128", help="Only used when --runtime-exe is set.")
    ap.add_argument("--llama-unload-after-chat", default="", help="Only used when --runtime-exe is set. e.g. 1/0")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = env["NO_PROXY"]
    os.environ["no_proxy"] = env["no_proxy"]
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        if k in os.environ:
            os.environ.pop(k, None)

    rt: Optional[subprocess.Popen] = None
    mem_pid: Optional[int] = None
    try:
        if args.runtime_exe:
            renv = env.copy()
            renv["RUNTIME_LISTEN_HOST"] = args.host
            renv["RUNTIME_LISTEN_PORT"] = str(args.port)
            renv["RUNTIME_PROVIDER"] = args.provider
            if args.llama_model_root:
                renv["LLAMA_CPP_MODEL"] = args.llama_model_root
            if args.lmdeploy_host:
                renv["LMDEPLOY_HOST"] = args.lmdeploy_host
            if args.ollama_host:
                renv["OLLAMA_HOST"] = args.ollama_host
            if args.mnn_host:
                renv["MNN_HOST"] = args.mnn_host

            if args.provider == "llama_cpp":
                renv["LLAMA_CPP_FLASH_ATTN"] = str(args.llama_flash_attn)
                renv["LLAMA_CPP_N_BATCH"] = str(args.llama_n_batch)
                renv["LLAMA_CPP_N_UBATCH"] = str(args.llama_n_ubatch)
                if args.llama_unload_after_chat != "":
                    renv["LLAMA_CPP_UNLOAD_AFTER_CHAT"] = str(args.llama_unload_after_chat)

            rt = start_process([args.runtime_exe], env=renv)
            wait_ready(base_url, timeout_s=60.0)
            mem_pid = rt.pid
        else:
            wait_ready(base_url, timeout_s=30.0)
            if args.pid and args.pid > 0:
                mem_pid = int(args.pid)
            else:
                mem_pid = find_pid_by_listen_port_windows(args.port)

        st, _, body = http_json("GET", f"{base_url}/v1/models", None, timeout_s=30.0)
        if st < 200 or st >= 300:
            raise RuntimeError(f"/v1/models failed: http {st}: {body}")
        models = parse_models(body)
        model = choose_model(models, args.model, args.model_contains)

        print(f"base_url={base_url}")
        print(f"model={model}")
        if mem_pid is not None:
            print(f"runtime_pid={mem_pid}")

        def msg(role: str, content: str) -> Dict[str, Any]:
            if args.content_format == "array":
                return {"role": role, "content": [{"type": "text", "text": content}]}
            return {"role": role, "content": content}

        def content_text(v: Any) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                out = ""
                for part in v:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        out += part["text"]
                return out
            return ""

        for r in range(args.rounds):
            mem0 = get_process_memory_mb_windows(mem_pid) if mem_pid is not None else None
            t0 = time.time()
            last_assistant = ""

            if args.session_mode == "client":
                messages: List[Dict[str, Any]] = [
                    msg("system", "你是一个严谨的助手。"),
                    msg("user", f"第{r + 1}轮：用一句话概括‘内存回收’的含义。"),
                ]

                for t in range(args.turns_per_round):
                    last_assistant, _ = chat_once(base_url, model, messages, timeout_s=300.0)
                    messages.append(msg("assistant", last_assistant))
                    if t + 1 < args.turns_per_round:
                        messages.append(msg("user", "基于上一句，再补充一个要点（不要重复）。"))
            else:
                session_id: Optional[str] = None
                for t in range(args.turns_per_round):
                    if t == 0:
                        messages = [
                            msg("system", "你是一个严谨的助手。"),
                            msg("user", f"第{r + 1}轮：用一句话概括‘内存回收’的含义。"),
                        ]
                        extra_payload = {"use_server_history": True}
                    else:
                        messages = [msg("user", "基于上一句，再补充一个要点（不要重复）。")]
                        extra_payload = None

                    extra_headers = None
                    if session_id:
                        extra_headers = {"x-session-id": session_id}

                    last_assistant, resp_headers = chat_once(
                        base_url,
                        model,
                        messages,
                        timeout_s=300.0,
                        extra_headers=extra_headers,
                        extra_payload=extra_payload,
                    )
                    if session_id is None:
                        session_id = resp_headers.get("x-session-id") or resp_headers.get("X-Session-Id")

            dt_ms = (time.time() - t0) * 1000.0
            mem1 = get_process_memory_mb_windows(mem_pid) if mem_pid is not None else None
            assistant_len = len(content_text(last_assistant))

            def fmt_mem(m: Optional[ProcMem]) -> str:
                if m is None:
                    return "n/a"
                return f"ws={m.working_set_mb:.1f}MB private={m.private_mb:.1f}MB"

            print(f"round={r + 1}/{args.rounds} dt={dt_ms:.0f}ms assistant_chars={assistant_len} mem_before={fmt_mem(mem0)} mem_after={fmt_mem(mem1)}")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

            if rt is not None and rt.poll() is not None:
                drain_output_nonblocking(rt, max_lines=200)
                raise RuntimeError(f"runtime exited early with code {rt.returncode}")

        return 0
    finally:
        stop_process(rt)


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def http_json(url, payload, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, dict(e.headers), body


def http_post(url):
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, dict(e.headers), body


def http_get(url):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, dict(e.headers), body


def wait_ready(url, timeout_s=5, method="GET"):
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            if method == "POST":
                req = urllib.request.Request(url, data=b"{}", method="POST")
                req.add_header("Content-Type", "application/json")
            else:
                req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                resp.read()
                return True
        except Exception:
            time.sleep(0.05)
    return False


def start_process(argv, env=None, stdout=None, stderr=None):
    if stdout is None:
        stdout = subprocess.DEVNULL
    if stderr is None:
        stderr = subprocess.DEVNULL
    p = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
    return p


def stop_process(p):
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=2)
        return
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def extract_chat_content(body: str) -> str:
    j = json.loads(body)
    return str(j["choices"][0]["message"]["content"])


def extract_float_token(s: str, key: str):
    needle = key + "="
    i = s.find(needle)
    if i < 0:
        return None
    j = i + len(needle)
    k = j
    while k < len(s) and s[k] not in (" ", "\n", "\r", "\t", ",", ";"):
        k += 1
    try:
        return float(s[j:k])
    except Exception:
        return None


def assert_close(actual, expected, tol=1e-3):
    assert actual is not None
    assert abs(float(actual) - float(expected)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="./build/local-ai-runtime")
    ap.add_argument("--mcp-port", type=int, default=19001)
    ap.add_argument("--openai-port", type=int, default=19002)
    ap.add_argument("--runtime-port", type=int, default=18080)
    ap.add_argument("--workspace-root", default=os.getcwd())
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    mcp = start_process([sys.executable, "tools/mock_mcp_server.py", "--port", str(args.mcp_port), "--mode", "lsp"], env=env)
    openai = start_process([sys.executable, "tools/mock_openai_server.py", "--port", str(args.openai_port)], env=env)
    try:
        if not wait_ready(f"http://127.0.0.1:{args.mcp_port}/", timeout_s=3, method="POST"):
            raise RuntimeError("mock mcp server did not start")
        if not wait_ready(f"http://127.0.0.1:{args.openai_port}/v1/models", timeout_s=3, method="GET"):
            raise RuntimeError("mock openai server did not start")

        renv = env.copy()
        renv["RUNTIME_LISTEN_HOST"] = "127.0.0.1"
        renv["RUNTIME_LISTEN_PORT"] = str(args.runtime_port)
        renv["MCP_HOSTS"] = f"http://127.0.0.1:{args.mcp_port}/"
        renv["RUNTIME_WORKSPACE_ROOT"] = args.workspace_root
        renv["RUNTIME_PROVIDER"] = "mnn"
        renv["MNN_HOST"] = f"http://127.0.0.1:{args.openai_port}"
        renv["LMDEPLOY_HOST"] = f"http://127.0.0.1:{args.openai_port}"
        store_dir = os.path.join(
            args.workspace_root,
            f".runtime_session_store_dir_{args.runtime_port}_{int(time.time() * 1000)}",
        )
        store_file = os.path.join(store_dir, "sessions.json")
        renv["RUNTIME_SESSION_STORE"] = store_dir
        renv["RUNTIME_SESSION_STORE_NAMESPACE"] = "regression"
        if os.name == "nt":
            rt_path = os.path.abspath(args.runtime)
            rt_dir = os.path.dirname(rt_path)
            base_dir = os.path.dirname(rt_dir)
            dll_dirs = [
                rt_dir,
                os.path.join(base_dir, "bin", "Release"),
                os.path.join(base_dir, "bin", "Debug"),
            ]
            dll_dirs = [p for p in dll_dirs if os.path.isdir(p)]
            if dll_dirs:
                renv["PATH"] = os.pathsep.join(dll_dirs + [renv.get("PATH", "")])
        rt_log_path = os.path.join(args.workspace_root, f".runtime_regression_{args.runtime_port}.log")
        rt_log = open(rt_log_path, "wb")
        rt = start_process([args.runtime], env=renv, stdout=rt_log, stderr=rt_log)
        try:
            if not wait_ready(f"http://127.0.0.1:{args.runtime_port}/v1/models", timeout_s=3, method="GET"):
                rt_log.flush()
                msg = "runtime did not start"
                if rt.poll() is not None:
                    msg += f" (exited={rt.returncode})"
                try:
                    with open(rt_log_path, "rb") as f:
                        tail = f.read()[-4000:].decode("utf-8", errors="replace")
                    if tail.strip():
                        msg += "\n\n--- runtime log tail ---\n" + tail
                except Exception:
                    pass
                raise RuntimeError(msg)

            st, _, body = http_get(f"http://127.0.0.1:{args.runtime_port}/v1/models")
            assert st == 200
            jm = json.loads(body)
            ids = [x.get("id") for x in jm.get("data", []) if isinstance(x, dict)]
            assert "mock-model" in ids
            assert "lmdeploy:mock-model" in ids

            st, _, body = http_post(f"http://127.0.0.1:{args.runtime_port}/internal/refresh_mcp_tools")
            assert st == 200
            j = json.loads(body)
            assert j.get("ok") is True
            assert j.get("servers") == 1
            assert j.get("registered", 0) >= 5

            payload = {
                "model": "fake-tool",
                "messages": [{"role": "user", "content": "请使用 ide.read_file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ide.read_file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                        },
                    }
                ],
            }
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            assert "TOOL_RESULT ide.read_file" in body

            payload = {
                "model": "fake-tool",
                "trace": True,
                "planner": {"enabled": True, "max_plan_steps": 2, "max_rewrites": 1},
                "messages": [{"role": "user", "content": "bad_args ide.hover"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ide.hover",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "uri": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "character": {"type": "integer"},
                                },
                                "required": ["uri", "line", "character"],
                            },
                        },
                    }
                ],
            }
            st, headers, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            trace = headers.get("x-runtime-trace", "")
            assert '"used_planner":true' in trace
            assert '"plan_rewrites":1' in trace
            assert "TOOL_RESULT ide.hover" in body

            payload = {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            assert "mock:n=1 last=hi" in body

            payload = {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "min_p": 0.01,
            }
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            content = extract_chat_content(body)
            assert_close(extract_float_token(content, "temp"), 0.7)
            assert_close(extract_float_token(content, "top_p"), 0.9)
            assert_close(extract_float_token(content, "min_p"), 0.01)

            payload = {
                "model": "glm-mock",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.1,
                "top_p": 0.2,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ide.read_file",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                        },
                    }
                ],
            }
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            content = extract_chat_content(body)
            assert_close(extract_float_token(content, "temp"), 0.7)
            assert_close(extract_float_token(content, "top_p"), 1.0)

            payload = {"model": "lmdeploy:mock-model", "messages": [{"role": "user", "content": "hi2"}]}
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            assert "mock:n=1 last=hi2" in body

            payload = {"model": "mock-model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            assert "mock:n=1 last=hi" in body

            payload = {"model": "mock-model", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}
            st, headers, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 200
            sid = headers.get("x-session-id") or headers.get("X-Session-Id")
            assert sid

            payload = {"model": "mock-model", "messages": [{"role": "user", "content": "next"}]}
            st, _, body = http_json(
                f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions",
                payload,
                headers={"x-session-id": sid},
            )
            assert st == 200
            assert "mock:n=4 last=next" in body
            assert os.path.exists(store_file)
            with open(store_file, "rb") as f:
                store = json.loads(f.read().decode("utf-8", errors="replace"))
            key = f"regression:{sid}"
            assert isinstance(store, dict)
            assert isinstance(store.get("sessions"), dict)
            assert key in store["sessions"]
            sess = store["sessions"][key]
            assert isinstance(sess, dict)
            assert sess.get("session_id") == sid
            assert isinstance(sess.get("turns"), list)
            assert len(sess["turns"]) >= 2

            payload = {"model": "mock-model", "input": "x"}
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/embeddings", payload)
            assert st == 200
            assert '"embedding":[0.1,0.2,0.3]' in body.replace(" ", "")

            payload = {"model": "llama_cpp:any", "messages": [{"role": "user", "content": "hi"}]}
            st, _, body = http_json(f"http://127.0.0.1:{args.runtime_port}/v1/chat/completions", payload)
            assert st == 502
            assert "llama_cpp:" in body

        finally:
            stop_process(rt)
            try:
                rt_log.close()
            except Exception:
                pass
            try:
                if "store_dir" in locals() and os.path.isdir(store_dir):
                    for name in os.listdir(store_dir):
                        try:
                            os.remove(os.path.join(store_dir, name))
                        except Exception:
                            pass
                    try:
                        os.rmdir(store_dir)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if os.path.exists(rt_log_path):
                    os.remove(rt_log_path)
            except Exception:
                pass
    finally:
        stop_process(mcp)
        stop_process(openai)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

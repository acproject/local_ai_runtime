import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def http_json(url, payload, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, dict(resp.headers), body


def http_post(url):
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, dict(resp.headers), body


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


def start_process(argv, env=None):
    p = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="./build/local-ai-runtime")
    ap.add_argument("--mcp-port", type=int, default=19001)
    ap.add_argument("--runtime-port", type=int, default=18080)
    ap.add_argument("--workspace-root", default=os.getcwd())
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    mcp = start_process([sys.executable, "tools/mock_mcp_server.py", "--port", str(args.mcp_port), "--mode", "lsp"], env=env)
    try:
        if not wait_ready(f"http://127.0.0.1:{args.mcp_port}/", timeout_s=3, method="POST"):
            raise RuntimeError("mock mcp server did not start")

        renv = env.copy()
        renv["RUNTIME_LISTEN_HOST"] = "127.0.0.1"
        renv["RUNTIME_LISTEN_PORT"] = str(args.runtime_port)
        renv["MCP_HOSTS"] = f"http://127.0.0.1:{args.mcp_port}/"
        renv["RUNTIME_WORKSPACE_ROOT"] = args.workspace_root
        rt = start_process([args.runtime], env=renv)
        try:
            if not wait_ready(f"http://127.0.0.1:{args.runtime_port}/v1/models", timeout_s=3, method="GET"):
                raise RuntimeError("runtime did not start")

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

        finally:
            stop_process(rt)
    finally:
        stop_process(mcp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

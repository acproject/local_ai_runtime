import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def tools_for_mode(mode: str):
    if mode == "echo":
        return [
            {
                "name": "mcp.echo",
                "title": "Echo",
                "description": "Echo back text",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ]
    if mode == "lsp":
        return [
            {
                "name": "fs.read_file",
                "title": "Read File",
                "description": "Read a UTF-8 text file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "fs.search",
                "title": "Search",
                "description": "Search text in files under a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["path", "query"],
                },
            },
            {
                "name": "lsp.hover",
                "title": "Hover",
                "description": "Return hover info for a position",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["uri", "line", "character"],
                },
            },
            {
                "name": "lsp.definition",
                "title": "Definition",
                "description": "Return definition location for a position",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["uri", "line", "character"],
                },
            },
            {
                "name": "lsp.diagnostics",
                "title": "Diagnostics",
                "description": "Return diagnostics for a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"uri": {"type": "string"}},
                    "required": ["uri"],
                },
            },
        ]
    raise ValueError(f"unknown mode: {mode}")


def call_for_mode(mode: str, name: str, arguments: dict):
    if mode == "echo" and name == "mcp.echo":
        text = str(arguments.get("text", ""))
        return {"content": [{"type": "text", "text": text}], "isError": False}
    if mode == "lsp" and name == "lsp.hover":
        uri = arguments.get("uri")
        line = arguments.get("line")
        character = arguments.get("character")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"hover({uri}:{line}:{character})",
                }
            ],
            "isError": False,
        }
    if mode == "lsp" and name == "lsp.definition":
        uri = arguments.get("uri")
        return {
            "content": [{"type": "text", "text": f"definition({uri})"}],
            "isError": False,
        }
    if mode == "lsp" and name == "lsp.diagnostics":
        uri = str(arguments.get("uri") or "")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"diagnostics({uri})",
                }
            ],
            "isError": False,
        }
    if mode == "lsp" and name == "fs.read_file":
        path = str(arguments.get("path") or "")
        try:
            data = open(path, "r", encoding="utf-8", errors="replace").read()
        except Exception as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        return {"content": [{"type": "text", "text": data[:2000]}], "isError": False}
    if mode == "lsp" and name == "fs.search":
        import os

        base = str(arguments.get("path") or "")
        query = str(arguments.get("query") or "")
        max_results = int(arguments.get("max_results") or 20)
        hits = []
        for root, _, files in os.walk(base):
            for fn in files:
                if len(hits) >= max_results:
                    break
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if query in line:
                                hits.append({"path": fp, "line": i, "text": line.strip()[:200]})
                                break
                except Exception:
                    continue
            if len(hits) >= max_results:
                break
        return {"content": [{"type": "text", "text": json.dumps(hits)}], "isError": False}
    return {"content": [{"type": "text", "text": "unknown tool"}], "isError": True}


class Handler(BaseHTTPRequestHandler):
    def _auth_snapshot(self):
        out = {}
        for k in ["authorization", "api-key", "x-api-key", "api_key"]:
            v = self.headers.get(k)
            if v is None:
                continue
            s = str(v)
            out[k] = {"present": True, "len": len(s)}
        return out

    def do_POST(self):
        snap = self._auth_snapshot()
        if snap:
            keys = ",".join(sorted(snap.keys()))
            print(f"[mock-mcp] POST auth_keys={keys}")
        else:
            print("[mock-mcp] POST auth=none")

        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)
        req_id = req.get("id")
        method = req.get("method")

        if method == "initialize":
            result = {"capabilities": {"tools": {}}, "received_auth": snap}
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.server.tools, "received_auth": snap},
            }
        elif method == "tools/call":
            delay_ms = int(getattr(self.server, "delay_ms", 0) or 0)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            params = req.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            result = call_for_mode(self.server.mode, name, args)
            result["received_auth"] = snap
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "method not found"},
            }

        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=["echo", "lsp"], default="echo")
    ap.add_argument("--delay-ms", type=int, default=0)
    args = ap.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    srv.mode = args.mode
    srv.tools = tools_for_mode(args.mode)
    srv.delay_ms = int(args.delay_ms or 0)
    srv.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"

curl_common=(--noproxy '*' -sS)

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

printf '%s\n' "GET ${BASE_URL}/v1/models"
curl "${curl_common[@]}" "${BASE_URL}/v1/models" >/dev/null || fail "runtime not reachable"

printf '%s\n' "POST ${BASE_URL}/internal/refresh_mcp_tools"
refresh="$(curl "${curl_common[@]}" -X POST "${BASE_URL}/internal/refresh_mcp_tools" || true)"
printf '%s\n' "$refresh"
printf '\n'

printf '%s\n' "POST ${BASE_URL}/v1/chat/completions (ide.read_file)"
read_file_req='{"model":"fake-tool","messages":[{"role":"user","content":"请使用 ide.read_file"}],"tools":[{"type":"function","function":{"name":"ide.read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'
read_file_resp="$(curl "${curl_common[@]}" -H 'Content-Type: application/json' -d "${read_file_req}" "${BASE_URL}/v1/chat/completions")"
printf '%s\n' "$read_file_resp"
printf '\n'

printf '%s\n' "POST ${BASE_URL}/v1/chat/completions (planner bad_args rewrite + trace)"
planner_req='{"model":"fake-tool","trace":true,"planner":{"enabled":true,"max_plan_steps":2,"max_rewrites":1},"messages":[{"role":"user","content":"bad_args ide.hover"}],"tools":[{"type":"function","function":{"name":"ide.hover","parameters":{"type":"object","properties":{"uri":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"}},"required":["uri","line","character"]}}}]}'
curl "${curl_common[@]}" -i -H 'Content-Type: application/json' -d "${planner_req}" "${BASE_URL}/v1/chat/completions" | sed -n '1,20p'
printf '\n'

printf '%s\n' "OK"

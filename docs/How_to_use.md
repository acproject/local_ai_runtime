### 在windows下使用
```ps
$Env:PATH='D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\bin\Release;' + $Env:PATH; $Env:RUNTIME_LISTEN_HOST='127.0.0.1'; $Env:RUNTIME_LISTEN_PORT='18080'; $Env:RUNTIME_PROVIDER='lmdeploy'; $Env:LMDEPLOY_HOST='http://127.0.0.1:23333'; $Env:NO_PROXY='127.0.0.1,localhost'; $Env:no_proxy='127.0.0.1,localhost'; & 'D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\Release\local-ai-runtime.exe'
```
- 启动多个Provider
```ps
$Env:PATH='D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\bin\Release;' + $Env:PATH;$Env:RUNTIME_LISTEN_HOST='127.0.0.1';$Env:RUNTIME_LISTEN_PORT='18081';$Env:RUNTIME_PROVIDER='llama_cpp';$Env:LLAMA_CPP_MODEL='M:\llm_models';$Env:OLLAMA_HOST='http://127.0.0.1:11434';$Env:LMDEPLOY_HOST='http://127.0.0.1:23333';$Env:NO_PROXY='127.0.0.1,localhost';$Env:no_proxy='127.0.0.1,localhost';& 'D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\Release\local-ai-runtime.exe'

## 支持cuda可以使用下面的内容：
$Env:PATH=' D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64-cuda\bin\Release;' + $Env:PATH;$Env:RUNTIME_LISTEN_HOST='127.0.0.1';$Env:RUNTIME_LISTEN_PORT='18081';$Env:RUNTIME_PROVIDER='llama_cpp';$Env:LLAMA_CPP_MODEL='M:\llm_models';$Env:OLLAMA_HOST='http://127.0.0.1:11434';$Env:LMDEPLOY_HOST='http://127.0.0.1:23333';$Env:NO_PROXY='127.0.0.1,localhost';$Env:no_proxy='127.0.0.1,localhost';& ' D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64-cuda\bin\Release\local-ai-runtime.exe'
```
* 注意$Env:RUNTIME_LISTEN_PORT='18080'一定要和CloudToLocal中本地的local_agent.exe中，参数--openai=http://127.0.0.1:18080，端口一致

* 注意如果使用本地服务，还需要下面的设置：
```ps
PowerShell 里加这些环境变量（会覆盖默认）：
- $env:LLAMA_CPP_FLASH_ATTN="disabled"
- $env:LLAMA_CPP_N_BATCH="256"
- $env:LLAMA_CPP_N_UBATCH="128"
```

### API的使用方式

opencode 侧建议把 OpenAI Base URL 指向以下任意一层（路径保持一致）：

- 直接连 runtime（local-ai-runtime.exe）：`http://<runtime_host>:<runtime_port>`（例：`http://127.0.0.1:18080`）
- 连 CloudToLocal 的 http_proxy（cloud_server.exe）：`http://<cloud_server_host>:8081`（例：`http://127.0.0.1:8081`）
- 连 llm-servers 对外域名：`https://www.owiseman.com`

#### 鉴权（token）传递

当 llm-servers 配置了 `LLM_V1_PROXY_TOKEN` 或 `OPENAI_PROXY_TOKEN` 时需要鉴权，推荐统一使用：

```http
Authorization: Bearer <token>
```

也支持：

- Header：`api-key: <token>`
- Query：`?api-key=<token>` 或 `?api_key=<token>`
- 仅 `/v1/chat/completions`：JSON body 里也可放 `api-key` / `api_key`（llm-servers 会读取后移除再转发）

#### 模型命名规则

`/v1/models` 返回的 `data[].id` 可能带 provider 前缀：

- `ollama:<id>` / `lmdeploy:<id>` / `mnn:<id>` / `llama_cpp:<id>`

也可能不带前缀（表示 runtime 的默认 provider，由 `RUNTIME_PROVIDER` 决定）。opencode 侧应当直接使用返回的 `id` 原样作为请求里的 `model`。

#### GET /v1/models（模型列表）

```bash
curl -s http://127.0.0.1:18080/v1/models
```

返回示例（简化）：

```json
{
  "object": "list",
  "data": [
    {"id":"ollama:qwen2.5:latest","object":"model","created":1700000000,"owned_by":"ollama"},
    {"id":"llama_cpp:default","object":"model","created":1700000000,"owned_by":"llama_cpp"}
  ]
}
```

#### POST /v1/chat/completions（聊天）

非流式（`stream=false`）：

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"ollama:qwen2.5:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":false}"
```

流式（`stream=true`，SSE）：

```bash
curl -N http://127.0.0.1:18080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"ollama:qwen2.5:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"讲个笑话\"}],\"stream\":true}"
```

SSE 输出为多条：

- `data: <json>\n\n`
- 结束：`data: [DONE]\n\n`

opencode 侧增量拼接规则：

- 普通文本：拼接 `choices[0].delta.content`
- 工具调用：如果出现 `choices[0].delta.tool_calls`，其中 `function.arguments` 会分片多次下发，需要按 `id/index` 聚合拼接 arguments 字符串

#### POST /v1/embeddings（向量）

```bash
curl -s http://127.0.0.1:18080/v1/embeddings ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"ollama:qwen2.5:latest\",\"input\":\"hello\"}"
```

#### POST /v1/responses（最小子集，非流式）

```bash
curl -s http://127.0.0.1:18080/v1/responses ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"ollama:qwen2.5:latest\",\"input\":\"say hi\"}"
```

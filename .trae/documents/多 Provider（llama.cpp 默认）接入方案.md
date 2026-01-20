## 需求拆解（你这次要达成什么）
1. 把 llama.cpp 作为默认推理引擎（不再默认依赖 Ollama）。
2. 基于 llama.cpp 在项目内实现“自研推理服务/流程”（为了自定义特殊流程）。
3. 新增 MNN / LMDeploy 两个 Provider，让用户可选择不同后端。
4. 保持现有 OpenAI 兼容层（/v1/models、/v1/chat/completions、/v1/embeddings、/v1/responses）与 Tool Loop/Planner/MCP 功能不回退。

## 现状调研结论（需要先重构才能多 Provider）
- 当前没有 Provider 抽象：路由层 OpenAiRouter 直接持有 `OllamaProvider*`，ToolLoop/Planner 也显式依赖 `OllamaProvider*`。
  - 代码入口：[openai_router.hpp](file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/openai_router.hpp)、[openai_router.cpp](file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/openai_router.cpp)、[ollama_provider.hpp](file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/ollama_provider.hpp)、[main.cpp](file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp)
- 因此“新增 Provider”不是加几个类就行，必须先把路由层 + tool loop/planner 的模型调用抽象化。

## 设计决策（最合理的默认实现路径）
### A. llama.cpp 采用“进程内（in-proc）集成”作为默认 Provider
- 这最符合“保证特殊流程可自定义”：你能控制加载/缓存、采样策略、KV cache、grammar/json 约束、logits processor、tokens 预算、甚至把某些 workflow 固化。
- 相比“再起一个 llama.cpp HTTP server”：
  - 优点：少一次 HTTP/JSON 序列化、低延迟、控制力强。
  - 风险：需要处理并发、模型生命周期、跨平台构建（但我们本来就是 CMake+C++20）。

### B. MNN / LMDeploy 先走“外部服务适配（HTTP）”
- 因为这两个后端在 C++ 侧集成成本不确定（依赖、ABI、GPU 选择等），先做 Provider 适配层最稳：
  - LMDeploy 通常可提供 OpenAI 兼容接口（或至少能封装成 OpenAI 兼容）。
  - MNN 可先支持一个你自研的 MNN-serving（或用户自备服务）并通过 runtime 适配。
- 后续如果要“进程内集成”，再加编译开关与依赖即可，不影响接口。

## 需要新增/调整的核心抽象
### 1) Provider 接口（IProvider）
- 新增 `src/providers/provider.hpp`：定义 provider-agnostic 的请求/响应结构。
  - `ListModels()`
  - `Embeddings(model, input)`（可选 capability）
  - `ChatOnce(ChatRequest)` + `ChatStream(ChatRequest, callbacks)`（可选 capability）
- 把现有 `OllamaChatRequest/Response` 改为通用 `ChatRequest/ChatResponse`（避免上层被 Ollama 类型锁死）。

### 2) ProviderRegistry + 选择规则
- 新增 `ProviderRegistry`：
  - `default_provider`（默认 llama_cpp）
  - `Get(provider_name)`
  - `Resolve(model)`（支持 `provider:model` 前缀、或按配置映射 model->provider）
- 目标：
  - `/v1/chat/completions` / `/v1/embeddings` / `/v1/responses` 都能先选 provider，再执行。
  - ToolLoop/Planner 内部“二次调用模型”必须复用同一个 provider（避免计划阶段用 A、执行阶段用 B）。

## 具体实施步骤（落地顺序）
### Step 1：抽象化（让多 Provider 成为可能）
1. 新增 provider 接口与 registry。
2. 改造 OpenAiRouter：把 `OllamaProvider*` 替换成 `ProviderRegistry*` 或 `IProvider*`。
3. 改造 ToolLoop/Planner：把 `ChatOnceText/RunToolLoop/RunPlanner` 的 `OllamaProvider*` 参数换成 `IProvider&`。
4. 先把现有 `OllamaProvider` 迁移为 `OllamaHttpProvider : IProvider`（逻辑基本复用）。
- 验收：现有所有接口在 `provider=ollama` 下行为不变，回归脚本仍能跑通。

### Step 2：llama.cpp Provider（默认）
1. 引入 llama.cpp：
   - 推荐用 CMake FetchContent（或 git submodule）拉取 llama.cpp 源码并编译为静态库。
2. 新增 `LlamaCppProvider : IProvider`（进程内）：
   - 支持 `ListModels`（基于配置的模型目录扫描/单模型注册）。
   - 支持 `ChatOnce/ChatStream`：
     - 把 OpenAI messages 转成 prompt（支持 system/user/assistant role）。
     - 支持停止词、最大 tokens、温度、top_p 等（从请求体读取，先实现子集）。
   - 可选：`Embeddings`（llama.cpp 有 embedding 模式时再加；或者先 capability=false）。
3. 让默认 provider 指向 llama_cpp（配置默认值变更）。
- 验收：不依赖 Ollama，也能完成 chat/completions 的 tool loop/planner 基本闭环。

### Step 3：MNN Provider（HTTP 适配 v0）
1. 新增 `MnnHttpProvider : IProvider`：
   - 通过配置的 `MNN_HOST`（或 base_url）调用自研/第三方 MNN serving。
   - 先实现 `ChatOnce`，流式与 embeddings 作为 capability=false。
2. Model 列表可先从配置静态提供（或调用服务 `/models`）。
- 验收：`provider=mnn` 能跑通基础非流式 chat。

### Step 4：LMDeploy Provider（HTTP 适配 v0）
1. 新增 `LmDeployProvider : IProvider`：
   - 优先适配 LMDeploy 的 OpenAI 兼容接口（若存在），直接复用 OpenAI ChatCompletions/Embeddings schema。
   - 若不完全兼容，则写一个薄适配把 LMDeploy 返回映射到内部 `ChatResponse`。
2. capability 先保证 `ChatOnce`，再逐步补 `ChatStream/Embeddings`。

### Step 5：配置与对外选择方式
1. `config.hpp/cpp` 增加：
   - `RUNTIME_PROVIDER`（默认 `llama_cpp`）
   - 各 provider 的 endpoint：`OLLAMA_HOST`、`MNN_HOST`、`LMDEPLOY_HOST` 等
   - llama.cpp 模型参数：模型路径/模型目录、context size、threads、gpu layers 等
2. 选择规则（不破坏现有调用方）：
   - 若请求 body 里 `model` 写 `ollama:llama3` / `mnn:xxx` / `lmdeploy:xxx`，按前缀路由。
   - 没前缀则走默认 provider。

## 测试与回归策略（保证不会越改越乱）
1. 扩展现有 `tools/regression.py`：
   - 新增用例：`default_provider=llama_cpp` 下跑通 tool loop/planner 的基础链路（先用 fake-tool 或 llama.cpp 小模型）。
2. 新增最小单测/自测脚本：
   - `tools/smoke_providers.sh`：分别对 `llama_cpp/ollama/mnn/lmdeploy` 做一次 `/v1/models` 与一次 `/v1/chat/completions`。
3. 性能/稳定性：
   - llama.cpp provider 增加基本并发限制（同一模型串行或队列），避免多请求 OOM 或互相污染上下文。

## 交付物（你能立即拿去用的）
- Provider 抽象 + registry：路由层不再绑定 Ollama。
- `llama.cpp` 进程内默认 Provider：能跑通 chat/completions（含 tool loop/planner）。
- `MNN`/`LMDeploy` Provider（HTTP 适配 v0）：用户可通过 `model` 前缀或默认配置选择。
- 配置文档与回归脚本更新。

## 风险与规避
- llama.cpp 构建/平台差异：先 CPU 跑通（Metal/CUDA 后续加开关）。
- embeddings 能力差异：用 capability 标识，路由层对不支持的 provider 返回明确错误。
- 流式一致性：先保证非流式闭环，流式作为第二阶段能力开关。

我将按 Step 1→5 顺序实现，先把“可插拔 Provider”打通，再落地 llama.cpp 默认 Provider，最后补齐 MNN/LMDeploy 适配与回归。
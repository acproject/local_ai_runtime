# LLamaCpp Agent Server

基于 llama.cpp 的高性能 Agent 服务器，支持工具调用（Function Calling），兼容 OpenAI API 格式。

## 特性

- **🔧 工具调用（Function Calling）**：支持 JSON Schema 定义的工具，自动转换为 GBNF 语法约束
- **🚀 高性能**：基于 llama.cpp，支持 Metal GPU 加速（macOS）
- **🌐 OpenAI 兼容 API**：兼容 `/v1/chat/completions` 等标准端点
- **📝 流式响应**：支持 SSE 流式输出（开发中）
- **🔄 对话管理**：自动维护多轮对话历史
- **⚡ C++23 现代代码**：使用 `std::expected`、`std::format` 等现代特性

## 快速开始

### 环境要求

- **macOS**: Xcode 15+ (支持 C++23)
- **Linux**: GCC 13+ 或 Clang 17+
- **CMake**: 3.20+
- **Git**: 用于子模块

### 构建

```bash
# 克隆仓库
git clone <repository-url>
cd llama_cpp_agent

# 初始化子模块
git submodule update --init --recursive

# 构建
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 运行

```bash
# 下载模型（示例使用 Qwen2.5）
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf

# 启动服务器
./llama_agent_server qwen2.5-7b-instruct-q4_k_m.gguf

# 服务器默认运行在 http://localhost:8080
```

## API 使用

### 健康检查

```bash
curl http://localhost:8080/health
```

### 对话补全

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-agent",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 带工具调用的对话

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-agent",
    "messages": [
      {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

## 项目结构

```
llama_cpp_agent/
├── include/llama_agent/      # 公共头文件
│   ├── agent_runtime.hpp     # Agent 运行时核心
│   ├── gbnf_generator.hpp    # JSON Schema → GBNF
│   ├── llama_wrapper.hpp     # llama.cpp 封装
│   ├── tool_manager.hpp      # 工具管理
│   ├── tool_call_parser.hpp  # 工具调用解析
│   ├── conversation.hpp      # 对话历史
│   └── http_server.hpp       # HTTP 服务器
├── src/                      # 实现文件
├── tests/                    # 单元测试
├── extern/                   # 第三方依赖
│   ├── llama.cpp/           # llama.cpp 子模块
│   ├── cpp-httplib/         # HTTP 服务器库
│   └── json/                # nlohmann/json
└── build/                    # 构建目录
```

## 架构

```
┌─────────────────────────────────────────┐
│           HTTP API Server               │
│     (OpenAI-compatible endpoints)       │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           Agent Runtime                 │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │  State   │  │  Tool    │  │  Conv  ││
│  │ Machine  │  │  Manager │  │History ││
│  └──────────┘  └──────────┘  └────────┘│
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│        GrammarGenerator                 │
│    (JSON Schema → GBNF)                 │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│          LlamaWrapper                   │
│    (llama.cpp C++ Wrapper)              │
└─────────────────────────────────────────┘
```

## 核心组件

### 1. AgentRuntime

管理对话流程、状态机和工具调用：

```cpp
AgentRuntime runtime(
    std::make_unique<LlamaWrapper>(config),
    std::make_unique<ToolManager>(),
    agentConfig
);

// 注册工具
runtime.registerTool(toolDef, [](const nlohmann::json& params) {
    return nlohmann::json{{"temperature", 25.0}};
});

// 处理消息
auto response = runtime.processMessage("What's the weather?");
```

### 2. GrammarGenerator

将 JSON Schema 转换为 GBNF 语法：

```cpp
GrammarGenerator gen;
auto grammar = gen.generateFromSchema(schema);
auto toolGrammar = gen.generateToolCallGrammar(tools);
```

### 3. ToolCallParser

解析 LLM 输出的工具调用：

```cpp
ToolCallParser parser;
auto toolCalls = parser.parse(llmResponse);
for (const auto& call : toolCalls) {
    auto result = executeTool(call);
}
```

## 配置选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `systemPrompt` | string | - | 系统提示词 |
| `maxIterations` | int | 10 | 最大工具调用迭代次数 |
| `maxTokensPerResponse` | int | 512 | 每次响应最大 token 数 |
| `temperature` | float | 0.7 | 采样温度 |
| `enableToolUse` | bool | true | 启用工具调用 |
| `retryAttempts` | int | 3 | 错误重试次数 |

## 测试

```bash
# 运行测试
cd build
ctest --output-on-failure

# 运行特定测试
./tests/test_tool_call_parser
```

## 开发计划

- [x] Phase 1: 项目骨架搭建
- [x] Phase 2: 核心组件开发
- [x] Phase 3: Agent Runtime 完善
- [ ] Phase 4: 测试与文档
  - [x] 基础单元测试
  - [ ] 集成测试
  - [x] API 文档
- [ ] Phase 5: 性能优化
  - [ ] 批处理推理
  - [ ] 模型量化支持
  - [ ] 并发请求处理

## 技术栈

- **C++23**: `std::expected`, `std::format`, concepts
- **llama.cpp**: 推理后端
- **cpp-httplib**: HTTP 服务器
- **nlohmann/json**: JSON 处理
- **GoogleTest**: 测试框架

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - 优秀的 LLM 推理库
- [nlohmann/json](https://github.com/nlohmann/json) - 现代 C++ JSON 库
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) - C++ HTTP 库

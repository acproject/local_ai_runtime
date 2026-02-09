# LLamaCpp Agent Server - AGENTS.md

Guidelines for AI coding agents working on this C++23 Agent Runtime project.

## Build Commands

```bash
# Configure (Release)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_ENABLE=ON

# Build all targets
cmake --build . -j$(nproc)

# Build specific target
cmake --build . --target llama_agent_server
cmake --build . --target test_gbnf_generator

# Run all tests
ctest --output-on-failure

# Run single test (specific)
ctest -R test_gbnf_generator -V
ctest -R test_agent_runtime --output-on-failure

# Run with verbose output
ctest -V

# Clean build
rm -rf build && mkdir build && cd build && cmake .. && cmake --build .
```

## Code Style Guidelines

### Formatting
- Use `clang-format` with `.clang-format` (LLVM style, 4-space indent)
- Max line width: 100 characters
- Brace style: Attach (K&R style)
- Always use braces for control structures

### Naming Conventions
- **Classes/Structs**: PascalCase (`AgentRuntime`, `ToolManager`, `GrammarGenerator`)
- **Functions/Methods**: camelCase (`generateGrammar()`, `processRequest()`)
- **Private Members**: camelCase_ suffix (`llamaContext_`, `sessionId_`)
- **Public Members**: camelCase (no suffix)
- **Macros/Constants**: UPPER_SNAKE_CASE (`MAX_CONTEXT_SIZE`, `DEFAULT_PORT`)
- **Template Params**: T, InputIt, or descriptive names (`Allocator`)
- **Filenames**: snake_case.cpp / snake_case.hpp
- **Namespaces**: lowercase (`llama_agent`, `detail`)

### C++23 Features (Use Freely)
- `std::expected<T, E>` for error handling
- `std::format` for string formatting (avoid sprintf/iostream)
- `std::optional` and `std::variant` for type safety
- `constexpr` and `consteval` for compile-time evaluation
- Structured bindings: `auto [id, name] = parseId(input);`
- Ranges: `std::views::filter`, `std::views::transform`
- Deducing `this` for CRTP alternatives

## Import Order

```cpp
// 1. C++ Standard Library
#include <memory>
#include <string>
#include <vector>
#include <expected>
#include <format>

// 2. Third-party libraries
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <llama.h>

// 3. Project headers
#include "llama_agent/agent_runtime.hpp"
#include "llama_agent/tool_manager.hpp"
```

## Error Handling

```cpp
// Prefer std::expected for fallible operations
std::expected<std::string, Error> parseToolCall(const std::string& json);

// Use std::optional for nullable returns
std::optional<ToolDefinition> findTool(const std::string& name);

// Exceptions only for truly exceptional cases (constructor failures)
// Always document noexcept functions
```

## Project Structure

```
llama_cpp_agent/
├── CMakeLists.txt              # Root build config
├── extern/                     # Git submodules
│   ├── llama.cpp/
│   ├── cpp-httplib/
│   └── json/
├── include/llama_agent/        # Public headers
│   ├── agent_runtime.hpp
│   ├── gbnf_generator.hpp
│   ├── llama_wrapper.hpp
│   ├── tool_manager.hpp
│   └── http_server.hpp
├── src/                        # Implementation
│   ├── agent_runtime.cpp
│   ├── gbnf_generator.cpp
│   ├── llama_wrapper.cpp
│   ├── tool_manager.cpp
│   └── http_server.cpp
└── tests/                      # Unit tests
    ├── test_gbnf_generator.cpp
    ├── test_agent_runtime.cpp
    └── test_tool_manager.cpp
```

## Code Patterns

### Class Design
```cpp
class AgentRuntime {
public:
    explicit AgentRuntime(Config config);
    ~AgentRuntime() = default;
    
    // Disable copy, enable move
    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;
    AgentRuntime(AgentRuntime&&) noexcept = default;
    AgentRuntime& operator=(AgentRuntime&&) noexcept = default;
    
    std::expected<Response, Error> process(const Request& req);
    
private:
    std::unique_ptr<LlamaWrapper> llama_;
    ToolManager toolManager_;
};
```

### String Formatting
```cpp
// Use std::format (C++23)
auto msg = std::format("Processing request {} for model {}", 
                       requestId, modelName);
```

### JSON Handling
```cpp
using json = nlohmann::json;

// Prefer structured bindings when parsing
auto j = json::parse(jsonStr);
auto [name, params] = std::pair{
    j["name"].get<std::string>(),
    j["parameters"]
};
```

## Testing

- Use GoogleTest framework
- Test file naming: `test_<component>.cpp`
- One test class per component
- Use `TEST_F` for fixtures
- Mock external dependencies (llama.cpp C API)

## Memory Management

- Prefer `std::unique_ptr` for ownership
- Use `std::shared_ptr` only for shared ownership
- Avoid raw `new/delete`
- RAII for all resources (files, network, etc.)

## Performance Guidelines

- Use `const&` for large input parameters
- Use `std::move` for output parameters when appropriate
- Reserve vector capacity when size is known
- Profile before optimizing

## Documentation

- Document public APIs with Doxygen-style comments
- Explain WHY, not WHAT (code shows what)
- Document thread-safety guarantees
- Mark experimental features with `[[deprecated]]`

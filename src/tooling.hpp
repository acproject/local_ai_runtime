#pragma once

#include <nlohmann/json.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_agent {
class ToolManager;
}

namespace runtime {

struct RuntimeConfig;

struct ToolSchema {
  std::string name;
  std::string description;
  nlohmann::json parameters;
};

struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;
};

struct ToolResult {
  std::string tool_call_id;
  std::string name;
  nlohmann::json result;
  bool ok = true;
  std::string error;
};

using ToolHandler = std::function<ToolResult(const std::string& tool_call_id, const nlohmann::json& arguments)>;

class ToolRegistry {
 public:
  ToolRegistry() = default;
  ~ToolRegistry();
  ToolRegistry(const ToolRegistry&) = delete;
  ToolRegistry& operator=(const ToolRegistry&) = delete;
  ToolRegistry(ToolRegistry&& other) noexcept;
  ToolRegistry& operator=(ToolRegistry&& other) noexcept;

  void RegisterTool(ToolSchema schema, ToolHandler handler);
  bool HasTool(const std::string& name) const;
  std::optional<ToolSchema> GetSchema(const std::string& name) const;
  std::optional<ToolHandler> GetHandler(const std::string& name) const;

  std::vector<ToolSchema> ListSchemas() const;
  std::vector<ToolSchema> FilterSchemas(const std::vector<std::string>& allow_names) const;

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, ToolSchema> schemas_;
  std::unordered_map<std::string, ToolHandler> handlers_;
  std::unique_ptr<llama_agent::ToolManager> tool_manager_;
};

ToolRegistry BuildDefaultToolRegistry(const RuntimeConfig& cfg);

std::vector<std::string> ExtractToolNames(const std::vector<ToolSchema>& tools);

std::optional<nlohmann::json> ParseJsonLoose(const std::string& text);
std::optional<std::vector<ToolCall>> ParseToolCallsFromAssistantText(const std::string& assistant_text);

}  // namespace runtime

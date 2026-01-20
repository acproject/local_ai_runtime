#include "tooling.hpp"

#include "session_manager.hpp"

#include <chrono>
#include <sstream>
#include <string>

namespace runtime {
namespace {

static std::string Trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
  return s.substr(start, end - start);
}

static std::optional<std::string> ExtractFirstJsonObject(const std::string& text) {
  auto pos = text.find('{');
  if (pos == std::string::npos) return std::nullopt;
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = pos; i < text.size(); i++) {
    char c = text[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') depth++;
    if (c == '}') {
      depth--;
      if (depth == 0) return text.substr(pos, i - pos + 1);
    }
  }
  return std::nullopt;
}

static nlohmann::json ErrorResult(const std::string& message) {
  nlohmann::json j;
  j["ok"] = false;
  j["error"] = message;
  return j;
}

}  // namespace

ToolRegistry::ToolRegistry(ToolRegistry&& other) noexcept {
  std::unique_lock<std::shared_mutex> lock(other.mu_);
  schemas_ = std::move(other.schemas_);
  handlers_ = std::move(other.handlers_);
}

ToolRegistry& ToolRegistry::operator=(ToolRegistry&& other) noexcept {
  if (this == &other) return *this;
  std::unique_lock<std::shared_mutex> lock_other(other.mu_);
  std::unique_lock<std::shared_mutex> lock_this(mu_);
  schemas_ = std::move(other.schemas_);
  handlers_ = std::move(other.handlers_);
  return *this;
}

void ToolRegistry::RegisterTool(ToolSchema schema, ToolHandler handler) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  const auto name = schema.name;
  schemas_[name] = std::move(schema);
  handlers_[name] = std::move(handler);
}

bool ToolRegistry::HasTool(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return schemas_.find(name) != schemas_.end() && handlers_.find(name) != handlers_.end();
}

std::optional<ToolSchema> ToolRegistry::GetSchema(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = schemas_.find(name);
  if (it == schemas_.end()) return std::nullopt;
  return it->second;
}

std::optional<ToolHandler> ToolRegistry::GetHandler(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = handlers_.find(name);
  if (it == handlers_.end()) return std::nullopt;
  return it->second;
}

std::vector<ToolSchema> ToolRegistry::ListSchemas() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<ToolSchema> out;
  out.reserve(schemas_.size());
  for (const auto& [_, schema] : schemas_) out.push_back(schema);
  return out;
}

std::vector<ToolSchema> ToolRegistry::FilterSchemas(const std::vector<std::string>& allow_names) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<ToolSchema> out;
  out.reserve(allow_names.size());
  for (const auto& name : allow_names) {
    auto it = schemas_.find(name);
    if (it != schemas_.end()) out.push_back(it->second);
  }
  return out;
}

ToolRegistry BuildDefaultToolRegistry() {
  ToolRegistry reg;

  {
    ToolSchema schema;
    schema.name = "runtime.echo";
    schema.description = "Echo back the provided text.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"text", {{"type", "string"}}}}},
                         {"required", {"text"}}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.echo";
      if (!arguments.is_object() || !arguments.contains("text") || !arguments["text"].is_string()) {
        r.ok = false;
        r.error = "missing required field: text";
        r.result = ErrorResult(r.error);
        return r;
      }
      r.result = {{"ok", true}, {"text", arguments["text"].get<std::string>()}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "runtime.add";
    schema.description = "Add two numbers and return the sum.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
                         {"required", {"a", "b"}}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.add";
      if (!arguments.is_object() || !arguments.contains("a") || !arguments.contains("b")) {
        r.ok = false;
        r.error = "missing required fields: a, b";
        r.result = ErrorResult(r.error);
        return r;
      }
      if (!(arguments["a"].is_number() && arguments["b"].is_number())) {
        r.ok = false;
        r.error = "fields a and b must be numbers";
        r.result = ErrorResult(r.error);
        return r;
      }
      double a = arguments["a"].get<double>();
      double b = arguments["b"].get<double>();
      r.result = {{"ok", true}, {"sum", a + b}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "runtime.time";
    schema.description = "Get current unix time in seconds.";
    schema.parameters = {{"type", "object"}, {"properties", nlohmann::json::object()}, {"required", nlohmann::json::array()}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.time";
      auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      r.result = {{"ok", true}, {"unix_seconds", now}};
      return r;
    });
  }

  return reg;
}

std::vector<std::string> ExtractToolNames(const std::vector<ToolSchema>& tools) {
  std::vector<std::string> out;
  out.reserve(tools.size());
  for (const auto& t : tools) out.push_back(t.name);
  return out;
}

std::optional<nlohmann::json> ParseJsonLoose(const std::string& text) {
  auto trimmed = Trim(text);
  if (trimmed.empty()) return std::nullopt;
  if (auto j = nlohmann::json::parse(trimmed, nullptr, false); !j.is_discarded()) return j;
  if (auto obj = ExtractFirstJsonObject(trimmed)) {
    auto j = nlohmann::json::parse(*obj, nullptr, false);
    if (!j.is_discarded()) return j;
  }
  return std::nullopt;
}

std::optional<std::vector<ToolCall>> ParseToolCallsFromAssistantText(const std::string& assistant_text) {
  auto jopt = ParseJsonLoose(assistant_text);
  if (!jopt) return std::nullopt;
  const auto& j = *jopt;
  if (!j.is_object()) return std::nullopt;
  if (!j.contains("tool_calls") || !j["tool_calls"].is_array()) return std::nullopt;

  std::vector<ToolCall> calls;
  for (size_t i = 0; i < j["tool_calls"].size(); i++) {
    const auto& item = j["tool_calls"][i];
    if (!item.is_object()) continue;
    ToolCall c;
    c.id = NewId("call");
    if (item.contains("id") && item["id"].is_string()) c.id = item["id"].get<std::string>();
    if (item.contains("name") && item["name"].is_string()) c.name = item["name"].get<std::string>();
    if (item.contains("function") && item["function"].is_object() && item["function"].contains("name") &&
        item["function"]["name"].is_string()) {
      c.name = item["function"]["name"].get<std::string>();
    }
    if (item.contains("arguments")) {
      if (item["arguments"].is_string()) {
        c.arguments_json = item["arguments"].get<std::string>();
      } else if (!item["arguments"].is_null()) {
        c.arguments_json = item["arguments"].dump();
      }
    } else if (item.contains("function") && item["function"].is_object() && item["function"].contains("arguments")) {
      const auto& a = item["function"]["arguments"];
      if (a.is_string()) {
        c.arguments_json = a.get<std::string>();
      } else if (!a.is_null()) {
        c.arguments_json = a.dump();
      }
    }
    if (!c.name.empty()) calls.push_back(std::move(c));
  }
  if (calls.empty()) return std::nullopt;
  return calls;
}

}  // namespace runtime

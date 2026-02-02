#include "tooling.hpp"

#include "session_manager.hpp"

#include <chrono>
#include <cctype>
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

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromJson(const nlohmann::json& original) {
  if (!original.is_object()) return std::nullopt;

  const nlohmann::json* root = &original;
  if (root->contains("opencode") && (*root)["opencode"].is_object()) root = &(*root)["opencode"];

  auto make_call = [&](const nlohmann::json& item) -> std::optional<ToolCall> {
    if (!item.is_object()) return std::nullopt;
    ToolCall c;
    c.id = NewId("call");
    if (item.contains("id") && item["id"].is_string()) c.id = item["id"].get<std::string>();

    if (item.contains("name") && item["name"].is_string()) c.name = item["name"].get<std::string>();
    if (c.name.empty() && item.contains("tool") && item["tool"].is_string()) c.name = item["tool"].get<std::string>();
    if (c.name.empty() && item.contains("toolName") && item["toolName"].is_string()) c.name = item["toolName"].get<std::string>();
    if (c.name.empty() && item.contains("function") && item["function"].is_object() && item["function"].contains("name") &&
        item["function"]["name"].is_string()) {
      c.name = item["function"]["name"].get<std::string>();
    }

    auto set_args = [&](const nlohmann::json& a) {
      if (a.is_string()) {
        c.arguments_json = a.get<std::string>();
      } else if (a.is_object() || a.is_array() || a.is_number() || a.is_boolean()) {
        c.arguments_json = a.dump();
      } else if (a.is_null()) {
        c.arguments_json = "{}";
      } else {
        c.arguments_json = a.dump();
      }
    };

    if (item.contains("arguments")) {
      set_args(item["arguments"]);
    } else if (item.contains("args")) {
      set_args(item["args"]);
    } else if (item.contains("input")) {
      set_args(item["input"]);
    } else if (item.contains("function") && item["function"].is_object() && item["function"].contains("arguments")) {
      set_args(item["function"]["arguments"]);
    }

    if (c.arguments_json.empty()) c.arguments_json = "{}";
    if (c.name.empty()) return std::nullopt;
    return c;
  };

  for (const auto& key : {"tool_call", "toolCall", "toolcall"}) {
    if (root->contains(key) && (*root)[key].is_object()) {
      if (auto c = make_call((*root)[key])) return std::vector<ToolCall>{*c};
    }
  }

  if (auto c = make_call(*root)) return std::vector<ToolCall>{*c};

  const nlohmann::json* tool_calls = nullptr;
  for (const auto& key : {"tool_calls", "toolCalls", "toolcalls"}) {
    if (root->contains(key) && (*root)[key].is_array()) {
      tool_calls = &(*root)[key];
      break;
    }
  }
  if (!tool_calls) return std::nullopt;

  std::vector<ToolCall> calls;
  for (size_t i = 0; i < tool_calls->size(); i++) {
    if (auto c = make_call((*tool_calls)[i])) calls.push_back(std::move(*c));
  }
  if (calls.empty()) return std::nullopt;
  return calls;
}

static bool IsToolNameChar(char ch) {
  const unsigned char c = static_cast<unsigned char>(ch);
  return std::isalnum(c) || ch == '_' || ch == '-' || ch == '.' || ch == ':' || ch == '/';
}

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromTaggedText(const std::string& assistant_text) {
  std::string lower;
  lower.reserve(assistant_text.size());
  for (char ch : assistant_text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

  const std::string tool_tag = "<tool_call";
  const std::string tool_tag2 = "<toolcall";
  const std::string arg_tag = "<arg_value>";
  const std::string arg_end = "</arg_value>";

  std::vector<ToolCall> calls;
  size_t pos = 0;
  while (pos < lower.size()) {
    size_t start = lower.find(tool_tag, pos);
    if (start == std::string::npos) start = lower.find(tool_tag2, pos);
    if (start == std::string::npos) break;

    size_t tag_close = lower.find('>', start);
    if (tag_close == std::string::npos) break;

    std::string name;
    std::string tag_text = assistant_text.substr(start, tag_close - start + 1);
    std::string tag_lower = lower.substr(start, tag_close - start + 1);
    auto find_attr = [&](const std::string& attr) -> std::optional<std::string> {
      auto p = tag_lower.find(attr);
      if (p == std::string::npos) return std::nullopt;
      p += attr.size();
      while (p < tag_text.size() && std::isspace(static_cast<unsigned char>(tag_text[p]))) p++;
      if (p >= tag_text.size() || tag_text[p] != '=') return std::nullopt;
      p++;
      while (p < tag_text.size() && std::isspace(static_cast<unsigned char>(tag_text[p]))) p++;
      if (p >= tag_text.size()) return std::nullopt;
      if (tag_text[p] == '"' || tag_text[p] == '\'') {
        const char q = tag_text[p++];
        size_t qend = tag_text.find(q, p);
        if (qend == std::string::npos) return std::nullopt;
        return tag_text.substr(p, qend - p);
      }
      size_t e = p;
      while (e < tag_text.size() && !std::isspace(static_cast<unsigned char>(tag_text[e])) && tag_text[e] != '>') e++;
      if (e <= p) return std::nullopt;
      return tag_text.substr(p, e - p);
    };

    if (auto n = find_attr("name")) name = Trim(*n);
    size_t after_name = tag_close + 1;
    if (name.empty()) {
      size_t name_start = tag_close + 1;
      while (name_start < assistant_text.size() && std::isspace(static_cast<unsigned char>(assistant_text[name_start]))) name_start++;
      size_t name_end = name_start;
      while (name_end < assistant_text.size() && IsToolNameChar(assistant_text[name_end])) name_end++;
      name = Trim(assistant_text.substr(name_start, name_end - name_start));
      after_name = name_end;
    }

    if (name.empty()) {
      pos = tag_close + 1;
      continue;
    }

    size_t block_start = tag_close + 1;
    size_t next_tool = lower.find(tool_tag, block_start);
    size_t next_tool2 = lower.find(tool_tag2, block_start);
    if (next_tool == std::string::npos || (next_tool2 != std::string::npos && next_tool2 < next_tool)) next_tool = next_tool2;
    size_t block_end = (next_tool == std::string::npos) ? assistant_text.size() : next_tool;

    std::string args_text;
    size_t astart = lower.find(arg_tag, after_name);
    if (astart != std::string::npos && astart < block_end) {
      astart += arg_tag.size();
      size_t aend = lower.find(arg_end, astart);
      if (aend == std::string::npos || aend > block_end) aend = block_end;
      args_text = Trim(assistant_text.substr(astart, aend - astart));
    } else {
      size_t maybe_close = lower.find(arg_end, after_name);
      if (maybe_close != std::string::npos && maybe_close < block_end) {
        size_t raw_start = maybe_close + arg_end.size();
        args_text = Trim(assistant_text.substr(raw_start, block_end - raw_start));
      } else {
        args_text = Trim(assistant_text.substr(after_name, block_end - after_name));
      }
    }

    if (!args_text.empty()) {
      if (auto first = ExtractFirstJsonObject(args_text)) args_text = Trim(*first);
    }

    ToolCall c;
    c.id = NewId("call");
    c.name = name;
    if (args_text.empty()) {
      c.arguments_json = "{}";
    } else {
      c.arguments_json = args_text;
    }
    calls.push_back(std::move(c));

    pos = block_end;
  }

  if (calls.empty()) return std::nullopt;
  return calls;
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
  if (jopt) {
    if (auto from_json = ExtractToolCallsFromJson(*jopt)) return from_json;
  }
  return ExtractToolCallsFromTaggedText(assistant_text);
}

}  // namespace runtime

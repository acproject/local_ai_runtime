#include "openai_router.hpp"
#include "config.hpp"
#include "llama_agent/gbnf_generator.hpp"
#include "llama_agent/tool_manager.hpp"
#include "ollama_provider.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>

namespace runtime {
namespace {

static int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string GetEnvStr(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static std::string ToLowerAscii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

static bool IsGlmFamilyModel(const std::string& model) {
  return ToLowerAscii(model).find("glm") != std::string::npos;
}

static std::string NormalizePrefix(std::string p) {
  if (p.empty()) return {};
  if (p == "/") return {};
  if (!p.empty() && p.back() == '/') p.pop_back();
  if (p.empty()) return {};
  if (p.front() != '/') p.insert(p.begin(), '/');
  return p;
}

static std::vector<std::string> GetApiPrefixes() {
  std::string mode = ToLowerAscii(GetEnvStr("RUNTIME_API_PREFIX_MODE"));
  if (mode.empty()) mode = "auto";

  if (mode == "v1" || mode == "none" || mode == "off") {
    return {""};
  }
  if (mode == "api") {
    return {"/api"};
  }
  return {"", "/api"};
}

static nlohmann::json MakeError(const std::string& message, const std::string& type) {
  nlohmann::json j;
  j["error"] = {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}};
  return j;
}

static nlohmann::json MakeAnthropicError(const std::string& message, const std::string& type) {
  nlohmann::json j;
  j["type"] = "error";
  j["error"] = {{"type", type}, {"message", message}};
  return j;
}

static void SendJson(httplib::Response* res, int status, const nlohmann::json& body) {
  res->status = status;
  res->set_header("Content-Type", "application/json");
  res->set_content(body.dump(), "application/json");
}

static std::string SseData(const nlohmann::json& j) {
  return std::string("data: ") + j.dump() + "\n\n";
}

static std::string SseDone() {
  return "data: [DONE]\n\n";
}

static std::string SseEvent(const std::string& event, const nlohmann::json& j) {
  return std::string("event: ") + event + "\n" + "data: " + j.dump() + "\n\n";
}

static nlohmann::json ParseJsonBody(const httplib::Request& req) {
  return nlohmann::json::parse(req.body, nullptr, false);
}

static std::string RedactHeaderValue(const std::string& key, const std::string& value) {
  const auto k = ToLowerAscii(key);
  if (k == "authorization" || k == "proxy-authorization" || k == "api-key" || k == "api_key" || k == "x-api-key") {
    return "<redacted>";
  }
  return value;
}

static std::string SanitizeBodyForLog(const std::string& body) {
  if (body.empty()) return {};
  auto j = nlohmann::json::parse(body, nullptr, false);
  if (j.is_discarded()) return body;
  if (j.is_object()) {
    for (const auto& key : {"api_key", "api-key", "authorization", "apiKey"}) {
      if (j.contains(key)) j.erase(key);
    }
    if (j.contains("headers") && j["headers"].is_object()) {
      auto& h = j["headers"];
      for (const auto& key : {"authorization", "proxy-authorization", "api-key", "api_key", "x-api-key"}) {
        if (h.contains(key)) h.erase(key);
      }
    }
  }
  return j.dump();
}

static std::string TruncateForLog(std::string s, size_t max_chars) {
  if (max_chars == 0) return {};
  if (s.size() <= max_chars) return s;
  constexpr const char* kSuffix = "...(truncated)";
  if (max_chars <= std::strlen(kSuffix)) return std::string(kSuffix).substr(0, max_chars);
  s.resize(max_chars - std::strlen(kSuffix));
  s += kSuffix;
  return s;
}

static bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static std::string ToolKindForLog(const std::string& tool_name) {
  if (StartsWith(tool_name, "lsp.")) return "lsp";
  if (StartsWith(tool_name, "ide.")) return "ide";
  if (StartsWith(tool_name, "fs.")) return "fs";
  if (StartsWith(tool_name, "runtime.")) return "runtime";
  if (StartsWith(tool_name, "mcp.")) return "mcp";
  if (tool_name.find(".mcp.") != std::string::npos) return "mcp";
  return "tool";
}

static void LogRequestRaw(const httplib::Request& req) {
  std::cout << "[request] " << req.method << " " << req.path << "\n";
  for (const auto& it : req.headers) {
    std::cout << "  " << it.first << ": " << RedactHeaderValue(it.first, it.second) << "\n";
  }
  if (!req.body.empty()) {
    std::cout << "  body: " << SanitizeBodyForLog(req.body) << "\n";
  }
}

static std::string TrimAscii(std::string s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
    s.erase(s.begin());
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
    s.pop_back();
  }
  return s;
}

static std::optional<std::string> ExtractBearerToken(const std::string& authorization_value) {
  const auto v = TrimAscii(authorization_value);
  const auto lower = ToLowerAscii(v);
  constexpr const char* kPrefix = "bearer ";
  if (lower.size() < std::strlen(kPrefix)) return std::nullopt;
  if (lower.compare(0, std::strlen(kPrefix), kPrefix) != 0) return std::nullopt;
  auto token = TrimAscii(v.substr(std::strlen(kPrefix)));
  if (token.empty()) return std::nullopt;
  return token;
}

static RequestHeaderList BuildUpstreamAuthHeaders(const std::string& key) {
  if (key.empty()) return {};
  RequestHeaderList out;
  out.emplace_back("Authorization", std::string("Bearer ") + key);
  out.emplace_back("x-api-key", key);
  out.emplace_back("api-key", key);
  return out;
}

static RequestHeaderList ExtractUpstreamAuthHeaders(const httplib::Request& req) {
  if (const auto token = ExtractBearerToken(req.get_header_value("authorization"))) {
    return BuildUpstreamAuthHeaders(*token);
  }
  const auto x_api_key = TrimAscii(req.get_header_value("x-api-key"));
  if (!x_api_key.empty()) return BuildUpstreamAuthHeaders(x_api_key);
  const auto api_key = TrimAscii(req.get_header_value("api-key"));
  if (!api_key.empty()) return BuildUpstreamAuthHeaders(api_key);
  const auto api_key2 = TrimAscii(req.get_header_value("api_key"));
  if (!api_key2.empty()) return BuildUpstreamAuthHeaders(api_key2);
  return {};
}

static void LogProviderUse(const std::string& provider_name, const std::string& model) {
  std::cout << "[provider] " << provider_name << " model=" << model << "\n";
}

static void LogClientMessage(const std::string& session_id, const std::vector<ChatMessage>& messages) {
  std::cout << "[client-message] session_id=" << session_id << "\n";
  for (const auto& m : messages) {
    std::cout << "  " << m.role << ": " << m.content << "\n";
  }
}

static std::string ExtractMessageContent(const nlohmann::json& content) {
  if (content.is_string()) return content.get<std::string>();
  if (content.is_object()) {
    if (content.contains("type") && content["type"].is_string()) {
      const auto type = content["type"].get<std::string>();
      if (type == "text" || type == "input_text") {
        if (content.contains("text") && content["text"].is_string()) return content["text"].get<std::string>();
        if (content.contains("content") && content["content"].is_string()) return content["content"].get<std::string>();
      }
    }
    if (content.contains("text") && content["text"].is_string()) return content["text"].get<std::string>();
    if (content.contains("content") && content["content"].is_string()) return content["content"].get<std::string>();
    if (content.contains("parts")) return ExtractMessageContent(content["parts"]);
    return {};
  }
  if (!content.is_array()) return {};
  std::string out;
  for (const auto& part : content) {
    if (!part.is_object()) continue;
    if (part.contains("type") && part["type"].is_string()) {
      const auto type = part["type"].get<std::string>();
      if (type == "text" || type == "input_text") {
        if (part.contains("text") && part["text"].is_string()) {
          out += part["text"].get<std::string>();
        } else if (part.contains("content") && part["content"].is_string()) {
          out += part["content"].get<std::string>();
        }
        continue;
      }
      continue;
    }
    if (part.contains("text") && part["text"].is_string()) out += part["text"].get<std::string>();
  }
  return out;
}

static std::vector<ChatMessage> ParseChatMessages(const nlohmann::json& j, bool* ok) {
  std::vector<ChatMessage> out;
  *ok = false;
  if (!j.contains("messages") || !j["messages"].is_array()) return out;
  for (const auto& m : j["messages"]) {
    if (!m.is_object()) continue;
    ChatMessage cm;
    if (m.contains("role") && m["role"].is_string()) cm.role = m["role"].get<std::string>();
    if (m.contains("content")) cm.content = ExtractMessageContent(m["content"]);
    if (!cm.role.empty()) out.push_back(std::move(cm));
  }
  *ok = true;
  return out;
}

static std::vector<std::string> ParseRequestedToolNames(const nlohmann::json& j) {
  std::vector<std::string> out;
  if (!j.contains("tools") || !j["tools"].is_array()) return out;
  for (const auto& t : j["tools"]) {
    if (t.is_string()) {
      out.push_back(t.get<std::string>());
      continue;
    }
    if (!t.is_object()) continue;
    if (t.contains("function") && t["function"].is_object()) {
      if (t["function"].contains("name") && t["function"]["name"].is_string()) {
        out.push_back(t["function"]["name"].get<std::string>());
        continue;
      }
      if (t["function"].contains("tool") && t["function"]["tool"].is_string()) {
        out.push_back(t["function"]["tool"].get<std::string>());
        continue;
      }
    }
    if (t.contains("name") && t["name"].is_string()) {
      out.push_back(t["name"].get<std::string>());
      continue;
    }
    if (t.contains("tool") && t["tool"].is_string()) {
      out.push_back(t["tool"].get<std::string>());
      continue;
    }
  }
  return out;
}

static bool ToolChoiceIsNone(const nlohmann::json& j) {
  if (!j.contains("tool_choice")) return false;
  const auto& tc = j["tool_choice"];
  if (tc.is_string()) return tc.get<std::string>() == "none";
  if (tc.is_object() && tc.contains("type") && tc["type"].is_string()) return tc["type"].get<std::string>() == "none";
  return false;
}

static bool WantsServerToolLoop(const nlohmann::json& j) {
  for (const auto& k : {"max_steps", "max_tool_calls", "planner", "trace"}) {
    if (j.contains(k)) return true;
  }
  if (j.contains("tools") && j["tools"].is_array()) {
    for (const auto& t : j["tools"]) {
      if (t.is_string()) return true;
      if (t.is_object()) {
        const nlohmann::json* obj = &t;
        if (t.contains("function") && t["function"].is_object()) obj = &t["function"];
        if ((obj->contains("name") && (*obj)["name"].is_string() && !(*obj)["name"].get<std::string>().empty()) ||
            (obj->contains("tool") && (*obj)["tool"].is_string() && !(*obj)["tool"].get<std::string>().empty())) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool ToolsContainFullSchemas(const nlohmann::json& j) {
  if (!j.contains("tools") || !j["tools"].is_array()) return false;
  for (const auto& t : j["tools"]) {
    if (!t.is_object()) continue;
    const nlohmann::json* obj = &t;
    if (t.contains("function") && t["function"].is_object()) obj = &t["function"];
    if (obj->contains("parameters") || obj->contains("description")) return true;
  }
  return false;
}

static std::vector<ToolSchema> ParseRequestedToolSchemas(const nlohmann::json& j) {
  std::vector<ToolSchema> out;
  if (!j.contains("tools") || !j["tools"].is_array()) return out;
  for (const auto& t : j["tools"]) {
    if (!t.is_object()) continue;
    ToolSchema s;
    if (t.contains("function") && t["function"].is_object()) {
      const auto& fn = t["function"];
      if (fn.contains("name") && fn["name"].is_string()) s.name = fn["name"].get<std::string>();
      if (fn.contains("description") && fn["description"].is_string()) s.description = fn["description"].get<std::string>();
      if (fn.contains("parameters")) s.parameters = fn["parameters"];
    } else {
      if (t.contains("name") && t["name"].is_string()) s.name = t["name"].get<std::string>();
      if (t.contains("description") && t["description"].is_string()) s.description = t["description"].get<std::string>();
      if (t.contains("parameters")) s.parameters = t["parameters"];
    }
    if (s.name.empty()) continue;
    if (s.parameters.is_null()) s.parameters = nlohmann::json::object();
    out.push_back(std::move(s));
  }
  return out;
}

static std::optional<std::string> ExtractForcedToolName(const nlohmann::json& j) {
  if (!j.contains("tool_choice")) return std::nullopt;
  const auto& tc = j["tool_choice"];
  if (!tc.is_object()) return std::nullopt;
  if (tc.contains("type") && tc["type"].is_string() && tc["type"].get<std::string>() == "function") {
    if (tc.contains("function") && tc["function"].is_object() && tc["function"].contains("name") && tc["function"]["name"].is_string()) {
      return tc["function"]["name"].get<std::string>();
    }
  }
  return std::nullopt;
}

static std::string BuildToolSystemPromptClientManaged(const std::vector<ToolSchema>& tools, const std::optional<std::string>& forced_tool) {
  nlohmann::json tool_list = nlohmann::json::array();
  for (const auto& t : tools) {
    tool_list.push_back({{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}});
  }
  nlohmann::json spec;
  spec["tools"] = tool_list;

  std::string prompt;
  prompt += "You are a tool-using assistant.\n";
  prompt += "Tool results will be provided as messages with role \"tool\".\n";
  if (forced_tool && !forced_tool->empty()) {
    prompt += "When calling a tool, you MUST call: " + *forced_tool + "\n";
  }
  prompt += "When you need to call tool(s), respond ONLY with a single JSON object:\n";
  prompt += "{\"tool_calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"arguments\":{...}}]}\n";
  prompt += "If you can answer without tools, respond ONLY with:\n";
  prompt += "{\"final\":\"...\"}\n";
  prompt += "Never include any extra text outside the JSON.\n";
  prompt += "Available tools spec:\n";
  prompt += spec.dump();
  return prompt;
}

static nlohmann::json BuildOpenAiToolCalls(const std::vector<ToolCall>& calls) {
  nlohmann::json out = nlohmann::json::array();
  for (const auto& c : calls) {
    out.push_back({{"id", c.id}, {"type", "function"}, {"function", {{"name", c.name}, {"arguments", c.arguments_json}}}});
  }
  return out;
}

static void NormalizeClientManagedToolCalls(const std::vector<ToolSchema>& tools, std::vector<ToolCall>* calls) {
  if (!calls) return;

  auto find_schema = [&](const std::string& name) -> const ToolSchema* {
    for (const auto& t : tools) {
      if (t.name == name) return &t;
    }
    return nullptr;
  };

  auto guess_single_key = [&](const nlohmann::json& params) -> std::optional<std::string> {
    if (!params.is_object()) return std::nullopt;
    if (!params.contains("type") || !params["type"].is_string() || params["type"].get<std::string>() != "object") return std::nullopt;
    if (!params.contains("properties") || !params["properties"].is_object()) return std::nullopt;
    const auto& props = params["properties"];

    if (params.contains("required") && params["required"].is_array() && params["required"].size() == 1 && params["required"][0].is_string()) {
      return params["required"][0].get<std::string>();
    }
    if (props.size() == 1) return props.begin().key();
    for (const auto& cand : {"filePath", "path", "uri", "content", "text", "input", "command"}) {
      if (props.contains(cand)) return std::string(cand);
    }
    return std::nullopt;
  };

  for (auto& c : *calls) {
    const auto* schema = find_schema(c.name);
    auto j = ParseJsonLoose(c.arguments_json);
    if (!j) {
      if (schema) {
        auto key = guess_single_key(schema->parameters);
        if (key && !key->empty()) {
          nlohmann::json obj = nlohmann::json::object();
          obj[*key] = c.arguments_json;
          c.arguments_json = obj.dump();
          continue;
        }
      }
      c.arguments_json = nlohmann::json(c.arguments_json).dump();
      continue;
    }
    if (!j->is_string()) continue;

    const auto raw = j->get<std::string>();
    if (!schema) {
      c.arguments_json = nlohmann::json(raw).dump();
      continue;
    }
    auto key = guess_single_key(schema->parameters);
    if (!key || key->empty()) {
      c.arguments_json = nlohmann::json(raw).dump();
      continue;
    }
    nlohmann::json obj = nlohmann::json::object();
    obj[*key] = raw;
    c.arguments_json = obj.dump();
  }
}

static bool LooksLikePathLike(const std::string& s) {
  if (s.empty()) return false;
  if (s.find('/') != std::string::npos || s.find('\\') != std::string::npos) return true;
  if (s.size() >= 2 && std::isalpha(static_cast<unsigned char>(s[0])) && s[1] == ':') return true;
  if (s[0] == '.' || s[0] == '~') return true;
  return false;
}

static std::optional<std::string> GuessSingleKeyFromParams(const nlohmann::json& params, const std::string& raw) {
  if (!params.is_object()) return std::nullopt;
  if (!params.contains("type") || !params["type"].is_string()) return std::nullopt;
  if (params["type"].get<std::string>() != "object") return std::nullopt;
  if (!params.contains("properties") || !params["properties"].is_object()) return std::nullopt;
  const auto& props = params["properties"];

  if (params.contains("required") && params["required"].is_array() && params["required"].size() == 1 && params["required"][0].is_string()) {
    return params["required"][0].get<std::string>();
  }
  if (props.size() == 1) return props.begin().key();

  if (LooksLikePathLike(raw)) {
    for (const auto& cand : {"filePath", "path", "uri"}) {
      if (props.contains(cand)) return std::string(cand);
    }
  }
  for (const auto& cand : {"command", "text", "input", "content"}) {
    if (props.contains(cand)) return std::string(cand);
  }
  return std::nullopt;
}

static void NormalizeToolArgsObject(const ToolSchema& schema, nlohmann::json* args) {
  if (!args || !args->is_object()) return;
  if (!schema.parameters.is_object()) return;
  if (!schema.parameters.contains("properties") || !schema.parameters["properties"].is_object()) return;
  const auto& props = schema.parameters["properties"];

  auto move_key = [&](const char* dst, std::initializer_list<const char*> srcs) {
    if (!props.contains(dst)) return;
    if (args->contains(dst)) return;
    for (const auto* src : srcs) {
      if (src && args->contains(src)) {
        (*args)[dst] = std::move((*args)[src]);
        args->erase(src);
        return;
      }
    }
  };

  move_key("filePath", {"path", "filepath", "file_path", "file", "filename", "uri"});
  move_key("path", {"filePath", "filepath", "file_path", "file", "filename", "uri"});
  move_key("uri", {"url", "path", "filePath"});
  move_key("content", {"text", "data", "body", "contents"});
  move_key("text", {"content", "data", "body"});
  move_key("oldString", {"old", "from", "pattern", "search", "oldText"});
  move_key("newString", {"new", "to", "replacement", "replace", "newText"});
  move_key("replaceAll", {"all", "global"});
}

static std::string MapFinishReasonToAnthropicStopReason(const std::string& finish_reason) {
  if (finish_reason == "length") return "max_tokens";
  if (finish_reason == "content_filter") return "end_turn";
  return "end_turn";
}

static std::string BuildToolSystemPrompt(const std::vector<ToolSchema>& tools) {
  nlohmann::json tool_list = nlohmann::json::array();
  for (const auto& t : tools) {
    tool_list.push_back({{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}});
  }
  nlohmann::json spec;
  spec["tools"] = tool_list;

  std::string prompt;
  prompt += "You are a tool-using assistant.\n";
  prompt += "If you need to call tools, respond ONLY with a single JSON object:\n";
  prompt += "{\"tool_calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"arguments\":{...}}]}\n";
  prompt += "If you can answer without tools, respond ONLY with:\n";
  prompt += "{\"final\":\"...\"}\n";
  prompt += "Never include any extra text outside the JSON.\n";
  prompt += "Available tools spec:\n";
  prompt += spec.dump();
  return prompt;
}

static std::string BuildToolLoopGrammar(const std::vector<ToolSchema>& tools) {
  std::vector<llama_agent::ToolDefinition> defs;
  defs.reserve(tools.size());
  for (const auto& t : tools) {
    llama_agent::ToolDefinition d;
    d.name = t.name;
    d.description = t.description;
    defs.push_back(std::move(d));
  }

  llama_agent::GrammarGenerator gen;
  auto tool_part = gen.generateToolCallGrammar(defs);
  if (!tool_part.has_value()) {
    std::string g;
    g += "root ::= ws (final_object | tool_calls_object) ws\n\n";
    g += "final_object ::= \"{\" ws final_pair ws \"}\" ws\n";
    g += "final_pair ::= \"\\\"final\\\"\" ws \":\" ws string\n\n";
    g += "tool_calls_object ::= \"{\" ws tool_calls_pair ws \"}\" ws\n";
    g += "tool_calls_pair ::= \"\\\"tool_calls\\\"\" ws \":\" ws tool_calls\n\n";
    g += "tool_calls ::= \"[\" ws tool_call_list? \"]\" ws\n";
    g += "tool_call_list ::= tool_call (\",\" ws tool_call)*\n";
    g += "tool_call ::= \"{\" ws id_pair \",\" ws name_pair \",\" ws arguments_pair ws \"}\" ws\n";
    g += "id_pair ::= \"\\\"id\\\"\" ws \":\" ws string\n";
    g += "name_pair ::= \"\\\"name\\\"\" ws \":\" ws string\n";
    g += "arguments_pair ::= \"\\\"arguments\\\"\" ws \":\" ws json_value\n\n";
    g += R"(
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
number ::= ("-"? [0-9]+) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
json_object ::= "{" ws (json_pair ("," ws json_pair)*)? "}" ws
json_pair ::= string ":" ws json_value
json_array ::= "[" ws (json_value ("," ws json_value)*)? "]" ws
json_value ::= json_object | json_array | string | number | ("true" | "false" | "null") ws
ws ::= [ \t\n]*
)";
    return g;
  }

  std::string g;
  g += "root ::= ws (final_object | tool_calls_object) ws\n\n";
  g += "final_object ::= \"{\" ws final_pair ws \"}\" ws\n";
  g += "final_pair ::= \"\\\"final\\\"\" ws \":\" ws string\n\n";
  g += "tool_calls_object ::= \"{\" ws tool_calls_pair ws \"}\" ws\n";
  g += "tool_calls_pair ::= \"\\\"tool_calls\\\"\" ws \":\" ws tool_calls\n\n";
  g += *tool_part;
  return g;
}

static std::string BuildPlannerSystemPrompt(const std::vector<ToolSchema>& tools, int max_plan_steps) {
  nlohmann::json tool_list = nlohmann::json::array();
  for (const auto& t : tools) {
    tool_list.push_back({{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}});
  }
  nlohmann::json spec;
  spec["tools"] = tool_list;

  std::string prompt;
  prompt += "You are a planner.\n";
  prompt += "Return ONLY a single JSON object and no extra text.\n";
  prompt += "If tools are needed, output:\n";
  prompt += "{\"plan\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n";
  prompt += "The plan length MUST be <= " + std::to_string(max_plan_steps) + ".\n";
  prompt += "If no tools are needed, output:\n";
  prompt += "{\"final\":\"...\"}\n";
  prompt += "Available tools spec:\n";
  prompt += spec.dump();
  return prompt;
}

static std::string BuildPlannerFinalSystemPrompt() {
  std::string prompt;
  prompt += "You are a tool result summarizer.\n";
  prompt += "You have been given TOOL_RESULT messages.\n";
  prompt += "Return ONLY a single JSON object and no extra text:\n";
  prompt += "{\"final\":\"...\"}\n";
  return prompt;
}

static std::optional<std::string> ExtractFinalFromAssistantJson(const std::string& text) {
  auto j = ParseJsonLoose(text);
  if (!j || !j->is_object()) return std::nullopt;
  const nlohmann::json* root = &(*j);
  if (root->contains("opencode") && (*root)["opencode"].is_object()) root = &(*root)["opencode"];
  if (root->contains("final") && (*root)["final"].is_string()) return (*root)["final"].get<std::string>();
  if (root->contains("content") && (*root)["content"].is_string()) return (*root)["content"].get<std::string>();
  if (root->contains("text") && (*root)["text"].is_string()) return (*root)["text"].get<std::string>();
  return std::nullopt;
}

static std::string FakeModelOnce(const std::vector<ChatMessage>& messages) {
  bool has_tool_result = false;
  std::string last_user;
  std::string last_system;
  for (const auto& m : messages) {
    if (m.role == "user") last_user = m.content;
    if (m.role == "system") last_system = m.content;
    if (m.role == "user" && m.content.find("TOOL_RESULT") != std::string::npos) {
      has_tool_result = true;
      break;
    }
  }

  auto extract_uri_arg = [&](const std::string& text) -> std::optional<std::string> {
    auto pos = text.find("uri=");
    if (pos == std::string::npos) return std::nullopt;
    pos += 4;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
    if (pos >= text.size()) return std::nullopt;
    char quote = 0;
    if (text[pos] == '"' || text[pos] == '\'') {
      quote = text[pos];
      pos++;
    }
    size_t end = pos;
    if (quote) {
      end = text.find(quote, pos);
      if (end == std::string::npos) return std::nullopt;
    } else {
      while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) end++;
    }
    if (end <= pos) return std::nullopt;
    return text.substr(pos, end - pos);
  };

  if (last_system.find("You are a planner.") != std::string::npos) {
    if (last_user.find("bad_args") != std::string::npos) {
      return R"({"plan":[{"name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":"x","character":2}}]})";
    }
    if (last_user.find("ide.read_file") != std::string::npos) {
      return R"({"plan":[{"name":"ide.read_file","arguments":{"path":"src/main.cpp"}}]})";
    }
    if (last_user.find("ide.search") != std::string::npos) {
      return R"({"plan":[{"name":"ide.search","arguments":{"query":"OpenAiRouter","path":"src"}}]})";
    }
    if (last_user.find("ide.hover") != std::string::npos) {
      return R"({"plan":[{"name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.definition") != std::string::npos) {
      return R"({"plan":[{"name":"ide.definition","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.diagnostics") != std::string::npos) {
      if (auto uri = extract_uri_arg(last_user)) {
        return std::string(R"({"plan":[{"name":"ide.diagnostics","arguments":{"uri":)") + nlohmann::json(*uri).dump() +
               R"(}}]})";
      }
      return R"({"plan":[{"name":"ide.diagnostics","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp"}}]})";
    }
    if (last_user.find("lsp.hover") != std::string::npos) {
      return R"({"plan":[{"name":"lsp.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("read_file") != std::string::npos) {
      return R"({"plan":[{"name":"read_file","arguments":{"filePath":"src/main.cpp","offset":0,"limit":50}}]})";
    }
    if (last_user.find("readFile") != std::string::npos) {
      return R"({"plan":[{"name":"readFile","arguments":{"filePath":"src/main.cpp","offset":0,"limit":50}}]})";
    }
    if (last_user.find("writeFile") != std::string::npos) {
      return R"({"plan":[{"name":"writeFile","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","content":"hello"}}]})";
    }
    if (last_user.find("editFile") != std::string::npos) {
      return R"({"plan":[{"name":"editFile","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","oldString":"hello","newString":"hello2","replaceAll":false}}]})";
    }
    if (last_user.find("edit") != std::string::npos) {
      return R"({"plan":[{"name":"edit","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","oldString":"hello","newString":"hello2","replaceAll":false}}]})";
    }
    if (last_user.find("glob") != std::string::npos) {
      return R"({"plan":[{"name":"glob","arguments":{"pattern":"*.cpp","path":"src"}}]})";
    }
    if (last_user.find("grep") != std::string::npos) {
      return R"({"plan":[{"name":"grep","arguments":{"pattern":"BuildDefaultToolRegistry","path":"src"}}]})";
    }
    if (last_user.find("list") != std::string::npos) {
      return R"({"plan":[{"name":"list","arguments":{"path":"src"}}]})";
    }
    return R"({"plan":[{"name":"runtime.add","arguments":{"a":2,"b":3}}]})";
  }
  if (last_system.find("tool result summarizer") != std::string::npos) {
    auto pos = last_user.find("TOOL_RESULT");
    if (pos != std::string::npos) return std::string("{\"final\":") + nlohmann::json(last_user).dump() + "}";
    return R"({"final":"done"})";
  }
  if (!has_tool_result) {
    if (last_user.find("mcp2.mcp.echo") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"mcp2.mcp.echo","arguments":{"text":"hello2"}}]})";
    }
    if (last_user.find("mcp.echo") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"mcp.echo","arguments":{"text":"hello"}}]})";
    }
    if (last_user.find("runtime.infer_task_status") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"runtime.infer_task_status","arguments":{"session_id":"test"}}]})";
    }
    if (last_user.find("ide.read_file") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.read_file","arguments":{"path":"src/main.cpp"}}]})";
    }
    if (last_user.find("ide.search") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.search","arguments":{"query":"OpenAiRouter","path":"src"}}]})";
    }
    if (last_user.find("ide.hover") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.definition") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.definition","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.diagnostics") != std::string::npos) {
      if (auto uri = extract_uri_arg(last_user)) {
        return std::string(R"({"tool_calls":[{"id":"call_1","name":"ide.diagnostics","arguments":{"uri":)") +
               nlohmann::json(*uri).dump() + R"(}}]})";
      }
      return R"({"tool_calls":[{"id":"call_1","name":"ide.diagnostics","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp"}}]})";
    }
    if (last_user.find("lsp.hover") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"lsp.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("read_file") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"filePath":"src/main.cpp","offset":0,"limit":50}}]})";
    }
    if (last_user.find("readFile") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"readFile","arguments":{"filePath":"src/main.cpp","offset":0,"limit":50}}]})";
    }
    if (last_user.find("write_mismatch") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"write","arguments":{"path":"build-vs2022-x64-cuda/opencode_tool_test_mismatch.txt","content":"hello"}}]})";
    }
    if (last_user.find("edit_mismatch") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"edit","arguments":{"path":"build-vs2022-x64-cuda/opencode_tool_test_mismatch.txt","old":"hello","new":"hello2","replaceAll":false}}]})";
    }
    if (last_user.find("writeFile") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"writeFile","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","content":"hello"}}]})";
    }
    if (last_user.find("editFile") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"editFile","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","oldString":"hello","newString":"hello2","replaceAll":false}}]})";
    }
    if (last_user.find("edit") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"edit","arguments":{"filePath":"build-vs2022-x64-cuda/opencode_tool_test.txt","oldString":"hello","newString":"hello2","replaceAll":false}}]})";
    }
    if (last_user.find("glob") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"glob","arguments":{"pattern":"*.cpp","path":"src"}}]})";
    }
    if (last_user.find("grep") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"grep","arguments":{"pattern":"BuildDefaultToolRegistry","path":"src"}}]})";
    }
    if (last_user.find("list") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"list","arguments":{"path":"src"}}]})";
    }
    return R"({"tool_calls":[{"id":"call_1","name":"runtime.add","arguments":{"a":2,"b":3}}]})";
  }
  if (last_user.find("mcp.echo") != std::string::npos || last_user.find("mcp2.mcp.echo") != std::string::npos ||
      last_user.find("runtime.infer_task_status") != std::string::npos || last_user.find("lsp.hover") != std::string::npos ||
      last_user.find("ide.hover") != std::string::npos || last_user.find("ide.read_file") != std::string::npos ||
      last_user.find("ide.search") != std::string::npos || last_user.find("ide.definition") != std::string::npos ||
      last_user.find("ide.diagnostics") != std::string::npos || last_user.find("read_file") != std::string::npos ||
      last_user.find("readFile") != std::string::npos || last_user.find("writeFile") != std::string::npos ||
      last_user.find("editFile") != std::string::npos || last_user.find("edit") != std::string::npos ||
      last_user.find("glob") != std::string::npos || last_user.find("grep") != std::string::npos ||
      last_user.find("list") != std::string::npos) {
    auto pos = last_user.find("TOOL_RESULT");
    if (pos != std::string::npos) {
      return std::string("{\"final\":") + nlohmann::json(last_user).dump() + "}";
    }
    return R"({"final":"done"})";
  }
  return R"({"final":"2 + 3 = 5"})";
}

struct ToolLoopResult {
  std::string final_text;
  std::vector<ToolCall> executed_calls;
  std::vector<ToolResult> results;
  int steps = 0;
  bool hit_step_limit = false;
  bool hit_tool_limit = false;
  bool used_planner = false;
  bool planner_failed = false;
  int plan_steps = 0;
  int plan_rewrites = 0;
  nlohmann::json plan = nlohmann::json::array();
};

static nlohmann::json BuildRuntimeTrace(const ToolLoopResult& loop) {
  nlohmann::json j;
  j["steps"] = loop.steps;
  j["hit_step_limit"] = loop.hit_step_limit;
  j["hit_tool_limit"] = loop.hit_tool_limit;
  j["used_planner"] = loop.used_planner;
  j["planner_failed"] = loop.planner_failed;
  j["plan_steps"] = loop.plan_steps;
  j["plan_rewrites"] = loop.plan_rewrites;
  j["plan"] = loop.plan;
  j["tool_calls"] = nlohmann::json::array();
  for (const auto& c : loop.executed_calls) {
    j["tool_calls"].push_back({{"id", c.id}, {"name", c.name}, {"arguments", c.arguments_json}});
  }
  j["tool_results"] = nlohmann::json::array();
  for (const auto& r : loop.results) {
    j["tool_results"].push_back({{"tool_call_id", r.tool_call_id}, {"name", r.name}, {"ok", r.ok}, {"result", r.result}});
  }
  return j;
}

struct PlannerPlanStep {
  std::string name;
  nlohmann::json arguments;
};

static std::optional<std::vector<PlannerPlanStep>> ParsePlannerPlan(const std::string& assistant_text) {
  auto j = ParseJsonLoose(assistant_text);
  if (!j || !j->is_object()) return std::nullopt;
  if (j->contains("final") && (*j)["final"].is_string()) return std::vector<PlannerPlanStep>{};
  if (!j->contains("plan") || !(*j)["plan"].is_array()) return std::nullopt;
  std::vector<PlannerPlanStep> out;
  for (const auto& s : (*j)["plan"]) {
    if (!s.is_object()) continue;
    if (!s.contains("name") || !s["name"].is_string()) continue;
    PlannerPlanStep step;
    step.name = s["name"].get<std::string>();
    if (s.contains("arguments") && s["arguments"].is_object()) {
      step.arguments = s["arguments"];
    } else {
      step.arguments = nlohmann::json::object();
    }
    if (!step.name.empty()) out.push_back(std::move(step));
  }
  return out;
}

static bool CheckType(const std::string& t, const nlohmann::json& v) {
  if (t == "string") return v.is_string();
  if (t == "integer") return v.is_number_integer();
  if (t == "number") return v.is_number();
  if (t == "boolean") return v.is_boolean();
  if (t == "object") return v.is_object();
  if (t == "array") return v.is_array();
  return true;
}

static bool ValidateSchemaLoose(const nlohmann::json& schema, const nlohmann::json& args, std::string* err) {
  if (!schema.is_object()) return true;
  if (schema.contains("type") && schema["type"].is_string()) {
    auto t = schema["type"].get<std::string>();
    if (!CheckType(t, args)) {
      if (err) *err = "arguments type mismatch";
      return false;
    }
  }
  if (schema.contains("required") && schema["required"].is_array() && args.is_object()) {
    for (const auto& r : schema["required"]) {
      if (!r.is_string()) continue;
      auto k = r.get<std::string>();
      if (!args.contains(k)) {
        if (err) *err = "missing required field: " + k;
        return false;
      }
    }
  }
  if (schema.contains("properties") && schema["properties"].is_object() && args.is_object()) {
    for (const auto& [k, ps] : schema["properties"].items()) {
      if (!args.contains(k)) continue;
      if (!ps.is_object()) continue;
      if (ps.contains("type") && ps["type"].is_string()) {
        if (!CheckType(ps["type"].get<std::string>(), args[k])) {
          if (err) *err = "field type mismatch: " + k;
          return false;
        }
      }
    }
  }
  return true;
}

static std::string ChatOnceText(const std::string& model,
                                const std::vector<ChatMessage>& messages,
                                const std::optional<int>& max_tokens,
                                const std::optional<float>& temperature,
                                const std::optional<float>& top_p,
                                const std::optional<float>& min_p,
                                IProvider* provider,
                                std::string* err) {
  if (model == "fake-tool") return FakeModelOnce(messages);
  ChatRequest req;
  req.model = model;
  req.stream = false;
  req.max_tokens = max_tokens;
  req.temperature = temperature;
  req.top_p = top_p;
  req.min_p = min_p;
  req.messages = messages;
  auto resp = provider->ChatOnce(req, err);
  if (!resp) return {};
  return resp->content;
}

static ToolLoopResult RunPlanner(const std::string& model,
                                 const std::string& session_id,
                                 const std::vector<ChatMessage>& full_messages,
                                 const std::vector<ToolSchema>& allowed_tools,
                                 const ToolRegistry& registry,
                                 IProvider* provider,
                                 const std::optional<int>& max_tokens,
                                 const std::optional<float>& temperature,
                                 const std::optional<float>& top_p,
                                 const std::optional<float>& min_p,
                                 int max_plan_steps,
                                 int max_plan_rewrites,
                                 int max_tool_calls,
                                 std::string* err) {
  ToolLoopResult out;
  out.used_planner = true;

  if (max_plan_steps <= 0) max_plan_steps = 1;
  if (max_plan_rewrites < 0) max_plan_rewrites = 0;
  if (max_tool_calls < 0) max_tool_calls = 0;

  std::unordered_set<std::string> allowed_names;
  for (const auto& t : allowed_tools) allowed_names.insert(t.name);

  std::vector<ChatMessage> plan_msgs;
  plan_msgs.reserve(full_messages.size() + 2);
  plan_msgs.push_back({"system", BuildPlannerSystemPrompt(allowed_tools, max_plan_steps)});
  for (const auto& m : full_messages) plan_msgs.push_back(m);

  std::optional<std::vector<PlannerPlanStep>> plan;
  std::string plan_text;
  int rewrites = 0;
  for (int attempt = 0; attempt <= max_plan_rewrites; attempt++) {
    plan_text = ChatOnceText(model, plan_msgs, max_tokens, temperature, top_p, min_p, provider, err);
    if (plan_text.empty() && err && !err->empty()) {
      out.planner_failed = true;
      return out;
    }
    if (auto final = ExtractFinalFromAssistantJson(plan_text)) {
      out.final_text = *final;
      out.steps = 1;
      out.plan_steps = 0;
      return out;
    }
    plan = ParsePlannerPlan(plan_text);
    if (!plan) {
      if (attempt == max_plan_rewrites) {
        out.planner_failed = true;
        out.final_text = plan_text;
        out.steps = 1;
        return out;
      }
      plan_msgs.push_back({"user", "Plan invalid JSON. Return a corrected plan JSON only."});
      continue;
    }
    bool ok = true;
    std::string why;
    for (const auto& s : *plan) {
      if (!allowed_names.empty() && allowed_names.find(s.name) == allowed_names.end()) {
        ok = false;
        why = "tool not allowed: " + s.name;
        break;
      }
      auto schema = registry.GetSchema(s.name);
      if (!schema) {
        ok = false;
        why = "tool not found: " + s.name;
        break;
      }
      std::string schema_err;
      if (!ValidateSchemaLoose(schema->parameters, s.arguments, &schema_err)) {
        ok = false;
        why = "invalid arguments for " + s.name + ": " + schema_err;
        break;
      }
    }
    if (ok) break;
    if (attempt == max_plan_rewrites) {
      out.planner_failed = true;
      out.final_text = why;
      out.steps = 1;
      return out;
    }
    plan_msgs.push_back({"user", "Plan rejected: " + why + ". Return a corrected plan JSON only."});
    plan.reset();
    rewrites = attempt + 1;
  }

  if (!plan) {
    out.planner_failed = true;
    out.final_text = plan_text;
    out.steps = 1;
    return out;
  }

  if (static_cast<int>(plan->size()) > max_plan_steps) plan->resize(static_cast<size_t>(max_plan_steps));
  out.plan_steps = static_cast<int>(plan->size());
  out.plan_rewrites = rewrites;
  out.plan = nlohmann::json::array();
  for (const auto& s : *plan) out.plan.push_back({{"name", s.name}, {"arguments", s.arguments}});

  std::vector<ChatMessage> exec_msgs;
  exec_msgs.reserve(full_messages.size() + out.plan_steps + 4);
  for (const auto& m : full_messages) exec_msgs.push_back(m);

  int tool_calls_used = 0;
  for (size_t i = 0; i < plan->size(); i++) {
    if (tool_calls_used >= max_tool_calls) {
      out.hit_tool_limit = true;
      out.final_text = "tool call limit exceeded";
      out.steps = static_cast<int>(i + 1);
      return out;
    }

    const auto& s = (*plan)[i];
    ToolCall c;
    c.id = "plan_" + std::to_string(i + 1);
    c.name = s.name;
    c.arguments_json = s.arguments.dump();
    out.executed_calls.push_back(c);

    ToolResult r;
    r.tool_call_id = c.id;
    r.name = c.name;

    std::cout << "[tool-call] session_id=" << session_id << " id=" << c.id << " name=" << c.name
              << " kind=" << ToolKindForLog(c.name) << " arguments="
              << TruncateForLog(SanitizeBodyForLog(c.arguments_json), 2000) << "\n";

    if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
      r.ok = false;
      r.error = "tool not allowed";
      r.result = {{"ok", false}, {"error", r.error}};
      out.results.push_back(r);
      std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                << " ok=0 error=" << r.error << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
      exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      tool_calls_used++;
      continue;
    }
    auto handler = registry.GetHandler(c.name);
    if (!handler) {
      r.ok = false;
      r.error = "tool not found";
      r.result = {{"ok", false}, {"error", r.error}};
      out.results.push_back(r);
      std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                << " ok=0 error=" << r.error << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
      exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      tool_calls_used++;
      continue;
    }

    r = (*handler)(c.id, s.arguments);
    out.results.push_back(r);
    std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
              << " ok=" << (r.ok ? 1 : 0) << " error=" << (r.error.empty() ? "-" : r.error)
              << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
    exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
    tool_calls_used++;
  }

  std::vector<ChatMessage> final_msgs;
  final_msgs.reserve(exec_msgs.size() + 2);
  final_msgs.push_back({"system", BuildPlannerFinalSystemPrompt()});
  for (const auto& m : exec_msgs) final_msgs.push_back(m);

  auto final_text = ChatOnceText(model, final_msgs, max_tokens, temperature, top_p, min_p, provider, err);
  out.steps = 2;
  if (auto final = ExtractFinalFromAssistantJson(final_text)) {
    out.final_text = *final;
    return out;
  }
  out.final_text = final_text;
  return out;
}

static ToolLoopResult RunToolLoop(const std::string& model,
                                  const std::string& session_id,
                                  const std::vector<ChatMessage>& full_messages,
                                  const std::vector<ToolSchema>& allowed_tools,
                                  const ToolRegistry& registry,
                                  IProvider* provider,
                                  const std::optional<int>& max_tokens,
                                  const std::optional<float>& temperature,
                                  const std::optional<float>& top_p,
                                  const std::optional<float>& min_p,
                                  int max_steps,
                                  int max_tool_calls,
                                  std::string* err) {
  ToolLoopResult out;
  std::vector<ChatMessage> msgs;
  msgs.reserve(full_messages.size() + 8);

  std::unordered_set<std::string> allowed_names;
  for (const auto& t : allowed_tools) allowed_names.insert(t.name);

  if (!allowed_tools.empty()) {
    msgs.push_back({"system", BuildToolSystemPrompt(allowed_tools)});
  }
  for (const auto& m : full_messages) msgs.push_back(m);

  if (max_steps <= 0) max_steps = 1;
  if (max_tool_calls < 0) max_tool_calls = 0;

  int tool_calls_used = 0;
  for (int step = 0; step < max_steps; step++) {
    out.steps = step + 1;
    std::string assistant_text;
    if (model == "fake-tool") {
      assistant_text = FakeModelOnce(msgs);
    } else {
      ChatRequest req;
      req.model = model;
      req.stream = false;
      req.max_tokens = max_tokens;
      req.temperature = temperature;
      req.top_p = top_p;
      req.min_p = min_p;
      if (provider && provider->Name() == "llama_cpp" && !allowed_tools.empty()) {
        req.grammar = BuildToolLoopGrammar(allowed_tools);
      }
      req.messages = msgs;
      auto resp = provider->ChatOnce(req, err);
      if (!resp) return out;
      assistant_text = resp->content;
    }

    if (auto calls = ParseToolCallsFromAssistantText(assistant_text)) {
      for (auto& c : *calls) {
        if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "tool not allowed";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          std::cout << "[tool-call] session_id=" << session_id << " id=" << c.id << " name=" << c.name
                    << " kind=" << ToolKindForLog(c.name) << " arguments="
                    << TruncateForLog(SanitizeBodyForLog(c.arguments_json), 2000) << "\n";
          std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                    << " ok=0 error=" << r.error << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        if (!registry.HasTool(c.name)) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "tool not found";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          std::cout << "[tool-call] session_id=" << session_id << " id=" << c.id << " name=" << c.name
                    << " kind=" << ToolKindForLog(c.name) << " arguments="
                    << TruncateForLog(SanitizeBodyForLog(c.arguments_json), 2000) << "\n";
          std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                    << " ok=0 error=" << r.error << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        if (tool_calls_used >= max_tool_calls) {
          out.hit_tool_limit = true;
          out.final_text = "tool call limit exceeded";
          return out;
        }
        auto schema = registry.GetSchema(c.name);
        auto jargs = ParseJsonLoose(c.arguments_json);
        if (schema && jargs && jargs->is_string()) {
          std::string raw = jargs->get<std::string>();
          const auto& p = schema->parameters;
          if (p.is_object() && p.contains("type") && p["type"].is_string() && p["type"].get<std::string>() == "string") {
            jargs = nlohmann::json(raw);
          } else if (auto key = GuessSingleKeyFromParams(p, raw)) {
            nlohmann::json obj = nlohmann::json::object();
            obj[*key] = raw;
            jargs = std::move(obj);
          }
        }
        if (!jargs) {
          if (schema) {
            std::string raw = c.arguments_json;
            size_t s = 0;
            while (s < raw.size() && std::isspace(static_cast<unsigned char>(raw[s]))) s++;
            size_t e = raw.size();
            while (e > s && std::isspace(static_cast<unsigned char>(raw[e - 1]))) e--;
            raw = raw.substr(s, e - s);

            const auto& p = schema->parameters;
            if (p.is_object() && p.contains("type") && p["type"].is_string() && p["type"].get<std::string>() == "string") {
              jargs = nlohmann::json(raw);
            } else if (auto key = GuessSingleKeyFromParams(p, raw)) {
              jargs = nlohmann::json::object();
              (*jargs)[*key] = raw;
            }
          }
        }
        if (schema && jargs && jargs->is_object()) NormalizeToolArgsObject(*schema, &(*jargs));
        if (!jargs) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "invalid tool arguments json";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          std::cout << "[tool-call] session_id=" << session_id << " id=" << c.id << " name=" << c.name
                    << " kind=" << ToolKindForLog(c.name) << " arguments=" << TruncateForLog(c.arguments_json, 2000) << "\n";
          std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                    << " ok=0 error=" << r.error << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        auto handler = registry.GetHandler(c.name);
        if (!handler) continue;
        std::cout << "[tool-call] session_id=" << session_id << " id=" << c.id << " name=" << c.name
                  << " kind=" << ToolKindForLog(c.name) << " arguments="
                  << TruncateForLog(SanitizeBodyForLog(jargs->dump()), 2000) << "\n";
        auto r = (*handler)(c.id, *jargs);
        tool_calls_used++;
        out.executed_calls.push_back(c);
        out.results.push_back(r);
        std::cout << "[tool-result] session_id=" << session_id << " id=" << r.tool_call_id << " name=" << r.name
                  << " ok=" << (r.ok ? 1 : 0) << " error=" << (r.error.empty() ? "-" : r.error)
                  << " result=" << TruncateForLog(SanitizeBodyForLog(r.result.dump()), 2000) << "\n";
        msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      }
      continue;
    }

    if (auto final = ExtractFinalFromAssistantJson(assistant_text)) {
      out.final_text = *final;
      return out;
    }

    out.final_text = assistant_text;
    return out;
  }

  out.hit_step_limit = true;
  out.final_text = "tool loop exceeded max steps";
  return out;
}

}  // namespace

OpenAiRouter::OpenAiRouter(SessionManager* sessions, ProviderRegistry* providers, ToolRegistry tools)
    : sessions_(sessions), providers_(providers), tools_(std::move(tools)) {}

ToolRegistry* OpenAiRouter::MutableTools() {
  return &tools_;
}

void OpenAiRouter::Register(httplib::Server* server) {
  const auto prefixes = GetApiPrefixes();

  auto models_handler = [&](const httplib::Request& req, httplib::Response& res) {
    LogRequestRaw(req);
    ScopedRequestAuthHeaders scope(ExtractUpstreamAuthHeaders(req));
    nlohmann::json out;
    out["object"] = "list";
    out["data"] = nlohmann::json::array();
    nlohmann::json provider_status = nlohmann::json::object();
    const std::string default_provider = providers_ ? providers_->DefaultProviderName() : "";
    if (providers_) {
      for (auto* p : providers_->List()) {
        std::string err;
        auto models = p->ListModels(&err);
        if (p->Name() == "ollama") {
          if (auto* op = dynamic_cast<OllamaProvider*>(p)) {
            std::string ps_err;
            auto ps = op->GetPs(&ps_err);
            if (ps) {
              provider_status["ollama"] = {{"ps", *ps}};
            } else if (!ps_err.empty()) {
              provider_status["ollama"] = {{"ps_error", ps_err}};
            }
          }
        }
        for (const auto& m : models) {
          nlohmann::json item;
          if (p->Name() == default_provider) {
            item["id"] = m.id;
          } else {
            item["id"] = p->Name() + ":" + m.id;
          }
          item["object"] = "model";
          item["created"] = NowSeconds();
          item["owned_by"] = m.owned_by.empty() ? p->Name() : m.owned_by;
          out["data"].push_back(std::move(item));
        }
      }
    }
    if (!provider_status.empty()) {
      out["provider_status"] = std::move(provider_status);
    }
    SendJson(&res, 200, out);
  };

  auto embeddings_handler = [&](const httplib::Request& req, httplib::Response& res) {
    LogRequestRaw(req);
    ScopedRequestAuthHeaders scope(ExtractUpstreamAuthHeaders(req));
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    std::string input;
    if (j.contains("input") && j["input"].is_string()) {
      input = j["input"].get<std::string>();
    } else if (j.contains("input") && j["input"].is_array() && !j["input"].empty() && j["input"][0].is_string()) {
      input = j["input"][0].get<std::string>();
    } else {
      return SendJson(&res, 400, MakeError("missing field: input", "invalid_request_error"));
    }

    std::string err;
    auto resolved = providers_ ? providers_->Resolve(model) : std::nullopt;
    if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
    if (providers_) {
      auto sw = providers_->Activate(resolved->provider_name);
      if (sw.switched) {
        std::cout << "[provider-switch] from=" << sw.from << " to=" << sw.to << "\n";
      }
    }
    LogProviderUse(resolved->provider_name, resolved->model);
    auto vec = resolved->provider->Embeddings(resolved->model, input, &err);
    if (!vec) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));

    nlohmann::json out;
    out["object"] = "list";
    out["data"] = nlohmann::json::array();
    nlohmann::json item;
    item["object"] = "embedding";
    item["embedding"] = *vec;
    item["index"] = 0;
    out["data"].push_back(std::move(item));
    out["model"] = model;
    out["usage"] = {{"prompt_tokens", nullptr}, {"total_tokens", nullptr}};
    SendJson(&res, 200, out);
  };

  auto chat_completions_handler = [&](const httplib::Request& req, httplib::Response& res) {
    LogRequestRaw(req);
    const auto auth_headers = ExtractUpstreamAuthHeaders(req);
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    bool ok = false;
    auto req_messages = ParseChatMessages(j, &ok);
    if (!ok) return SendJson(&res, 400, MakeError("missing field: messages", "invalid_request_error"));

    std::string preferred_session_id;
    if (j.contains("session_id") && j["session_id"].is_string()) preferred_session_id = j["session_id"].get<std::string>();
    if (preferred_session_id.empty()) {
      preferred_session_id = req.get_header_value("x-session-id");
      if (preferred_session_id.empty()) preferred_session_id = req.get_header_value("X-Session-Id");
    }
    std::string session_id = sessions_->EnsureSessionId(preferred_session_id);
    res.set_header("x-session-id", session_id);
    LogClientMessage(session_id, req_messages);

    bool use_server_history = false;
    bool use_server_history_explicit = false;
    if (j.contains("use_server_history") && j["use_server_history"].is_boolean()) {
      use_server_history = j["use_server_history"].get<bool>();
      use_server_history_explicit = true;
    }
    bool has_assistant = false;
    for (const auto& m : req_messages) {
      if (m.role == "assistant" || m.role == "tool") {
        has_assistant = true;
        break;
      }
    }
    if (!use_server_history_explicit) {
      use_server_history = !has_assistant;
    }

    std::vector<ChatMessage> full_messages;
    if (use_server_history) {
      auto sess = sessions_->GetOrCreate(session_id);
      full_messages = sess.history;
      full_messages.insert(full_messages.end(), req_messages.begin(), req_messages.end());
    } else {
      full_messages = req_messages;
    }

    bool stream = false;
    if (j.contains("stream") && j["stream"].is_boolean()) stream = j["stream"].get<bool>();
    std::optional<int> max_tokens;
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
      max_tokens = j["max_tokens"].get<int>();
    } else if (j.contains("max_completion_tokens") && j["max_completion_tokens"].is_number_integer()) {
      max_tokens = j["max_completion_tokens"].get<int>();
    }
    std::optional<float> temperature;
    if (j.contains("temperature") && j["temperature"].is_number()) temperature = j["temperature"].get<float>();
    std::optional<float> top_p;
    if (j.contains("top_p") && j["top_p"].is_number()) top_p = j["top_p"].get<float>();
    std::optional<float> min_p;
    if (j.contains("min_p") && j["min_p"].is_number()) min_p = j["min_p"].get<float>();

    int max_steps = 6;
    if (j.contains("max_steps") && j["max_steps"].is_number_integer()) max_steps = j["max_steps"].get<int>();
    int max_tool_calls = 16;
    if (j.contains("max_tool_calls") && j["max_tool_calls"].is_number_integer()) {
      max_tool_calls = j["max_tool_calls"].get<int>();
    }
    bool planner = false;
    int max_plan_steps = 6;
    int max_plan_rewrites = 2;
    if (j.contains("planner")) {
      const auto& p = j["planner"];
      if (p.is_boolean()) {
        planner = p.get<bool>();
      } else if (p.is_object()) {
        if (p.contains("enabled") && p["enabled"].is_boolean()) planner = p["enabled"].get<bool>();
        if (p.contains("max_plan_steps") && p["max_plan_steps"].is_number_integer()) {
          max_plan_steps = p["max_plan_steps"].get<int>();
        }
        if (p.contains("max_rewrites") && p["max_rewrites"].is_number_integer()) {
          max_plan_rewrites = p["max_rewrites"].get<int>();
        }
      }
    }
    bool trace = false;
    if (j.contains("trace") && j["trace"].is_boolean()) trace = j["trace"].get<bool>();

    auto turn_id = NewId("turn");
    TurnRecord turn;
    turn.turn_id = turn_id;
    turn.input_messages = req_messages;

    std::vector<ToolSchema> allowed_tools;
    const bool tool_choice_none = ToolChoiceIsNone(j);
    const bool client_managed_tools = (!tool_choice_none && ToolsContainFullSchemas(j));
    const bool server_tool_loop = (!tool_choice_none && WantsServerToolLoop(j) && !client_managed_tools);
    if (server_tool_loop) {
      auto names = ParseRequestedToolNames(j);
      allowed_tools = tools_.FilterSchemas(names);
    }
    std::vector<ToolSchema> client_tools = client_managed_tools ? ParseRequestedToolSchemas(j) : std::vector<ToolSchema>();
    std::optional<std::string> forced_tool = client_managed_tools ? ExtractForcedToolName(j) : std::nullopt;

    IProvider* provider = nullptr;
    std::string provider_model = model;
    if (model != "fake-tool") {
      auto resolved = providers_->Resolve(model);
      if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
      provider = resolved->provider;
      provider_model = resolved->model;
      if (providers_) {
        auto sw = providers_->Activate(resolved->provider_name);
        if (sw.switched) {
          std::cout << "[provider-switch] from=" << sw.from << " to=" << sw.to << "\n";
        }
      }
      LogProviderUse(resolved->provider_name, resolved->model);
    }

    if ((!allowed_tools.empty() || !client_tools.empty()) && IsGlmFamilyModel(provider_model)) {
      temperature = 0.7f;
      top_p = 1.0f;
    }

    if (!stream) {
      ScopedRequestAuthHeaders scope(auth_headers);
      std::string err;
      ToolLoopResult loop;
      std::string finish_reason = "stop";
      if (!client_tools.empty()) {
        std::vector<ChatMessage> msgs;
        msgs.reserve(full_messages.size() + 2);
        msgs.push_back({"system", BuildToolSystemPromptClientManaged(client_tools, forced_tool)});
        for (const auto& m : full_messages) msgs.push_back(m);

        std::string assistant_text;
        if (model == "fake-tool") {
          assistant_text = FakeModelOnce(msgs);
        } else {
          ChatRequest creq;
          creq.model = provider_model;
          creq.stream = false;
          creq.max_tokens = max_tokens;
          creq.temperature = temperature;
          creq.top_p = top_p;
          creq.min_p = min_p;
          creq.messages = msgs;
          auto resp = provider->ChatOnce(creq, &err);
          if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
          assistant_text = resp->content;
          finish_reason = resp->finish_reason;
        }

        if (auto calls = ParseToolCallsFromAssistantText(assistant_text)) {
          NormalizeClientManagedToolCalls(client_tools, &(*calls));
          turn.output_text = assistant_text;
          sessions_->AppendTurn(session_id, turn);
          if (use_server_history) {
            sessions_->AppendToHistory(session_id, req_messages);
            sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", assistant_text}});
          }

          nlohmann::json out;
          out["id"] = NewId("chatcmpl");
          out["object"] = "chat.completion";
          out["created"] = NowSeconds();
          out["model"] = model;
          out["choices"] = nlohmann::json::array();
          nlohmann::json choice;
          choice["index"] = 0;
          choice["message"] = {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", BuildOpenAiToolCalls(*calls)}};
          choice["finish_reason"] = "tool_calls";
          out["choices"].push_back(std::move(choice));
          out["usage"] = {{"prompt_tokens", nullptr}, {"completion_tokens", nullptr}, {"total_tokens", nullptr}};
          return SendJson(&res, 200, out);
        }

        if (auto final = ExtractFinalFromAssistantJson(assistant_text)) {
          loop.final_text = *final;
          finish_reason = "stop";
        } else {
          loop.final_text = assistant_text;
        }
      } else if (!allowed_tools.empty()) {
        if (planner) {
          loop =
              RunPlanner(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p, max_plan_steps,
                         max_plan_rewrites, max_tool_calls, &err);
          if (loop.planner_failed) {
            loop = RunToolLoop(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p, max_steps,
                               max_tool_calls, &err);
          }
        } else {
          loop = RunToolLoop(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p, max_steps,
                             max_tool_calls, &err);
        }
      } else {
        if (model == "fake-tool") {
          loop.final_text = FakeModelOnce(full_messages);
        } else {
          ChatRequest creq;
          creq.model = provider_model;
          creq.stream = false;
          creq.max_tokens = max_tokens;
          creq.temperature = temperature;
          creq.top_p = top_p;
          creq.min_p = min_p;
          creq.messages = full_messages;
          auto resp = provider->ChatOnce(creq, &err);
          if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
          loop.final_text = resp->content;
          finish_reason = resp->finish_reason;
        }
      }
      if (loop.final_text.empty() && !err.empty()) return SendJson(&res, 502, MakeError(err, "api_error"));
      if (trace) res.set_header("x-runtime-trace", BuildRuntimeTrace(loop).dump());

      turn.output_text = loop.final_text;
      sessions_->AppendTurn(session_id, turn);
      if (use_server_history) {
        sessions_->AppendToHistory(session_id, req_messages);
        for (const auto& tc : loop.executed_calls) {
          sessions_->AppendToHistory(session_id,
                                     {ChatMessage{"assistant", "TOOL_CALL " + tc.name + " " + tc.arguments_json}});
        }
        for (const auto& tr : loop.results) {
          sessions_->AppendToHistory(session_id, {ChatMessage{"user", "TOOL_RESULT " + tr.name + " " + tr.result.dump()}});
        }
        sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", loop.final_text}});
      }

      std::cout << "[chat] session_id=" << session_id << " stream=0 max_tokens=" << (max_tokens.has_value() ? max_tokens.value() : -1)
                << " finish_reason=" << finish_reason << " output_chars=" << loop.final_text.size() << "\n";

      nlohmann::json out;
      out["id"] = NewId("chatcmpl");
      out["object"] = "chat.completion";
      out["created"] = NowSeconds();
      out["model"] = model;
      out["choices"] = nlohmann::json::array();
      nlohmann::json choice;
      choice["index"] = 0;
      choice["message"] = {{"role", "assistant"}, {"content", loop.final_text}};
      choice["finish_reason"] = finish_reason;
      out["choices"].push_back(std::move(choice));
      out["usage"] = {{"prompt_tokens", nullptr}, {"completion_tokens", nullptr}, {"total_tokens", nullptr}};
      return SendJson(&res, 200, out);
    }

    res.status = 200;
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "close");
    res.set_header("X-Accel-Buffering", "no");
    res.set_header("x-turn-id", turn_id);

    auto id = NewId("chatcmpl");
    auto created = NowSeconds();

    if (!client_tools.empty()) {
      ScopedRequestAuthHeaders scope(auth_headers);
      std::string err;

      std::vector<ChatMessage> msgs;
      msgs.reserve(full_messages.size() + 2);
      msgs.push_back({"system", BuildToolSystemPromptClientManaged(client_tools, forced_tool)});
      for (const auto& m : full_messages) msgs.push_back(m);

      std::string assistant_text;
      if (model == "fake-tool") {
        assistant_text = FakeModelOnce(msgs);
      } else {
        ChatRequest creq;
        creq.model = provider_model;
        creq.stream = false;
        creq.max_tokens = max_tokens;
        creq.temperature = temperature;
        creq.top_p = top_p;
        creq.min_p = min_p;
        creq.messages = msgs;
        auto resp = provider->ChatOnce(creq, &err);
        if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
        assistant_text = resp->content;
      }

      auto calls = ParseToolCallsFromAssistantText(assistant_text);
      if (calls) NormalizeClientManagedToolCalls(client_tools, &(*calls));
      std::string final_text;
      if (!calls) {
        if (auto final = ExtractFinalFromAssistantJson(assistant_text)) {
          final_text = *final;
        } else {
          final_text = assistant_text;
        }
      }

      res.set_chunked_content_provider(
          "text/event-stream",
          [=,
           this,
           turn = std::move(turn),
           assistant_text = std::move(assistant_text),
           calls = std::move(calls),
           final_text = std::move(final_text)](size_t, httplib::DataSink& sink) mutable {
            try {
              auto write_bytes = [&](const std::string& s) -> bool {
                if (sink.is_writable && !sink.is_writable()) return false;
                if (!sink.write) return false;
                return sink.write(s.data(), s.size());
              };

              if (!write_bytes(std::string(": init\n") + std::string(2048, ' ') + "\n\n")) {
                sink.done();
                return false;
              }

              auto write_chunk = [&](const nlohmann::json& delta, const nlohmann::json& finish_reason) -> bool {
                nlohmann::json chunk;
                chunk["id"] = id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = model;
                nlohmann::json choice;
                choice["index"] = 0;
                choice["delta"] = delta;
                choice["finish_reason"] = finish_reason;
                chunk["choices"] = nlohmann::json::array({choice});
                return write_bytes(SseData(chunk));
              };

              nlohmann::json role_delta;
              role_delta["role"] = "assistant";
              if (!write_chunk(role_delta, nullptr)) {
                sink.done();
                return false;
              }

              if (calls) {
                constexpr size_t kArgChunk = 48;
                for (size_t i = 0; i < calls->size(); i++) {
                  const auto& c = (*calls)[i];
                  std::string args = c.arguments_json.empty() ? "{}" : c.arguments_json;
                  for (size_t off = 0; off < args.size(); off += kArgChunk) {
                    const bool first_piece = (off == 0);
                    nlohmann::json tool_delta;
                    nlohmann::json tc;
                    tc["index"] = static_cast<int>(i);
                    tc["id"] = c.id;
                    tc["type"] = "function";
                    nlohmann::json func;
                    if (first_piece) func["name"] = c.name;
                    func["arguments"] = args.substr(off, kArgChunk);
                    tc["function"] = std::move(func);
                    tool_delta["tool_calls"] = nlohmann::json::array({tc});
                    if (!write_chunk(tool_delta, nullptr)) {
                      sink.done();
                      return false;
                    }
                  }
                }
                if (!write_chunk(nlohmann::json::object(), "tool_calls")) {
                  sink.done();
                  return false;
                }
              } else {
                if (!final_text.empty()) {
                  nlohmann::json d;
                  d["content"] = final_text;
                  if (!write_chunk(d, nullptr)) {
                    sink.done();
                    return false;
                  }
                }
                if (!write_chunk(nlohmann::json::object(), "stop")) {
                  sink.done();
                  return false;
                }
              }

              if (calls) {
                turn.output_text = assistant_text;
              } else {
                turn.output_text = final_text;
              }
              sessions_->AppendTurn(session_id, turn);
              if (use_server_history) {
                sessions_->AppendToHistory(session_id, turn.input_messages);
                sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", calls ? assistant_text : final_text}});
              }

              write_bytes(SseDone());
              sink.done();
              return false;
            } catch (...) {
              sink.done();
              return false;
            }
          },
          [](bool) {});
      return;
    }

    if (allowed_tools.empty() && model != "fake-tool") {
      ChatRequest creq;
      creq.model = provider_model;
      creq.stream = true;
      creq.max_tokens = max_tokens;
      creq.temperature = temperature;
      creq.top_p = top_p;
      creq.min_p = min_p;
      creq.messages = full_messages;

      res.set_chunked_content_provider(
          "text/event-stream",
          [=,
           this,
           turn = std::move(turn),
           creq = std::move(creq)](size_t, httplib::DataSink& sink) mutable {
            try {
              ScopedRequestAuthHeaders scope(auth_headers);
              std::string acc;
              bool wrote_role = false;
              bool write_ok = true;
              std::string finish_reason = "stop";

              auto write_bytes = [&](const std::string& s) -> bool {
                if (sink.is_writable && !sink.is_writable()) return false;
                if (!sink.write) return false;
                return sink.write(s.data(), s.size());
              };

              write_ok = write_bytes(std::string(": init\n") + std::string(2048, ' ') + "\n\n");
              if (!write_ok) {
                sink.done();
                return false;
              }

              auto write_chunk = [&](const nlohmann::json& delta, const nlohmann::json& finish_reason) -> bool {
                nlohmann::json chunk;
                chunk["id"] = id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = model;
                nlohmann::json choice;
                choice["index"] = 0;
                choice["delta"] = delta;
                choice["finish_reason"] = finish_reason;
                chunk["choices"] = nlohmann::json::array({choice});
                return write_bytes(SseData(chunk));
              };

              std::string stream_err;
              const bool ok = provider->ChatStream(
                  creq,
                  [&](const std::string& delta_text) -> bool {
                    if (!write_ok) return false;
                    nlohmann::json delta;
                    if (!wrote_role) {
                      delta["role"] = "assistant";
                      wrote_role = true;
                    }
                    delta["content"] = delta_text;
                    if (!write_chunk(delta, nullptr)) {
                      write_ok = false;
                      return false;
                    }
                    acc += delta_text;
                    return true;
                  },
                  [&](const std::string& fr) {
                    finish_reason = fr;
                  },
                  &stream_err);

              if (!ok && !stream_err.empty()) {
                std::cout << "[provider-error] " << stream_err << "\n";
              }
              bool finish_ok = false;
              bool done_ok = false;
              if (write_ok) {
                finish_ok = write_chunk(nlohmann::json::object(), finish_reason);
                write_ok = finish_ok;
              }

              if (write_ok) {
                turn.output_text = acc;
                sessions_->AppendTurn(session_id, turn);
                if (use_server_history) {
                  sessions_->AppendToHistory(session_id, turn.input_messages);
                  sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", acc}});
                }
              }

              if (write_ok) {
                done_ok = write_bytes(SseDone());
                write_ok = done_ok;
              }
              std::cout << "[chat] session_id=" << session_id << " stream=1 max_tokens="
                        << (creq.max_tokens.has_value() ? creq.max_tokens.value() : -1) << " finish_reason=" << finish_reason
                        << " output_chars=" << acc.size() << " finish_ok=" << (finish_ok ? 1 : 0) << " done_ok=" << (done_ok ? 1 : 0)
                        << "\n";
              sink.done();
              return false;
            } catch (...) {
              sink.done();
              return false;
            }
          },
          [](bool) {});
      return;
    }

    std::optional<ToolLoopResult> precomputed_loop;
    std::string precomputed_err;
    if (trace) {
      ScopedRequestAuthHeaders scope(auth_headers);
      if (!allowed_tools.empty()) {
        if (planner) {
          precomputed_loop =
              RunPlanner(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p, max_plan_steps,
                         max_plan_rewrites, max_tool_calls, &precomputed_err);
          if (precomputed_loop->planner_failed) {
            precomputed_loop = RunToolLoop(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p,
                                           max_steps, max_tool_calls, &precomputed_err);
          }
        } else {
          precomputed_loop = RunToolLoop(provider_model, session_id, full_messages, allowed_tools, tools_, provider, max_tokens, temperature, top_p, min_p,
                                         max_steps, max_tool_calls, &precomputed_err);
        }
      } else {
        ToolLoopResult tmp;
        tmp.final_text = FakeModelOnce(full_messages);
        precomputed_loop = std::move(tmp);
      }

      if (!precomputed_err.empty() && precomputed_loop->final_text.empty()) {
        return SendJson(&res, 502, MakeError(precomputed_err, "api_error"));
      }
      res.set_header("x-runtime-trace", BuildRuntimeTrace(*precomputed_loop).dump());
    }

    res.set_chunked_content_provider(
        "text/event-stream",
        [=,
         this,
         turn = std::move(turn),
         precomputed_loop = std::move(precomputed_loop),
         precomputed_err = std::move(precomputed_err)](size_t, httplib::DataSink& sink) mutable {
          try {
            std::string acc;
            bool wrote_role = false;

            auto write_bytes = [&](const std::string& s) -> bool {
              if (sink.is_writable && !sink.is_writable()) return false;
              if (!sink.write) return false;
              return sink.write(s.data(), s.size());
            };

            if (!write_bytes(std::string(": init\n") + std::string(2048, ' ') + "\n\n")) {
              sink.done();
              return false;
            }

            auto write_chunk = [&](const nlohmann::json& delta, const nlohmann::json& finish_reason) -> bool {
              nlohmann::json chunk;
              chunk["id"] = id;
              chunk["object"] = "chat.completion.chunk";
              chunk["created"] = created;
              chunk["model"] = model;
              nlohmann::json choice;
              choice["index"] = 0;
              choice["delta"] = delta;
              choice["finish_reason"] = finish_reason;
              chunk["choices"] = nlohmann::json::array({choice});
              return write_bytes(SseData(chunk));
            };

            if (!wrote_role) {
              nlohmann::json delta;
              delta["role"] = "assistant";
              wrote_role = true;
              if (!write_chunk(delta, nullptr)) {
                sink.done();
                return false;
              }
            }

            ToolLoopResult loop;
            std::string err;
            if (precomputed_loop) {
              loop = std::move(*precomputed_loop);
              err = std::move(precomputed_err);
            } else {
              auto last_keepalive = std::chrono::steady_clock::now();
              auto last_progress = std::chrono::steady_clock::now();
              int model_timeout_s = 900;
              int tool_timeout_s = 300;
              int progress_ms = 2000;
              if (const char* v = std::getenv("RUNTIME_STREAM_MODEL_TIMEOUT_S"); v && *v) model_timeout_s = std::atoi(v);
              if (const char* v = std::getenv("RUNTIME_STREAM_TOOL_TIMEOUT_S"); v && *v) tool_timeout_s = std::atoi(v);
              if (const char* v = std::getenv("RUNTIME_STREAM_PROGRESS_MS"); v && *v) progress_ms = std::atoi(v);

              auto maybe_progress = [&]() -> bool {
                if (progress_ms <= 0) return true;
                auto now = std::chrono::steady_clock::now();
                if (now - last_progress < std::chrono::milliseconds(progress_ms)) return true;
                last_progress = now;
                return write_bytes(std::string(": progress ") + std::string(256, ' ') + "\n\n");
              };

              auto maybe_keepalive = [&]() -> bool {
                auto now = std::chrono::steady_clock::now();
                if (now - last_keepalive < std::chrono::seconds(1)) return true;
                last_keepalive = now;
                return write_bytes(": keepalive\n\n");
              };

              enum class WaitState { Ready, Disconnected, Timeout };
              auto wait_ready = [&](auto& fut, int timeout_s) -> WaitState {
                const auto start = std::chrono::steady_clock::now();
                while (fut.wait_for(std::chrono::milliseconds(250)) != std::future_status::ready) {
                  if (!maybe_keepalive()) return WaitState::Disconnected;
                  if (!maybe_progress()) return WaitState::Disconnected;
                  if (timeout_s > 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (now - start >= std::chrono::seconds(timeout_s)) return WaitState::Timeout;
                  }
                }
                return WaitState::Ready;
              };

              auto write_tool_call = [&](int index, const ToolCall& c) -> bool {
                constexpr size_t kArgChunk = 48;
                std::string args = c.arguments_json.empty() ? "{}" : c.arguments_json;
                for (size_t off = 0; off < args.size(); off += kArgChunk) {
                  const bool first_piece = (off == 0);
                  nlohmann::json tool_delta;
                  nlohmann::json tc;
                  tc["index"] = index;
                  tc["id"] = c.id;
                  tc["type"] = "function";
                  nlohmann::json func;
                  if (first_piece) func["name"] = c.name;
                  func["arguments"] = args.substr(off, kArgChunk);
                  tc["function"] = std::move(func);
                  tool_delta["tool_calls"] = nlohmann::json::array({tc});
                  if (!write_chunk(tool_delta, nullptr)) return false;
                }
                return true;
              };

              auto write_tool_result = [&](const ToolResult& r) -> bool {
                nlohmann::json delta;
                delta["tool_result"] = {{"id", r.tool_call_id}, {"name", r.name}, {"ok", r.ok}, {"error", r.error}};
                return write_chunk(delta, nullptr);
              };

              auto run_chat_once = [&](const std::vector<ChatMessage>& messages, std::string* out_text) -> bool {
                if (provider_model == "fake-tool") {
                  *out_text = FakeModelOnce(messages);
                  return true;
                }
                std::promise<std::pair<std::string, std::string>> p;
                auto fut = p.get_future();
                std::thread([=, p = std::move(p)]() mutable {
                  ScopedRequestAuthHeaders scope(auth_headers);
                  std::string local_err;
                  ChatRequest req;
                  req.model = provider_model;
                  req.stream = false;
                  req.max_tokens = max_tokens;
                  req.temperature = temperature;
                  req.top_p = top_p;
                  req.min_p = min_p;
                  if (provider && provider->Name() == "llama_cpp" && !allowed_tools.empty()) {
                    req.grammar = BuildToolLoopGrammar(allowed_tools);
                  }
                  req.messages = messages;
                  auto resp = provider->ChatOnce(req, &local_err);
                  if (!resp) {
                    p.set_value(std::make_pair(std::string(), local_err.empty() ? std::string("upstream error") : local_err));
                    return;
                  }
                  p.set_value(std::make_pair(resp->content, std::string()));
                }).detach();

                auto st = wait_ready(fut, model_timeout_s);
                if (st == WaitState::Disconnected) return false;
                if (st == WaitState::Timeout) {
                  if (out_text) *out_text = "";
                  err = "timeout waiting for model";
                  return true;
                }

                auto got = fut.get();
                if (!got.second.empty()) {
                  if (out_text) *out_text = "";
                  err = std::move(got.second);
                  return true;
                }
                if (out_text) *out_text = std::move(got.first);
                return true;
              };

              auto execute_tool = [&](const ToolCall& c, const nlohmann::json& jargs, ToolResult* out_r) -> bool {
                auto handler = tools_.GetHandler(c.name);
                if (!handler) {
                  ToolResult r;
                  r.tool_call_id = c.id;
                  r.name = c.name;
                  r.ok = false;
                  r.error = "tool not found";
                  r.result = {{"ok", false}, {"error", r.error}};
                  *out_r = std::move(r);
                  return true;
                }
                std::promise<ToolResult> p;
                auto fut = p.get_future();
                std::thread([&, p = std::move(p)]() mutable {
                  ScopedRequestAuthHeaders scope(auth_headers);
                  p.set_value((*handler)(c.id, jargs));
                }).detach();

                auto st = wait_ready(fut, tool_timeout_s);
                if (st == WaitState::Disconnected) return false;
                if (st == WaitState::Timeout) {
                  ToolResult r;
                  r.tool_call_id = c.id;
                  r.name = c.name;
                  r.ok = false;
                  r.error = "timeout waiting for tool";
                  r.result = {{"ok", false}, {"error", r.error}};
                  *out_r = std::move(r);
                  return true;
                }

                *out_r = fut.get();
                return true;
              };

              std::unordered_set<std::string> allowed_names;
              for (const auto& t : allowed_tools) allowed_names.insert(t.name);

              std::vector<ChatMessage> msgs;
              msgs.reserve(full_messages.size() + 8);
              if (!allowed_tools.empty()) msgs.push_back({"system", BuildToolSystemPrompt(allowed_tools)});
              for (const auto& m : full_messages) msgs.push_back(m);

              if (max_steps <= 0) max_steps = 1;
              if (max_tool_calls < 0) max_tool_calls = 0;

              int tool_calls_used = 0;

              auto run_tool_loop_stream = [&]() -> bool {
                for (int step = 0; step < max_steps; step++) {
                  loop.steps = step + 1;
                  std::string assistant_text;
                  if (!run_chat_once(msgs, &assistant_text)) return false;
                  if (!err.empty() && assistant_text.empty()) return true;

                  if (auto calls = ParseToolCallsFromAssistantText(assistant_text)) {
                    for (auto& c : *calls) {
                      int idx = static_cast<int>(loop.executed_calls.size());
                      loop.executed_calls.push_back(c);
                      if (!write_tool_call(idx, c)) return false;

                      if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
                        ToolResult r;
                        r.tool_call_id = c.id;
                        r.name = c.name;
                        r.ok = false;
                        r.error = "tool not allowed";
                        r.result = {{"ok", false}, {"error", r.error}};
                        loop.results.push_back(r);
                        msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                        if (!write_tool_result(r)) return false;
                        continue;
                      }
                      if (!tools_.HasTool(c.name)) {
                        ToolResult r;
                        r.tool_call_id = c.id;
                        r.name = c.name;
                        r.ok = false;
                        r.error = "tool not found";
                        r.result = {{"ok", false}, {"error", r.error}};
                        loop.results.push_back(r);
                        msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                        if (!write_tool_result(r)) return false;
                        continue;
                      }
                      if (tool_calls_used >= max_tool_calls) {
                        loop.hit_tool_limit = true;
                        loop.final_text = "tool call limit exceeded";
                        return true;
                      }

                      auto schema = tools_.GetSchema(c.name);
                      auto jargs = ParseJsonLoose(c.arguments_json);
                      if (schema && jargs && jargs->is_string()) {
                        std::string raw = jargs->get<std::string>();
                        const auto& p = schema->parameters;
                        if (p.is_object() && p.contains("type") && p["type"].is_string() && p["type"].get<std::string>() == "string") {
                          jargs = nlohmann::json(raw);
                        } else if (auto key = GuessSingleKeyFromParams(p, raw)) {
                          nlohmann::json obj = nlohmann::json::object();
                          obj[*key] = raw;
                          jargs = std::move(obj);
                        }
                      }
                      if (!jargs) {
                        if (schema) {
                          std::string raw = c.arguments_json;
                          size_t s = 0;
                          while (s < raw.size() && std::isspace(static_cast<unsigned char>(raw[s]))) s++;
                          size_t e = raw.size();
                          while (e > s && std::isspace(static_cast<unsigned char>(raw[e - 1]))) e--;
                          raw = raw.substr(s, e - s);

                          const auto& p = schema->parameters;
                          if (p.is_object() && p.contains("type") && p["type"].is_string() && p["type"].get<std::string>() == "string") {
                            jargs = nlohmann::json(raw);
                          } else if (auto key = GuessSingleKeyFromParams(p, raw)) {
                            jargs = nlohmann::json::object();
                            (*jargs)[*key] = raw;
                          }
                        }
                      }

                      if (schema && jargs && jargs->is_object()) NormalizeToolArgsObject(*schema, &(*jargs));
                      if (!jargs) {
                        ToolResult r;
                        r.tool_call_id = c.id;
                        r.name = c.name;
                        r.ok = false;
                        r.error = "invalid tool arguments json";
                        r.result = {{"ok", false}, {"error", r.error}};
                        loop.results.push_back(r);
                        msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                        if (!write_tool_result(r)) return false;
                        continue;
                      }

                      ToolResult r;
                      if (!execute_tool(c, *jargs, &r)) return false;
                      tool_calls_used++;
                      loop.results.push_back(r);
                      msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                      if (!write_tool_result(r)) return false;
                    }
                    continue;
                  }

                  if (auto final = ExtractFinalFromAssistantJson(assistant_text)) {
                    loop.final_text = *final;
                    return true;
                  }

                  loop.final_text = assistant_text;
                  return true;
                }
                loop.hit_step_limit = true;
                loop.final_text = "tool loop exceeded max steps";
                return true;
              };

              auto run_planner_stream = [&]() -> bool {
                loop.used_planner = true;
                if (max_plan_steps <= 0) max_plan_steps = 1;
                if (max_plan_rewrites < 0) max_plan_rewrites = 0;
                if (max_tool_calls < 0) max_tool_calls = 0;

                std::vector<ChatMessage> plan_msgs;
                plan_msgs.reserve(full_messages.size() + 2);
                plan_msgs.push_back({"system", BuildPlannerSystemPrompt(allowed_tools, max_plan_steps)});
                for (const auto& m : full_messages) plan_msgs.push_back(m);

                std::optional<std::vector<PlannerPlanStep>> plan;
                std::string plan_text;
                int rewrites = 0;
                for (int attempt = 0; attempt <= max_plan_rewrites; attempt++) {
                  if (!run_chat_once(plan_msgs, &plan_text)) return false;
                  if (!err.empty() && plan_text.empty()) {
                    loop.planner_failed = true;
                    return true;
                  }
                  if (auto final = ExtractFinalFromAssistantJson(plan_text)) {
                    loop.final_text = *final;
                    loop.steps = 1;
                    loop.plan_steps = 0;
                    return true;
                  }
                  plan = ParsePlannerPlan(plan_text);
                  if (!plan) {
                    if (attempt == max_plan_rewrites) {
                      loop.planner_failed = true;
                      loop.final_text = plan_text;
                      loop.steps = 1;
                      return true;
                    }
                    plan_msgs.push_back({"user", "Plan invalid JSON. Return a corrected plan JSON only."});
                    continue;
                  }
                  bool ok = true;
                  std::string why;
                  for (const auto& s : *plan) {
                    if (!allowed_names.empty() && allowed_names.find(s.name) == allowed_names.end()) {
                      ok = false;
                      why = "tool not allowed: " + s.name;
                      break;
                    }
                    auto schema = tools_.GetSchema(s.name);
                    if (!schema) {
                      ok = false;
                      why = "tool not found: " + s.name;
                      break;
                    }
                    std::string schema_err;
                    if (!ValidateSchemaLoose(schema->parameters, s.arguments, &schema_err)) {
                      ok = false;
                      why = "invalid arguments for " + s.name + ": " + schema_err;
                      break;
                    }
                  }
                  if (ok) break;
                  if (attempt == max_plan_rewrites) {
                    loop.planner_failed = true;
                    loop.final_text = why;
                    loop.steps = 1;
                    return true;
                  }
                  plan_msgs.push_back({"user", "Plan rejected: " + why + ". Return a corrected plan JSON only."});
                  plan.reset();
                  rewrites = attempt + 1;
                }

                if (!plan) {
                  loop.planner_failed = true;
                  loop.final_text = plan_text;
                  loop.steps = 1;
                  return true;
                }

                if (static_cast<int>(plan->size()) > max_plan_steps) plan->resize(static_cast<size_t>(max_plan_steps));
                loop.plan_steps = static_cast<int>(plan->size());
                loop.plan_rewrites = rewrites;
                loop.plan = nlohmann::json::array();
                for (const auto& s : *plan) loop.plan.push_back({{"name", s.name}, {"arguments", s.arguments}});

                std::vector<ChatMessage> exec_msgs;
                exec_msgs.reserve(full_messages.size() + loop.plan_steps + 4);
                for (const auto& m : full_messages) exec_msgs.push_back(m);

                int tool_calls_used_local = 0;
                for (size_t i = 0; i < plan->size(); i++) {
                  if (tool_calls_used_local >= max_tool_calls) {
                    loop.hit_tool_limit = true;
                    loop.final_text = "tool call limit exceeded";
                    loop.steps = static_cast<int>(i + 1);
                    return true;
                  }

                  const auto& s = (*plan)[i];
                  ToolCall c;
                  c.id = "plan_" + std::to_string(i + 1);
                  c.name = s.name;
                  c.arguments_json = s.arguments.dump();
                  int idx = static_cast<int>(loop.executed_calls.size());
                  loop.executed_calls.push_back(c);
                  if (!write_tool_call(idx, c)) return false;

                  ToolResult r;
                  r.tool_call_id = c.id;
                  r.name = c.name;

                  if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
                    r.ok = false;
                    r.error = "tool not allowed";
                    r.result = {{"ok", false}, {"error", r.error}};
                    loop.results.push_back(r);
                    exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                    tool_calls_used_local++;
                    if (!write_tool_result(r)) return false;
                    continue;
                  }

                  if (!tools_.HasTool(c.name)) {
                    r.ok = false;
                    r.error = "tool not found";
                    r.result = {{"ok", false}, {"error", r.error}};
                    loop.results.push_back(r);
                    exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
                    tool_calls_used_local++;
                    if (!write_tool_result(r)) return false;
                    continue;
                  }

                  ToolResult got_r;
                  if (!execute_tool(c, s.arguments, &got_r)) return false;
                  loop.results.push_back(got_r);
                  exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + got_r.result.dump()});
                  tool_calls_used_local++;
                  if (!write_tool_result(got_r)) return false;
                }

                std::vector<ChatMessage> final_msgs;
                final_msgs.reserve(exec_msgs.size() + 2);
                final_msgs.push_back({"system", BuildPlannerFinalSystemPrompt()});
                for (const auto& m : exec_msgs) final_msgs.push_back(m);

                std::string final_text;
                if (!run_chat_once(final_msgs, &final_text)) return false;
                if (!err.empty() && final_text.empty()) return true;

                loop.steps = 2;
                if (auto final = ExtractFinalFromAssistantJson(final_text)) {
                  loop.final_text = *final;
                  return true;
                }
                loop.final_text = final_text;
                return true;
              };

              if (!allowed_tools.empty()) {
                if (planner) {
                  if (!run_planner_stream()) return false;
                  if (loop.planner_failed && err.empty()) {
                    loop = ToolLoopResult();
                    if (!run_tool_loop_stream()) return false;
                  }
                } else {
                  if (!run_tool_loop_stream()) return false;
                }
              } else {
                loop.final_text = FakeModelOnce(full_messages);
              }
            }

            if (!err.empty() && loop.final_text.empty()) {
              nlohmann::json delta;
              delta["content"] = err;
              write_chunk(delta, "stop");
              write_bytes(SseDone());
              sink.done();
              return false;
            }

            constexpr size_t kTextChunk = 64;
            for (size_t off = 0; off < loop.final_text.size(); off += kTextChunk) {
              auto piece = loop.final_text.substr(off, kTextChunk);
              nlohmann::json delta;
              delta["content"] = piece;
              if (!write_chunk(delta, nullptr)) {
                sink.done();
                return false;
              }
              acc += piece;
            }

            if (!write_chunk(nlohmann::json::object(), "stop")) {
              sink.done();
              return false;
            }

            turn.output_text = acc;
            sessions_->AppendTurn(session_id, turn);
            if (use_server_history) {
              sessions_->AppendToHistory(session_id, turn.input_messages);
              for (const auto& tc : loop.executed_calls) {
                sessions_->AppendToHistory(session_id,
                                           {ChatMessage{"assistant", "TOOL_CALL " + tc.name + " " + tc.arguments_json}});
              }
              for (const auto& tr : loop.results) {
                sessions_->AppendToHistory(session_id, {ChatMessage{"user", "TOOL_RESULT " + tr.name + " " + tr.result.dump()}});
              }
              sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", acc}});
            }

            if (!write_bytes(SseDone())) {
              sink.done();
              return false;
            }
            sink.done();
            return false;
          } catch (...) {
            sink.done();
            return false;
          }
        },
        [](bool) {});
  };

  auto responses_handler = [&](const httplib::Request& req, httplib::Response& res) {
    LogRequestRaw(req);
    ScopedRequestAuthHeaders scope(ExtractUpstreamAuthHeaders(req));
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    std::string input;
    if (j.contains("input") && j["input"].is_string()) {
      input = j["input"].get<std::string>();
    } else if (j.contains("input") && j["input"].is_array() && !j["input"].empty()) {
      const auto& v = j["input"][0];
      if (v.is_string()) input = v.get<std::string>();
      if (v.is_object() && v.contains("content") && v["content"].is_string()) input = v["content"].get<std::string>();
    } else {
      return SendJson(&res, 400, MakeError("missing field: input", "invalid_request_error"));
    }

    std::string err;
    std::string content;
    if (model == "fake-tool") {
      content = FakeModelOnce({ChatMessage{"user", input}});
    } else {
      auto resolved = providers_ ? providers_->Resolve(model) : std::nullopt;
      if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
      if (providers_) {
        auto sw = providers_->Activate(resolved->provider_name);
        if (sw.switched) {
          std::cout << "[provider-switch] from=" << sw.from << " to=" << sw.to << "\n";
        }
      }
      LogProviderUse(resolved->provider_name, resolved->model);
      ChatRequest creq;
      creq.model = resolved->model;
      creq.stream = false;
      creq.messages = {ChatMessage{"user", input}};
      auto resp = resolved->provider->ChatOnce(creq, &err);
      if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
      content = resp->content;
    }

    nlohmann::json out;
    out["id"] = NewId("resp");
    out["object"] = "response";
    out["created"] = NowSeconds();
    out["model"] = model;
    out["output"] = nlohmann::json::array();
    nlohmann::json msg;
    msg["id"] = NewId("msg");
    msg["type"] = "message";
    msg["role"] = "assistant";
    msg["content"] = nlohmann::json::array();
    msg["content"].push_back({{"type", "output_text"}, {"text", content}});
    out["output"].push_back(std::move(msg));
    SendJson(&res, 200, out);
  };

  auto anthropic_messages_handler = [&](const httplib::Request& req, httplib::Response& res) {
    LogRequestRaw(req);
    const auto auth_headers = ExtractUpstreamAuthHeaders(req);
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeAnthropicError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeAnthropicError("missing field: model", "invalid_request_error"));
    }
    if (!j.contains("messages") || !j["messages"].is_array()) {
      return SendJson(&res, 400, MakeAnthropicError("missing field: messages", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    bool ok = false;
    auto req_messages = ParseChatMessages(j, &ok);
    if (!ok) return SendJson(&res, 400, MakeAnthropicError("missing field: messages", "invalid_request_error"));

    std::string system_text;
    if (j.contains("system")) system_text = ExtractMessageContent(j["system"]);

    std::vector<ChatMessage> full_messages;
    if (!system_text.empty()) full_messages.push_back(ChatMessage{"system", system_text});
    full_messages.insert(full_messages.end(), req_messages.begin(), req_messages.end());

    bool stream = false;
    if (j.contains("stream") && j["stream"].is_boolean()) stream = j["stream"].get<bool>();
    std::optional<int> max_tokens;
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) max_tokens = j["max_tokens"].get<int>();

    IProvider* provider = nullptr;
    std::string provider_model = model;
    if (model != "fake-tool") {
      auto resolved = providers_ ? providers_->Resolve(model) : std::nullopt;
      if (!resolved) return SendJson(&res, 400, MakeAnthropicError("unknown provider in model", "invalid_request_error"));
      provider = resolved->provider;
      provider_model = resolved->model;
      if (providers_) {
        auto sw = providers_->Activate(resolved->provider_name);
        if (sw.switched) {
          std::cout << "[provider-switch] from=" << sw.from << " to=" << sw.to << "\n";
        }
      }
      LogProviderUse(resolved->provider_name, resolved->model);
    }

    if (!stream) {
      ScopedRequestAuthHeaders scope(auth_headers);
      std::string err;
      std::string content;
      std::string finish_reason = "stop";
      if (model == "fake-tool") {
        content = FakeModelOnce(full_messages);
      } else {
        ChatRequest creq;
        creq.model = provider_model;
        creq.stream = false;
        creq.max_tokens = max_tokens;
        creq.messages = full_messages;
        auto resp = provider->ChatOnce(creq, &err);
        if (!resp) return SendJson(&res, 502, MakeAnthropicError(err.empty() ? "upstream error" : err, "api_error"));
        content = resp->content;
        finish_reason = resp->finish_reason;
      }

      nlohmann::json out;
      out["id"] = NewId("msg");
      out["type"] = "message";
      out["role"] = "assistant";
      out["content"] = nlohmann::json::array({{{"type", "text"}, {"text", content}}});
      out["model"] = model;
      out["stop_reason"] = MapFinishReasonToAnthropicStopReason(finish_reason);
      out["stop_sequence"] = nullptr;
      out["usage"] = {{"input_tokens", nullptr}, {"output_tokens", nullptr}};
      return SendJson(&res, 200, out);
    }

    res.status = 200;
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "close");
    res.set_header("X-Accel-Buffering", "no");

    auto id = NewId("msg");
    if (model == "fake-tool") {
      res.set_chunked_content_provider(
          "text/event-stream",
          [=](size_t, httplib::DataSink& sink) mutable {
            auto write_bytes = [&](const std::string& s) -> bool {
              if (sink.is_writable && !sink.is_writable()) return false;
              if (!sink.write) return false;
              return sink.write(s.data(), s.size());
            };

            const auto content = FakeModelOnce(full_messages);
            nlohmann::json start;
            start["type"] = "message_start";
            start["message"] = {{"id", id},
                                {"type", "message"},
                                {"role", "assistant"},
                                {"content", nlohmann::json::array()},
                                {"model", model},
                                {"stop_reason", nullptr},
                                {"stop_sequence", nullptr},
                                {"usage", {{"input_tokens", nullptr}, {"output_tokens", nullptr}}}};
            if (!write_bytes(SseEvent("message_start", start))) {
              sink.done();
              return false;
            }

            nlohmann::json cbs;
            cbs["type"] = "content_block_start";
            cbs["index"] = 0;
            cbs["content_block"] = {{"type", "text"}, {"text", ""}};
            if (!write_bytes(SseEvent("content_block_start", cbs))) {
              sink.done();
              return false;
            }

            nlohmann::json delta;
            delta["type"] = "content_block_delta";
            delta["index"] = 0;
            delta["delta"] = {{"type", "text_delta"}, {"text", content}};
            if (!write_bytes(SseEvent("content_block_delta", delta))) {
              sink.done();
              return false;
            }

            nlohmann::json cbstop;
            cbstop["type"] = "content_block_stop";
            cbstop["index"] = 0;
            if (!write_bytes(SseEvent("content_block_stop", cbstop))) {
              sink.done();
              return false;
            }

            nlohmann::json md;
            md["type"] = "message_delta";
            md["delta"] = {{"stop_reason", "end_turn"}, {"stop_sequence", nullptr}};
            md["usage"] = {{"output_tokens", nullptr}};
            if (!write_bytes(SseEvent("message_delta", md))) {
              sink.done();
              return false;
            }

            nlohmann::json stop;
            stop["type"] = "message_stop";
            if (!write_bytes(SseEvent("message_stop", stop))) {
              sink.done();
              return false;
            }
            sink.done();
            return false;
          },
          [](bool) {});
      return;
    }

    ChatRequest creq;
    creq.model = provider_model;
    creq.stream = true;
    creq.max_tokens = max_tokens;
    creq.messages = full_messages;

    res.set_chunked_content_provider(
        "text/event-stream",
        [=](size_t, httplib::DataSink& sink) mutable {
          try {
            ScopedRequestAuthHeaders scope(auth_headers);
            auto write_bytes = [&](const std::string& s) -> bool {
              if (sink.is_writable && !sink.is_writable()) return false;
              if (!sink.write) return false;
              return sink.write(s.data(), s.size());
            };

            nlohmann::json start;
            start["type"] = "message_start";
            start["message"] = {{"id", id},
                                {"type", "message"},
                                {"role", "assistant"},
                                {"content", nlohmann::json::array()},
                                {"model", model},
                                {"stop_reason", nullptr},
                                {"stop_sequence", nullptr},
                                {"usage", {{"input_tokens", nullptr}, {"output_tokens", nullptr}}}};
            if (!write_bytes(SseEvent("message_start", start))) {
              sink.done();
              return false;
            }

            nlohmann::json cbs;
            cbs["type"] = "content_block_start";
            cbs["index"] = 0;
            cbs["content_block"] = {{"type", "text"}, {"text", ""}};
            if (!write_bytes(SseEvent("content_block_start", cbs))) {
              sink.done();
              return false;
            }

            std::string finish_reason = "stop";
            std::string stream_err;
            bool wrote_any = false;
            const bool ok_stream = provider->ChatStream(
                creq,
                [&](const std::string& delta_text) -> bool {
                  wrote_any = true;
                  nlohmann::json delta;
                  delta["type"] = "content_block_delta";
                  delta["index"] = 0;
                  delta["delta"] = {{"type", "text_delta"}, {"text", delta_text}};
                  if (!write_bytes(SseEvent("content_block_delta", delta))) return false;
                  return true;
                },
                [&](const std::string& fr) { finish_reason = fr; },
                &stream_err);
            if (!ok_stream && !stream_err.empty()) {
              sink.done();
              return false;
            }

            if (!wrote_any) {
              nlohmann::json delta;
              delta["type"] = "content_block_delta";
              delta["index"] = 0;
              delta["delta"] = {{"type", "text_delta"}, {"text", ""}};
              if (!write_bytes(SseEvent("content_block_delta", delta))) {
                sink.done();
                return false;
              }
            }

            nlohmann::json cbstop;
            cbstop["type"] = "content_block_stop";
            cbstop["index"] = 0;
            if (!write_bytes(SseEvent("content_block_stop", cbstop))) {
              sink.done();
              return false;
            }

            nlohmann::json md;
            md["type"] = "message_delta";
            md["delta"] = {{"stop_reason", MapFinishReasonToAnthropicStopReason(finish_reason)}, {"stop_sequence", nullptr}};
            md["usage"] = {{"output_tokens", nullptr}};
            if (!write_bytes(SseEvent("message_delta", md))) {
              sink.done();
              return false;
            }

            nlohmann::json stop;
            stop["type"] = "message_stop";
            if (!write_bytes(SseEvent("message_stop", stop))) {
              sink.done();
              return false;
            }

            sink.done();
            return false;
          } catch (...) {
            sink.done();
            return false;
          }
        },
        [](bool) {});
  };

  for (const auto& raw_prefix : prefixes) {
    const auto prefix = NormalizePrefix(raw_prefix);
    server->Get(prefix + "/v1/models", models_handler);
    server->Post(prefix + "/v1/embeddings", embeddings_handler);
    server->Post(prefix + "/v1/chat/completions", chat_completions_handler);
    server->Post(prefix + "/v1/responses", responses_handler);
    server->Post(prefix + "/v1/messages", anthropic_messages_handler);
  }
}

}  // namespace runtime

#include "config.hpp"
#if LOCAL_AI_RUNTIME_WITH_LLAMA_CPP
#include "llama_cpp_provider.hpp"
#endif
#include "mcp_client.hpp"
#include "openai_compatible_http_provider.hpp"
#include "ollama_provider.hpp"
#include "openai_router.hpp"
#include "providers/registry.hpp"
#include "session_manager.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>

namespace {

static std::string Trim(std::string s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
  return s.substr(start, end - start);
}

static bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static std::string ToLower(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
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

static std::string SanitizeJsonForLog(const nlohmann::json& body) {
  if (body.is_null()) return "null";
  if (!body.is_object()) return body.dump();
  auto j = body;
  for (const auto& key : {"api_key", "api-key", "authorization", "apiKey"}) {
    if (j.contains(key)) j.erase(key);
  }
  if (j.contains("headers") && j["headers"].is_object()) {
    auto& h = j["headers"];
    for (const auto& key : {"authorization", "proxy-authorization", "api-key", "api_key", "x-api-key"}) {
      if (h.contains(key)) h.erase(key);
    }
  }
  return j.dump();
}

static void LogMcpCall(const std::string& tool_call_id,
                       const std::string& exposed_name,
                       const std::string& remote_name,
                       const nlohmann::json& arguments) {
  std::cout << "[mcp-call] id=" << tool_call_id << " exposed=" << exposed_name << " remote=" << remote_name
            << " arguments=" << TruncateForLog(SanitizeJsonForLog(arguments), 2000) << "\n";
}

static void LogMcpResult(const std::string& tool_call_id,
                         const std::string& exposed_name,
                         const std::string& remote_name,
                         bool ok,
                         const std::string& error,
                         const nlohmann::json& result) {
  std::cout << "[mcp-result] id=" << tool_call_id << " exposed=" << exposed_name << " remote=" << remote_name
            << " ok=" << (ok ? 1 : 0) << " error=" << (error.empty() ? "-" : error)
            << " result=" << TruncateForLog(SanitizeJsonForLog(result), 2000) << "\n";
}

static std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream iss(text);
  std::string line;
  while (std::getline(iss, line)) out.push_back(line);
  return out;
}

static int StatusScore(const std::string& status) {
  if (status == "completed") return 3;
  if (status == "in_progress") return 2;
  if (status == "pending") return 1;
  return 0;
}

static std::optional<std::pair<std::string, std::string>> ParseTodoLine(const std::string& raw_line) {
  auto line = Trim(raw_line);
  if (line.empty()) return std::nullopt;

  auto lower = ToLower(line);
  const std::vector<std::pair<std::string, std::string>> kCheckboxPrefixes = {
      {"- [ ]", "pending"}, {"* [ ]", "pending"}, {"- [x]", "completed"}, {"* [x]", "completed"}};
  for (const auto& [prefix, status] : kCheckboxPrefixes) {
    if (StartsWith(lower, prefix)) {
      auto text = Trim(line.substr(prefix.size()));
      if (text.empty()) return std::nullopt;
      return std::make_pair(text, status);
    }
  }

  if (StartsWith(line, "- ") || StartsWith(line, "* ")) {
    auto text = Trim(line.substr(2));
    if (text.empty()) return std::nullopt;
    if (lower.find("in progress") != std::string::npos || lower.find("in_progress") != std::string::npos) {
      return std::make_pair(text, "in_progress");
    }
    if (lower.find("completed") != std::string::npos || lower.find("done") != std::string::npos) {
      return std::make_pair(text, "completed");
    }
    if (lower.find("pending") != std::string::npos) {
      return std::make_pair(text, "pending");
    }
    return std::make_pair(text, "unknown");
  }

  return std::nullopt;
}

static std::optional<int> GetEnvInt(const char* key) {
  const char* v = std::getenv(key);
  if (!v || !*v) return std::nullopt;
  char* end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v) return std::nullopt;
  if (n > INT32_MAX) n = INT32_MAX;
  if (n < INT32_MIN) n = INT32_MIN;
  return static_cast<int>(n);
}

static void MergeTodo(std::unordered_map<std::string, std::string>* best, const std::string& text, const std::string& status) {
  auto it = best->find(text);
  if (it == best->end()) {
    (*best)[text] = status;
    return;
  }
  if (StatusScore(status) > StatusScore(it->second)) it->second = status;
}

static nlohmann::json InferTodosFromSession(const runtime::Session& s, int max_history_messages) {
  std::unordered_map<std::string, std::string> best;

  int start = 0;
  if (max_history_messages > 0 && static_cast<int>(s.history.size()) > max_history_messages) {
    start = static_cast<int>(s.history.size()) - max_history_messages;
  }

  for (size_t i = static_cast<size_t>(start); i < s.history.size(); i++) {
    const auto& m = s.history[i];
    if (m.role != "assistant" && m.role != "user") continue;
    for (const auto& line : SplitLines(m.content)) {
      auto item = ParseTodoLine(line);
      if (!item) continue;
      MergeTodo(&best, item->first, item->second);
    }
  }

  nlohmann::json todos = nlohmann::json::array();
  for (const auto& [text, status] : best) {
    todos.push_back({{"text", text}, {"status", status}});
  }
  return todos;
}

static nlohmann::json ExtractRecentToolResults(const runtime::Session& s, int max_items) {
  nlohmann::json out = nlohmann::json::array();
  if (max_items <= 0) return out;

  for (int i = static_cast<int>(s.history.size()) - 1; i >= 0 && static_cast<int>(out.size()) < max_items; i--) {
    const auto& m = s.history[static_cast<size_t>(i)];
    if (m.role != "user") continue;
    const std::string prefix = "TOOL_RESULT ";
    if (!StartsWith(m.content, prefix)) continue;

    auto rest = m.content.substr(prefix.size());
    auto sp = rest.find(' ');
    if (sp == std::string::npos) continue;
    auto name = rest.substr(0, sp);
    auto payload = Trim(rest.substr(sp + 1));
    auto jr = nlohmann::json::parse(payload, nullptr, false);
    bool ok = true;
    if (jr.is_object() && jr.contains("ok") && jr["ok"].is_boolean()) ok = jr["ok"].get<bool>();
    out.push_back({{"name", name}, {"ok", ok}, {"result", jr.is_discarded() ? nlohmann::json(payload) : jr}});
  }
  return out;
}

}  // namespace

int main() {
  std::cout.setf(std::ios::unitbuf);
  auto cfg = runtime::LoadConfigFromEnv();

  runtime::SessionStoreConfig store_cfg;
  store_cfg.type = cfg.session_store_type;
  store_cfg.file_path = cfg.session_store_path;
  store_cfg.endpoint = cfg.session_store_endpoint;
  store_cfg.password = cfg.session_store_password;
  store_cfg.db = cfg.session_store_db;
  store_cfg.store_namespace = cfg.session_store_namespace;
  store_cfg.reset_on_boot = cfg.session_store_reset_on_boot;
  runtime::SessionManager sessions(store_cfg);
  runtime::ProviderRegistry providers(cfg.default_provider);
#if LOCAL_AI_RUNTIME_WITH_LLAMA_CPP
  providers.Register(std::make_unique<runtime::LlamaCppProvider>(cfg.llama_cpp_model_path));
#endif
  providers.Register(std::make_unique<runtime::OllamaProvider>(cfg.ollama));
  if (cfg.mnn_enabled) providers.Register(std::make_unique<runtime::OpenAiCompatibleHttpProvider>("mnn", cfg.mnn));
  if (cfg.lmdeploy_enabled) providers.Register(std::make_unique<runtime::OpenAiCompatibleHttpProvider>("lmdeploy", cfg.lmdeploy));
  runtime::OpenAiRouter router(&sessions, &providers, runtime::BuildDefaultToolRegistry(cfg));

  {
    auto* tools = router.MutableTools();
    runtime::ToolSchema schema;
    schema.name = "runtime.infer_task_status";
    schema.description = "Infer todo/task status from server session context.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"session_id", {{"type", "string"}}},
                           {"max_history_messages", {{"type", "integer"}}},
                           {"max_recent_tool_results", {{"type", "integer"}}}}},
                         {"required", {"session_id"}}};
    tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      runtime::ToolResult tr;
      tr.tool_call_id = tool_call_id;
      tr.name = schema.name;

      if (!arguments.is_object() || !arguments.contains("session_id") || !arguments["session_id"].is_string()) {
        tr.ok = false;
        tr.error = "missing required field: session_id";
        tr.result = {{"ok", false}, {"error", tr.error}};
        return tr;
      }

      int max_history_messages = 200;
      int max_recent_tool_results = 20;
      if (arguments.contains("max_history_messages") && arguments["max_history_messages"].is_number_integer()) {
        max_history_messages = arguments["max_history_messages"].get<int>();
      }
      if (arguments.contains("max_recent_tool_results") && arguments["max_recent_tool_results"].is_number_integer()) {
        max_recent_tool_results = arguments["max_recent_tool_results"].get<int>();
      }

      auto session_id = arguments["session_id"].get<std::string>();
      auto s = sessions.GetOrCreate(session_id);

      nlohmann::json result;
      result["ok"] = true;
      result["session_id"] = s.session_id;
      result["history_messages"] = static_cast<int>(s.history.size());
      result["turns"] = static_cast<int>(s.turns.size());
      if (!s.turns.empty()) result["last_turn_id"] = s.turns.back().turn_id;
      result["todos"] = InferTodosFromSession(s, max_history_messages);
      result["recent_tool_results"] = ExtractRecentToolResults(s, max_recent_tool_results);
      tr.result = std::move(result);
      return tr;
    });
  }

  std::cout << "[runtime] default_provider=" << cfg.default_provider << "\n";
  std::cout << "[provider] llama_cpp model_path=" << (cfg.llama_cpp_model_path.empty() ? "<empty>" : cfg.llama_cpp_model_path)
            << "\n";
  std::cout << "[provider] ollama endpoint=" << cfg.ollama.scheme << "://" << cfg.ollama.host << ":" << cfg.ollama.port
            << cfg.ollama.base_path << "\n";
  std::cout << "[provider] mnn enabled=" << (cfg.mnn_enabled ? "true" : "false") << " endpoint=" << cfg.mnn.scheme << "://"
            << cfg.mnn.host << ":" << cfg.mnn.port << cfg.mnn.base_path << "\n";
  std::cout << "[provider] lmdeploy enabled=" << (cfg.lmdeploy_enabled ? "true" : "false") << " endpoint="
            << cfg.lmdeploy.scheme << "://" << cfg.lmdeploy.host << ":" << cfg.lmdeploy.port << cfg.lmdeploy.base_path << "\n";
  std::cout << "[provider] mcp enabled=" << (cfg.mcp_enabled ? "true" : "false") << " endpoint=" << cfg.mcp.scheme << "://"
            << cfg.mcp.host << ":" << cfg.mcp.port << cfg.mcp.base_path << "\n";

  httplib::Server server;
  router.Register(&server);

  std::vector<std::shared_ptr<runtime::McpClient>> mcp_servers;
  std::vector<std::unordered_map<std::string, std::string>> mcp_name_maps;

  auto normalize_under_root = [&](const std::string& path_or_uri, std::string* out_path, std::string* err) -> bool {
    auto percent_decode = [&](const std::string& in) -> std::string {
      std::string out;
      out.reserve(in.size());
      for (size_t i = 0; i < in.size(); i++) {
        const char ch = in[i];
        if (ch == '%' && i + 2 < in.size()) {
          const char a = in[i + 1];
          const char b = in[i + 2];
          auto hex = [](char x) -> int {
            if (x >= '0' && x <= '9') return x - '0';
            if (x >= 'a' && x <= 'f') return 10 + (x - 'a');
            if (x >= 'A' && x <= 'F') return 10 + (x - 'A');
            return -1;
          };
          const int ha = hex(a);
          const int hb = hex(b);
          if (ha >= 0 && hb >= 0) {
            out.push_back(static_cast<char>((ha << 4) | hb));
            i += 2;
            continue;
          }
        }
        out.push_back(ch);
      }
      return out;
    };

    std::string raw = path_or_uri;
    const std::string lower = ToLower(raw);
    constexpr const char* kFileScheme = "file://";
    if (lower.rfind(kFileScheme, 0) == 0) {
      raw = raw.substr(std::strlen(kFileScheme));
      if (raw.rfind("localhost/", 0) == 0) raw = raw.substr(std::strlen("localhost/"));
      if (!raw.empty() && raw[0] == '/' && raw.size() >= 3 && std::isalpha(static_cast<unsigned char>(raw[1])) &&
          raw[2] == ':') {
        raw = raw.substr(1);
      }
      raw = percent_decode(raw);
    }
    std::filesystem::path p = raw;
    if (!cfg.workspace_root.empty() && p.is_relative()) p = std::filesystem::path(cfg.workspace_root) / p;
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(p, ec);
    if (ec) {
      if (err) *err = "invalid path";
      return false;
    }
    if (!cfg.workspace_root.empty()) {
      auto root = std::filesystem::weakly_canonical(std::filesystem::path(cfg.workspace_root), ec);
      if (ec) {
        if (err) *err = "invalid workspace root";
        return false;
      }
      auto canon_s = canon.generic_string();
      auto root_s = root.generic_string();
      if (!root_s.empty() && canon_s.rfind(root_s, 0) != 0) {
        if (err) *err = "path is outside workspace root";
        return false;
      }
    }
    if (out_path) *out_path = canon.generic_string();
    return true;
  };

  auto make_file_uri = [&](const std::string& normalized_path) -> std::string {
    if (normalized_path.empty()) return "file:///";
    if (normalized_path[0] == '/') return std::string("file://") + normalized_path;
    return std::string("file:///") + normalized_path;
  };

  auto refresh_mcp_tools = [&]() {
    nlohmann::json out;
    out["ok"] = true;
    out["servers"] = mcp_servers.size();
    out["registered"] = 0;
    out["errors"] = nlohmann::json::array();

    auto* tools = router.MutableTools();
    for (size_t i = 0; i < mcp_servers.size(); i++) {
      std::string err;
      auto list = mcp_servers[i]->ListTools(&err);
      if (!err.empty()) {
        out["errors"].push_back({{"server", static_cast<int>(i + 1)}, {"error", err}});
        continue;
      }
      for (const auto& t : list) {
        if (t.name.empty()) continue;
        std::string exposed_name;
        auto it = mcp_name_maps[i].find(t.name);
        if (it != mcp_name_maps[i].end()) {
          exposed_name = it->second;
        } else {
          exposed_name = t.name;
          if (tools->HasTool(exposed_name)) exposed_name = "mcp" + std::to_string(i + 1) + "." + exposed_name;
          mcp_name_maps[i][t.name] = exposed_name;
        }

        runtime::ToolSchema schema;
        schema.name = exposed_name;
        schema.description = t.description.empty() ? t.title : t.description;
        schema.parameters = t.input_schema.is_null() ? nlohmann::json::object() : t.input_schema;
        tools->RegisterTool(schema, [mcp = mcp_servers[i], exposed_name, remote_name = t.name](
                                       const std::string& tool_call_id, const nlohmann::json& arguments) {
          std::string call_err;
          runtime::ToolResult r;
          r.tool_call_id = tool_call_id;
          r.name = exposed_name;
          LogMcpCall(tool_call_id, exposed_name, remote_name, arguments);
          auto result = mcp->CallTool(remote_name, arguments, &call_err);
          if (!result) {
            r.ok = false;
            r.error = call_err.empty() ? "mcp: call failed" : call_err;
            r.result = {{"ok", false}, {"error", r.error}};
            LogMcpResult(tool_call_id, exposed_name, remote_name, false, r.error, r.result);
            return r;
          }
          r.result = *result;
          if (result->contains("isError") && (*result)["isError"].is_boolean()) r.ok = !(*result)["isError"].get<bool>();
          LogMcpResult(tool_call_id, exposed_name, remote_name, r.ok, r.error, r.result);
          return r;
        });
        out["registered"] = out["registered"].get<int>() + 1;
      }
    }
    return out;
  };

  if (cfg.mcp_enabled) {
    std::vector<runtime::HttpEndpoint> hosts;
    if (!cfg.mcp_hosts.empty()) {
      hosts = cfg.mcp_hosts;
    } else {
      hosts.push_back(cfg.mcp);
    }

    for (size_t i = 0; i < hosts.size(); i++) {
      auto mcp = std::make_shared<runtime::McpClient>(hosts[i]);
      if (auto v = GetEnvInt("MCP_CONNECT_TIMEOUT_S")) mcp->SetTimeouts(*v, 0, 0);
      if (auto v = GetEnvInt("MCP_READ_TIMEOUT_S")) mcp->SetTimeouts(0, *v, 0);
      if (auto v = GetEnvInt("MCP_WRITE_TIMEOUT_S")) mcp->SetTimeouts(0, 0, *v);
      if (auto v = GetEnvInt("MCP_MAX_IN_FLIGHT")) mcp->SetMaxInFlight(*v);
      std::string err;
      if (!mcp->Initialize(&err)) continue;
      mcp_servers.push_back(std::move(mcp));
      mcp_name_maps.push_back({});
    }
    refresh_mcp_tools();
  }

  auto call_any_mcp = [&](const std::string& tool_name, const nlohmann::json& args, std::string* err) {
    for (const auto& mcp : mcp_servers) {
      std::string call_err;
      auto r = mcp->CallTool(tool_name, args, &call_err);
      if (r) return r;
      if (err) *err = call_err;
    }
    return std::optional<nlohmann::json>();
  };

  if (!mcp_servers.empty()) {
    auto* tools = router.MutableTools();

    {
      runtime::ToolSchema schema;
      schema.name = "ide.read_file";
      schema.description = "Read a text file under workspace root.";
      schema.parameters = {{"type", "object"},
                           {"properties", {{"path", {{"type", "string"}}}}},
                           {"required", {"path"}}};
      tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
        runtime::ToolResult tr;
        tr.tool_call_id = tool_call_id;
        tr.name = schema.name;
        if (!arguments.is_object() || !arguments.contains("path") || !arguments["path"].is_string()) {
          tr.ok = false;
          tr.error = "missing required field: path";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        std::string norm, err;
        if (!normalize_under_root(arguments["path"].get<std::string>(), &norm, &err)) {
          tr.ok = false;
          tr.error = err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        nlohmann::json args;
        args["path"] = norm;
        LogMcpCall(tool_call_id, schema.name, "fs.read_file", args);
        auto r = call_any_mcp("fs.read_file", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          LogMcpResult(tool_call_id, schema.name, "fs.read_file", false, tr.error, tr.result);
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
        LogMcpResult(tool_call_id, schema.name, "fs.read_file", tr.ok, tr.error, tr.result);
        return tr;
      });
    }

    {
      runtime::ToolSchema schema;
      schema.name = "ide.search";
      schema.description = "Search text in workspace files.";
      schema.parameters = {{"type", "object"},
                           {"properties",
                            {{"query", {{"type", "string"}}},
                             {"path", {{"type", "string"}}},
                             {"max_results", {{"type", "integer"}}}}},
                           {"required", {"query"}}};
      tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
        runtime::ToolResult tr;
        tr.tool_call_id = tool_call_id;
        tr.name = schema.name;
        if (!arguments.is_object() || !arguments.contains("query") || !arguments["query"].is_string()) {
          tr.ok = false;
          tr.error = "missing required field: query";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        nlohmann::json args;
        args["query"] = arguments["query"];
        if (arguments.contains("max_results") && arguments["max_results"].is_number_integer()) {
          args["max_results"] = arguments["max_results"];
        }
        if (arguments.contains("path") && arguments["path"].is_string()) {
          std::string norm, err;
          if (!normalize_under_root(arguments["path"].get<std::string>(), &norm, &err)) {
            tr.ok = false;
            tr.error = err;
            tr.result = {{"ok", false}, {"error", tr.error}};
            return tr;
          }
          args["path"] = norm;
        } else if (!cfg.workspace_root.empty()) {
          args["path"] = cfg.workspace_root;
        }
        std::string err;
        LogMcpCall(tool_call_id, schema.name, "fs.search", args);
        auto r = call_any_mcp("fs.search", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          LogMcpResult(tool_call_id, schema.name, "fs.search", false, tr.error, tr.result);
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
        LogMcpResult(tool_call_id, schema.name, "fs.search", tr.ok, tr.error, tr.result);
        return tr;
      });
    }

    {
      runtime::ToolSchema schema;
      schema.name = "ide.diagnostics";
      schema.description = "Get diagnostics for a file.";
      schema.parameters = {{"type", "object"},
                           {"properties", {{"uri", {{"type", "string"}}}}},
                           {"required", {"uri"}}};
      tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
        runtime::ToolResult tr;
        tr.tool_call_id = tool_call_id;
        tr.name = schema.name;
        if (!arguments.is_object() || !arguments.contains("uri") || !arguments["uri"].is_string()) {
          tr.ok = false;
          tr.error = "missing required field: uri";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        std::string norm, err;
        if (!normalize_under_root(arguments["uri"].get<std::string>(), &norm, &err)) {
          tr.ok = false;
          tr.error = err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        nlohmann::json args;
        args["uri"] = make_file_uri(norm);
        LogMcpCall(tool_call_id, schema.name, "lsp.diagnostics", args);
        auto r = call_any_mcp("lsp.diagnostics", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          LogMcpResult(tool_call_id, schema.name, "lsp.diagnostics", false, tr.error, tr.result);
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
        LogMcpResult(tool_call_id, schema.name, "lsp.diagnostics", tr.ok, tr.error, tr.result);
        return tr;
      });
    }

    {
      runtime::ToolSchema schema;
      schema.name = "ide.hover";
      schema.description = "Get hover information at a position.";
      schema.parameters = {{"type", "object"},
                           {"properties",
                            {{"uri", {{"type", "string"}}},
                             {"line", {{"type", "integer"}}},
                             {"character", {{"type", "integer"}}}}},
                           {"required", {"uri", "line", "character"}}};
      tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
        runtime::ToolResult tr;
        tr.tool_call_id = tool_call_id;
        tr.name = schema.name;
        if (!arguments.is_object() || !arguments.contains("uri") || !arguments["uri"].is_string()) {
          tr.ok = false;
          tr.error = "missing required field: uri";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        if (!arguments.contains("line") || !arguments["line"].is_number_integer() || !arguments.contains("character") ||
            !arguments["character"].is_number_integer()) {
          tr.ok = false;
          tr.error = "missing required fields: line, character";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        std::string norm, err;
        if (!normalize_under_root(arguments["uri"].get<std::string>(), &norm, &err)) {
          tr.ok = false;
          tr.error = err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        nlohmann::json args;
        args["uri"] = make_file_uri(norm);
        args["line"] = arguments["line"];
        args["character"] = arguments["character"];
        LogMcpCall(tool_call_id, schema.name, "lsp.hover", args);
        auto r = call_any_mcp("lsp.hover", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          LogMcpResult(tool_call_id, schema.name, "lsp.hover", false, tr.error, tr.result);
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
        LogMcpResult(tool_call_id, schema.name, "lsp.hover", tr.ok, tr.error, tr.result);
        return tr;
      });
    }

    {
      runtime::ToolSchema schema;
      schema.name = "ide.definition";
      schema.description = "Get definition location at a position.";
      schema.parameters = {{"type", "object"},
                           {"properties",
                            {{"uri", {{"type", "string"}}},
                             {"line", {{"type", "integer"}}},
                             {"character", {{"type", "integer"}}}}},
                           {"required", {"uri", "line", "character"}}};
      tools->RegisterTool(schema, [&, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
        runtime::ToolResult tr;
        tr.tool_call_id = tool_call_id;
        tr.name = schema.name;
        if (!arguments.is_object() || !arguments.contains("uri") || !arguments["uri"].is_string()) {
          tr.ok = false;
          tr.error = "missing required field: uri";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        if (!arguments.contains("line") || !arguments["line"].is_number_integer() || !arguments.contains("character") ||
            !arguments["character"].is_number_integer()) {
          tr.ok = false;
          tr.error = "missing required fields: line, character";
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        std::string norm, err;
        if (!normalize_under_root(arguments["uri"].get<std::string>(), &norm, &err)) {
          tr.ok = false;
          tr.error = err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        nlohmann::json args;
        args["uri"] = make_file_uri(norm);
        args["line"] = arguments["line"];
        args["character"] = arguments["character"];
        LogMcpCall(tool_call_id, schema.name, "lsp.definition", args);
        auto r = call_any_mcp("lsp.definition", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          LogMcpResult(tool_call_id, schema.name, "lsp.definition", false, tr.error, tr.result);
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
        LogMcpResult(tool_call_id, schema.name, "lsp.definition", tr.ok, tr.error, tr.result);
        return tr;
      });
    }
  }

  server.Post("/internal/refresh_mcp_tools", [&](const httplib::Request&, httplib::Response& res) {
    auto out = refresh_mcp_tools();
    res.status = 200;
    res.set_content(out.dump(), "application/json");
  });

  server.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string message = "unknown exception";
    if (ep) {
      try {
        std::rethrow_exception(ep);
      } catch (const std::exception& e) {
        message = e.what();
      } catch (...) {
      }
    }
    nlohmann::json j;
    j["error"] = {{"message", message}, {"type", "server_error"}, {"param", nullptr}, {"code", nullptr}};
    res.status = 500;
    res.set_content(j.dump(), "application/json");
  });

  server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (!res.body.empty()) return;
    std::string message;
    std::string type = "invalid_request_error";
    if (res.status == 404) {
      message = "not found";
    } else if (res.status >= 500) {
      message = "upstream error";
      type = "api_error";
    } else {
      message = "bad request";
    }
    nlohmann::json j;
    j["error"] = {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}};
    res.set_content(j.dump(), "application/json");
  });

  server.set_keep_alive_timeout(5);
  server.set_read_timeout(60);
  server.set_write_timeout(60);

  server.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    nlohmann::json j;
    j["ok"] = true;
    j["unix_seconds"] =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    res.status = 200;
    res.set_content(j.dump(), "application/json");
  });

  std::cout << "[http] listen host=" << cfg.listen.host << " port=" << cfg.listen.port << "\n";
  const bool ok = server.listen(cfg.listen.host, cfg.listen.port);
  std::cout << "[http] listen returned ok=" << (ok ? 1 : 0) << "\n";
  return ok ? 0 : 1;
}

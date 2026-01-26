#include "config.hpp"
#include "llama_cpp_provider.hpp"
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
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

int main() {
  auto cfg = runtime::LoadConfigFromEnv();

  runtime::SessionManager sessions(cfg.session_store_path);
  runtime::ProviderRegistry providers(cfg.default_provider);
  providers.Register(std::make_unique<runtime::LlamaCppProvider>(cfg.llama_cpp_model_path));
  providers.Register(std::make_unique<runtime::OllamaProvider>(cfg.ollama));
  if (cfg.mnn_enabled) providers.Register(std::make_unique<runtime::OpenAiCompatibleHttpProvider>("mnn", cfg.mnn));
  if (cfg.lmdeploy_enabled) providers.Register(std::make_unique<runtime::OpenAiCompatibleHttpProvider>("lmdeploy", cfg.lmdeploy));
  runtime::OpenAiRouter router(&sessions, &providers, runtime::BuildDefaultToolRegistry());

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
    std::string raw = path_or_uri;
    constexpr const char* kFileScheme = "file://";
    if (raw.rfind(kFileScheme, 0) == 0) raw = raw.substr(std::strlen(kFileScheme));
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
          auto result = mcp->CallTool(remote_name, arguments, &call_err);
          if (!result) {
            r.ok = false;
            r.error = call_err.empty() ? "mcp: call failed" : call_err;
            r.result = {{"ok", false}, {"error", r.error}};
            return r;
          }
          r.result = *result;
          if (result->contains("isError") && (*result)["isError"].is_boolean()) r.ok = !(*result)["isError"].get<bool>();
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
        auto r = call_any_mcp("fs.read_file", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
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
        auto r = call_any_mcp("fs.search", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
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
        args["uri"] = std::string("file://") + norm;
        auto r = call_any_mcp("lsp.diagnostics", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
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
        args["uri"] = std::string("file://") + norm;
        args["line"] = arguments["line"];
        args["character"] = arguments["character"];
        auto r = call_any_mcp("lsp.hover", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
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
        args["uri"] = std::string("file://") + norm;
        args["line"] = arguments["line"];
        args["character"] = arguments["character"];
        auto r = call_any_mcp("lsp.definition", args, &err);
        if (!r) {
          tr.ok = false;
          tr.error = err.empty() ? "mcp: call failed" : err;
          tr.result = {{"ok", false}, {"error", tr.error}};
          return tr;
        }
        tr.result = *r;
        if (r->contains("isError") && (*r)["isError"].is_boolean()) tr.ok = !(*r)["isError"].get<bool>();
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

  server.listen(cfg.listen.host, cfg.listen.port);
  return 0;
}

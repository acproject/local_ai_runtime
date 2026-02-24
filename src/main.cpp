#include "config.hpp"
#if LOCAL_AI_RUNTIME_WITH_LLAMA_CPP
#include "llama_cpp_provider.hpp"
#endif
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
  runtime::OpenAiRouter router(&sessions, &providers);

  std::cout << "[runtime] default_provider=" << cfg.default_provider << "\n";
  std::cout << "[provider] llama_cpp model_path=" << (cfg.llama_cpp_model_path.empty() ? "<empty>" : cfg.llama_cpp_model_path)
            << "\n";
  std::cout << "[provider] ollama endpoint=" << cfg.ollama.scheme << "://" << cfg.ollama.host << ":" << cfg.ollama.port
            << cfg.ollama.base_path << "\n";
  std::cout << "[provider] mnn enabled=" << (cfg.mnn_enabled ? "true" : "false") << " endpoint=" << cfg.mnn.scheme << "://"
            << cfg.mnn.host << ":" << cfg.mnn.port << cfg.mnn.base_path << "\n";
  std::cout << "[provider] lmdeploy enabled=" << (cfg.lmdeploy_enabled ? "true" : "false") << " endpoint="
            << cfg.lmdeploy.scheme << "://" << cfg.lmdeploy.host << ":" << cfg.lmdeploy.port << cfg.lmdeploy.base_path << "\n";

  httplib::Server server;
  router.Register(&server);

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

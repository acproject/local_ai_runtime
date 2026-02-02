#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace runtime {

struct HttpListenConfig {
  std::string host = "0.0.0.0";
  int port = 8080;
};

struct HttpEndpoint {
  std::string scheme = "http";
  std::string host = "127.0.0.1";
  int port = 11434;
  std::string base_path;
};

struct RuntimeConfig {
  HttpListenConfig listen;
  std::string default_provider = "llama_cpp";
  std::string llama_cpp_model_path;
  std::string session_store_type = "memory";
  std::string session_store_path;
  HttpEndpoint session_store_endpoint;
  std::string session_store_password;
  int session_store_db = 0;
  std::string session_store_namespace;
  bool session_store_reset_on_boot = false;
  HttpEndpoint ollama;
  HttpEndpoint mnn;
  HttpEndpoint lmdeploy;
  bool mnn_enabled = false;
  bool lmdeploy_enabled = false;
  HttpEndpoint mcp;
  bool mcp_enabled = false;
  std::vector<HttpEndpoint> mcp_hosts;
  std::string workspace_root;
};

RuntimeConfig LoadConfigFromEnv();

using RequestHeaderList = std::vector<std::pair<std::string, std::string>>;

const RequestHeaderList& CurrentRequestAuthHeaders();
void SetCurrentRequestAuthHeaders(RequestHeaderList headers);

class ScopedRequestAuthHeaders {
 public:
  explicit ScopedRequestAuthHeaders(RequestHeaderList headers);
  ~ScopedRequestAuthHeaders();

 private:
  RequestHeaderList prev_;
};

}  // namespace runtime

#pragma once

#include <cstdint>
#include <string>
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
  HttpEndpoint ollama;
  HttpEndpoint mcp;
  bool mcp_enabled = false;
  std::vector<HttpEndpoint> mcp_hosts;
  std::string workspace_root;
};

RuntimeConfig LoadConfigFromEnv();

}  // namespace runtime

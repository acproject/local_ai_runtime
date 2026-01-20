#include "config.hpp"

#include <cstdlib>
#include <string>
#include <vector>

namespace runtime {
namespace {

static bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static HttpEndpoint ParseHttpEndpoint(const std::string& url, int default_port) {
  HttpEndpoint ep;
  std::string s = url;
  if (StartsWith(s, "http://")) {
    ep.scheme = "http";
    s = s.substr(7);
  } else if (StartsWith(s, "https://")) {
    ep.scheme = "https";
    s = s.substr(8);
  }

  auto slash_pos = s.find('/');
  if (slash_pos != std::string::npos) {
    ep.base_path = s.substr(slash_pos);
    s = s.substr(0, slash_pos);
  }

  auto colon_pos = s.rfind(':');
  if (colon_pos != std::string::npos) {
    ep.host = s.substr(0, colon_pos);
    ep.port = std::atoi(s.substr(colon_pos + 1).c_str());
  } else if (!s.empty()) {
    ep.host = s;
  }
  if (ep.base_path.empty()) ep.base_path = "";
  if (ep.port == 0) ep.port = default_port;
  if (ep.host.empty()) ep.host = "127.0.0.1";
  return ep;
}

static std::string GetEnvStr(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  for (auto& v : out) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
  }
  std::vector<std::string> filtered;
  for (auto& v : out) {
    if (!v.empty()) filtered.push_back(std::move(v));
  }
  return filtered;
}

}  // namespace

RuntimeConfig LoadConfigFromEnv() {
  RuntimeConfig cfg;

  if (auto host = GetEnvStr("RUNTIME_LISTEN_HOST"); !host.empty()) cfg.listen.host = host;
  if (auto port = GetEnvStr("RUNTIME_LISTEN_PORT"); !port.empty()) cfg.listen.port = std::atoi(port.c_str());

  if (auto p = GetEnvStr("RUNTIME_PROVIDER"); !p.empty()) cfg.default_provider = p;
  if (auto model = GetEnvStr("LLAMA_CPP_MODEL"); !model.empty()) cfg.llama_cpp_model_path = model;

  auto ollama = GetEnvStr("OLLAMA_HOST");
  if (!ollama.empty()) cfg.ollama = ParseHttpEndpoint(ollama, 11434);

  auto mnn = GetEnvStr("MNN_HOST");
  if (!mnn.empty()) {
    cfg.mnn = ParseHttpEndpoint(mnn, 8000);
    cfg.mnn_enabled = true;
  }

  auto lmdeploy = GetEnvStr("LMDEPLOY_HOST");
  if (!lmdeploy.empty()) {
    cfg.lmdeploy = ParseHttpEndpoint(lmdeploy, 23333);
    cfg.lmdeploy_enabled = true;
  }

  auto mcp = GetEnvStr("MCP_HOST");
  if (!mcp.empty()) {
    cfg.mcp = ParseHttpEndpoint(mcp, 9000);
    cfg.mcp_enabled = true;
  }

  auto mcp_hosts = GetEnvStr("MCP_HOSTS");
  if (!mcp_hosts.empty()) {
    cfg.mcp_hosts.clear();
    for (const auto& url : SplitCsv(mcp_hosts)) {
      cfg.mcp_hosts.push_back(ParseHttpEndpoint(url, 9000));
    }
    if (!cfg.mcp_hosts.empty()) cfg.mcp_enabled = true;
  }

  if (auto root = GetEnvStr("RUNTIME_WORKSPACE_ROOT"); !root.empty()) cfg.workspace_root = root;

  return cfg;
}

}  // namespace runtime

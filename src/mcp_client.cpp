#include "mcp_client.hpp"

#include <httplib.h>

#include <memory>
#include <string>
#include <utility>

namespace runtime {
namespace {

static std::unique_ptr<httplib::Client> MakeClient(const HttpEndpoint& ep,
                                                   int connect_timeout_seconds,
                                                   int read_timeout_seconds,
                                                   int write_timeout_seconds) {
  auto cli = std::make_unique<httplib::Client>(ep.host, ep.port);
  cli->set_connection_timeout(connect_timeout_seconds);
  cli->set_read_timeout(read_timeout_seconds);
  cli->set_write_timeout(write_timeout_seconds);
  const auto& auth = CurrentRequestAuthHeaders();
  if (!auth.empty()) {
    httplib::Headers headers;
    for (const auto& kv : auth) {
      headers.emplace(kv.first, kv.second);
    }
    cli->set_default_headers(std::move(headers));
  }
  return cli;
}

static std::string JoinPath(const std::string& base, const std::string& path) {
  if (base.empty()) return path;
  if (base.back() == '/' && !path.empty() && path.front() == '/') return base + path.substr(1);
  if (base.back() != '/' && !path.empty() && path.front() != '/') return base + "/" + path;
  return base + path;
}

static std::string ExtractJsonRpcError(const nlohmann::json& resp) {
  if (!resp.is_object()) return "invalid json-rpc response";
  if (!resp.contains("error") || !resp["error"].is_object()) return {};
  const auto& e = resp["error"];
  std::string msg;
  if (e.contains("message") && e["message"].is_string()) msg = e["message"].get<std::string>();
  if (msg.empty()) msg = "json-rpc error";
  return msg;
}

static std::optional<nlohmann::json> ExtractJsonRpcResult(const nlohmann::json& resp) {
  if (!resp.is_object()) return std::nullopt;
  if (resp.contains("result")) return resp["result"];
  return std::nullopt;
}

}  // namespace

McpClient::McpClient(HttpEndpoint endpoint) : endpoint_(std::move(endpoint)) {}

void McpClient::SetTimeouts(int connect_seconds, int read_seconds, int write_seconds) {
  if (connect_seconds > 0) connect_timeout_seconds_ = connect_seconds;
  if (read_seconds > 0) read_timeout_seconds_ = read_seconds;
  if (write_seconds > 0) write_timeout_seconds_ = write_seconds;
}

void McpClient::SetMaxInFlight(int max_in_flight) {
  if (max_in_flight > 0) max_in_flight_ = max_in_flight;
}

bool McpClient::Initialize(std::string* err) {
  nlohmann::json params;
  params["protocolVersion"] = "2024-11-05";
  params["capabilities"] = nlohmann::json::object();
  params["clientInfo"] = {{"name", "local-ai-runtime"}, {"version", "0.1.0"}};
  auto r = Rpc("initialize", params, err);
  return r.has_value();
}

std::vector<McpToolInfo> McpClient::ListTools(std::string* err) {
  std::vector<McpToolInfo> out;
  std::string cursor;
  for (int page = 0; page < 64; page++) {
    nlohmann::json params = nlohmann::json::object();
    if (!cursor.empty()) params["cursor"] = cursor;
    auto r = Rpc("tools/list", params, err);
    if (!r) return {};
    if (!r->contains("tools") || !(*r)["tools"].is_array()) return out;
    for (const auto& t : (*r)["tools"]) {
      if (!t.is_object()) continue;
      McpToolInfo info;
      if (t.contains("name") && t["name"].is_string()) info.name = t["name"].get<std::string>();
      if (t.contains("title") && t["title"].is_string()) info.title = t["title"].get<std::string>();
      if (t.contains("description") && t["description"].is_string()) info.description = t["description"].get<std::string>();
      if (t.contains("inputSchema") && t["inputSchema"].is_object()) info.input_schema = t["inputSchema"];
      if (!info.name.empty()) out.push_back(std::move(info));
    }
    if (r->contains("nextCursor") && (*r)["nextCursor"].is_string()) {
      cursor = (*r)["nextCursor"].get<std::string>();
      if (cursor.empty()) break;
    } else {
      break;
    }
  }
  return out;
}

std::optional<nlohmann::json> McpClient::CallTool(const std::string& name,
                                                  const nlohmann::json& arguments,
                                                  std::string* err) {
  nlohmann::json params;
  params["name"] = name;
  params["arguments"] = arguments;
  return Rpc("tools/call", params, err);
}

std::optional<nlohmann::json> McpClient::Rpc(const std::string& method,
                                             const nlohmann::json& params,
                                             std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (in_flight_ >= max_in_flight_) {
      if (err) *err = "mcp: too many in-flight requests";
      return std::nullopt;
    }
    in_flight_++;
  }

  auto dec = [&]() {
    std::lock_guard<std::mutex> lock(mu_);
    in_flight_--;
  };

  auto cli = MakeClient(endpoint_, connect_timeout_seconds_, read_timeout_seconds_, write_timeout_seconds_);
  nlohmann::json req;
  req["jsonrpc"] = "2.0";
  req["id"] = next_id_++;
  req["method"] = method;
  req["params"] = params;

  auto path = endpoint_.base_path.empty() ? "/" : endpoint_.base_path;
  auto res = cli->Post(JoinPath("", path), req.dump(), "application/json");
  if (!res) {
    if (err) *err = "mcp: failed to connect";
    dec();
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "mcp: http " + std::to_string(res->status);
    dec();
    return std::nullopt;
  }

  auto resp = nlohmann::json::parse(res->body, nullptr, false);
  if (resp.is_discarded()) {
    if (err) *err = "mcp: invalid json response";
    dec();
    return std::nullopt;
  }

  auto rpc_err = ExtractJsonRpcError(resp);
  if (!rpc_err.empty()) {
    if (err) *err = rpc_err;
    dec();
    return std::nullopt;
  }
  auto result = ExtractJsonRpcResult(resp);
  if (!result) {
    if (err) *err = "mcp: missing result";
    dec();
    return std::nullopt;
  }
  dec();
  return result;
}

}  // namespace runtime

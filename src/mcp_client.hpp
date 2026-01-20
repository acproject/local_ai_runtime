#pragma once

#include "config.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace runtime {

struct McpToolInfo {
  std::string name;
  std::string title;
  std::string description;
  nlohmann::json input_schema;
};

class McpClient {
 public:
  explicit McpClient(HttpEndpoint endpoint);

  bool Initialize(std::string* err);
  std::vector<McpToolInfo> ListTools(std::string* err);
  std::optional<nlohmann::json> CallTool(const std::string& name, const nlohmann::json& arguments, std::string* err);

  void SetTimeouts(int connect_seconds, int read_seconds, int write_seconds);
  void SetMaxInFlight(int max_in_flight);

 private:
  HttpEndpoint endpoint_;
  int64_t next_id_ = 1;
  int connect_timeout_seconds_ = 5;
  int read_timeout_seconds_ = 60;
  int write_timeout_seconds_ = 30;
  int max_in_flight_ = 4;

  std::mutex mu_;
  int in_flight_ = 0;

  std::optional<nlohmann::json> Rpc(const std::string& method, const nlohmann::json& params, std::string* err);
};

}  // namespace runtime

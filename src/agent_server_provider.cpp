#include "agent_server_provider.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace runtime {
namespace {

static bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string TrimWs(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == '\n' || s[i] == '\r' || s[i] == ' ' || s[i] == '\t')) i++;
  if (i > 0) s.erase(0, i);
  return s;
}

static std::string JoinPath(const std::string& base, const std::string& path) {
  if (base.empty()) return path;
  std::string b = base;
  std::string p = path;
  if (EndsWith(b, "/")) b.pop_back();
  if (StartsWith(p, "/")) p = p.substr(1);
  if (p.empty()) return b;
  return b + "/" + p;
}

static std::unique_ptr<httplib::Client> MakeClient(const HttpEndpoint& ep) {
  std::unique_ptr<httplib::Client> cli;
  if (ep.scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    cli = std::make_unique<httplib::SSLClient>(ep.host, ep.port);
#else
    std::cerr << "[agent_server] HTTPS requested but SSL not supported\n";
    cli = std::make_unique<httplib::Client>(ep.host, ep.port);
#endif
  } else {
    cli = std::make_unique<httplib::Client>(ep.host, ep.port);
  }
  cli->set_connection_timeout(5, 0);
  cli->set_read_timeout(300, 0);
  cli->set_write_timeout(30, 0);
  cli->set_keep_alive(true);
  
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

static std::string ToLower(std::string s) {
  for (auto& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// Parse SSE events from OpenAI streaming format
struct SseEvent {
  std::string event_type;
  std::string data;
};

static std::vector<SseEvent> ParseSseEvents(const std::string& chunk) {
  std::vector<SseEvent> events;
  std::istringstream ss(chunk);
  std::string line;
  SseEvent current_event;
  
  while (std::getline(ss, line)) {
    line = TrimWs(line);
    
    if (line.empty()) {
      // Empty line signals end of event
      if (!current_event.event_type.empty() || !current_event.data.empty()) {
        events.push_back(std::move(current_event));
        current_event = SseEvent();
      }
      continue;
    }
    
    if (StartsWith(line, "event:")) {
      current_event.event_type = TrimWs(line.substr(6));
    } else if (StartsWith(line, "data:")) {
      if (!current_event.data.empty()) {
        current_event.data += "\n";
      }
      current_event.data += TrimWs(line.substr(5));
    }
  }
  
  // Handle last event if no trailing empty line
  if (!current_event.event_type.empty() || !current_event.data.empty()) {
    events.push_back(std::move(current_event));
  }
  
  return events;
}

}  // namespace

AgentServerProvider::AgentServerProvider(HttpEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {
  std::cout << "[agent_server] initialized with endpoint " 
            << endpoint_.scheme << "://" << endpoint_.host 
            << ":" << endpoint_.port << endpoint_.base_path << "\n";
}

AgentServerProvider::~AgentServerProvider() {
  Stop();
}

void AgentServerProvider::Start() {
  std::string err;
  server_available_ = CheckHealth(&err);
  if (server_available_) {
    std::cout << "[agent_server] server is available\n";
  } else {
    std::cout << "[agent_server] server not available: " << err << "\n";
  }
}

void AgentServerProvider::Stop() {
  server_available_ = false;
}

std::string AgentServerProvider::Name() const {
  return "agent_server";
}

bool AgentServerProvider::CheckHealth(std::string* err) {
  auto cli = MakeClient(endpoint_);
  auto res = cli->Get(JoinPath(endpoint_.base_path, "/health"));
  if (!res) {
    if (err) *err = "failed to connect to server";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "health check returned status " + std::to_string(res->status);
    return false;
  }
  return true;
}

std::vector<ModelInfo> AgentServerProvider::ListModels(std::string* err) {
  auto cli = MakeClient(endpoint_);
  auto res = cli->Get(JoinPath(endpoint_.base_path, "/v1/models"));
  
  if (!res) {
    if (err) *err = "failed to connect to server for models";
    return {};
  }
  
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "/v1/models returned status " + std::to_string(res->status);
    return {};
  }
  
  auto j = nlohmann::json::parse(res->body, nullptr, false);
  if (j.is_discarded() || !j.contains("data") || !j["data"].is_array()) {
    if (err) *err = "invalid json from /v1/models";
    return {};
  }
  
  std::vector<ModelInfo> out;
  for (const auto& item : j["data"]) {
    if (!item.is_object()) continue;
    ModelInfo m;
    if (item.contains("id") && item["id"].is_string()) {
      m.id = item["id"].get<std::string>();
    }
    if (item.contains("owned_by") && item["owned_by"].is_string()) {
      m.owned_by = item["owned_by"].get<std::string>();
    }
    if (m.owned_by.empty()) m.owned_by = "llama-server";
    if (!m.id.empty()) {
      out.push_back(std::move(m));
    }
  }
  return out;
}

std::optional<std::vector<double>> AgentServerProvider::Embeddings(
    const std::string& model, const std::string& input, std::string* err) {
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = model;
  j["input"] = input;
  
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/v1/embeddings"), 
                       j.dump(), "application/json");
  
  if (!res) {
    if (err) *err = "failed to connect to server for embeddings";
    return std::nullopt;
  }
  
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "/v1/embeddings returned status " + std::to_string(res->status);
    return std::nullopt;
  }
  
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("data") || !jr["data"].is_array() || 
      jr["data"].empty() || !jr["data"][0].is_object() ||
      !jr["data"][0].contains("embedding") || !jr["data"][0]["embedding"].is_array()) {
    if (err) *err = "invalid json from /v1/embeddings";
    return std::nullopt;
  }
  
  std::vector<double> vec;
  for (const auto& v : jr["data"][0]["embedding"]) {
    if (v.is_number_float() || v.is_number_integer()) {
      vec.push_back(v.get<double>());
    }
  }
  return vec;
}

std::optional<ChatResponse> AgentServerProvider::ChatOnce(const ChatRequest& req, std::string* err) {
  auto cli = MakeClient(endpoint_);
  
  nlohmann::json j;
  j["model"] = req.model;
  j["stream"] = false;
  
  if (req.max_tokens.has_value() && req.max_tokens.value() > 0) {
    j["max_tokens"] = req.max_tokens.value();
  }
  if (req.temperature.has_value()) {
    j["temperature"] = req.temperature.value();
  }
  if (req.top_p.has_value()) {
    j["top_p"] = req.top_p.value();
  }
  
  j["messages"] = nlohmann::json::array();
  for (const auto& m : req.messages) {
    j["messages"].push_back({{"role", m.role}, {"content", m.content}});
  }
  
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/v1/chat/completions"), 
                       j.dump(), "application/json");
  
  if (!res) {
    if (err) *err = "failed to connect to server for chat";
    return std::nullopt;
  }
  
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "/v1/chat/completions returned status " + std::to_string(res->status);
    return std::nullopt;
  }
  
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("choices") || !jr["choices"].is_array() || 
      jr["choices"].empty() || !jr["choices"][0].is_object() ||
      !jr["choices"][0].contains("message") || !jr["choices"][0]["message"].is_object() ||
      !jr["choices"][0]["message"].contains("content")) {
    if (err) *err = "invalid json from /v1/chat/completions";
    return std::nullopt;
  }
  
  ChatResponse out;
  out.model = req.model;
  if (jr["choices"][0]["message"]["content"].is_string()) {
    out.content = jr["choices"][0]["message"]["content"].get<std::string>();
  }
  if (jr["choices"][0].contains("finish_reason") && 
      jr["choices"][0]["finish_reason"].is_string()) {
    out.finish_reason = jr["choices"][0]["finish_reason"].get<std::string>();
  }
  out.done = true;
  return out;
}

bool AgentServerProvider::ChatStream(const ChatRequest& req,
                                      const std::function<bool(const std::string&)>& on_delta,
                                      const std::function<void(const std::string& finish_reason)>& on_done,
                                      std::string* err) {
  auto cli = MakeClient(endpoint_);
  
  // Build request body with streaming enabled
  nlohmann::json j;
  j["model"] = req.model;
  j["stream"] = true;  // Enable streaming
  
  if (req.max_tokens.has_value() && req.max_tokens.value() > 0) {
    j["max_tokens"] = req.max_tokens.value();
  }
  if (req.temperature.has_value()) {
    j["temperature"] = req.temperature.value();
  }
  if (req.top_p.has_value()) {
    j["top_p"] = req.top_p.value();
  }
  
  j["messages"] = nlohmann::json::array();
  for (const auto& m : req.messages) {
    j["messages"].push_back({{"role", m.role}, {"content", m.content}});
  }
  
  std::string path = JoinPath(endpoint_.base_path, "/v1/chat/completions");
  
  // Build request for streaming
  httplib::Request http_req;
  http_req.method = "POST";
  http_req.path = path;
  http_req.set_header("Content-Type", "application/json");
  http_req.body = j.dump();
  
  std::string buffer;
  std::string finish_reason = "stop";
  bool has_error = false;
  std::string error_msg;
  bool client_cancelled = false;
  
  // Set up content receiver for streaming response
  http_req.content_receiver = [&](const char* data, size_t len, uint64_t /*offset*/, uint64_t /*total*/) -> bool {
    if (has_error || client_cancelled) return false;
    
    buffer.append(data, len);
    
    // Try to parse complete SSE events
    auto events = ParseSseEvents(buffer);
    
    for (const auto& event : events) {
      // OpenAI streaming format: data: {...}
      if (event.data == "[DONE]") {
        // Stream finished
        continue;
      }
      
      auto ej = nlohmann::json::parse(event.data, nullptr, false);
      if (ej.is_discarded()) continue;
      
      // Parse delta content
      if (ej.contains("choices") && ej["choices"].is_array() && !ej["choices"].empty()) {
        const auto& choice = ej["choices"][0];
        if (choice.is_object()) {
          // Get delta content
          if (choice.contains("delta") && choice["delta"].is_object()) {
            const auto& delta = choice["delta"];
            if (delta.contains("content") && delta["content"].is_string()) {
              std::string content = delta["content"].get<std::string>();
              if (!content.empty()) {
                if (!on_delta(content)) {
                  client_cancelled = true;
                  return false;
                }
              }
            }
          }
          // Get finish reason
          if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
            if (choice["finish_reason"].is_string()) {
              finish_reason = choice["finish_reason"].get<std::string>();
            }
          }
        }
      }
    }
    
    // Clear processed events from buffer
    size_t last_event_end = buffer.rfind("\n\n");
    if (last_event_end != std::string::npos) {
      buffer = buffer.substr(last_event_end + 2);
    }
    
    return true;
  };
  
  auto res = cli->send(http_req);
  
  if (!res) {
    if (err) *err = "failed to connect to server for streaming chat";
    return false;
  }
  
  if (has_error) {
    if (err) *err = error_msg;
    return false;
  }
  
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "streaming chat returned status " + std::to_string(res->status);
    return false;
  }
  
  on_done(finish_reason);
  return true;
}

}  // namespace runtime

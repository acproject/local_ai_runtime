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

static std::string GetEnvStr(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static std::string ToLower(std::string s) {
  for (auto& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
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
  std::lock_guard<std::mutex> lock(mu_);
  session_map_.clear();
  default_agent_session_.clear();
  server_available_ = false;
}

std::string AgentServerProvider::Name() const {
  return "agent_server";
}

bool AgentServerProvider::CheckHealth(std::string* err) {
  auto cli = MakeClient(endpoint_);
  auto res = cli->Get(JoinPath(endpoint_.base_path, "/health"));
  if (!res) {
    if (err) *err = "failed to connect to agent server";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "health check returned status " + std::to_string(res->status);
    return false;
  }
  auto j = nlohmann::json::parse(res->body, nullptr, false);
  if (j.is_discarded()) {
    if (err) *err = "invalid JSON from health check";
    return false;
  }
  // Health endpoint returns {"status": "ok"}
  if (j.contains("status") && j["status"].is_string()) {
    return j["status"].get<std::string>() == "ok";
  }
  return true;
}

std::vector<ModelInfo> AgentServerProvider::ListModels(std::string* err) {
  // Agent server doesn't have a /v1/models endpoint
  // We return a single model entry to indicate the service is available
  if (!server_available_) {
    if (err) *err = "agent server not available";
    return {};
  }
  
  std::vector<ModelInfo> out;
  ModelInfo m;
  m.id = "agent";
  m.owned_by = "agent_server";
  out.push_back(std::move(m));
  return out;
}

std::optional<std::vector<double>> AgentServerProvider::Embeddings(
    const std::string&, const std::string&, std::string* err) {
  if (err) *err = "agent_server: embeddings not supported";
  return std::nullopt;
}

std::optional<std::string> AgentServerProvider::CreateSession(std::string* err) {
  auto cli = MakeClient(endpoint_);
  
  // Create session with optional configuration
  nlohmann::json body = nlohmann::json::object();
  
  // Read agent session config from environment
  std::string yolo_mode = GetEnvStr("AGENT_YOLO_MODE");
  if (!yolo_mode.empty()) {
    body["yolo"] = (ToLower(yolo_mode) == "true" || yolo_mode == "1");
  }
  
  std::string working_dir = GetEnvStr("AGENT_WORKING_DIR");
  if (!working_dir.empty()) {
    body["working_dir"] = working_dir;
  }
  
  std::string enable_skills = GetEnvStr("AGENT_ENABLE_SKILLS");
  if (!enable_skills.empty()) {
    body["enable_skills"] = (ToLower(enable_skills) == "true" || enable_skills == "1");
  }
  
  // Subagent configuration
  std::string max_subagent_depth = GetEnvStr("AGENT_MAX_SUBAGENT_DEPTH");
  if (!max_subagent_depth.empty()) {
    try {
      int depth = std::stoi(max_subagent_depth);
      // Clamp to valid range: 0-5
      if (depth < 0) depth = 0;
      if (depth > 5) depth = 5;
      body["max_subagent_depth"] = depth;
    } catch (...) {
      // Invalid number, ignore
    }
  }
  
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/v1/agent/session"), 
                       body.dump(), "application/json");
  if (!res) {
    if (err) *err = "failed to connect to agent server for session creation";
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "session creation returned status " + std::to_string(res->status);
    return std::nullopt;
  }
  
  auto j = nlohmann::json::parse(res->body, nullptr, false);
  if (j.is_discarded() || !j.contains("session_id") || !j["session_id"].is_string()) {
    if (err) *err = "invalid JSON from session creation";
    return std::nullopt;
  }
  
  return j["session_id"].get<std::string>();
}

std::vector<AgentServerProvider::SseEvent> AgentServerProvider::ParseSseEvents(const std::string& chunk) {
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

std::optional<ChatResponse> AgentServerProvider::ChatOnce(const ChatRequest& req, std::string* err) {
  std::string full_content;
  std::string finish_reason = "stop";
  
  const bool ok = ChatStream(
      req,
      [&](const std::string& delta) {
        full_content += delta;
        return true;
      },
      [&](const std::string& fr) {
        finish_reason = fr;
      },
      err);
  
  if (!ok) return std::nullopt;
  
  ChatResponse resp;
  resp.model = req.model;
  resp.content = std::move(full_content);
  resp.done = true;
  resp.finish_reason = std::move(finish_reason);
  return resp;
}

bool AgentServerProvider::ChatStream(const ChatRequest& req,
                                      const std::function<bool(const std::string&)>& on_delta,
                                      const std::function<void(const std::string& finish_reason)>& on_done,
                                      std::string* err) {
  // Get or create agent session
  std::string agent_session_id;
  {
    std::lock_guard<std::mutex> lock(mu_);
    
    // Map local session_id to agent session
    if (req.session_id.has_value() && !req.session_id->empty()) {
      auto it = session_map_.find(*req.session_id);
      if (it != session_map_.end()) {
        agent_session_id = it->second;
      } else {
        // Create new agent session for this local session
        std::string create_err;
        auto new_session = CreateSession(&create_err);
        if (!new_session) {
          if (err) *err = "failed to create agent session: " + create_err;
          return false;
        }
        agent_session_id = *new_session;
        session_map_[*req.session_id] = agent_session_id;
      }
    } else {
      // Use default session
      if (default_agent_session_.empty()) {
        std::string create_err;
        auto new_session = CreateSession(&create_err);
        if (!new_session) {
          if (err) *err = "failed to create default agent session: " + create_err;
          return false;
        }
        default_agent_session_ = *new_session;
      }
      agent_session_id = default_agent_session_;
    }
  }
  
  // Build the message content from the request
  // For agent server, we send the last user message
  std::string content;
  if (!req.messages.empty()) {
    // Find the last user message
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it) {
      if (it->role == "user") {
        content = it->content;
        break;
      }
    }
    // If no user message, use the last message
    if (content.empty()) {
      content = req.messages.back().content;
    }
  }
  
  if (content.empty()) {
    if (err) *err = "no message content provided";
    return false;
  }
  
  auto cli = MakeClient(endpoint_);
  
  // Build request body
  nlohmann::json body;
  body["content"] = content;
  
  std::string path = JoinPath(endpoint_.base_path, "/v1/agent/session/" + agent_session_id + "/chat");
  
  // Build request
  httplib::Request http_req;
  http_req.method = "POST";
  http_req.path = path;
  http_req.set_header("Content-Type", "application/json");
  http_req.body = body.dump();
  
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
      if (event.event_type == "text_delta") {
        // Parse the JSON data
        auto j = nlohmann::json::parse(event.data, nullptr, false);
        if (!j.is_discarded() && j.contains("delta") && j["delta"].is_string()) {
          if (!on_delta(j["delta"].get<std::string>())) {
            client_cancelled = true;
            return false;  // Client cancelled
          }
        }
      } else if (event.event_type == "completed") {
        auto j = nlohmann::json::parse(event.data, nullptr, false);
        if (!j.is_discarded() && j.contains("finish_reason") && j["finish_reason"].is_string()) {
          finish_reason = j["finish_reason"].get<std::string>();
        }
      } else if (event.event_type == "error") {
        auto j = nlohmann::json::parse(event.data, nullptr, false);
        if (!j.is_discarded() && j.contains("message") && j["message"].is_string()) {
          error_msg = j["message"].get<std::string>();
        } else {
          error_msg = event.data;
        }
        has_error = true;
        return false;
      }
      // Other event types: reasoning_delta, tool_start, tool_result, etc.
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
    if (err) *err = "failed to connect to agent server";
    return false;
  }
  
  if (has_error) {
    if (err) *err = error_msg;
    return false;
  }
  
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "chat request returned status " + std::to_string(res->status);
    return false;
  }
  
  on_done(finish_reason);
  return true;
}

}  // namespace runtime

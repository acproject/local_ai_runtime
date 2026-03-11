#pragma once

#include "config.hpp"
#include "providers/provider.hpp"

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace runtime {

// Provider that connects to llama-agent-server HTTP API
// This provider acts as an HTTP client to the agent-server service
// which provides agent capabilities with tool calling support
class AgentServerProvider : public IProvider {
 public:
  explicit AgentServerProvider(HttpEndpoint endpoint);
  ~AgentServerProvider() override;

  void Start() override;
  void Stop() override;

  std::string Name() const override;
  std::vector<ModelInfo> ListModels(std::string* err) override;
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err) override;

  std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) override;
  bool ChatStream(const ChatRequest& req,
                  const std::function<bool(const std::string&)>& on_delta,
                  const std::function<void(const std::string& finish_reason)>& on_done,
                  std::string* err) override;

 private:
  // Create a new agent session on the server
  std::optional<std::string> CreateSession(std::string* err);
  
  // Check if the agent server is healthy
  bool CheckHealth(std::string* err);
  
  // Parse SSE event from the chat stream
  struct SseEvent {
    std::string event_type;
    std::string data;
  };
  static std::vector<SseEvent> ParseSseEvents(const std::string& chunk);

  HttpEndpoint endpoint_;
  std::mutex mu_;
  
  // Cache for agent sessions (maps session_id -> agent_session_id)
  std::unordered_map<std::string, std::string> session_map_;
  
  // Default agent session ID (created on first use)
  std::string default_agent_session_;
  
  // Flag indicating if server is available
  bool server_available_ = false;
};

}  // namespace runtime

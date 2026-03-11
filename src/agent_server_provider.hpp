#pragma once

#include "config.hpp"
#include "providers/provider.hpp"

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace runtime {

// Provider that connects to llama-server (llama.cpp HTTP server)
// Uses standard OpenAI-compatible API:
// - GET /v1/models - List available models
// - POST /v1/chat/completions - Chat completion (streaming supported)
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
  // Check if the server is healthy
  bool CheckHealth(std::string* err);

  HttpEndpoint endpoint_;
  std::mutex mu_;
  bool server_available_ = false;
};

}  // namespace runtime

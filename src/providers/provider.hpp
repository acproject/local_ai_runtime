#pragma once

#include "session_manager.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace runtime {

struct ModelInfo {
  std::string id;
  std::string owned_by;
};

struct ChatRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  bool stream = false;
  std::optional<int> max_tokens;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> min_p;
};

struct ChatResponse {
  std::string model;
  std::string content;
  bool done = true;
  std::string finish_reason = "stop";
};

class IProvider {
 public:
  virtual ~IProvider() = default;

  virtual void Start() {}
  virtual void Stop() {}

  virtual std::string Name() const = 0;
  virtual std::vector<ModelInfo> ListModels(std::string* err) = 0;
  virtual std::optional<std::vector<double>> Embeddings(const std::string& model,
                                                        const std::string& input,
                                                        std::string* err) = 0;

  virtual std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) = 0;
  virtual bool ChatStream(const ChatRequest& req,
                          const std::function<bool(const std::string&)>& on_delta,
                          const std::function<void(const std::string& finish_reason)>& on_done,
                          std::string* err) = 0;
};

}  // namespace runtime


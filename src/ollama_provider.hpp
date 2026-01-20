#pragma once

#include "config.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace runtime {

struct ModelInfo {
  std::string id;
  std::string owned_by;
};

struct OllamaChatRequest {
  std::string model;
  struct Message {
    std::string role;
    std::string content;
  };
  std::vector<Message> messages;
  bool stream = false;
};

struct OllamaChatResponse {
  std::string model;
  std::string content;
  bool done = true;
};

class OllamaProvider {
 public:
  explicit OllamaProvider(HttpEndpoint endpoint);

  std::vector<ModelInfo> ListModels(std::string* err);
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err);

  std::optional<OllamaChatResponse> ChatOnce(const OllamaChatRequest& req, std::string* err);
  bool ChatStream(const OllamaChatRequest& req,
                  const std::function<void(const std::string&)>& on_delta,
                  const std::function<void()>& on_done,
                  std::string* err);

 private:
  HttpEndpoint endpoint_;
};

}  // namespace runtime

